"""Post-processing of SongFormer outputs into ``MsaInfo`` segments.

Peak picking is adapted from all-in-one (MIT license):
https://github.com/mir-aidj/all-in-one/blob/main/src/allin1/postprocessing/helpers.py
The rest follows upstream SongFormer ``postprocessing/functional.py`` and ``infer/infer.py``.
"""

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

from ..data.labels import END_LABEL, ID_TO_LABEL, MsaInfo

__all__ = [
    "local_maxima",
    "peak_picking",
    "decode_structure",
    "rule_post_processing",
    "stitch_chunk_logits",
    "structure_from_tokens",
]


def local_maxima(x: torch.Tensor, filter_size: int = 41) -> torch.Tensor:
    """Keep values that are the maximum of their centred window; zero elsewhere. ``x`` is ``(T,)``."""
    if filter_size % 2 != 1:
        raise ValueError("filter_size must be odd")
    padded = F.pad(x.unsqueeze(0), (filter_size // 2, filter_size // 2), value=-torch.inf)
    windows = padded.unfold(1, filter_size, 1)[0]
    is_peak = windows[:, filter_size // 2] == windows.max(dim=-1).values
    return torch.where(is_peak, x, torch.zeros_like(x))


def peak_picking(activation: np.ndarray, window_past: int = 12, window_future: int = 6) -> np.ndarray:
    """Strength of each local maximum (activation minus mean of the surrounding windows); zero elsewhere."""
    if (window_past + window_future) % 2 != 0:
        raise ValueError("window_past + window_future must be even")
    window_size = window_past + window_future + 1

    padded = np.pad(activation, (window_past, window_future), mode="constant")
    is_peak = (activation == sliding_window_view(padded, window_size).max(axis=-1)) & (activation > 0)

    past_mean = sliding_window_view(padded[: -(window_future + 1)], window_past).mean(axis=-1)
    future_mean = sliding_window_view(padded[window_past + 1 :], window_future).mean(axis=-1)
    strength = activation - (past_mean + future_mean) / 2

    out = np.zeros_like(activation)
    out[is_peak] = strength[is_peak]
    return out


def decode_structure(
    boundary_logits: torch.Tensor,
    function_logits: torch.Tensor,
    frame_rate: float,
    local_maxima_filter_size: int = 3,
    peak_window: float = 12.0,
) -> MsaInfo:
    """Turn one song's logits into segments.

    Args:
        boundary_logits: ``(T,)``.
        function_logits: ``(T, C)``; disallowed classes should already be ``-inf``.
        frame_rate: Frames per second of the logits.
        local_maxima_filter_size: Window (frames) for the first local-maximum filter.
        peak_window: Past / future window (s) for peak picking.

    Returns:
        ``[(start, label), ..., (end_time, "end")]``. Each segment takes the class with the highest mean probability.
    """
    boundary_prob = local_maxima(torch.sigmoid(boundary_logits.float()), filter_size=local_maxima_filter_size).cpu().numpy()
    function_prob = torch.softmax(function_logits.float(), dim=-1).cpu().numpy()

    window = int(peak_window * frame_rate)
    boundary_frames = np.flatnonzero(peak_picking(boundary_prob, window_past=window, window_future=window) > 0)

    duration = len(boundary_prob) / frame_rate
    times = boundary_frames / frame_rate
    if len(times) == 0 or times[0] != 0:
        times = np.insert(times, 0, 0.0)
    if times[-1] != duration:
        times = np.append(times, duration)

    segment_probs = np.split(function_prob, boundary_frames[boundary_frames > 0], axis=0)
    labels = [ID_TO_LABEL[int(p.mean(axis=0).argmax())] for p in segment_probs]

    segments: MsaInfo = [(float(start), label) for start, label in zip(times[:-1], labels)]
    segments.append((float(times[-1]), END_LABEL))
    return segments


def rule_post_processing(msa_info: MsaInfo, min_edge_duration: float = 1.0, max_merge_duration: float = 10.0) -> MsaInfo:
    """Clean the song edges: absorb segments shorter than ``min_edge_duration`` at the start / end, then merge a
    short (``<= max_merge_duration``) first / last segment into its neighbour when both share a label."""
    result = list(msa_info)

    while len(result) > 2 and result[1][0] - result[0][0] < min_edge_duration:
        result = [(result[0][0], result[1][1])] + result[2:]

    while len(result) > 2 and result[-1][0] - result[-2][0] < min_edge_duration:
        result = result[:-2] + [result[-1]]

    while len(result) > 2 and result[0][1] == result[1][1] and result[1][0] <= max_merge_duration:
        result = [result[0]] + result[2:]

    while len(result) > 2 and result[-2][1] == result[-3][1] and result[-1][0] - result[-2][0] <= max_merge_duration:
        result = result[:-2] + [result[-1]]

    return result


def stitch_chunk_logits(chunks: Iterable[Tuple[float, Dict[str, torch.Tensor]]], frame_rate: float) -> Dict[str, torch.Tensor]:
    """Place per-chunk frame tensors on the song timeline.

    Floating-point tensors (logits) are averaged where chunks overlap; other tensors (tokens) take the later chunk.

    Args:
        chunks: ``(start_sec, {name: (T, ...)})`` per chunk, e.g. ``boundary_logits`` ``(T,)`` and ``function_logits``
            ``(T, C)``.
        frame_rate: Frames per second of the tensors.
    """
    placed = [(int(start * frame_rate), tensors) for start, tensors in chunks]
    length = max(offset + next(iter(tensors.values())).shape[0] for offset, tensors in placed)

    stitched: Dict[str, torch.Tensor] = {}
    for name, first in placed[0][1].items():
        is_float = first.is_floating_point()
        out = torch.zeros((length, *first.shape[1:]), dtype=torch.float32 if is_float else first.dtype, device=first.device)
        count = torch.zeros(length, device=first.device)
        for offset, tensors in placed:
            value = tensors[name]
            n = value.shape[0]
            if is_float:
                out[offset : offset + n] += value.float()
            else:
                out[offset : offset + n] = value
            count[offset : offset + n] += 1
        if is_float:
            out = out / count.clamp(min=1).view(-1, *([1] * (out.dim() - 1)))
        stitched[name] = out
    return stitched


def structure_from_tokens(boundary_tokens: torch.Tensor, function_tokens: torch.Tensor, frame_rate: float, event_id: int, num_classes: int) -> MsaInfo:
    """Segments from fully unmasked tokens ``(T,)`` of one song.

    Every ``event_id`` frame starts a segment; its label is the most frequent label token inside it. Segments without
    a label token (only ``MASK`` / ``PAD``) are dropped, so the previous segment extends over them.
    """
    boundary = boundary_tokens.detach().cpu().numpy()
    function = function_tokens.detach().cpu().numpy()
    length = len(boundary)

    cuts = [0, *(int(i) for i in np.flatnonzero(boundary == event_id) if i > 0), length]
    segments: MsaInfo = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        labels = function[start:end]
        labels = labels[(labels >= 0) & (labels < num_classes)]
        if len(labels) > 0:
            segments.append((start / frame_rate, ID_TO_LABEL[int(np.bincount(labels).argmax())]))

    if not segments:
        segments.append((0.0, ID_TO_LABEL[0]))
    segments.append((length / frame_rate, END_LABEL))
    return segments
