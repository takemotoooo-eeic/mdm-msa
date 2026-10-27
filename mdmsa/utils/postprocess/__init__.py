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
    """Place per-chunk logits on the song timeline and average where chunks overlap.

    Args:
        chunks: ``(start_sec, {"boundary_logits": (T,), "function_logits": (T, C)})`` per chunk.
        frame_rate: Frames per second of the logits.
    """
    placed = [(int(start * frame_rate), logits) for start, logits in chunks]
    length = max(offset + logits["boundary_logits"].shape[0] for offset, logits in placed)
    num_classes = placed[0][1]["function_logits"].shape[-1]
    device = placed[0][1]["boundary_logits"].device

    boundary = torch.zeros(length, device=device)
    function = torch.zeros(length, num_classes, device=device)
    count = torch.zeros(length, device=device)
    for offset, logits in placed:
        n = logits["boundary_logits"].shape[0]
        boundary[offset : offset + n] += logits["boundary_logits"].float()
        function[offset : offset + n] += logits["function_logits"].float()
        count[offset : offset + n] += 1

    count = count.clamp(min=1)
    return {"boundary_logits": boundary / count, "function_logits": function / count.unsqueeze(-1)}
