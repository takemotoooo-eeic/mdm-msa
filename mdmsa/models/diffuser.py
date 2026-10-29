"""Masked discrete diffusion over boundary and function tokens.

Ported from md-songformer (``models/masked_diffuser.py``, ``utils/collator/masking.py``, ``utils/tokens.py``).

Each output frame carries two tokens:

- boundary: ``MASK`` / ``NO_EVENT`` / ``EVENT`` / ``PAD``
- function: a label id in ``[0, num_classes)``, ``MASK`` (= ``num_classes``) or ``PAD`` (= ``num_classes + 1``)

Training (:meth:`MaskedDiffuser.add_mask`) hides part of the ground-truth tokens and restricts the loss to hidden frames.
Inference (:meth:`MaskedDiffuser.inference`) starts from all-``MASK`` and reveals tokens by confidence
(:meth:`MaskedDiffuser.unmask`); :meth:`MaskedDiffuser.decode` turns the tokens into segments.
"""

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.data.labels import MsaInfo
from ..utils.postprocess import structure_from_tokens

__all__ = ["MaskedDiffuser"]

Batch = Dict[str, Any]

_BOUNDARY_MASKINGS = ("random", "all_mask")
_FUNCTION_MASKINGS = ("random", "random_span", "section", "section_merged", "all_mask")
_PATTERN_KEYS = {"type", "prob", "ratio_min", "ratio_max", "span_seconds"}
_BOUNDARY_UNMASK_STRATEGIES = ("dynamic", "confidence")
_CONFIDENCES = ("logit_abs", "softmax")
_FUNCTION_UNMASK_UNITS = ("frame", "segment")


class MaskedDiffuser:
    """Corruption (training) and iterative unmasking (inference) for masked diffusion SongFormer.

    Args:
        num_classes: Size of the functional label space.
        frame_rate: Frame rate of the tokens (Hz).
        boundary_masking: Masking patterns of the boundary channel, one drawn per song. Each is a dict with ``type``,
            ``prob`` (the probabilities must sum to 1) and, except for ``all_mask``, the mask ratio range
            ``ratio_min`` / ``ratio_max`` (ratio ~ ``U(ratio_min, ratio_max)``). Types: ``random`` frames (hidden
            ``NO_EVENT`` / ``EVENT`` keep the song's own ratio) and ``all_mask`` (every token hidden, the inference
            starting point).
        function_masking: Masking patterns of the function channel, drawn per song independently of the boundary
            channel. Types: ``random`` frames, ``random_span`` (label-independent spans of mean ``span_seconds``,
            default 5), ``section`` (whole sections split at ground-truth boundaries), ``section_merged`` (runs of the
            same label) and ``all_mask``.
        boundary_inference: Inference settings of the boundary channel: ``num_inference_steps``, ``unmask_strategy``
            (``dynamic`` / ``confidence``), ``confidence`` (``logit_abs`` / ``softmax``) and ``peak_window``: after
            each boundary step, keep one ``EVENT`` per window (frames, odd) and set the rest of the window to
            ``NO_EVENT`` (``0`` disables it).
        function_inference: Inference settings of the function channel: ``num_inference_steps``, ``confidence`` and
            ``unmask_unit``. ``frame`` (default) reveals single frames together with the boundary steps. ``segment``
            waits until every boundary is revealed, then reveals whole segments (between ``EVENT`` s) with the label of
            highest mean probability, most confident segments first; it adds ``num_inference_steps`` model calls after
            the boundary steps and ignores ``confidence``.
        sync_mask: Hide the same frames in both channels: ``function_masking`` draws one mask per song (on frames where
            either channel is supervised) and ``boundary_masking`` is not used. ``all_mask`` then hides both channels.
        function_noise: Corrupt part of the *visible* function tokens so the model learns that revealed tokens can be
            wrong (at inference they are its own predictions). ``prob``: share of songs that get noise (``0`` disables);
            ``max_ratio``: per song, a share ``U(0, max_ratio)`` of the visible label runs gets another label of the same
            song, drawn in proportion to its frame count (majority labels more often). In those songs every visible
            function frame is also supervised, so the model learns to keep correct tokens and fix wrong ones.
    """

    BOUNDARY_MASK = 0
    BOUNDARY_NO_EVENT = 1
    BOUNDARY_EVENT = 2
    BOUNDARY_PAD = 3
    BOUNDARY_VOCAB_SIZE = 4

    def __init__(
        self,
        num_classes: int = 128,
        frame_rate: float = 25 / 3,
        boundary_masking: Optional[Sequence[Mapping[str, Any]]] = None,
        function_masking: Optional[Sequence[Mapping[str, Any]]] = None,
        sync_mask: bool = False,
        function_noise: Optional[Mapping[str, Any]] = None,
        boundary_inference: Optional[Mapping[str, Any]] = None,
        function_inference: Optional[Mapping[str, Any]] = None,
    ) -> None:
        boundary = dict(boundary_inference or {})
        function = dict(function_inference or {})
        function_noise = dict(function_noise or {})
        for name, given, allowed in (
            ("boundary_inference", boundary, {"num_inference_steps", "unmask_strategy", "confidence", "peak_window"}),
            ("function_inference", function, {"num_inference_steps", "confidence", "unmask_unit"}),
            ("function_noise", function_noise, {"prob", "max_ratio"}),
        ):
            if set(given) - allowed:
                raise ValueError(f"{name}: unknown keys {sorted(set(given) - allowed)}.")

        peak_window = int(boundary.get("peak_window", 15))
        if peak_window < 0 or (peak_window > 0 and peak_window % 2 == 0):
            raise ValueError("boundary_inference.peak_window must be 0 or a positive odd integer.")

        self.num_classes = int(num_classes)
        self.frame_rate = float(frame_rate)
        self.FUNCTION_MASK = self.num_classes
        self.FUNCTION_PAD = self.num_classes + 1

        default = [{"type": "random", "prob": 0.6}, {"type": "all_mask", "prob": 0.4}]
        self.boundary_masking = self._parse_patterns(boundary_masking or default, _BOUNDARY_MASKINGS, "boundary_masking")
        default = [{"type": "random_span", "prob": 0.6}, {"type": "all_mask", "prob": 0.4}]
        self.function_masking = self._parse_patterns(function_masking or default, _FUNCTION_MASKINGS, "function_masking")
        self.sync_mask = bool(sync_mask)

        self.boundary_steps = int(boundary.get("num_inference_steps", 1))
        self.boundary_unmask_strategy = boundary.get("unmask_strategy", "dynamic")
        self.boundary_confidence = boundary.get("confidence", "logit_abs")
        self.function_steps = int(function.get("num_inference_steps", 1))
        self.function_confidence = function.get("confidence", "logit_abs")
        self.function_unmask_unit = function.get("unmask_unit", "frame")
        self.boundary_peak_window = peak_window
        self.function_noise_prob = float(function_noise.get("prob", 0.0))
        self.function_noise_max_ratio = float(function_noise.get("max_ratio", 0.3))
        if not 0 <= self.function_noise_prob <= 1 or not 0 <= self.function_noise_max_ratio <= 1:
            raise ValueError("function_noise.prob and function_noise.max_ratio must be in [0, 1].")

        if self.boundary_steps < 1 or self.function_steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        if self.boundary_unmask_strategy not in _BOUNDARY_UNMASK_STRATEGIES:
            raise ValueError(f"boundary_inference.unmask_strategy must be one of {_BOUNDARY_UNMASK_STRATEGIES}.")
        if self.boundary_confidence not in _CONFIDENCES or self.function_confidence not in _CONFIDENCES:
            raise ValueError(f"confidence must be one of {_CONFIDENCES}.")
        if self.function_unmask_unit not in _FUNCTION_UNMASK_UNITS:
            raise ValueError(f"function_inference.unmask_unit must be one of {_FUNCTION_UNMASK_UNITS}.")

    def _parse_patterns(self, patterns: Sequence[Mapping[str, Any]], types: Tuple[str, ...], name: str) -> List[Dict[str, Any]]:
        parsed = []
        for pattern in patterns:
            pattern = dict(pattern)
            unknown = set(pattern) - _PATTERN_KEYS
            if unknown:
                raise ValueError(f"{name}: unknown keys {sorted(unknown)}.")
            if pattern.get("type") not in types:
                raise ValueError(f"{name}: type must be one of {types}.")
            if "span_seconds" in pattern and pattern["type"] != "random_span":
                raise ValueError(f"{name}: span_seconds is only for random_span.")
            item = {
                "type": pattern["type"],
                "prob": float(pattern.get("prob", 0.0)),
                "ratio_min": float(pattern.get("ratio_min", 0.05)),
                "ratio_max": float(pattern.get("ratio_max", 1.0)),
                "span_frames": max(1.0, float(pattern.get("span_seconds", 5.0)) * self.frame_rate),
            }
            if item["prob"] < 0 or not 0 <= item["ratio_min"] <= item["ratio_max"] <= 1:
                raise ValueError(f"{name}: requires prob >= 0 and 0 <= ratio_min <= ratio_max <= 1.")
            parsed.append(item)
        if not parsed or not math.isclose(sum(p["prob"] for p in parsed), 1.0, abs_tol=1e-6):
            raise ValueError(f"{name}: the probabilities must sum to 1.")
        return parsed

    @staticmethod
    def function_vocab_size(num_classes: int) -> int:
        """Labels plus ``MASK`` and ``PAD``."""
        return num_classes + 2

    @property
    def num_inference_steps(self) -> int:
        if self.function_unmask_unit == "segment":
            return self.boundary_steps + self.function_steps
        return max(self.boundary_steps, self.function_steps)

    # ------------------------------------------------------------------ training

    def tokenize(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ground-truth tokens ``(B, T)`` from a :class:`mdmsa.utils.collator.SongFormerCollator` batch.

        Frames ignored by the dataset (padding, ``boundary_mask`` / ``function_mask``) become ``PAD``.
        """
        padding = batch["padding_mask"]
        boundary = torch.where(batch["boundary_events"] > 0.5, self.BOUNDARY_EVENT, self.BOUNDARY_NO_EVENT)
        boundary = boundary.masked_fill(padding | batch["boundary_mask"], self.BOUNDARY_PAD)
        function = batch["function_targets"].argmax(dim=-1)
        function = function.masked_fill(padding | batch["function_mask"], self.FUNCTION_PAD)
        return boundary.long(), function.long()

    def add_mask(self, batch: Batch) -> Batch:
        """Corrupt a training batch.

        Adds ``boundary_tokens`` / ``function_tokens`` (model inputs) and adds the visible frames to
        ``boundary_mask`` / ``function_mask``, so the losses only see hidden frames. Each song draws one pattern per
        channel from ``boundary_masking`` / ``function_masking``, with its own mask ratio (with ``sync_mask``, one
        ``function_masking`` pattern hides the same frames in both channels).
        """
        boundary, function = self.tokenize(batch)
        boundary_valid, function_valid = boundary != self.BOUNDARY_PAD, function != self.FUNCTION_PAD
        if self.sync_mask:
            valid = boundary_valid | function_valid
            hidden = self._sample_hidden(
                self.function_masking, valid, lambda p, rows, r: self._function_hidden(p, function[rows], boundary[rows], r, valid=valid[rows])
            )
            boundary_hidden, function_hidden = hidden & boundary_valid, hidden & function_valid
        else:
            boundary_hidden = self._sample_hidden(self.boundary_masking, boundary_valid, lambda p, rows, r: self._boundary_hidden(p, boundary[rows], r))
            function_hidden = self._sample_hidden(
                self.function_masking, function_valid, lambda p, rows, r: self._function_hidden(p, function[rows], boundary[rows], r)
            )
        boundary_noisy = boundary.masked_fill(boundary_hidden, self.BOUNDARY_MASK)
        function_noisy = function.masked_fill(function_hidden, self.FUNCTION_MASK)

        if self.function_noise_prob > 0:
            function_noisy, supervised = self._corrupt_visible_function(function_noisy, function)
            function_hidden = function_hidden | supervised

        return {
            **batch,
            "boundary_tokens": boundary_noisy,
            "function_tokens": function_noisy,
            "boundary_mask": batch["boundary_mask"] | ~boundary_hidden,
            "function_mask": batch["function_mask"] | ~function_hidden,
        }

    @staticmethod
    def _sample_hidden(
        patterns: List[Dict[str, Any]], valid: torch.Tensor, hide: Callable[[Dict[str, Any], torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Hidden frames ``(B, T)``: one pattern per song, ``hide(pattern, rows, ratios)`` for the songs that drew it."""
        probs = torch.tensor([p["prob"] for p in patterns])
        choice = torch.multinomial(probs, valid.shape[0], replacement=True).to(valid.device)
        hidden = torch.zeros_like(valid)
        for index, pattern in enumerate(patterns):
            rows = choice == index
            n_rows = int(rows.sum().item())
            if n_rows == 0:
                continue
            if pattern["type"] == "all_mask":
                hidden[rows] = valid[rows]
                continue
            ratios = torch.empty(n_rows, device=valid.device).uniform_(pattern["ratio_min"], pattern["ratio_max"])
            hidden[rows] = hide(pattern, rows, ratios) & valid[rows]
        return hidden

    @staticmethod
    def _boundary_hidden(pattern: Dict[str, Any], tokens: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        return torch.rand(tokens.shape, device=tokens.device) < ratios.unsqueeze(1)  # random

    def _function_hidden(
        self, pattern: Dict[str, Any], tokens: torch.Tensor, boundary: torch.Tensor, ratios: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``valid`` (default: non-``PAD`` function frames) sets where ``random_span`` places its spans."""
        if pattern["type"] == "random":
            return torch.rand(tokens.shape, device=tokens.device) < ratios.unsqueeze(1)
        if pattern["type"] == "random_span":
            return self._random_span_mask(tokens != self.FUNCTION_PAD if valid is None else valid, ratios, pattern["span_frames"])
        sections = self._section_ids(tokens, boundary if pattern["type"] == "section" else None)
        n_sections = sections.amax(dim=-1)
        k = torch.minimum(torch.round(ratios * n_sections.float()).long().clamp(min=0), n_sections)
        # A positive ratio on a short song would otherwise round down to zero sections.
        k = torch.where((ratios > 0) & (n_sections > 0) & (k == 0), torch.ones_like(k), k)
        return self._select_sections(sections, k)

    def _corrupt_visible_function(self, noisy: torch.Tensor, clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Give some visible label runs another label of the same song (see ``function_noise``).

        Returns the corrupted tokens and the visible frames of the noised songs (to be supervised).
        """
        batch_size = noisy.shape[0]
        visible = (noisy != self.FUNCTION_MASK) & (noisy != self.FUNCTION_PAD)
        starts = torch.ones_like(visible)
        starts[:, 1:] = (noisy[:, 1:] != noisy[:, :-1]) | ~visible[:, :-1]
        runs = (starts & visible).long().cumsum(dim=-1).masked_fill(~visible, 0)  # (B, T), 0 = not visible
        n_runs = runs.amax(dim=-1)
        noised_rows = (torch.rand(batch_size, device=noisy.device) < self.function_noise_prob) & (n_runs > 0)
        if not noised_rows.any():
            return noisy, torch.zeros_like(visible)

        ratios = torch.rand(batch_size, device=noisy.device) * self.function_noise_max_ratio
        k = torch.round(ratios * n_runs.float()).long().masked_fill(~noised_rows, 0)
        chosen = self._select_sections(runs, k)  # frames of the chosen runs

        # Replacement label per run: another label of the song, weighted by its frame count.
        valid = clean != self.FUNCTION_PAD
        counts = torch.zeros(batch_size, self.num_classes, device=noisy.device)
        counts.scatter_add_(1, clean.masked_fill(~valid, 0), valid.float())
        max_runs = int(n_runs.max().item())
        run_label = torch.zeros(batch_size, max_runs + 1, dtype=torch.long, device=noisy.device)
        run_label.scatter_(1, runs, noisy.masked_fill(~visible, 0))
        weights = counts.unsqueeze(1).repeat(1, max_runs + 1, 1).scatter_(2, run_label.unsqueeze(-1), 0.0)
        has_other = weights.sum(dim=-1) > 0
        weights = torch.where(has_other.unsqueeze(-1), weights, torch.ones_like(weights))
        new_label = torch.multinomial(weights.flatten(0, 1), 1).view(batch_size, max_runs + 1)

        corrupt = chosen & has_other.gather(1, runs)
        noisy = torch.where(corrupt, new_label.gather(1, runs), noisy)
        return noisy, visible & noised_rows.unsqueeze(1)

    def _section_ids(self, tokens: torch.Tensor, boundary: Optional[torch.Tensor]) -> torch.Tensor:
        """Section index per frame (0 for ``PAD``). A section is a run of one label, also split at ``EVENT`` frames."""
        valid = tokens != self.FUNCTION_PAD
        changed = torch.ones_like(valid)
        changed[:, 1:] = tokens[:, 1:] != tokens[:, :-1]
        if boundary is not None:
            changed |= boundary == self.BOUNDARY_EVENT
        return (changed & valid).long().cumsum(dim=-1).masked_fill(~valid, 0)

    @staticmethod
    def _select_sections(sections: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Frames of ``k`` random sections per row."""
        max_sections = int(sections.max().item()) if sections.numel() else 0
        max_k = int(k.max().item()) if k.numel() else 0
        if max_sections == 0 or max_k <= 0:
            return torch.zeros_like(sections, dtype=torch.bool)

        batch_size = sections.shape[0]
        ids = torch.arange(max_sections + 1, device=sections.device).unsqueeze(0)
        priority = torch.rand(batch_size, max_sections + 1, device=sections.device)
        priority = priority.masked_fill((ids < 1) | (ids > sections.amax(dim=-1, keepdim=True)), -1.0)
        top_priority, top_ids = priority.topk(max_k, dim=-1)
        take = (torch.arange(max_k, device=sections.device).unsqueeze(0) < k.unsqueeze(1)) & (top_priority >= 0)
        chosen = torch.zeros(batch_size, max_sections + 1, dtype=torch.bool, device=sections.device).scatter_(1, top_ids, take)
        return chosen.gather(1, sections)

    @staticmethod
    def _random_partition(total: int, parts: int, device: torch.device) -> torch.Tensor:
        """Uniform positive integer composition of ``total`` into ``parts`` lengths."""
        if parts == 1:
            return torch.tensor([total], device=device)
        cuts = (torch.randperm(total - 1, device=device)[: parts - 1] + 1).sort().values
        return torch.cat((cuts.new_zeros(1), cuts, cuts.new_full((1,), total))).diff()

    @classmethod
    def _random_span_mask(cls, valid: torch.Tensor, ratios: torch.Tensor, span_frames: float) -> torch.Tensor:
        """Hide ``round(ratio * valid_frames)`` frames as label-independent random spans.

        Each contiguous valid run gets its proportional share (randomized rounding keeps the song total). Within a run,
        hidden and visible spans alternate with random lengths (mean ``span_frames``) and a random rotation.
        """
        batch_size = valid.shape[0]
        n_valid = valid.sum(dim=-1)
        budgets = torch.round(ratios * n_valid).long()
        budgets = torch.minimum(torch.where((ratios > 0) & (n_valid > 0), budgets.clamp(min=1), budgets), n_valid)

        valid_cpu = valid.cpu()
        plans = torch.stack((n_valid, budgets), dim=-1).tolist()
        rounding = torch.rand(batch_size).tolist()
        hidden = torch.zeros_like(valid)
        for row, (count, budget) in enumerate(plans):
            if budget == 0:
                continue
            if budget == count:
                hidden[row] = valid[row]
                continue
            edges = F.pad(valid_cpu[row].to(torch.int8), (1, 1)).diff()
            starts = (edges == 1).nonzero(as_tuple=True)[0].tolist()
            ends = (edges == -1).nonzero(as_tuple=True)[0].tolist()
            offset = min(int(rounding[row] * count), count - 1)
            cumulative, allocated = 0, 0
            for start, end in zip(starts, ends):
                length = end - start
                cumulative += length
                next_allocated = (cumulative * budget + offset) // count
                n_hidden, allocated = next_allocated - allocated, next_allocated
                if n_hidden == 0:
                    continue
                if n_hidden == length:
                    hidden[row, start:end] = True
                    continue
                n_spans = max(1, min(round(n_hidden / span_frames), n_hidden, length - n_hidden))
                hidden_lengths = cls._random_partition(n_hidden, n_spans, valid.device)
                visible_lengths = cls._random_partition(length - n_hidden, n_spans, valid.device)
                span_ends = torch.stack((visible_lengths, hidden_lengths), dim=-1).flatten().cumsum(dim=0)
                shift = torch.randint(length, (1,), device=valid.device)
                positions = (torch.arange(length, device=valid.device) + shift) % length
                hidden[row, start:end] = torch.searchsorted(span_ends, positions, right=True).remainder(2) == 1
        return hidden

    # ----------------------------------------------------------------- inference

    def init_tokens(self, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-``MASK`` tokens, ``PAD`` where ``padding_mask`` is true."""
        boundary = torch.full(padding_mask.shape, self.BOUNDARY_MASK, dtype=torch.long, device=padding_mask.device)
        function = torch.full(padding_mask.shape, self.FUNCTION_MASK, dtype=torch.long, device=padding_mask.device)
        return boundary.masked_fill(padding_mask, self.BOUNDARY_PAD), function.masked_fill(padding_mask, self.FUNCTION_PAD)

    @staticmethod
    def _num_to_unmask(num_valid: torch.Tensor, remaining: torch.Tensor, step: int, num_steps: int) -> torch.Tensor:
        """Linear schedule: ``ceil(valid / steps)`` per step, everything left at the last step."""
        if step >= num_steps - 1:
            return remaining
        return torch.minimum((num_valid + num_steps - 1) // num_steps, remaining)

    @torch.no_grad()
    def unmask(
        self,
        boundary_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        outputs: Mapping[str, torch.Tensor],
        step: int,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reveal this step's share of ``MASK`` tokens from the model outputs. A channel stops after its own last step."""
        if step < self.boundary_steps:
            k = self._num_to_unmask(
                (boundary_tokens != self.BOUNDARY_PAD).sum(dim=-1), (boundary_tokens == self.BOUNDARY_MASK).sum(dim=-1), step, self.boundary_steps
            )
            boundary_tokens = self._unmask_boundary(boundary_tokens, outputs["boundary_logits"], k)

        logits = outputs["function_logits"]
        if label_mask is not None:
            logits = logits.masked_fill(label_mask.unsqueeze(1), -float("inf"))
        if self.function_unmask_unit == "segment":
            function_step = step - self.boundary_steps
            if 0 <= function_step < self.function_steps:
                function_tokens = self._unmask_function_segments(function_tokens, boundary_tokens, logits, function_step)
        elif step < self.function_steps:
            k = self._num_to_unmask(
                (function_tokens != self.FUNCTION_PAD).sum(dim=-1), (function_tokens == self.FUNCTION_MASK).sum(dim=-1), step, self.function_steps
            )
            function_tokens = self._unmask_function(function_tokens, logits, k)

        return boundary_tokens, function_tokens

    def _unmask_boundary(self, tokens: torch.Tensor, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        predicted = torch.where(logits > 0, self.BOUNDARY_EVENT, self.BOUNDARY_NO_EVENT)
        confidence = logits.abs()
        if self.boundary_confidence == "softmax":
            # Probability of the chosen class is sigmoid(|z|); rank its log to avoid rounding to 1.
            confidence = F.logsigmoid(confidence.float())
        still_masked = tokens == self.BOUNDARY_MASK

        if self.boundary_unmask_strategy == "confidence":
            tokens = self._unmask_topk(tokens, predicted, confidence, still_masked, k)
        else:
            remaining = still_masked.sum(dim=-1)
            k = torch.minimum(k.clamp(min=0), remaining)
            predicted_event = still_masked & (predicted == self.BOUNDARY_EVENT)
            predicted_no_event = still_masked & (predicted == self.BOUNDARY_NO_EVENT)
            # dynamic: follow the model's own EVENT ratio among the masked frames.
            k_event = torch.round(k.double() * predicted_event.sum(dim=-1) / remaining.clamp(min=1)).long()
            k_no_event = k - k_event
            tokens = self._unmask_topk(tokens, predicted, confidence, predicted_event, k_event)
            tokens = self._unmask_topk(tokens, predicted, confidence, predicted_no_event, k_no_event)

        return self._suppress_boundary_neighbors(tokens, logits)

    def _suppress_boundary_neighbors(self, tokens: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Keep one ``EVENT`` (highest logit) per window and close the rest of the window as ``NO_EVENT``.

        Training boundaries are single-frame impulses, so neighbouring ``EVENT`` s never co-occur.
        """
        window = self.boundary_peak_window
        if window <= 1:
            return tokens
        radius = window // 2
        is_event = tokens == self.BOUNDARY_EVENT
        score = F.pad(logits.float().masked_fill(~is_event, -float("inf")), (radius, radius), value=-float("inf"))
        is_peak = is_event & (score.unfold(1, window, 1).argmax(dim=-1) == radius)
        tokens = tokens.masked_fill(is_event & ~is_peak, self.BOUNDARY_NO_EVENT)
        near_peak = F.max_pool1d(is_peak.float().unsqueeze(1), kernel_size=window, stride=1, padding=radius).squeeze(1) > 0
        return tokens.masked_fill(near_peak & (tokens == self.BOUNDARY_MASK), self.BOUNDARY_NO_EVENT)

    def _unmask_function(self, tokens: torch.Tensor, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        predicted = logits.argmax(dim=-1)
        scores = logits.float().softmax(dim=-1) if self.function_confidence == "softmax" else logits.abs()
        confidence = scores.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
        return self._unmask_topk(tokens, predicted, confidence, tokens == self.FUNCTION_MASK, k)

    def _unmask_function_segments(self, tokens: torch.Tensor, boundary_tokens: torch.Tensor, logits: torch.Tensor, step: int) -> torch.Tensor:
        """Reveal this step's share of masked segments, each filled with its highest mean-probability label.

        Segments start at the first frame and at every ``EVENT`` (as in :meth:`decode`); boundaries must be revealed.
        """
        valid = tokens != self.FUNCTION_PAD
        starts = boundary_tokens == self.BOUNDARY_EVENT
        starts[:, 0] = True
        segments = (starts & valid).long().cumsum(dim=-1).masked_fill(~valid, 0)  # (B, T), 0 = padding
        num_segments = int(segments.max().item()) + 1

        probs = logits.float().softmax(dim=-1) * valid.unsqueeze(-1)
        index = segments.unsqueeze(-1).expand(-1, -1, probs.shape[-1])
        mean = probs.new_zeros(probs.shape[0], num_segments, probs.shape[-1]).scatter_add_(1, index, probs)
        counts = torch.zeros(segments.shape[0], num_segments, device=tokens.device).scatter_add_(1, segments, valid.float())
        confidence, label = (mean / counts.clamp(min=1).unsqueeze(-1)).max(dim=-1)  # (B, S)

        seg_ids = torch.arange(num_segments, device=tokens.device).unsqueeze(0)
        masked_frames = torch.zeros_like(counts).scatter_add_(1, segments, (tokens == self.FUNCTION_MASK).float())
        eligible = (seg_ids > 0) & (masked_frames > 0)
        k = self._num_to_unmask(((seg_ids > 0) & (counts > 0)).sum(dim=-1), eligible.sum(dim=-1), step, self.function_steps)
        chosen = self._unmask_topk(torch.zeros_like(label), torch.ones_like(label), confidence, eligible, k)  # 1 = reveal
        reveal = chosen.gather(1, segments).bool() & (tokens == self.FUNCTION_MASK)
        return torch.where(reveal, label.gather(1, segments), tokens)

    @staticmethod
    def _unmask_topk(tokens: torch.Tensor, predicted: torch.Tensor, confidence: torch.Tensor, eligible: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Replace the ``k`` most confident ``eligible`` tokens per row by their predictions."""
        k = torch.minimum(k.to(device=tokens.device, dtype=torch.long), eligible.sum(dim=-1))
        max_k = int(k.max().item()) if k.numel() else 0
        if max_k <= 0:
            return tokens
        top_confidence, idx = confidence.masked_fill(~eligible, -float("inf")).topk(max_k, dim=-1)
        take = (torch.arange(max_k, device=tokens.device).unsqueeze(0) < k.unsqueeze(1)) & torch.isfinite(top_confidence)
        return tokens.scatter(1, idx, torch.where(take, predicted.gather(1, idx), tokens.gather(1, idx)))

    @torch.no_grad()
    def inference(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        dataset_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Denoise from all-``MASK``.

        Returns the last step's ``boundary_logits`` / ``function_logits`` (unmasked by ``label_mask``) plus the final
        ``boundary_tokens`` / ``function_tokens``, all ``(B, T, ...)``.
        """
        length = embeddings.shape[1] // model.downsample_rate
        token_padding = padding_mask if padding_mask is not None else torch.zeros(embeddings.shape[0], length, dtype=torch.bool, device=embeddings.device)
        boundary_tokens, function_tokens = self.init_tokens(token_padding)

        outputs: Dict[str, torch.Tensor] = {}
        for step in range(self.num_inference_steps):
            outputs = model(embeddings, dataset_ids, padding_mask=padding_mask, boundary_tokens=boundary_tokens, function_tokens=function_tokens)
            boundary_tokens, function_tokens = self.unmask(boundary_tokens, function_tokens, outputs, step, label_mask=label_mask)

        return {**outputs, "boundary_tokens": boundary_tokens, "function_tokens": function_tokens}

    def decode(self, outputs: Mapping[str, torch.Tensor]) -> MsaInfo:
        """Segments of one song from its final tokens ``(T,)``."""
        return structure_from_tokens(
            outputs["boundary_tokens"],
            outputs["function_tokens"],
            frame_rate=self.frame_rate,
            event_id=self.BOUNDARY_EVENT,
            num_classes=self.num_classes,
        )
