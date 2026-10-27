"""Music structure analysis metrics, following upstream SongFormer ``evaluation/eval_infer_results.py``.

Boundary hit rates, pairwise F and normalized-entropy scores use ``mir_eval`` with the same settings as
``msaf.eval.compute_results``. Frame accuracy and IoU follow upstream ``postprocessing/calc_acc.py`` / ``calc_iou.py``.
"""

import bisect
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mir_eval
import numpy as np

from ..data.labels import END_LABEL, MsaInfo

__all__ = [
    "replace_label",
    "frame_accuracy",
    "label_overlaps",
    "segment_metrics",
    "summarize",
]


def replace_label(msa_info: MsaInfo, source: str, target: Optional[str]) -> MsaInfo:
    """Rename ``source`` segments to ``target`` (e.g. ``pre-chorus -> verse`` for 7-class scoring)."""
    if target is None:
        return list(msa_info)
    return [(t, target if label == source else label) for t, label in msa_info]


def _intervals(msa_info: MsaInfo) -> Tuple[np.ndarray, List[str]]:
    times = np.array([t for t, _ in msa_info])
    return np.stack([times[:-1], times[1:]], axis=1), [label for _, label in msa_info[:-1]]


def frame_accuracy(reference: MsaInfo, estimate: MsaInfo, digits: int = 3) -> float:
    """Fraction of the commonly covered time where the labels agree."""
    ref_times = [int(round(t, digits) * 10**digits) for t, _ in reference]
    est_times = [int(round(t, digits) * 10**digits) for t, _ in estimate]
    start, end = max(ref_times[0], est_times[0]), min(ref_times[-1], est_times[-1])

    points = sorted({start, end} | {t for t in ref_times + est_times if start <= t <= end})
    total, correct = 0, 0
    for left, right in zip(points[:-1], points[1:]):
        ref_label = reference[bisect.bisect_right(ref_times, left) - 1][1]
        est_label = estimate[bisect.bisect_right(est_times, left) - 1][1]
        total += right - left
        correct += (right - left) * (ref_label == est_label)
    return correct / total


def label_overlaps(reference: MsaInfo, estimate: MsaInfo) -> Dict[str, Tuple[float, float]]:
    """Per label occurring in either side: ``(intersection, union)`` durations."""
    ref_inter, ref_labels = _intervals(reference)
    est_inter, est_labels = _intervals(estimate)

    result = {}
    for label in set(ref_labels) | set(est_labels):
        a = [(s, e) for (s, e), lab in zip(ref_inter, ref_labels) if lab == label]
        b = [(s, e) for (s, e), lab in zip(est_inter, est_labels) if lab == label]
        intersection = sum(max(0.0, min(ea, eb) - max(sa, sb)) for sa, ea in a for sb, eb in b)
        union = sum(e - s for s, e in a) + sum(e - s for s, e in b) - intersection
        result[label] = (float(intersection), float(union))
    return result


def segment_metrics(reference: MsaInfo, estimate: MsaInfo) -> Dict[str, float]:
    """Boundary hit rates (0.5 / 1 / 3 s), pairwise F, normalized conditional entropy, and frame accuracy."""
    if reference[-1][1] != END_LABEL or estimate[-1][1] != END_LABEL:
        raise ValueError("MsaInfo must end with an 'end' entry")
    ref_inter, ref_labels = _intervals(reference)
    est_inter, est_labels = _intervals(estimate)

    result: Dict[str, float] = {}
    for window, name in [(0.5, "0.5"), (1.0, "1"), (3.0, "3")]:
        p, r, f = mir_eval.segment.detection(ref_inter, est_inter, window=window, trim=False)
        result.update({f"HitRate_{name}P": p, f"HitRate_{name}R": r, f"HitRate_{name}F": f})

    ref_inter_adj, ref_labels_adj = mir_eval.util.adjust_intervals(ref_inter, ref_labels)
    est_inter_adj, est_labels_adj = mir_eval.util.adjust_intervals(est_inter, est_labels, t_min=0.0, t_max=ref_inter_adj.max())
    result["PWP"], result["PWR"], result["PWF"] = mir_eval.segment.pairwise(ref_inter_adj, ref_labels_adj, est_inter_adj, est_labels_adj)
    result["So"], result["Su"], result["Sf"] = mir_eval.segment.nce(ref_inter_adj, ref_labels_adj, est_inter_adj, est_labels_adj)

    result["acc"] = frame_accuracy(reference, estimate)
    return {k: float(v) for k, v in result.items()}


def summarize(per_song: List[Dict[str, float]], overlaps: List[Dict[str, Tuple[float, float]]]) -> Dict[str, float]:
    """Dataset-level summary: song-averaged metrics and duration-pooled IoU (overall and per label)."""
    summary: Dict[str, float] = {"num_samples": float(len(per_song))}
    for key, name in [("HitRate_0.5F", "HR.5F"), ("HitRate_3F", "HR3F"), ("HitRate_1F", "HR1F"), ("PWF", "PWF"), ("Sf", "Sf"), ("acc", "acc")]:
        summary[name] = float(np.mean([m[key] for m in per_song])) if per_song else float("nan")

    intersection: Dict[str, float] = defaultdict(float)
    union: Dict[str, float] = defaultdict(float)
    for song in overlaps:
        for label, (i, u) in song.items():
            intersection[label] += i
            union[label] += u
    total_union = sum(union.values())
    summary["iou"] = sum(intersection.values()) / total_union if total_union > 0 else 0.0
    for label in sorted(union):
        summary[f"iou_{label}"] = intersection[label] / union[label] if union[label] > 0 else 0.0
    return summary
