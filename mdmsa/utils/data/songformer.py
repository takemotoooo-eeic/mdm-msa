"""Datasets over pre-extracted SongFormer SSL features.

Features are stored per 420 s window as ``<song_id>_<start_sec>.npy`` with shape ``(1, T, 1024)`` at 25 Hz,
one directory per stream. Streams are concatenated along the feature axis in the order given.
Ported from upstream ``dataset/SongFormerDataset.py``, ``HookTheoryAdapter.py`` and ``GeminiOnlyLabelAdapter.py``.
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from ..logging import get_logger
from .labels import DATASET_TO_ID, END_LABEL, LABEL_TO_ID, NO_LABEL, MsaInfo, label_mask

__all__ = [
    "SongFormDBDataset",
    "SongFormBenchDataset",
    "load_features",
    "load_msa_txt",
    "split_chunk_id",
]

_EPS = 1e-6
# HookTheory labels -> SongFormer labels.
_HOOK_LABELS = {
    "chorus": "chorus",
    "intro": "intro",
    "bridge": "bridge",
    "verse": "verse",
    "pre-chorus": "pre-chorus",
    "solo": "inst",
    "instrumental": "inst",
    "outro": "outro",
    NO_LABEL: NO_LABEL,
}


def split_chunk_id(stem: str) -> Tuple[str, int]:
    """``"HX_0003_6foot7foot_420" -> ("HX_0003_6foot7foot", 420)``."""
    song_id, start = stem.rsplit("_", 1)
    return song_id, int(start)


def load_features(embedding_dirs: Sequence[str], stem: str, max_length_diff: int = 4) -> np.ndarray:
    """Load and concatenate all streams of one chunk: ``(T, sum(D))``."""
    streams = [np.load(Path(d) / f"{stem}.npy").squeeze(0) for d in embedding_dirs]
    lengths = [x.shape[0] for x in streams]
    if max(lengths) - min(lengths) > max_length_diff:
        raise ValueError(f"Feature lengths of {stem} differ too much: {lengths}")
    return np.concatenate([x[: min(lengths)] for x in streams], axis=-1).astype(np.float32)


def load_msa_txt(path: str) -> MsaInfo:
    """Read ``<time> <label>`` lines ending with an ``end`` line."""
    msa_info: MsaInfo = []
    with open(path) as f:
        for line in f:
            if line.strip():
                time_, label = line.split()
                msa_info.append((float(time_), label))
    if not msa_info or msa_info[-1][1] != END_LABEL:
        raise ValueError(f"{path} must end with '{END_LABEL}'")
    return msa_info


def _list_chunk_stems(directory: str) -> Set[str]:
    return {p.stem for p in Path(directory).iterdir() if p.suffix == ".npy"}


def _read_ids(path: str) -> Set[str]:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


@dataclass
class _Subset:
    name: str
    dataset: str
    embedding_dirs: List[str]
    label_format: str
    supervise_boundary: bool
    # timeline: song_id -> (times, labels); segments: song_id -> [segment dicts]
    labels: Dict[str, Any]


class SongFormDBDataset(Dataset):
    """420 s training / validation slices of SongFormDB.

    Args:
        subsets: One dict per subset with keys

            - ``name``: short name used in ``data_id``.
            - ``dataset``: key of :data:`DATASET_TO_ID` (selects the label set and conditioning id).
            - ``label_path``: jsonl annotations.
            - ``split_path``: song ids to use, one per line.
            - ``embedding_dirs``: feature directories, concatenated in this order.
            - ``multiplier`` (default 1): how many times each slice is repeated.
            - ``label_format`` (default ``"timeline"``): ``"timeline"`` for ``{"id", "labels": [[time, label], ...]}``
              or ``"segments"`` for HookTheory rows ``{"ori_audio_path", "segment_start", "segment_end", "label"}``.
            - ``supervise_boundary`` (default true): false masks the whole boundary loss (labels-only data).
        num_classes: Size of the functional label space.
        frame_rate: Output frame rate of the model (Hz).
        slice_dur: Length of one feature chunk (s).
        num_neighbors: Width of the Gaussian used to widen boundary targets (frames).
    """

    def __init__(
        self,
        subsets: List[Dict[str, Any]],
        num_classes: int = 128,
        frame_rate: float = 25 / 3,
        slice_dur: int = 420,
        num_neighbors: int = 10,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.frame_rate = frame_rate
        self.slice_dur = slice_dur
        self.num_neighbors = num_neighbors
        self.num_frames = int(slice_dur * frame_rate)

        logger = get_logger()
        self.subsets: List[_Subset] = []
        self.items: List[Tuple[int, str]] = []

        for index, spec in enumerate(subsets):
            label_format = spec.get("label_format", "timeline")
            if label_format == "timeline":
                labels = self._load_timeline(spec["label_path"])
            elif label_format == "segments":
                labels = self._load_segments(spec["label_path"])
            else:
                raise ValueError(f"Unknown label_format: {label_format}")

            subset = _Subset(
                name=spec["name"],
                dataset=spec["dataset"],
                embedding_dirs=list(spec["embedding_dirs"]),
                label_format=label_format,
                supervise_boundary=spec.get("supervise_boundary", True),
                labels=labels,
            )
            self.subsets.append(subset)

            stems = set.intersection(*(_list_chunk_stems(d) for d in subset.embedding_dirs))
            split_ids = _read_ids(spec["split_path"])
            stems = sorted(s for s in stems if split_chunk_id(s)[0] in split_ids and split_chunk_id(s)[0] in labels)

            multiplier = int(spec.get("multiplier", 1))
            logger.info(f"{subset.name}: {len(stems)} slices x {multiplier}")
            self.items.extend([(index, stem) for stem in stems] * multiplier)

        self.label_masks = {name: label_mask(name, num_classes) for name in DATASET_TO_ID}

    @staticmethod
    def _load_timeline(path: str) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        labels = {}
        with open(path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    item = json.loads(line)
                    labels[item["id"]] = (np.array([float(t) for t, _ in item["labels"]]), [label for _, label in item["labels"]])
        return labels

    @staticmethod
    def _load_segments(path: str) -> Dict[str, List[Dict[str, Any]]]:
        labels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    labels[Path(item["ori_audio_path"]).stem].append(
                        {
                            "start": round(float(item["segment_start"]), 3),
                            "end": round(float(item["segment_end"]), 3),
                            "labels": [_HOOK_LABELS[x] for x in item["label"]],
                        }
                    )
        return dict(labels)

    def __len__(self) -> int:
        return len(self.items)

    def _time2frame(self, time_: float) -> int:
        return int(time_ * self.frame_rate)

    def _widen(self, events: np.ndarray) -> np.ndarray:
        sigma = self.num_neighbors / 3.0
        smoothed = gaussian_filter1d(events, sigma=sigma) * (np.sqrt(2 * np.pi) * sigma)
        return np.clip(smoothed, 0.0, 1.0)

    def _timeline_from_segments(self, segments: List[Dict[str, Any]], start: float, end: float) -> Tuple[np.ndarray, List[str]]:
        """Cut HookTheory segments to ``[start, end]``; gaps become ``NO_LABEL``, parts < 2 s are merged randomly."""
        points = {start, end}
        for seg in segments:
            for t in (seg["start"], seg["end"]):
                if start <= t <= end:
                    points.add(t)
        points_sorted = sorted(points)

        parts = []
        for left, right in zip(points_sorted[:-1], points_sorted[1:]):
            labels = sorted({label for seg in segments if seg["start"] <= left and seg["end"] >= right for label in seg["labels"]})
            parts.append({"start": left, "end": right, "labels": labels or [NO_LABEL]})

        for _ in range(2):
            parts = self._merge_short_parts(parts, min_duration=2.0)

        return np.array([p["start"] for p in parts]), [random.choice(p["labels"]) for p in parts]

    @staticmethod
    def _merge_short_parts(parts: List[Dict[str, Any]], min_duration: float) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for index, part in enumerate(parts):
            if part["end"] - part["start"] >= min_duration:
                merged.append(part)
                continue
            has_prev, has_next = len(merged) > 0, index + 1 < len(parts)
            if has_prev and (not has_next or random.choice([True, False])):
                merged[-1]["end"] = part["end"]
            elif has_next:
                parts[index + 1]["start"] = part["start"]
        return merged

    def _build_targets(
        self,
        times: np.ndarray,
        labels: List[str],
        time_range: Tuple[float, float],
        supervise_boundary: bool,
        mask_no_label_boundaries: bool,
    ) -> Optional[Dict[str, Any]]:
        """Frame targets for one slice. ``times`` are relative to the slice start."""
        time_l, time_r = time_range
        inner = np.flatnonzero((time_l + _EPS < times) & (times < time_r - _EPS))
        if len(inner) == 0:
            return None

        # Segments inside the slice: [(start, label), ..., (time_r, "end")]
        msa_info: MsaInfo = [(float(max(times[inner[0] - 1], time_l)), labels[inner[0] - 1])]
        msa_info += [(float(times[i]), labels[i]) for i in inner]
        msa_info.append((float(time_r), END_LABEL))

        n = self.num_frames
        padding_mask = np.ones(n, dtype=bool)
        padding_mask[self._time2frame(time_l) : self._time2frame(time_r)] = False

        boundaries = np.zeros(n, dtype=np.float64)
        boundaries[[self._time2frame(times[i]) for i in inner]] = 1.0

        function_targets = np.zeros((n, self.num_classes), dtype=np.float32)
        function_mask = np.zeros(n, dtype=bool)
        boundary_mask = np.zeros(n, dtype=bool) if supervise_boundary else np.ones(n, dtype=bool)

        for index, ((seg_start, label), (seg_end, _)) in enumerate(zip(msa_info[:-1], msa_info[1:])):
            start_frame, end_frame = self._time2frame(seg_start), self._time2frame(seg_end)
            function_targets[start_frame:end_frame, LABEL_TO_ID[label]] = 1.0
            if label != NO_LABEL:
                continue
            function_mask[start_frame:end_frame] = True
            if mask_no_label_boundaries:
                # Ignore boundaries inside unlabeled gaps, except within 4 s of labeled neighbours.
                left = 0 if index == 0 else self._time2frame(seg_start + 4)
                right = n if index == len(msa_info) - 2 else self._time2frame(seg_end - 4)
                if right > left:
                    boundary_mask[max(left, 0) : min(right, n)] = True

        return {
            "padding_mask": padding_mask,
            "boundary_targets": self._widen(boundaries).astype(np.float32),
            "function_targets": function_targets,
            "boundary_mask": boundary_mask,
            "function_mask": function_mask,
            "msa_info": msa_info,
        }

    def __getitem__(self, index: int) -> Optional[Dict[str, Any]]:
        subset_index, stem = self.items[index]
        subset = self.subsets[subset_index]
        song_id, start = split_chunk_id(stem)

        if subset.label_format == "segments":
            dur = int(math.ceil(self.slice_dur))
            times, labels = self._timeline_from_segments(subset.labels[song_id], float(start), float(start + dur))
            targets = self._build_targets(times - start, labels, (0.0, float(dur)), subset.supervise_boundary, mask_no_label_boundaries=True)
        else:
            times, labels = subset.labels[song_id]
            local = times - start
            time_range = (max(0.0, float(local.min())), min(float(self.slice_dur), float(local.max())))
            targets = self._build_targets(local, labels, time_range, subset.supervise_boundary, mask_no_label_boundaries=False)

        if targets is None:
            return None

        return {
            "data_id": f"{subset.name}_{stem}",
            "embeddings": load_features(subset.embedding_dirs, stem),
            "dataset_id": DATASET_TO_ID[subset.dataset],
            "label_mask": self.label_masks[subset.dataset],
            **targets,
        }


class SongFormBenchDataset(Dataset):
    """Whole songs for evaluation: all feature chunks of a song plus its reference annotation.

    Args:
        subsets: One dict per subset with ``name``, ``embedding_dirs`` and optionally ``label_dir``
            (``<song_id>.txt`` references). Without ``label_dir`` every song with features is used and
            ``msa_info`` is ``None`` (inference only).
        dataset: Key of :data:`DATASET_TO_ID` used for conditioning and the allowed label set.
        num_classes: Size of the functional label space.
    """

    def __init__(self, subsets: List[Dict[str, Any]], dataset: str = "SongForm-HX-8Class", num_classes: int = 128) -> None:
        super().__init__()
        self.dataset_id = DATASET_TO_ID[dataset]
        self.label_mask = label_mask(dataset, num_classes)
        self.items: List[Dict[str, Any]] = []

        logger = get_logger()
        for spec in subsets:
            embedding_dirs = list(spec["embedding_dirs"])
            chunks: Dict[str, List[int]] = defaultdict(list)
            for stem in set.intersection(*(_list_chunk_stems(d) for d in embedding_dirs)):
                song_id, start = split_chunk_id(stem)
                chunks[song_id].append(start)

            label_dir = spec.get("label_dir")
            if label_dir:
                songs = sorted(p.stem for p in Path(label_dir).glob("*.txt"))
                missing = [s for s in songs if s not in chunks]
                if missing:
                    logger.warning(f"{spec['name']}: no features for {len(missing)} songs, e.g. {missing[:3]}")
            else:
                songs = sorted(chunks)

            for song_id in songs:
                if song_id in chunks:
                    self.items.append(
                        {
                            "subset": spec["name"],
                            "song_id": song_id,
                            "starts": sorted(chunks[song_id]),
                            "embedding_dirs": embedding_dirs,
                            "label_path": str(Path(label_dir) / f"{song_id}.txt") if label_dir else None,
                        }
                    )
            logger.info(f"{spec['name']}: {sum(1 for x in self.items if x['subset'] == spec['name'])} songs")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            "subset": item["subset"],
            "song_id": item["song_id"],
            "chunks": [(start, load_features(item["embedding_dirs"], f"{item['song_id']}_{start}")) for start in item["starts"]],
            "msa_info": load_msa_txt(item["label_path"]) if item["label_path"] else None,
            "dataset_id": self.dataset_id,
            "label_mask": self.label_mask,
        }
