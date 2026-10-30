"""Datasets whose functional label space is a compact list of labels (e.g. the 8 classes of SongFormDB-HX).

:class:`SongFormDBDataset` / :class:`SongFormBenchDataset` use the 128-way upstream id space, most of which never
occurs in the data. Discrete diffusion builds transition matrices over the label space, so unused classes would
receive uniform-noise mass. These wrappers keep the parent datasets unchanged and only re-index the targets:
class ``i`` is ``labels[i]``.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .labels import LABEL_TO_ID
from .songformer import SongFormBenchDataset, SongFormDBDataset

__all__ = [
    "CompactLabelSongFormDBDataset",
    "CompactLabelSongFormBenchDataset",
]

_UPSTREAM_NUM_CLASSES = 128


class CompactLabelSongFormDBDataset(SongFormDBDataset):
    """:class:`SongFormDBDataset` with ``function_targets`` ``(T, len(labels))``.

    Frames whose label is not in ``labels`` (and ``NO_LABEL`` frames) get an all-zero target and
    ``function_mask = True``, so they are excluded from functional losses.

    Args:
        labels: Label names in class order.
        **kwargs: :class:`SongFormDBDataset` arguments (``num_classes`` is fixed to the upstream 128).
    """

    def __init__(self, labels: List[str], **kwargs: Any) -> None:
        kwargs.pop("num_classes", None)
        super().__init__(num_classes=_UPSTREAM_NUM_CLASSES, **kwargs)
        self.labels = list(labels)
        self.label_ids = np.array([LABEL_TO_ID[label] for label in self.labels])

    def __getitem__(self, index: int) -> Optional[Dict[str, Any]]:
        item = super().__getitem__(index)
        if item is None:
            return None
        targets = item["function_targets"][:, self.label_ids]
        unlabeled = (targets.sum(-1) == 0) & ~item["padding_mask"]
        return {
            **item,
            "function_targets": targets,
            "function_mask": item["function_mask"] | unlabeled,
            "label_mask": np.zeros(len(self.labels), dtype=bool),
        }


class CompactLabelSongFormBenchDataset(SongFormBenchDataset):
    """:class:`SongFormBenchDataset` whose ``label_mask`` covers ``len(labels)`` classes (all allowed).

    Args:
        labels: Label names in class order.
        **kwargs: :class:`SongFormBenchDataset` arguments.
    """

    def __init__(self, labels: List[str], **kwargs: Any) -> None:
        kwargs.pop("num_classes", None)
        super().__init__(num_classes=_UPSTREAM_NUM_CLASSES, **kwargs)
        self.labels = list(labels)
        self.label_mask = np.zeros(len(self.labels), dtype=bool)
