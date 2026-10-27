"""Label and dataset-id tables for SongFormer.

Ids match upstream ``dataset/label2id.py`` (the classifier head has 128 outputs and the dataset
embedding 64 rows), but only labels / datasets that appear in SongFormDB and SongFormBench are listed.
"""

from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "MsaInfo",
    "END_LABEL",
    "NO_LABEL",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "DATASET_TO_ID",
    "DATASET_ALLOWED_LABELS",
    "label_mask",
]

# [(start_time, label), ..., (end_time, "end")]
MsaInfo = List[Tuple[float, str]]

END_LABEL = "end"
# Frames with this label are excluded from the functional loss.
NO_LABEL = "NO_LABEL"

LABEL_TO_ID: Dict[str, int] = {
    "intro": 0,
    "verse": 1,
    "chorus": 2,
    "bridge": 3,
    "inst": 4,
    "outro": 5,
    "silence": 6,
    "pre-chorus": 26,
    "interlude": 39,
    "ending": 64,
    "no-vocal-intro": 65,
    "no-vocal-interlude": 66,
    "no-vocal-outro": 67,
    NO_LABEL: 68,
}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

DATASET_TO_ID: Dict[str, int] = {
    "SongForm-Private": 3,  # SongFormDB-Ext
    "SongForm-HX-8Class": 5,  # SongFormDB-HX / SongFormBench (pre-chorus kept)
    "SongForm-Hook": 6,  # SongFormDB-Hook
    "SongForm-Gem": 7,  # SongFormDB-Gem (labels + boundaries)
    "SongForm-Gem-Only-Label": 8,  # SongFormDB-Gem (labels only)
}

_EIGHT_CLASSES = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence", "pre-chorus"]
DATASET_ALLOWED_LABELS: Dict[str, List[str]] = {
    "SongForm-Private": [*_EIGHT_CLASSES, "interlude", "ending", "no-vocal-intro", "no-vocal-interlude", "no-vocal-outro"],
    "SongForm-HX-8Class": _EIGHT_CLASSES,
    "SongForm-Hook": _EIGHT_CLASSES,
    "SongForm-Gem": _EIGHT_CLASSES,
    "SongForm-Gem-Only-Label": _EIGHT_CLASSES,
}


def label_mask(dataset: str, num_classes: int) -> np.ndarray:
    """``(num_classes,)`` bool array, ``True`` at classes not used by ``dataset``."""
    mask = np.ones(num_classes, dtype=bool)
    mask[[LABEL_TO_ID[label] for label in DATASET_ALLOWED_LABELS[dataset]]] = False
    return mask
