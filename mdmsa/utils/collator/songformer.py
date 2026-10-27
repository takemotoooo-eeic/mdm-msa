from typing import Any, Dict, List, Optional

import numpy as np
import torch

__all__ = [
    "SongFormerCollator",
    "SongCollator",
]


class SongFormerCollator:
    """Pads :class:`mdmsa.utils.data.SongFormDBDataset` slices into a batch.

    Features are padded to the longest slice (input rate) and frame targets to ``T_in // downsample_rate``
    (output rate). Padded frames have ``padding_mask = True``. Slices the dataset returned as ``None`` are dropped.

    Args:
        downsample_rate: Ratio between input and output frame rates of the model.
    """

    def __init__(self, downsample_rate: int = 3) -> None:
        self.downsample_rate = downsample_rate

    def __call__(self, batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        batch = [x for x in batch if x is not None]
        if not batch:
            return None

        batch_size = len(batch)
        input_length = max(x["embeddings"].shape[0] for x in batch)
        length = input_length // self.downsample_rate
        feature_dim = batch[0]["embeddings"].shape[1]
        num_classes = batch[0]["function_targets"].shape[1]

        embeddings = np.zeros((batch_size, input_length, feature_dim), dtype=np.float32)
        padding_mask = np.ones((batch_size, length), dtype=bool)
        boundary_targets = np.zeros((batch_size, length), dtype=np.float32)
        function_targets = np.zeros((batch_size, length, num_classes), dtype=np.float32)
        boundary_mask = np.zeros((batch_size, length), dtype=bool)
        function_mask = np.zeros((batch_size, length), dtype=bool)

        for i, item in enumerate(batch):
            embeddings[i, : item["embeddings"].shape[0]] = item["embeddings"]
            n = min(length, item["padding_mask"].shape[0])
            padding_mask[i, :n] = item["padding_mask"][:n]
            boundary_targets[i, :n] = item["boundary_targets"][:n]
            function_targets[i, :n] = item["function_targets"][:n]
            boundary_mask[i, :n] = item["boundary_mask"][:n]
            function_mask[i, :n] = item["function_mask"][:n]

        return {
            "data_ids": [x["data_id"] for x in batch],
            "embeddings": torch.from_numpy(embeddings),
            "padding_mask": torch.from_numpy(padding_mask),
            "boundary_targets": torch.from_numpy(boundary_targets),
            "function_targets": torch.from_numpy(function_targets),
            "boundary_mask": torch.from_numpy(boundary_mask),
            "function_mask": torch.from_numpy(function_mask),
            "dataset_ids": torch.tensor([x["dataset_id"] for x in batch], dtype=torch.long),
            "label_mask": torch.from_numpy(np.stack([x["label_mask"] for x in batch])),
            "msa_infos": [x["msa_info"] for x in batch],
        }


class SongCollator:
    """Batch of one whole song (:class:`mdmsa.utils.data.SongFormBenchDataset`): returns the item unchanged."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("SongCollator supports batch_size=1 only.")
        return batch[0]
