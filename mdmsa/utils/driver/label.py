"""Label-only SongFormer drivers: frame-wise functional labels without boundary prediction.

Segments are the runs of equal frame labels, so boundaries in the written estimations come from label changes only.
Used with the compact label datasets (:mod:`mdmsa.utils.data.compact`); ``train.labels`` gives the class names.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...amp import autocast
from ..data.labels import END_LABEL, MsaInfo
from ..postprocess import stitch_chunk_logits
from .songformer import SongFormerEvaluator, SongFormerTrainer, _to_device

__all__ = [
    "LabelSongFormerTrainer",
    "LabelSongFormerEvaluator",
    "frames_to_msa",
    "frame_label_accuracy",
]

Batch = Dict[str, Any]


def frames_to_msa(frame_labels: Sequence[int], label_names: Sequence[str], frame_rate: float) -> MsaInfo:
    """Runs of equal frame labels as ``[(start, label), ..., (end, "end")]``."""
    labels = np.asarray(frame_labels)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    msa_info: MsaInfo = [(float(i / frame_rate), label_names[int(labels[i])]) for i in starts]
    msa_info.append((float(len(labels) / frame_rate), END_LABEL))
    return msa_info


def frame_label_accuracy(predictions: torch.Tensor, batch: Batch) -> List[float]:
    """Per-song accuracy of ``(B, T)`` predicted ids on labeled, unpadded frames."""
    targets = batch["function_targets"]
    valid = ~batch["padding_mask"] & ~batch["function_mask"] & (targets.sum(-1) > 0)
    correct = (predictions == targets.argmax(-1)) & valid
    return [float(c.sum() / v.sum()) for c, v in zip(correct, valid) if v.any()]


class LabelSongFormerTrainer(SongFormerTrainer):
    """:class:`SongFormerTrainer` whose validation scores frame label accuracy (``acc``) instead of decoded segments."""

    def _validation_outputs(self, model: nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        return self._forward(model, batch)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        model = self.ema.ema_model if self.ema is not None else self.model
        model.eval()

        losses: List[float] = []
        accuracies: List[float] = []
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue
            batch = _to_device(batch, self.device)
            with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                outputs = self._validation_outputs(model, batch)
                losses.append(float(self.criterion(outputs, batch)))
            logits = outputs["function_logits"].masked_fill(batch["label_mask"].unsqueeze(1), -float("inf"))
            accuracies += frame_label_accuracy(logits.argmax(-1), batch)
        return {"loss": float(np.mean(losses)), "acc": float(np.mean(accuracies))} if losses else {}


class LabelSongFormerEvaluator(SongFormerEvaluator):
    """:class:`SongFormerEvaluator` that labels every frame by the argmax of the stitched chunk scores.

    No boundary decoding and no rule-based post-processing; segments are runs of equal labels.
    """

    def chunk_scores(self, embeddings: torch.Tensor, dataset_ids: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        """``(T, K)`` scores of one chunk; averaged over overlapping chunks before the argmax."""
        with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
            outputs = self.model(embeddings, dataset_ids, label_mask=label_mask.unsqueeze(0))
        return outputs["function_logits"][0].float()

    @torch.no_grad()
    def predict(self, song: Dict[str, Any]) -> MsaInfo:
        dataset_ids = torch.tensor([song["dataset_id"]], device=self.device)
        label_mask = torch.from_numpy(song["label_mask"]).to(self.device)

        chunks = []
        for start, embeddings in song["chunks"]:
            scores = self.chunk_scores(torch.from_numpy(embeddings).unsqueeze(0).to(self.device), dataset_ids, label_mask)
            chunks.append((start, {"boundary_logits": torch.zeros(scores.shape[0], device=self.device), "function_logits": scores}))

        frame_rate = self.config.train.postprocess.frame_rate
        scores = stitch_chunk_logits(chunks, frame_rate=frame_rate)["function_logits"]
        scores = scores.masked_fill(label_mask.unsqueeze(0), -float("inf"))
        return frames_to_msa(scores.argmax(-1).tolist(), list(self.config.train.labels), frame_rate)
