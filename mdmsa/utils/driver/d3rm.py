"""Training / evaluation of :class:`mdmsa.models.SongFormerD3RM` with :class:`mdmsa.models.D3RMDiffusion`.

Reference: https://github.com/hanshounsu/d3rm, ``transcription/diffusion/trainer.py``
(``DiscreteDiffusion.forward`` for training, ``DiscreteDiffusion.sample`` for inference). As in the reference, the audio
encoder output is computed once and reused for every denoising step.
"""

from typing import Any, Callable, Dict, List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...amp import autocast
from .label import LabelSongFormerEvaluator, LabelSongFormerTrainer, frame_label_accuracy
from .songformer import _to_device

__all__ = [
    "D3RMTrainer",
    "D3RMEvaluator",
]

Batch = Dict[str, Any]


def _denoiser(model: nn.Module, embeddings: torch.Tensor, dataset_ids: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> Callable:
    audio = model.encode_audio(embeddings, dataset_ids)
    return lambda tokens, timesteps: model.denoise(audio, tokens, timesteps, padding_mask=padding_mask)


class D3RMTrainer(LabelSongFormerTrainer):
    """Trains the denoiser on ``train.diffuser`` (a :class:`mdmsa.models.D3RMDiffusion`).

    The model output dict holds ``vb_loss`` / ``aux_loss`` (per song) for :mod:`mdmsa.loss.d3rm`, and
    ``function_logits`` = ``log p(x_0 | x_t)`` at the sampled step.

    Validation (EMA model, fixed seed) reports the loss at random steps, ``acc_onestep`` (argmax of ``p(x_0 | x_T)``,
    all frames masked) and, when ``train.validation.sample`` is true, ``acc`` of full ``T``-step ancestral sampling on
    at most ``train.validation.max_sample_batches`` slices (``null``: all). Without sampling ``acc`` is ``acc_onestep``.
    """

    diffuser: nn.Module

    def _forward(self, model: nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        valid = ~batch["padding_mask"] & ~batch["function_mask"] & (batch["function_targets"].sum(-1) > 0)
        x_start = batch["function_targets"].argmax(-1)
        denoise = _denoiser(model, batch["embeddings"], batch["dataset_ids"], batch["padding_mask"])
        out = self.diffuser.training_losses(denoise, x_start, valid, update_history=model.training)
        return {"vb_loss": out["vb_loss"], "aux_loss": out["aux_loss"], "function_logits": out["log_x0_recon"][:, :-1, :].transpose(1, 2)}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        model = self.ema.ema_model if self.ema is not None else self.model
        model.eval()
        validation_config = self.config.train.validation
        do_sample = bool(validation_config.get("sample", True))
        max_sample_batches = validation_config.get("max_sample_batches")

        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.config.train.seed)

        rows: Dict[str, List[float]] = {"loss": [], "vb_loss": [], "aux_loss": []}
        acc_onestep: List[float] = []
        acc_sample: List[float] = []
        for index, batch in enumerate(tqdm(dataloader, desc="Validation")):
            if batch is None:
                continue
            batch = _to_device(batch, self.device)
            with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                outputs = self._forward(model, batch)
                rows["loss"].append(float(self.criterion(outputs, batch)))
                rows["vb_loss"].append(float(outputs["vb_loss"].mean()))
                rows["aux_loss"].append(float(outputs["aux_loss"].mean()))

                denoise = _denoiser(model, batch["embeddings"], batch["dataset_ids"], batch["padding_mask"])
                b, length = batch["padding_mask"].shape
                if do_sample and (max_sample_batches is None or index < max_sample_batches):
                    sampled = self.diffuser.sample(denoise, b, length, self.device)
                    acc_onestep += frame_label_accuracy(sampled["x0_first"], batch)
                    acc_sample += frame_label_accuracy(sampled["tokens"], batch)
                else:
                    tokens = torch.full((b, length), self.diffuser.mask_token, device=self.device, dtype=torch.long)
                    t = torch.full((b,), self.diffuser.num_timesteps - 1, device=self.device, dtype=torch.long)
                    acc_onestep += frame_label_accuracy(denoise(tokens, t).argmax(-1), batch)

        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

        if not rows["loss"]:
            return {}
        metrics = {key: float(np.mean(values)) for key, values in rows.items()}
        metrics["acc_onestep"] = float(np.mean(acc_onestep))
        metrics["acc"] = float(np.mean(acc_sample)) if acc_sample else metrics["acc_onestep"]
        return metrics

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "D3RMTrainer":
        trainer = super().build_from_config(config)
        trainer.diffuser = hydra.utils.instantiate(config.train.diffuser).to(trainer.device)
        return trainer


class D3RMEvaluator(LabelSongFormerEvaluator):
    """Labels every chunk by ``T``-step ancestral sampling (``train.evaluation.prediction: sample``) or by the
    one-step prediction ``argmax p(x_0 | x_T)`` (``onestep``). Chunk one-hots are averaged where chunks overlap."""

    diffuser: nn.Module

    def chunk_scores(self, embeddings: torch.Tensor, dataset_ids: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        prediction = self.config.train.evaluation.get("prediction", "sample")
        length = embeddings.shape[1] // self.model.downsample_rate
        with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
            denoise = _denoiser(self.model, embeddings, dataset_ids, None)
            if prediction == "sample":
                tokens = self.diffuser.sample(denoise, 1, length, self.device)["tokens"][0]
            elif prediction == "onestep":
                masked = torch.full((1, length), self.diffuser.mask_token, device=self.device, dtype=torch.long)
                t = torch.full((1,), self.diffuser.num_timesteps - 1, device=self.device, dtype=torch.long)
                tokens = denoise(masked, t)[0].argmax(-1)
            else:
                raise ValueError(f"Unknown train.evaluation.prediction: {prediction}")
        return torch.nn.functional.one_hot(tokens, self.model.num_classes).float()

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "D3RMEvaluator":
        evaluator = super().build_from_config(config)
        evaluator.diffuser = hydra.utils.instantiate(config.train.diffuser).to(evaluator.device)
        return evaluator
