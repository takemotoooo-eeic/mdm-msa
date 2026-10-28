"""Training and evaluation drivers for SongFormer music structure analysis."""

import copy
import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...amp import autocast, get_autocast_device_type, should_enable_amp
from ...amp.grad_scaler import GradScaler
from ...loss import TotalLoss
from ...models.diffuser import MaskedDiffuser
from .. import convert_dtype, set_device, set_seed
from .._omegaconf import replace_missing_with_none
from .._tensorboard import get_writer
from ..data.labels import MsaInfo
from ..logging import get_logger
from ..metrics import label_overlaps, replace_label, segment_metrics, summarize
from ..postprocess import decode_structure, rule_post_processing, stitch_chunk_logits
from ..wandb import DummyWandbLogger, WandbLogger
from .balancer import Balancer
from .base import Driver
from .ema import ExponentialMovingAverage

__all__ = [
    "SongFormerTrainer",
    "SongFormerEvaluator",
    "load_model_weights",
]

Batch = Dict[str, Any]


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


_TOKEN_KEYS = ("boundary_tokens", "function_tokens")


def _forward(model: nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
    """Training forward. Token inputs are passed when the batch has them (masked diffusion)."""
    tokens = {key: batch[key] for key in _TOKEN_KEYS if key in batch}
    return model(batch["embeddings"], batch["dataset_ids"], padding_mask=batch["padding_mask"], **tokens)


def _infer(
    model: nn.Module,
    diffuser: Optional[MaskedDiffuser],
    embeddings: torch.Tensor,
    dataset_ids: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Raw logits (regression) or denoised tokens plus last-step logits (masked diffusion)."""
    if diffuser is None:
        return model(embeddings, dataset_ids, padding_mask=padding_mask)
    return diffuser.inference(model, embeddings, dataset_ids, padding_mask=padding_mask, label_mask=label_mask)


def _decode(outputs: Dict[str, torch.Tensor], label_mask: torch.Tensor, postprocess: DictConfig, diffuser: Optional[MaskedDiffuser]) -> MsaInfo:
    """Segments of one song from its (unbatched) outputs."""
    if diffuser is not None:
        return diffuser.decode(outputs)
    return decode_structure(
        outputs["boundary_logits"],
        outputs["function_logits"].masked_fill(label_mask.unsqueeze(0), -float("inf")),
        frame_rate=postprocess.frame_rate,
        local_maxima_filter_size=postprocess.local_maxima_filter_size,
        peak_window=postprocess.peak_window,
    )


def _build_diffuser(config: DictConfig) -> Optional[MaskedDiffuser]:
    return hydra.utils.instantiate(config.diffuser) if config.get("diffuser") else None


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_model_weights(model: nn.Module, path: str, use_ema: bool = True) -> str:
    """Load SongFormer weights from a checkpoint of this repository or of upstream SongFormer.

    Supported: ``*.safetensors`` (upstream EMA weights), upstream ``*.pt`` (``model`` / ``model_ema``) and
    checkpoints written by :class:`SongFormerTrainer` (``model`` / ``ema``).

    Returns:
        Which weights were loaded (``"ema"`` or ``"model"``).
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint: Dict[str, Any] = {"ema": load_file(path)}
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    ema_state = checkpoint.get("ema", checkpoint.get("model_ema"))
    if use_ema and ema_state is not None:
        model.load_state_dict(_strip_prefix(ema_state, "ema_model."))
        return "ema"
    model.load_state_dict(checkpoint["model"])
    return "model"


def _loss_name(module: nn.Module) -> str:
    inner = module.loss_module
    if isinstance(inner, TotalLoss):
        return "+".join(_loss_name(x) for x in inner.losses)
    return type(inner).__name__


class SongFormerTrainer(Driver):
    """Iteration-based trainer with gradient accumulation, EMA weights, a loss balancer and early stopping.

    The top-level terms of ``criterion`` (a :class:`mdmsa.loss.TotalLoss`) are combined by ``balancer`` when one is
    given, otherwise summed. Validation runs on the EMA model every ``train.validation.every`` optimizer steps.

    With a ``diffuser`` (masked diffusion), each training batch is corrupted by :meth:`MaskedDiffuser.add_mask` and
    validation decodes by :meth:`MaskedDiffuser.inference`; without one, the model is a regressor.
    """

    def __init__(
        self,
        *,
        training_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader],
        model: nn.Module,
        criterion: TotalLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        ema: Optional[ExponentialMovingAverage],
        balancer: Optional[Balancer],
        config: DictConfig,
        device: torch.device,
        diffuser: Optional[MaskedDiffuser] = None,
    ) -> None:
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.balancer = balancer
        self.diffuser = diffuser
        self.config = config
        self.loss_names = [_loss_name(loss) for loss in criterion.losses]

        self._reset(config, device=device)

    def _reset(self, config: DictConfig, device: torch.device) -> None:
        training_config = config.train

        self.device = device
        self.dtype = convert_dtype(training_config.torch_dtype)
        self.enable_amp = should_enable_amp(self.dtype)
        self.device_type = get_autocast_device_type()
        self.scaler = GradScaler(device=self.device_type, enabled=self.enable_amp)

        os.makedirs(training_config.output.tensorboard_dir, exist_ok=True)
        self.logger = get_logger()
        self.writer = get_writer(log_dir=training_config.output.tensorboard_dir)
        self.wandb_logger: Union[WandbLogger, DummyWandbLogger] = hydra.utils.instantiate(
            training_config.wandb,
            run_config=OmegaConf.to_container(config, resolve=True),
            model=self.model,
            _recursive_=False,
        )
        self.set_commit_hash()

        self.epoch = 0
        self.iteration = 0
        self.best_scores: Dict[str, float] = {}
        self.num_bad_validations = 0
        self.stopped_early = False

        self.logger.info(self.model)
        self.logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        if training_config.checkpoint.resume_from:
            self.load_checkpoint(training_config.checkpoint.resume_from)

    def _log_scalars(self, metrics: Dict[str, float], prefix: str) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, global_step=self.iteration)
        if self.wandb_logger.enabled:
            self.wandb_logger.add_scalars({f"{prefix}/{k}": v for k, v in metrics.items()}, global_step=self.iteration)

    def _finished(self) -> bool:
        steps = self.config.train.steps
        if self.stopped_early:
            return True
        if steps.iterations is not None and self.iteration >= steps.iterations:
            return True
        return steps.epochs is not None and self.epoch >= steps.epochs

    def _optimizer_step(self) -> None:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.iteration += 1

    def train_for_epoch(self, dataloader: DataLoader) -> None:
        training_config = self.config.train
        accumulation = training_config.gradient_accumulation_steps
        validate_every = training_config.validation.every

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        for batch in pbar:
            if batch is None:
                continue
            batch = _to_device(batch, self.device)
            if self.diffuser is not None:
                batch = self.diffuser.add_mask(batch)

            with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                outputs = _forward(self.model, batch)
                components = [loss(outputs, batch) for loss in self.criterion.losses]

            if self.balancer is not None:
                total = self.balancer(components, list(self.model.parameters()))
            else:
                total = sum(components)

            self.scaler.scale(total / accumulation).backward()
            micro_step += 1
            if micro_step % accumulation != 0:
                continue

            self._optimizer_step()

            if self.iteration % training_config.log_every == 0:
                metrics = {"loss": float(total.detach()), **{name: float(c.detach()) for name, c in zip(self.loss_names, components)}}
                metrics.update({f"learning_rate_{i}": group["lr"] for i, group in enumerate(self.optimizer.param_groups)})
                if self.balancer is not None:
                    metrics.update({f"balancer_{k}": v for k, v in self.balancer.metrics.items()})
                self._log_scalars(metrics, prefix="training")
            pbar.set_postfix({"iter": self.iteration, "loss": f"{float(total.detach()):.4f}"})

            if self.validation_dataloader is not None and self.iteration % validate_every == 0:
                self._validate_and_save()
                self.model.train()

            if self._finished():
                return

        self.epoch += 1

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Slice-level validation: loss and segment metrics of decoded predictions (EMA model if available)."""
        model = self.ema.ema_model if self.ema is not None else self.model
        model.eval()
        postprocess = self.config.train.postprocess

        rows: List[Dict[str, float]] = []
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue
            batch = _to_device(batch, self.device)
            with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                outputs = _infer(model, self.diffuser, batch["embeddings"], batch["dataset_ids"], batch["padding_mask"], batch["label_mask"])
                loss = float(self.criterion(outputs, batch))

            for index, reference in enumerate(batch["msa_infos"]):
                try:
                    estimate = _decode({k: v[index] for k, v in outputs.items()}, batch["label_mask"][index], postprocess, self.diffuser)
                    rows.append({"loss": loss, **segment_metrics(reference, estimate)})
                except ValueError as error:
                    self.logger.warning(f"Skip {batch['data_ids'][index]} in validation: {error}")

        keys = ["loss", "HitRate_0.5F", "HitRate_3F", "HitRate_1F", "PWF", "Sf", "acc"]
        return {key: float(np.mean([row[key] for row in rows])) for key in keys} if rows else {}

    def _validate_and_save(self) -> None:
        training_config = self.config.train
        metrics = self.validate(self.validation_dataloader)
        self._log_scalars(metrics, prefix="validation")
        self.logger.info(f"[Iteration {self.iteration}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        improved = False
        for key in training_config.validation.monitor:
            if metrics.get(key, -float("inf")) > self.best_scores.get(key, -float("inf")):
                self.best_scores[key] = metrics[key]
                improved = True
        self.num_bad_validations = 0 if improved else self.num_bad_validations + 1

        checkpoint_config = training_config.output.model
        self.save_checkpoint(checkpoint_config.iteration.path.format(iteration=self.iteration))
        self.save_checkpoint(checkpoint_config.last.path.format(iteration=self.iteration))
        if improved:
            self.save_checkpoint(checkpoint_config.best.path.format(iteration=self.iteration))

        patience = training_config.validation.early_stopping_patience
        if patience is not None and self.num_bad_validations >= patience:
            self.logger.info(f"Early stopping: no improvement in {patience} validations.")
            self.stopped_early = True

    def run(self) -> Dict[str, float]:
        steps = self.config.train.steps
        if steps.iterations is None and steps.epochs is None:
            raise ValueError("Set config.train.steps.iterations and/or config.train.steps.epochs.")

        while not self._finished():
            self.train_for_epoch(self.training_dataloader)

        if self.validation_dataloader is not None and self.iteration % self.config.train.validation.every != 0:
            self._validate_and_save()

        self.writer.close()
        if self.wandb_logger.enabled:
            self.wandb_logger.finish()
        return self.best_scores

    def save_checkpoint(self, path: str) -> None:
        from ... import __version__ as _version

        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "best_scores": self.best_scores,
            "num_bad_validations": self.num_bad_validations,
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

        config = copy.deepcopy(self.config)
        replace_missing_with_none(config)
        checkpoint["resolved_config"] = OmegaConf.to_container(config, resolve=True)
        checkpoint["_metadata"] = {"version": _version, "driver": self.__class__.__name__, "commit_hash": self.commit_hash}

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        self.logger.info(f"Resume from {path}.")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint.get("epoch", 0)
        self.iteration = checkpoint.get("iteration", 0)
        self.best_scores = checkpoint.get("best_scores", {})
        self.num_bad_validations = checkpoint.get("num_bad_validations", 0)
        return checkpoint

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "SongFormerTrainer":
        training_config = config.train
        set_seed(training_config.seed)
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        training_dataloader = hydra.utils.instantiate(config.dataloader.train)
        validation_dataloader = hydra.utils.instantiate(config.dataloader.validate) if config.dataloader.validate.get("_target_") else None

        model = set_device(hydra.utils.instantiate(config.model), accelerator=accelerator)
        criterion = set_device(hydra.utils.instantiate(config.loss), accelerator=accelerator)

        optimizer = hydra.utils.instantiate(config.optimizer.optimizer, model.parameters())
        scheduler_config = config.optimizer.scheduler
        scheduler = hydra.utils.instantiate(scheduler_config, optimizer) if scheduler_config.get("_target_") else None

        ema = hydra.utils.instantiate(training_config.ema, model) if training_config.get("ema") else None
        balancer = hydra.utils.instantiate(training_config.balancer) if training_config.get("balancer") else None

        return cls(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            balancer=balancer,
            config=config,
            device=next(model.parameters()).device,
            diffuser=_build_diffuser(config),
        )


class SongFormerEvaluator(Driver):
    """Whole-song inference on pre-extracted features, with scoring when references are available.

    For each song, chunk outputs (logits, or tokens with a ``diffuser``) are stitched, decoded into segments
    (optionally with rule-based cleanup) and written to ``<output_dir>/<subset>/estimations/<song_id>.txt``. With
    references, per-song metrics and summaries are written for every entry of ``train.evaluation.label_mappings``
    (e.g. pre-chorus -> verse).
    """

    def __init__(
        self, *, dataloader: DataLoader, model: nn.Module, config: DictConfig, device: torch.device, diffuser: Optional[MaskedDiffuser] = None
    ) -> None:
        self.dataloader = dataloader
        self.model = model
        self.diffuser = diffuser
        self.config = config
        self.device = device

        training_config = config.train
        self.dtype = convert_dtype(training_config.torch_dtype)
        self.enable_amp = should_enable_amp(self.dtype)
        self.device_type = get_autocast_device_type()
        self.logger = get_logger()
        self.set_commit_hash()

        checkpoint_path = training_config.checkpoint.resume_from
        if not checkpoint_path:
            raise ValueError("Set train.checkpoint.resume_from to a checkpoint path for evaluation.")
        loaded = load_model_weights(self.model, checkpoint_path, use_ema=training_config.evaluation.use_ema)
        self.logger.info(f"Loaded {loaded} weights from {checkpoint_path}.")

    @torch.no_grad()
    def predict(self, song: Dict[str, Any]) -> MsaInfo:
        """Segments of one song from :class:`mdmsa.utils.data.SongFormBenchDataset`."""
        dataset_ids = torch.tensor([song["dataset_id"]], device=self.device)
        label_mask = torch.from_numpy(song["label_mask"]).to(self.device)

        chunks: List[Tuple[float, Dict[str, torch.Tensor]]] = []
        for start, embeddings in song["chunks"]:
            with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                embeddings = torch.from_numpy(embeddings).unsqueeze(0).to(self.device)
                outputs = _infer(self.model, self.diffuser, embeddings, dataset_ids, label_mask=label_mask.unsqueeze(0))
            chunks.append((start, {k: v[0] for k, v in outputs.items()}))

        postprocess = self.config.train.postprocess
        stitched = stitch_chunk_logits(chunks, frame_rate=postprocess.frame_rate)
        estimate = _decode(stitched, label_mask, postprocess, self.diffuser)
        if self.config.train.evaluation.rule_post_processing:
            estimate = rule_post_processing(estimate)
        return estimate

    def run(self) -> Dict[str, Dict[str, float]]:
        evaluation_config = self.config.train.evaluation
        output_dir = evaluation_config.output_dir
        mappings = dict(evaluation_config.label_mappings)
        self.model.eval()

        per_song: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        overlaps: Dict[Tuple[str, str], List[Dict[str, Tuple[float, float]]]] = defaultdict(list)

        for song in tqdm(self.dataloader, desc="Evaluation"):
            estimate = self.predict(song)

            estimation_dir = os.path.join(output_dir, song["subset"], "estimations")
            os.makedirs(estimation_dir, exist_ok=True)
            with open(os.path.join(estimation_dir, f"{song['song_id']}.txt"), "w") as f:
                f.write("\n".join(f"{t:.3f} {label}" for t, label in estimate) + "\n")

            if song["msa_info"] is None:
                continue
            for name, target in mappings.items():
                reference = replace_label(song["msa_info"], "pre-chorus", target)
                mapped = replace_label(estimate, "pre-chorus", target)
                try:
                    metrics = segment_metrics(reference, mapped)
                except ValueError as error:
                    self.logger.warning(f"Skip {song['song_id']} ({name}): {error}")
                    continue
                per_song[(song["subset"], name)].append({"id": song["song_id"], **metrics})
                overlaps[(song["subset"], name)].append(label_overlaps(reference, mapped))

        summaries: Dict[str, Dict[str, float]] = {}
        for (subset, name), rows in sorted(per_song.items()):
            result_dir = os.path.join(output_dir, subset, name)
            os.makedirs(result_dir, exist_ok=True)
            self._write_csv(os.path.join(result_dir, "per_song.csv"), rows)

            summary = summarize(rows, overlaps[(subset, name)])
            self._write_csv(os.path.join(result_dir, "summary.csv"), [summary])
            summaries[f"{subset}/{name}"] = summary
            self.logger.info(f"[{subset}/{name}] " + ", ".join(f"{k}={summary[k]:.4f}" for k in ["HR.5F", "HR3F", "HR1F", "PWF", "Sf", "acc", "iou"]))

        if summaries:
            self._write_markdown(os.path.join(output_dir, "summary.md"), summaries)
        self.logger.info(f"Results saved to {output_dir}")
        return summaries

    @staticmethod
    def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_markdown(path: str, summaries: Dict[str, Dict[str, float]]) -> None:
        keys = ["num_samples", "HR.5F", "HR3F", "HR1F", "PWF", "Sf", "acc", "iou"]
        lines = ["| subset/mapping | " + " | ".join(keys) + " |", "|" + " --- |" * (len(keys) + 1)]
        for name, summary in summaries.items():
            lines.append(f"| {name} | " + " | ".join(f"{summary[k]:.4f}" for k in keys) + " |")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "SongFormerEvaluator":
        set_seed(config.train.seed)
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        dataloader = hydra.utils.instantiate(config.dataloader.test)
        model = set_device(hydra.utils.instantiate(config.model), accelerator=accelerator)
        return cls(dataloader=dataloader, model=model, config=config, device=next(model.parameters()).device, diffuser=_build_diffuser(config))
