"""Training and evaluation drivers for image classification."""

import copy
import os
from typing import Any, Dict, Optional, Tuple, Union

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...amp import autocast, get_autocast_device_type, should_enable_amp
from ...amp.grad_scaler import GradScaler
from .. import convert_dtype, set_device, set_seed
from .._omegaconf import replace_missing_with_none
from .._tensorboard import get_writer
from ..logging import get_logger
from ..wandb import DummyWandbLogger, WandbLogger
from .base import Driver


class ClassifierTrainer(Driver):
    """Trainer for classification models."""

    def __init__(
        self,
        *,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: DictConfig,
        device: torch.device,
    ) -> None:
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self._reset(config, device=device)

    def _reset(
        self,
        config: DictConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        training_config = config.train

        if device is None:
            device = next(self.model.parameters()).device

        dtype = convert_dtype(training_config.torch_dtype)
        enable_amp = should_enable_amp(dtype)
        device_type = get_autocast_device_type()

        tensorboard_dir = os.path.join(training_config.output.tensorboard_dir)
        os.makedirs(tensorboard_dir, exist_ok=True)

        logger = get_logger()
        writer = get_writer(log_dir=tensorboard_dir)
        wandb_logger = hydra.utils.instantiate(
            training_config.wandb,
            run_config=OmegaConf.to_container(config, resolve=True),
            model=self.model,
            _recursive_=False,
        )

        self.set_commit_hash()

        self.device = device
        self.dtype = dtype
        self.enable_amp = enable_amp
        self.device_type = device_type
        self.scaler = GradScaler(device=device_type, enabled=enable_amp)

        self.logger = logger
        self.writer = writer
        self.wandb_logger: Union[WandbLogger, DummyWandbLogger] = wandb_logger

        self.epoch = 0
        self.iteration = 0
        self.best_validation_accuracy = -float("inf")
        self.history = {
            "training_loss": [],
            "training_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }
        for index, _ in enumerate(self.optimizer.param_groups):
            self.history[f"learning_rate_{index}"] = []

        self.logger.info(self.model)

        if training_config.checkpoint.resume_from:
            self.load_checkpoint(training_config.checkpoint.resume_from)

    def _prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        return images.to(self.device), labels.to(self.device)

    def _log_scalar(
        self,
        tag: str,
        value: float,
        global_step: int,
        wandb_step: Optional[int] = None,
    ) -> None:
        self.writer.add_scalar(tag, value, global_step=global_step)
        if self.wandb_logger.enabled:
            self.wandb_logger.add_scalar(tag, value, global_step=wandb_step or global_step)

    def _log_images(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        global_step: int,
        wandb_step: Optional[int] = None,
    ) -> None:
        mean = 0.1307
        std = 0.3081
        vis = images[:8].detach().cpu() * std + mean
        vis = vis.clamp(0.0, 1.0)

        self.writer.add_images("samples/images", vis, global_step=global_step)

        if self.wandb_logger.enabled:
            step = wandb_step or global_step
            caption = " | ".join(f"y={int(label)} p={int(pred)}" for label, pred in zip(labels[:8].cpu(), predictions[:8].cpu()))
            self.wandb_logger.add_image("samples/images", vis, global_step=step)
            self.logger.info(f"Sample predictions: {caption}")

    def train_for_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")

        for batch in pbar:
            images, labels = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            average_loss = total_loss / total_samples
            accuracy = total_correct / total_samples

            self.iteration += 1
            self._log_scalar("training_loss/iteration", loss.item(), global_step=self.iteration)

            for index, param_group in enumerate(self.optimizer.param_groups):
                self._log_scalar(
                    f"learning_rate_{index}/iteration",
                    param_group["lr"],
                    global_step=self.iteration,
                )

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{average_loss:.4f}",
                    "acc": f"{accuracy:.4f}",
                }
            )

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            "training_loss": total_loss / max(total_samples, 1),
            "training_accuracy": total_correct / max(total_samples, 1),
        }
        for index, param_group in enumerate(self.optimizer.param_groups):
            metrics[f"learning_rate_{index}"] = param_group["lr"]

        return metrics

    def validate_for_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        last_images: Optional[torch.Tensor] = None
        last_labels: Optional[torch.Tensor] = None
        last_predictions: Optional[torch.Tensor] = None

        pbar = tqdm(dataloader, desc="Validation")

        with torch.no_grad():
            for batch in pbar:
                images, labels = self._prepare_batch(batch)

                with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                predictions = logits.argmax(dim=1)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (predictions == labels).sum().item()
                total_samples += batch_size

                last_images = images
                last_labels = labels
                last_predictions = predictions

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{total_correct / total_samples:.4f}",
                    }
                )

        metrics = {
            "validation_loss": total_loss / max(total_samples, 1),
            "validation_accuracy": total_correct / max(total_samples, 1),
        }

        global_step = self.epoch + 1
        if (
            self.wandb_logger.log_image_every > 0
            and global_step % self.wandb_logger.log_image_every == 0
            and last_images is not None
            and last_labels is not None
            and last_predictions is not None
        ):
            self._log_images(
                last_images,
                last_labels,
                last_predictions,
                global_step=global_step,
                wandb_step=self.iteration,
            )

        return metrics

    def run_for_epoch(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        training_config = self.config.train
        checkpoint_config = training_config.output.model

        training_metrics = self.train_for_epoch(training_dataloader)

        for metric_key, metric_value in training_metrics.items():
            self._log_scalar(
                f"{metric_key}/epoch",
                metric_value,
                global_step=self.epoch + 1,
                wandb_step=self.iteration,
            )

        if validation_dataloader is not None:
            validation_metrics = self.validate_for_epoch(validation_dataloader)

            for metric_key, metric_value in validation_metrics.items():
                self._log_scalar(
                    f"{metric_key}/epoch",
                    metric_value,
                    global_step=self.epoch + 1,
                    wandb_step=self.iteration,
                )

            self.epoch += 1

            if self.epoch % checkpoint_config.epoch.every == 0:
                path = checkpoint_config.epoch.path.format(epoch=self.epoch)
                self.save_checkpoint(path)

            last_path = checkpoint_config.last_epoch.path
            if last_path is not None:
                self.save_checkpoint(last_path.format(epoch=self.epoch))

            self.logger.info(
                f"[Epoch {self.epoch}]: "
                f"training_loss={training_metrics['training_loss']:.4f}, "
                f"training_accuracy={training_metrics['training_accuracy']:.4f}, "
                f"validation_loss={validation_metrics['validation_loss']:.4f}, "
                f"validation_accuracy={validation_metrics['validation_accuracy']:.4f}"
            )

            if validation_metrics["validation_accuracy"] > self.best_validation_accuracy:
                self.best_validation_accuracy = validation_metrics["validation_accuracy"]
                path = checkpoint_config.best_epoch.path
                if path is not None:
                    self.save_checkpoint(path.format(epoch=self.epoch))
        else:
            self.epoch += 1

            if self.epoch % checkpoint_config.epoch.every == 0:
                path = checkpoint_config.epoch.path.format(epoch=self.epoch)
                self.save_checkpoint(path)

            last_path = checkpoint_config.last_epoch.path
            if last_path is not None:
                self.save_checkpoint(last_path.format(epoch=self.epoch))

            self.logger.info(
                f"[Epoch {self.epoch}]: training_loss={training_metrics['training_loss']:.4f}, training_accuracy={training_metrics['training_accuracy']:.4f}"
            )
            validation_metrics = {}

        return training_metrics, validation_metrics

    def run(
        self,
        training_dataloader: Optional[DataLoader] = None,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        if training_dataloader is None:
            training_dataloader = self.training_dataloader

        if validation_dataloader is None:
            validation_dataloader = self.validation_dataloader

        training_config = self.config.train
        epochs = training_config.steps.epochs
        iterations = training_config.steps.iterations

        if (epochs is None) == (iterations is None):
            raise ValueError("Set either of config.train.steps.epochs or config.train.steps.iterations.")

        if iterations is not None:
            raise NotImplementedError("iterations is not supported")

        history = copy.deepcopy(self.history)

        for _ in range(self.epoch, epochs):
            training_metrics, validation_metrics = self.run_for_epoch(training_dataloader, validation_dataloader)

            for metric_key, metric_value in training_metrics.items():
                if metric_key in history:
                    history[metric_key].append(metric_value)

            for metric_key, metric_value in validation_metrics.items():
                if metric_key in history:
                    history[metric_key].append(metric_value)

        if self.writer is not None:
            self.writer.close()

        if self.wandb_logger.enabled:
            self.wandb_logger.finish()

        return history

    def save_checkpoint(self, path: str) -> None:
        from ... import __version__ as _version

        model_dir = os.path.dirname(path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "best_validation_accuracy": self.best_validation_accuracy,
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        config = copy.deepcopy(self.config)
        replace_missing_with_none(config)
        checkpoint["resolved_config"] = OmegaConf.to_container(config, resolve=True)
        checkpoint["_metadata"] = {
            "version": _version,
            "driver": self.__class__.__name__,
            "commit_hash": self.commit_hash,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer_state_dict: bool = True,
        load_scheduler_state_dict: bool = True,
    ) -> Dict[str, Any]:
        self.logger.info(f"Load weights from {path}.")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        if load_optimizer_state_dict and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if load_scheduler_state_dict and self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.epoch = checkpoint.get("epoch", 0)
        self.iteration = checkpoint.get("iteration", 0)
        self.best_validation_accuracy = checkpoint.get("best_validation_accuracy", -float("inf"))

        return checkpoint

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "ClassifierTrainer":
        dataloader_config = config.dataloader
        training_config = config.train
        model_config = config.model
        optimizer_config = config.optimizer.optimizer
        scheduler_config = config.optimizer.scheduler

        set_seed(training_config.seed)
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        training_dataloader = hydra.utils.instantiate(dataloader_config.train)
        validation_dataloader = hydra.utils.instantiate(dataloader_config.validate)

        model = hydra.utils.instantiate(model_config)
        criterion = hydra.utils.instantiate(config.loss)
        model = set_device(model, accelerator=accelerator)
        criterion = set_device(criterion, accelerator=accelerator)

        optimizer = hydra.utils.instantiate(optimizer_config, model.parameters())
        scheduler = hydra.utils.instantiate(scheduler_config, optimizer) if scheduler_config.get("_target_") else None
        device = next(model.parameters()).device

        return cls(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
        )


class ClassifierEvaluator(Driver):
    """Evaluator for classification models."""

    def __init__(
        self,
        *,
        dataloader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        config: DictConfig,
        device: torch.device,
    ) -> None:
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device

        training_config = config.train
        dtype = convert_dtype(training_config.torch_dtype)
        self.dtype = dtype
        self.enable_amp = should_enable_amp(dtype)
        self.device_type = get_autocast_device_type()
        self.logger = get_logger()
        self.set_commit_hash()

        checkpoint_path = training_config.checkpoint.resume_from
        if not checkpoint_path:
            raise ValueError("Set train.checkpoint.resume_from to a checkpoint path for evaluation.")

        self.logger.info(f"Load weights from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

    def run(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.dataloader, desc="Evaluation")

        with torch.no_grad():
            for batch in pbar:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                predictions = logits.argmax(dim=1)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (predictions == labels).sum().item()
                total_samples += batch_size

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{total_correct / total_samples:.4f}",
                    }
                )

        metrics = {
            "test_loss": total_loss / max(total_samples, 1),
            "test_accuracy": total_correct / max(total_samples, 1),
        }
        self.logger.info(f"test_loss={metrics['test_loss']:.4f}, test_accuracy={metrics['test_accuracy']:.4f}")
        return metrics

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "ClassifierEvaluator":
        dataloader_config = config.dataloader
        training_config = config.train
        model_config = config.model

        set_seed(training_config.seed)
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        if getattr(dataloader_config, "test", None) is not None and dataloader_config.test.get("_target_"):
            dataloader = hydra.utils.instantiate(dataloader_config.test)
        else:
            dataloader = hydra.utils.instantiate(dataloader_config.validate)

        model = hydra.utils.instantiate(model_config)
        criterion = hydra.utils.instantiate(config.loss)
        model = set_device(model, accelerator=accelerator)
        criterion = set_device(criterion, accelerator=accelerator)
        device = next(model.parameters()).device

        return cls(
            dataloader=dataloader,
            model=model,
            criterion=criterion,
            config=config,
            device=device,
        )
