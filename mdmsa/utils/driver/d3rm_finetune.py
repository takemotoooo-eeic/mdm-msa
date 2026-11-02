"""Two-stage D3RM fine-tuning from released SongFormer weights.

Stage 1 (the first ``train.freeze.iterations`` optimizer steps) trains only the parameters added to SongFormer
(:meth:`mdmsa.models.SongFormerD3RM.new_parameter_names`: label-token embedding and AdaLN step conditioning); the
pretrained body is frozen. Stage 2 trains everything. The new parameters use ``train.freeze.new_lr`` throughout and
the body uses the optimizer's ``lr``; both groups follow the same scheduler.

Not in the reference (https://github.com/hanshounsu/d3rm), which trains from scratch; the diffusion objective is the
same as :class:`mdmsa.utils.driver.D3RMTrainer`.
"""

from typing import Set

import hydra
import torch
from omegaconf import DictConfig

from .. import set_device, set_seed
from .d3rm import D3RMTrainer

__all__ = ["D3RMTwoStageTrainer"]


class D3RMTwoStageTrainer(D3RMTrainer):
    """:class:`D3RMTrainer` with a frozen pretrained body for the first ``train.freeze.iterations`` steps."""

    new_parameter_names: Set[str]
    freeze_iterations: int

    def _set_body_trainable(self, trainable: bool) -> None:
        for name, parameter in self.model.named_parameters():
            if name not in self.new_parameter_names:
                parameter.requires_grad_(trainable)

    def _optimizer_step(self) -> None:
        super()._optimizer_step()
        if self.iteration == self.freeze_iterations:
            self._set_body_trainable(True)
            self.logger.info(f"Iteration {self.iteration}: unfroze the pretrained body.")

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "D3RMTwoStageTrainer":
        # Same as SongFormerTrainer.build_from_config, except for the two optimizer parameter groups (built before
        # the trainer so that resuming loads a matching optimizer state).
        training_config = config.train
        set_seed(training_config.seed)
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        training_dataloader = hydra.utils.instantiate(config.dataloader.train)
        validation_dataloader = hydra.utils.instantiate(config.dataloader.validate) if config.dataloader.validate.get("_target_") else None

        model = set_device(hydra.utils.instantiate(config.model), accelerator=accelerator)
        criterion = set_device(hydra.utils.instantiate(config.loss), accelerator=accelerator)

        new_names = set(model.new_parameter_names())
        body = [p for n, p in model.named_parameters() if n not in new_names]
        new = [p for n, p in model.named_parameters() if n in new_names]
        groups = [{"params": body}, {"params": new, "lr": float(training_config.freeze.new_lr)}]
        optimizer = hydra.utils.instantiate(config.optimizer.optimizer, groups)
        scheduler_config = config.optimizer.scheduler
        scheduler = hydra.utils.instantiate(scheduler_config, optimizer) if scheduler_config.get("_target_") else None

        ema = hydra.utils.instantiate(training_config.ema, model) if training_config.get("ema") else None
        balancer = hydra.utils.instantiate(training_config.balancer) if training_config.get("balancer") else None

        trainer = cls(
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
        )
        trainer.diffuser = hydra.utils.instantiate(config.train.diffuser).to(trainer.device)
        trainer.new_parameter_names = new_names
        trainer.freeze_iterations = int(training_config.freeze.iterations)
        if trainer.iteration < trainer.freeze_iterations:
            trainer._set_body_trainable(False)
            trainer.logger.info(f"Pretrained body frozen until iteration {trainer.freeze_iterations}; {sum(p.numel() for p in new)} parameters train.")
        return trainer
