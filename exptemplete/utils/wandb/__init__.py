import importlib
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_wandb = importlib.import_module("wandb")

__all__ = [
    "DummyWandbLogger",
    "WandbLogger",
    "get_wandb_logger",
]


class WandbLogger:
    def __init__(self, run: Any, log_image_every: int = 1) -> None:
        self._run = run
        self.log_image_every = log_image_every

    @property
    def enabled(self) -> bool:
        return True

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        _wandb.log({tag: scalar_value}, step=global_step)

    def add_scalars(self, metrics: Dict[str, float], global_step: int) -> None:
        _wandb.log(metrics, step=global_step)

    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.ndarray],
        global_step: int,
    ) -> None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()

        _wandb.log({tag: _wandb.Image(image)}, step=global_step)

    def watch(self, model: torch.nn.Module) -> None:
        _wandb.watch(model)

    def finish(self) -> None:
        _wandb.finish()


class DummyWandbLogger:
    def __init__(self, log_image_every: int = 1) -> None:
        self.log_image_every = log_image_every

    @property
    def enabled(self) -> bool:
        return False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        pass

    def add_scalars(self, metrics: Dict[str, float], global_step: int) -> None:
        pass

    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.ndarray],
        global_step: int,
    ) -> None:
        pass

    def watch(self, model: torch.nn.Module) -> None:
        pass

    def finish(self) -> None:
        pass


def get_wandb_logger(
    *,
    enabled: bool = True,
    project: str = "exp-templete",
    name: Optional[str] = None,
    entity: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
    watch: bool = False,
    model: Optional[torch.nn.Module] = None,
    log_image_every: int = 1,
) -> Union[WandbLogger, DummyWandbLogger]:
    if not enabled:
        return DummyWandbLogger(log_image_every=log_image_every)

    if isinstance(run_config, DictConfig):
        run_config = OmegaConf.to_container(run_config, resolve=True)
    name = str(name)

    run = _wandb.init(
        project=project,
        name=name,
        entity=entity,
        config=run_config,
    )
    logger = WandbLogger(run, log_image_every=log_image_every)

    if watch and model is not None:
        logger.watch(model)

    return logger
