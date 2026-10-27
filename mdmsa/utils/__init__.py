import random
from typing import Optional

import torch
import torch.nn as nn

from ._torch import convert_dtype

__all__ = [
    "convert_dtype",
    "set_seed",
    "set_device",
    "select_device",
]


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def set_device(
    module: nn.Module,
    accelerator: str,
) -> nn.Module:
    device = select_device(accelerator)
    module = module.to(device)
    return module


def select_device(accelerator: Optional[str]) -> str:
    if accelerator is None:
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    if accelerator in ["cpu", "cuda", "mps"]:
        device = accelerator
    elif accelerator == "gpu":
        device = "cuda"
    else:
        raise ValueError(f"Unknown accelerator {accelerator} is specified.")

    return device
