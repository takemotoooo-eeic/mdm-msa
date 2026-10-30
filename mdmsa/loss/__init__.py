import torch
import torch.nn as nn

from .d3rm import D3RMAuxLoss, D3RMVBLoss
from .songformer import BoundaryBCELoss, BoundaryTVLoss, FunctionCrossEntropyLoss, FunctionFocalLoss

__all__ = [
    "D3RMVBLoss",
    "D3RMAuxLoss",
    "BoundaryBCELoss",
    "BoundaryTVLoss",
    "FunctionCrossEntropyLoss",
    "FunctionFocalLoss",
    "Loss",
    "TotalLoss",
]


class Loss(nn.Module):
    def __init__(self, loss_module: nn.Module, weight: float = 1.0) -> None:
        super().__init__()
        self.loss_module = loss_module
        self.weight = weight

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.loss_module(*args, **kwargs) * self.weight


class TotalLoss(nn.Module):
    def __init__(self, losses: list[Loss]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return sum(loss(*args, **kwargs) for loss in self.losses)
