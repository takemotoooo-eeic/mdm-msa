import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CrossEntropyLoss",
    "Loss",
    "TotalLoss",
]


class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input, target)


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
