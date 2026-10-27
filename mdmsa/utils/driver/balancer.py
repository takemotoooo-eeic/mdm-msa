from typing import Dict, List, Optional, Sequence

import torch
from torch import autograd

__all__ = ["Balancer"]


class Balancer:
    """Gradient-norm loss balancer (EnCodec style, as used by upstream SongFormer training).

    Each loss is rescaled by ``ratio * total_norm / ema(||d loss / d params||)`` so that every loss contributes
    gradients of the same magnitude regardless of its scale.

    Args:
        weights: Relative weights of the losses (default: equal).
        rescale_grads: If false, return the plain weighted sum.
        total_norm: Reference gradient norm.
        ema_decay: Decay of the moving average of gradient norms.
        epsilon: Numerical stability term.
    """

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        rescale_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        epsilon: float = 1e-12,
    ) -> None:
        self.weights = list(weights) if weights is not None else None
        self.rescale_grads = rescale_grads
        self.total_norm = total_norm
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        self._norm_sum: Dict[int, float] = {}
        self._weight_sum: Dict[int, float] = {}
        self.metrics: Dict[str, float] = {}

    def _average(self, index: int, value: float) -> float:
        self._norm_sum[index] = self._norm_sum.get(index, 0.0) * self.ema_decay + value
        self._weight_sum[index] = self._weight_sum.get(index, 0.0) * self.ema_decay + 1.0
        return self._norm_sum[index] / self._weight_sum[index]

    def __call__(self, losses: List[torch.Tensor], params: Sequence[torch.Tensor]) -> torch.Tensor:
        weights = self.weights if self.weights is not None else [1.0] * len(losses)
        if len(weights) != len(losses):
            raise ValueError(f"Got {len(losses)} losses but {len(weights)} weights.")

        if not self.rescale_grads:
            return sum(w * loss for w, loss in zip(weights, losses))

        params = [p for p in params if p.requires_grad]
        norms = []
        for index, loss in enumerate(losses):
            grads = autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            norm = torch.cat([g.reshape(-1) for g in grads if g is not None]).norm()
            norms.append(self._average(index, float(norm)))

        total = sum(norms)
        self.metrics = {f"ratio_{index}": norm / total for index, norm in enumerate(norms)}

        weight_sum = sum(weights)
        return sum(loss * (w / weight_sum) * self.total_norm / (self.epsilon + norm) for loss, w, norm in zip(losses, weights, norms))
