import copy

import torch
import torch.nn as nn

__all__ = ["ExponentialMovingAverage"]


class ExponentialMovingAverage(nn.Module):
    """EMA of model weights with a warm-up decay schedule (same rule as ``ema_pytorch.EMA`` 0.7).

    ``state_dict()`` holds ``ema_model.*``, ``initted`` and ``step``, the layout of upstream SongFormer checkpoints.

    Args:
        model: Model to track. A frozen copy is kept in ``ema_model``.
        beta: Maximum decay.
        update_after_step: Until this step the EMA copies the model.
        update_every: Update interval (steps).
        inv_gamma: Decay schedule parameter.
        power: Decay schedule exponent.
        min_value: Minimum decay.
    """

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.9999,
        update_after_step: int = 100,
        update_every: int = 10,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def current_decay(self) -> float:
        epoch = max(int(self.step.item()) - self.update_after_step - 1, 0)
        if epoch <= 0:
            return 0.0
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return min(max(value, self.min_value), self.beta)

    @torch.no_grad()
    def _copy_from(self, model: nn.Module) -> None:
        for ema_tensor, tensor in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
            ema_tensor.copy_(tensor)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        step = int(self.step.item())
        self.step += 1

        if not bool(self.initted.item()):
            self._copy_from(model)
            self.initted.fill_(True)
            return

        if step % self.update_every != 0:
            return
        if step <= self.update_after_step:
            self._copy_from(model)
            return

        weight = 1.0 - self.current_decay()
        for ema_tensor, tensor in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
            if ema_tensor.is_floating_point():
                ema_tensor.lerp_(tensor, weight)
            else:
                ema_tensor.copy_(tensor)

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.ema_model(*args, **kwargs)
