import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["CosineWarmupLR"]


class CosineWarmupLR(LambdaLR):
    """Linear warm-up then cosine decay to zero, stepped once per optimizer step.

    Same schedule as ``transformers.get_cosine_schedule_with_warmup``.
    """

    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1) -> None:
        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
