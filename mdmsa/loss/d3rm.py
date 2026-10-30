"""D3RM loss terms. The diffusion process computes them per song (:meth:`mdmsa.models.D3RMDiffusion.training_losses`);
these modules only average them so that ``Loss`` / ``TotalLoss`` can weight them from YAML.

Reference: https://github.com/hanshounsu/d3rm, ``transcription/diffusion/trainer.py`` (``DiscreteDiffusion._train_loss``):
``loss = vb_loss + auxiliary_loss_weight * aux_loss``, with ``auxiliary_loss_weight = 5e-4`` in ``configs/D3RM_cli.yaml``.
"""

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["D3RMVBLoss", "D3RMAuxLoss"]

Batch = Dict[str, torch.Tensor]


class D3RMVBLoss(nn.Module):
    """Variational bound term ``KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)) / p(t)`` (decoder NLL at ``t = 0``)."""

    def forward(self, input: Batch, target: Batch) -> torch.Tensor:
        return input["vb_loss"].mean()


class D3RMAuxLoss(nn.Module):
    """Auxiliary ``KL(x_0 || p(x_0|x_t)) / p(t)`` on all labeled frames, including unmasked (possibly replaced) ones."""

    def forward(self, input: Batch, target: Batch) -> torch.Tensor:
        return input["aux_loss"].mean()
