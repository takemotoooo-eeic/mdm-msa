"""SongFormer losses, one ``nn.Module`` per term.

Every module takes ``(input, target)`` where ``input`` is the model output dict
(``boundary_logits`` ``(B, T)``, ``function_logits`` ``(B, T, C)``) and ``target`` is the batch dict
produced by :class:`mdmsa.utils.collator.SongFormerCollator`. Frames with ``padding_mask`` or the task's
``boundary_mask`` / ``function_mask`` set are ignored.

``reduction`` selects how per-frame terms are averaged:

- ``"mean"`` (default): over *all* ``B * T`` frames after zeroing ignored frames, as upstream does, so the weighted
  sum of these modules reproduces upstream ``Model.compute_losses`` exactly:

      loss_section  = 0.2 * (BoundaryBCELoss + 0.05 * BoundaryTVLoss)
      loss_function = 0.8 * (FunctionCrossEntropyLoss + 0.2 * FunctionFocalLoss)

- ``"masked_mean"``: over the supervised frames only. Used for masked diffusion, where
  :meth:`mdmsa.models.MaskedDiffuser.add_mask` leaves only the hidden frames supervised.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BoundaryBCELoss",
    "BoundaryTVLoss",
    "FunctionCrossEntropyLoss",
    "FunctionFocalLoss",
]

Batch = Dict[str, torch.Tensor]
_REDUCTIONS = ("mean", "masked_mean")


def _boundary_weight(target: Batch) -> torch.Tensor:
    """``(B, T)`` float weight: 1 on frames that supervise boundaries."""
    return ((~target["padding_mask"]) & (~target["boundary_mask"])).float()


def _function_weight(target: Batch) -> torch.Tensor:
    """``(B, T)`` float weight: 1 on frames that supervise functional labels."""
    return ((~target["padding_mask"]) & (~target["function_mask"])).float()


def _reduce(loss: torch.Tensor, weight: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "masked_mean":
        return (weight * loss).sum() / weight.sum().clamp(min=1.0)
    return (weight * loss).mean()


class _FrameLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}")
        self.reduction = reduction


class BoundaryBCELoss(_FrameLoss):
    """Binary cross entropy against Gaussian-widened boundary targets."""

    def forward(self, input: Batch, target: Batch) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(input["boundary_logits"], target["boundary_targets"], reduction="none")
        return _reduce(loss, _boundary_weight(target), self.reduction)


class BoundaryTVLoss(_FrameLoss):
    """Total-variation smoothness penalty on boundary logits, relaxed near true boundaries.

    Args:
        beta: Exponent of the absolute difference.
        lambda_tv: Scale of the penalty.
        boundary_threshold: Target value above which a frame counts as a boundary region.
        reduction_weight: Penalty scale inside boundary regions.
        reduction: ``"mean"`` scales the penalty by the supervised-frame ratio (upstream); ``"masked_mean"`` does not.
    """

    def __init__(
        self, beta: float = 0.6, lambda_tv: float = 0.4, boundary_threshold: float = 0.01, reduction_weight: float = 0.1, reduction: str = "mean"
    ) -> None:
        super().__init__(reduction)
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.boundary_threshold = boundary_threshold
        self.reduction_weight = reduction_weight

    def forward(self, input: Batch, target: Batch) -> torch.Tensor:
        pred = input["boundary_logits"]
        boundary = target["boundary_targets"]

        tv = (pred[:, 1:] - pred[:, :-1]).abs().add(1e-8).pow(self.beta)
        near_boundary = (boundary[:, :-1] > self.boundary_threshold) | (boundary[:, 1:] > self.boundary_threshold)
        weight = torch.where(near_boundary, torch.full_like(tv, self.reduction_weight), torch.ones_like(tv))
        tv_loss = self.lambda_tv * (tv * weight).mean()

        if self.reduction == "masked_mean":
            return tv_loss
        # Upstream adds this scalar to every frame before the masked mean.
        return tv_loss * _boundary_weight(target).mean()


class FunctionCrossEntropyLoss(_FrameLoss):
    """Cross entropy with (possibly all-zero) one-hot targets ``(B, T, C)``."""

    def forward(self, input: Batch, target: Batch) -> torch.Tensor:
        loss = F.cross_entropy(input["function_logits"].transpose(1, 2), target["function_targets"].transpose(1, 2), reduction="none")
        return _reduce(loss, _function_weight(target), self.reduction)


class FunctionFocalLoss(_FrameLoss):
    """Softmax focal loss for mutually exclusive functional labels.

    Args:
        alpha: Weight of the positive class (0 disables alpha weighting).
        gamma: Focusing parameter.
        reduction: See the module docstring.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: Batch, target: Batch) -> torch.Tensor:
        log_probs = F.log_softmax(input["function_logits"], dim=-1)
        onehot = target["function_targets"]

        p_t = (log_probs.exp() * onehot).sum(dim=-1).clamp(min=1e-8, max=1.0 - 1e-8)
        if self.alpha > 0:
            alpha_t = self.alpha * onehot + (1 - self.alpha) * (1 - onehot)
            alpha_weight = (alpha_t * onehot).sum(dim=-1)
        else:
            alpha_weight = 1.0

        ce = -(log_probs * onehot).sum(dim=-1)
        loss = alpha_weight * (1 - p_t) ** self.gamma * ce
        return _reduce(loss, _function_weight(target), self.reduction)
