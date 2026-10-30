"""D3RM discrete diffusion (mask + uniform transition matrices) over frame-wise functional labels.

Reference: https://github.com/hanshounsu/d3rm (commit 8f7f2f6), ``transcription/diffusion/trainer.py``
(``alpha_schedule``, ``DiscreteDiffusion``; itself adapted from VQ-Diffusion). The forward process, posterior,
training loss and sampler follow that file; the encoder / decoder it wraps are replaced by a callable
``denoise(tokens, timesteps) -> logits`` (see :class:`mdmsa.models.SongFormerD3RM`).

Tokens are class ids ``0 .. K-1`` plus ``MASK = K``. Log one-hot tensors have the layout ``(B, K + 1, L)`` as in the
reference. Deviations from the reference, all needed because our sequences are padded to different lengths:

- Frames outside ``valid`` (padding / unlabeled) are given the MASK token as model input and get zero loss weight.
- Per-song KL terms are averaged over valid frames instead of summed over a fixed ``label_seq_len`` and then divided
  by ``B * L``; for equal lengths both give the same loss.
- ``gamma_bar_T >= 0.99999`` is clipped to ``0.99999`` (the reference's value for the absorbing reverse schedule),
  since ``gamma_bar_T = 1`` makes ``beta_bar`` negative in ``alpha_schedule``.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["D3RMDiffusion", "alpha_schedule"]

Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_MAX_GAMMA_BAR = 0.99999


def _log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    return torch.log(1 - a.exp() + 1e-40)


def _log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _index_to_log_onehot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``(B, L)`` ids -> ``(B, num_classes, L)`` log one-hot."""
    onehot = F.one_hot(x, num_classes).permute(0, 2, 1)
    return torch.log(onehot.float().clamp(min=1e-30))


def _log_onehot_to_index(log_x: torch.Tensor) -> torch.Tensor:
    return log_x.argmax(1)


def alpha_schedule(
    time_step: int, N: int, att_1: float = 0.99999, att_T: float = 0.000009, ctt_1: float = 0.000009, ctt_T: float = 0.99999
) -> Tuple[np.ndarray, ...]:
    """Linear schedules of the keep (alpha), uniform-replace (beta) and mask (gamma) probabilities.

    Returns per-step ``at, bt, ct`` (length ``T``) and cumulative ``att, btt, ctt`` (length ``T + 1``; the last entry
    is the no-noise state used for ``t = -1``). Copied from the reference ``alpha_schedule``.
    """
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


class _Schedule(nn.Module):
    """Log-space buffers of one :func:`alpha_schedule` (forward or reverse)."""

    def __init__(self, num_timesteps: int, num_states: int, gamma_bar_T: float) -> None:
        super().__init__()
        at, bt, ct, att, btt, ctt = alpha_schedule(num_timesteps, N=num_states, ctt_T=gamma_bar_T)
        tensors = {name: torch.tensor(value.astype("float64")) for name, value in dict(at=at, bt=bt, ct=ct, att=att, btt=btt, ctt=ctt).items()}
        log_ct, log_cumprod_ct = torch.log(tensors["ct"]), torch.log(tensors["ctt"])
        buffers = {
            "log_at": torch.log(tensors["at"]),
            "log_bt": torch.log(tensors["bt"]),
            "log_ct": log_ct,
            "log_cumprod_at": torch.log(tensors["att"]),
            "log_cumprod_bt": torch.log(tensors["btt"]),
            "log_cumprod_ct": log_cumprod_ct,
            "log_1_min_ct": _log_1_min_a(log_ct),
            "log_1_min_cumprod_ct": _log_1_min_a(log_cumprod_ct),
        }
        assert _log_add_exp(buffers["log_ct"], buffers["log_1_min_ct"]).abs().sum().item() < 1.0e-5
        assert _log_add_exp(buffers["log_cumprod_ct"], buffers["log_1_min_cumprod_ct"]).abs().sum().item() < 1.0e-5
        for name, value in buffers.items():
            self.register_buffer(name, value.float(), persistent=False)


class D3RMDiffusion(nn.Module):
    """Discrete denoising diffusion with mask-and-uniform corruption (D3RM / VQ-Diffusion).

    Args:
        num_classes: Number of real classes ``K`` (the MASK token is ``K``).
        diffusion_step: Number of diffusion steps ``T``.
        gamma_bar_T: Final cumulative mask probability; ``1 - gamma_bar_T`` of the tokens are uniformly replaced at ``T``.
        adaptive_auxiliary_loss: Scale the auxiliary term by ``2 - t / T`` as the reference does.
        mask_weight: KL weights of ``[masked, unmasked]`` positions of ``x_t``.
        reverse_sampling: Sample with the absorbing (``gamma_bar_T ~ 1``) posterior ("absorbing state sampling").
        time_sampling: ``"importance"`` (reference default, uniform until every step was seen 10 times) or ``"uniform"``.
    """

    def __init__(
        self,
        num_classes: int,
        diffusion_step: int = 100,
        gamma_bar_T: float = 0.6,
        adaptive_auxiliary_loss: bool = True,
        mask_weight: Optional[List[float]] = None,
        reverse_sampling: bool = True,
        time_sampling: str = "importance",
    ) -> None:
        super().__init__()
        if time_sampling not in ("importance", "uniform"):
            raise ValueError(f"Unknown time_sampling: {time_sampling}")
        self.num_real_classes = num_classes
        self.num_classes = num_classes + 1  # + MASK, as in the reference
        self.mask_token = num_classes
        self.num_timesteps = diffusion_step
        self.gamma_bar_T = min(float(gamma_bar_T), _MAX_GAMMA_BAR)
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = list(mask_weight) if mask_weight is not None else [1.0, 1.0]
        self.reverse_sampling = reverse_sampling
        self.time_sampling = time_sampling

        self.forward_schedule = _Schedule(diffusion_step, num_classes, self.gamma_bar_T)
        self.reverse_schedule = _Schedule(diffusion_step, num_classes, _MAX_GAMMA_BAR) if reverse_sampling else None
        self.register_buffer("Lt_history", torch.zeros(diffusion_step))
        self.register_buffer("Lt_count", torch.zeros(diffusion_step))

    # ----- forward process (reference: q_pred_one_timestep, q_pred, q_posterior) -----

    @staticmethod
    def _q_pred_one_timestep(s: _Schedule, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        log_at = _extract(s.log_at, t, log_x_t.shape)
        log_bt = _extract(s.log_bt, t, log_x_t.shape)
        log_ct = _extract(s.log_ct, t, log_x_t.shape)
        log_1_min_ct = _extract(s.log_1_min_ct, t, log_x_t.shape)
        return torch.cat([_log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt), _log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)], dim=1)

    def _q_pred(self, s: _Schedule, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``q(x_t | x_0)``."""
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = _extract(s.log_cumprod_at, t, log_x_start.shape)
        log_cumprod_bt = _extract(s.log_cumprod_bt, t, log_x_start.shape)
        log_cumprod_ct = _extract(s.log_cumprod_ct, t, log_x_start.shape)
        log_1_min_cumprod_ct = _extract(s.log_1_min_cumprod_ct, t, log_x_start.shape)
        return torch.cat(
            [
                _log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                _log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct),
            ],
            dim=1,
        )

    def _q_posterior(self, s: _Schedule, log_x_start: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``q(x_{t-1} | x_t, x_0)`` averaged over ``x_0 ~ exp(log_x_start)`` (reference ``q_posterior[_reverse]``)."""
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size, _, length = log_x_start.shape
        mask = (_log_onehot_to_index(log_x_t) == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, length)

        log_qt = self._q_pred(s, log_x_t, t)[:, :-1, :]
        log_cumprod_ct = _extract(s.log_cumprod_ct, t, log_x_start.shape)
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self._q_pred_one_timestep(s, log_x_t, t)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = _extract(s.log_ct, t, log_x_start.shape)
        ct_vector = torch.cat((log_ct.expand(-1, self.num_classes - 1, -1), log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_ev = self._q_pred(s, q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_ev, -70, 0)

    @staticmethod
    def _log_sample_categorical(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Gumbel-max sample over dim 1, returned as log one-hot."""
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        return _index_to_log_onehot((gumbel_noise + logits).argmax(dim=1), num_classes)

    def q_sample(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._log_sample_categorical(self._q_pred(self.forward_schedule, log_x_start, t), self.num_classes)

    # ----- model (reference: predict_start) -----

    def predict_start(self, denoise: Denoiser, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``log p(x_0 | x_t)`` as ``(B, K + 1, L)`` with the MASK row at ``-70``."""
        x_t = _log_onehot_to_index(log_x_t)
        out = denoise(x_t, t).transpose(1, 2)  # (B, K, L)
        assert out.size(1) == self.num_classes - 1
        log_pred = F.log_softmax(out.double(), dim=1).float()
        zero_vector = torch.full_like(log_pred[:, :1, :], -70.0)
        return torch.clamp(torch.cat((log_pred, zero_vector), dim=1), -70, 0)

    # ----- training (reference: sample_time, multinomial_kl, _train_loss) -----

    def _sample_time(self, b: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.time_sampling == "importance" and bool((self.Lt_count > 10).all()):
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            return t, pt_all.gather(dim=0, index=t)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return t, torch.ones_like(t).float() / self.num_timesteps

    @staticmethod
    def _multinomial_kl(log_prob1: torch.Tensor, log_prob2: torch.Tensor) -> torch.Tensor:
        return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)

    def training_losses(self, denoise: Denoiser, x_start: torch.Tensor, valid: torch.Tensor, update_history: bool = True) -> Dict[str, torch.Tensor]:
        """One stochastic VLB term per song.

        Args:
            denoise: ``(tokens (B, L), timesteps (B,)) -> logits (B, L, K)``.
            x_start: ``(B, L)`` clean class ids (any id at invalid frames).
            valid: ``(B, L)`` bool, frames that carry a label.
            update_history: Update the importance-sampling statistics (off for validation).

        Returns:
            ``vb_loss`` and ``aux_loss`` ``(B,)`` (already divided by ``p(t)``; the auxiliary weight is applied by the
            loss config), ``t`` ``(B,)``, ``log_x0_recon`` ``(B, K + 1, L)`` and ``x_t`` ``(B, L)``.
        """
        b, device = x_start.size(0), x_start.device
        t, pt = self._sample_time(b, device)

        x_start = torch.where(valid, x_start, torch.zeros_like(x_start))
        log_x_start = _index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start, t)
        # Invalid frames are shown as MASK and carry no loss.
        log_xt = torch.where(valid.unsqueeze(1), log_xt, _index_to_log_onehot(torch.full_like(x_start, self.mask_token), self.num_classes))
        xt = _log_onehot_to_index(log_xt)

        log_x0_recon = self.predict_start(denoise, log_xt, t)
        log_model_prob = self._q_posterior(self.forward_schedule, log_x0_recon, log_xt, t)
        log_true_prob = self._q_posterior(self.forward_schedule, log_x_start, log_xt, t)

        weight = valid.float()
        count = weight.sum(-1).clamp(min=1)
        mask_region = (xt == self.mask_token).float()
        mask_weight = mask_region * self.mask_weight[0] + (1.0 - mask_region) * self.mask_weight[1]

        kl = (self._multinomial_kl(log_true_prob, log_model_prob) * mask_weight * weight).sum(-1) / count
        decoder_nll = (-(log_x_start.exp() * log_model_prob).sum(dim=1) * weight).sum(-1) / count
        is_first = (t == 0).float()
        kl_loss = is_first * decoder_nll + (1.0 - is_first) * kl

        if update_history:
            Lt2 = kl_loss.detach().pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            self.Lt_history.scatter_(dim=0, index=t, src=(0.1 * Lt2 + 0.9 * Lt2_prev))
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        kl_aux = (self._multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :]) * mask_weight * weight).sum(-1) / count
        kl_aux_loss = is_first * decoder_nll + (1.0 - is_first) * kl_aux
        aux_weight = (1 - t / self.num_timesteps) + 1.0 if self.adaptive_auxiliary_loss else torch.ones_like(kl_aux_loss)

        return {
            "vb_loss": kl_loss / pt,
            "aux_loss": aux_weight * kl_aux_loss / pt,
            "t": t,
            "log_x0_recon": log_x0_recon,
            "x_t": xt,
        }

    # ----- sampling (reference: p_pred, p_sample, sample) -----

    @torch.no_grad()
    def sample(self, denoise: Denoiser, batch_size: int, length: int, device: torch.device, return_trajectory: bool = False) -> Dict[str, torch.Tensor]:
        """Ancestral sampling from the fully masked sequence over all ``T`` steps.

        Returns:
            ``tokens`` ``(B, L)`` and ``x0_first`` ``(B, L)``: argmax of ``p(x_0 | x_T)`` at the first step
            (the one-step prediction). With ``return_trajectory``, ``trajectory`` ``(T, B, L)`` holds ``x_{t-1}`` after
            every step.
        """
        schedule = self.reverse_schedule if self.reverse_sampling else self.forward_schedule
        # Fully masked start (reference: sample_from_fully_masked=True); clamped log one-hot instead of log(0).
        log_z = _index_to_log_onehot(torch.full((batch_size, length), self.mask_token, device=device, dtype=torch.long), self.num_classes)
        x0_first = None
        trajectory = []
        for diffusion_index in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
            log_x_recon = self.predict_start(denoise, log_z, t)
            if x0_first is None:
                x0_first = log_x_recon[:, :-1, :].argmax(1)
            model_log_prob = self._q_posterior(schedule, log_x_recon, log_z, t)
            log_z = self._log_sample_categorical(model_log_prob, self.num_classes)
            if return_trajectory:
                trajectory.append(_log_onehot_to_index(log_z))
        out = {"tokens": _log_onehot_to_index(log_z), "x0_first": x0_first}
        if return_trajectory:
            out["trajectory"] = torch.stack(trajectory)
        return out
