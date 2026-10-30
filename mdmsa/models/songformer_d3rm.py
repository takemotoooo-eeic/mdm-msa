"""SongFormer as the D3RM denoiser: label-token embeddings plus diffusion-step AdaLN.

Only two changes are made to :class:`SongFormer`:

1. An embedding of the current label tokens ``x_t`` (``K`` classes + MASK) is added to the encoded audio before the
   transformer. (D3RM concatenates label embeddings with encoder features: https://github.com/hanshounsu/d3rm,
   ``transcription/decoder.py``, ``TransModel.forward``.)
2. Every pre-norm of the transformer and its final norm become :class:`TimestepAdaLayerNorm`, which adds a
   diffusion-step dependent scale and shift (reference ``transcription/diffusion/natten_diffusion.py``,
   ``AdaLayerNorm`` with ``timestep_type: adalayernorm``, i.e. a per-layer ``nn.Embedding`` of the step).

All other module and parameter names are SongFormer's, so released SongFormer weights load into this model;
the new modules are zero-initialised so that the loaded model starts as SongFormer applied to the audio.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.data.labels import LABEL_TO_ID
from .songformer import SongFormer

__all__ = ["SongFormerD3RM", "TimestepAdaLayerNorm"]


class TimestepAdaLayerNorm(nn.Module):
    """``LayerNorm(x) * gamma * (1 + scale(t)) + shift(t)``.

    Keeps the ``gamma`` / ``unit_offset`` of the x-transformers ``LayerNorm`` it replaces. ``scale, shift`` are
    ``Linear(SiLU(E[t]))`` with a per-layer step embedding ``E`` as in the reference ``AdaLayerNorm``. The embedding
    is applied to a one-hot step vector (``condition`` ``(B, 1, T)``) so it can be passed through x-transformers'
    ``condition`` argument. The output linear is zero-initialised (the reference uses the default init).
    """

    def __init__(self, dim: int, diffusion_step: int, unit_offset: bool = False) -> None:
        super().__init__()
        self.unit_offset = unit_offset
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.full((dim,), 1.0 - float(unit_offset)))
        self.emb = nn.Linear(diffusion_step, dim, bias=False)
        nn.init.normal_(self.emb.weight)  # same init as nn.Embedding
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @classmethod
    def replace(cls, norm: nn.Module, diffusion_step: int) -> "TimestepAdaLayerNorm":
        """Build from an x-transformers ``LayerNorm``, reusing its ``gamma``."""
        module = cls(norm.gamma.shape[0], diffusion_step, unit_offset=bool(norm.unit_offset))
        module.gamma = norm.gamma
        return module

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = torch.chunk(self.linear(self.silu(self.emb(condition))), 2, dim=-1)
        gamma = self.gamma + float(self.unit_offset)
        return self.ln(x) * gamma * (1 + scale) + shift


class SongFormerD3RM(SongFormer):
    """:class:`SongFormer` that predicts ``p(x_0 | x_t, t, audio)`` over ``num_classes`` labels.

    Args:
        diffusion_step: Number of diffusion steps ``T`` (size of the step embeddings).
        zero_init_label_embed: Start the label-token embedding at zero (useful when fine-tuning pretrained weights).
        pretrained_path: Optional released SongFormer weights (``.safetensors`` / ``.pt``) to initialise from.
        pretrained_labels: Label names of this model's classes; selects the rows of the pretrained 128-way
            ``function_head`` output layer. Required with ``pretrained_path``.
        **kwargs: :class:`SongFormer` arguments (``num_classes`` is the number of real labels ``K``).
    """

    def __init__(
        self,
        diffusion_step: int = 100,
        zero_init_label_embed: bool = False,
        pretrained_path: Optional[str] = None,
        pretrained_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.diffusion_step = diffusion_step
        self.label_token_embed = nn.Embedding(self.num_classes + 1, self.down_sample_conv.pointwise_conv.out_channels)
        if zero_init_label_embed:
            nn.init.zeros_(self.label_token_embed.weight)

        encoder = self.transformer.transformer
        for layer in encoder.layers:
            norms = layer[0]
            norms[0] = TimestepAdaLayerNorm.replace(norms[0], diffusion_step)
        encoder.final_norm = TimestepAdaLayerNorm.replace(encoder.final_norm, diffusion_step)
        # Let x-transformers pass `condition` to the norms.
        encoder.norm_need_condition = True
        encoder.need_condition = True
        encoder.dim_condition = diffusion_step

        if pretrained_path:
            self.load_pretrained(pretrained_path, pretrained_labels)

    def load_pretrained(self, path: str, labels: Optional[List[str]]) -> None:
        """Load SongFormer weights; only the new (AdaLN step / label-token) parameters stay at their init."""
        if labels is None or len(labels) != self.num_classes:
            raise ValueError("pretrained_labels must list num_classes label names.")
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(path)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state = checkpoint.get("ema", checkpoint.get("model_ema", checkpoint.get("model")))
        state = {k[len("ema_model.") :] if k.startswith("ema_model.") else k: v for k, v in state.items() if k not in ("step", "initted")}

        rows = torch.tensor([LABEL_TO_ID[label] for label in labels])
        last = max(int(k.split(".")[2]) for k in state if k.startswith("function_head.net."))
        for name in ("weight", "bias"):
            key = f"function_head.net.{last}.{name}"
            state[key] = state[key][rows]

        missing, unexpected = self.load_state_dict(state, strict=False)
        new = ("label_token_embed.", ".emb.", ".linear.")
        bad_missing = [k for k in missing if not any(n in k for n in new)]
        if bad_missing or unexpected:
            raise RuntimeError(f"Pretrained weights do not match: missing={bad_missing}, unexpected={unexpected}")

    def encode_audio(self, embeddings: torch.Tensor, dataset_ids: torch.Tensor) -> torch.Tensor:
        """Downsampled audio features with the dataset prefix, ``(B, T, D)``; reused across diffusion steps."""
        x = self.input_norm(self.mixed_win_downsample(embeddings))
        x = self.down_sample_conv(x)
        return x + self.dataset_class_prefix(dataset_ids).unsqueeze(1)

    def denoise(self, audio: torch.Tensor, tokens: torch.Tensor, timesteps: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``(B, T)`` tokens and ``(B,)`` steps -> function logits ``(B, T, num_classes)``."""
        x = audio + self.label_token_embed(tokens)
        condition = F.one_hot(timesteps, self.diffusion_step).to(x.dtype)
        mask = None if padding_mask is None else ~padding_mask
        x = self.transformer.transformer(self.transformer.input_proj(x), mask=mask, condition=condition)
        return self.function_head(x)

    def forward(
        self,
        embeddings: torch.Tensor,
        dataset_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        *,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """``function_logits`` ``(B, T, num_classes)`` of ``p(x_0 | x_t = tokens, t = timesteps)``."""
        logits = self.denoise(self.encode_audio(embeddings, dataset_ids), tokens, timesteps, padding_mask=padding_mask)
        if label_mask is not None:
            logits = logits.masked_fill(label_mask.unsqueeze(1), -float("inf"))
        return {"function_logits": logits}
