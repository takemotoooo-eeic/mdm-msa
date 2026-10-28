"""Masked diffusion SongFormer: SongFormer conditioned on partially masked boundary / function tokens."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .diffuser import MaskedDiffuser
from .songformer import SongFormer

__all__ = ["SongFormerDiffusion"]


class SongFormerDiffusion(SongFormer):
    """:class:`SongFormer` whose transformer input also gets embeddings of the current tokens.

    Token embeddings are added to the encoded audio (after the dataset prefix), so every other module and parameter
    name is shared with :class:`SongFormer`. Token ids follow :class:`mdmsa.models.MaskedDiffuser`.

    Args:
        scale_token_embed: Multiply token embeddings by ``sqrt(transformer_encoder_input_dim)``.
        **kwargs: :class:`SongFormer` arguments.
    """

    def __init__(self, scale_token_embed: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        dim = self.transformer_encoder_input_dim
        self.boundary_token_embed = nn.Embedding(MaskedDiffuser.BOUNDARY_VOCAB_SIZE, dim)
        self.function_token_embed = nn.Embedding(MaskedDiffuser.function_vocab_size(self.num_classes), dim)
        self.token_embed_scale = dim**0.5 if scale_token_embed else 1.0

    def forward(
        self,
        embeddings: torch.Tensor,
        dataset_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        *,
        boundary_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Same as :meth:`SongFormer.forward`, plus ``boundary_tokens`` / ``function_tokens`` ``(B, T)``."""
        x = self.encode(embeddings, dataset_ids)
        x = x + self.token_embed_scale * (self.boundary_token_embed(boundary_tokens) + self.function_token_embed(function_tokens))
        return self.predict(x, padding_mask=padding_mask, label_mask=label_mask)
