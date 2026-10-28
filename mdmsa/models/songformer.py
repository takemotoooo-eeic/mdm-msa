"""SongFormer: music structure analysis from pre-extracted SSL features.

Ported from https://github.com/ASLP-lab/SongFormer (``models/SongFormer.py``).
Submodule names are kept identical to upstream so released checkpoints load as-is.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from x_transformers import Encoder

__all__ = [
    "SongFormer",
    "Head",
    "TimeDownsample",
    "TransformerEncoder",
]


class Head(nn.Module):
    """Frame-wise MLP head: ``(B, T, C_in) -> (B, T, C_out)``."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None, activation: str = "silu") -> None:
        super().__init__()
        activations = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        dims = [input_dim, *(hidden_dims or []), output_dim]
        layers: List[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(activations[activation.lower()]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeDownsample(nn.Module):
    """Depthwise-separable strided conv with an average-pooled residual branch."""

    def __init__(
        self,
        dim_in: int,
        dim_out: Optional[int] = None,
        kernel_size: int = 5,
        stride: int = 5,
        padding: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dim_out = dim_out or dim_in

        self.depthwise_conv = nn.Conv1d(dim_in, dim_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=dim_in, bias=False)
        self.pointwise_conv = nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size, stride, padding=padding)
        self.residual_conv = nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=False) if dim_in != dim_out else None
        self.norm1 = nn.LayerNorm(dim_out)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D_in) -> (B, T // stride, D_out)``."""
        x = x.transpose(1, 2)
        y = self.pointwise_conv(self.depthwise_conv(x))

        residual = self.pool(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        y = (y + residual).transpose(1, 2)
        return self.dropout1(self.act1(self.norm1(y)))


class TransformerEncoder(nn.Module):
    """Input projection followed by an x-transformers encoder with rotary positions."""

    def __init__(self, input_dim: int, dim: int, num_layers: int = 1, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim != dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(dim, dim),
            )
        else:
            self.input_proj = nn.Identity()

        self.transformer = Encoder(
            dim=dim,
            depth=num_layers,
            heads=num_heads,
            layer_dropout=dropout,
            attn_dropout=dropout,
            ff_dropout=dropout,
            attn_flash=True,
            rotary_pos_emb=True,
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``padding_mask`` is ``(B, T)`` with ``True`` at padded frames."""
        mask = None if padding_mask is None else ~padding_mask
        return self.transformer(self.input_proj(x), mask=mask)


class SongFormer(nn.Module):
    """Predicts per-frame section boundaries and functional labels.

    Input is the concatenation of four SSL streams (MusicFM / MuQ, 30 s and 420 s windows) at 25 Hz.
    Output frames are downsampled by ``down_sample_conv_stride``.

    With ``use_dataset_id=False`` the dataset (data source) prefix embedding is not created and ``dataset_ids`` is
    ignored, so annotation-policy differences between training datasets are not given to the model.
    """

    def __init__(
        self,
        input_dim_raw: int = 4096,
        input_dim: int = 2048,
        down_sample_conv_kernel_size: int = 3,
        down_sample_conv_stride: int = 3,
        down_sample_conv_padding: int = 0,
        down_sample_conv_dropout: float = 0.1,
        transformer_encoder_input_dim: int = 1024,
        transformer_input_dim: int = 512,
        num_transformer_layers: int = 4,
        transformer_nhead: int = 8,
        transformer_dropout: float = 0.1,
        num_classes: int = 128,
        num_dataset_classes: int = 64,
        use_dataset_id: bool = True,
    ) -> None:
        super().__init__()
        self.downsample_rate = down_sample_conv_stride
        self.num_classes = num_classes
        self.transformer_encoder_input_dim = transformer_encoder_input_dim
        self.use_dataset_id = use_dataset_id

        self.mixed_win_downsample = nn.Linear(input_dim_raw, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.down_sample_conv = TimeDownsample(
            dim_in=input_dim,
            dim_out=transformer_encoder_input_dim,
            kernel_size=down_sample_conv_kernel_size,
            stride=down_sample_conv_stride,
            padding=down_sample_conv_padding,
            dropout=down_sample_conv_dropout,
        )
        if use_dataset_id:
            self.dataset_class_prefix = nn.Embedding(num_dataset_classes, transformer_encoder_input_dim)
        self.transformer = TransformerEncoder(
            input_dim=transformer_encoder_input_dim,
            dim=transformer_input_dim,
            num_layers=num_transformer_layers,
            num_heads=transformer_nhead,
            dropout=transformer_dropout,
        )
        self.boundary_head = Head(transformer_input_dim, 1)
        self.function_head = Head(transformer_input_dim, num_classes)

    def encode(self, embeddings: torch.Tensor, dataset_ids: torch.Tensor) -> torch.Tensor:
        """Downsampled audio features with the dataset prefix: ``(B, T, transformer_encoder_input_dim)``."""
        x = self.input_norm(self.mixed_win_downsample(embeddings))
        x = self.down_sample_conv(x)
        if not self.use_dataset_id:
            return x
        return x + self.dataset_class_prefix(dataset_ids).unsqueeze(1)

    def predict(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, label_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Transformer and heads on encoded features."""
        x = self.transformer(x, padding_mask=padding_mask)

        boundary_logits = self.boundary_head(x).squeeze(-1)
        function_logits = self.function_head(x)
        if label_mask is not None:
            function_logits = function_logits.masked_fill(label_mask.unsqueeze(1), -float("inf"))

        return {"boundary_logits": boundary_logits, "function_logits": function_logits}

    def forward(
        self,
        embeddings: torch.Tensor,
        dataset_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: ``(B, T_in, input_dim_raw)`` SSL features.
            dataset_ids: ``(B,)`` dataset (label-set) ids used as a conditioning prefix.
            padding_mask: ``(B, T)`` output-rate mask, ``True`` at frames to ignore in attention.
            label_mask: ``(B, num_classes)``, ``True`` at classes to exclude from ``function_logits``.

        Returns:
            ``boundary_logits`` ``(B, T)`` and ``function_logits`` ``(B, T, num_classes)``.
        """
        return self.predict(self.encode(embeddings, dataset_ids), padding_mask=padding_mask, label_mask=label_mask)
