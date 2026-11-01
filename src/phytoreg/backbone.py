from __future__ import annotations

"""Native PyTorch NTv3-style backbone for supervised regulatory modeling.

The architecture follows InstaDeep's public Nucleotide Transformer v3 design:
single-base embeddings, a seven-stage convolution tower, a RoPE/SwiGLU
Transformer torso, and a symmetric U-Net decoder.  This module is an
independent PyTorch implementation and does not import the upstream JAX code.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import BackboneConfig


def _conv1d_same(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
) -> nn.Conv1d:
    if kernel_size % 2 == 0:
        raise ValueError("same-padded convolution requires an odd kernel")
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
    )


class _ConvBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.conv = _conv1d_same(dim_in, dim_out, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x).transpose(1, 2)
        return F.gelu(self.conv(x).transpose(1, 2), approximate="tanh")


class _ResidualConvBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = _ConvBlock(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _DownsampleBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.conv = _ConvBlock(dim_in, dim_out, 5)
        self.residual = _ResidualConvBlock(dim_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.residual(self.conv(x))
        return F.avg_pool1d(x.transpose(1, 2), 2, 2).transpose(1, 2)


class _UpsampleConvBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.ConvTranspose1d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2 - 1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x).transpose(1, 2)
        # JAX lax.conv_transpose(..., padding="SAME") uses (3, 2)
        # transposed-convolution padding for kernel=5/stride=2 and does not
        # transpose the spatial kernel. PyTorch uses the opposite kernel
        # convention and symmetric padding, so flip the kernel and remove the
        # single extra value on the right to preserve NTv3's base alignment.
        x = F.conv_transpose1d(
            x,
            self.conv.weight.flip(-1),
            self.conv.bias,
            stride=2,
            padding=self.conv.padding,
        )[..., :-1]
        return F.gelu(x.transpose(1, 2), approximate="tanh")


class _UpsampleBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.up = _UpsampleConvBlock(dim_in, dim_out, 5)
        self.residual = _ResidualConvBlock(dim_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(self.up(x))


def _apply_rotary(x: Tensor, *, theta: float) -> Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("rotary embedding requires an even head dimension")
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    frequencies = torch.arange(
        0,
        head_dim,
        2,
        device=x.device,
        dtype=torch.float32,
    )
    frequencies = theta ** (-frequencies / head_dim)
    angles = torch.outer(positions, frequencies)
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
    # NTv3 follows the ESM/RoFormer convention: dimension i is paired with
    # i + head_dim / 2, rather than pairing adjacent even/odd dimensions.
    first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )


class _RotarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        dropout: float,
        theta: float,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.theta = theta
        # The upstream implementation disables the extra learned K/V tokens,
        # but its Q/K/V and output nnx.Linear projections retain their default
        # additive biases. A fused QKV projection is algebraically equivalent.
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor, valid_mask: Tensor | None) -> Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(value: Tensor) -> Tensor:
            return value.view(
                batch,
                length,
                self.heads,
                self.head_dim,
            ).transpose(1, 2)

        q = _apply_rotary(split_heads(q), theta=self.theta)
        k = _apply_rotary(split_heads(k), theta=self.theta)
        v = split_heads(v)
        attention_bias: Tensor | None = None
        if valid_mask is not None:
            if not torch.all(valid_mask.any(dim=1)):
                raise ValueError("each sequence must contain at least one valid token")
            attention_bias = torch.zeros(
                batch,
                1,
                1,
                length,
                dtype=x.dtype,
                device=x.device,
            )
            attention_bias = attention_bias.masked_fill(
                ~valid_mask[:, None, None, :],
                torch.finfo(x.dtype).min,
            )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out(attended)


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.input(x).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * value))


class _TransformerBlock(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        dim = config.transformer_dim
        self.attention_norm = nn.LayerNorm(dim, eps=1e-5)
        self.attention = _RotarySelfAttention(
            dim,
            config.transformer_heads,
            dropout=config.attention_dropout,
            theta=config.rotary_theta,
        )
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = _SwiGLU(
            dim,
            config.transformer_ffn_dim,
            config.transformer_dropout,
        )
        self.residual_dropout = nn.Dropout(config.transformer_dropout)

    def forward(self, x: Tensor, valid_mask: Tensor | None) -> Tensor:
        x = x + self.residual_dropout(
            self.attention(self.attention_norm(x), valid_mask)
        )
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class NTv3Backbone(nn.Module):
    """NTv3 U-Net backbone for PhytoReg regulatory features at 32-bp resolution."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        config.validate()
        assert config.conv_init_embed_dim is not None
        self.config = config
        self.base_embedding = nn.Embedding(6, config.token_embed_dim)
        self.stem = _conv1d_same(
            config.token_embed_dim,
            config.conv_init_embed_dim,
            15,
        )
        filter_list = (config.conv_init_embed_dim, *config.encoder_channels)
        self.down_blocks = nn.ModuleList(
            _DownsampleBlock(dim_in, dim_out)
            for dim_in, dim_out in zip(
                filter_list[:-1],
                filter_list[1:],
                strict=True,
            )
        )
        self.transformer = nn.ModuleList(
            _TransformerBlock(config)
            for _ in range(config.transformer_layers)
        )
        reversed_filters = tuple(reversed(filter_list))
        self.up_blocks = nn.ModuleList(
            _UpsampleBlock(dim_in, dim_out)
            for dim_in, dim_out in zip(
                reversed_filters[:-1],
                reversed_filters[1:],
                strict=True,
            )
        )

    @property
    def output_dim(self) -> int:
        assert self.config.conv_init_embed_dim is not None
        return self.config.conv_init_embed_dim

    def _validate_inputs(
        self,
        dna: Tensor,
        bottleneck_valid_mask: Tensor | None,
    ) -> None:
        if dna.ndim != 3 or tuple(dna.shape[1:]) != (
            self.config.input_length_bp,
            self.config.input_channels,
        ):
            raise ValueError(
                "dna must have shape "
                f"[batch, {self.config.input_length_bp}, "
                f"{self.config.input_channels}], got {tuple(dna.shape)}"
            )
        if bottleneck_valid_mask is not None and tuple(
            bottleneck_valid_mask.shape
        ) != (dna.shape[0], self.config.bottleneck_tokens):
            raise ValueError(
                "bottleneck_valid_mask must have shape "
                f"[batch, {self.config.bottleneck_tokens}]"
            )

    def _embed_dna(self, dna: Tensor) -> Tensor:
        token_ids = dna.argmax(dim=-1)
        dynamic_mask = dna.abs().sum(dim=-1) == 0
        token_ids = token_ids.masked_fill(dynamic_mask, 5)
        return self.base_embedding(token_ids)

    def _transformer_forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        for block in self.transformer:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden, module=block: module(hidden, valid_mask),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, valid_mask)
        return x

    def forward_base_features(
        self,
        dna: Tensor,
        bottleneck_valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return full-resolution U-Net features derived from DNA alone."""

        self._validate_inputs(dna, bottleneck_valid_mask)
        x = F.gelu(
            self.stem(self._embed_dna(dna).transpose(1, 2)),
            approximate="tanh",
        ).transpose(1, 2)
        residuals: list[Tensor] = []
        for block in self.down_blocks:
            residuals.append(x)
            x = (checkpoint(block, x, use_reentrant=False)
                 if self.config.gradient_checkpointing and self.training and x.requires_grad
                 else block(x))
        bottleneck = self._transformer_forward(x, bottleneck_valid_mask)
        x = bottleneck
        for block, residual in zip(
            self.up_blocks,
            reversed(residuals),
            strict=True,
        ):
            x = (checkpoint(block, x, use_reentrant=False)
                 if self.config.gradient_checkpointing and self.training and x.requires_grad
                 else block(x))
            if x.shape != residual.shape:
                raise RuntimeError(
                    "NTv3 decoder/skip shape mismatch: "
                    f"{tuple(x.shape)} versus {tuple(residual.shape)}"
                )
            if self.config.use_skip_connection:
                x = x + residual
        if x.shape[1] != self.config.input_length_bp:
            raise RuntimeError(
                f"NTv3 decoder produced {x.shape[1]} bases, expected "
                f"{self.config.input_length_bp}"
            )
        return {
            "base_1bp": x,
            "bottleneck_128bp": bottleneck,
        }

    def central_crop_32bp(self, full_32bp: Tensor) -> Tensor:
        output_bins = self.config.downstream_output_bins
        crop_each_side = (full_32bp.shape[1] - output_bins) // 2
        return full_32bp[:, crop_each_side : crop_each_side + output_bins]

    def forward_features(
        self,
        dna: Tensor,
        bottleneck_valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        features = self.forward_base_features(dna, bottleneck_valid_mask)
        base = features["base_1bp"]
        latent_32 = F.avg_pool1d(
            base.transpose(1, 2),
            kernel_size=32,
            stride=32,
        ).transpose(1, 2)
        return {
            "latent_32bp": latent_32,
            "regulatory_32bp": self.central_crop_32bp(latent_32),
            "bottleneck_128bp": features["bottleneck_128bp"],
        }

    def forward(
        self,
        dna: Tensor,
        bottleneck_valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.forward_features(dna, bottleneck_valid_mask)


def ntv3_filter_list(
    conv_init_embed_dim: int,
    transformer_dim: int,
    num_downsamples: int = 7,
) -> tuple[int, ...]:
    """Return NTv3's linearly spaced convolution widths."""

    if num_downsamples <= 0:
        raise ValueError("num_downsamples must be positive")
    return tuple(
        int(
            conv_init_embed_dim
            + (transformer_dim - conv_init_embed_dim)
            * index
            / num_downsamples
        )
        for index in range(num_downsamples + 1)
    )
