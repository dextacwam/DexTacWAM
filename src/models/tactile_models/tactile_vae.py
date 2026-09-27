"""Stage 1 Tactile VAE main classes.

Top-level wrapper :class:`TactileVAE` and its three sub-modules:

* :class:`TactileEncoder` — per-finger 3D causal CNN backbone followed by
  ``N`` interleaved transformer blocks (``CrossAttn(pose) → SelfAttn(finger)
  → FFN``) and an LTX-style ``mu/logvar`` head. Produces a per-finger
  spatial-temporal latent ``z (B, 5, 128, T_lat, 3, 4)`` that satisfies
  the C1–C4 alignment contracts from ``docs/tactile_vae_stage1.md``.

* :class:`TactileFlowDecoder` — per-finger pixel-shuffle upsampling stack
  built from LTX modules (``LTXVideoResnetBlock3d`` + ``LTXVideoUpsampler3d``)
  that maps ``z`` to ``(B, 5, T_out, 24, 32, 3)`` flow ``(dx, dy, divergence)``.

* :class:`TactilePoseDecoder` — auxiliary global head that mean-pools ``z``
  over ``(T_lat, h, w)`` while keeping the 5-finger axis, then a small MLP
  predicts the 22-dim hand pose. Acts as diffuse pose-aware regularization
  on the latent (see plan §2b).

The internal :class:`TransformerBlock` and the two MHA helpers are kept in
this file because they are not reused outside the encoder.

Conventions
-----------
* All attention uses ``F.scaled_dot_product_attention`` (Flash / mem-eff
  paths transparently selected by PyTorch), matching the LTX codebase.
* Pre-norm transformer blocks: ``x = x + sublayer(LayerNorm(x))``.
* QKV / output projections have no bias (modern style); FFN has bias.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..ltx_models.autoencoder_kl_ltx import (
    LTXVideoCausalConv3d,
    LTXVideoResnetBlock3d,
    LTXVideoUpsampler3d,
)
from .tactile_modules import (
    FingerCausalConv3d,
    FingerPositionEmbedding,
    ModalityEmbedding,
    PoseTokenizer,
)


# ---------------------------------------------------------------------------
# Multi-head attention helpers (built on F.scaled_dot_product_attention)
# ---------------------------------------------------------------------------


class _MultiHeadCrossAttention(nn.Module):
    """Standard MHA cross-attention with separate Q, K, V projections.

    Shape contract::

        x       (B, N, C)        Q source (queries)
        context (B, M, C)        K/V source (keys/values)
        out     (B, N, C)
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # SDPA applies its own scale; kept for clarity.

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        m = context.shape[1]

        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(b, n, c)
        return self.out_proj(attn)


class _MultiHeadSelfAttention(nn.Module):
    """Standard MHA self-attention with a fused QKV projection."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(b, n, c)
        return self.out_proj(attn)


class _FFN(nn.Module):
    """Standard transformer FFN: ``Linear → GELU → Linear`` with optional dropout."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# Transformer block: PoseCrossAttn -> InterFingerSelfAttn -> FFN  (pre-norm)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """One interleaved tactile transformer block.

    Three sub-layers, each pre-normed and residually added::

        x ← x + CrossAttn(LN(x), LN(pose_kv))                # pose conditioning
        x ← x + SelfAttn over 5-finger axis(LN(x))           # inter-finger fusion
        x ← x + FFN(LN(x))                                   # non-linear capacity

    The self-attention temporarily reshapes the token sequence so that the
    finger axis becomes the attention axis: each ``(t, h, w)`` cell sees a
    length-5 mini-sequence of fingers.

    Shape contract::

        x         (B, F * T_lat * H_lat * W_lat, C)          tactile token grid
        pose_kv   (B, K_pose, C)                             reused across blocks
        out       same shape as ``x``
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        self.norm_q_ca = nn.LayerNorm(dim)
        self.norm_kv_ca = nn.LayerNorm(dim)
        self.cross_attn = _MultiHeadCrossAttention(dim, num_heads, dropout)

        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn_finger = _MultiHeadSelfAttention(dim, num_heads, dropout)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, ffn_mult * dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        pose_kv: torch.Tensor,
        num_fingers: int,
    ) -> torch.Tensor:
        # 1) Pose cross-attention (Q=tactile, K/V=pose tokens, normed independently).
        x = x + self.cross_attn(self.norm_q_ca(x), self.norm_kv_ca(pose_kv))

        # 2) Inter-finger self-attention.
        b, n, c = x.shape
        if n % num_fingers != 0:
            raise ValueError(
                f"Token count ({n}) is not divisible by num_fingers ({num_fingers})."
            )
        thw = n // num_fingers
        # (B, F*thw, C) -> (B*thw, F, C) so attention happens across 5 fingers
        # at each (t, h, w) cell, with shared weights across all cells.
        x_finger = rearrange(x, "b (f thw) c -> (b thw) f c", f=num_fingers, thw=thw)
        x_finger = x_finger + self.self_attn_finger(self.norm_sa(x_finger))
        x = rearrange(x_finger, "(b thw) f c -> b (f thw) c", b=b, thw=thw)

        # 3) Position-wise FFN.
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# TactileEncoder: CNN -> embeddings -> N transformer blocks -> mu/logvar head
# ---------------------------------------------------------------------------


class TactileEncoder(nn.Module):
    """Cross-modal tactile encoder.

    Forward pipeline (with default config and ``T=9``)::

        tactile (B, 5, 1, 9, 192, 256), hand_pose (B, 22)
            │
            ▼  FingerCausalConv3d (shared per-finger CNN)
        (B, 5, 256, 2, 3, 4)
            │  + FingerPositionEmbedding + ModalityEmbedding
            ▼  rearrange 'b f c t h w -> b (f t h w) c'
        (B, 120, 256)                tactile tokens
            │  PoseTokenizer(hand_pose) -> (B, 5, 256) pose K/V (reused per block)
            ▼  TransformerBlock × N=3   (CA -> SA over fingers -> FFN)
        (B, 120, 256)
            │  rearrange back -> (B, 5, 256, 2, 3, 4)
            ▼  per-finger Conv3d head: 256 -> 128 + 1
        mu     (B, 5, 128, 2, 3, 4)
        logvar (B, 5,   1, 2, 3, 4)  scalar logvar in C-channel sense
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        latent_channels: int = 128,
        num_fingers: int = 5,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        # Backbone CNN config (mirrors LTXVideoEncoder3d shapes).
        patch_size: int = 8,
        patch_size_t: int = 1,
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 256),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (1, 1, 1, 1, 2),
        # Pose tokenizer config.
        pose_dim: int = 22,
        num_pose_tokens: int = 5,
        pose_hidden_dim: int = 256,
        is_causal: bool = True,
    ):
        super().__init__()

        if block_out_channels[-1] != hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal the final entry of "
                f"block_out_channels ({block_out_channels[-1]}); the CNN "
                "backbone outputs `block_out_channels[-1]` channels."
            )

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_channels = latent_channels
        self.num_fingers = num_fingers
        self.num_layers = num_layers
        self.is_causal = is_causal

        self.backbone = FingerCausalConv3d(
            in_channels=in_channels,
            block_out_channels=block_out_channels,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            is_causal=is_causal,
        )

        self.finger_pe = FingerPositionEmbedding(num_fingers=num_fingers, dim=hidden_dim)
        self.modality = ModalityEmbedding(dim=hidden_dim)

        self.pose_tokenizer = PoseTokenizer(
            pose_dim=pose_dim,
            num_tokens=num_pose_tokens,
            hidden_dim=pose_hidden_dim,
            token_dim=hidden_dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(hidden_dim)

        # LTX-style head: 128 mu channels + 1 scalar logvar channel (C4 contract).
        # We use a causal Conv3d so the head respects the encoder's causal pattern.
        self.head = LTXVideoCausalConv3d(
            in_channels=hidden_dim,
            out_channels=latent_channels + 1,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
        )

    def forward(
        self, tactile: torch.Tensor, hand_pose: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tactile.ndim != 6:
            raise ValueError(
                f"tactile must be (B, F, C, T, H, W); got shape {tuple(tactile.shape)}."
            )
        if hand_pose.ndim != 2:
            raise ValueError(
                f"hand_pose must be (B, pose_dim); got shape {tuple(hand_pose.shape)}."
            )

        b, f, _, _, _, _ = tactile.shape
        if f != self.num_fingers:
            raise ValueError(
                f"Expected {self.num_fingers} fingers; got {f}."
            )

        h = self.backbone(tactile)                 # (B, 5, 256, T_lat, 3, 4)
        _, _, _, t_lat, h_lat, w_lat = h.shape

        h = self.finger_pe(h)
        h = self.modality(h)

        tokens = rearrange(h, "b f c t h w -> b (f t h w) c")  # (B, 5*T_lat*3*4, 256)
        pose_kv = self.pose_tokenizer(hand_pose)               # (B, K_pose, 256)

        for block in self.blocks:
            tokens = block(tokens, pose_kv, num_fingers=f)

        tokens = self.norm_out(tokens)
        h = rearrange(
            tokens, "b (f t h w) c -> b f c t h w",
            f=f, t=t_lat, h=h_lat, w=w_lat,
        )

        # Per-finger head with shared weights via batch flatten.
        h_flat = rearrange(h, "b f c t h w -> (b f) c t h w")
        out = self.head(h_flat)                                # ((b f), C+1, T_lat, 3, 4)
        out = rearrange(out, "(b f) c t h w -> b f c t h w", b=b, f=f)

        mu = out[:, :, : self.latent_channels]                 # (B, 5, 128, T_lat, 3, 4)
        logvar = out[:, :, self.latent_channels : self.latent_channels + 1]
        # logvar shape: (B, 5, 1, T_lat, 3, 4)  -- one scalar logvar per spatial-temporal cell
        return mu, logvar


# ---------------------------------------------------------------------------
# TactileFlowDecoder: pixel-shuffle upsampling -> (dx, dy, divergence)
# ---------------------------------------------------------------------------


class _SpatialUp2x(nn.Module):
    """One spatial-only 2× pixel-shuffle up-stage with channel preservation.

    Wraps an :class:`LTXVideoUpsampler3d` configured with ``stride=(1,2,2)``
    and ``upscale_factor=1`` so the internal conv inflates channels by 4×
    and the pixel-shuffle restores the original channel count while
    doubling spatial resolution. A pre-upsample :class:`LTXVideoResnetBlock3d`
    provides feature processing at the lower resolution.
    """

    def __init__(self, channels: int, is_causal: bool = True):
        super().__init__()
        self.resnet = LTXVideoResnetBlock3d(
            in_channels=channels, out_channels=channels, is_causal=is_causal
        )
        self.upsample = LTXVideoUpsampler3d(
            in_channels=channels,
            stride=(1, 2, 2),
            is_causal=is_causal,
            residual=False,
            upscale_factor=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.upsample(x)
        return x


class _TemporalUp8x(nn.Module):
    """One temporal-only 8× pixel-shuffle up-stage matching encoder's 8× compression.

    Stride ``(8, 1, 1)`` with ``upscale_factor=1``: conv inflates channels
    by 8×, pixel-shuffle restores channels and produces ``8 * T_lat`` frames.
    The causal trim drops the leading ``stride[0] - 1 = 7`` frames so the
    final temporal length is ``T_out = 8 * T_lat - 7 = 1 + (T_lat - 1) * 8``,
    which equals the encoder input ``T`` for both ``T_lat=1`` (→ ``T=1``) and
    ``T_lat=2`` (→ ``T=9``).
    """

    def __init__(self, channels: int, is_causal: bool = True):
        super().__init__()
        self.resnet = LTXVideoResnetBlock3d(
            in_channels=channels, out_channels=channels, is_causal=is_causal
        )
        self.upsample = LTXVideoUpsampler3d(
            in_channels=channels,
            stride=(8, 1, 1),
            is_causal=is_causal,
            residual=False,
            upscale_factor=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.upsample(x)
        return x


class TactileFlowDecoder(nn.Module):
    """Per-finger decoder that maps ``z`` to the 3-channel tactile flow.

    Per-finger pipeline (shared weights via batch flatten)::

        z_i (B, 128, T_lat, 3, 4)
            ▼  Conv3d(128 → 64, k=1)                      channel reduction
        (B, 64, T_lat, 3, 4)
            ▼  3 × _SpatialUp2x                           3 × 2 spatial = 8 ×
        (B, 64, T_lat, 24, 32)
            ▼  _TemporalUp8x                              8 × temporal
        (B, 64, T_out, 24, 32)
            ▼  LTXVideoResnetBlock3d                      final feature processing
            ▼  Conv3d(64 → 3, k=3)                        output projection
        flow_pred_i (B, 3, T_out, 24, 32)

    Output (after stacking 5 fingers): ``(B, 5, T_out, 24, 32, 3)`` matching
    the dataset's flow ground-truth layout (channel last).

    The decoder uses a *separate* :class:`FingerPositionEmbedding` from the
    encoder because it operates at the latent's 128-dim channel, whereas
    the encoder PE is at the backbone's 256-dim channel. The CLASS is
    reused (same logic, same init) but Parameter weights are independent.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        decoder_channels: int = 64,
        out_channels: int = 3,
        num_fingers: int = 5,
        is_causal: bool = True,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels
        self.num_fingers = num_fingers

        # Re-tag finger identity at the latent's 128-dim channel before
        # any shared per-finger processing.
        self.finger_pe = FingerPositionEmbedding(num_fingers=num_fingers, dim=latent_channels)

        self.conv_in = LTXVideoCausalConv3d(
            in_channels=latent_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            stride=1,
            is_causal=is_causal,
        )

        self.spatial_up_1 = _SpatialUp2x(decoder_channels, is_causal=is_causal)
        self.spatial_up_2 = _SpatialUp2x(decoder_channels, is_causal=is_causal)
        self.spatial_up_3 = _SpatialUp2x(decoder_channels, is_causal=is_causal)
        self.temporal_up = _TemporalUp8x(decoder_channels, is_causal=is_causal)

        self.resnet_out = LTXVideoResnetBlock3d(
            in_channels=decoder_channels, out_channels=decoder_channels, is_causal=is_causal
        )
        self.conv_out = LTXVideoCausalConv3d(
            in_channels=decoder_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 6:
            raise ValueError(
                f"z must be (B, F, C, T_lat, H, W); got shape {tuple(z.shape)}."
            )
        b, f, c, t_lat, h_lat, w_lat = z.shape
        if f != self.num_fingers:
            raise ValueError(
                f"Expected {self.num_fingers} fingers; got {f}."
            )
        if c != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels; got {c}."
            )

        z = self.finger_pe(z)

        # Shared per-finger processing via batch flatten.
        x = rearrange(z, "b f c t h w -> (b f) c t h w")
        x = self.conv_in(x)
        x = self.spatial_up_1(x)
        x = self.spatial_up_2(x)
        x = self.spatial_up_3(x)
        x = self.temporal_up(x)
        x = self.resnet_out(x)
        x = self.conv_out(x)                                   # ((b f), 3, T_out, 24, 32)

        # (B, 5, T_out, 24, 32, 3) — channel-last to match GT layout.
        flow = rearrange(x, "(b f) c t h w -> b f t h w c", b=b, f=f)
        return flow


# ---------------------------------------------------------------------------
# TactilePoseDecoder: spatial-temporal pool (keep finger) -> MLP -> 22-dim
# ---------------------------------------------------------------------------


class TactilePoseDecoder(nn.Module):
    """Auxiliary head regressing 22-dim hand pose from the latent.

    Pipeline::

        z (B, 5, 128, T_lat, 3, 4)
            → mean over (T_lat, h, w)          (B, 5, 128)
            → flatten finger × channel         (B, 640)
            → Linear(640 → hidden) → GELU
            → Linear(hidden → 22)
        pose_hat (B, 22)

    See plan §2b for the design rationale (T-invariance, diffuse
    regularization, complementary global signal to the flow decoder).
    """

    def __init__(
        self,
        latent_channels: int = 128,
        num_fingers: int = 5,
        hidden_dim: int = 256,
        pose_dim: int = 22,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_fingers = num_fingers
        self.pose_dim = pose_dim

        self.mlp = nn.Sequential(
            nn.Linear(num_fingers * latent_channels, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, pose_dim, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 6:
            raise ValueError(
                f"z must be (B, F, C, T_lat, H, W); got shape {tuple(z.shape)}."
            )
        # Mean over (T_lat, h, w); keep (B, F, C). dims=(3,4,5).
        pooled = z.mean(dim=(3, 4, 5))                         # (B, F, C)
        flat = pooled.flatten(1)                               # (B, F * C)
        return self.mlp(flat)


# ---------------------------------------------------------------------------
# TactileVAE top-level wrapper
# ---------------------------------------------------------------------------


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE reparameterization. ``logvar`` may broadcast against ``mu``."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) summed over the channel axis, mean over everything else.

    ``logvar`` is broadcast against ``mu`` so the formula is per-mu-element:
    ``-0.5 * (1 + logvar - mu^2 - exp(logvar))``.
    """
    logvar_b = logvar.expand_as(mu)
    kl_per_elem = -0.5 * (1.0 + logvar_b - mu.pow(2) - logvar_b.exp())
    return kl_per_elem.mean()


class TactileVAE(nn.Module):
    """Top-level Stage 1 wrapper. Bundles encoder + flow decoder + pose decoder.

    Stage 1 ``forward`` returns a dict with everything the trainer needs::

        {
          "mu":        (B, 5, 128, T_lat, 3, 4),
          "logvar":    (B, 5,   1, T_lat, 3, 4),
          "z":         (B, 5, 128, T_lat, 3, 4),
          "flow_pred": (B, 5, T_out, 24, 32, 3),
          "pose_pred": (B, 22),
        }

    Use ``sample=False`` (defaults to ``True``) at validation/inference to
    take the latent mean instead of sampling.
    """

    def __init__(
        self,
        # Encoder.
        in_channels: int = 1,
        hidden_dim: int = 256,
        latent_channels: int = 128,
        num_fingers: int = 5,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        patch_size: int = 8,
        patch_size_t: int = 1,
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 256),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (1, 1, 1, 1, 2),
        pose_dim: int = 22,
        num_pose_tokens: int = 5,
        pose_hidden_dim: int = 256,
        is_causal: bool = True,
        # Decoder.
        flow_decoder_channels: int = 64,
        flow_out_channels: int = 3,
        # Pose head.
        pose_decoder_hidden: int = 256,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.num_fingers = num_fingers
        self.is_causal = is_causal

        self.encoder = TactileEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            latent_channels=latent_channels,
            num_fingers=num_fingers,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            block_out_channels=block_out_channels,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            pose_dim=pose_dim,
            num_pose_tokens=num_pose_tokens,
            pose_hidden_dim=pose_hidden_dim,
            is_causal=is_causal,
        )

        self.flow_decoder = TactileFlowDecoder(
            latent_channels=latent_channels,
            decoder_channels=flow_decoder_channels,
            out_channels=flow_out_channels,
            num_fingers=num_fingers,
            is_causal=is_causal,
        )

        self.pose_decoder = TactilePoseDecoder(
            latent_channels=latent_channels,
            num_fingers=num_fingers,
            hidden_dim=pose_decoder_hidden,
            pose_dim=pose_dim,
        )

    # ------------------------------------------------------------------
    # Encoding / decoding API
    # ------------------------------------------------------------------

    def encode(
        self, tactile: torch.Tensor, hand_pose: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = tactile.shape[3]
        mu, logvar = self.encoder(tactile, hand_pose)

        # Runtime enforcement of the C1, C3, C4 contracts (cheap, catches drift early).
        expected_T_lat = 1 + (T - 1) // 8 if T > 1 else 1
        assert mu.shape[2] == self.latent_channels, (
            f"C1 violated: latent channel = {mu.shape[2]}, expected "
            f"{self.latent_channels}."
        )
        assert mu.shape[3] == expected_T_lat, (
            f"C3 violated: T_lat = {mu.shape[3]}, expected {expected_T_lat} "
            f"for T = {T}."
        )
        assert logvar.shape[2] == 1, (
            f"C4 violated: logvar must have 1 channel (LTX scalar logvar); "
            f"got {logvar.shape[2]}."
        )
        return mu, logvar

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow_decoder(z), self.pose_decoder(z)

    # ------------------------------------------------------------------
    # Full forward (used by the trainer)
    # ------------------------------------------------------------------

    def forward(
        self,
        tactile: torch.Tensor,
        hand_pose: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(tactile, hand_pose)
        z = reparameterize(mu, logvar) if sample else mu
        flow_pred, pose_pred = self.decode(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "flow_pred": flow_pred,
            "pose_pred": pose_pred,
        }

    # ------------------------------------------------------------------
    # Convenience: gradient checkpointing toggle (forwarded to backbone)
    # ------------------------------------------------------------------

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.encoder.backbone.gradient_checkpointing = enabled
