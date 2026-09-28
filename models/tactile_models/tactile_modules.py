"""Building blocks for the Stage 1 Tactile VAE.

Contains four reusable modules shared by `TactileEncoder`, `TactileFlowDecoder`,
and `TactilePoseDecoder`:

  * ``FingerCausalConv3d`` — per-finger 3D causal convolutional backbone that
    mirrors :class:`models.ltx_models.autoencoder_kl_ltx.LTXVideoEncoder3d`
    *up to but excluding* the final mu/logvar head. Weights are shared across
    the 5 fingers via batch flattening.
  * ``PoseTokenizer`` — maps a 22-dim per-hand joint vector to ``K`` learned
    virtual pose tokens (``(B, K, dim)``) used as cross-attention K/V.
  * ``FingerPositionEmbedding`` — learnable ``(F, dim)`` table broadcast over
    every spatio-temporal cell. The same instance is reused by the encoder
    (after the CNN) and the flow decoder (after the upsampling stack) to
    re-tag finger identity for shared per-finger weights.
  * ``ModalityEmbedding`` — single learnable bias added to every tactile
    token to mark the modality, so a downstream DiT can distinguish tactile
    tokens from visual ones in Stage 3.

See :mod:`models.tactile_models.tactile_vae` for the surrounding
architecture and the C1–C4 latent alignment contracts.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

# RMSNorm + SiLU are reused for the post-backbone activation, mirroring LTX's
# encoder so that the channel statistics entering the attention stack match
# what an LTX-style head expects.
from diffusers.models.normalization import RMSNorm

from ..ltx_models.autoencoder_kl_ltx import (
    LTXVideoCausalConv3d,
    LTXVideoDownBlock3D,
    LTXVideoMidBlock3d,
)


# ---------------------------------------------------------------------------
# Per-finger 3D causal conv backbone (shared weights across 5 fingers)
# ---------------------------------------------------------------------------


class FingerCausalConv3d(nn.Module):
    """Per-finger 3D causal conv backbone with shared weights.

    Mirrors the structure of LTX's ``LTXVideoEncoder3d`` (patchify ``→``
    ``conv_in`` ``→`` down blocks ``→`` mid block ``→`` ``norm_out`` ``→``
    activation) but **stops before** the ``conv_out`` mu/logvar head — the
    tactile encoder inserts attention layers between the backbone and the
    head, so we cannot bake the head in here.

    Shape contract::

        Input :  (B, F, C_in,  T,     H,     W)         per-hand tactile clip
        Output:  (B, F, C_out, T_lat, H_lat, W_lat)     per-finger features

    With the defaults (``patch_size=8``, four down blocks with
    ``spatio_temporal_scaling=(T,T,T,F)``, and the LTX causal padding rule
    ``T_lat = 1 + (T-1)/8``)::

        (B, 5, 1, T,   192, 256)  ->  (B, 5, 256, T_lat, 3, 4)

    which satisfies the spatial half of contract C2 (3×4 per finger gives
    60 tokens per hand per latent frame, balanced against the visual VAE)
    and the temporal half of contract C3 (8× causal compression matching
    ``AutoencoderKLLTXVideo``).

    Weight sharing across the 5 fingers is implemented by flattening the
    finger axis into the batch axis at the start of ``forward``: a single
    set of weights sees ``B * F`` per-finger clips.
    """

    def __init__(
        self,
        in_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 256),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (1, 1, 1, 1, 2),
        patch_size: int = 8,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = True,
    ):
        super().__init__()

        if len(spatio_temporal_scaling) != len(block_out_channels):
            raise ValueError(
                "spatio_temporal_scaling must have the same length as "
                f"block_out_channels (got {len(spatio_temporal_scaling)} vs "
                f"{len(block_out_channels)})."
            )
        if len(layers_per_block) != len(block_out_channels) + 1:
            raise ValueError(
                "layers_per_block must have length len(block_out_channels) + 1 "
                f"(got {len(layers_per_block)} vs "
                f"{len(block_out_channels) + 1}); the last entry is for the "
                "mid-block."
            )

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.block_out_channels = tuple(block_out_channels)
        self.spatio_temporal_scaling = tuple(spatio_temporal_scaling)
        self.is_causal = is_causal

        # After patchify, channels are inflated by p_t * p * p, just like LTX.
        patched_in_channels = in_channels * patch_size * patch_size * patch_size_t

        output_channel = block_out_channels[0]
        self.conv_in = LTXVideoCausalConv3d(
            in_channels=patched_in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
        )

        # Down blocks. Channel progression matches LTX:
        # input_channel = previous output, output_channel = block_out_channels[i+1]
        # (or block_out_channels[i] for the last block).
        num_down = len(block_out_channels)
        down_blocks = []
        for i in range(num_down):
            input_channel = output_channel
            output_channel = (
                block_out_channels[i + 1]
                if i + 1 < num_down
                else block_out_channels[i]
            )
            down_blocks.append(
                LTXVideoDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    is_causal=is_causal,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_block = LTXVideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
        )

        # Mirror LTX's post-backbone normalization so downstream channel
        # statistics are comparable. RMSNorm with elementwise_affine=False has
        # no learnable per-channel weights, so the size argument is ignored
        # at runtime — kept here only for code-path symmetry.
        self.norm_out = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
        self.act_out = nn.SiLU()

        self.out_channels = output_channel
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Patchify (replicates LTXVideoEncoder3d's reshape/permute exactly)
    # ------------------------------------------------------------------

    def _patchify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` -> ``(B, C * p_t * p * p, T/p_t, H/p, W/p)``."""

        p, p_t = self.patch_size, self.patch_size_t
        b, c, t, h, w = hidden_states.shape
        if t % p_t != 0 or h % p != 0 or w % p != 0:
            raise ValueError(
                f"Input shape ({t},{h},{w}) is not divisible by patch sizes "
                f"(p_t={p_t}, p={p})."
            )
        post_t = t // p_t
        post_h = h // p
        post_w = w // p
        hidden_states = hidden_states.reshape(
            b, c, post_t, p_t, post_h, p, post_w, p
        )
        # b, c, n_p_t, p_t, n_p_h, p, n_p_w, p
        # -> b, (c * p_t * p * p), n_p_t, n_p_h, n_p_w
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        return hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a per-hand tactile clip.

        Args:
            x: ``(B, F, C_in, T, H, W)`` per-hand tactile clip (typically
                ``F=5`` fingers, ``C_in=1`` grayscale, ``H=192``, ``W=256``).

        Returns:
            ``(B, F, C_out, T_lat, H_lat, W_lat)`` per-finger features at
            ``C_out = block_out_channels[-1]`` channels (no mu/logvar head
            applied).
        """

        if x.ndim != 6:
            raise ValueError(
                f"FingerCausalConv3d expects a 6-D tensor (B,F,C,T,H,W); got "
                f"shape {tuple(x.shape)}."
            )

        b, f, c, t, h, w = x.shape
        hidden_states = rearrange(x, "b f c t h w -> (b f) c t h w")

        hidden_states = self._patchify(hidden_states)
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    down_block, hidden_states, use_reentrant=False
                )
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.mid_block, hidden_states, use_reentrant=False
            )
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)
            hidden_states = self.mid_block(hidden_states)

        # Channel-last RMSNorm (mirrors LTXVideoEncoder3d.forward).
        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.act_out(hidden_states)

        return rearrange(hidden_states, "(b f) c t h w -> b f c t h w", b=b, f=f)


# ---------------------------------------------------------------------------
# Pose tokenizer  (22-dim hand pose  ->  K virtual tokens)
# ---------------------------------------------------------------------------


class PoseTokenizer(nn.Module):
    """Map a per-hand joint vector to ``K`` learned virtual pose tokens.

    Implementation: ``Linear(pose_dim → hidden) → GELU → Linear(hidden → K * dim)``
    followed by a reshape. Producing **multiple** tokens (default ``K=5``) is
    crucial: a single K/V token would make cross-attention degenerate to an
    additive bias (FiLM), since softmax over a single key always returns 1.

    Shape contract::

        Input :  (B, pose_dim)                e.g. (B, 22) hand joints
        Output:  (B, num_tokens, token_dim)   K/V tokens for cross-attention
    """

    def __init__(
        self,
        pose_dim: int = 22,
        num_tokens: int = 5,
        hidden_dim: int = 256,
        token_dim: int = 256,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * token_dim),
        )

    def forward(self, hand_pose: torch.Tensor) -> torch.Tensor:
        if hand_pose.ndim != 2 or hand_pose.shape[-1] != self.pose_dim:
            raise ValueError(
                f"PoseTokenizer expects (B, {self.pose_dim}); got shape "
                f"{tuple(hand_pose.shape)}."
            )
        b = hand_pose.shape[0]
        return self.proj(hand_pose).view(b, self.num_tokens, self.token_dim)


# ---------------------------------------------------------------------------
# Per-finger position embedding (reused by encoder and flow decoder)
# ---------------------------------------------------------------------------


class FingerPositionEmbedding(nn.Module):
    """Learnable per-finger position embedding.

    Stores a ``(num_fingers, dim)`` table that is **added once** after the
    CNN backbone (in the encoder) and **reused** after the upsampling stack
    (in the flow decoder) to re-tag finger identity for the shared
    per-finger weights.

    The embedding broadcasts over every spatio-temporal cell so that the
    finger identity is visible to every downstream attention/decoder
    operation without being recomputed at each layer.
    """

    def __init__(self, num_fingers: int = 5, dim: int = 256):
        super().__init__()
        self.num_fingers = num_fingers
        self.dim = dim
        self.embed = nn.Parameter(torch.zeros(num_fingers, dim))
        nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the finger embedding to a per-finger feature tensor.

        Args:
            x: either ``(B, F, C, T, H, W)`` (encoder/decoder feature map) or
                ``(B, F, C)`` (pooled per-finger summary).

        Returns:
            ``x`` with ``self.embed`` broadcast-added along the channel axis.
        """

        if x.ndim == 6:
            b, f, c, t, h, w = x.shape
            if f != self.num_fingers or c != self.dim:
                raise ValueError(
                    f"FingerPositionEmbedding shape mismatch: feature "
                    f"({f},{c}) vs embedding ({self.num_fingers},{self.dim})."
                )
            return x + self.embed.view(1, f, c, 1, 1, 1)
        if x.ndim == 3:
            b, f, c = x.shape
            if f != self.num_fingers or c != self.dim:
                raise ValueError(
                    f"FingerPositionEmbedding shape mismatch: feature "
                    f"({f},{c}) vs embedding ({self.num_fingers},{self.dim})."
                )
            return x + self.embed.view(1, f, c)
        raise ValueError(
            f"FingerPositionEmbedding expects 6-D or 3-D input; got shape "
            f"{tuple(x.shape)}."
        )


# ---------------------------------------------------------------------------
# Modality embedding  (one learnable bias marking "this is tactile")
# ---------------------------------------------------------------------------


class ModalityEmbedding(nn.Module):
    """Single learnable bias added to every tactile token.

    Cheap (~``dim`` parameters) but useful for Stage 3, where tactile and
    visual tokens are concatenated into the DiT's input sequence — the bias
    gives the DiT a constant "tactile here" signal it can read alongside
    positional and finger embeddings.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 6:
            return x + self.bias.view(1, 1, -1, 1, 1, 1)
        if x.ndim == 5:
            return x + self.bias.view(1, -1, 1, 1, 1)
        if x.ndim == 3:
            return x + self.bias.view(1, 1, -1)
        raise ValueError(
            f"ModalityEmbedding expects a 3-, 5-, or 6-D input; got shape "
            f"{tuple(x.shape)}."
        )
