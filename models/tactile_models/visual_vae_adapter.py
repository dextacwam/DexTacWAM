"""Stage 1 lite: frozen LTX visual VAE + tactile adapter (probe).

This module implements the architecture described in the
`stage1_lite_visual_vae_adapter_probe` plan: keep the LTX visual VAE frozen,
encode each finger's grayscale tactile image through it (after a learnable
1->3 channel adapter), then fuse the per-finger latents into a single
per-hand latent that strictly matches the visual VAE's `(B, 128, T_lat, 6, 8)`
token layout (the C5 contract for Stage 2 DiT plug-in).

Pipeline (per hand)::

    gray tactile (B, 5, 1, T, 192, 256)
        -> GrayToRGB (init = ones-repeat: weight=1, bias=0)        -> RGB
        -> FROZEN AutoencoderKLLTXVideo.encode (latent_dist.mode by default)
        -> per-finger latent (B, 5, 128, T_lat, 6, 8)

    Diagnostic branch (always logged, weighted into loss with `lambda_loc_pre`):
        -> AuxPreFuseFlowHead (per-finger shared-weight upsampler)
        -> flow_pred_pre (B, 5, T_out, 24, 32, 3)

    Main branch:
        -> FingerAttentionAdapter
              .  Q = hand_query + pos_embed[h, w]
              .  K, V = z + finger_embed[5]
              .  base = sum_f softmax(finger_logits)[f] * z[:, f]   (learnable
                      weighted finger mean; finger_logits=0 init -> mean-pool)
              .  out  = base + alpha * attn_out                    (alpha=0 init)
        -> per-hand latent (B, 128, T_lat, 6, 8)
        -> ModalityEmbedding bias
        -> AuxFlowDecoder (post-fuse: 1 hand -> 5 fingers)
        -> AuxPoseDecoder (per-hand 22-dim pose)

The visual VAE is frozen via `requires_grad_(False)` ONLY; encode is NOT
wrapped in `torch.no_grad()` so gradients can still flow back through it to
`GrayToRGB`. The smoke test (`scripts/smoke_visual_vae_adapter.py`) verifies
this invariant.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..ltx_models.autoencoder_kl_ltx import (
    AutoencoderKLLTXVideo,
    LTXVideoCausalConv3d,
    LTXVideoResnetBlock3d,
)
from .tactile_modules import ModalityEmbedding
from .tactile_vae import _SpatialUp2x, _TemporalUp8x

__all__ = [
    "GrayToRGB",
    "FingerAttentionAdapter",
    "ConcatChannelAdapter",
    "FingerSetTransformerAdapter",
    "AuxPreFuseFlowHead",
    "AuxFlowDecoder",
    "AuxPoseDecoder",
    "VisualVAEAdapterModel",
]


# ---------------------------------------------------------------------------
# 1->3 channel grayscale-to-RGB adapter (4-6 params)
# ---------------------------------------------------------------------------


class GrayToRGB(nn.Module):
    """Map a single-channel tactile image to 3 RGB channels.

    Default init (`ones_repeat`) sets ``weight=1.0`` and ``bias=0.0`` so that
    at step 0 the forward is exactly ``gray.repeat_interleave(3, dim=1)`` --
    each RGB channel equals the input grayscale. Crucially this does NOT
    compress luminance by 3x (which the legacy ``uniform_third`` init would).
    The 6 parameters are then free to learn a chromatic remapping that the
    frozen LTX VAE finds informative.

    Accepts both 4-D and 5-D inputs:
        (B, 1, H, W)        -> (B, 3, H, W)
        (B, 1, T, H, W)     -> (B, 3, T, H, W)   (per-frame Conv2d with batch
                                                  flatten; cheaper than 3-D conv)
    """

    def __init__(self, init_mode: str = "ones_repeat"):
        super().__init__()
        if init_mode not in {"ones_repeat", "uniform_third"}:
            raise ValueError(
                f"GrayToRGB init_mode must be 'ones_repeat' or 'uniform_third'; "
                f"got {init_mode!r}."
            )
        self.init_mode = init_mode
        self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            if init_mode == "ones_repeat":
                self.conv.weight.fill_(1.0)
            else:  # uniform_third
                self.conv.weight.fill_(1.0 / 3.0)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.conv(x)
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            if c != 1:
                raise ValueError(
                    f"GrayToRGB expects C=1 on dim 1; got shape {tuple(x.shape)}."
                )
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv(x)
            return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        raise ValueError(
            f"GrayToRGB expects 4-D or 5-D input; got shape {tuple(x.shape)}."
        )


# ---------------------------------------------------------------------------
# Adapter B (default): per-(t, h, w) finger attention
# ---------------------------------------------------------------------------


class FingerAttentionAdapter(nn.Module):
    """Fuse 5 per-finger latents into a single per-hand latent.

    For each spatio-temporal cell ``(t, h, w)``::

        Q = hand_query[1, C] + pos_embed[h, w]                  (broadcast)
        K = V = z[:, :, :, t, h, w] + finger_embed[5]            (B, 5, C)
        attn_out  = MultiheadAttention(Q, K, V).squeeze(1)       (B, C)

        if residual == "weighted_mean":
            base = sum_f softmax(finger_logits)[f] * z[:, f, :, t, h, w]
        else:  # mean_pool
            base = z[:, :, :, t, h, w].mean(dim=1)

        out  = base + alpha * attn_out                  alpha is zero-init so
                                                        the layer starts as a
                                                        pure (weighted) mean
                                                        pool over fingers.

    Shapes::

        Input :  (B, 5, 128, T_lat, 6, 8)
        Output:  (B,    128, T_lat, 6, 8)

    The output spatial layout `(6, 8)` is preserved exactly so the post-adapter
    latent drop-in replaces a single visual VAE view in the Stage 2 DiT. This
    is the C5 contract from the plan.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_fingers: int = 5,
        num_heads: int = 4,
        h: int = 6,
        w: int = 8,
        use_finger_embed: bool = True,
        use_pos_query: bool = True,
        residual: str = "weighted_mean",
    ):
        super().__init__()
        if residual not in {"weighted_mean", "mean_pool"}:
            raise ValueError(
                f"FingerAttentionAdapter residual must be 'weighted_mean' or "
                f"'mean_pool'; got {residual!r}."
            )

        self.embed_dim = embed_dim
        self.num_fingers = num_fingers
        self.num_heads = num_heads
        self.h = h
        self.w = w
        self.use_finger_embed = use_finger_embed
        self.use_pos_query = use_pos_query
        self.residual = residual

        # Global learnable hand query (1, C).
        self.hand_query = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.hand_query, std=0.02)

        # Per-spatial-cell positional bias added to the query at each (h, w).
        if use_pos_query:
            self.pos_embed = nn.Parameter(torch.zeros(h, w, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)

        # Per-finger identity embedding added to K/V so the adapter can tell
        # thumb / index / middle / ring / pinky apart instead of relying on
        # the frozen visual VAE to encode finger identity implicitly.
        if use_finger_embed:
            self.finger_embed = nn.Parameter(torch.zeros(num_fingers, embed_dim))
            nn.init.trunc_normal_(self.finger_embed, std=0.02)
        else:
            self.register_parameter("finger_embed", None)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Learnable softmax-weighted finger mean residual. Zero-init logits
        # -> uniform softmax -> exact mean-pool at step 0.
        if residual == "weighted_mean":
            self.finger_logits = nn.Parameter(torch.zeros(num_fingers))
        else:
            self.register_parameter("finger_logits", None)

        # Zero-init alpha: at step 0 the attention path contributes nothing,
        # so the adapter starts as a pure (weighted) mean over fingers. This
        # is the AdaLN-Zero style residual: stable starting point + the model
        # learns the attention correction during training.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 6:
            raise ValueError(
                f"FingerAttentionAdapter expects (B, F, C, T, H, W); got "
                f"shape {tuple(z.shape)}."
            )
        b, f, c, t, h, w = z.shape
        if f != self.num_fingers:
            raise ValueError(
                f"Expected {self.num_fingers} fingers; got {f}."
            )
        if c != self.embed_dim:
            raise ValueError(
                f"Expected channel dim {self.embed_dim}; got {c}."
            )
        if h != self.h or w != self.w:
            raise ValueError(
                f"Spatial mismatch: input ({h},{w}) vs configured "
                f"({self.h},{self.w})."
            )

        # Token view: each (b, t, h, w) location is one independent attention
        # sample over the 5 fingers; weights are shared across all locations.
        # z: (B, F, C, T, H, W) -> z_kv: (B*T*H*W, F, C)
        z_kv = rearrange(z, "b f c t h w -> (b t h w) f c")

        # Query: (B*T*H*W, 1, C). Add positional bias per (h, w).
        n_loc = b * t * h * w
        if self.use_pos_query:
            # pos_embed: (H, W, C) -> tile across (B, T) -> (B*T*H*W, C)
            pos = self.pos_embed.view(1, h, w, c).expand(b * t, h, w, c)
            pos = rearrange(pos, "n h w c -> (n h w) c")
            q = self.hand_query.expand(n_loc, c) + pos
        else:
            q = self.hand_query.expand(n_loc, c)
        q = q.unsqueeze(1)  # (N, 1, C)

        # Add finger identity embedding to K/V.
        if self.use_finger_embed:
            kv = z_kv + self.finger_embed.view(1, f, c)
        else:
            kv = z_kv

        # Pre-attention norms (LayerNorm-Pre style; helps stability with
        # identity-style residual). Independent norms for Q and K/V.
        q_n = self.q_norm(q)
        kv_n = self.kv_norm(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        attn_out = attn_out.squeeze(1)  # (N, C)

        # Residual base: (weighted) mean over fingers.
        if self.residual == "weighted_mean":
            w_finger = F.softmax(self.finger_logits, dim=0)  # (F,)
            base = (z_kv * w_finger.view(1, f, 1)).sum(dim=1)  # (N, C)
        else:
            base = z_kv.mean(dim=1)

        out = base + self.alpha * attn_out
        out = rearrange(
            out, "(b t h w) c -> b c t h w", b=b, t=t, h=h, w=w,
        )
        return out


# ---------------------------------------------------------------------------
# Adapter A (ablation): naive concat-channel Conv3d
# ---------------------------------------------------------------------------


class ConcatChannelAdapter(nn.Module):
    """Cheapest fusion baseline: concat the 5 finger latents along channels
    and reduce with a 1x1x1 Conv3d.

    Used as an ablation only; the default in v0 is :class:`FingerAttentionAdapter`.

    Init: weights randomized (Conv3d default), bias zero. Not initialized to
    mean-pool because we want this baseline to compete on its own footing.

    Shapes::

        Input :  (B, F, C, T, H, W)
        Output:  (B,    C, T, H, W)
    """

    def __init__(self, embed_dim: int = 128, num_fingers: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fingers = num_fingers
        self.conv = nn.Conv3d(num_fingers * embed_dim, embed_dim, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 6:
            raise ValueError(
                f"ConcatChannelAdapter expects 6-D input; got shape "
                f"{tuple(z.shape)}."
            )
        b, f, c, t, h, w = z.shape
        if f != self.num_fingers or c != self.embed_dim:
            raise ValueError(
                f"Shape mismatch: F,C=({f},{c}) vs "
                f"({self.num_fingers},{self.embed_dim})."
            )
        x = rearrange(z, "b f c t h w -> b (f c) t h w")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Adapter v2 (v0c-A): deeper transformer set-encoder over [hand_query + 5 fingers]
# ---------------------------------------------------------------------------


class _FingerSetEncoderBlock(nn.Module):
    """Pre-LN transformer encoder block over a 6-token sequence.

    Layout::

        x  -> LN -> MHA(self-attn) -> + residual
        x  -> LN -> Linear(C, FFN) -> GELU -> Linear(FFN, C) -> + residual

    The input sequence is ``[hand_query, finger_0, ..., finger_4]`` with shape
    ``(N, 6, C)`` where ``N = B*T*H*W`` (one independent set per spatio-temporal
    cell).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T_tok, C) -- T_tok = 1 + num_fingers (= 6 for the v0c default).
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# v0d additions: pose encoder + TimeSformer post-adapter helpers
# ---------------------------------------------------------------------------
#
# These are the building blocks for v0d's two upgrades over v0c-A:
#
#   1. _PoseEncoder maps per-frame hand pose `(B, T_lat, 22)` -> per-frame
#      pose embedding `(B, T_lat, 128)` for additive injection into
#      `FingerSetTransformerAdapter.hand_query` (Option D in the v0d plan).
#
#   2. _TimeSformerBlock applies divided space-time attention over the
#      adapter's `(B, T_lat, H*W, 128)` per-cell hand-out tokens for
#      temporal context refinement (Path 2 in the v0d plan). Inspired by
#      TouchAnything's TemporalTransformer but operating at our compressed
#      latent resolution (T_lat in {1, 2}, S = 6*8 = 48) instead of
#      TouchAnything's raw-frame resolution.
#
# Both modules are defined here but NOT yet wired into
# `FingerSetTransformerAdapter` -- that wiring lives in the `adapter-class`
# / `adapter-forward` todos. At wiring time both modules become alpha-gated
# submodules whose alpha is zero-initialized, so v0d at step 0 is
# bit-identical to v0c-A and warm-start from v0c-A `step_30000` is clean.


class _PoseEncoder(nn.Module):
    """Per-frame hand-pose encoder: ``(B, T, P)`` -> ``(B, T, C)``.

    A 2-layer MLP with LayerNorm + GELU. Used by
    :class:`FingerSetTransformerAdapter` (after the `adapter-class` todo)
    to enrich ``hand_query`` with pose information via additive bias::

        q_enriched = hand_query + alpha_pose * _PoseEncoder(hand_pose)

    At v0d step 0 the residual is gated by ``alpha_pose=0``, so this
    module's output is unused and v0d behaves exactly as v0c-A.

    Param count at the v0d default (``pose_dim=22, embed_dim=128,
    hidden_dim=128``)::

        Linear(22 -> 128) + LayerNorm(128) + GELU + Dropout(0)
        + Linear(128 -> 128) + LayerNorm(128)
        ~= 20 K params

    Args:
        pose_dim: input pose dimensionality. For DexVTAM Stage 1 lite this
            is 22 (one hand's finger joint positions, a slice of the 58-d
            LeRobot state vector; see ``data/tactile_dataset.py``
            ``_LEFT_HAND_SLICE`` / ``_RIGHT_HAND_SLICE``).
        embed_dim: output embedding dim. Must equal the adapter's
            ``embed_dim`` (= 128 for the C5 contract) so the additive
            injection into ``hand_query`` is shape-compatible.
        hidden_dim: MLP hidden width. Default 128 (= embed_dim) keeps the
            module compact; raising it grows params by ~ ``pose_dim *
            delta + delta``.
        dropout: applied between the two Linears. Default 0.0 to mirror
            the v0c-A adapter's dropout-free baseline.
    """

    def __init__(
        self,
        pose_dim: int = 22,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 3:
            raise ValueError(
                f"_PoseEncoder expects 3-D input (B, T, P); got shape "
                f"{tuple(pose.shape)}."
            )
        if pose.shape[-1] != self.pose_dim:
            raise ValueError(
                f"_PoseEncoder expects last dim = {self.pose_dim}; got "
                f"{pose.shape[-1]}."
            )
        return self.encoder(pose)


class _TimeSformerBlock(nn.Module):
    """Single divided space-time attention block (TimeSformer-style).

    Operates on a token grid ``(B, T, S, C)`` where ``S = H*W`` is the
    spatial axis flattened. Each forward pass does (pre-LN)::

        x = x + temporal_attn(LN_t(x))   # MHA over T (per spatial position)
        x = x + spatial_attn(LN_s(x))    # MHA over S (per temporal position)
        x = x + ffn(LN_ffn(x))           # 4x FFN expansion

    Used (after the `adapter-class` todo) by
    :class:`FingerSetTransformerAdapter` as a post-adapter refinement
    block, alpha-gated::

        hand_out = hand_out + alpha_temp * _TimeSformerBlock(hand_out_grid)

    At v0d step 0 ``alpha_temp=0`` so this block's output is unused and
    v0d behaves exactly as v0c-A.

    Param count at the v0d default (``embed_dim=128, num_heads=8,
    ffn_dim=512, dropout=0``)::

        temporal_attn (MultiheadAttention, embed=128):  ~66 K
            in_proj  3*(128*128) + 3*128                = 49,536
            out_proj 128*128 + 128                      = 16,512
        spatial_attn  same shape:                       ~66 K
        ffn  (Linear 128->512 + Linear 512->128):       ~131 K
        3 LayerNorms (128 each, weight + bias):         ~ 0.8 K
        => ~ 264 K params per block, ~ 528 K for 2 blocks.

    (The v0d plan's "~ 1.0 M for 2 blocks" was a loose upper bound; actual
    is roughly half. This is in our favor: fewer params => faster
    warm-start convergence.)

    Inspired by TouchAnything's ``TemporalTransformer`` but operating at
    our compressed latent resolution (T in {1, 2}, S = 6*8 = 48) instead
    of raw-frame resolution (T = 30+, S = 256+).

    Args:
        embed_dim: token channel dim. Must equal the adapter's embed_dim
            (= 128 for the C5 contract).
        num_heads: MHA head count. Requires ``embed_dim % num_heads == 0``.
        ffn_dim: FFN hidden width. Default 512 = 4 * embed_dim, mirroring
            the standard transformer-block expansion ratio.
        dropout: applied inside both MHAs and inside the FFN. Default 0.0
            to mirror the v0c-A adapter's dropout-free baseline.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"_TimeSformerBlock: embed_dim ({embed_dim}) must be "
                f"divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm_t = nn.LayerNorm(embed_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_s = nn.LayerNorm(embed_dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"_TimeSformerBlock expects 4-D input (B, T, S, C); got "
                f"shape {tuple(x.shape)}."
            )
        b, t, s, c = x.shape
        if c != self.embed_dim:
            raise ValueError(
                f"_TimeSformerBlock channel mismatch: input C={c} vs "
                f"configured embed_dim={self.embed_dim}."
            )

        # Temporal attention: every spatial position attends across time.
        # (B, T, S, C) -> (B*S, T, C) so MHA's batch dim covers (B, S).
        # T=1 is a degenerate single-token attention (still valid; the
        # residual just learns identity for that batch).
        x_t_in = self.norm_t(x)
        x_t_in = rearrange(x_t_in, "b t s c -> (b s) t c")
        attn_t, _ = self.temporal_attn(x_t_in, x_t_in, x_t_in, need_weights=False)
        attn_t = rearrange(attn_t, "(b s) t c -> b t s c", b=b, s=s)
        x = x + attn_t

        # Spatial attention: every time position attends across spatial.
        # (B, T, S, C) -> (B*T, S, C).
        x_s_in = self.norm_s(x)
        x_s_in = rearrange(x_s_in, "b t s c -> (b t) s c")
        attn_s, _ = self.spatial_attn(x_s_in, x_s_in, x_s_in, need_weights=False)
        attn_s = rearrange(attn_s, "(b t) s c -> b t s c", b=b, t=t)
        x = x + attn_s

        # FFN over (B, T, S, C); LayerNorm + Linear apply per-token by
        # default so no further reshape is needed.
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FingerSetTransformerAdapter(nn.Module):
    """Deeper adapter (v0c-A): stack of transformer encoder blocks over the
    set ``[hand_query, finger_0, ..., finger_4]`` per spatio-temporal cell.

    Same I/O contract as :class:`FingerAttentionAdapter` (the C5 contract for
    Stage 2 DiT plug-in is preserved):

    Shapes::

        Input :  (B, 5, 128, T_lat, 6, 8)
        Output: (B,    128, T_lat, 6, 8)

    Per spatio-temporal cell ``(t, h, w)``::

        tokens    = [hand_query + pos_embed[h, w]]                  (1 token)
                    + [z[:, f, :, t, h, w] + finger_embed[f]
                       for f in range(num_fingers)]                 (5 tokens)
        tokens   -> N transformer encoder blocks (pre-LN, MHA + FFN)
        hand_out  = LN(tokens)[..., 0, :]                           (B, C)

        if residual == 'weighted_mean':
            base = sum_f softmax(finger_logits)[f] * z[:, f, ..., t, h, w]
        else:  # mean_pool
            base = z[:, :, :, t, h, w].mean(dim=1)

        out      = base + alpha * hand_out                          alpha = 0 init
                                                                    -> at step 0 the
                                                                    transformer path
                                                                    contributes 0 and
                                                                    the adapter is
                                                                    exactly the v1
                                                                    weighted-mean
                                                                    baseline (the
                                                                    smoke test
                                                                    asserts this).

    Params (default 3 layers / 8 heads / FFN 1024 at C=128):
        ~ 1.6 M (vs. 70 K for the v0 :class:`FingerAttentionAdapter`); still
        << the frozen LTX VAE backbone (~500 M). Designed to test whether the
        v0_full step-30k gap=0.29 is an adapter-capacity bottleneck.

    The ``hand_query``, ``pos_embed`` and ``finger_embed`` parameters and the
    learnable ``finger_logits`` for the weighted-mean residual mirror v1
    exactly so the adapter starts at the same point in function space; the only
    new capacity comes from the stacked transformer blocks gated by ``alpha``.

    v0d additions (defaults reduce to v0c-A so this class is a pure superset)::

        use_pose_injection : bool = False
            If True, allocate a :class:`_PoseEncoder` plus an ``alpha_pose``
            scalar parameter (zero-init). Forward wiring is deferred to a
            separate todo (`adapter-forward`); until that wiring lands the
            new submodules are unused and v0c-A behavior is byte-identical.
        use_timesformer    : bool = False
            If True, allocate ``timesformer_num_blocks`` instances of
            :class:`_TimeSformerBlock` plus an ``alpha_temp`` scalar
            parameter (zero-init). Same scope discipline as above.
        finger_dropout     : float in [0, 1) = 0.0
            If > 0, allocate a learnable ``mask_token`` for replacing
            randomly-dropped finger tokens during training. Forward
            wiring is deferred to the same later todo.

    All v0d residual paths are alpha-zero-init (``alpha_pose=0``,
    ``alpha_temp=0``); v0c-A's ``alpha`` is also zero-init. So even after
    the forward wiring lands, v0d at step 0 with all flags ON is still
    functionally identical to v0c-A at step 0. Warm-starting v0d from a
    v0c-A checkpoint with ``strict=False`` loads cleanly: only the new
    ``pose_encoder.*``, ``timesformer_blocks.*``, ``alpha_pose``,
    ``alpha_temp`` and ``mask_token`` keys are missing, and all of them
    are initialized to a zero-residual / zero-effect configuration.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_fingers: int = 5,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        h: int = 6,
        w: int = 8,
        use_finger_embed: bool = True,
        use_pos_query: bool = True,
        residual: str = "weighted_mean",
        # ----- v0d additions (defaults reduce to v0c-A) -----------------
        use_pose_injection: bool = False,
        pose_dim: int = 22,
        use_timesformer: bool = False,
        timesformer_num_blocks: int = 2,
        timesformer_num_heads: int = 8,
        timesformer_ffn_dim: int = 512,
        finger_dropout: float = 0.0,
    ):
        super().__init__()
        if residual not in {"weighted_mean", "mean_pool"}:
            raise ValueError(
                f"FingerSetTransformerAdapter residual must be 'weighted_mean' "
                f"or 'mean_pool'; got {residual!r}."
            )
        if num_layers < 1:
            raise ValueError(
                f"FingerSetTransformerAdapter num_layers must be >= 1; got "
                f"{num_layers}."
            )
        # v0d-side validation. Cheap to check unconditionally.
        if pose_dim < 1:
            raise ValueError(
                f"FingerSetTransformerAdapter pose_dim must be >= 1; got "
                f"{pose_dim}."
            )
        if timesformer_num_blocks < 0:
            raise ValueError(
                f"FingerSetTransformerAdapter timesformer_num_blocks must be "
                f">= 0; got {timesformer_num_blocks}."
            )
        if use_timesformer and timesformer_num_blocks == 0:
            raise ValueError(
                "FingerSetTransformerAdapter: use_timesformer=True requires "
                "timesformer_num_blocks >= 1; set use_timesformer=False to "
                "disable the post-adapter, or pass num_blocks >= 1."
            )
        if not (0.0 <= finger_dropout < 1.0):
            raise ValueError(
                f"FingerSetTransformerAdapter finger_dropout must be in "
                f"[0, 1); got {finger_dropout}."
            )

        self.embed_dim = embed_dim
        self.num_fingers = num_fingers
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.h = h
        self.w = w
        self.use_finger_embed = use_finger_embed
        self.use_pos_query = use_pos_query
        self.residual = residual
        # v0d flags / hyperparams, stored verbatim for introspection and so
        # forward (added in `adapter-forward` todo) can branch on them
        # without re-reading any external config object.
        self.use_pose_injection = use_pose_injection
        self.pose_dim = pose_dim
        self.use_timesformer = use_timesformer
        self.timesformer_num_blocks = timesformer_num_blocks
        self.timesformer_num_heads = timesformer_num_heads
        self.timesformer_ffn_dim = timesformer_ffn_dim
        self.finger_dropout = finger_dropout

        self.hand_query = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.hand_query, std=0.02)

        if use_pos_query:
            self.pos_embed = nn.Parameter(torch.zeros(h, w, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)

        if use_finger_embed:
            self.finger_embed = nn.Parameter(torch.zeros(num_fingers, embed_dim))
            nn.init.trunc_normal_(self.finger_embed, std=0.02)
        else:
            self.register_parameter("finger_embed", None)

        self.blocks = nn.ModuleList([
            _FingerSetEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)

        if residual == "weighted_mean":
            self.finger_logits = nn.Parameter(torch.zeros(num_fingers))
        else:
            self.register_parameter("finger_logits", None)

        # Zero-init alpha: at step 0 the transformer path contributes nothing,
        # so the adapter starts as a pure (weighted) mean over fingers --
        # identical function to v1 at step 0. The smoke test verifies this.
        self.alpha = nn.Parameter(torch.zeros(1))

        # ----- v0d submodules ------------------------------------------
        # Each branch is wrapped so that with the default flag values (all
        # off) the module namespace and state_dict are byte-identical to
        # v0c-A. The `None` registration trick mirrors the existing v0c-A
        # pattern (see `pos_embed`, `finger_embed`, `finger_logits` above):
        # `nn.Module` slots use `add_module(name, None)` and `nn.Parameter`
        # slots use `register_parameter(name, None)`, which keep the
        # attribute accessible (returns None) but skip it from state_dict
        # so a flags-off v0d ckpt is bit-equal to a v0c-A ckpt.
        #
        # Forward is left untouched in this todo: these submodules are
        # allocated but unused. The next todo (`adapter-forward`) wires
        # them in behind alpha-zero residual gates so v0d at step 0 with
        # all flags ON is still functionally identical to v0c-A at step 0.
        if use_pose_injection:
            self.pose_encoder = _PoseEncoder(
                pose_dim=pose_dim,
                embed_dim=embed_dim,
                hidden_dim=embed_dim,
                dropout=0.0,
            )
            # alpha_pose: zero-init residual gate for additive injection
            #   q_enriched = hand_query + alpha_pose * pose_encoder(pose)
            self.alpha_pose = nn.Parameter(torch.zeros(1))
        else:
            self.add_module("pose_encoder", None)
            self.register_parameter("alpha_pose", None)

        if use_timesformer:
            self.timesformer_blocks = nn.ModuleList([
                _TimeSformerBlock(
                    embed_dim=embed_dim,
                    num_heads=timesformer_num_heads,
                    ffn_dim=timesformer_ffn_dim,
                    dropout=0.0,
                )
                for _ in range(timesformer_num_blocks)
            ])
            # alpha_temp: zero-init residual gate for the post-adapter
            # refinement
            #   hand_grid = hand_grid + alpha_temp * timesformer_stack(hand_grid)
            self.alpha_temp = nn.Parameter(torch.zeros(1))
        else:
            self.add_module("timesformer_blocks", None)
            self.register_parameter("alpha_temp", None)

        if finger_dropout > 0.0:
            # Learnable replacement token for randomly-dropped finger tokens
            # during training (Bernoulli per spatio-temporal cell). Inference
            # path (later todo) keeps all 5 fingers, so this token is only
            # used when `self.training and finger_dropout > 0`.
            self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            self.register_parameter("mask_token", None)

    def forward(
        self,
        z: torch.Tensor,
        hand_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            z: per-finger latent grid, shape ``(B, F, C, T, H, W)``.
            hand_pose: per-frame hand pose, shape ``(B, T, P)``. Required
                when ``use_pose_injection=True``; silently ignored
                otherwise (so callers that don't carry pose still work
                with v0c-A-style adapters).

        Returns:
            Per-hand latent grid, shape ``(B, C, T, H, W)``. The C5 contract
            for Stage 2 DiT plug-in is preserved.

        v0d additions, all alpha-zero-gated so step 0 = v0c-A step 0:

            * pose injection (Option D): adds
              ``alpha_pose * PoseEncoder(hand_pose)`` (tiled across H, W)
              to ``hand_query`` before the set-transformer.
            * finger dropout: at training time, randomly replaces finger
              tokens in the KV stream with the learnable ``mask_token``
              (Bernoulli per spatio-temporal cell). The residual baseline
              path keeps the original (un-dropped) finger latents so the
              weighted-mean shortcut does not leak ``mask_token`` into the
              output.
            * TimeSformer post-block (Path 2): refines the adapter's
              ``(B, T, S, C)`` token grid with divided space-time
              attention and adds ``alpha_temp * (refined - out)`` as a
              residual.
        """
        if z.ndim != 6:
            raise ValueError(
                f"FingerSetTransformerAdapter expects (B, F, C, T, H, W); got "
                f"shape {tuple(z.shape)}."
            )
        b, f, c, t, h, w = z.shape
        if f != self.num_fingers:
            raise ValueError(
                f"Expected {self.num_fingers} fingers; got {f}."
            )
        if c != self.embed_dim:
            raise ValueError(
                f"Expected channel dim {self.embed_dim}; got {c}."
            )
        if h != self.h or w != self.w:
            raise ValueError(
                f"Spatial mismatch: input ({h},{w}) vs configured "
                f"({self.h},{self.w})."
            )

        # ----- v0d hand_pose contract validation -----
        if self.use_pose_injection:
            if hand_pose is None:
                raise ValueError(
                    "FingerSetTransformerAdapter.forward: hand_pose is "
                    "required when use_pose_injection=True; got None."
                )
            if hand_pose.ndim != 3:
                raise ValueError(
                    f"hand_pose must be 3-D (B, T_lat, P); got shape "
                    f"{tuple(hand_pose.shape)}."
                )
            if (
                hand_pose.shape[0] != b
                or hand_pose.shape[1] != t
                or hand_pose.shape[2] != self.pose_dim
            ):
                raise ValueError(
                    f"hand_pose shape mismatch: expected "
                    f"(B={b}, T_lat={t}, P={self.pose_dim}); got "
                    f"{tuple(hand_pose.shape)}."
                )
        # When use_pose_injection=False we silently ignore hand_pose so
        # the v0c-A call site `adapter(z)` still works unchanged.

        # Token view: each (b, t, h, w) location is one independent set of
        # 6 tokens (hand-query + 5 fingers). Weights are shared across all
        # locations.
        z_kv = rearrange(z, "b f c t h w -> (b t h w) f c")
        n_loc = z_kv.shape[0]

        # ----- v0d finger dropout (training only, KV path only) -----
        # Replace dropped finger tokens at the z_kv level (BEFORE finger_embed
        # is added) so the learnable mask_token represents "value missing"
        # while finger_embed retains positional identity for the dropped slot.
        # The residual base path below uses the ORIGINAL z_kv -- so the
        # weighted-mean shortcut never sees mask_token. This means dropout
        # only perturbs the transformer's view of the input, not the
        # alpha-gated baseline.
        if (
            self.training
            and self.finger_dropout > 0.0
            and self.mask_token is not None
        ):
            mask = torch.empty(
                b, t, h, w, f, device=z.device, dtype=torch.bool,
            )
            mask.bernoulli_(self.finger_dropout)
            mask = rearrange(mask, "b t h w f -> (b t h w) f")
            mask_token_exp = self.mask_token.to(dtype=z_kv.dtype).expand(
                n_loc, f, c,
            )
            z_kv_attn = torch.where(mask.unsqueeze(-1), mask_token_exp, z_kv)
        else:
            z_kv_attn = z_kv  # alias, no copy

        if self.use_finger_embed:
            kv = z_kv_attn + self.finger_embed.view(1, f, c)
        else:
            kv = z_kv_attn

        if self.use_pos_query:
            # pos_embed: (H, W, C) -> tile across (B, T) -> (B*T*H*W, C)
            pos = self.pos_embed.view(1, h, w, c).expand(b * t, h, w, c)
            pos = rearrange(pos, "n h w c -> (n h w) c")
            q = self.hand_query.expand(n_loc, c) + pos
        else:
            q = self.hand_query.expand(n_loc, c)

        # ----- v0d pose injection (additive bias on hand_query) -----
        # alpha_pose=0 -> exactly zero contribution -> bit-equal to v0c-A.
        # Even though pose_encoder is run unconditionally when the flag is
        # on (so its forward FLOPs and gradient-graph are wired), the
        # `0 * pose_emb` term collapses to a finite zero on each element.
        # alpha_pose itself receives a non-trivial gradient (proportional
        # to dloss/dq . pose_emb), so it can move off zero on step 1 and
        # then unblock pose_encoder's own learning -- this is the standard
        # alpha-residual training dynamic, mirroring v0c-A's `alpha`.
        if self.use_pose_injection and self.pose_encoder is not None:
            pose_emb = self.pose_encoder(hand_pose)  # (B, T_lat, C)
            pose_emb = pose_emb.view(b, t, 1, 1, self.embed_dim).expand(
                b, t, h, w, self.embed_dim,
            )
            pose_emb = rearrange(pose_emb, "b t h w c -> (b t h w) c")
            q = q + self.alpha_pose * pose_emb

        q = q.unsqueeze(1)  # (N, 1, C)

        # Sequence of 6 tokens: [hand_query, finger_0, ..., finger_4].
        tokens = torch.cat([q, kv], dim=1)  # (N, 1 + F, C)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.final_ln(tokens)
        hand_out = tokens[:, 0, :]  # (N, C) -- read the hand-query slot.

        # Residual base: identical to v1. Uses the ORIGINAL z_kv (never the
        # dropout-perturbed z_kv_attn) so the weighted-mean shortcut path
        # is invariant to finger_dropout -- only the transformer (alpha-
        # gated) path sees the mask_token.
        if self.residual == "weighted_mean":
            w_finger = F.softmax(self.finger_logits, dim=0)  # (F,)
            base = (z_kv * w_finger.view(1, f, 1)).sum(dim=1)  # (N, C)
        else:
            base = z_kv.mean(dim=1)

        out = base + self.alpha * hand_out
        out = rearrange(
            out, "(b t h w) c -> b c t h w", b=b, t=t, h=h, w=w,
        )

        # ----- v0d TimeSformer post-block (alpha-gated residual) -----
        # Same alpha-zero-init story as pose injection: at step 0
        # alpha_temp=0 collapses the residual to bit-equal v0c-A. The
        # `(refined - out)` form (rather than just `refined`) is what
        # gives that exact zero-residual semantics on every element,
        # since `out + 0 * (refined - out) == out` in fp32 for finite
        # tensors regardless of `refined`'s value.
        if self.use_timesformer and self.timesformer_blocks is not None:
            # (B, C, T, H, W) -> (B, T, H*W, C)
            refined = rearrange(out, "b c t h w -> b t (h w) c")
            for block in self.timesformer_blocks:
                refined = block(refined)
            refined = rearrange(
                refined, "b t (h w) c -> b c t h w", h=h, w=w,
            )
            out = out + self.alpha_temp * (refined - out)

        return out


# ---------------------------------------------------------------------------
# Auxiliary heads
# ---------------------------------------------------------------------------


class _PerFingerUpsampleStack(nn.Module):
    """Shared-weight per-finger upsampler used by both pre- and post-fuse heads.

    Per-finger pipeline (same as TactileFlowDecoder but starting from the
    visual VAE's spatial resolution `(6, 8)` instead of `(3, 4)`)::

        x (B, C_lat, T_lat, 6, 8)
            -> Conv3d(C_lat -> C_dec, k=1)         channel reduction
        (B, C_dec, T_lat, 6, 8)
            -> 2 x _SpatialUp2x                    2 x 2 spatial = 4 x
        (B, C_dec, T_lat, 24, 32)
            -> _TemporalUp8x                       8 x temporal
        (B, C_dec, T_out, 24, 32)
            -> ResNet -> Conv3d(C_dec -> C_out, k=3)
        (B, C_out, T_out, 24, 32)

    With ``T_out = 1 + 8 * (T_lat - 1)`` so T_lat=1 -> T_out=1 and
    T_lat=2 -> T_out=9, matching the v3/v4 dataset's flow temporal resolution.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        decoder_channels: int = 64,
        out_channels: int = 3,
        is_causal: bool = True,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels

        self.conv_in = LTXVideoCausalConv3d(
            in_channels=latent_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            stride=1,
            is_causal=is_causal,
        )
        self.spatial_up_1 = _SpatialUp2x(decoder_channels, is_causal=is_causal)
        self.spatial_up_2 = _SpatialUp2x(decoder_channels, is_causal=is_causal)
        self.temporal_up = _TemporalUp8x(decoder_channels, is_causal=is_causal)
        self.resnet_out = LTXVideoResnetBlock3d(
            in_channels=decoder_channels,
            out_channels=decoder_channels,
            is_causal=is_causal,
        )
        self.conv_out = LTXVideoCausalConv3d(
            in_channels=decoder_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_lat, T_lat, H, W) -- caller flattens whatever batch axis.
        x = self.conv_in(x)
        x = self.spatial_up_1(x)
        x = self.spatial_up_2(x)
        x = self.temporal_up(x)
        x = self.resnet_out(x)
        x = self.conv_out(x)
        return x


class AuxPreFuseFlowHead(nn.Module):
    """Per-finger pre-fusion flow head (diagnostic).

    Decodes flow directly from the per-finger latent BEFORE the adapter, so
    the post-vs-pre comparison attributes failure to either the encoder
    (both bad) or the adapter (pre good, post bad).

    Shapes::

        Input :  (B, F, C_lat, T_lat, H, W)
        Output:  (B, F, T_out, 4*H, 4*W, C_out)   (channel-last to match GT)

    Implementation: shared per-finger upsampler via batch flatten -- so the
    pre-fuse head has a constant parameter count regardless of `F`.
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
        self.num_fingers = num_fingers
        self.out_channels = out_channels
        self.stack = _PerFingerUpsampleStack(
            latent_channels=latent_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            is_causal=is_causal,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 6:
            raise ValueError(
                f"AuxPreFuseFlowHead expects 6-D input; got shape "
                f"{tuple(z.shape)}."
            )
        b, f, c, t_lat, h, w = z.shape
        if f != self.num_fingers:
            raise ValueError(
                f"Expected {self.num_fingers} fingers; got {f}."
            )
        x = rearrange(z, "b f c t h w -> (b f) c t h w")
        x = self.stack(x)
        flow = rearrange(x, "(b f) c t h w -> b f t h w c", b=b, f=f)
        return flow


class AuxFlowDecoder(nn.Module):
    """Post-fusion flow head: 1 hand latent -> 5 fingers of flow.

    Shapes::

        Input :  (B, C_lat, T_lat, H, W)               per-hand fused latent
        Output:  (B, F, T_out, 4*H, 4*W, C_out)        per-finger flow

    Implementation: same upsampling stack as the pre-fuse head, but the
    final conv emits ``F * C_out`` channels which are reshaped into 5
    finger predictions. The 5 fingers thus share spatial/temporal feature
    extraction up to the very last 3x3x3 conv, which expands them.
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
        self.num_fingers = num_fingers
        self.out_channels = out_channels

        self.conv_in = LTXVideoCausalConv3d(
            in_channels=latent_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            stride=1,
            is_causal=is_causal,
        )
        self.spatial_up_1 = _SpatialUp2x(decoder_channels, is_causal=is_causal)
        self.spatial_up_2 = _SpatialUp2x(decoder_channels, is_causal=is_causal)
        self.temporal_up = _TemporalUp8x(decoder_channels, is_causal=is_causal)
        self.resnet_out = LTXVideoResnetBlock3d(
            in_channels=decoder_channels,
            out_channels=decoder_channels,
            is_causal=is_causal,
        )
        self.conv_out = LTXVideoCausalConv3d(
            in_channels=decoder_channels,
            out_channels=num_fingers * out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(
                f"AuxFlowDecoder expects 5-D input (B, C, T, H, W); got "
                f"shape {tuple(z.shape)}."
            )
        x = self.conv_in(z)
        x = self.spatial_up_1(x)
        x = self.spatial_up_2(x)
        x = self.temporal_up(x)
        x = self.resnet_out(x)
        x = self.conv_out(x)  # (B, F*C_out, T_out, 4H, 4W)
        flow = rearrange(
            x,
            "b (f c) t h w -> b f t h w c",
            f=self.num_fingers,
            c=self.out_channels,
        )
        return flow


class AuxPoseDecoder(nn.Module):
    """Pool the per-hand latent over (T, H, W) and regress to a 22-dim pose.

    Mirrors the v3/v4 ``TactilePoseDecoder`` API but operates on the post-
    fusion per-hand latent (no finger axis; channel pool only).
    """

    def __init__(
        self,
        latent_channels: int = 128,
        hidden_dim: int = 256,
        pose_dim: int = 22,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(
                f"AuxPoseDecoder expects 5-D input (B, C, T, H, W); got "
                f"shape {tuple(z.shape)}."
            )
        pooled = z.mean(dim=(2, 3, 4))  # (B, C)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Wrapper: VAE + GrayToRGB + adapter + aux heads + modality bias
# ---------------------------------------------------------------------------


class VisualVAEAdapterModel(nn.Module):
    """End-to-end Stage 1 lite probe model.

    Everything except the LTX visual VAE encoder is trainable. The VAE is
    frozen via ``requires_grad_(False)`` and ``eval()`` once at construction;
    its decoder is unused (we never call ``vae.decode``) but kept on the
    module so that the safetensors checkpoint loads cleanly.

    Critical correctness invariant: ``forward`` does NOT wrap ``vae.encode``
    in ``torch.no_grad()``. The VAE parameters have ``requires_grad=False`` so
    they are not updated, but the autograd graph is preserved so gradients
    flow back through the VAE to ``GrayToRGB``. The smoke test
    ``scripts/smoke_visual_vae_adapter.py`` asserts this invariant by checking
    that ``gray_to_rgb.weight.grad`` is non-None and non-zero after
    ``loss.backward()`` while every ``vae.parameters()`` entry has
    ``grad is None``.
    """

    def __init__(
        self,
        vae: AutoencoderKLLTXVideo,
        latent_channels: int = 128,
        adapter_kind: str = "finger_attention",
        num_fingers: int = 5,
        num_heads: int = 4,
        spatial_h: int = 6,
        spatial_w: int = 8,
        use_finger_embed: bool = True,
        use_pos_query: bool = True,
        adapter_residual: str = "weighted_mean",
        gray_to_rgb_init: str = "ones_repeat",
        latent_mode: str = "mean",
        enable_pre_fuse: bool = True,
        # ---- v0c-A (FingerSetTransformerAdapter) -----------------------
        # Ignored for adapter_kind in {"finger_attention", "concat_channel"}.
        adapter_n_layers: int = 3,
        adapter_ffn_dim: int = 1024,
        adapter_dropout: float = 0.0,
        # ---- v0d (only consumed by adapter_kind="finger_set_transformer") --
        # All defaults reduce v0d to v0c-A; passing a non-default value with
        # a non-finger_set_transformer adapter_kind raises ValueError to fail
        # loud rather than silently dropping the kwarg.
        adapter_use_pose_injection: bool = False,
        adapter_pose_dim: int = 22,
        adapter_use_timesformer: bool = False,
        adapter_timesformer_num_blocks: int = 2,
        adapter_timesformer_num_heads: int = 8,
        adapter_timesformer_ffn_dim: int = 512,
        adapter_finger_dropout: float = 0.0,
    ):
        super().__init__()

        if latent_mode not in {"mean", "sample"}:
            raise ValueError(
                f"latent_mode must be 'mean' or 'sample'; got {latent_mode!r}."
            )

        # v0d kwargs only make sense for the finger_set_transformer adapter.
        # If a non-default v0d value is passed with another adapter_kind we
        # raise loudly instead of silently dropping it -- silent drop is the
        # main footgun this gate is here to prevent.
        v0d_active = (
            adapter_use_pose_injection
            or adapter_use_timesformer
            or adapter_finger_dropout > 0.0
        )
        if v0d_active and adapter_kind != "finger_set_transformer":
            raise ValueError(
                f"v0d adapter kwargs (use_pose_injection / use_timesformer / "
                f"finger_dropout) are only valid for "
                f"adapter_kind='finger_set_transformer'; got "
                f"adapter_kind={adapter_kind!r}."
            )

        self.vae = vae
        self.latent_channels = latent_channels
        self.num_fingers = num_fingers
        self.latent_mode = latent_mode
        self.enable_pre_fuse = enable_pre_fuse
        # Cache the v0d pose-injection flag so callers (trainer, smoke) and
        # internal sub-stages can branch on it without poking into the
        # adapter's namespace.
        self.use_pose_injection = bool(adapter_use_pose_injection)
        self.pose_dim = int(adapter_pose_dim)
        # Latch for the one-time "pose-broadcast misconfiguration" warning
        # emitted by fuse_per_hand. Lives on the model instance (not a
        # module-level global) so multi-model test harnesses each get their
        # own warning, while a single training run only logs the warning
        # once even though _compute_losses is called every batch.
        self._pose_broadcast_warned: bool = False

        # Freeze VAE: requires_grad=False blocks updates; we deliberately do
        # NOT use torch.no_grad() in forward so the autograd graph stays
        # alive and gradients can reach `GrayToRGB`.
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.gray_to_rgb = GrayToRGB(init_mode=gray_to_rgb_init)

        if adapter_kind == "finger_attention":
            self.adapter = FingerAttentionAdapter(
                embed_dim=latent_channels,
                num_fingers=num_fingers,
                num_heads=num_heads,
                h=spatial_h,
                w=spatial_w,
                use_finger_embed=use_finger_embed,
                use_pos_query=use_pos_query,
                residual=adapter_residual,
            )
        elif adapter_kind == "concat_channel":
            self.adapter = ConcatChannelAdapter(
                embed_dim=latent_channels, num_fingers=num_fingers,
            )
        elif adapter_kind == "finger_set_transformer":
            # v0c-A / v0d: deeper transformer set-encoder. Same I/O contract
            # as FingerAttentionAdapter (preserves the C5 contract for Stage
            # 2). v0d additions are alpha-zero-gated so step 0 with all flags
            # ON is bit-identical to v0c-A.
            self.adapter = FingerSetTransformerAdapter(
                embed_dim=latent_channels,
                num_fingers=num_fingers,
                num_heads=num_heads,
                num_layers=adapter_n_layers,
                ffn_dim=adapter_ffn_dim,
                dropout=adapter_dropout,
                h=spatial_h,
                w=spatial_w,
                use_finger_embed=use_finger_embed,
                use_pos_query=use_pos_query,
                residual=adapter_residual,
                use_pose_injection=adapter_use_pose_injection,
                pose_dim=adapter_pose_dim,
                use_timesformer=adapter_use_timesformer,
                timesformer_num_blocks=adapter_timesformer_num_blocks,
                timesformer_num_heads=adapter_timesformer_num_heads,
                timesformer_ffn_dim=adapter_timesformer_ffn_dim,
                finger_dropout=adapter_finger_dropout,
            )
        else:
            raise ValueError(
                f"Unknown adapter_kind: {adapter_kind!r}. Must be "
                f"'finger_attention', 'concat_channel', or "
                f"'finger_set_transformer'."
            )

        self.modality_embed = ModalityEmbedding(dim=latent_channels)

        self.aux_flow_post = AuxFlowDecoder(
            latent_channels=latent_channels,
            num_fingers=num_fingers,
        )

        # v0d Option A: when pose is injected on the encoder side it is no
        # longer a supervision target (it would be cycle-consistency in the
        # most trivial sense -- "decoder recovers the input"). Mirrors
        # TouchAnything's tactile_prediction branch which uses PoseEncoder
        # but no PoseDecoder. Setting self.aux_pose=None lets the trainer
        # gate loss_pose on `model.aux_pose is None`.
        if self.use_pose_injection:
            self.aux_pose = None
        else:
            self.aux_pose = AuxPoseDecoder(latent_channels=latent_channels)

        if enable_pre_fuse:
            self.aux_flow_pre = AuxPreFuseFlowHead(
                latent_channels=latent_channels,
                num_fingers=num_fingers,
            )
        else:
            self.aux_flow_pre = None

    # ------------------------------------------------------------------
    # Forward sub-stages -- exposed for inspection / smoke / inferencer
    # ------------------------------------------------------------------

    def encode_per_finger(
        self,
        tactile: torch.Tensor,
        latent_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode a per-hand tactile clip into per-finger visual VAE latents.

        Args:
            tactile: ``(B, F, 1, T, H, W)`` grayscale tactile in ``[-1, 1]``.
            latent_mode: ``'mean'`` or ``'sample'``; defaults to the model's
                configured ``self.latent_mode``.

        Returns:
            ``(B, F, C_lat, T_lat, H_lat, W_lat)`` per-finger latent tensor.
        """
        if tactile.ndim != 6:
            raise ValueError(
                f"Expected (B, F, 1, T, H, W); got shape {tuple(tactile.shape)}."
            )
        b, f, c, t, h, w = tactile.shape
        if c != 1:
            raise ValueError(f"Expected grayscale (C=1); got C={c}.")
        if f != self.num_fingers:
            raise ValueError(
                f"Expected {self.num_fingers} fingers; got {f}."
            )

        latent_mode = latent_mode or self.latent_mode

        # Flatten finger axis into batch for the VAE (shared weights across
        # fingers since the LTX encoder doesn't know about hand structure).
        x = rearrange(tactile, "b f c t h w -> (b f) c t h w")
        x_rgb = self.gray_to_rgb(x)  # (B*F, 3, T, H, W)

        # NOTE: deliberately NOT wrapped in torch.no_grad(). VAE params have
        # requires_grad=False so they are not updated, but the autograd graph
        # must remain alive for gradients to reach `gray_to_rgb`.
        out = self.vae.encode(x_rgb)
        latent_dist = out.latent_dist

        if latent_mode == "mean":
            z = latent_dist.mode()
        else:  # "sample"
            z = latent_dist.sample()

        z = rearrange(z, "(b f) c t h w -> b f c t h w", b=b, f=f)
        return z

    @staticmethod
    def _align_pose_to_lat(
        pose: torch.Tensor, T_lat: int,
    ) -> torch.Tensor:
        """Linspace-resample ``(B, T_raw, P)`` -> ``(B, T_lat, P)``.

        Two paths are reached by the official v0d_full config
        (``pose_mode='per_frame'``):

          * ``T_raw == T_lat`` (incl. T=1->T_lat=1): identity.
          * ``T_raw=9, T_lat=2``: linspace picks indices ``[0, 8]``, i.e.
            clip-start + clip-end. Slot 1 is bit-identical to v0c-A's
            last-frame ``state[last_frame]`` ground truth, so this
            alignment is a strict information superset of v0c-A's per-clip
            pose (Section 6.3 of the v0d plan).

        A third "broadcast" path (``T_raw=1, T_lat>1``) is reached ONLY
        under misconfiguration; see the inline comment below.
        """
        if pose.ndim != 3:
            raise ValueError(
                f"_align_pose_to_lat expects 3-D (B, T_raw, P); got shape "
                f"{tuple(pose.shape)}."
            )
        if T_lat < 1:
            raise ValueError(f"T_lat must be >= 1; got {T_lat}.")
        T_raw = pose.shape[1]
        if T_raw == T_lat:
            return pose
        if T_raw == 1:
            # Defensive fallback only. In the official v0d config we set
            # pose_mode="per_frame", so T=9 produces pose shape (B, 9, P)
            # and this branch is never used. This branch only fires when
            # use_pose_injection=True is enabled while the dataset still
            # emits last_frame pose, i.e. (B, 1, P), and the VAE output
            # has T_lat>1. In that degraded/backward-compatible case,
            # broadcast the single pose to all latent slots instead of
            # crashing. The trainer / wrapper emits a one-time warning
            # before invoking this path so the misconfiguration is
            # visible without spamming the log every batch.
            return pose.expand(pose.shape[0], T_lat, pose.shape[2])
        # General linspace resample. For T_raw=9, T_lat=2 -> indices=[0, 8]
        # (clip first + last frame). round() before long() guarantees we
        # land on actual frame indices rather than truncating early.
        indices = torch.linspace(
            0, T_raw - 1, T_lat, device=pose.device,
        ).round().long()
        return pose.index_select(dim=1, index=indices)

    def fuse_per_hand(
        self,
        z_per_finger: torch.Tensor,
        hand_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the adapter + modality embedding on a per-finger latent.

        Args:
            z_per_finger: ``(B, F, C, T_lat, H, W)`` per-finger latent.
            hand_pose: ``(B, T_raw, P)`` per-hand pose at the raw (dataset)
                temporal resolution. Required only when the adapter has
                ``use_pose_injection=True`` (v0d); silently ignored
                otherwise. If supplied it is resampled to ``T_lat`` via
                :meth:`_align_pose_to_lat` before being passed to the
                adapter, so the adapter always sees pose at latent-time
                resolution.

        Returns:
            ``(B, C, T_lat, H, W)`` per-hand latent ready for Stage 2 /
            aux heads.
        """
        # Branch only when the adapter actually consumes hand_pose. Calling
        # the v0/v0b/v0c adapter signatures with an extra hand_pose kwarg
        # would either silently drop it (FingerAttentionAdapter) or raise
        # (the v0d adapter when use_pose_injection=True), so we route here.
        if self.use_pose_injection:
            if hand_pose is None:
                raise ValueError(
                    "VisualVAEAdapterModel.fuse_per_hand: hand_pose is "
                    "required when adapter use_pose_injection=True; got None."
                )
            T_lat = z_per_finger.shape[3]
            # One-time misconfiguration warning: T_raw=1 (last_frame dataset
            # mode) combined with T_lat>1 (T=9 clips through the LTX VAE)
            # silently degrades to broadcasting a single pose across all
            # latent slots. For the official v0d_full config (pose_mode=
            # 'per_frame') this never fires; if it does, the user almost
            # certainly forgot to flip pose_mode in the YAML.
            if (
                not self._pose_broadcast_warned
                and hand_pose.ndim == 3
                and hand_pose.shape[1] == 1
                and T_lat > 1
            ):
                warnings.warn(
                    "use_pose_injection=True with single-frame pose "
                    f"(T_raw=1) and T_lat={T_lat}>1. This is a fallback "
                    "path: the single pose is broadcast to every latent "
                    "slot. For v0d_full, set pose_mode='per_frame' in the "
                    "dataset YAML so per-frame pose is emitted instead. "
                    "This warning is emitted once per model instance.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._pose_broadcast_warned = True
            pose_at_lat = self._align_pose_to_lat(hand_pose, T_lat)
            z_hand = self.adapter(z_per_finger, hand_pose=pose_at_lat)
        else:
            z_hand = self.adapter(z_per_finger)
        z_hand = self.modality_embed(z_hand)
        return z_hand

    @torch.no_grad()
    def encode_per_hand(
        self,
        tactile: torch.Tensor,
        hand_pose: Optional[torch.Tensor] = None,
        latent_mode: Optional[str] = None,
        return_per_finger: bool = False,
    ) -> torch.Tensor:
        """Stage-2-only API: encode a multi-hand tactile clip into per-hand
        fused latents in ONE call, with NO aux heads invoked.

        Wraps ``encode_per_finger`` -> ``fuse_per_hand`` for the multi-hand
        case ``(B, V_hand, ...)``. The decorator ``@torch.no_grad()`` is
        deliberate: Stage 2 freezes the entire ``VisualVAEAdapterModel``, so
        gradients should never flow back through this call. Stage-1 callers
        that DO need gradient flow (e.g. for fine-tuning ``GrayToRGB``)
        should keep using ``encode_per_finger`` + ``fuse_per_hand`` directly.

        For ``V_hand == 1`` and ``aux_flow_pre`` disabled, this method is
        bit-for-bit identical to ``forward()['z_per_hand']`` (modulo the
        no_grad envelope); the smoke test
        ``scripts/smoke_visual_vae_adapter.py`` proves this with
        ``torch.allclose(rtol=0, atol=0)``.

        Args:
            tactile: ``(B, V_hand, F, T, H, W)`` grayscale tactile clip per
                hand. F == ``self.num_fingers`` (= 5 by default). NO explicit
                C=1 channel dim -- the method inserts it internally to match
                ``encode_per_finger``'s signature. dtype: ``float`` in
                ``[-1, 1]`` (matching the trainer / dataset convention).
            hand_pose: optional ``(B, V_hand, T_raw, P)`` per-hand pose at the
                raw temporal resolution. Required when the wrapper was
                constructed with ``adapter_use_pose_injection=True``
                (v0d); silently ignored otherwise. The V_hand axis is
                flattened into batch the same way as ``tactile`` so the
                adapter sees ``(B*V_hand, T_raw, P)`` before
                :meth:`_align_pose_to_lat` resamples to ``T_lat``.
            latent_mode: ``'mean'`` (default) or ``'sample'``; defaults to
                ``self.latent_mode``.
            return_per_finger: if True, also return the per-finger LTX VAE
                latent (before adapter fusion) alongside the per-hand
                result. Used by the Stage-2 val monitor to track
                distribution shift across the encoding chain. Shape:
                ``(B, V_hand, F, C_lat, T_lat, H_lat, W_lat)``.

        Returns:
            When ``return_per_finger=False`` (default):
                ``(B, V_hand, C_lat, T_lat, H_lat, W_lat)`` per-hand fused
                latent.
            When ``return_per_finger=True``:
                Tuple of (per-hand fused latent, per-finger latent).

            For the C5 contract, ``C_lat == 128``, ``H_lat == 6``,
            ``W_lat == 8`` (verified by ``scripts/audit_stage2_shapes.py``).
            The returned tensors have ``requires_grad == False`` because of
            the ``@torch.no_grad()`` decorator.
        """
        if tactile.ndim != 6:
            raise ValueError(
                f"encode_per_hand expects (B, V_hand, F, T, H, W) -- 6-D; "
                f"got shape {tuple(tactile.shape)}."
            )
        b, v_hand, f, t, h, w = tactile.shape
        if f != self.num_fingers:
            raise ValueError(
                f"encode_per_hand: F dim = {f}, but model expects "
                f"num_fingers = {self.num_fingers}."
            )
        if v_hand <= 0:
            raise ValueError(f"V_hand must be > 0; got {v_hand}.")

        if self.use_pose_injection:
            if hand_pose is None:
                raise ValueError(
                    "encode_per_hand: hand_pose is required when the "
                    "adapter has use_pose_injection=True; got None."
                )
            if hand_pose.ndim != 4:
                raise ValueError(
                    f"encode_per_hand: hand_pose must be 4-D "
                    f"(B, V_hand, T_raw, P); got shape "
                    f"{tuple(hand_pose.shape)}."
                )
            if (
                hand_pose.shape[0] != b
                or hand_pose.shape[1] != v_hand
                or hand_pose.shape[3] != self.pose_dim
            ):
                raise ValueError(
                    f"encode_per_hand: hand_pose shape mismatch -- "
                    f"expected (B={b}, V_hand={v_hand}, T_raw=*, "
                    f"P={self.pose_dim}); got {tuple(hand_pose.shape)}."
                )
            pose_flat = hand_pose.reshape(b * v_hand, hand_pose.shape[2], self.pose_dim)
        else:
            pose_flat = None

        # Insert grayscale C=1 dim and flatten V_hand into batch for the
        # shared encoder + adapter (the LTX VAE encoder does not know about
        # hand structure, so per-hand latents are just per-batch latents).
        tac_flat = tactile.reshape(b * v_hand, f, 1, t, h, w)

        z_per_finger = self.encode_per_finger(
            tac_flat, latent_mode=latent_mode,
        )                                       # (B*V_hand, F, C, T_lat, H, W)
        z_per_hand_flat = self.fuse_per_hand(
            z_per_finger, hand_pose=pose_flat,
        )  # (B*V_hand, C, T_lat, H, W)

        _, c_lat, t_lat, h_lat, w_lat = z_per_hand_flat.shape
        result = z_per_hand_flat.reshape(b, v_hand, c_lat, t_lat, h_lat, w_lat)

        if return_per_finger:
            _, pf_f, pf_c, pf_t, pf_h, pf_w = z_per_finger.shape
            z_pf_out = z_per_finger.reshape(
                b, v_hand, pf_f, pf_c, pf_t, pf_h, pf_w,
            )
            assert z_pf_out.shape[:3] == (b, v_hand, f), (
                f"per-finger reshape ordering broken: expected "
                f"({b}, {v_hand}, {f}, ...) but got {z_pf_out.shape}"
            )
            assert z_pf_out.shape[4] == result.shape[3], (
                f"per-finger T_lat ({z_pf_out.shape[4]}) != "
                f"per-hand T_lat ({result.shape[3]})"
            )
            return result, z_pf_out

        return result

    # ------------------------------------------------------------------
    # Top-level forward (returns dict of all preds + intermediate latents)
    # ------------------------------------------------------------------

    def forward(
        self,
        tactile: torch.Tensor,
        hand_pose: Optional[torch.Tensor] = None,
        latent_mode: Optional[str] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """End-to-end Stage-1 forward.

        Args:
            tactile: ``(B, F, 1, T, H, W)`` grayscale tactile clip per hand.
            hand_pose: optional ``(B, T_raw, P)`` per-hand pose at the raw
                temporal resolution. Required when ``use_pose_injection=True``
                (v0d); silently ignored otherwise. ``T_raw`` may equal 1
                (last-frame mode -- broadcast to every latent slot) or T
                (per-frame mode -- linspace-resampled to ``T_lat``).
            latent_mode: ``'mean'`` (default) or ``'sample'``; defaults to
                ``self.latent_mode``.

        Returns:
            Dict with keys ``flow_pred_post``, ``flow_pred_pre`` (or None),
            ``pose_pred`` (None when use_pose_injection=True -- v0d Option
            A: AuxPoseDecoder is dropped because pose is an input, not a
            target), ``z_per_finger``, ``z_per_hand``.
        """
        z_per_finger = self.encode_per_finger(tactile, latent_mode=latent_mode)

        if self.aux_flow_pre is not None:
            flow_pred_pre = self.aux_flow_pre(z_per_finger)
        else:
            flow_pred_pre = None

        z_per_hand = self.fuse_per_hand(z_per_finger, hand_pose=hand_pose)

        flow_pred_post = self.aux_flow_post(z_per_hand)
        # v0d Option A: when pose is injected on the encoder side, the aux
        # pose head is dropped (mirrors TouchAnything tactile_prediction).
        pose_pred = (
            self.aux_pose(z_per_hand) if self.aux_pose is not None else None
        )

        return {
            "flow_pred_post": flow_pred_post,
            "flow_pred_pre": flow_pred_pre,
            "pose_pred": pose_pred,
            "z_per_finger": z_per_finger,
            "z_per_hand": z_per_hand,
        }

    # ------------------------------------------------------------------
    # Optimizer helper
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only parameters that should be optimized.

        Filters out the frozen VAE; intended for use as
        ``optimizer = torch.optim.AdamW(model.trainable_parameters(), ...)``.
        """
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def named_trainable_parameters(self):
        """Like :meth:`trainable_parameters` but yields ``(name, param)``."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p
