"""Stage 2: TactileProjector -- the only Stage-2-trainable tactile-side module.

Given a frozen v0c-A per-hand latent ``(B, V_hand, C, T_lat, H, W)`` matching the
LTX visual VAE's per-camera latent layout (the C5 contract verified by
``scripts/audit_stage2_shapes.py``), the projector applies a lightweight
per-cell residual MLP plus a modality bias and a per-view (per-hand) embedding,
yielding the same shape so the trainer can ``torch.cat`` it onto the visual
views along the ``n_view`` axis::

    out = x + alpha * MLP(LN(x)) + modality_bias + view_bias[view_idx]

with the residual gated by a zero-initialized scalar ``alpha`` so at step 0::

    out_step0 = x + modality_bias + view_bias[view_idx]

i.e. the projector is functionally a marker injection at init, with the
trainable residual path inactive. This is the LLaVA-style "warm-start"
recipe: phase 1 trains the projector against a frozen DiT, alpha leaves
zero, then phase 2 unfreezes the DiT and joint training proceeds.

Why a separate module instead of reusing ``DiT.proj_in``:

  * ``DiT.proj_in`` is internal to the visual DiT and tuned for visual stats;
    re-purposing it for tactile would lose (1) per-view differentiation
    (no view_id), (2) alpha-zero warm-start (no stable transition), (3) the
    modality marker readable at the DiT input level, and (4) the phase-1
    trainable degree of freedom.

Param budget (default ``latent_dim=128, hidden_dim=256, num_views=2``):

  * LayerNorm:        2 * 128                =      256
  * MLP[0] Linear:    128 * 256 + 256        =   33 024
  * MLP[2] Linear:    256 * 128 + 128        =   32 896  (zero-init at start)
  * alpha scalar:     1                      =        1
  * modality_bias:    128                    =      128
  * view_embed:       2 * 128                =      256
  * Total:                                   ~  66 561  parameters

This module is the contract-defining thing for Stage 2 phase 1; do NOT add
cross-token mixing here. View / time / space coupling is the DiT's job.

See the paper's Section on the finger- and pose-aware tactile compressor
for the design rationale behind this contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .tactile_modules import ModalityEmbedding


__all__ = ["TactileProjector"]


class TactileProjector(nn.Module):
    """Per-cell residual MLP + modality / view marker on a per-hand latent.

    Args:
        latent_dim: channel dim of the input latent (``C`` in
            ``(B, V_hand, C, T_lat, H, W)``). Defaults to ``128`` (LTX C5).
        hidden_dim: hidden width of the residual MLP. Defaults to ``256``.
        num_views: number of distinct hand views (``V_hand``). The
            ``view_embed`` table has this many rows. Defaults to ``2``
            (left + right hand). Set higher for forward-compat ablations
            (e.g. 10 for per-finger views in Stage 3).

    Shapes (all forward calls):
        Input  ``x`` :       ``(B, V_hand, C, T_lat, H, W)``
        Input  ``view_idx``: ``(B, V_hand)`` int64 indices into ``[0, num_views)``
        Output:              ``(B, V_hand, C, T_lat, H, W)``  (same as ``x``)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_views: int = 2,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0; got {latent_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0; got {hidden_dim}")
        if num_views <= 0:
            raise ValueError(f"num_views must be > 0; got {num_views}")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_views = num_views

        # Per-cell channel-axis LayerNorm + MLP. Operates on the C dim with
        # all spatio-temporal positions treated as independent batch entries.
        # MLP uses standard PyTorch defaults (Kaiming for Linear weights, zero
        # for biases) -- we deliberately do NOT zero-init the last linear.
        # See "Init invariant" note below.
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # alpha is fp32 by default (recommended -- accelerator + autocast
        # handle bf16 in forward; trainable params keep fp32 master copies).
        #
        # Init invariant (AdaLN-Zero / DiT pattern, mirrors
        # FingerSetTransformerAdapter.alpha):
        #   alpha = 0 means at step 0, ``alpha * MLP(LN(x)) = 0 * non-zero = 0``
        #   exactly, so ``out_step0 = x + modality_bias + view_bias[view_idx]``.
        #   Crucially, ``mlp_out`` ITSELF is NOT zero (last linear is Kaiming-
        #   init), so ``d(loss)/d(alpha) = upstream * mlp_out`` is non-zero
        #   at step 0 -- alpha can immediately leave zero. Once alpha != 0,
        #   gradients propagate through ``alpha * mlp_out`` to every MLP +
        #   LayerNorm parameter. If we ALSO zero-init the last linear, the
        #   path is doubly-zero and alpha never moves -- a dead state. Hence
        #   single-zero alpha + standard MLP init is the correct recipe.
        self.alpha = nn.Parameter(torch.zeros(1))

        # ModalityEmbedding adds a single learnable bias broadcast over all
        # tactile tokens. trunc_normal_(0.02) init lives in tactile_modules.
        self.modality_bias = ModalityEmbedding(dim=latent_dim)

        # view_embed: per-hand bias. Init small so step-0 markers are mild;
        # trainable so phase 1 can learn discriminative left vs right.
        self.view_embed = nn.Embedding(num_views, latent_dim)
        nn.init.trunc_normal_(self.view_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, view_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the residual MLP + modality / view markers.

        Args:
            x: ``(B, V_hand, C, T_lat, H, W)`` per-hand latent.
            view_idx: ``(B, V_hand)`` int64 indices into ``[0, num_views)``.
                In the default 2-hand convention, ``0 = left``, ``1 = right``;
                the trainer / dataset MUST keep this order consistent across
                phase 1 / phase 2 / inference.

        Returns:
            Tensor with the same shape as ``x``.
        """
        if x.ndim != 6:
            raise ValueError(
                f"TactileProjector expects (B, V_hand, C, T_lat, H, W); got "
                f"shape {tuple(x.shape)}."
            )
        b, v_hand, c, t_lat, h, w = x.shape
        if c != self.latent_dim:
            raise ValueError(
                f"Channel dim {c} != latent_dim {self.latent_dim}."
            )
        if view_idx.shape != (b, v_hand):
            raise ValueError(
                f"view_idx must be (B, V_hand) = ({b}, {v_hand}); got "
                f"{tuple(view_idx.shape)}."
            )
        if view_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"view_idx must be int32 or int64; got dtype {view_idx.dtype}."
            )
        # Bounds check; an out-of-range index is a wiring bug, not a numerical
        # corner case, so fail loudly at forward time.
        if view_idx.numel() > 0:
            v_max = int(view_idx.max().item())
            v_min = int(view_idx.min().item())
            if v_min < 0 or v_max >= self.num_views:
                raise ValueError(
                    f"view_idx out of range [0, {self.num_views}); got "
                    f"min={v_min}, max={v_max}."
                )

        # ---------- per-cell channel-axis LN + MLP residual ----------
        # Treat each spatio-temporal cell of each (B, V_hand) as an
        # independent feature vector of dim C; LN + MLP act on C only.
        x_flat = rearrange(x, "b v c t h w -> (b v t h w) c")
        residual = self.mlp(self.norm(x_flat))
        x_flat = x_flat + self.alpha * residual
        out = rearrange(
            x_flat, "(b v t h w) c -> b v c t h w",
            b=b, v=v_hand, t=t_lat, h=h, w=w,
        )

        # ---------- modality marker ----------
        # ModalityEmbedding handles ndim=6 layout natively (broadcasts on C).
        out = self.modality_bias(out)

        # ---------- per-view (per-hand) marker ----------
        # view_embed: (num_views, C); index with view_idx (B, V_hand) ->
        # (B, V_hand, C); broadcast-add along (T, H, W).
        view_bias = self.view_embed(view_idx)            # (B, V_hand, C)
        out = out + view_bias.view(b, v_hand, c, 1, 1, 1)

        return out

    # ------------------------------------------------------------------
    # Helpers (used by trainer / smoke / inferencer)
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield all parameters (every projector parameter is trainable)."""
        for p in self.parameters():
            yield p

    def named_trainable_parameters(self):
        """Like :meth:`trainable_parameters` but yields ``(name, param)``."""
        for n, p in self.named_parameters():
            yield n, p

    def alpha_value(self) -> float:
        """Return the scalar alpha as a python float (cheap; for logging)."""
        return float(self.alpha.detach().cpu().item())

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim}, "
            f"num_views={self.num_views}"
        )
