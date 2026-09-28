"""Tactile flow visualization utility shared by Stage-1 and Stage-2 trainers.

Renders side-by-side GT vs predicted tactile flow as a matplotlib grid:
each (hand, finger) is one row showing 3 GT channels (dx/dy/div) followed
by 3 predicted channels for the same finger. This is the visualization
format used by Stage-1 ``VisualVAEAdapterTrainer._save_flow_viz`` and
``TactileVAETrainer._save_flow_viz``; this module is the single source of
truth so Stage-2 WM validation can reuse it without coupling to either
Stage-1 trainer module.

Single-hand (Stage-1 use)::

    flow_gt   : (F, H, W, C)         # F=5 fingers, C=3 (dx, dy, div)
    flow_pred : (F, H, W, C)
    -> F x 6 grid (3 GT cols + 3 pred cols), one row per finger.

Multi-hand (Stage-2 WM use)::

    flow_gt   : (V_hand, F, H, W, C) # V_hand=2 (left, right)
    flow_pred : (V_hand, F, H, W, C)
    -> (V_hand * F) x 6 grid; rows are grouped by hand.

The function accepts either torch tensors or numpy arrays and returns the
absolute save path. matplotlib uses the Agg backend (set on import) so the
util is safe to call from any process / worker, including dataloader
workers and DDP rank-0 only contexts.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_DEFAULT_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_DEFAULT_CHAN_NAMES = ("dx", "dy", "div")


def resolve_frame_indices(spec: str, T_out: int) -> List[int]:
    """Parse a frame-index CLI spec into a concrete list of indices.

    Used by online / offline tactile flow viz to pick which future
    latent frames to render. Lives next to ``save_flow_compare_grid``
    so callers only depend on a single module for viz-side utilities.

    Accepted forms (case-insensitive)::

        "all"   -> [0, 1, ..., T_out - 1]
        "last"  -> [T_out - 1]
        "<int>" -> [int]; negative values count from the end
                   (e.g. "-1" -> [T_out - 1])

    Raises:
        ValueError: if ``spec`` is non-numeric / non-keyword, or if a
            numeric index resolves outside ``[0, T_out)``.
    """
    spec_norm = spec.strip().lower()
    if spec_norm == "all":
        return list(range(T_out))
    if spec_norm == "last":
        return [T_out - 1]
    try:
        idx = int(spec_norm)
    except ValueError as e:
        raise ValueError(
            f"resolve_frame_indices: spec must be 'all', 'last', or an int; "
            f"got {spec!r}"
        ) from e
    if idx < 0:
        idx = T_out + idx
    if idx < 0 or idx >= T_out:
        raise ValueError(
            f"resolve_frame_indices: {spec!r} resolves to {idx}, out of range "
            f"[0, {T_out})"
        )
    return [idx]


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def save_flow_compare_grid(
    flow_gt: Union[torch.Tensor, np.ndarray],
    flow_pred: Union[torch.Tensor, np.ndarray],
    save_path: str,
    *,
    title: str = "",
    finger_names: Sequence[str] = _DEFAULT_FINGER_NAMES,
    chan_names: Sequence[str] = _DEFAULT_CHAN_NAMES,
    hand_names: Optional[Sequence[str]] = None,
    cmap: str = "seismic",
    dpi: int = 80,
    figsize_per_row: float = 2.0,
    figsize_per_col: float = 2.3,
) -> str:
    """Render a (V_hand * F) x 6 grid of GT vs predicted tactile flow.

    Args:
        flow_gt: GT flow tensor / array. Accepts either ``(F, H, W, C)``
            (single hand) or ``(V_hand, F, H, W, C)`` (multi-hand). C is
            the channel axis (typically 3: dx, dy, div).
        flow_pred: predicted flow with identical shape to ``flow_gt``.
        save_path: absolute path to the output PNG.
        title: optional figure suptitle.
        finger_names: per-finger row labels. Length must be >= F.
        chan_names: per-channel column labels. Length must be >= C.
        hand_names: optional per-hand row-group labels. When provided,
            length must be == V_hand. Defaults to ``["hand0", "hand1", ...]``
            when V_hand > 1.
        cmap: matplotlib colormap (default ``"seismic"`` for signed flow).
        dpi: output PNG DPI.
        figsize_per_row / figsize_per_col: figure size scaling.

    Returns:
        The absolute save path that was written.

    Raises:
        ValueError: if shape contract is violated.
    """
    gt = _to_numpy(flow_gt)
    pred = _to_numpy(flow_pred)

    if gt.shape != pred.shape:
        raise ValueError(
            f"save_flow_compare_grid: gt.shape {gt.shape} != pred.shape "
            f"{pred.shape}"
        )

    if gt.ndim == 4:
        # Single-hand input. Lift to (1, F, H, W, C) for uniform handling.
        gt = gt[None, ...]
        pred = pred[None, ...]
    elif gt.ndim != 5:
        raise ValueError(
            f"save_flow_compare_grid: expected ndim==4 (F,H,W,C) or "
            f"ndim==5 (V_hand,F,H,W,C); got shape {gt.shape}"
        )

    V_hand, F, H, W, C = gt.shape
    if C < len(chan_names):
        raise ValueError(
            f"save_flow_compare_grid: flow has C={C} channels but "
            f"len(chan_names)={len(chan_names)}; pass a shorter chan_names "
            f"or supply more channels."
        )
    if F > len(finger_names):
        raise ValueError(
            f"save_flow_compare_grid: flow has F={F} fingers but only "
            f"{len(finger_names)} finger names provided."
        )
    if hand_names is None:
        hand_names = (
            (f"hand{vh}" for vh in range(V_hand))
            if V_hand > 1
            else ("",)
        )
        hand_names = tuple(hand_names)
    elif len(hand_names) != V_hand:
        raise ValueError(
            f"save_flow_compare_grid: hand_names length {len(hand_names)} "
            f"!= V_hand {V_hand}"
        )

    n_chan_show = len(chan_names)
    n_cols = 2 * n_chan_show
    n_rows = V_hand * F

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_col * n_cols, figsize_per_row * n_rows),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=12)

    for vh in range(V_hand):
        h_tag = hand_names[vh]
        for fi in range(F):
            row = vh * F + fi
            f_label = finger_names[fi]
            row_label = f"{h_tag} {f_label}".strip()
            for ci, c_name in enumerate(chan_names):
                axes[row, ci].imshow(gt[vh, fi, :, :, ci], cmap=cmap)
                axes[row, ci].set_title(
                    f"GT {row_label} {c_name}", fontsize=8,
                )
                axes[row, ci].axis("off")

                axes[row, n_chan_show + ci].imshow(
                    pred[vh, fi, :, :, ci], cmap=cmap,
                )
                axes[row, n_chan_show + ci].set_title(
                    f"Pred {row_label} {c_name}", fontsize=8,
                )
                axes[row, n_chan_show + ci].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)
