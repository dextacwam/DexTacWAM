"""Compute normalization stats for the Stage-1 Tactile VAE inputs.

Produces TWO JSON files in ``--out_dir``:

* ``flow_stats.json`` -- per-channel mean/std of ``tactile_flow[..., (dx, dy, divergence)]``
  (parquet channels ``[0, 1, 3]`` -- ``magnitude`` is dropped because it is
  redundant with ``dx``/``dy``). Shape: ``mean (3,) std (3,)``.
* ``pose_stats.json`` -- per-joint mean/std of the 22-dim hand pose extracted
  from ``state[7:29]`` (left hand) and ``state[36:58]`` (right hand). Both
  hands are pooled into one population so the stats apply symmetrically.
  Shape: ``mean (22,) std (22,)``.

Both JSON files match the schema consumed by ``data.tactile_dataset._load_stats``::

    {"mean": [...], "std": [...], "q01": [...], "q99": [...], "n_samples": int}

(``q01``/``q99``/``n_samples`` are diagnostic-only -- the dataset reads only
``mean``/``std``.)

Usage (single root, v1 form)::

    python scripts/compute_tactile_stats.py \\
        --data_root data/wipe_plate \\
        --episodes 0..19 \\
        --out_dir DexVTAM/data/stats/wipe_plate

Usage (multi-root, v2 data-scaling)::

    python scripts/compute_tactile_stats.py \\
        --data_roots data/datasets/wipe_plate data/datasets/wrap_adhesive \\
        --episode_lists "0..19" "0..34" \\
        --out_dir DexVTAM/data/stats/wipe_plate_plus_wrap

The single- and multi-root forms are mutually exclusive. In multi-root mode,
all listed episodes from all listed roots are pooled into one population so
the resulting mean/std applies symmetrically across the combined train corpus
- this is what the multi-root :class:`TactileDataset` expects.

The training yaml then references those JSONs via ``flow_stats_path`` /
``pose_stats_path`` under ``data.train`` and ``data.val``.

Implementation notes:

* Two-pass online accumulation (sum, sum-of-squares). Values are O(1) for
  flow/pose, so catastrophic cancellation is not a concern at this scale.
* The flow channel sub-selection ``[0, 1, 3]`` is enforced here so the JSON
  always has dim 3, matching the runtime contract of the tactile dataset.
* Per-episode iteration with explicit ``del`` keeps peak memory bounded even
  on the largest LeRobot tasks (~hundreds of episodes).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants -- kept in sync with data/tactile_dataset.py.
# ---------------------------------------------------------------------------

_FLOW_CHANNELS_USED = (0, 1, 3)        # dx, dy, divergence
# Per-hand slice indices depend on the state-layout convention:
#   state_dim==58 (cube/488 legacy [L_arm, L_hand, R_arm, R_hand]):
#       L_hand=state[:, 7:29],  R_hand=state[:, 36:58]
#   state_dim>=90 (new convert_to_vtam.py [L_arm, R_arm, L_hand, R_hand, ...]):
#       L_hand=state[:, 14:36], R_hand=state[:, 36:58]
# R_hand is identical in both layouts (7+22+7 = 7+7+22 = 36).
# See data/dex_vtam_dataset.py::_extract_hand_pose for the mirror dispatch.
_LEFT_HAND_SLICE_58 = slice(7, 29)
_RIGHT_HAND_SLICE_58 = slice(36, 58)
_LEFT_HAND_SLICE_90 = slice(14, 36)
_RIGHT_HAND_SLICE_90 = slice(36, 58)
# Back-compat aliases for any caller importing these names directly:
_LEFT_HAND_SLICE = _LEFT_HAND_SLICE_58
_RIGHT_HAND_SLICE = _RIGHT_HAND_SLICE_58
_FLOW_RAW_CHANNELS = 4                  # parquet channel layout
_PARQUET_COLUMNS = ["tactile_flow", "state"]


def _hand_slices_for_state_dim(state_dim: int) -> Tuple[slice, slice]:
    """Return (left_slice, right_slice) consistent with the dataset's
    `_extract_hand_pose` dispatch on `state_dim`."""
    if state_dim == 58:
        return _LEFT_HAND_SLICE_58, _RIGHT_HAND_SLICE_58
    if state_dim == 90:
        return _LEFT_HAND_SLICE_90, _RIGHT_HAND_SLICE_90
    raise ValueError(
        f"Unsupported state_dim={state_dim}; expected exactly 58 (cube/488 legacy) "
        f"or >=90 (new convert_to_vtam.py arms-first layout)."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_stack(arr: np.ndarray) -> np.ndarray:
    """Recursively stack a nested-object numpy array into one dense array.

    LeRobot stores multi-dim fields as nested ``object`` arrays; this rebuilds
    the dense form in a single pass. Mirror of the helper in
    ``data/tactile_dataset.py`` to keep this script standalone.
    """
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return np.stack([_deep_stack(x) for x in arr])
    return np.asarray(arr)


def _episode_parquet_path(data_root: str, episode_index: int) -> str:
    chunk = episode_index // 1000
    return os.path.join(
        data_root, "data", f"chunk-{chunk:03d}", f"episode_{episode_index:06d}.parquet"
    )


def _load_episode(data_root: str, episode_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(tactile_flow, state)`` arrays for one episode.

    Shapes: ``tactile_flow`` is ``(N, 2, 5, 24, 32, 4) float32``,
    ``state`` is ``(N, 58) float32``.
    """
    path = _episode_parquet_path(data_root, episode_index)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path, columns=_PARQUET_COLUMNS)
    flow = np.stack([_deep_stack(v) for v in df["tactile_flow"].values]).astype(np.float32)
    state = np.stack([_deep_stack(v) for v in df["state"].values]).astype(np.float32)
    return flow, state


# ---------------------------------------------------------------------------
# Online accumulator
# ---------------------------------------------------------------------------


class _MeanStdAccumulator:
    """Two-pass online (sum, sum_sq) accumulator for per-feature mean/std."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n: int = 0
        self.sum: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.sum_sq: np.ndarray = np.zeros(dim, dtype=np.float64)

    def update(self, chunk: np.ndarray) -> None:
        """Update with a 2-D ``(B, D)`` block."""
        if chunk.size == 0:
            return
        if chunk.ndim != 2 or chunk.shape[1] != self.dim:
            raise ValueError(
                f"Accumulator dim={self.dim} but chunk shape {chunk.shape}."
            )
        chunk_d = chunk.astype(np.float64, copy=False)
        self.sum += chunk_d.sum(axis=0)
        self.sum_sq += np.square(chunk_d).sum(axis=0)
        self.n += chunk.shape[0]

    def mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise RuntimeError("Accumulator is empty; cannot compute mean/std.")
        mean = self.sum / self.n
        var = np.maximum(self.sum_sq / self.n - mean * mean, 0.0)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Stats workers
# ---------------------------------------------------------------------------


def _flatten_flow(flow: np.ndarray, max_frames: int) -> np.ndarray:
    """``(N, 2, 5, 24, 32, 4) → (N*2*5*24*32, 3)`` after channel sub-selection.

    Optionally subsamples the time axis to ``max_frames`` (uniform stride) so
    that very long episodes do not dominate the stats / memory.
    """
    if flow.shape[-1] != _FLOW_RAW_CHANNELS:
        raise ValueError(
            f"Expected raw flow with {_FLOW_RAW_CHANNELS} channels, got {flow.shape[-1]}."
        )
    if max_frames is not None and flow.shape[0] > max_frames:
        idx = np.linspace(0, flow.shape[0] - 1, num=max_frames, dtype=np.int64)
        flow = flow[idx]
    flow = flow[..., list(_FLOW_CHANNELS_USED)]            # (N', 2, 5, 24, 32, 3)
    return flow.reshape(-1, len(_FLOW_CHANNELS_USED))      # (N'*2*5*24*32, 3)


def _flatten_pose(state: np.ndarray, max_frames: int) -> np.ndarray:
    """``(N, state_dim) → (2*N, 22)``: stack left + right hands into one
    population. Per-hand slices are layout-aware; see ``_hand_slices_for_state_dim``."""
    left_slice, right_slice = _hand_slices_for_state_dim(state.shape[-1])
    if max_frames is not None and state.shape[0] > max_frames:
        idx = np.linspace(0, state.shape[0] - 1, num=max_frames, dtype=np.int64)
        state = state[idx]
    left = state[:, left_slice]
    right = state[:, right_slice]
    return np.concatenate([left, right], axis=0)           # (2N', 22)


def compute_stats(
    roots_and_episodes: Sequence[Tuple[str, Sequence[int]]],
    max_frames_per_episode: int = -1,
    quantile_subsample: int = 200_000,
) -> Tuple[dict, dict]:
    """Run the two passes over a list of ``(data_root, episodes)`` pairs and
    return ``(flow_stats, pose_stats)`` aggregated across the whole population.

    For diagnostics, the function also gathers up to ``quantile_subsample`` rows
    of pose / flow so that q01 and q99 can be reported in the JSON. These are
    NOT used by the dataset; they exist for the user to sanity-check.
    """
    if not roots_and_episodes:
        raise ValueError("roots_and_episodes must contain at least one entry.")

    flow_acc = _MeanStdAccumulator(dim=len(_FLOW_CHANNELS_USED))
    pose_acc = _MeanStdAccumulator(dim=22)

    rng = np.random.default_rng(0)
    flow_q_pool: List[np.ndarray] = []
    pose_q_pool: List[np.ndarray] = []
    flow_q_budget = quantile_subsample
    pose_q_budget = quantile_subsample

    mfpe = max_frames_per_episode if max_frames_per_episode > 0 else None

    # Flatten the (root, ep) pairs into one list so the progress bar reflects
    # the true total work across all roots.
    flat_jobs: List[Tuple[str, int]] = []
    for data_root, episodes in roots_and_episodes:
        for ep in episodes:
            flat_jobs.append((data_root, int(ep)))

    for data_root, ep in tqdm(flat_jobs, desc="episodes", unit="ep"):
        flow, state = _load_episode(data_root, ep)
        flow_flat = _flatten_flow(flow, max_frames=mfpe)
        pose_flat = _flatten_pose(state, max_frames=mfpe)

        flow_acc.update(flow_flat)
        pose_acc.update(pose_flat)

        # Reservoir-light quantile pool (random per-episode subsample).
        if flow_q_budget > 0:
            take = min(flow_q_budget, flow_flat.shape[0])
            idx = rng.choice(flow_flat.shape[0], size=take, replace=False)
            flow_q_pool.append(flow_flat[idx].copy())
            flow_q_budget -= take
        if pose_q_budget > 0:
            take = min(pose_q_budget, pose_flat.shape[0])
            idx = rng.choice(pose_flat.shape[0], size=take, replace=False)
            pose_q_pool.append(pose_flat[idx].copy())
            pose_q_budget -= take

        del flow, state, flow_flat, pose_flat

    flow_mean, flow_std = flow_acc.mean_std()
    pose_mean, pose_std = pose_acc.mean_std()

    flow_q = np.concatenate(flow_q_pool, axis=0) if flow_q_pool else np.zeros((0, 3), np.float32)
    pose_q = np.concatenate(pose_q_pool, axis=0) if pose_q_pool else np.zeros((0, 22), np.float32)

    sources = [
        {"data_root": str(root), "episodes": [int(e) for e in eps]}
        for root, eps in roots_and_episodes
    ]
    n_roots = len(roots_and_episodes)

    flow_stats = {
        "mean": flow_mean.tolist(),
        "std": flow_std.tolist(),
        "q01": np.percentile(flow_q, 1, axis=0).tolist() if flow_q.size else [None] * 3,
        "q99": np.percentile(flow_q, 99, axis=0).tolist() if flow_q.size else [None] * 3,
        "n_samples": int(flow_acc.n),
        "channel_order": ["dx", "dy", "divergence"],
        "source_channels": list(_FLOW_CHANNELS_USED),
        "sources": sources,
    }
    pose_stats = {
        "mean": pose_mean.tolist(),
        "std": pose_std.tolist(),
        "q01": np.percentile(pose_q, 1, axis=0).tolist() if pose_q.size else [None] * 22,
        "q99": np.percentile(pose_q, 99, axis=0).tolist() if pose_q.size else [None] * 22,
        "n_samples": int(pose_acc.n),
        "joint_order": "state[7:29] (left) and state[36:58] (right), pooled",
        "sources": sources,
    }

    # Backward-compat keys for single-root callers (v1 stats files write
    # ``data_root`` and ``episodes`` at the top level). Always emitted for the
    # first source so existing tooling that grep'd these strings keeps working.
    flow_stats["data_root"] = sources[0]["data_root"]
    flow_stats["episodes"] = sources[0]["episodes"]
    pose_stats["data_root"] = sources[0]["data_root"]
    pose_stats["episodes"] = sources[0]["episodes"]
    if n_roots > 1:
        flow_stats["multi_root"] = True
        pose_stats["multi_root"] = True

    return flow_stats, pose_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_episodes_arg(values: List[str]) -> List[int]:
    """Allow ``--episodes 0..19`` shorthand alongside explicit lists."""
    out: List[int] = []
    for v in values:
        if ".." in v:
            lo, hi = v.split("..", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(v))
    return out


def _resolve_roots_and_episodes(args) -> List[Tuple[str, List[int]]]:
    """Validate the argparse namespace and return a list of ``(root, episodes)``
    pairs. Exactly one of (``--data_root`` + ``--episodes``) or
    (``--data_roots`` + ``--episode_lists``) must be provided.
    """
    single_form = args.data_root is not None or args.episodes is not None
    multi_form = args.data_roots is not None or args.episode_lists is not None
    if single_form and multi_form:
        raise SystemExit(
            "Pass either --data_root + --episodes (single-root, v1) OR "
            "--data_roots + --episode_lists (multi-root, v2), not both."
        )
    if not single_form and not multi_form:
        raise SystemExit(
            "Must pass either --data_root + --episodes (single-root) OR "
            "--data_roots + --episode_lists (multi-root)."
        )

    if single_form:
        if args.data_root is None or args.episodes is None:
            raise SystemExit("Single-root mode requires both --data_root and --episodes.")
        eps = _parse_episodes_arg(args.episodes)
        return [(args.data_root, eps)]

    if args.data_roots is None or args.episode_lists is None:
        raise SystemExit("Multi-root mode requires both --data_roots and --episode_lists.")
    if len(args.data_roots) != len(args.episode_lists):
        raise SystemExit(
            f"--data_roots ({len(args.data_roots)} given) and --episode_lists "
            f"({len(args.episode_lists)} given) must have the same number of entries."
        )
    pairs: List[Tuple[str, List[int]]] = []
    for root, ep_spec in zip(args.data_roots, args.episode_lists):
        # Each --episode_lists entry is one string. Accept comma-separated list,
        # space-separated list (already split by argparse if quoted), or range
        # syntax "0..19".
        tokens = [tok for tok in ep_spec.replace(",", " ").split() if tok]
        if not tokens:
            raise SystemExit(f"Empty episode list for data_root={root!r}.")
        eps = _parse_episodes_arg(tokens)
        pairs.append((root, eps))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n", 1)[0],
    )
    # Single-root form (v1, kept for backward compatibility):
    parser.add_argument("--data_root", default=None, help="LeRobot dataset root (single-root form).")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help='Episode indices for the single-root form, e.g. "0 1 2" or "0..19".',
    )
    # Multi-root form (v2, data scaling):
    parser.add_argument(
        "--data_roots",
        nargs="+",
        default=None,
        help="Two or more LeRobot dataset roots (multi-root form). Pair with --episode_lists.",
    )
    parser.add_argument(
        "--episode_lists",
        nargs="+",
        default=None,
        help=(
            "One episode-spec string per --data_roots entry. Each string accepts "
            "comma- or space-separated indices and ``a..b`` ranges, e.g. "
            "'0..19' '0,1,2,3..7'."
        ),
    )
    parser.add_argument("--out_dir", required=True, help="Where to write the two JSON files.")
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=-1,
        help="If positive, uniformly subsample frames per episode to bound memory/time.",
    )
    parser.add_argument(
        "--quantile_subsample",
        type=int,
        default=200_000,
        help="Total rows pooled across episodes for q01/q99 diagnostics.",
    )
    args = parser.parse_args()

    roots_and_episodes = _resolve_roots_and_episodes(args)
    total_eps = sum(len(eps) for _, eps in roots_and_episodes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stats] mode       = {'multi-root' if len(roots_and_episodes) > 1 else 'single-root'}")
    for i, (root, eps) in enumerate(roots_and_episodes):
        print(f"[stats] root[{i}]    = {root}")
        print(f"[stats] eps[{i}]     = {eps}")
    print(f"[stats] total_eps  = {total_eps}")
    print(f"[stats] out_dir    = {out_dir}")
    print(f"[stats] max_frames = {args.max_frames_per_episode}")

    flow_stats, pose_stats = compute_stats(
        roots_and_episodes=roots_and_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        quantile_subsample=args.quantile_subsample,
    )

    flow_path = out_dir / "flow_stats.json"
    pose_path = out_dir / "pose_stats.json"
    with open(flow_path, "w") as f:
        json.dump(flow_stats, f, indent=2)
    with open(pose_path, "w") as f:
        json.dump(pose_stats, f, indent=2)

    def _fmt(arr):
        return "[" + ", ".join(f"{x:+.4f}" for x in arr) + "]"

    print(f"\n[flow] n={flow_stats['n_samples']:,}")
    print(f"[flow] mean = {_fmt(flow_stats['mean'])}    (dx, dy, divergence)")
    print(f"[flow] std  = {_fmt(flow_stats['std'])}")
    print(f"[flow] q01  = {_fmt(flow_stats['q01'])}")
    print(f"[flow] q99  = {_fmt(flow_stats['q99'])}")

    print(f"\n[pose] n={pose_stats['n_samples']:,}")
    print(f"[pose] mean[:5] = {_fmt(pose_stats['mean'][:5])}  ...")
    print(f"[pose] std [:5] = {_fmt(pose_stats['std'][:5])}  ...")

    print(f"\n[stats] wrote {flow_path}")
    print(f"[stats] wrote {pose_path}")


if __name__ == "__main__":
    main()
