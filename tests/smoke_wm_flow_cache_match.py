#!/usr/bin/env python3
"""Mandatory R2 smoke gate: cached WM flow vs live parquet flow.

Checks that `tactile_flow.npy` cache content matches flow decoded from parquet
for selected episodes/frames (after selecting channels [0,1,3]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.tactile_dataset import _deep_stack, _FLOW_CHANNELS_USED


def _episode_parquet_path(data_root: str, domain: str, episode_idx: int) -> str:
    domain_dir = os.path.join(data_root, domain)
    if os.path.isfile(os.path.join(domain_dir, "meta", "info.json")):
        meta_dir = os.path.join(domain_dir, "meta")
        data_dir = os.path.join(domain_dir, "data")
    else:
        meta_dir = os.path.join(data_root, "meta")
        data_dir = os.path.join(data_root, "data")
    with open(os.path.join(meta_dir, "info.json"), "r") as f:
        info = json.load(f)
    chunks_size = int(info["chunks_size"])
    chunk_idx = episode_idx // chunks_size
    return os.path.join(
        data_dir, f"chunk-{chunk_idx:03d}", f"episode_{episode_idx:06d}.parquet"
    )


def _load_live_flow(parquet_path: str, frame_idxs: List[int]) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["tactile_flow"])
    rows = df["tactile_flow"].to_list()
    flow = np.stack([_deep_stack(rows[i]) for i in frame_idxs]).astype(np.float32)
    flow = flow[..., list(_FLOW_CHANNELS_USED)]  # (T, 2, 5, 24, 32, 3)
    return flow


def _load_cached_flow(cache_dir: str, episode_idx: int, frame_idxs: List[int]) -> np.ndarray:
    flow_path = os.path.join(cache_dir, f"episode_{episode_idx:06d}", "tactile_flow.npy")
    if not os.path.isfile(flow_path):
        raise FileNotFoundError(f"missing cached flow file: {flow_path}")
    flow = np.load(flow_path, mmap_mode="r")[frame_idxs]  # (T, 2, 5, 24, 32, 4)
    flow = np.asarray(flow).astype(np.float32)[..., list(_FLOW_CHANNELS_USED)]
    return flow


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test cache flow equals live parquet flow")
    p.add_argument("--data-root", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--episodes", required=True, help="Comma-separated episode ids")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    if not episodes:
        raise ValueError("episodes list is empty")

    max_abs_all = 0.0
    for ep in episodes:
        parquet_path = _episode_parquet_path(args.data_root, args.domain, ep)
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"missing parquet: {parquet_path}")
        df_len = len(pd.read_parquet(parquet_path, columns=["tactile_flow"]))
        n = min(args.num_frames, df_len)
        frame_idxs = sorted(rng.choice(df_len, size=n, replace=False).tolist())

        live = _load_live_flow(parquet_path, frame_idxs)
        cached = _load_cached_flow(args.cache_dir, ep, frame_idxs)
        diff = np.abs(live - cached)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        max_abs_all = max(max_abs_all, max_abs)
        print(
            f"episode={ep} frames={len(frame_idxs)} max_abs={max_abs:.8f} "
            f"mean_abs={mean_abs:.8f}"
        )

        if not np.allclose(live, cached, atol=args.atol, rtol=0.0):
            raise AssertionError(
                f"cache-vs-live mismatch at episode={ep}: max_abs={max_abs:.8f} "
                f"(atol={args.atol})"
            )

    print(f"PASS: all episodes match within atol={args.atol}. global_max_abs={max_abs_all:.8f}")


if __name__ == "__main__":
    main()

