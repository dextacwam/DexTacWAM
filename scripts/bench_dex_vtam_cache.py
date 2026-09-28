"""Benchmark ``DexVTAMDataset.__getitem__`` latency: cache vs live.

Measures steady-state per-sample wall-clock for both pipelines, on real
data, with the same dataset config that production training uses. Two
modes:

  * ``--single``  -- synchronous loop, single-process, no DataLoader. Good
    isolation of the per-sample CPU work that the cache eliminates
    (PIL decode + ``_deep_stack`` + resize). Reported as
    ``mean / median / p99`` ms/sample over N iterations after a warmup.

  * ``--dataloader`` -- ``torch.utils.data.DataLoader`` with the same
    ``num_workers`` / ``prefetch_factor`` / ``persistent_workers`` /
    ``pin_memory`` knobs the production yaml uses, batched. Reports
    ``s/batch`` mean + p99 + period-8-spike presence. Closer to training
    reality.

The bench builds two datasets (``ds_live`` cache_dir=None, ``ds_cached``
cache_dir=...) and only samples episode indexes that exist in the cache
(``--episodes``). Both datasets walk the same meta/episodes.jsonl so the
comparison is apples-to-apples.

Usage::

    python scripts/bench_dex_vtam_cache.py --single \\
      --data-root data/pick_cube/lerobot_dataset_right_hand_pick_cube_100_episodes \\
      --domain    right_hand_pick_cube \\
      --cache-dir data/cache/right_hand_pick_cube_v1_test \\
      --stat-file configs/tongs/20260801_placed_tong_right_only_relative_stats.json \\
      --episodes 0,1,2 --iters 30 --warmup 5

    python scripts/bench_dex_vtam_cache.py --dataloader \\
      ... \\
      --batch-size 8 --num-workers 4 --prefetch 4 --batches 30
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dex_vtam_dataset import DexVTAMDataset  # noqa: E402


def _build_dataset(
    data_root: str, domain: str, sample_size: Tuple[int, int],
    valid_cam: List[str], cache_dir: Optional[str], stat_file: str,
    chunk: int, n_previous: int, sample_n_frames: int,
    valid_act_dim: Optional[int], valid_sta_dim: Optional[int],
    fix_epiidx: Optional[int] = None,
) -> DexVTAMDataset:
    return DexVTAMDataset(
        data_roots=[data_root],
        domains=[domain],
        sample_size=sample_size,
        sample_n_frames=sample_n_frames,
        preprocess="resize",
        valid_cam=valid_cam,
        chunk=chunk,
        action_chunk=chunk,
        n_previous=n_previous,
        previous_pick_mode="uniform",
        random_crop=False,
        action_type="absolute",
        action_space="joint",
        train_dataset=True,
        action_key="actions",
        state_key="state",
        extra_parquet_index=False,
        valid_act_dim=valid_act_dim,
        valid_sta_dim=valid_sta_dim,
        read_tactile=True,
        tactile_key="tactile",
        repeat_dataset=1,
        stat_file=stat_file,
        cache_dir=cache_dir,
        fix_epiidx=fix_epiidx,
    )


def _summary(label: str, samples_ms: List[float]) -> Dict[str, float]:
    s = sorted(samples_ms)
    n = len(s)
    mean = statistics.fmean(s)
    median = s[n // 2]
    p90 = s[int(n * 0.90)]
    p99 = s[min(n - 1, int(n * 0.99))]
    print(f"  {label:>14s}: n={n:3d}  mean={mean:7.2f} ms  median={median:7.2f} ms  "
          f"p90={p90:7.2f} ms  p99={p99:7.2f} ms  max={s[-1]:7.2f} ms  min={s[0]:7.2f} ms")
    return {"mean": mean, "median": median, "p90": p90, "p99": p99, "max": s[-1], "min": s[0]}


def bench_single(args: argparse.Namespace) -> None:
    print(f"\n{'='*70}")
    print("Single-process __getitem__ bench (no DataLoader)")
    print(f"{'='*70}")

    sample_size = tuple(args.sample_size)
    valid_cam = list(args.valid_cam)
    ep_indexes = [int(x) for x in args.episodes.split(",") if x.strip()]
    n_previous = args.n_previous
    chunk = args.chunk
    sample_n_frames = n_previous + chunk

    print(f"  data_root  = {args.data_root}")
    print(f"  domain     = {args.domain}")
    print(f"  cache_dir  = {args.cache_dir}")
    print(f"  episodes   = {ep_indexes}")
    print(f"  iters      = {args.iters}  (after {args.warmup} warmup)")
    print(f"  sample_n_frames = {sample_n_frames}  chunk = {chunk}  n_previous = {n_previous}")

    rng = np.random.default_rng(123)

    print("\n  building ds_live (cache_dir=None)...")
    ds_live = _build_dataset(
        args.data_root, args.domain, sample_size, valid_cam,
        cache_dir=None, stat_file=args.stat_file,
        chunk=chunk, n_previous=n_previous, sample_n_frames=sample_n_frames,
        valid_act_dim=args.valid_act_dim, valid_sta_dim=args.valid_sta_dim,
    )
    print(f"  ds_live len = {len(ds_live)}")

    print("\n  building ds_cached (cache_dir set)...")
    ds_cached = _build_dataset(
        args.data_root, args.domain, sample_size, valid_cam,
        cache_dir=args.cache_dir, stat_file=args.stat_file,
        chunk=chunk, n_previous=n_previous, sample_n_frames=sample_n_frames,
        valid_act_dim=args.valid_act_dim, valid_sta_dim=args.valid_sta_dim,
    )
    print(f"  ds_cached len = {len(ds_cached)}")

    def run_one_pass(ds: DexVTAMDataset, label: str, n_iters: int, n_warmup: int) -> List[float]:
        # Use a deterministic per-iter idx so live and cached see the same episode mix.
        idx_seq = rng.choice(ep_indexes, size=n_iters + n_warmup).tolist()
        # Warmup
        for k in range(n_warmup):
            _ = ds[int(idx_seq[k])]
        # Timed
        times_ms: List[float] = []
        for k in range(n_iters):
            t0 = time.perf_counter()
            _ = ds[int(idx_seq[n_warmup + k])]
            dt_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt_ms)
        return times_ms

    print("\n  WARMUP+RUN ds_live...")
    t_live = run_one_pass(ds_live, "live", args.iters, args.warmup)

    print("\n  WARMUP+RUN ds_cached...")
    t_cached = run_one_pass(ds_cached, "cached", args.iters, args.warmup)

    print(f"\n  results (ms per __getitem__):")
    s_live = _summary("live", t_live)
    s_cached = _summary("cached", t_cached)

    speedup_mean = s_live["mean"] / max(s_cached["mean"], 1e-9)
    speedup_p99 = s_live["p99"] / max(s_cached["p99"], 1e-9)
    print(f"\n  speedup mean = {speedup_mean:.2f}x   speedup p99 = {speedup_p99:.2f}x")
    if speedup_mean >= 2.0:
        print(f"  >> Cache is {speedup_mean:.1f}x faster on mean. Worth the kill+restart.")
    else:
        print(f"  >> Cache speedup mean is only {speedup_mean:.2f}x. Re-evaluate.")


def bench_dataloader(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, Subset

    print(f"\n{'='*70}")
    print("DataLoader bench (closer to training reality)")
    print(f"{'='*70}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  num_workers     = {args.num_workers}")
    print(f"  prefetch_factor = {args.prefetch}")
    print(f"  pin_memory      = {args.pin_memory}")
    print(f"  persistent_w    = {args.persistent}")
    print(f"  batches         = {args.batches}  (warmup {args.warmup_batches})")

    sample_size = tuple(args.sample_size)
    valid_cam = list(args.valid_cam)
    ep_indexes = [int(x) for x in args.episodes.split(",") if x.strip()]
    n_previous = args.n_previous
    chunk = args.chunk
    sample_n_frames = n_previous + chunk

    print("\n  building ds_live...")
    ds_live = _build_dataset(
        args.data_root, args.domain, sample_size, valid_cam,
        cache_dir=None, stat_file=args.stat_file,
        chunk=chunk, n_previous=n_previous, sample_n_frames=sample_n_frames,
        valid_act_dim=args.valid_act_dim, valid_sta_dim=args.valid_sta_dim,
    )
    ds_live = Subset(ds_live, ep_indexes * (args.batches * args.batch_size // max(1, len(ep_indexes)) + 4))

    print("  building ds_cached...")
    ds_cached = _build_dataset(
        args.data_root, args.domain, sample_size, valid_cam,
        cache_dir=args.cache_dir, stat_file=args.stat_file,
        chunk=chunk, n_previous=n_previous, sample_n_frames=sample_n_frames,
        valid_act_dim=args.valid_act_dim, valid_sta_dim=args.valid_sta_dim,
    )
    ds_cached = Subset(ds_cached, ep_indexes * (args.batches * args.batch_size // max(1, len(ep_indexes)) + 4))

    def make_dl(ds):
        kwargs = dict(
            dataset=ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        if args.num_workers > 0:
            kwargs.update(prefetch_factor=args.prefetch, persistent_workers=args.persistent)
        return DataLoader(**kwargs)

    def run_pass(ds, label):
        dl = make_dl(ds)
        it = iter(dl)
        for _ in range(args.warmup_batches):
            _ = next(it)
        times_s: List[float] = []
        for _ in range(args.batches):
            t0 = time.perf_counter()
            _ = next(it)
            times_s.append(time.perf_counter() - t0)
        del dl, it
        return times_s

    print("\n  WARMUP+RUN ds_live...")
    t_live = run_pass(ds_live, "live")
    print(f"  live: {_fmt_pass(t_live)}")

    print("\n  WARMUP+RUN ds_cached...")
    t_cached = run_pass(ds_cached, "cached")
    print(f"  cached: {_fmt_pass(t_cached)}")

    mean_live = statistics.fmean(t_live)
    mean_cached = statistics.fmean(t_cached)
    speedup = mean_live / max(mean_cached, 1e-9)
    print(f"\n  speedup mean = {speedup:.2f}x")


def _fmt_pass(t_s: List[float]) -> str:
    s = sorted(t_s)
    n = len(s)
    mean = statistics.fmean(s)
    p99 = s[min(n - 1, int(n * 0.99))]
    spike = s[-1] / max(mean, 1e-9)
    return (f"n={n}  mean={mean*1000:7.1f} ms  median={s[n//2]*1000:7.1f} ms  "
            f"p99={p99*1000:7.1f} ms  max={s[-1]*1000:7.1f} ms  max/mean={spike:.1f}x")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true",
                      help="Single-process __getitem__ bench.")
    mode.add_argument("--dataloader", action="store_true",
                      help="DataLoader bench (matches training).")

    p.add_argument("--data-root", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--stat-file", required=True)
    p.add_argument("--episodes", default="0,1,2",
                   help="Comma-separated episode indexes. Must all be present in cache_dir.")
    p.add_argument("--sample-size", nargs=2, type=int, default=[192, 256])
    p.add_argument("--valid-cam", nargs="+", default=["head_img"])
    p.add_argument("--chunk", type=int, default=9)
    p.add_argument("--n-previous", type=int, default=4)
    p.add_argument("--valid-act-dim", type=int, default=None)
    p.add_argument("--valid-sta-dim", type=int, default=None)

    # single mode
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)

    # dataloader mode
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent", action="store_true")
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--warmup-batches", type=int, default=3)

    args = p.parse_args()
    if args.single:
        bench_single(args)
    elif args.dataloader:
        bench_dataloader(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
