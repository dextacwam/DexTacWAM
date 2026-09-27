"""Strict consistency test for ``DexVTAMDataset`` offline cache.

Verifies that, for the same fixed window indexes, the cache fast path
(``cache_dir`` set) produces identical training tensors to the live
decode path (``cache_dir=None``):

  * ``video``   max abs error < 1e-2 (uint8 quantization tolerance after
    a non-trivial resize). When ``transforms.Resize`` is a no-op (i.e.
    parquet frames are already at ``sample_size``, e.g. right_hand_pick_cube
    192x256), this is byte-identical (max abs == 0).
  * ``tactile`` max abs error == 0 (no resize on the tactile path).
  * ``actions`` max abs error < 1e-7 (only float ops, deterministic).
  * ``state``   max abs error < 1e-7.
  * ``caption`` exact str equality.

Also covers:
  * ``schema.json`` mismatch -> dataset construction must raise.
  * ``random_crop=True`` + ``cache_dir`` -> dataset construction must raise.

Two modes:

  Test A (``--hermetic``)
    Build a synthetic LeRobot dataset on disk, run preprocess in-process,
    compare per episode. CPU-only, hermetic, ~30 s.

  Test B (``--real ...``)
    Use a real ``--data-root`` and a cache produced by
    ``preprocess_dex_vtam_cache.py``. Compares ``--limit`` episodes; the
    full pipeline (parquet read + PIL decode + ``_deep_stack``) runs for
    each comparison so tolerance reflects real data quirks.

Usage::

    python scripts/test_dex_vtam_cache_consistency.py --hermetic

    python scripts/test_dex_vtam_cache_consistency.py --real \\
      --data-root data/pick_cube \\
      --domain    lerobot_dataset_right_hand_pick_cube_100_episodes \\
      --cache-dir data/cache/right_hand_pick_cube_v1_test \\
      --episodes 0,1,2

    # right-only relative corpus (tong): 2 cameras, V_hand=1, action 75 /
    # state 45, eef-suffixed stats keys. --read-hand-pose also covers
    # _extract_hand_pose, which is the one right-only path where a wrong
    # arm_layout would silently slice the WRONG 22 dims out of the 45-D state.
    python scripts/test_dex_vtam_cache_consistency.py --real \\
      --data-root data/datasets_lerobot/20260725_pick_cherry_tomato_with_tong_right_only \\
      --domain    20260725_pick_cherry_tomato_with_tong_right_only \\
      --cache-dir data/cache/tong_right_only_v1_test \\
      --valid-cam head_img right_wrist_img \\
      --action-type relative_eef_rot6d --action-space eef --arm-layout right_only \\
      --read-hand-pose --pose-stats-path data/stats/diverse_488/pose_stats.json \\
      --stat-file configs/tongs/20260801_placed_tong_right_only_relative_stats.json \\
      --n-previous 4 --episodes 0,1,2

Exits 0 on all-pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dex_vtam_dataset import DexVTAMDataset  # noqa: E402
from scripts.preprocess_dex_vtam_cache import (  # noqa: E402
    SCHEMA_VERSION,
    process_one_episode,
)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

# When sample_size == native resolution, transforms.Resize is a no-op and
# uint8 video cache is byte-identical to the live float pipeline. Use 0
# for that case. Otherwise allow 2/255 for a single quantization round-trip
# after Normalize(0.5, 0.5).
TOL_VIDEO_NOOP_RESIZE = 0.0
TOL_VIDEO_NONNOOP_RESIZE = 2.0 / 255.0 + 1e-6
TOL_TACTILE = 0.0
TOL_FLOAT = 1e-7
TOL_FIELD_DEFAULTS = {
    "actions": TOL_FLOAT,
    "state": TOL_FLOAT,
    "tactile": TOL_TACTILE,
    "video": TOL_VIDEO_NOOP_RESIZE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n[{title}]")


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.dtype != b.dtype:
        a = a.float()
        b = b.float()
    return float((a.float() - b.float()).abs().max().item())


def _compare_samples(
    s_live: Dict[str, Any],
    s_cached: Dict[str, Any],
    field_tols: Dict[str, float],
    label: str,
) -> List[str]:
    """Return list of failure messages (empty list = all pass)."""
    failures: List[str] = []
    expected_keys = set(s_live.keys())
    actual_keys = set(s_cached.keys())
    if expected_keys != actual_keys:
        failures.append(f"{label}: key set differs live={expected_keys} cached={actual_keys}")

    for key in expected_keys & actual_keys:
        live_v = s_live[key]
        cached_v = s_cached[key]
        if isinstance(live_v, torch.Tensor):
            if not isinstance(cached_v, torch.Tensor):
                failures.append(f"{label}: {key} type mismatch live=Tensor cached={type(cached_v).__name__}")
                continue
            if tuple(live_v.shape) != tuple(cached_v.shape):
                failures.append(
                    f"{label}: {key} shape mismatch live={tuple(live_v.shape)} cached={tuple(cached_v.shape)}"
                )
                continue
            if live_v.dtype != cached_v.dtype:
                failures.append(
                    f"{label}: {key} dtype mismatch live={live_v.dtype} cached={cached_v.dtype}"
                )
                continue
            err = _max_abs_err(live_v, cached_v)
            tol = field_tols.get(key, TOL_FLOAT)
            ok = err <= tol
            status = _green("OK") if ok else _red("FAIL")
            print(f"  {status} {label}.{key}: max abs err = {err:.3e} (tol {tol:.3e}, "
                  f"shape={tuple(live_v.shape)} dtype={live_v.dtype})")
            if not ok:
                failures.append(f"{label}: {key} max abs err {err} > tol {tol}")
        elif isinstance(live_v, str):
            if live_v != cached_v:
                failures.append(f"{label}: {key} str mismatch live={live_v!r} cached={cached_v!r}")
            else:
                print(f"  {_green('OK')} {label}.{key}: str match {live_v!r}")
        else:
            if live_v != cached_v:
                failures.append(f"{label}: {key} value mismatch live={live_v!r} cached={cached_v!r}")
    return failures


def _build_dataset_pair(
    data_roots: List[str],
    domains: List[str],
    cache_dir: Optional[str],
    sample_size: Tuple[int, int],
    valid_cam: List[str],
    n_action: int,
    n_state: int,
    stat_file: str,
    fix_epiidx: int,
    fix_sidx: int,
    fix_mem_idx: List[int],
    chunk: int,
    n_previous: int,
    sample_n_frames: int,
    valid_act_dim: Optional[int],
    valid_sta_dim: Optional[int],
    action_type: str = "absolute",
    action_space: str = "joint",
    arm_layout: str = "bimanual",
    read_hand_pose: bool = False,
    pose_stats_path: Optional[str] = None,
    pose_mode: str = "per_frame",
) -> DexVTAMDataset:
    """Build one dataset. Defaults reproduce the original absolute/joint probe.

    ``action_space`` must match the SUFFIX of the stats keys in ``stat_file``
    ({domain}_joint vs {domain}_eef); the relative configs use ``eef``. The
    right-only corpora additionally need ``arm_layout="right_only"``, which is
    consulted by the relative action build and by ``_extract_hand_pose``.
    """
    return DexVTAMDataset(
        data_roots=data_roots,
        domains=domains,
        sample_size=sample_size,
        sample_n_frames=sample_n_frames,
        preprocess="resize",
        valid_cam=valid_cam,
        chunk=chunk,
        action_chunk=chunk,
        n_previous=n_previous,
        previous_pick_mode="uniform",
        random_crop=False,
        action_type=action_type,
        action_space=action_space,
        arm_layout=arm_layout,
        train_dataset=True,
        action_key="actions",
        state_key="state",
        extra_parquet_index=False,
        valid_act_dim=valid_act_dim,
        valid_sta_dim=valid_sta_dim,
        read_tactile=True,
        tactile_key="tactile",
        repeat_dataset=1,
        fix_epiidx=fix_epiidx,
        fix_sidx=fix_sidx,
        fix_mem_idx=fix_mem_idx,
        stat_file=stat_file,
        cache_dir=cache_dir,
        read_hand_pose=read_hand_pose,
        pose_stats_path=pose_stats_path,
        pose_mode=pose_mode,
    )


# ---------------------------------------------------------------------------
# Test A: hermetic synthetic
# ---------------------------------------------------------------------------


def _to_nested_object_array_4d(arr: np.ndarray) -> np.ndarray:
    """Convert dense ``(V, F, H, W)`` array to LeRobot's 4-level nested
    object-array layout (each axis is a 1-D object array, leaves are 1-D
    uint8 arrays per row). Pyarrow only stores 1-D leaves in ``object``
    columns; this matches what real LeRobot parquet files contain so
    ``_deep_stack`` round-trips identically.
    """
    V, F_, H, W = arr.shape
    out = np.empty(V, dtype=object)
    for v in range(V):
        out_v = np.empty(F_, dtype=object)
        for f in range(F_):
            out_vf = np.empty(H, dtype=object)
            for h in range(H):
                out_vf[h] = arr[v, f, h, :].copy()
            out_v[f] = out_vf
        out[v] = out_v
    return out


def _make_synthetic_lerobot(
    root: str,
    n_episodes: int,
    n_frames: int,
    n_action: int,
    n_state: int,
    sample_h: int,
    sample_w: int,
    n_tactile_hands: int = 2,
    n_tactile_fingers: int = 5,
    seed: int = 42,
) -> None:
    """Materialize a complete minimal LeRobot dataset under ``root``.

    Layout: ``<root>/{meta,data/chunk-000}/``. Image bytes are PNG-encoded
    at ``(sample_h, sample_w)`` so resize is a no-op and the test asserts
    byte-identity for video. ``tactile`` is stored as 4-level nested object
    arrays to match real LeRobot parquet files (so ``_deep_stack``
    exercises the recursive path).
    """
    rng = np.random.default_rng(seed)
    meta_dir = Path(root) / "meta"
    data_dir = Path(root) / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "info.json", "w") as f:
        json.dump({"total_chunks": 1, "chunks_size": 1000}, f)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "synthetic test caption"}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in range(n_episodes):
            f.write(json.dumps({
                "episode_index": ep,
                "tasks": [f"synthetic caption episode {ep}"],
                "length": n_frames,
            }) + "\n")

    for ep in range(n_episodes):
        rows: List[Dict[str, Any]] = []
        for t in range(n_frames):
            img_arr = rng.integers(0, 256, (sample_h, sample_w, 3), dtype=np.uint8)
            img = Image.fromarray(img_arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            tactile_dense = rng.integers(
                0, 256,
                (n_tactile_hands, n_tactile_fingers, sample_h, sample_w),
                dtype=np.uint8,
            )
            row = {
                "head_img": {"bytes": buf.getvalue(), "path": ""},
                "tactile": _to_nested_object_array_4d(tactile_dense),
                "actions": rng.uniform(-0.5, 0.5, n_action).astype(np.float32),
                "state": rng.uniform(-0.5, 0.5, n_state).astype(np.float32),
                "frame_index": t,
                "episode_index": ep,
            }
            rows.append(row)
        pd.DataFrame(rows).to_parquet(data_dir / f"episode_{ep:06d}.parquet")


def _write_synthetic_stat_file(path: str, domain: str, n_action: int, n_state: int) -> None:
    """Write a stat_file shaped like StatisticInfo with q01/q99 spanning [-1, 1]."""
    stat = {
        domain + "_joint": {
            "mean": [0.0] * n_action,
            "std": [1.0] * n_action,
            "q01": [-1.0] * n_action,
            "q99": [1.0] * n_action,
        },
        domain + "_state_joint": {
            "mean": [0.0] * n_state,
            "std": [1.0] * n_state,
            "q01": [-1.0] * n_state,
            "q99": [1.0] * n_state,
        },
    }
    with open(path, "w") as f:
        json.dump(stat, f)


def _run_preprocess_in_process(
    data_root: str,
    domain: str,
    cache_dir: str,
    sample_size: Tuple[int, int],
    valid_cam: List[str],
) -> None:
    """Equivalent of preprocess_dex_vtam_cache CLI but called in-process so
    pytest-style hermetic invocation doesn't pay subprocess startup cost."""
    from datetime import datetime

    domain_dir = os.path.join(data_root, domain)
    if os.path.isfile(os.path.join(domain_dir, "meta", "tasks.jsonl")):
        meta_dir = os.path.join(domain_dir, "meta")
        data_dir = os.path.join(domain_dir, "data")
    else:
        meta_dir = os.path.join(data_root, "meta")
        data_dir = os.path.join(data_root, "data")

    with open(os.path.join(meta_dir, "info.json")) as f:
        info = json.load(f)
    chunks_size = int(info["chunks_size"])

    eps = []
    with open(os.path.join(meta_dir, "episodes.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                eps.append(json.loads(line))

    os.makedirs(cache_dir, exist_ok=True)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "sample_size": list(sample_size),
        "preprocess": "resize",
        "valid_cam": list(valid_cam),
        "tactile_key": "tactile",
        "action_key": "actions",
        "state_key": "state",
        "extra_parquet_index": False,
        "video_layout": "T_V_H_W_C_uint8_post_resize",
        "tactile_layout": "T_Vhand_F_H_W_uint8_pre_normalize",
        "action_layout": "T_C_float32_raw_pre_normalize",
        "state_layout": "T_C_float32_raw_pre_normalize",
        "domain": domain,
        "data_root": data_root,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "cache_dir": cache_dir,
        "overwrite": True,
    }
    with open(os.path.join(cache_dir, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    for ep in eps:
        episode_index = int(ep["episode_index"])
        episode_chunk = episode_index // chunks_size
        parquet_path = os.path.join(
            data_dir, f"chunk-{episode_chunk:03d}", f"episode_{episode_index:06d}.parquet"
        )
        tasks = ep["tasks"]
        caption = tasks[0] if isinstance(tasks, list) and tasks else str(tasks)
        ep_idx, status, err = process_one_episode((
            {"episode_index": episode_index, "parquet_path": parquet_path, "caption": caption},
            schema,
        ))
        if status == "error":
            raise RuntimeError(f"preprocess in-process failed on ep {ep_idx}: {err}")


def test_hermetic() -> int:
    print(f"\n{'='*70}\nTest A: HERMETIC (synthetic LeRobot, in-process preprocess)\n{'='*70}")

    n_episodes = 3
    n_frames = 30
    n_action = 7
    n_state = 7
    sample_h, sample_w = 192, 256
    sample_size = (sample_h, sample_w)
    chunk = 4
    n_previous = 12
    sample_n_frames = n_previous + chunk

    failures: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        data_root = os.path.join(tmp, "data_root")
        domain = "synthetic_corpus"
        domain_root = os.path.join(data_root, domain)
        cache_dir = os.path.join(tmp, "cache_dir")
        cache_dir_bad = os.path.join(tmp, "cache_dir_bad")
        stat_file = os.path.join(tmp, "stat.json")

        _section("0] make synthetic LeRobot tree")
        _make_synthetic_lerobot(
            domain_root,
            n_episodes=n_episodes, n_frames=n_frames,
            n_action=n_action, n_state=n_state,
            sample_h=sample_h, sample_w=sample_w,
        )
        _write_synthetic_stat_file(stat_file, domain, n_action, n_state)
        print(f"  synthetic data: {n_episodes} episode(s), {n_frames} frames each, "
              f"size=({sample_h},{sample_w})")

        _section("1] run preprocess in-process")
        _run_preprocess_in_process(data_root, domain, cache_dir, sample_size, ["head_img"])
        ep_dirs = sorted(os.listdir(cache_dir))
        print(f"  cache_dir contains: {ep_dirs}")

        _section("2] schema mismatch must raise (sample_size flipped)")
        os.makedirs(cache_dir_bad, exist_ok=True)
        with open(os.path.join(cache_dir, "schema.json")) as f:
            schema = json.load(f)
        bad_schema = dict(schema)
        bad_schema["sample_size"] = [256, 192]   # swapped H/W
        with open(os.path.join(cache_dir_bad, "schema.json"), "w") as f:
            json.dump(bad_schema, f)
        try:
            _build_dataset_pair(
                data_roots=[data_root], domains=[domain],
                cache_dir=cache_dir_bad, sample_size=sample_size,
                valid_cam=["head_img"], n_action=n_action, n_state=n_state,
                stat_file=stat_file, fix_epiidx=0, fix_sidx=4, fix_mem_idx=list(range(12)),
                chunk=chunk, n_previous=n_previous, sample_n_frames=sample_n_frames,
                valid_act_dim=None, valid_sta_dim=None,
            )
            failures.append("schema mismatch did not raise")
            print(f"  {_red('FAIL')} expected ValueError, got success")
        except ValueError as e:
            print(f"  {_green('OK')} ValueError raised as expected: {str(e).splitlines()[0]}")
        except Exception as e:
            failures.append(f"schema mismatch raised wrong error type: {type(e).__name__}: {e}")
            print(f"  {_red('FAIL')} wrong error type: {type(e).__name__}: {e}")

        _section("3] random_crop=True + cache_dir must raise")
        try:
            DexVTAMDataset(
                data_roots=[data_root], domains=[domain],
                sample_size=sample_size, sample_n_frames=sample_n_frames,
                preprocess="resize", valid_cam=["head_img"],
                chunk=chunk, action_chunk=chunk, n_previous=n_previous,
                random_crop=True,  # <-- the trigger
                action_key="actions", state_key="state",
                extra_parquet_index=False, read_tactile=True, tactile_key="tactile",
                stat_file=stat_file, cache_dir=cache_dir,
            )
            failures.append("random_crop=True + cache_dir did not raise")
            print(f"  {_red('FAIL')} expected ValueError")
        except ValueError as e:
            print(f"  {_green('OK')} ValueError raised: {str(e).splitlines()[0]}")
        except Exception as e:
            failures.append(f"random_crop raised wrong type: {type(e).__name__}: {e}")
            print(f"  {_red('FAIL')} wrong error type: {type(e).__name__}: {e}")

        _section("4] per-episode field-by-field consistency (live vs cached)")
        for ep in range(n_episodes):
            print(f"\n  --- episode {ep} ---")
            fix_sidx = 16
            fix_mem_idx = list(range(12))
            common = dict(
                data_roots=[data_root], domains=[domain], sample_size=sample_size,
                valid_cam=["head_img"], n_action=n_action, n_state=n_state,
                stat_file=stat_file, fix_epiidx=ep, fix_sidx=fix_sidx,
                fix_mem_idx=fix_mem_idx, chunk=chunk, n_previous=n_previous,
                sample_n_frames=sample_n_frames, valid_act_dim=None, valid_sta_dim=None,
            )
            ds_live = _build_dataset_pair(cache_dir=None, **common)
            ds_cached = _build_dataset_pair(cache_dir=cache_dir, **common)
            s_live = ds_live[0]
            s_cached = ds_cached[0]
            field_tols = dict(TOL_FIELD_DEFAULTS)
            # Hermetic synthetic data has parquet at sample_size, so resize is no-op
            # -> video byte-identical.
            field_tols["video"] = TOL_VIDEO_NOOP_RESIZE
            failures += _compare_samples(s_live, s_cached, field_tols, label=f"ep{ep}")

    print(f"\n{'='*70}")
    if failures:
        print(_red(f"Test A FAILED: {len(failures)} failure(s)"))
        for f in failures:
            print(f"  - {f}")
        return 1
    print(_green("Test A PASSED"))
    return 0


# ---------------------------------------------------------------------------
# Test B: real-data spot-check
# ---------------------------------------------------------------------------


def test_real(args: argparse.Namespace) -> int:
    print(f"\n{'='*70}\nTest B: REAL-DATA SPOT-CHECK\n{'='*70}")
    print(f"  data_root  = {args.data_root}")
    print(f"  domain     = {args.domain}")
    print(f"  cache_dir  = {args.cache_dir}")
    print(f"  episodes   = {args.episodes}")

    if args.episodes:
        ep_indexes = [int(x) for x in args.episodes.split(",") if x.strip()]
    else:
        ep_indexes = list(range(args.limit))

    sample_size = tuple(args.sample_size)
    valid_cam = list(args.valid_cam)

    chunk = args.chunk
    n_previous = args.n_previous
    sample_n_frames = n_previous + chunk

    fix_sidx = args.fix_sidx
    fix_mem_idx = list(range(n_previous))

    # Detect resize triviality from a sample frame to pick the right tolerance.
    # The dataset constructor accepts either layout: data_root contains the
    # corpus dir directly (production yaml style) OR data_root + domain joins
    # to the corpus dir (preprocess-script style). We replicate that fallback
    # here so the test CLI matches whatever convention the user uses.
    print("\n  detecting resize-triviality...")
    domain_dir = os.path.join(args.data_root, args.domain)
    if os.path.isfile(os.path.join(domain_dir, "meta", "tasks.jsonl")):
        parquet_dir = os.path.join(domain_dir, "data", "chunk-000")
    elif os.path.isfile(os.path.join(args.data_root, "meta", "tasks.jsonl")):
        parquet_dir = os.path.join(args.data_root, "data", "chunk-000")
    else:
        raise FileNotFoundError(
            f"Could not find LeRobot meta/tasks.jsonl under {domain_dir} or {args.data_root}"
        )
    parquet_path = os.path.join(parquet_dir, f"episode_{ep_indexes[0]:06d}.parquet")
    df_head = pd.read_parquet(parquet_path)
    cell0 = df_head[valid_cam[0]].iloc[0]
    img0 = Image.open(io.BytesIO(cell0["bytes"]))
    orig_w, orig_h = img0.size
    print(f"  parquet image native (W,H) = ({orig_w}, {orig_h}); sample_size (H,W) = {sample_size}")
    is_resize_noop = (orig_h, orig_w) == sample_size
    field_tols = dict(TOL_FIELD_DEFAULTS)
    field_tols["video"] = TOL_VIDEO_NOOP_RESIZE if is_resize_noop else TOL_VIDEO_NONNOOP_RESIZE
    print(f"  resize is {'no-op' if is_resize_noop else 'non-trivial'}; "
          f"video tol = {field_tols['video']:.3e}")

    failures: List[str] = []
    for ep in ep_indexes:
        print(f"\n  --- episode {ep} ---")
        common = dict(
            data_roots=[args.data_root], domains=[args.domain],
            sample_size=sample_size, valid_cam=valid_cam,
            n_action=0, n_state=0,
            stat_file=args.stat_file, fix_epiidx=ep, fix_sidx=fix_sidx,
            fix_mem_idx=fix_mem_idx, chunk=chunk, n_previous=n_previous,
            sample_n_frames=sample_n_frames,
            valid_act_dim=args.valid_act_dim,
            valid_sta_dim=args.valid_sta_dim,
            action_type=args.action_type,
            action_space=args.action_space,
            arm_layout=args.arm_layout,
            read_hand_pose=args.read_hand_pose,
            pose_stats_path=args.pose_stats_path,
        )
        try:
            ds_live = _build_dataset_pair(cache_dir=None, **common)
            ds_cached = _build_dataset_pair(cache_dir=args.cache_dir, **common)
            s_live = ds_live[0]
            s_cached = ds_cached[0]
        except Exception:
            traceback.print_exc()
            failures.append(f"ep{ep}: dataset construction or sample fetch raised")
            continue
        failures += _compare_samples(s_live, s_cached, field_tols, label=f"ep{ep}")

    print(f"\n{'='*70}")
    if failures:
        print(_red(f"Test B FAILED: {len(failures)} failure(s)"))
        for f in failures:
            print(f"  - {f}")
        return 1
    print(_green("Test B PASSED"))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub_grp = p.add_mutually_exclusive_group(required=True)
    sub_grp.add_argument("--hermetic", action="store_true",
                         help="Run Test A: synthetic LeRobot dataset, in-process preprocess.")
    sub_grp.add_argument("--real", action="store_true",
                         help="Run Test B: real data + cache produced by preprocess_dex_vtam_cache.py.")

    # Test B args
    p.add_argument("--data-root", default=None)
    p.add_argument("--domain", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--sample-size", nargs=2, type=int, default=[192, 256])
    p.add_argument("--valid-cam", nargs="+", default=["head_img"])
    p.add_argument("--episodes", default="0,1,2",
                   help="Comma-separated episode indexes for Test B.")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--chunk", type=int, default=9)
    p.add_argument("--n-previous", type=int, default=8)
    p.add_argument("--fix-sidx", type=int, default=20)
    p.add_argument("--stat-file", default=None,
                   help="Optional stat_file path for Test B; if omitted, falls back "
                        "to data.utils.statistics.StatisticInfo.")
    p.add_argument("--valid-act-dim", type=int, default=None)
    p.add_argument("--valid-sta-dim", type=int, default=None)
    # Defaults keep Test B exactly as before; the relative / right-only corpora
    # need these to match their yaml (see the right-only example above).
    p.add_argument("--action-type", default="absolute",
                   choices=["absolute", "relative_eef_rot6d"])
    p.add_argument("--action-space", default="joint", choices=["joint", "eef"],
                   help="stats-key suffix; MUST match the stat_file's keys")
    p.add_argument("--arm-layout", default="bimanual",
                   choices=["bimanual", "right_only"])
    p.add_argument("--read-hand-pose", action="store_true",
                   help="also compare hand_pose (V_hand, T, 22) live vs cached; "
                        "V_hand follows --arm-layout")
    p.add_argument("--pose-stats-path", default=None)
    args = p.parse_args()

    if args.real and args.action_type == "relative_eef_rot6d" \
            and (args.valid_act_dim is not None or args.valid_sta_dim is not None):
        p.error("relative_eef_rot6d does not support --valid-act-dim/--valid-sta-dim")

    if args.hermetic:
        return test_hermetic()

    if args.real:
        if not all([args.data_root, args.domain, args.cache_dir]):
            p.error("--real requires --data-root --domain --cache-dir")
        return test_real(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
