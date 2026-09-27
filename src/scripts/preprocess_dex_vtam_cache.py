"""Offline preprocessing cache for ``DexVTAMDataset``.

Pre-decodes parquet frames, stacks tactile via ``_deep_stack``, applies the
same resize transform the dataset would at training time, and writes
per-episode mmap-friendly numpy arrays plus a global ``schema.json``.

Why this exists
---------------
At training time, ``DexVTAMDataset.get_batch`` does, per ``__getitem__``:

  * ``Image.open(io.BytesIO(...))`` per frame  (~1 ms x 16 frames/sample),
  * ``_deep_stack`` on the tactile object array (~5 ms x 16 frames),
  * a torchvision ``Resize`` per sample,
  * an ``np.stack`` over per-row action / state arrays.

With 4 ranks x 8 dataloader workers, this drives a strict period-8
head-of-line stall pattern (~21 s spike vs ~2 s steady-state) that pinned
phase 2 ETA at ~62 h. The fix mirrors stage-1 adapter's caching idea but
goes one step further: pre-bake the per-episode invariant work to disk so
each training ``__getitem__`` collapses to::

    arr = np.load(..., mmap_mode='r')[vid_indexes]   # zero-copy view
    tensor = torch.from_numpy(arr.copy()).float() / 255
    permute / normalize

OS pagecache then shares the bytes across the 32 dataloader worker
processes for free, sidestepping the in-memory cache strategy's
``num_workers x cache_size`` RAM blow-up.

Layout written
--------------
::

    <cache_dir>/
      schema.json
      episode_000000/
        video.npy        # uint8 (T_total, V_rgb, H_target, W_target, 3) post-resize
        tactile.npy      # uint8 (T_total, V_hand=2, F=5, 192, 256)
        action.npy       # float32 (T_total, action_dim) RAW, pre-normalize
        state.npy        # float32 (T_total, state_dim)  RAW, pre-normalize
        meta.json        # caption, total_frames, parquet_mtime/size, schema_version
      episode_000001/
        ...

``action.npy`` / ``state.npy`` are stored RAW (pre q01/q99 normalization)
because the stat file may change without re-extracting frames; the
normalization step is cheap enough to keep at training time.

Byte identity
-------------
Video cache uses the same ``transforms.Resize`` invocation the live
pipeline uses, but rounds back to ``uint8`` to keep disk small. If
``sample_size == original_resolution`` (which is the case for
right_hand_pick_cube: native 192x256, sample_size [192, 256]), the
``Resize`` is a no-op and cache is *byte-identical* to live. Otherwise
the cache introduces at most ``2/255`` max abs error per pixel after
the downstream ``Normalize(0.5, 0.5)`` (one quantization round-trip).

Resume
------
A successful episode is committed atomically: per-episode write goes to
``episode_NNNNNN.tmp/`` then renames to ``episode_NNNNNN/``. ``meta.json``
is the commit marker -- a re-run skips any episode whose ``meta.json``
plus all four ``.npy`` files already exist. Pass ``--overwrite`` to
force rebuild.

CLI examples
------------
::

    # Test: --limit 3 to a *_test cache_dir, num-workers low so we don't
    # fight phase 2's running dataloaders for CPU / file handles.
    python scripts/preprocess_dex_vtam_cache.py \\
      --data-root data/pick_cube \\
      --domain    lerobot_dataset_right_hand_pick_cube_100_episodes \\
      --cache-dir data/cache/right_hand_pick_cube_v1_test \\
      --sample-size 192 256 \\
      --preprocess resize \\
      --valid-cam head_img \\
      --tactile-key tactile \\
      --action-key actions \\
      --state-key state \\
      --num-workers 4 \\
      --limit 3 \\
      --overwrite

    # Production full: only after Tests A+B pass and phase 2 is killed.
    python scripts/preprocess_dex_vtam_cache.py \\
      ... \\
      --cache-dir data/cache/right_hand_pick_cube_v1 \\
      --num-workers 8

Note on parquet column names
----------------------------
The right_hand_pick_cube corpus uses non-standard short column names
(``head_img`` / ``tactile`` / ``actions`` / ``state``), NOT the LeRobot
standard ``observation.images.*`` / ``action`` / ``observation.state``.
The CLI must match the parquet, not the LeRobot convention.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Inlined to avoid importing data.tactile_dataset (which imports torch
# transitively from runner/, slowing each Pool worker startup on NFS).
# Bit-for-bit identical to data.tactile_dataset._deep_stack.
def _deep_stack(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return np.stack([_deep_stack(x) for x in arr])
    return np.asarray(arr)


SCHEMA_VERSION = "v1"


def load_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_resize_transform(sample_size: Tuple[int, int], preprocess: str):
    """Mirror ``DexVTAMDataset.pixel_transforms_resize`` exactly."""
    if preprocess == "center_crop_resize":
        return transforms.Compose([
            transforms.Resize(min(sample_size)),
            transforms.CenterCrop(sample_size),
        ])
    if preprocess == "resize":
        return transforms.Compose([
            transforms.Resize(sample_size),
        ])
    raise ValueError(f"unsupported preprocess: {preprocess!r}")


def episode_dir(cache_root: str, episode_index: int) -> str:
    return os.path.join(cache_root, f"episode_{episode_index:06d}")


def is_episode_done(ep_dir: str) -> bool:
    if not os.path.isfile(os.path.join(ep_dir, "meta.json")):
        return False
    for fname in ("video.npy", "tactile.npy", "action.npy", "state.npy"):
        if not os.path.isfile(os.path.join(ep_dir, fname)):
            return False
    return True


def is_flow_done(ep_dir: str) -> bool:
    if not os.path.isfile(os.path.join(ep_dir, "meta.json")):
        return False
    return os.path.isfile(os.path.join(ep_dir, "tactile_flow.npy"))


def process_one_episode(args: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple[int, str, Optional[str]]:
    """Decode one episode, write cache subdir, return (ep_idx, status, err)."""
    ep_info, schema = args
    episode_index = int(ep_info["episode_index"])
    parquet_path = ep_info["parquet_path"]
    caption = ep_info["caption"]
    cache_dir = schema["cache_dir"]

    ep_dir = episode_dir(cache_dir, episode_index)
    add_flow = bool(schema.get("add_flow", False))
    if (not schema.get("overwrite", False)) and is_episode_done(ep_dir):
        if (not add_flow) or is_flow_done(ep_dir):
            return episode_index, "skipped", None
        # Flow backfill path: keep existing cached arrays untouched; only
        # materialize tactile_flow.npy from parquet.
        try:
            df = pd.read_parquet(parquet_path)
            tactile_flow_key = str(schema.get("tactile_flow_key", "tactile_flow"))
            if tactile_flow_key not in df:
                raise KeyError(
                    f"column {tactile_flow_key!r} missing in {parquet_path}"
                )
            tactile_flow_rows = df[tactile_flow_key].to_list()
            tactile_flow = np.stack([_deep_stack(r) for r in tactile_flow_rows])
            if tactile_flow.dtype != np.float32:
                tactile_flow = tactile_flow.astype(np.float32)
            np.save(os.path.join(ep_dir, "tactile_flow.npy"), tactile_flow)
            return episode_index, "flow_backfilled", None
        except Exception as e:
            return episode_index, "error", f"{e!r}\n{traceback.format_exc()}"

    try:
        valid_cam: List[str] = list(schema["valid_cam"])
        tactile_key: str = schema["tactile_key"]
        tactile_flow_key: str = str(schema.get("tactile_flow_key", "tactile_flow"))
        action_key: str = schema["action_key"]
        state_key: str = schema["state_key"]
        sample_size: Tuple[int, int] = tuple(schema["sample_size"])  # (H, W)
        preprocess: str = schema["preprocess"]
        extra_idx: bool = bool(schema["extra_parquet_index"])

        df = pd.read_parquet(parquet_path)
        T_total = len(df)

        # action / state -- mirror DexVTAMDataset.get_batch lines 489-505
        if extra_idx:
            action = np.stack([df[action_key].iloc[i][0] for i in range(T_total)])
            state = np.stack([df[state_key].iloc[i][0] for i in range(T_total)])
        else:
            action = np.stack([df[action_key].iloc[i] for i in range(T_total)])
            state = np.stack([df[state_key].iloc[i] for i in range(T_total)])
        action = action.astype(np.float32)
        state = state.astype(np.float32)

        # tactile -- mirror lines 562-565
        tactile_rows = df[tactile_key].to_list()
        tactile = np.stack([_deep_stack(r) for r in tactile_rows])
        if tactile.dtype != np.uint8:
            tactile = tactile.astype(np.uint8)

        tactile_flow = None
        if bool(schema.get("add_flow", False)):
            if tactile_flow_key not in df:
                raise KeyError(
                    f"tactile_flow_key={tactile_flow_key!r} missing in parquet "
                    f"{parquet_path}"
                )
            tactile_flow_rows = df[tactile_flow_key].to_list()
            tactile_flow = np.stack([_deep_stack(r) for r in tactile_flow_rows])
            if tactile_flow.dtype != np.float32:
                tactile_flow = tactile_flow.astype(np.float32)

        # video per cam -- mirror lines 537-550
        # PIL decode: use a context manager + np.array (which copies the
        # raster) so the PIL Image, the underlying file handle, and the
        # BytesIO are released as soon as the per-frame loop body exits.
        # The previous implementation kept all T_total PIL objects (and
        # implicitly their backing BytesIO buffers) alive until the
        # outer np.stack ran, which under the 488 full-corpus run would
        # peak at ~2 GB of transient buffers per worker (488 episodes
        # x ~349 frames x ~12 KB/frame, x num_workers in flight).
        resize_t = build_resize_transform(sample_size, preprocess)
        per_cam = []
        for cam in valid_cam:
            cam_bytes = df[cam].to_list()
            frames: List[np.ndarray] = []
            for i in range(T_total):
                with Image.open(io.BytesIO(cam_bytes[i]["bytes"])) as img:
                    # np.array() forces a copy out of PIL's lazy buffer
                    # (vs np.asarray which shares memory with img.tobytes
                    # and could be invalidated when img.__exit__ runs).
                    frames.append(np.array(img.convert("RGB")))
            arr = np.stack(frames)                                          # (T, H_orig, W_orig, 3) uint8
            tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous().float() / 255.0
            tensor = resize_t(tensor)                                       # (3, T, H_target, W_target) [0,1]
            tensor_uint8 = (tensor.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
            cam_arr = tensor_uint8.permute(1, 2, 3, 0).cpu().numpy()        # (T, H, W, 3)
            per_cam.append(cam_arr)
        video = np.stack(per_cam, axis=1)                                   # (T, V_rgb, H, W, 3)

        H_t, W_t = sample_size
        V_rgb = len(valid_cam)
        if video.shape != (T_total, V_rgb, H_t, W_t, 3):
            raise RuntimeError(f"video shape mismatch: got {video.shape}, "
                               f"expected ({T_total}, {V_rgb}, {H_t}, {W_t}, 3)")
        if tactile.shape[0] != T_total:
            raise RuntimeError(f"tactile T mismatch: got {tactile.shape[0]}, expected {T_total}")
        if action.shape[0] != T_total:
            raise RuntimeError(f"action T mismatch: got {action.shape[0]}, expected {T_total}")
        if state.shape[0] != T_total:
            raise RuntimeError(f"state T mismatch: got {state.shape[0]}, expected {T_total}")

        # Atomic-ish commit: write to .tmp then rename. meta.json is the marker.
        tmp_dir = ep_dir + ".tmp"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        np.save(os.path.join(tmp_dir, "video.npy"), video)
        np.save(os.path.join(tmp_dir, "tactile.npy"), tactile)
        if tactile_flow is not None:
            np.save(os.path.join(tmp_dir, "tactile_flow.npy"), tactile_flow)
        np.save(os.path.join(tmp_dir, "action.npy"), action)
        np.save(os.path.join(tmp_dir, "state.npy"), state)

        meta = {
            "schema_version": SCHEMA_VERSION,
            "episode_index": int(episode_index),
            "caption": caption,
            "total_frames": int(T_total),
            "parquet_path": parquet_path,
            "parquet_mtime": os.path.getmtime(parquet_path),
            "parquet_size": os.path.getsize(parquet_path),
            "video_shape": list(video.shape),
            "tactile_shape": list(tactile.shape),
            "tactile_flow_shape": (
                list(tactile_flow.shape) if tactile_flow is not None else None
            ),
            "action_shape": list(action.shape),
            "state_shape": list(state.shape),
        }
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if os.path.isdir(ep_dir):
            shutil.rmtree(ep_dir)
        os.rename(tmp_dir, ep_dir)

        return episode_index, "done", None
    except Exception as e:
        return episode_index, "error", f"{e!r}\n{traceback.format_exc()}"


def main():
    p = argparse.ArgumentParser(description="Offline preprocess cache for DexVTAMDataset.")
    p.add_argument("--data-root", required=True,
                   help="Root containing <domain>/{meta,data}/...  or just meta/data/...")
    p.add_argument("--domain", required=True,
                   help="Domain folder name (used to namespace + as DexVTAMDataset 'domain' arg).")
    p.add_argument("--cache-dir", required=True,
                   help="Output directory; will contain schema.json + per-episode subdirs.")
    p.add_argument("--sample-size", nargs=2, type=int, required=True, metavar=("H", "W"),
                   help="Target (H, W) for the cached video; matches yaml data.sample_size.")
    p.add_argument("--preprocess", choices=["resize", "center_crop_resize"], default="resize")
    p.add_argument("--valid-cam", nargs="+", required=True,
                   help="One or more parquet image columns. NOTE: right_hand_pick_cube uses "
                        "short names (e.g. 'head_img'), not 'observation.images.head_img'.")
    p.add_argument("--tactile-key", default="tactile")
    p.add_argument("--tactile-flow-key", default="tactile_flow",
                   help="Parquet column name for optical flow cache backfill.")
    p.add_argument("--add-flow", action="store_true",
                   help="Also cache tactile_flow.npy (N,2,5,24,32,4) float32. "
                        "With existing cached episodes, backfills only flow file.")
    p.add_argument("--action-key", default="actions",
                   help="Parquet column name. right_hand_pick_cube uses 'actions' (plural).")
    p.add_argument("--state-key", default="state",
                   help="Parquet column name. right_hand_pick_cube uses 'state', not 'observation.state'.")
    p.add_argument("--extra-parquet-index", action="store_true",
                   help="Set if action/state arrays are [T, 1, C] instead of [T, C].")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N episodes from episodes.jsonl.")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated episode indexes (overrides --limit).")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild every selected episode even if its cache is committed.")
    args = p.parse_args()

    domain_dir = os.path.join(args.data_root, args.domain)
    if os.path.isfile(os.path.join(domain_dir, "meta", "tasks.jsonl")):
        meta_dir = os.path.join(domain_dir, "meta")
        data_dir = os.path.join(domain_dir, "data")
    elif os.path.isfile(os.path.join(args.data_root, "meta", "tasks.jsonl")):
        meta_dir = os.path.join(args.data_root, "meta")
        data_dir = os.path.join(args.data_root, "data")
    else:
        raise FileNotFoundError(
            f"Could not find LeRobot meta/tasks.jsonl under {domain_dir} or {args.data_root}"
        )

    with open(os.path.join(meta_dir, "info.json")) as f:
        info = json.load(f)
    chunks_size = int(info["chunks_size"])

    episodes_list = load_jsonl(os.path.join(meta_dir, "episodes.jsonl"))

    if args.episodes is not None:
        wanted = set(int(x) for x in args.episodes.split(",") if x.strip())
        if not wanted:
            raise ValueError("--episodes parsed to an empty set; pass a non-empty comma-separated list.")
        available = {int(e["episode_index"]) for e in episodes_list}
        missing = sorted(wanted - available)
        if missing:
            # Fail-loud: silently caching a subset of the requested
            # ids would let a stale yaml `episodes:` list look complete
            # at training time (the DexVTAMDataset A3 filter does the
            # same fail-loud check; mirror that contract here).
            extras = sorted(available - wanted)[:5]
            extras_hint = f" (corpus has e.g. {extras}...)" if extras else ""
            raise ValueError(
                f"--episodes requested {len(wanted)} ids but {len(missing)} are not in "
                f"meta/episodes.jsonl for domain={args.domain!r}: missing={missing}{extras_hint}. "
                f"Fix the --episodes list or pass a different --domain."
            )
        episodes_list = [e for e in episodes_list if int(e["episode_index"]) in wanted]
    elif args.limit is not None:
        episodes_list = episodes_list[: args.limit]

    ep_infos: List[Dict[str, Any]] = []
    for ep in episodes_list:
        episode_index = int(ep["episode_index"])
        episode_chunk = episode_index // chunks_size
        parquet_path = os.path.join(
            data_dir,
            f"chunk-{episode_chunk:03d}",
            f"episode_{episode_index:06d}.parquet",
        )
        if not os.path.exists(parquet_path):
            print(f"[WARN] parquet missing, skipping ep {episode_index}: {parquet_path}")
            continue
        tasks = ep["tasks"]
        caption = tasks[0] if isinstance(tasks, list) and tasks else str(tasks)
        ep_infos.append({
            "episode_index": episode_index,
            "parquet_path": parquet_path,
            "caption": caption,
        })

    os.makedirs(args.cache_dir, exist_ok=True)

    # Schema -- contract dataset will validate against.
    schema: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_size": list(args.sample_size),
        "preprocess": args.preprocess,
        "valid_cam": list(args.valid_cam),
        "tactile_key": args.tactile_key,
        "tactile_flow_key": args.tactile_flow_key,
        "add_flow": bool(args.add_flow),
        "action_key": args.action_key,
        "state_key": args.state_key,
        "extra_parquet_index": bool(args.extra_parquet_index),
        "video_layout": "T_V_H_W_C_uint8_post_resize",
        "tactile_layout": "T_Vhand_F_H_W_uint8_pre_normalize",
        "action_layout": "T_C_float32_raw_pre_normalize",
        "state_layout": "T_C_float32_raw_pre_normalize",
        "domain": args.domain,
        "data_root": args.data_root,
        "created_at": datetime.utcnow().isoformat() + "Z",
        # Per-run fields below; ignored when comparing existing schema.
        "cache_dir": args.cache_dir,
        "overwrite": bool(args.overwrite),
    }

    schema_path = os.path.join(args.cache_dir, "schema.json")
    if os.path.exists(schema_path) and not args.overwrite:
        with open(schema_path) as f:
            existing = json.load(f)
        ignore = {
            "created_at", "cache_dir", "overwrite", "data_root",
            # Added in R1-min; do not invalidate old schema-only caches.
            "add_flow", "tactile_flow_key",
        }
        diff_keys = sorted(
            k for k in set(schema) | set(existing)
            if k not in ignore and existing.get(k) != schema.get(k)
        )
        if diff_keys:
            raise RuntimeError(
                f"Existing schema.json mismatch on keys {diff_keys}.\n"
                f"  existing: {{ {', '.join(f'{k}={existing.get(k)!r}' for k in diff_keys)} }}\n"
                f"  new:      {{ {', '.join(f'{k}={schema.get(k)!r}'   for k in diff_keys)} }}\n"
                f"Use --overwrite or pick a different --cache-dir."
            )
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"[preprocess] {len(ep_infos)} episode(s) to process")
    print(f"[preprocess] cache_dir = {args.cache_dir}")
    print(f"[preprocess] schema    -> {schema_path}")
    print(f"[preprocess] num_workers = {args.num_workers}")

    t0 = time.time()
    if args.num_workers <= 1:
        results = [process_one_episode((ep, schema)) for ep in ep_infos]
    else:
        ctx_args = [(ep, schema) for ep in ep_infos]
        with Pool(args.num_workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(process_one_episode, ctx_args), 1):
                ep_idx, status, err = r
                if status == "error":
                    print(f"[preprocess][ERR  {i}/{len(ep_infos)}  ep={ep_idx}] {err.splitlines()[0]}")
                else:
                    print(f"[preprocess][{status:>7}  {i}/{len(ep_infos)}  ep={ep_idx}]")
                results.append(r)

    n_done = sum(1 for _, s, _ in results if s == "done")
    n_skip = sum(1 for _, s, _ in results if s == "skipped")
    n_err = sum(1 for _, s, _ in results if s == "error")
    elapsed = time.time() - t0
    print(f"[preprocess] FINISHED in {elapsed:.1f}s "
          f"| done={n_done} skipped={n_skip} error={n_err}")

    if n_err:
        for ep_idx, status, err in results:
            if status == "error":
                print(f"\n[preprocess][full traceback ep={ep_idx}]\n{err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
