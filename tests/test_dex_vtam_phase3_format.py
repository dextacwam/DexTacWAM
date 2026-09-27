"""Phase 3 (action_full) dataset format check.

Verifies that ``DexVTAMDataset`` -- as configured by a Stage-3 yaml such as
``configs/ltx_model/right_hand_pick_cube/action_model_right_hand_pick_cube_tactile.yaml``
-- produces samples whose shape / dtype / value-range / cat layout / state
prompt layout match what ``TactileDiTTrainer`` consumes on the
``train_mode='action_full'`` + ``return_action=True`` + ``add_state=True``
code path (``runner/tactile_dit_trainer.py`` lines ~1605-1719).

Phase 2 production (``train_mode='video_only'``) never executes that path,
so even though phase 2 trains fine the contract for phase 3 is unverified.
This is a CPU-only, ~30 s sanity check that runs without competing with
phase 2 for GPUs.

It does NOT exercise the model forward pass. The end-to-end model-side
verification is a separate, deferred step: after phase 2 finishes, run
the production yaml with ``train_steps=20`` override on 2 GPUs.

Usage::

    python scripts/test_dex_vtam_phase3_format.py \\
      --config configs/ltx_model/right_hand_pick_cube/action_model_right_hand_pick_cube_tactile.yaml

Optional overrides (handy when running on a host whose ``data_roots`` /
``cache_dir`` differ from the production yaml):

    python scripts/test_dex_vtam_phase3_format.py \\
      --config configs/ltx_model/right_hand_pick_cube/action_model_right_hand_pick_cube_tactile.yaml \\
      --data-root data/pick_cube \\
      --cache-dir data/cache/right_hand_pick_cube_v1_test \\
      --indexes 0,1,2

Exits 0 on all-pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dex_vtam_dataset import DexVTAMDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n[{title}]")


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    status = _green("OK  ") if cond else _red("FAIL")
    msg = f"  {status} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return cond


# ---------------------------------------------------------------------------
# Yaml loader
# ---------------------------------------------------------------------------


def _resolve_relative(path: Optional[str], yaml_path: str) -> Optional[str]:
    """Resolve a path that may be relative to the repo root.

    The yaml stores ``stat_file: configs/...`` as repo-relative; resolve so
    the test works regardless of cwd.
    """
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def load_yaml_data_train(
    yaml_path: str,
    data_root_override: Optional[str],
    cache_dir_override: Optional[str],
) -> Tuple[Dict[str, Any], int]:
    """Load ``data.train`` kwargs and ``diffusion_model.config.action_in_channels``."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    data_train = dict(cfg["data"]["train"])

    # Apply overrides.
    if data_root_override is not None:
        data_train["data_roots"] = [data_root_override]
    if cache_dir_override is not None:
        data_train["cache_dir"] = cache_dir_override

    # Resolve repo-relative paths.
    data_train["stat_file"] = _resolve_relative(
        data_train.get("stat_file"), yaml_path
    )

    # Speed up dataset construction: 1000x repeat is for training cadence,
    # not needed for a format check on a few indexes.
    data_train["repeat_dataset"] = 1

    action_in_channels = int(
        cfg["diffusion_model"]["config"]["action_in_channels"]
    )
    return data_train, action_in_channels


def load_action_state_dims(
    stat_file: str, domain: str, action_space: str,
    valid_act_dim: Optional[int], valid_sta_dim: Optional[int],
) -> Tuple[int, int]:
    """Read raw action_dim / state_dim from the stat file (q01 length).

    Honors valid_act_dim / valid_sta_dim if set in the yaml (clip).
    """
    with open(stat_file) as f:
        stats = json.load(f)
    action_key = f"{domain}_{action_space}"
    state_key = f"{domain}_state_{action_space}"
    if action_key not in stats:
        raise KeyError(
            f"stat_file missing entry {action_key!r}. Got keys: {list(stats)[:8]}..."
        )
    if state_key not in stats:
        raise KeyError(
            f"stat_file missing entry {state_key!r}. Got keys: {list(stats)[:8]}..."
        )
    act_dim = len(stats[action_key]["q01"])
    sta_dim = len(stats[state_key]["q01"])
    if valid_act_dim is not None:
        act_dim = int(valid_act_dim)
    if valid_sta_dim is not None:
        sta_dim = int(valid_sta_dim)
    return act_dim, sta_dim


# ---------------------------------------------------------------------------
# Per-sample contract checks
# ---------------------------------------------------------------------------


def _check_sample_contract(
    sample: Dict[str, Any],
    idx: int,
    expected: Dict[str, Any],
) -> List[str]:
    """Return a list of failure messages (empty == all pass) for one sample."""
    failures: List[str] = []

    n_previous = expected["n_previous"]
    chunk = expected["chunk"]
    action_chunk = expected["action_chunk"]
    ignore_seek = expected["ignore_seek"]
    H, W = expected["sample_size"]
    V_rgb = expected["V_rgb"]
    action_in = expected["action_in_channels"]
    act_dim = expected["action_dim"]
    sta_dim = expected["state_dim"]
    has_tactile = expected["has_tactile"]

    # ---- (1) Shape & dtype ----
    actions = sample["actions"]
    state = sample["state"]
    video = sample["video"]
    caption = sample["caption"]

    expected_T_action = n_previous + action_chunk
    # ignore_seek=True (Stage-3 libero convention): video clip = mem + mem[-1:]
    # i.e. T_video = n_previous + 1. Future video frames are not loaded
    # because the action_full model predicts actions, not video.
    expected_T_video = n_previous + (1 if ignore_seek else chunk)

    ok = _check(
        f"idx={idx} actions.shape == ({expected_T_action}, {action_in})",
        tuple(actions.shape) == (expected_T_action, action_in)
        and actions.dtype == torch.float32,
        detail=f"got shape={tuple(actions.shape)} dtype={actions.dtype}",
    )
    if not ok:
        failures.append(f"actions shape/dtype")

    ok = _check(
        f"idx={idx} state.shape   == (1, {action_in})",
        tuple(state.shape) == (1, action_in) and state.dtype == torch.float32,
        detail=f"got shape={tuple(state.shape)} dtype={state.dtype}",
    )
    if not ok:
        failures.append(f"state shape/dtype")

    ok = _check(
        f"idx={idx} video.shape   == (3, {V_rgb}, {expected_T_video}, {H}, {W})",
        tuple(video.shape) == (3, V_rgb, expected_T_video, H, W)
        and video.dtype == torch.float32,
        detail=f"got shape={tuple(video.shape)} dtype={video.dtype}",
    )
    if not ok:
        failures.append(f"video shape/dtype")

    if has_tactile:
        if "tactile" not in sample:
            failures.append(f"tactile field missing in sample")
            _check(f"idx={idx} sample has tactile field", False)
        else:
            tactile = sample["tactile"]
            n_hand = expected["n_tactile_hands"]
            n_finger = expected["n_tactile_fingers"]
            ok = _check(
                f"idx={idx} tactile.shape == ({n_hand}, {n_finger}, {expected_T_video}, {H}, {W})",
                tuple(tactile.shape)
                == (n_hand, n_finger, expected_T_video, H, W)
                and tactile.dtype == torch.float32,
                detail=f"got shape={tuple(tactile.shape)} dtype={tactile.dtype}",
            )
            if not ok:
                failures.append(f"tactile shape/dtype")

    ok = _check(
        f"idx={idx} caption is str",
        isinstance(caption, str) and len(caption) > 0,
        detail=f"got type={type(caption).__name__}",
    )
    if not ok:
        failures.append(f"caption type")

    # ---- (2) Value range ----
    # q01/q99 normalize maps the [q01, q99] band to [-1, 1] but does not
    # clip outliers; some action dims (esp. quaternion / wrist columns
    # with near-zero range) can spike to 10-50x. We just sanity-check
    # for non-NaN / non-Inf and a generous absolute cap.
    actions_max = actions.abs().max().item()
    ok = _check(
        f"idx={idx} actions finite, abs.max() < 200 (q01/q99 norm, outliers OK)",
        bool(torch.isfinite(actions).all()) and actions_max < 200.0,
        detail=f"abs.max()={actions_max:.4f}",
    )
    if not ok:
        failures.append(f"actions value range")

    state_max = state.abs().max().item()
    ok = _check(
        f"idx={idx} state   finite, abs.max() < 200",
        bool(torch.isfinite(state).all()) and state_max < 200.0,
        detail=f"abs.max()={state_max:.4f}",
    )
    if not ok:
        failures.append(f"state value range")

    ok = _check(
        f"idx={idx} video   in [-1, 1]",
        bool(video.abs().max().item() <= 1.0 + 1e-6),
        detail=f"abs.max()={video.abs().max().item():.4f}",
    )
    if not ok:
        failures.append(f"video value range")

    if has_tactile and "tactile" in sample:
        tactile = sample["tactile"]
        ok = _check(
            f"idx={idx} tactile in [-1, 1]",
            bool(tactile.abs().max().item() <= 1.0 + 1e-6),
            detail=f"abs.max()={tactile.abs().max().item():.4f}",
        )
        if not ok:
            failures.append(f"tactile value range")

    # ---- (3) cat structure: actions[:, -sta_dim:] is the state portion ----
    # The dataset does action = cat(action_norm, state_norm, dim=1). So the
    # last sta_dim columns must (a) live in roughly [-1, 1] and (b) match
    # state[0, -sta_dim:] at the n_previous-1 row by construction (both come
    # from the same indexes[n_previous-1] timestep through the same
    # q01/q99 normalize).
    ok = _check(
        f"idx={idx} action_in_channels == act_dim + sta_dim ({act_dim}+{sta_dim}={act_dim+sta_dim})",
        action_in == act_dim + sta_dim,
        detail=f"yaml action_in_channels={action_in}, stat_file act+sta={act_dim+sta_dim}",
    )
    if not ok:
        failures.append(f"action_in_channels mismatch with stat_file dims")

    state_part_in_actions = actions[n_previous - 1, -sta_dim:]
    state_part_in_state = state[0, -sta_dim:]
    state_cat_err = float((state_part_in_actions - state_part_in_state).abs().max().item())
    ok = _check(
        f"idx={idx} actions[n_prev-1, -sta_dim:] ~= state[0, -sta_dim:]",
        state_cat_err < 1e-5,
        detail=f"max abs err = {state_cat_err:.3e}",
    )
    if not ok:
        failures.append(f"state cat consistency (err={state_cat_err})")

    # ---- (4) state prompt structure: state[0, :act_dim] == 0 ----
    state_prompt_zeros = state[0, :act_dim]
    zeros_max = float(state_prompt_zeros.abs().max().item())
    ok = _check(
        f"idx={idx} state[0, :{act_dim}] (action-history bootstrap) == zeros",
        zeros_max == 0.0,
        detail=f"abs.max()={zeros_max:.3e}",
    )
    if not ok:
        failures.append(f"state prompt zero-init (max abs={zeros_max})")

    # ---- (5) Trainer slicing mock (line 1606-1616) ----
    # batch dim faked with unsqueeze(0).
    a_b = actions.unsqueeze(0)
    s_b = state.unsqueeze(0)
    a_slice = a_b[:, -action_chunk:]  # noqa: future regression target
    if s_b.shape[1] != 1:
        s_slice = s_b[:, n_previous - 1 : n_previous]
    else:
        s_slice = s_b
    ok = _check(
        f"idx={idx} batch['actions'][:, -action_chunk:].shape == (1, {action_chunk}, {action_in})",
        tuple(a_slice.shape) == (1, action_chunk, action_in),
        detail=f"got {tuple(a_slice.shape)}",
    )
    if not ok:
        failures.append(f"trainer action slice shape")

    ok = _check(
        f"idx={idx} act_state slice .shape         == (1, 1, {action_in})",
        tuple(s_slice.shape) == (1, 1, action_in),
        detail=f"got {tuple(s_slice.shape)}",
    )
    if not ok:
        failures.append(f"trainer act_state slice shape")

    return failures


# ---------------------------------------------------------------------------
# DataLoader collate check
# ---------------------------------------------------------------------------


def _check_collate_contract(
    ds: DexVTAMDataset,
    indexes: List[int],
    expected: Dict[str, Any],
) -> List[str]:
    """Build a small DataLoader, pull one batch, assert (B, ...) collate shapes."""
    failures: List[str] = []

    n_previous = expected["n_previous"]
    chunk = expected["chunk"]
    action_chunk = expected["action_chunk"]
    ignore_seek = expected["ignore_seek"]
    H, W = expected["sample_size"]
    V_rgb = expected["V_rgb"]
    action_in = expected["action_in_channels"]
    has_tactile = expected["has_tactile"]
    n_hand = expected.get("n_tactile_hands")
    n_finger = expected.get("n_tactile_fingers")
    expected_T_action = n_previous + action_chunk
    expected_T_video = n_previous + (1 if ignore_seek else chunk)

    B = min(2, len(indexes))
    sub_indexes = indexes[:B]

    # Custom Subset so DataLoader iterates only the indexes we pre-selected
    # (skips the rng inside __getitem__'s while-True retry on accidental
    # cache misses for some idx not present in the cache subset).
    class _Subset(torch.utils.data.Dataset):
        def __init__(self, base, idxs):
            self.base = base
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            return self.base[self.idxs[i]]

    sub = _Subset(ds, sub_indexes)
    loader = torch.utils.data.DataLoader(
        sub, batch_size=B, num_workers=0, shuffle=False
    )
    batch = next(iter(loader))

    ok = _check(
        f"DataLoader B={B} batch['actions'].shape == ({B}, {expected_T_action}, {action_in})",
        tuple(batch["actions"].shape) == (B, expected_T_action, action_in)
        and batch["actions"].dtype == torch.float32,
        detail=f"got {tuple(batch['actions'].shape)} dtype={batch['actions'].dtype}",
    )
    if not ok:
        failures.append("collate actions shape")

    ok = _check(
        f"DataLoader B={B} batch['state'].shape   == ({B}, 1, {action_in})",
        tuple(batch["state"].shape) == (B, 1, action_in)
        and batch["state"].dtype == torch.float32,
        detail=f"got {tuple(batch['state'].shape)} dtype={batch['state'].dtype}",
    )
    if not ok:
        failures.append("collate state shape")

    ok = _check(
        f"DataLoader B={B} batch['video'].shape   == ({B}, 3, {V_rgb}, {expected_T_video}, {H}, {W})",
        tuple(batch["video"].shape)
        == (B, 3, V_rgb, expected_T_video, H, W)
        and batch["video"].dtype == torch.float32,
        detail=f"got {tuple(batch['video'].shape)} dtype={batch['video'].dtype}",
    )
    if not ok:
        failures.append("collate video shape")

    if has_tactile:
        if "tactile" not in batch:
            failures.append("collate tactile missing")
            _check("DataLoader batch has tactile field", False)
        else:
            ok = _check(
                f"DataLoader B={B} batch['tactile'].shape == ({B}, {n_hand}, {n_finger}, {expected_T_video}, {H}, {W})",
                tuple(batch["tactile"].shape)
                == (B, n_hand, n_finger, expected_T_video, H, W)
                and batch["tactile"].dtype == torch.float32,
                detail=f"got {tuple(batch['tactile'].shape)} dtype={batch['tactile'].dtype}",
            )
            if not ok:
                failures.append("collate tactile shape")

    captions = batch["caption"]
    ok = _check(
        f"DataLoader B={B} batch['caption'] is list[str] len={B}",
        isinstance(captions, list) and len(captions) == B
        and all(isinstance(c, str) for c in captions),
        detail=f"type={type(captions).__name__} len={len(captions) if hasattr(captions, '__len__') else '?'}",
    )
    if not ok:
        failures.append("collate caption type/len")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Stage-3 yaml (e.g. action_model_right_hand_pick_cube_tactile.yaml).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data.train.data_roots[0] (single root only). Useful "
        "when running on a host whose dataset path differs from the yaml.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override data.train.cache_dir. Useful for pointing at the "
        "_test (limit-3) cache on a non-prod host.",
    )
    parser.add_argument(
        "--indexes",
        default="0,100,500,1500",
        help="Comma-separated dataset indexes to spot-check (default 4).",
    )
    parser.add_argument(
        "--n-tactile-hands",
        type=int,
        default=2,
        help="Expected number of tactile hands per sample (default 2 for "
        "right_hand_pick_cube; v0c-A bundles {left, right}).",
    )
    parser.add_argument(
        "--n-tactile-fingers",
        type=int,
        default=5,
        help="Expected number of tactile fingers per hand bundle (default 5).",
    )
    args = parser.parse_args()

    yaml_path = os.path.abspath(args.config)
    if not os.path.isfile(yaml_path):
        print(f"ERROR: yaml not found: {yaml_path}", file=sys.stderr)
        return 1

    indexes = [int(x.strip()) for x in args.indexes.split(",") if x.strip()]
    if not indexes:
        print("ERROR: --indexes is empty", file=sys.stderr)
        return 1

    print("=" * 72)
    print("phase 3 dataset format check")
    print("=" * 72)
    print(f"  config       = {yaml_path}")

    data_train, action_in_channels = load_yaml_data_train(
        yaml_path, args.data_root, args.cache_dir
    )

    print(f"  data_roots   = {data_train['data_roots']}")
    print(f"  domain       = {data_train['domains'][0]}")
    print(f"  cache_dir    = {data_train.get('cache_dir')}")
    print(f"  sample_size  = {data_train['sample_size']}")
    print(f"  chunk        = {data_train['chunk']}")
    print(f"  action_chunk = {data_train['action_chunk']}")
    print(f"  n_previous   = {data_train['n_previous']}")
    print(f"  valid_cam    = {data_train['valid_cam']}")
    print(f"  action_in    = {action_in_channels} (from diffusion_model.config)")

    domain = data_train["domains"][0]
    action_space = data_train.get("action_space", "joint")
    act_dim, sta_dim = load_action_state_dims(
        data_train["stat_file"], domain, action_space,
        data_train.get("valid_act_dim"), data_train.get("valid_sta_dim"),
    )
    print(f"  action_dim   = {act_dim} (raw, from stat_file q01)")
    print(f"  state_dim    = {sta_dim} (raw, from stat_file q01)")

    # ``ignore_seek=True`` (libero action_model convention) bypasses the
    # chunked future-video sampling: ``frame_indexes = mem_indexes +
    # mem_indexes[-1:]`` -> T_video = n_previous + 1, regardless of
    # ``chunk``. With ``ignore_seek=False`` (GE world-model phase 2):
    # T_video = n_previous + chunk.
    ignore_seek = bool(data_train.get("ignore_seek", False))
    expected = {
        "n_previous": data_train["n_previous"],
        "chunk": data_train["chunk"],
        "action_chunk": data_train["action_chunk"],
        "ignore_seek": ignore_seek,
        "sample_size": tuple(data_train["sample_size"]),
        "V_rgb": len(data_train["valid_cam"]),
        "action_in_channels": action_in_channels,
        "action_dim": act_dim,
        "state_dim": sta_dim,
        "has_tactile": bool(data_train.get("read_tactile", True)),
        "n_tactile_hands": args.n_tactile_hands,
        "n_tactile_fingers": args.n_tactile_fingers,
    }
    print(f"  ignore_seek  = {ignore_seek}")

    _section("constructing DexVTAMDataset")
    ds = DexVTAMDataset(**data_train)
    print(f"  dataset length = {len(ds)} (unique episodes; repeat_dataset forced to 1 for the test)")
    print(f"  cache_dir live? {ds.cache_dir is not None}")

    all_failures: List[str] = []

    _section("per-sample contract")
    for idx in indexes:
        if idx >= len(ds):
            print(_red(f"  SKIP idx={idx}: dataset length is only {len(ds)}"))
            continue
        try:
            sample = ds[idx]
        except Exception as e:
            print(_red(f"  FAIL idx={idx}: __getitem__ raised {type(e).__name__}: {e}"))
            traceback.print_exc()
            all_failures.append(f"idx={idx} __getitem__ raised")
            continue
        failures = _check_sample_contract(sample, idx, expected)
        all_failures.extend([f"idx={idx} {f}" for f in failures])

    _section("DataLoader collate (B=2, num_workers=0)")
    try:
        valid_idxs = [i for i in indexes if i < len(ds)]
        if len(valid_idxs) >= 2:
            failures = _check_collate_contract(ds, valid_idxs, expected)
            all_failures.extend([f"collate {f}" for f in failures])
        else:
            print(_red("  SKIP: <2 valid indexes for batch_size=2"))
    except Exception as e:
        print(_red(f"  FAIL collate: {type(e).__name__}: {e}"))
        traceback.print_exc()
        all_failures.append(f"collate raised")

    print("\n" + "=" * 72)
    if not all_failures:
        print(_green("ALL CHECKS PASSED"))
        print(
            "  Phase 3 (action_full) trainer can consume samples produced by "
            "this dataset config. dataset shape contract verified end-to-end "
            "for actions/state/video/tactile/caption + DataLoader collate + "
            "trainer slicing mock."
        )
        return 0
    print(_red(f"{len(all_failures)} CHECK(S) FAILED:"))
    for f in all_failures:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
