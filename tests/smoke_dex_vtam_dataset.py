"""Stage 2 F3: structural smoke for ``DexVTAMDataset`` tactile read path.

Mirrors ``scripts/smoke_tactile_projector.py`` style: each test asserts a
single contract from the locked Stage-2 spec (Section 5: dataset). CPU-only,
no real LeRobot data needed -- a synthetic DataFrame is monkey-patched in
place of ``pd.read_parquet`` so the test is hermetic and runs in seconds on
any box that has pandas + torch + PIL.

Checks:

  1. Tactile field present + shape/dtype/range contract:
     sample["tactile"] is ``(V_hand=2, F=5, T=16, H=192, W=256) float32``
     with values in ``[-1, 1]``.
  2. Existing fields preserved (zero regression vs libero pipeline):
     sample has ``video, actions, caption, state`` with the original libero
     shapes/dtypes.
  3. ``read_tactile=False`` short-circuit:
     when constructed with ``read_tactile=False``, the sample dict has NO
     ``tactile`` key (strict GE baseline mode). This is required for
     byte-level reproducibility against vanilla GE training.
  4. Temporal alignment: tactile T dim equals video T dim (same vid_indexes
     used for both reads).

Run::

    python scripts/smoke_dex_vtam_dataset.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from data.dex_vtam_dataset import DexVTAMDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


SAMPLE_H, SAMPLE_W = 192, 256
TACTILE_V_HAND = 2
TACTILE_F = 5
ACTION_DIM = 7
TEST_DOMAIN = "test_domain"
TEST_CAM = "observation.images.front"
TOTAL_FRAMES = 30
ACTION_CHUNK = 4
N_PREVIOUS = 12
SAMPLE_N_FRAMES = 16  # n_previous + action_chunk
EXPECTED_T = 16  # mem_indexes(12) + video_end(4) when stride=1


def _make_synthetic_parquet_df(seed: int = 0) -> pd.DataFrame:
    """Build a minimal LeRobot-shaped DataFrame for one episode.

    Columns required by ``DexVTAMDataset.get_batch``:
      - ``action``: per-row ``(ACTION_DIM,)`` float32 array
      - ``observation.state``: per-row ``(ACTION_DIM,)`` float32 array
      - ``tactile``: per-row ``(V_hand, F, H, W)`` uint8 array
      - ``observation.images.front``: per-row dict with ``"bytes"`` PNG payload

    The tactile column intentionally uses dense uint8 ndarrays per row (not
    nested object arrays) -- ``_deep_stack`` handles both cases via its
    ``arr.dtype == object`` branch (falls through to ``np.asarray`` for
    dense leaves), so the shape contract is identical.
    """
    rng = np.random.default_rng(seed)

    tactile_rows = [
        rng.integers(0, 256, (TACTILE_V_HAND, TACTILE_F, SAMPLE_H, SAMPLE_W), dtype=np.uint8)
        for _ in range(TOTAL_FRAMES)
    ]

    cam_rows = []
    for _ in range(TOTAL_FRAMES):
        img_arr = rng.integers(0, 256, (SAMPLE_H, SAMPLE_W, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cam_rows.append({"bytes": buf.getvalue()})

    action_rows = [
        rng.uniform(-0.5, 0.5, ACTION_DIM).astype(np.float32) for _ in range(TOTAL_FRAMES)
    ]
    state_rows = [
        rng.uniform(-0.5, 0.5, ACTION_DIM).astype(np.float32) for _ in range(TOTAL_FRAMES)
    ]

    return pd.DataFrame(
        {
            "action": action_rows,
            "observation.state": state_rows,
            "tactile": tactile_rows,
            TEST_CAM: cam_rows,
        }
    )


def _make_dataset(read_tactile: bool = True) -> DexVTAMDataset:
    """Construct a ``DexVTAMDataset`` via ``__new__`` with manually-set attrs.

    Bypasses ``__init__`` (which scans a real LeRobot meta folder) so the
    test can run hermetically without any data on disk.
    """
    ds: DexVTAMDataset = DexVTAMDataset.__new__(DexVTAMDataset)

    ds.action_key = "action"
    ds.state_key = "observation.state"
    ds.action_type = "absolute"
    ds.action_space = "joint"
    ds.extra_parquet_index = False
    ds.valid_act_dim = None
    ds.valid_sta_dim = None
    ds.read_tactile = read_tactile
    ds.tactile_key = "tactile"

    ds.valid_cam = [TEST_CAM]
    ds.chunk = ACTION_CHUNK
    ds.action_chunk = ACTION_CHUNK
    ds.video_temporal_stride = 1
    ds.sample_n_frames = SAMPLE_N_FRAMES
    ds.n_previous = N_PREVIOUS
    ds.previous_pick_mode = "uniform"
    ds.fix_epiidx = None
    ds.fix_sidx = None
    ds.fix_mem_idx = None
    ds.ignore_seek = False
    ds.preprocess = "resize"
    ds.sample_size = (SAMPLE_H, SAMPLE_W)
    ds.random_crop = False

    ds.pixel_transforms_resize = transforms.Resize((SAMPLE_H, SAMPLE_W))
    ds.pixel_transforms_norm = transforms.Compose(
        [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
    )

    ds.dataset = [
        [
            "dummy_video.mp4",  # video_path (unused; video read goes via parquet img bytes)
            None,               # camera_info
            "dummy.parquet",    # parquet_path (intercepted by mock)
            TEST_DOMAIN,        # domain_name
            "",                 # domain_id
            None,               # task_info
            "test caption",     # caption
            TOTAL_FRAMES,       # length
        ]
    ]
    ds.length = 1

    ds.StatisticInfo = {
        TEST_DOMAIN + "_joint": {
            "mean": [0.0] * ACTION_DIM,
            "std": [1.0] * ACTION_DIM,
            "q01": [-1.0] * ACTION_DIM,
            "q99": [1.0] * ACTION_DIM,
        },
        TEST_DOMAIN + "_state_joint": {
            "mean": [0.0] * ACTION_DIM,
            "std": [1.0] * ACTION_DIM,
            "q01": [-1.0] * ACTION_DIM,
            "q99": [1.0] * ACTION_DIM,
        },
    }

    return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# 1. Tactile shape / dtype / range
# ---------------------------------------------------------------------------


def test_tactile_field_contract():
    _section("1] tactile field shape/dtype/range contract")
    ds = _make_dataset(read_tactile=True)
    fake_df = _make_synthetic_parquet_df(seed=0)

    with patch("pandas.read_parquet", return_value=fake_df):
        sample = ds[0]

    assert "tactile" in sample, (
        f"expected 'tactile' in sample (read_tactile=True); "
        f"got keys {list(sample.keys())}"
    )
    tac = sample["tactile"]
    assert isinstance(tac, torch.Tensor), f"tactile is {type(tac)}; expected Tensor"
    assert tac.dtype == torch.float32, f"tactile dtype {tac.dtype}; expected float32"

    expected_shape = (TACTILE_V_HAND, TACTILE_F, EXPECTED_T, SAMPLE_H, SAMPLE_W)
    assert tuple(tac.shape) == expected_shape, (
        f"tactile shape {tuple(tac.shape)}; expected {expected_shape}"
    )

    # Range: [-1, 1] (exact bounds because uint8/127.5 - 1.0 maps {0..255} -> {-1.0, 1.0})
    tac_min, tac_max = tac.min().item(), tac.max().item()
    assert -1.0 - 1e-6 <= tac_min <= 1.0 + 1e-6, f"tactile.min() = {tac_min}; expected >= -1"
    assert -1.0 - 1e-6 <= tac_max <= 1.0 + 1e-6, f"tactile.max() = {tac_max}; expected <= 1"

    print(f"  tactile.shape = {tuple(tac.shape)} OK")
    print(f"  tactile.dtype = {tac.dtype} OK")
    print(f"  tactile range = [{tac_min:.4f}, {tac_max:.4f}] OK")


# ---------------------------------------------------------------------------
# 2. Existing fields preserved
# ---------------------------------------------------------------------------


def test_existing_fields_preserved():
    _section("2] existing libero fields preserved (video, actions, caption, state)")
    ds = _make_dataset(read_tactile=True)
    fake_df = _make_synthetic_parquet_df(seed=1)

    with patch("pandas.read_parquet", return_value=fake_df):
        sample = ds[0]

    for key in ("video", "actions", "caption", "state"):
        assert key in sample, f"sample missing libero field '{key}'; keys = {list(sample.keys())}"

    # video: (c=3, v=1, T, H, W) per libero contract
    video = sample["video"]
    assert isinstance(video, torch.Tensor), f"video is {type(video)}"
    assert video.dtype == torch.float32, f"video dtype {video.dtype}"
    assert video.shape == (3, 1, EXPECTED_T, SAMPLE_H, SAMPLE_W), (
        f"video shape {tuple(video.shape)}; expected (3, 1, {EXPECTED_T}, {SAMPLE_H}, {SAMPLE_W})"
    )

    # actions: (T, 2*ACTION_DIM) -- libero concatenates [normalized_action, normalized_state]
    actions = sample["actions"]
    assert isinstance(actions, torch.Tensor), f"actions is {type(actions)}"
    assert actions.shape == (EXPECTED_T, 2 * ACTION_DIM), (
        f"actions shape {tuple(actions.shape)}; expected ({EXPECTED_T}, {2 * ACTION_DIM})"
    )

    assert sample["caption"] == "test caption", f"caption = {sample['caption']!r}"

    state = sample["state"]
    assert isinstance(state, torch.Tensor), f"state is {type(state)}"
    assert state.shape == (1, 2 * ACTION_DIM), f"state shape {tuple(state.shape)}"

    print(f"  video.shape   = {tuple(video.shape)} OK")
    print(f"  actions.shape = {tuple(actions.shape)} OK")
    print(f"  state.shape   = {tuple(state.shape)} OK")
    print(f"  caption       = {sample['caption']!r} OK")


# ---------------------------------------------------------------------------
# 3. read_tactile=False short-circuit
# ---------------------------------------------------------------------------


def test_read_tactile_false_omits_field():
    _section("3] read_tactile=False -> sample has no 'tactile' key (strict GE baseline)")
    ds = _make_dataset(read_tactile=False)
    fake_df = _make_synthetic_parquet_df(seed=2)

    with patch("pandas.read_parquet", return_value=fake_df):
        sample = ds[0]

    assert "tactile" not in sample, (
        f"read_tactile=False but sample still has 'tactile' key; "
        f"keys = {list(sample.keys())}"
    )

    # Strict equality vs libero: keys must be exactly (video, actions, caption, state)
    expected_keys = {"video", "actions", "caption", "state"}
    actual_keys = set(sample.keys())
    assert actual_keys == expected_keys, (
        f"read_tactile=False sample keys {actual_keys} != expected {expected_keys}"
    )
    print(f"  sample.keys() = {sorted(sample.keys())} OK")


# ---------------------------------------------------------------------------
# 5. Temporal alignment: tactile T == video T (same vid_indexes)
# ---------------------------------------------------------------------------


def test_video_tactile_temporal_alignment():
    _section("5] video and tactile share T dim (same vid_indexes used)")
    ds = _make_dataset(read_tactile=True)
    fake_df = _make_synthetic_parquet_df(seed=3)

    with patch("pandas.read_parquet", return_value=fake_df):
        sample = ds[0]

    video = sample["video"]
    tac = sample["tactile"]
    # video: (c, v, T, H, W); tactile: (V_hand, F, T, H, W)
    video_T = video.shape[2]
    tac_T = tac.shape[2]
    assert video_T == tac_T, (
        f"video T ({video_T}) != tactile T ({tac_T}); the two reads must use the "
        f"same vid_indexes."
    )
    print(f"  video.T = {video_T}, tactile.T = {tac_T} OK")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    print(f"torch={torch.__version__}  pandas={pd.__version__}")

    test_tactile_field_contract()
    test_existing_fields_preserved()
    test_read_tactile_false_omits_field()
    test_video_tactile_temporal_alignment()

    print("\nALL F3 SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
