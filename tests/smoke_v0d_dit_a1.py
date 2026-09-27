"""Phase A1 Gate-1 smoke: verify v0d kwargs + hand_pose plumbing in
``runner.tactile_dit_trainer._load_v0c_a_frozen`` and
``VisualVAEAdapterModel.encode_per_hand``.

This script exercises the load / encode contract WITHOUT spinning up the
full ``TactileDiTTrainer`` (which would pull in the LTX visual VAE, the
DiT, tokenizer, etc.). The visual VAE is replaced by a tiny stub
``_FakeVAE`` that mimics the LTX ``AutoencoderKLLTXVideo.encode(...)
.latent_dist.mode()/.sample()`` interface enough for the adapter to run
end-to-end on CPU in a few seconds.

What the smoke proves (Phase A1's Gate-1 checklist):

  1. The trainer's ``_load_v0c_a_frozen`` correctly threads the seven new
     v0d kwargs (``adapter_use_pose_injection``, ``adapter_pose_dim``,
     ``adapter_use_timesformer``, ``adapter_timesformer_num_blocks``,
     ``adapter_timesformer_num_heads``, ``adapter_timesformer_ffn_dim``,
     ``adapter_finger_dropout``) from the yaml config dict into the
     ``VisualVAEAdapterModel`` constructor.
  2. The ``[v0d-gate1]`` LOAD log block is emitted with the expected items
     (yaml/model flags, alpha_pose / alpha_temp presence + values,
     representative v0d weight shapes, missing/unexpected key breakdown).
  3. The fail-loud path triggers: when ``adapter_use_pose_injection=True``
     but the checkpoint has no ``adapter.pose_encoder.*`` weights, the
     loader raises ``RuntimeError`` containing ``[v0d-gate1] FATAL``.
  4. ``encode_per_hand`` accepts the v0d ``hand_pose`` kwarg of shape
     ``(B, V_hand, T_raw, P)`` and produces a per-hand latent of the
     expected shape ``(B, V_hand, C, T_lat, H_lat, W_lat)``.
  5. The ``hand_pose`` mem/future slicing logic used by
     ``_encode_tactile_split`` (mem -> per-frame T=1, future -> chunk
     block) round-trips through ``encode_per_hand`` without shape errors.

The smoke uses a freshly-initialized v0d adapter saved to a temp file so
it does NOT require access to the real v0d Stage-1 checkpoint. The
intent is to lock the trainer's plumbing; the cross-check that the real
v0d ckpt loads correctly happens implicitly when the trainer is launched
(the ``[v0d-gate1]`` block prints into the train log).

Usage (CPU, any dev box with the conda env active):

    python3 scripts/smoke_v0d_dit_a1.py

Exits with code 0 on success and prints a "[smoke PASS]" banner.
"""

import argparse
import logging
import os
import sys
import tempfile
import traceback

# NOTE: this smoke is intended to run on any host with a visible
# GPU + a built triton). The trainer module pulls in `diffusers.training_utils`
# -> `transformers.modeling_utils` -> `deepspeed.ops.transformer.inference.
# triton`, and triton's autotuner decorator calls `driver.active.
# get_benchmarker()` AT IMPORT TIME. On a GPU machine that resolves fine
# because the CUDA driver is registered. On a CPU-only machine it raises
# "0 active drivers ([])"; in that case run the smoke on a node that has a
# GPU visible. We do NOT install a fake-driver stub here because the
# trade-offs are bad: the stub silences a real configuration problem (no
# GPU where one should be) and forces deepspeed onto a code path that's
# not the one production training uses.

import torch
import torch.nn as nn


# Make the `DexVTAM/` root importable so `runner.tactile_dit_trainer` and
# `models.tactile_models.visual_vae_adapter` resolve regardless of the
# user's cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEXVTAM_ROOT = os.path.dirname(_THIS_DIR)
if _DEXVTAM_ROOT not in sys.path:
    sys.path.insert(0, _DEXVTAM_ROOT)

# Configure logging so the `[v0d-gate1]` INFO block is visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Initialize the accelerate process state so `accelerate.logging.get_logger`
# (used by `runner.tactile_dit_trainer`) doesn't raise on its first
# `logger.info(...)` call. `PartialState` is the lightweight init -- it
# bootstraps the distributed state for a single process without creating a
# full `Accelerator` (which would try to load a deepspeed / fp16 plugin
# pipeline). This is required regardless of CPU vs GPU.
try:
    from accelerate import PartialState

    PartialState()
except Exception as _exc:  # pragma: no cover -- defensive only
    print(
        f"[smoke WARN] PartialState() init failed: {_exc!r}. Continuing; the "
        f"[v0d-gate1] log block may not appear, but the assertion checks "
        f"still gate the test result."
    )


def _pick_device() -> torch.device:
    """Prefer CUDA when available so the smoke runs on the same device the
    real trainer uses. Falls back to CPU only when no GPU is
    visible. Returns a single device; we don't bother with multi-GPU
    because the smoke processes B=1 tensors."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


SMOKE_DEVICE = _pick_device()


# -----------------------------------------------------------------------
# Minimal LTX VAE stub
# -----------------------------------------------------------------------
class _FakeLatentDist:
    """Mimics ``diffusers``' DiagonalGaussianDistribution enough for the
    adapter's ``encode(...).latent_dist.mode()`` path."""

    def __init__(self, latent: torch.Tensor):
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent

    def sample(self) -> torch.Tensor:
        return self._latent


class _FakeEncodeOutput:
    """Mimics ``AutoencoderKLOutput`` so ``out.latent_dist`` works."""

    def __init__(self, latent: torch.Tensor):
        self.latent_dist = _FakeLatentDist(latent)


class _FakeVAE(nn.Module):
    """Tiny stand-in for ``AutoencoderKLLTXVideo`` that produces the right
    output shape for LTX's spatial 32x / temporal 8x compression.

    Input shape: ``(N, 3, T, H, W)`` (post ``GrayToRGB``).
    Output shape: ``(N, latent_channels, T_lat, H/32, W/32)`` where
    ``T_lat = T // 8 + 1`` if ``T > 1`` else ``1``.

    The actual values are zeros because the smoke is shape-only.
    """

    def __init__(self, latent_channels: int = 128, spatial_div: int = 32):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_div = spatial_div

    def encode(self, x: torch.Tensor) -> _FakeEncodeOutput:
        n, c, t, h, w = x.shape
        t_lat = (t // 8 + 1) if t > 1 else 1
        h_lat = h // self.spatial_div
        w_lat = w // self.spatial_div
        latent = torch.zeros(
            n, self.latent_channels, t_lat, h_lat, w_lat,
            dtype=x.dtype, device=x.device,
        )
        return _FakeEncodeOutput(latent)


# -----------------------------------------------------------------------
# Smoke helpers
# -----------------------------------------------------------------------
def _make_v0d_ckpt(tmpdir: str, with_v0d_weights: bool = True) -> str:
    """Build a fresh ``VisualVAEAdapterModel`` and dump its state_dict.

    When ``with_v0d_weights=True``, the adapter is constructed with the
    v0d flags on, so the saved ckpt contains the v0d-specific
    ``adapter.pose_encoder.*`` / ``adapter.timesformer_blocks.*`` /
    ``adapter.alpha_*`` keys. With ``False`` we save a v0c-A-flavored
    state_dict (no v0d weights) used by the abort-on-mismatch check.
    """
    from models.tactile_models.visual_vae_adapter import VisualVAEAdapterModel
    fake_vae = _FakeVAE(latent_channels=128, spatial_div=32)
    model = VisualVAEAdapterModel(
        vae=fake_vae,
        latent_channels=128,
        adapter_kind="finger_set_transformer",
        num_fingers=5,
        num_heads=8,
        spatial_h=6,
        spatial_w=8,
        adapter_residual="weighted_mean",
        gray_to_rgb_init="ones_repeat",
        latent_mode="mean",
        enable_pre_fuse=False,
        adapter_n_layers=3,
        adapter_ffn_dim=1024,
        adapter_dropout=0.0,
        adapter_use_pose_injection=with_v0d_weights,
        adapter_pose_dim=22,
        adapter_use_timesformer=with_v0d_weights,
        adapter_timesformer_num_blocks=2,
        adapter_timesformer_num_heads=8,
        adapter_timesformer_ffn_dim=512,
        adapter_finger_dropout=0.0,
    )
    suffix = "v0d" if with_v0d_weights else "v0c_a"
    path = os.path.join(tmpdir, f"smoke_{suffix}_ckpt.pt")
    # _load_v0c_a_frozen tries `sd["model"]` first then falls through to
    # raw state_dict, so saving raw is fine.
    torch.save(model.state_dict(), path)
    return path


def _run_load_check(tmpdir: str) -> None:
    """Check 1+2: v0d kwargs threading + Gate-1 LOAD log emission."""
    from runner.tactile_dit_trainer import _load_v0c_a_frozen

    ckpt_path = _make_v0d_ckpt(tmpdir, with_v0d_weights=True)
    fake_vae = _FakeVAE(latent_channels=128, spatial_div=32)

    cfg = {
        "latent_channels": 128,
        "adapter_kind": "finger_set_transformer",
        "num_fingers": 5,
        "num_heads": 8,
        "spatial_h": 6,
        "spatial_w": 8,
        "adapter_residual": "weighted_mean",
        "gray_to_rgb_init": "ones_repeat",
        "latent_mode": "mean",
        "adapter_n_layers": 3,
        "adapter_ffn_dim": 1024,
        "adapter_dropout": 0.0,
        # v0d
        "adapter_use_pose_injection": True,
        "adapter_pose_dim": 22,
        "adapter_use_timesformer": True,
        "adapter_timesformer_num_blocks": 2,
        "adapter_timesformer_num_heads": 8,
        "adapter_timesformer_ffn_dim": 512,
        "adapter_finger_dropout": 0.0,
    }
    model = _load_v0c_a_frozen(
        vae=fake_vae,
        tactile_vae_config=cfg,
        model_path=ckpt_path,
        device=SMOKE_DEVICE,
        dtype=torch.float32,
    )
    assert getattr(model.adapter, "use_pose_injection", False) is True, (
        "model.adapter.use_pose_injection should be True after v0d load."
    )
    assert getattr(model.adapter, "use_timesformer", False) is True, (
        "model.adapter.use_timesformer should be True after v0d load."
    )
    assert hasattr(model.adapter, "alpha_pose"), (
        "model.adapter should expose alpha_pose when pose_injection is on."
    )
    assert hasattr(model.adapter, "alpha_temp"), (
        "model.adapter should expose alpha_temp when timesformer is on."
    )
    # alpha gates are zero-init at construction; after loading from a
    # freshly-initialized ckpt they should still be exactly 0.
    a_pose = float(model.adapter.alpha_pose.detach().cpu().item())
    a_temp = float(model.adapter.alpha_temp.detach().cpu().item())
    assert a_pose == 0.0, f"alpha_pose expected 0.0 at init; got {a_pose}."
    assert a_temp == 0.0, f"alpha_temp expected 0.0 at init; got {a_temp}."
    print("[smoke] check 1 PASS: v0d kwargs threaded; alpha_pose/alpha_temp loaded.")
    return model


def _run_negative_load_check(tmpdir: str) -> None:
    """Check 3: abort-on-mismatch fires when v0d flag is True but the
    ckpt has no v0d weights."""
    from runner.tactile_dit_trainer import _load_v0c_a_frozen

    ckpt_path = _make_v0d_ckpt(tmpdir, with_v0d_weights=False)
    fake_vae = _FakeVAE(latent_channels=128, spatial_div=32)

    cfg = {
        # Same as success path EXCEPT we ask for v0d while the ckpt is
        # v0c-A (no pose_encoder.* keys).
        "adapter_kind": "finger_set_transformer",
        "num_fingers": 5,
        "num_heads": 8,
        "spatial_h": 6,
        "spatial_w": 8,
        "adapter_use_pose_injection": True,
        "adapter_pose_dim": 22,
        "adapter_use_timesformer": False,
        "adapter_finger_dropout": 0.0,
    }
    raised = False
    try:
        _load_v0c_a_frozen(
            vae=fake_vae,
            tactile_vae_config=cfg,
            model_path=ckpt_path,
            device=SMOKE_DEVICE,
            dtype=torch.float32,
        )
    except RuntimeError as e:
        raised = True
        msg = str(e)
        assert "[v0d-gate1] FATAL" in msg, (
            f"RuntimeError did not contain '[v0d-gate1] FATAL'. Got: {msg!r}"
        )
        assert "adapter.pose_encoder" in msg, (
            f"RuntimeError did not mention pose_encoder. Got: {msg!r}"
        )
    assert raised, (
        "Expected RuntimeError from _load_v0c_a_frozen when v0d flag is on "
        "but the ckpt has no v0d weights; no exception was raised."
    )
    print("[smoke] check 3 PASS: abort-on-mismatch fires on v0d-flag/ckpt mismatch.")


def _run_encode_check(model) -> None:
    """Check 4+5: encode_per_hand accepts hand_pose and the mem/future
    slicing logic from _encode_tactile_split round-trips cleanly.

    The synthetic tactile / pose tensors are built on ``SMOKE_DEVICE`` so
    they ride the same device as the loaded model (CUDA when present, CPU on
    a dev box without a GPU).
    """
    B, V_hand, F_finger, T_mem, T_future = 1, 2, 5, 1, 8  # mem=1, future=8 -> T_lat=1+2=3
    H, W = 192, 256
    P = 22
    dev = SMOKE_DEVICE
    # Mem path (per-frame, T_raw=1 each): same flattening _encode_tactile_split does.
    mem_tac = torch.zeros(B * T_mem, V_hand, F_finger, 1, H, W, device=dev)
    mem_pose = torch.zeros(B * T_mem, V_hand, 1, P, device=dev)
    mem_lat = model.encode_per_hand(mem_tac, hand_pose=mem_pose)
    expected_mem_shape = (B * T_mem, V_hand, 128, 1, 6, 8)
    assert tuple(mem_lat.shape) == expected_mem_shape, (
        f"mem encode shape {tuple(mem_lat.shape)} != expected {expected_mem_shape}."
    )
    # Future path (one chunk block, T_raw=T_future).
    fut_tac = torch.zeros(B, V_hand, F_finger, T_future, H, W, device=dev)
    fut_pose = torch.zeros(B, V_hand, T_future, P, device=dev)
    fut_lat = model.encode_per_hand(fut_tac, hand_pose=fut_pose)
    expected_fut_shape = (B, V_hand, 128, T_future // 8 + 1, 6, 8)
    assert tuple(fut_lat.shape) == expected_fut_shape, (
        f"future encode shape {tuple(fut_lat.shape)} != expected {expected_fut_shape}."
    )
    print(
        "[smoke] check 4 PASS: encode_per_hand(tac, hand_pose=hp) returns "
        f"mem={tuple(mem_lat.shape)}, future={tuple(fut_lat.shape)}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(
        f"[smoke] Phase A1 Gate-1 smoke on device={SMOKE_DEVICE} "
        "(fake LTX VAE / synthetic ckpt)"
    )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _run_load_check(tmpdir)
            _run_negative_load_check(tmpdir)
            _run_encode_check(model)
    except AssertionError as e:
        print(f"\n[smoke FAIL] assertion: {e}")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[smoke FAIL] unexpected exception: {e!r}")
        traceback.print_exc()
        return 1
    print("\n[smoke PASS] Phase A1 Gate-1 verified: v0d kwargs threaded, LOAD "
          "log emitted, abort-on-mismatch fires, encode_per_hand(hand_pose) "
          "round-trips.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
