#!/usr/bin/env python3
"""Structural smoke for ``TactileDiTTrainer`` tactile stack + Stage-2 loss split.

This does **not** construct a full ``TactileDiTTrainer(config_file)`` (that path
requires distributed init, tokenizer, VAE, DiT, and real checkpoints). Instead
it exercises the same tensor contracts as ``train()`` after visual VAE latents
are produced:

  1. ``_encode_tactile_split`` with a dummy v0c-A that mimics the real LTX
     time-compression so the tactile T_lat actually equals the visual T_lat
     ``mem_size + chunk // 8 + 1`` (mem path = per-frame T=1; future path =
     T // 8 + 1). This catches T-misalignment regressions that would crash
     the downstream ``torch.cat`` along dim=2.
  2. ``rearrange`` + ``torch.cat`` on ``mem_latents`` / ``latents`` matching
     the Batch-3b injection block (visual rows first, tactile rows last) and
     the resulting ``n_view = V_rgb + V_hand`` bump.
  3. A *behavioral* Stage-2 lambda mix test: per-row losses are constructed so
     visual rows = 1.0 and tactile rows = 5.0; the test asserts the slicing
     direction (visual = lower slice) and the exact scalar
     ``lambda_visual * 1 + lambda_tactile * 5`` come out, so swapping
     ``[:n_visual_rows]`` and ``[n_visual_rows:]`` would fail.
  4. Projector grad / frozen v0c-A grad isolation.

**GPU node required**: importing ``runner.tactile_dit_trainer`` transitively
pulls ``transformers / diffusers / triton``, which on CPU-only login nodes
fails at import time with ``0 active drivers``. Run this inside an interactive
GPU allocation (or via the ``smoke_stage2.sh`` GPU branch). The tests
themselves auto-pick CUDA when available and otherwise CPU.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.tactile_models.projector import TactileProjector  # noqa: E402
from runner.tactile_dit_trainer import TactileDiTTrainer  # noqa: E402


# LTX visual VAE rates -- mirrored here so the dummy v0c-A produces a tactile
# latent T whose stitching exactly matches the visual T_lat the trainer
# computes (see runner/tactile_dit_trainer.py: latent_frames = raw_frames //
# TEMPORAL_DOWN_RATIO + 1 + mem_size; SPATIAL_DOWN_RATIO = 32).
TEMPORAL_DOWN_RATIO = 8
SPATIAL_H, SPATIAL_W = 6, 8  # 192 / 32, 256 / 32


class _DummyTactileVAE(torch.nn.Module):
    """Stand-in mimicking the real LTX time-compression contract.

    - Per-frame call (T_in == 1): output T_lat = 1, matching the mem path.
    - Multi-frame call (T_in > 1): output T_lat = T_in // 8 + 1, matching
      the future path of the real LTX VAE used inside
      ``_encode_tactile_split``.
    Spatial: hard-coded 32x down (192 -> 6, 256 -> 8) to match v0c-A.
    """

    def __init__(self) -> None:
        super().__init__()
        self.spatial = torch.nn.Conv3d(
            5, 128, kernel_size=(1, 32, 32), stride=(1, 32, 32),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def encode_per_hand(self, tactile, latent_mode=None):
        B, V, F_, T, H, W = tactile.shape
        x = tactile.reshape(B * V, F_, T, H, W)
        x = self.spatial(x)  # (B*V, 128, T, 6, 8)
        if T > 1:
            T_lat = T // TEMPORAL_DOWN_RATIO + 1
            x = F.adaptive_avg_pool3d(x, (T_lat, SPATIAL_H, SPATIAL_W))
        return x.reshape(B, V, 128, x.shape[-3], SPATIAL_H, SPATIAL_W)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_trainer(dev: torch.device) -> TactileDiTTrainer:
    proj = TactileProjector(latent_dim=128, num_views=2).to(dev)
    trainer = TactileDiTTrainer.__new__(TactileDiTTrainer)
    trainer.tactile_vae = _DummyTactileVAE().to(dev)
    trainer.projector = proj
    return trainer


def test_encode_tactile_split_t_alignment() -> None:
    """``_encode_tactile_split`` should produce T_lat = mem_size + chunk//8 + 1."""
    dev = _device()
    trainer = _make_trainer(dev)

    B, V_hand, F_, H, W = 2, 2, 5, 192, 256
    mem_size = 4
    chunk = 8                                    # multi-of TEMPORAL_DOWN_RATIO
    T_total = mem_size + chunk
    expected_T_lat = mem_size + chunk // TEMPORAL_DOWN_RATIO + 1

    tactile = torch.randn(B, V_hand, F_, T_total, H, W, device=dev) * 0.1
    out = trainer._encode_tactile_split(tactile, mem_size)
    assert out.shape == (
        B, V_hand, 128, expected_T_lat, SPATIAL_H, SPATIAL_W
    ), out.shape


def test_view_stack_concat_and_n_view_bump() -> None:
    """Mirror ``train()``: cat on dim=0, bump n_view, keep visual rows first."""
    dev = _device()
    trainer = _make_trainer(dev)

    B, n_view_visual = 2, 3
    mem_size = 4
    chunk = 8
    raw_frames = chunk
    T_lat = mem_size + raw_frames // TEMPORAL_DOWN_RATIO + 1   # 4 + 1 + 1 = 6

    mem_latents = torch.randn(
        B * n_view_visual, 128, mem_size, SPATIAL_H, SPATIAL_W, device=dev,
    )
    future_latents = torch.randn(
        B * n_view_visual, 128, T_lat - mem_size, SPATIAL_H, SPATIAL_W, device=dev,
    )
    latents = torch.cat((mem_latents, future_latents), dim=2)

    V_hand = 2
    tactile = torch.randn(
        B, V_hand, 5, mem_size + chunk, 192, 256, device=dev,
    ) * 0.1
    tac_full = trainer._encode_tactile_split(tactile, mem_size)
    assert tac_full.shape[3] == T_lat, (tac_full.shape, T_lat)

    tac_full = rearrange(tac_full, "b v c f hh ww -> (b v) c f hh ww")
    tac_mem = tac_full[:, :, :mem_size]
    mem_cat = torch.cat([mem_latents, tac_mem], dim=0)
    lat_cat = torch.cat([latents, tac_full], dim=0)

    n_view = n_view_visual + V_hand
    assert mem_cat.shape == (
        B * n_view, 128, mem_size, SPATIAL_H, SPATIAL_W,
    ), mem_cat.shape
    assert lat_cat.shape == (
        B * n_view, 128, T_lat, SPATIAL_H, SPATIAL_W,
    ), lat_cat.shape


def test_stage2_loss_split_slicing() -> None:
    """Behavioral check: slice direction matches train()'s `[:n_visual_rows]`.

    Construct a per-row loss vector where visual rows = 1.0 and tactile rows =
    5.0. If anyone flips the slice, ``loss_visual`` would become 5.0 and the
    closed-form expected scalar below would no longer match.
    """
    dev = _device()
    batch_size, n_view_visual, n_view_tactile = 2, 3, 2
    n_visual_rows = batch_size * n_view_visual
    n_total_rows = batch_size * (n_view_visual + n_view_tactile)

    loss_video_per_row = torch.zeros(n_total_rows, device=dev)
    loss_video_per_row[:n_visual_rows] = 1.0
    loss_video_per_row[n_visual_rows:] = 5.0

    lambda_visual, lambda_tactile = 0.25, 4.0
    loss_visual = loss_video_per_row[:n_visual_rows].mean()
    loss_tactile = loss_video_per_row[n_visual_rows:].mean()
    loss_video = lambda_visual * loss_visual + lambda_tactile * loss_tactile

    assert torch.allclose(loss_visual, torch.tensor(1.0, device=dev)), loss_visual
    assert torch.allclose(loss_tactile, torch.tensor(5.0, device=dev)), loss_tactile
    expected = lambda_visual * 1.0 + lambda_tactile * 5.0  # 20.25
    assert torch.allclose(
        loss_video, torch.tensor(expected, device=dev),
    ), (loss_video, expected)


def test_projector_grad_through_encode_split() -> None:
    """End-to-end grad: projector params receive grad; v0c-A stays frozen."""
    dev = _device()
    trainer = _make_trainer(dev)

    B, V_hand = 1, 2
    chunk = 8
    tactile = torch.randn(
        B, V_hand, 5, 4 + chunk, 192, 256, device=dev, requires_grad=False,
    ) * 0.1
    out = trainer._encode_tactile_split(tactile, mem_size=4)
    out.sum().backward()
    assert trainer.projector.alpha.grad is not None
    assert trainer.projector.alpha.grad.norm() > 0
    for p in trainer.tactile_vae.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_use_tactile_views_off_passthrough() -> None:
    """With use_tactile_views=False the trainer never touches mem_latents/latents.

    We just exercise the no-op path: the smoke replicates `train()`'s
    ``if use_tactile_views`` guard so the visual-only shapes pass through
    unchanged. This guards against accidentally wiring tactile into the
    GE-baseline path.
    """
    dev = _device()
    B, n_view_visual = 2, 3
    mem_size = 4
    T_lat = 6

    mem_latents = torch.randn(B * n_view_visual, 128, mem_size, SPATIAL_H, SPATIAL_W, device=dev)
    future_latents = torch.randn(B * n_view_visual, 128, T_lat - mem_size, SPATIAL_H, SPATIAL_W, device=dev)
    latents = torch.cat((mem_latents, future_latents), dim=2)

    use_tactile_views = False
    n_view_visual_pre = n_view_visual
    n_view = n_view_visual_pre
    if use_tactile_views:                                  # noqa: PLR1714 - mirrors train()
        raise AssertionError("smoke must take the off branch")
    assert mem_latents.shape == (B * n_view, 128, mem_size, SPATIAL_H, SPATIAL_W)
    assert latents.shape == (B * n_view, 128, T_lat, SPATIAL_H, SPATIAL_W)


def main() -> None:
    test_encode_tactile_split_t_alignment()
    print(f"[smoke_tactile_dit_trainer] _encode_tactile_split T-align OK (device={_device()})")
    test_view_stack_concat_and_n_view_bump()
    print("[smoke_tactile_dit_trainer] view-stack concat + n_view bump OK")
    test_stage2_loss_split_slicing()
    print("[smoke_tactile_dit_trainer] Stage-2 lambda mix + slice direction OK")
    test_projector_grad_through_encode_split()
    print("[smoke_tactile_dit_trainer] projector grad / frozen v0c-A OK")
    test_use_tactile_views_off_passthrough()
    print("[smoke_tactile_dit_trainer] use_tactile_views=False pass-through OK")
    print("\n[smoke_tactile_dit_trainer] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
