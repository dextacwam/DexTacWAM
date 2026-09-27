#!/usr/bin/env python3
"""scripts/extract_latents_for_tsne.py

Extract VISION + TACTILE latents from ONE task's dataset using the frozen
visual LTX VAE + the v0d multi-finger tactile encoder, and save them as a
per-task `.npz` for `scripts/plot_latent_tsne.py`.

WHY the fused per-hand latent (not the raw per-finger VAE latent):
  The DiT actually consumes the *fused per-hand* tactile latent -- the
  output of the full multi-finger encoder:
      raw tactile (B,V_hand,F,T,H,W)
        -> encode_per_finger  (GrayToRGB + frozen LTX VAE)
        -> FingerSetTransformerAdapter (pose-aware fusion, 5 -> 1)
        -> fused per-hand (B,V_hand,C,T_lat,6,8)
  So the PRIMARY t-SNE analyzes the fused per-hand latent. The pre-fusion
  per-finger latent is kept only as a SUPPORTING diagnostic, and a
  leave-one-finger-out fusion-shift probe measures whether the fused hand
  latent is sensitive to every finger.

This script builds a fully-prepared `TactileDiTTrainer` (same overlay as
scripts/diagnose_vision_tactile_pollution.py) so the VAE + adapter are
wired byte-identically to training, then runs encoder-only forwards (NO
denoising, NO DiT loss). Run it once PER TASK with that task's tactile
config; then run plot_latent_tsne.py to combine all tasks.

Artifacts written to <out>/<tag>/:
  latents.npz  (schema documented in plot_latent_tsne.py)
  thumb.png    representative tactile thumbnail for the Fig-A cluster label

Usage (with the genie_envisioner env active):
  python scripts/extract_latents_for_tsne.py \
      --config configs/cube_handover/stage2_world_model.yaml \
      --tag chip \
      --out eval_artifacts/latent_tsne \
      --max-samples 300 --batch-size 4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Reuse the exact smoke overlay + runner-build helpers the pollution
# diagnostic uses, so project-root resolution, determinism, single-proc
# dist init, and the smoke config overlay stay consistent.
from smoke_v0d_dit_a2_numerical import (  # noqa: E402
    _ensure_project_root_on_path,
    _force_global_determinism,
    _ensure_dist_initialized,
    _override_config_for_smoke,
)


def _build_runner(config_path: str, seed: int, batch_size: int):
    """Build a prepared TactileDiTTrainer (VAE + v0d adapter wired)."""
    _ensure_project_root_on_path()
    _force_global_determinism(seed)
    _ensure_dist_initialized()

    smoke_yaml = _override_config_for_smoke(
        config_path, seed=seed, batch_size=batch_size, train_steps_cap=1,
    )
    from utils import import_custom_class

    Runner = import_custom_class(
        "TactileDiTTrainer", "runner/tactile_dit_trainer.py",
    )
    runner = Runner(smoke_yaml)
    runner.prepare_dataset()
    runner.prepare_models()

    # Single-GPU path skips DeepSpeed's autocast, so match dtypes the way
    # the pollution diagnostic does (projector / tactile_vae would else
    # stay fp32 and mismatch bf16 latents).
    weight_dtype = runner.state.weight_dtype
    if runner.projector is not None and weight_dtype != torch.float32:
        runner.projector.to(dtype=weight_dtype)
    if getattr(runner, "tactile_vae", None) is not None and weight_dtype != torch.float32:
        runner.tactile_vae.to(dtype=weight_dtype)
    return runner


def _pick_loader(runner, split: str | None):
    """Prefer a named/first val loader; fall back to the train loader."""
    loaders = getattr(runner, "val_loaders", None) or {}
    if split and split in loaders:
        return split, _with_meta(loaders[split])
    if loaders:
        name = next(iter(loaders.keys()))
        return name, _with_meta(loaders[name])
    tl = getattr(runner, "train_dataloader", None)
    if tl is None:
        raise RuntimeError("Runner has neither val_loaders nor train_dataloader.")
    return "train", _with_meta(tl)


def _with_meta(loader):
    """Ask the dataset for episode/phase provenance, if it supports it.

    Best-effort: datasets without ``return_meta`` (libero, agibot) are left
    alone and extraction proceeds without the provenance arrays.

    Setting the flag here is enough for worker processes too, because they are
    forked when the iterator is created, which happens after this returns. The
    exception is ``persistent_workers`` with an iterator already alive. Rather
    than guess, the caller checks for the key on each batch and the summary
    reports whether provenance actually came back.
    """
    ds = getattr(loader, "dataset", None)
    if ds is None or not hasattr(type(ds), "return_meta"):
        print("[extract] dataset has no return_meta; skipping episode/phase")
        return loader
    ds.return_meta = True
    return loader


def _pool(x: torch.Tensor) -> torch.Tensor:
    """Mean-pool a (..., C, T_lat, H, W) latent over (T_lat, H, W) -> (..., C)."""
    return x.float().mean(dim=(-1, -2, -3))


@torch.no_grad()
def _save_thumbnail(tactile: torch.Tensor, path: str):
    """Render a representative tactile thumbnail: hand-0, mid-frame, 5 fingers.

    tactile: (B, V_hand, F, T, H, W) in [-1, 1]. We render sample 0 / hand 0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tactile[0, 0].float().cpu().numpy()  # (F, T, H, W)
    F, T = t.shape[0], t.shape[1]
    mid = T // 2
    fig, axes = plt.subplots(1, F, figsize=(1.6 * F, 1.8))
    if F == 1:
        axes = [axes]
    for f in range(F):
        img = (t[f, mid] + 1.0) / 2.0  # -> [0,1]
        axes[f].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[f].axis("off")
    fig.subplots_adjust(wspace=0.05, left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=80, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="Tactile-enabled task yaml.")
    ap.add_argument("--tag", required=True, help="Task display name (subdir).")
    ap.add_argument("--out", default="eval_artifacts/latent_tsne")
    ap.add_argument("--split", default=None,
                    help="val_splits name; default first val loader or train.")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-passes", type=int, default=50,
                    help="Re-iterate the loader up to this many times to reach "
                         "--max-samples (val splits are small; each pass draws "
                         "fresh random windows). Set 1 for a single pass.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vision-view", type=int, default=0,
                    help="RGB view index to use for vision latents (0=head).")
    ap.add_argument("--ablation", choices=["zero", "mean", "both"],
                    default="both",
                    help="Leave-one-finger-out perturbation for the fusion "
                         "shift. 'zero' replaces the finger token with zeros; "
                         "'mean' replaces it with the mean of the other F-1 "
                         "finger tokens (neutral, in-distribution); 'both' "
                         "computes each and stores fusion_shift (zero) + "
                         "fusion_shift_mean.")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    runner = _build_runner(args.config, args.seed, args.batch_size)
    if not bool(getattr(runner.args, "use_tactile_views", False)):
        print(f"[extract:{args.tag}] FATAL: config has use_tactile_views=false; "
              f"this task has no tactile to encode.")
        return 1

    name, loader = _pick_loader(runner, args.split)
    weight_dtype = runner.state.weight_dtype
    device = next(runner.tactile_vae.parameters()).device
    pose_on = bool(getattr(runner.tactile_vae, "use_pose_injection", False))
    num_fingers = int(getattr(runner.tactile_vae, "num_fingers", 5))
    print(f"[extract:{args.tag}] split={name!r} device={device} dtype={weight_dtype} "
          f"pose_on={pose_on} F={num_fingers}")

    do_zero = args.ablation in ("zero", "both")
    do_mean = args.ablation in ("mean", "both")

    vision_L, hand_L, hand_side_L = [], [], []
    finger_L, finger_id_L, finger_side_L = [], [], []
    phase_L, episode_L = [], []
    shift_L = []
    shift_mean_L = []
    thumb_saved = False
    n_hand = 0

    stop = False
    for _pass in range(max(1, args.max_passes)):
      made_progress = False
      for batch in loader:
        made_progress = True
        # ---------- vision (single RGB view) ----------
        video = batch["video"].to(device, dtype=weight_dtype)  # (B,C,V,T,H,W)
        vidx = min(args.vision_view, video.shape[2] - 1)
        head = video[:, :, vidx].contiguous()  # (B,C,T,H,W)
        with torch.no_grad():
            vlat = runner.vae.encode(head).latent_dist.mode()  # (B,128,T_lat,H,W)
        vision_L.append(_pool(vlat).cpu().numpy())

        # ---------- tactile: fused per-hand + per-finger ----------
        tactile = batch["tactile"].to(device, dtype=weight_dtype).contiguous()
        B, V_hand, F, T, H, W = tactile.shape
        hp = None
        if pose_on:
            if "hand_pose" not in batch:
                raise KeyError(
                    f"[extract:{args.tag}] adapter use_pose_injection=True but "
                    f"batch has no 'hand_pose'. Set read_hand_pose+pose_stats_path "
                    f"in the config's data section.")
            hp = batch["hand_pose"].to(device, dtype=weight_dtype).contiguous()

        with torch.no_grad():
            fused, per_finger = runner.tactile_vae.encode_per_hand(
                tactile, hand_pose=hp, return_per_finger=True,
            )  # fused (B,V,C,Tl,h,w) ; per_finger (B,V,F,C,Tl,h,w)

            # pooled fused per-hand -> one point per (sample, hand)
            fused_p = _pool(fused)                       # (B,V,C)
            hand_L.append(fused_p.reshape(B * V_hand, -1).cpu().numpy())
            hand_side_L.append(
                np.tile(np.arange(V_hand), B).astype(np.int64))

            # Hand rows run sample-major with the hand axis fastest (the
            # reshape above), so repeating each per-sample scalar V_hand times
            # keeps provenance aligned with tactile_hand row-for-row.
            if "meta_phase" in batch:
                phase_L.append(np.repeat(
                    batch["meta_phase"].numpy().astype(np.float32), V_hand))
                episode_L.append(np.repeat(
                    batch["meta_episode"].numpy().astype(np.int64), V_hand))

            # pooled per-finger -> one point per (sample, hand, finger)
            finger_p = _pool(per_finger)                 # (B,V,F,C)
            finger_L.append(finger_p.reshape(B * V_hand * F, -1).cpu().numpy())
            fid = np.tile(np.arange(F), B * V_hand).astype(np.int64)
            fside = np.repeat(np.tile(np.arange(V_hand), B), F).astype(np.int64)
            finger_id_L.append(fid)
            finger_side_L.append(fside)

            # ---------- leave-one-finger-out fusion shift ----------
            # Flatten V_hand into batch to reuse fuse_per_hand's per-hand API.
            pf_flat = per_finger.reshape(B * V_hand, F, *per_finger.shape[3:])
            pose_flat = None
            if pose_on:
                pose_flat = hp.reshape(B * V_hand, hp.shape[2], hp.shape[3])
            z_all = runner.tactile_vae.fuse_per_hand(pf_flat, hand_pose=pose_flat)
            z_all_flat = z_all.reshape(z_all.shape[0], -1).float()
            denom = z_all_flat.norm(dim=1).clamp_min(1e-8)
            # Sum over fingers (for the leave-one-out mean token).
            pf_sum = pf_flat.sum(dim=1) if do_mean else None
            shifts = np.zeros((B * V_hand, F), dtype=np.float32)
            shifts_mean = np.zeros((B * V_hand, F), dtype=np.float32)
            for i in range(F):
                if do_zero:
                    masked = pf_flat.clone()
                    masked[:, i] = 0.0  # zero the finger token into the adapter
                    z_i = runner.tactile_vae.fuse_per_hand(
                        masked, hand_pose=pose_flat)
                    z_i_flat = z_i.reshape(z_i.shape[0], -1).float()
                    shifts[:, i] = ((z_all_flat - z_i_flat).norm(dim=1)
                                    / denom).cpu().numpy()
                if do_mean:
                    # Replace finger i with the mean of the OTHER F-1 fingers:
                    # a neutral, in-distribution token (marginalizes finger i).
                    mean_other = (pf_sum - pf_flat[:, i]) / max(F - 1, 1)
                    masked = pf_flat.clone()
                    masked[:, i] = mean_other
                    z_i = runner.tactile_vae.fuse_per_hand(
                        masked, hand_pose=pose_flat)
                    z_i_flat = z_i.reshape(z_i.shape[0], -1).float()
                    shifts_mean[:, i] = ((z_all_flat - z_i_flat).norm(dim=1)
                                         / denom).cpu().numpy()
            if do_zero:
                shift_L.append(shifts)
            if do_mean:
                shift_mean_L.append(shifts_mean)

        if not thumb_saved:
            _save_thumbnail(tactile, os.path.join(out_dir, "thumb.png"))
            thumb_saved = True

        n_hand += B * V_hand
        n_done = sum(len(a) for a in hand_L)
        if n_done >= args.max_samples:
            stop = True
            break
      if stop or not made_progress:
        break

    vision = np.concatenate(vision_L, axis=0)
    tactile_hand = np.concatenate(hand_L, axis=0)
    tactile_hand_side = np.concatenate(hand_side_L, axis=0)
    tactile_finger = np.concatenate(finger_L, axis=0)
    finger_id = np.concatenate(finger_id_L, axis=0)
    finger_side = np.concatenate(finger_side_L, axis=0)

    save_kw = dict(
        task=np.array(args.tag),
        vision=vision.astype(np.float32),
        tactile_hand=tactile_hand.astype(np.float32),
        tactile_hand_side=tactile_hand_side.astype(np.int64),
        tactile_finger=tactile_finger.astype(np.float32),
        finger_id=finger_id.astype(np.int64),
        finger_side=finger_side.astype(np.int64),
    )
    # Per-hand-row provenance: which episode the clip came from and how far
    # into that episode it ends. Lets the plotter separate "this task splits
    # into several clusters" from "this task has several contact phases".
    if phase_L:
        save_kw["clip_phase"] = np.concatenate(phase_L).astype(np.float32)
        save_kw["episode"] = np.concatenate(episode_L).astype(np.int64)
    else:
        print(f"[extract:{args.tag}] WARNING: no episode/phase provenance "
              f"captured; phase-coloured plots will not be available")
    fusion_shift = None
    if do_zero:
        fusion_shift = np.concatenate(shift_L, axis=0)
        save_kw["fusion_shift"] = fusion_shift.astype(np.float32)
    fusion_shift_mean = None
    if do_mean:
        fusion_shift_mean = np.concatenate(shift_mean_L, axis=0)
        save_kw["fusion_shift_mean"] = fusion_shift_mean.astype(np.float32)
        # If only mean was requested, also expose it under the default key so
        # the plotter (which reads `fusion_shift`) works without a flag.
        if not do_zero:
            save_kw["fusion_shift"] = fusion_shift_mean.astype(np.float32)

    npz_path = os.path.join(out_dir, "latents.npz")
    np.savez(npz_path, **save_kw)
    print(f"[extract:{args.tag}] wrote {npz_path}")
    print(f"  vision        {vision.shape}")
    print(f"  tactile_hand  {tactile_hand.shape}  (sides {np.bincount(tactile_hand_side)})")
    print(f"  tactile_finger{tactile_finger.shape}")
    if fusion_shift is not None:
        print(f"  fusion_shift       {fusion_shift.shape}  "
              f"mean/finger={np.round(fusion_shift.mean(0), 4).tolist()}")
    if fusion_shift_mean is not None:
        print(f"  fusion_shift_mean  {fusion_shift_mean.shape}  "
              f"mean/finger={np.round(fusion_shift_mean.mean(0), 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
