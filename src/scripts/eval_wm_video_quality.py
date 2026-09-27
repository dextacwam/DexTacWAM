#!/usr/bin/env python3
"""scripts/eval_wm_video_quality.py

Quantitative RGB video-prediction quality for a B2 world model, in PIXEL
space (the repo otherwise only has latent-space `loss_visual`).

Compares a checkpoint's predicted future RGB frames against ground-truth
future frames over a deterministic set of val samples, per RGB view and
aggregated. Metrics: pixel MSE + PSNR (always), SSIM + LPIPS (if
`torchmetrics` / `lpips` are importable). Also logs latent `loss_visual`
via the trainer's own `_compute_val_loss` for cross-reference.

Intended use (the "20k ablation"): run this twice with the SAME data
config / deterministic indices, once for the pure-vision WM checkpoint and
once for the tactile co-finetuned WM checkpoint, then diff the per-view
tables. Because RGB frame selection does not depend on tactile, both runs
see identical conditioning + GT when the seed and val split match.

Mirrors `TactileDiTTrainer.validate()` for the pipe.infer call and
`scripts/eval_wm_contact_recall.py::_load_ckpt` for checkpoint loading.

Usage (with the genie_envisioner env active):
  python scripts/eval_wm_video_quality.py \
      --config configs/ltx_model/0514_erase_whiteboard_with_wrist/video_model_0514_erase_whiteboard_with_wrist_tactile_v0d_proj_bypass.yaml \
      --checkpoint ./.../step_20000 \
      --tag tactile_20k \
      --out eval_artifacts/wm_video_quality \
      --n-samples 16 --num-inference-steps 30 --save-media 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from einops import rearrange

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ----------------------------------------------------------------------
# Metric backends (dependency-light; SSIM/LPIPS optional)
# ----------------------------------------------------------------------
def _make_metrics(device):
    """Return dict of callables (pred,gt)->float on [0,1] (B,C,H,W) tensors."""
    metrics = {}

    def _mse(pred, gt):
        return float(((pred - gt) ** 2).mean().item())

    def _psnr(pred, gt):
        mse = ((pred - gt) ** 2).mean().item()
        if mse <= 1e-12:
            return 99.0
        return float(10.0 * np.log10(1.0 / mse))

    metrics["mse"] = _mse
    metrics["psnr"] = _psnr

    try:
        from torchmetrics.functional import structural_similarity_index_measure as _ssim

        def _ssim_fn(pred, gt):
            return float(_ssim(pred, gt, data_range=1.0).item())

        metrics["ssim"] = _ssim_fn
        print("[metrics] SSIM: torchmetrics")
    except Exception as e:  # noqa: BLE001
        # Dependency-free fallback: Gaussian-window SSIM in pure torch.
        print(f"[metrics] SSIM: torchmetrics unavailable ({e}); using builtin.")

        def _gauss_win(ch, ws=11, sigma=1.5, device="cpu"):
            coords = torch.arange(ws, dtype=torch.float32, device=device) - (ws - 1) / 2.0
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = (g / g.sum())
            w2d = g[:, None] * g[None, :]
            return w2d.expand(ch, 1, ws, ws).contiguous()

        def _ssim_builtin(pred, gt):
            # pred,gt: (B,C,H,W) in [0,1]
            c = pred.shape[1]
            win = _gauss_win(c, device=pred.device).to(pred.dtype)
            pad = win.shape[-1] // 2
            mu_x = torch.nn.functional.conv2d(pred, win, padding=pad, groups=c)
            mu_y = torch.nn.functional.conv2d(gt, win, padding=pad, groups=c)
            mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
            sx = torch.nn.functional.conv2d(pred * pred, win, padding=pad, groups=c) - mu_x2
            sy = torch.nn.functional.conv2d(gt * gt, win, padding=pad, groups=c) - mu_y2
            sxy = torch.nn.functional.conv2d(pred * gt, win, padding=pad, groups=c) - mu_xy
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            ssim_map = ((2 * mu_xy + c1) * (2 * sxy + c2)) / \
                ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))
            return float(ssim_map.mean().item())

        metrics["ssim"] = _ssim_builtin

    try:
        import lpips as _lpips_mod

        _lpips_net = _lpips_mod.LPIPS(net="alex").to(device).eval()

        def _lpips_fn(pred, gt):
            # lpips expects [-1,1]
            with torch.no_grad():
                return float(_lpips_net(pred * 2 - 1, gt * 2 - 1).mean().item())

        metrics["lpips"] = _lpips_fn
        print("[metrics] LPIPS: lpips(alex)")
    except Exception as e:  # noqa: BLE001
        print(f"[metrics] LPIPS unavailable ({e}); skipping.")

    return metrics


def _build_runner(config_file: str, batch_size: int, seed: int, val_episodes=None):
    """Eval runner: single-GPU, no deepspeed, deterministic (mirrors eval_wm_contact_recall)."""
    import yaml
    from utils import import_custom_class
    from smoke_v0d_dit_a2_numerical import (
        _ensure_project_root_on_path, _force_global_determinism,
        _ensure_dist_initialized,
    )

    _ensure_project_root_on_path()
    _force_global_determinism(seed)
    _ensure_dist_initialized()

    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    cfg["use_deepspeed"] = False
    cfg.pop("deepspeed", None)
    cfg["batch_size"] = int(batch_size)
    cfg["dataloader_num_workers"] = 0
    cfg["persistent_workers"] = False
    cfg["use_color_jitter"] = False
    cfg["caption_dropout_p"] = 0.0
    cfg["seed"] = int(seed)
    cfg["output_dir"] = os.path.join(tempfile.gettempdir(), "wm_video_quality_out")

    # Optional episode override on the val dataset (e.g. to score IN-SAMPLE
    # training episodes for a symmetric in-sample comparison). The eval loop
    # iterates runner.val_dataset (data.val), so overriding data.val.episodes
    # is sufficient; we also patch data.val_splits for consistency.
    if val_episodes is not None:
        eps = [int(e) for e in val_episodes]
        data = cfg.get("data", {})
        if "val" in data and isinstance(data["val"], dict):
            data["val"]["episodes"] = list(eps)
        for _name, _split in (data.get("val_splits", {}) or {}).items():
            if isinstance(_split, dict):
                _split["episodes"] = list(eps)
        print(f"[val-episodes] overriding val episodes -> {eps}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name
    Runner = import_custom_class("TactileDiTTrainer", "runner/tactile_dit_trainer.py")
    try:
        runner = Runner(tmp_path)
    finally:
        os.unlink(tmp_path)
    runner.prepare_dataset()
    runner.prepare_models()
    runner.prepare_trainable_parameters()
    runner.prepare_optimizer()
    runner.prepare_for_training()
    return runner


def _load_ckpt(runner, checkpoint_dir: str):
    """Load DiT safetensors (+ optional projector.pt for tactile runs)."""
    from safetensors.torch import load_file

    dit_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
    if not os.path.isfile(dit_path):
        raise FileNotFoundError(f"missing DiT checkpoint: {dit_path}")
    dit_sd = load_file(dit_path)
    dit_module = (runner.diffusion_model.module
                  if hasattr(runner.diffusion_model, "module")
                  else runner.diffusion_model)
    missing, unexpected = dit_module.load_state_dict(dit_sd, strict=False)
    print(f"[ckpt] DiT {dit_path}\n       missing={len(missing)} unexpected={len(unexpected)}")
    # view_embed shape mismatch would show up as a missing key -> fail loud
    # (that means the config's max_view != the checkpoint's, i.e. wrong pair).
    ve_missing = [k for k in missing if "view_embed" in k]
    if ve_missing:
        print(f"[ckpt] WARNING: view_embed not loaded ({ve_missing}); "
              f"config max_view likely != checkpoint. Results INVALID for "
              f"this checkpoint unless intended.")

    proj_path = os.path.join(checkpoint_dir, "projector.pt")
    if runner.projector is not None and os.path.isfile(proj_path):
        proj_ckpt = torch.load(proj_path, map_location="cpu", weights_only=False)
        proj_sd = proj_ckpt.get("projector", proj_ckpt)
        proj_module = (runner.projector.module
                       if hasattr(runner.projector, "module") else runner.projector)
        proj_module.load_state_dict(proj_sd)
        proj_module.to(dtype=runner.state.weight_dtype)
        runner.projector.eval()
        print(f"[ckpt] projector {proj_path}")
    elif runner.projector is not None:
        print(f"[ckpt] NOTE: projector present but no projector.pt at {proj_path}")


@torch.no_grad()
def _predict_future(runner, batch, num_inference_steps):
    """Run pipe.infer for video only; return (pred (b v) c t h w, gt_future b c v t h w)."""
    from utils.model_utils import unwrap_model

    acc = getattr(runner.state, "accelerator", None)
    pipe = runner.pipeline_class(
        runner.scheduler, runner.vae, runner.text_encoder, runner.tokenizer,
        unwrap_model(acc, runner.diffusion_model) if acc is not None else runner.diffusion_model,
    )
    n_prev = runner.args.data["train"]["n_previous"]
    image = batch["video"][:, :, :, :n_prev].clone()  # b c v t h w (mem)
    prompt = batch["caption"]
    gt_video = batch["video"]  # b c v T h w
    b, c, v, t, h, w = image.shape
    image1 = rearrange(image[:1], "b c v t h w -> (b v) c t h w")

    preds = pipe.infer(
        image=image1,
        prompt=prompt[:1],
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        guidance_scale=1.0,
        height=h, width=w, n_view=v,
        return_action=False,
        n_prev=n_prev,
        chunk=(runner.args.data["train"]["chunk"] - 1) // runner.TEMPORAL_DOWN_RATIO + 1,
        return_video=True,
        noise_seed=42,
        action_chunk=runner.args.data["train"]["action_chunk"],
        history_action_state=None,
        pixel_wise_timestep=runner.args.pixel_wise_timestep,
        n_chunk=1,
        action_dim=None,
    )[0]
    pred_video = preds["video"].float().cpu()  # (b v) c t h w in [-1,1]
    gt_future = gt_video[:1, :, :, n_prev:].float().cpu()  # b c v (T-n_prev) h w
    return pred_video, gt_future, v


def _to01(x):
    return ((x.clamp(-1, 1) + 1.0) / 2.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="step_* dir with safetensors")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="eval_artifacts/wm_video_quality")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--val-episodes", nargs="*", type=int, default=None,
                    help="Override the val-dataset episode list (e.g. score "
                         "IN-SAMPLE training episodes 0..9 for a symmetric "
                         "in-sample comparison). Default: config's val split.")
    ap.add_argument("--num-inference-steps", type=int, default=30)
    ap.add_argument("--views", nargs="*", default=None,
                    help="RGB view names for the per-view table (len must = n_view).")
    ap.add_argument("--save-media", type=int, default=2,
                    help="How many pred-vs-GT sample MP4s/PNGs to dump.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent-loss", action="store_true",
                    help="Also compute latent loss_visual via _compute_val_loss.")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    runner = _build_runner(args.config, batch_size=1, seed=args.seed,
                           val_episodes=args.val_episodes)
    _load_ckpt(runner, args.checkpoint)
    device = next(runner.diffusion_model.parameters()).device
    metrics = _make_metrics(device)

    # Build our OWN batch_size=1 loader over the val DATASET so each
    # iteration is one distinct sample. runner.val_dataloader may batch the
    # whole val split into a single batch (validate() only ever uses [:1] of
    # it), which would score just one sample.
    base_ds = getattr(runner, "val_dataset", None)
    if base_ds is not None:
        loader = torch.utils.data.DataLoader(
            base_ds, batch_size=1, shuffle=False, num_workers=0,
        )
        print(f"[loader] iterating val_dataset (len={len(base_ds)}) at batch_size=1")
    else:
        loader = runner.val_dataloader
        print("[loader] no val_dataset; falling back to runner.val_dataloader")

    per_view_acc = {}   # view_idx -> metric_name -> [values]
    agg_acc = {m: [] for m in metrics}
    media_saved = 0
    n_done = 0

    from utils import save_video  # trainer's mp4 writer

    for batch in loader:
        pred_video, gt_future, v = _predict_future(runner, batch, args.num_inference_steps)
        # pred_video: (1*v) c t h w ; gt_future: 1 c v tg h w
        pred = rearrange(pred_video, "(b v) c t h w -> b v c t h w", v=v)  # 1 v c tp h w
        gt = rearrange(gt_future, "b c v t h w -> b v c t h w")            # 1 v c tg h w
        tp, tg = pred.shape[3], gt.shape[3]
        tmin = min(tp, tg)
        if tp != tg and n_done == 0:
            print(f"[align] pred_t={tp} gt_t={tg}; comparing last {tmin} frames.")
        pred = pred[:, :, :, tp - tmin:]   # trailing align
        gt = gt[:, :, :, tg - tmin:]

        pred01 = _to01(pred); gt01 = _to01(gt)  # 1 v c t h w in [0,1]
        for vi in range(v):
            # flatten time into batch for per-frame image metrics: (t) c h w
            p = pred01[0, vi].permute(1, 0, 2, 3).to(device)  # t c h w
            g = gt01[0, vi].permute(1, 0, 2, 3).to(device)
            per_view_acc.setdefault(vi, {m: [] for m in metrics})
            for mname, fn in metrics.items():
                val = fn(p, g)
                per_view_acc[vi][mname].append(val)
                agg_acc[mname].append(val)

        if media_saved < args.save_media:
            # side-by-side GT|pred per view, concatenated horizontally
            comp = torch.cat([gt01[0], pred01[0]], dim=-1)  # v c t h (2w)
            grid = rearrange(comp, "v c t h w -> c t h (v w)")
            try:
                save_video(grid, os.path.join(out_dir, f"sample{n_done:02d}_gt_vs_pred.mp4"), fps=6)
            except Exception as e:  # noqa: BLE001
                print(f"[media] save_video failed: {e}")
            media_saved += 1

        n_done += 1
        if n_done >= args.n_samples:
            break

    def _stat(vals):
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "n": int(a.size)} if a.size else {"mean": float("nan"), "n": 0}

    view_names = args.views if args.views and len(args.views) == len(per_view_acc) else \
        [f"view{vi}" for vi in sorted(per_view_acc)]
    report = {
        "tag": args.tag, "config": args.config, "checkpoint": args.checkpoint,
        "n_samples": n_done, "num_inference_steps": args.num_inference_steps,
        "aggregate": {m: _stat(agg_acc[m]) for m in metrics},
        "per_view": {
            view_names[i]: {m: _stat(per_view_acc[vi][m]) for m in metrics}
            for i, vi in enumerate(sorted(per_view_acc))
        },
        # Raw per-sample values, kept in evaluation order. Two checkpoints
        # scored on the same val episodes with the same seed walk the samples
        # in the same order, so entry i refers to the same episode in both
        # runs and the two can be compared as paired observations. That
        # matters: across-episode spread here is several dB, which swamps the
        # difference between two checkpoints when only means are available,
        # whereas pairing cancels the episode difficulty that both share.
        "per_sample": {
            view_names[i]: {m: [float(v) for v in per_view_acc[vi][m]]
                            for m in metrics}
            for i, vi in enumerate(sorted(per_view_acc))
        },
    }

    if args.latent_loss and getattr(runner, "val_loaders", None):
        name = next(iter(runner.val_loaders.keys()))
        vl = runner._compute_val_loss(runner.val_loaders[name],
                                      max_batches=args.n_samples, tag=name)
        report["latent_loss_visual"] = float(vl.get("loss_visual", float("nan")))
        report["latent_loss_video"] = float(vl.get("loss_video", float("nan")))

    jpath = os.path.join(out_dir, "metrics.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[video-quality:{args.tag}] wrote {jpath}")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
