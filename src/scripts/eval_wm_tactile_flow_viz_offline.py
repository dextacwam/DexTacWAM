#!/usr/bin/env python3
"""Offline Stage-2 WM tactile flow visualization.

Standalone counterpart to the online val PNG produced by
``runner/tactile_dit_trainer.py::validate()`` (which writes
``Validation_tactile_flow_compare_last.png`` next to ``Validation.mp4``
during the training loop). This script loads a saved checkpoint, runs
the FULL multi-step ``pipe.infer(...)`` with tactile injection on a
user-specified val split, and saves a GT vs predicted tactile flow
comparison PNG (via ``utils.tactile_flow_viz.save_flow_compare_grid``)
for each requested sample.

Position in the WM-tactile evaluation stack
-------------------------------------------

============================================  =====================================================
Online validation PNG (in trainer.validate)  fast sanity check during training (every steps_to_val)
``scripts/eval_wm_contact_recall.py``         offline quantitative recall / projected-MSE
``scripts/eval_wm_tactile_flow_viz_offline``  *this script* -- manual ckpt/split/sample PNG dumps
                                              for visual inspection (e.g. comparing two ckpts
                                              side-by-side after a run finishes)
============================================  =====================================================

Layout per sample under ``<output_dir>/sample_{i:03d}/``::

    tactile_flow_compare_t{T:03d}.png   # one per future T (--frame_index all, default)
                                         #   or _last.png   (--frame_index last)
                                         #   or _t{T}.png   (--frame_index <int>)
    Validation.mp4                       # visual prediction MP4 (default ON; --no_save_mp4 disables)
    Validation_gt.mp4                    # visual GT MP4         (default ON; --no_save_mp4 disables)
    manifest.json                        # caption + sample metadata

Hard requirements
-----------------

* ``use_tactile_views: true`` in the yaml.
* ``tactile_vae`` loaded (Stage-1 v0d adapter).
* ``tactile_vae.config.disable_projector: true`` (proj_bypass mode).
  Under bypass the WM target space is the Stage-1 ``tac_latent_pre``
  space, so ``aux_flow_post`` is a valid decoder for the WM's predicted
  tactile latent without any projector inverse approximation. Outside
  bypass mode the predicted latent is in projected space and would need
  the (non-contractive on v0d) 1st-order inverse used by
  ``eval_wm_contact_recall.py``; that path is intentionally NOT enabled
  here because it produces non-interpretable flow viz (see Section 7 of
  ``docs/b2_b3_v0d_proj_bypass_consistency_check.md``).

If any condition fails, the script exits with code 2 and a descriptive
error.

Example
-------

.. code-block:: bash

    cd .
    python scripts/eval_wm_tactile_flow_viz_offline.py \\
        --config configs/cube_handover/stage3_action_expert.yaml \\
        --checkpoint outputs/.../step_30000 \\
        --split holdout_488 \\
        --output_dir outputs/tactile_viz_offline/$(date +%Y_%m_%d_%H_%M_%S) \\
        --n_samples 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as _mp
from einops import rearrange
from safetensors.torch import load_file

_mp.set_sharing_strategy("file_system")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import import_custom_class, save_video
from utils.model_utils import unwrap_model
from utils.tactile_flow_viz import resolve_frame_indices, save_flow_compare_grid


logger = logging.getLogger("wm_tactile_flow_viz_offline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Setup helpers (mirrors eval_wm_contact_recall.py)
# ---------------------------------------------------------------------------


def _ensure_dist_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 500)))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def _build_runner(config_file: str, batch_size: int):
    """Build a fresh TactileDiTTrainer in eval-only mode.

    Mirrors the recipe in ``eval_wm_contact_recall.py`` but does NOT
    enable ``read_tactile_flow`` on val paths -- both the GT and the
    predicted flow used in this script are produced by
    ``tactile_vae.aux_flow_post`` applied to latents that are already
    in the model's adapter-pre-projector space, so the dataset-side
    cached flow stats are not required.
    """
    import yaml

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["use_deepspeed"] = False
    cfg.pop("deepspeed", None)
    if batch_size > 0:
        cfg["batch_size"] = batch_size

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


def _load_ckpt(runner, checkpoint_dir: str) -> None:
    """Load DiT (required) + projector (optional).

    The projector is OPTIONAL because pure proj_bypass training runs
    can omit ``projector.pt``; its forward is never called when
    ``_disable_projector_for_action`` is True. A clear warning is
    emitted so the user knows the projector file was missing.
    """
    dit_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
    if not os.path.isfile(dit_path):
        raise FileNotFoundError(f"missing DiT checkpoint: {dit_path}")
    dit_sd = load_file(dit_path)
    dit_module = (
        runner.diffusion_model.module
        if hasattr(runner.diffusion_model, "module")
        else runner.diffusion_model
    )
    missing, unexpected = dit_module.load_state_dict(dit_sd, strict=False)
    logger.info(
        "Loaded DiT ckpt: %s (missing=%d, unexpected=%d)",
        dit_path, len(missing), len(unexpected),
    )

    proj_path = os.path.join(checkpoint_dir, "projector.pt")
    if runner.projector is not None and os.path.isfile(proj_path):
        proj_ckpt = torch.load(proj_path, map_location="cpu", weights_only=False)
        proj_sd = proj_ckpt.get("projector", proj_ckpt)
        proj_module = (
            runner.projector.module
            if hasattr(runner.projector, "module")
            else runner.projector
        )
        proj_module.load_state_dict(proj_sd)
        proj_module.to(dtype=runner.state.weight_dtype)
        runner.projector.eval()
        logger.info("Loaded projector ckpt: %s", proj_path)
    else:
        logger.warning(
            "No projector.pt at %s; relying on runner.projector warmstart "
            "(only safe under disable_projector=true, which is checked separately).",
            proj_path,
        )


def _check_bypass_or_die(runner) -> None:
    use_tactile_views = bool(getattr(runner.args, "use_tactile_views", False))
    has_tactile_vae = runner.tactile_vae is not None
    is_bypass = bool(getattr(runner, "_disable_projector_for_action", False))
    if not (use_tactile_views and has_tactile_vae and is_bypass):
        raise SystemExit(
            f"offline tactile flow viz requires:\n"
            f"  use_tactile_views=True               (got {use_tactile_views})\n"
            f"  tactile_vae loaded                   (got {has_tactile_vae})\n"
            f"  _disable_projector_for_action=True   (got {is_bypass})\n"
            "Run a bypass yaml or use a non-bypass-only eval script."
        )


# ---------------------------------------------------------------------------
# Sample-level inference (mirrors validate() body, sans TB / writer)
# ---------------------------------------------------------------------------


def _make_pipe(runner, accelerator):
    """Construct the same pipeline `validate()` builds."""
    return runner.pipeline_class(
        runner.scheduler,
        runner.vae,
        runner.text_encoder,
        runner.tokenizer,
        unwrap_model(accelerator, runner.diffusion_model)
        if accelerator is not None
        else runner.diffusion_model,
    )


def _run_one_sample(
    runner,
    pipe,
    batch: Dict,
    out_dir: str,
    sample_id: int,
    *,
    batch_size: int,
    n_chunk: int,
    num_denois_steps: int,
    noise_seed: int,
    frame_index_spec: str,
    save_mp4: bool,
) -> Dict:
    """Run pipe.infer on a single val batch and dump PNG (+optional MP4s).

    Returns a small dict with shapes / paths for the manifest. The
    encoded inputs to ``pipe.infer`` follow ``TactileDiTTrainer.validate``
    one-for-one (line ~3848 onward in ``runner/tactile_dit_trainer.py``).
    """
    os.makedirs(out_dir, exist_ok=True)

    image = batch["video"][:, :, :, : runner.args.data["train"]["n_previous"]].clone()
    prompt = batch["caption"]
    gt_video = batch["video"]
    _, _, n_view_visual, _, h, w = image.shape

    image = image[:batch_size]
    image = rearrange(image, "b c v t h w -> (b v) c t h w")

    if runner.args.return_action and getattr(runner.args, "add_state", False):
        history_action_state = batch["state"][:batch_size]
        if history_action_state.shape[1] > 1:
            mem_size_raw = runner.args.data["train"]["n_previous"]
            history_action_state = history_action_state[
                :, mem_size_raw - 1 : mem_size_raw, :
            ]
        history_action_state = history_action_state.contiguous()
    else:
        history_action_state = None

    weight_dtype = next(runner.diffusion_model.parameters()).dtype
    infer_device = next(runner.diffusion_model.parameters()).device
    mem_size_raw = runner.args.data["train"]["n_previous"]

    # --- Encode tactile mem (exactly like validate()) ----------------------
    if "tactile" not in batch:
        raise KeyError(
            "offline tactile viz requires batch['tactile']; the chosen val "
            "split appears to be visual-only."
        )
    tactile_full = (
        batch["tactile"][:batch_size]
        .to(device=infer_device, dtype=weight_dtype)
        .contiguous()
    )
    if tactile_full.ndim != 6:
        raise ValueError(
            f"batch['tactile'] expected (B, V_hand, F=5, T, H, W); "
            f"got {tuple(tactile_full.shape)}."
        )
    n_view_hand = int(tactile_full.shape[1])

    use_pose_injection = bool(getattr(runner.tactile_vae, "use_pose_injection", False))
    if use_pose_injection:
        if "hand_pose" not in batch:
            raise KeyError(
                "use_pose_injection=true but val batch has no 'hand_pose'. "
                "Set read_hand_pose: true in the data.val yaml."
            )
        hand_pose_val = (
            batch["hand_pose"][:batch_size]
            .to(device=infer_device, dtype=weight_dtype)
            .contiguous()
        )
    else:
        hand_pose_val = None

    with torch.no_grad():
        _, tac_full_pre, _ = runner._encode_tactile_split(
            tactile_full,
            mem_size_raw,
            hand_pose=hand_pose_val,
            return_intermediates=True,
        )
    # _encode_tactile_split returns ``(B, V_hand, C, T_lat, H, W)`` (6-D).
    # The pipeline's tactile injection path expects 5-D
    # ``(B*V_hand, C, T_lat, H, W)`` so visual and tactile rows share the
    # same view-flattened layout. Flatten here and keep the rest of this
    # function operating on the 5-D layout.
    tac_full_pre = rearrange(tac_full_pre, "b vh c t h w -> (b vh) c t h w").contiguous()
    tac_mem_lat = tac_full_pre[:, :, :mem_size_raw].contiguous()

    # --- Multi-step inference --------------------------------------------
    preds = pipe.infer(
        image=image,
        prompt=prompt[:batch_size],
        negative_prompt="",
        num_inference_steps=num_denois_steps,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        guidance_scale=1.0,
        height=h,
        width=w,
        n_view=n_view_visual,
        return_action=runner.args.return_action,
        n_prev=runner.args.data["train"]["n_previous"],
        chunk=(runner.args.data["train"]["chunk"] - 1) // runner.TEMPORAL_DOWN_RATIO + 1,
        return_video=runner.args.return_video,
        noise_seed=noise_seed,
        action_chunk=runner.args.data["train"]["action_chunk"],
        history_action_state=history_action_state,
        pixel_wise_timestep=runner.args.pixel_wise_timestep,
        n_chunk=n_chunk,
        action_dim=(
            runner.args.diffusion_model["config"]["action_in_channels"]
            if runner.args.return_action
            else None
        ),
        tactile_mem_latents=tac_mem_lat,
        n_view_visual=n_view_visual,
        n_view_tactile=n_view_hand,
    )[0]

    info: Dict = {
        "sample_id": sample_id,
        "n_view_visual": n_view_visual,
        "n_view_tactile": n_view_hand,
        "caption": list(prompt[:batch_size]),
        "tactile_full_pre_shape": list(tac_full_pre.shape),
        "tactile_mem_lat_shape": list(tac_mem_lat.shape),
    }

    # --- Optional MP4s (visual rows only; matches online val) -------------
    if save_mp4:
        cap = "Validation"
        chunk_train = runner.args.data["train"]["chunk"]
        action_chunk = runner.args.data["train"]["action_chunk"]
        fps_basic = int(getattr(runner.args, "basic_fps", 30))
        fps = int(fps_basic / max(1, action_chunk // max(1, chunk_train)))
        gt_path = os.path.join(out_dir, f"{cap}_gt.mp4")
        save_video(
            rearrange(
                gt_video[0].data.cpu(),
                "c v t h w -> c t h (v w)",
                v=n_view_visual,
            ),
            gt_path,
            fps=fps,
        )
        info["mp4_gt"] = gt_path
        if runner.args.return_video and "video" in preds:
            pred_path = os.path.join(out_dir, f"{cap}.mp4")
            save_video(
                rearrange(
                    preds["video"].data.cpu(),
                    "(b v) c t h w -> b c t h (v w)",
                    v=n_view_visual,
                )[0],
                pred_path,
                fps=fps,
            )
            info["mp4_pred"] = pred_path

    # --- Decode tactile flow + save PNGs ---------------------------------
    if "tactile_latent" not in preds:
        raise RuntimeError(
            "pipe.infer did not return 'tactile_latent'; either the pipeline "
            "is older than the tactile-injection patch or n_view_tactile was "
            "not propagated correctly."
        )
    pred_tac_future = preds["tactile_latent"]
    T_lat_full = int(tac_full_pre.shape[2])
    T_lat_mem = int(tac_mem_lat.shape[2])
    if pred_tac_future.shape[0] != batch_size * n_view_hand:
        raise ValueError(
            f"pred_tac_future batch dim {pred_tac_future.shape[0]} != "
            f"batch_size*V_hand ({batch_size * n_view_hand})"
        )
    if pred_tac_future.shape[2] != T_lat_full - T_lat_mem:
        logger.warning(
            "[tac-viz-offline] pred_tac_future T=%d != expected %d "
            "(T_lat_full=%d, T_lat_mem=%d); decoding anyway.",
            pred_tac_future.shape[2], T_lat_full - T_lat_mem,
            T_lat_full, T_lat_mem,
        )
    tac_gt_future = tac_full_pre[:, :, T_lat_mem:].contiguous()

    with torch.no_grad():
        flow_pred_flat = runner.tactile_vae.aux_flow_post(
            pred_tac_future.to(dtype=tac_full_pre.dtype)
        )
        flow_gt_flat = runner.tactile_vae.aux_flow_post(tac_gt_future)
    flow_pred = rearrange(
        flow_pred_flat, "(b v) f t h w c -> b v f t h w c",
        b=batch_size, v=n_view_hand,
    )
    flow_gt = rearrange(
        flow_gt_flat, "(b v) f t h w c -> b v f t h w c",
        b=batch_size, v=n_view_hand,
    )
    if not torch.isfinite(flow_pred).all():
        raise ValueError("flow_pred contains NaN or Inf values")
    if flow_pred.abs().sum().item() == 0.0:
        logger.warning("[tac-viz-offline] flow_pred is all-zero; PNG will be blank.")

    T_out = int(flow_gt.shape[3])
    frame_indices = resolve_frame_indices(frame_index_spec, T_out)
    hand_names: Sequence[str] = (
        ("left", "right") if n_view_hand == 2
        else tuple(f"hand{vh}" for vh in range(n_view_hand))
    )

    png_paths: List[str] = []
    for t in frame_indices:
        gt_panel = flow_gt[0, :, :, t].detach().cpu().float().numpy()
        pred_panel = flow_pred[0, :, :, t].detach().cpu().float().numpy()
        if frame_index_spec.strip().lower() == "last":
            tag = "last"
        elif frame_index_spec.strip().lower() == "all":
            tag = f"t{t:03d}"
        else:
            tag = f"t{t:03d}"
        png_path = os.path.join(out_dir, f"tactile_flow_compare_{tag}.png")
        save_flow_compare_grid(
            flow_gt=gt_panel,
            flow_pred=pred_panel,
            save_path=png_path,
            title=(
                f"offline WM tactile flow (bypass) sample={sample_id} "
                f"frame={t}/{T_out - 1} V_hand={n_view_hand}"
            ),
            hand_names=hand_names,
        )
        png_paths.append(png_path)

    info["png_paths"] = png_paths
    info["T_out"] = T_out
    info["frame_indices_saved"] = frame_indices
    return info


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Offline tactile flow visualization (full multi-step pipe.infer "
            "with tactile injection; bypass-mode only)."
        )
    )
    p.add_argument("--config", required=True, help="path to action/WM yaml")
    p.add_argument(
        "--checkpoint", required=True,
        help="step_N directory with diffusion_pytorch_model.safetensors (+ optional projector.pt)",
    )
    p.add_argument(
        "--split", required=True,
        help="val split name from data.val_splits (e.g. holdout_488, cube_val)",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="root directory for sample_*/ outputs (created if missing)",
    )
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument(
        "--sample_indices", default=None,
        help="comma-separated batch indices to dump (overrides --n_samples)",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_chunk", type=int, default=1,
                   help="must be 1 (pipeline tactile injection limit)")
    p.add_argument(
        "--num_inference_steps", type=int, default=-1,
        help="-1 -> use yaml's args.num_inference_step",
    )
    p.add_argument("--noise_seed", type=int, default=42)
    p.add_argument(
        "--frame_index", default="all",
        help="'all' (default) | 'last' | <int>; controls which future frame(s) to save",
    )
    p.add_argument("--save_mp4", dest="save_mp4", action="store_true", default=True,
                   help="also save visual Validation.mp4 + Validation_gt.mp4 (default ON)")
    p.add_argument("--no_save_mp4", dest="save_mp4", action="store_false")
    args = p.parse_args()

    if args.n_chunk != 1:
        raise SystemExit(
            "tactile_mem_latents currently supports n_chunk=1 only "
            "(see custom_pipeline.py); pass --n_chunk 1."
        )

    _ensure_dist_initialized()
    runner = _build_runner(args.config, args.batch_size)
    _load_ckpt(runner, args.checkpoint)
    _check_bypass_or_die(runner)

    if args.split not in runner.val_loaders:
        raise KeyError(
            f"split={args.split!r} not found. available={list(runner.val_loaders.keys())}"
        )
    loader = runner.val_loaders[args.split]

    num_denois_steps = (
        args.num_inference_steps if args.num_inference_steps > 0
        else int(runner.args.num_inference_step)
    )

    runner.diffusion_model.eval()
    if runner.projector is not None:
        runner.projector.eval()
    if runner.tactile_vae is not None:
        runner.tactile_vae.eval()

    accelerator = getattr(runner.state, "accelerator", None)
    pipe = _make_pipe(runner, accelerator)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sample_indices:
        wanted = sorted(
            {int(x) for x in args.sample_indices.split(",") if x.strip()}
        )
    else:
        wanted = list(range(args.n_samples))
    if not wanted:
        raise SystemExit("no samples requested (n_samples=0 and sample_indices empty)")
    max_idx = max(wanted)

    manifest: List[Dict] = []
    for i, batch in enumerate(loader):
        if i > max_idx:
            break
        if i not in wanted:
            continue
        sample_dir = os.path.join(args.output_dir, f"sample_{i:03d}")
        logger.info("[tac-viz-offline] sample=%d -> %s", i, sample_dir)
        info = _run_one_sample(
            runner, pipe, batch, sample_dir, sample_id=i,
            batch_size=args.batch_size,
            n_chunk=args.n_chunk,
            num_denois_steps=num_denois_steps,
            noise_seed=args.noise_seed,
            frame_index_spec=args.frame_index,
            save_mp4=args.save_mp4,
        )
        with open(os.path.join(sample_dir, "manifest.json"), "w") as f:
            json.dump(info, f, indent=2)
        manifest.append(info)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "num_inference_steps": num_denois_steps,
                "noise_seed": args.noise_seed,
                "frame_index": args.frame_index,
                "save_mp4": args.save_mp4,
                "samples": manifest,
            },
            f,
            indent=2,
        )
    logger.info("[tac-viz-offline] wrote %d sample manifests; summary=%s",
                len(manifest), summary_path)

    # Clean dist teardown (same reason as eval_wm_contact_recall.py).
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
