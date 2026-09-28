#!/usr/bin/env python3
"""Standalone WM tactile contact recall / projected-space MSE evaluation.

Phase A (decode-chain sanity):
  clean:     GT tactile -> adapter -> AuxFlowDecoder -> V4B recall
  roundtrip: GT tactile -> adapter -> projector -> 1st-order inverse
             -> AuxFlowDecoder -> V4B recall   (kept as inverse-failure evidence)

Phase B (V1B, primary WM-quality metric):
  Full WM forward layout (visual+tactile rows, memory/future masks,
  text/action conditioning), one-forward x0 reconstruction. Compares x0_est
  and noisy_input against GT projected tactile latent in projected latent
  space (no inverse, no decode). Reports denoise_ratio = mse_x0 / mse_noisy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as _mp
from einops import rearrange

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from safetensors.torch import load_file

_mp.set_sharing_strategy("file_system")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runner.visual_vae_adapter_trainer import _V4BAccumulator
from utils import import_custom_class
from utils.data_utils import (
    gen_noise_from_condition_frame_latent,
    get_latents,
    get_text_conditions,
    randn_tensor,
)
from utils.model_utils import forward_pass


logger = logging.getLogger("wm_recall_eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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


def _denormalize_flow(x: torch.Tensor, flow_mean: torch.Tensor, flow_std: torch.Tensor) -> torch.Tensor:
    return x * flow_std.view(1, 1, 1, 1, 1, 3) + flow_mean.view(1, 1, 1, 1, 1, 3)


def _approximate_projector_inverse(projector, x_proj: torch.Tensor, view_idx: torch.Tensor) -> torch.Tensor:
    """1st-order inverse of TactileProjector.forward()."""
    out_dtype = x_proj.dtype
    proj_dtype = next(projector.parameters()).dtype
    x_proj = x_proj.to(proj_dtype)
    b, v, c, t, h, w = x_proj.shape
    view_bias = projector.view_embed(view_idx).view(b, v, c, 1, 1, 1)
    x = x_proj - view_bias
    x = x - projector.modality_bias.bias.view(1, 1, c, 1, 1, 1)
    x_flat = rearrange(x, "b v c t h w -> (b v t h w) c")
    residual = projector.mlp(projector.norm(x_flat))
    x_flat = x_flat - projector.alpha * residual
    out = rearrange(x_flat, "(b v t h w) c -> b v c t h w", b=b, v=v, t=t, h=h, w=w)
    return out.to(out_dtype)


def _decode_flow(tactile_vae, adapter_latent_future: torch.Tensor) -> torch.Tensor:
    """adapter_latent_future: (B, V_hand, C, T_lat_future, 6, 8)."""
    out_dtype = adapter_latent_future.dtype
    vae_dtype = next(tactile_vae.aux_flow_post.parameters()).dtype
    adapter_latent_future = adapter_latent_future.to(vae_dtype)
    b, v, c, t_lat, h, w = adapter_latent_future.shape
    x = rearrange(adapter_latent_future, "b v c t h w -> (b v) c t h w")
    pred = tactile_vae.aux_flow_post(x)  # (B*V, 5, T_out, 24, 32, 3)
    pred = rearrange(pred, "(b v) f t h w c -> b v f t h w c", b=b, v=v)
    return pred.to(out_dtype)


def _build_runner(config_file: str, flow_stats_path: str, batch_size: int):
    import yaml

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["use_deepspeed"] = False
    cfg.pop("deepspeed", None)
    if batch_size > 0:
        cfg["batch_size"] = batch_size

    # Enable tactile_flow read only on eval val paths.
    if "data" in cfg and "val" in cfg["data"]:
        cfg["data"]["val"]["read_tactile_flow"] = True
        cfg["data"]["val"]["flow_stats_path"] = flow_stats_path
    if "data" in cfg and "val_splits" in cfg["data"]:
        for split_name in cfg["data"]["val_splits"]:
            cfg["data"]["val_splits"][split_name]["read_tactile_flow"] = True
            cfg["data"]["val_splits"][split_name]["flow_stats_path"] = flow_stats_path

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
    dit_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
    if not os.path.isfile(dit_path):
        raise FileNotFoundError(f"missing DiT checkpoint: {dit_path}")
    dit_sd = load_file(dit_path)
    dit_module = runner.diffusion_model.module if hasattr(runner.diffusion_model, "module") else runner.diffusion_model
    missing, unexpected = dit_module.load_state_dict(dit_sd, strict=False)
    logger.info(f"Loaded DiT ckpt: {dit_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    proj_path = os.path.join(checkpoint_dir, "projector.pt")
    if not os.path.isfile(proj_path):
        raise FileNotFoundError(f"missing projector checkpoint: {proj_path}")
    proj_ckpt = torch.load(proj_path, map_location="cpu", weights_only=False)
    proj_sd = proj_ckpt.get("projector", proj_ckpt)
    proj_module = runner.projector.module if hasattr(runner.projector, "module") else runner.projector
    proj_module.load_state_dict(proj_sd)
    proj_module.to(dtype=runner.state.weight_dtype)
    runner.projector.eval()
    logger.info(f"Loaded projector ckpt: {proj_path}")


def _view_idx(batch_size: int, v_hand: int, device: torch.device, offset: int) -> torch.Tensor:
    idx = torch.arange(v_hand, device=device, dtype=torch.long) + int(offset)
    idx = idx.unsqueeze(0).expand(batch_size, v_hand).contiguous()
    return idx


def _accumulate_recall(
    device,
    pred_flow_future: torch.Tensor,
    gt_future: torch.Tensor,
    flow_mean: torch.Tensor,
    flow_std: torch.Tensor,
    contact_threshold: float,
) -> Dict:
    """Run Stage-1 V4B accumulator on already-decoded predicted flow."""
    pred_for_acc = rearrange(pred_flow_future, "b v f t h w c -> (b v) f t h w c")
    gt_for_acc = rearrange(gt_future, "b v f t h w c -> (b v) f t h w c")
    flow_loss = float((pred_for_acc - gt_for_acc).pow(2).mean().item())
    acc = _V4BAccumulator(device=device)
    acc.update(
        gt=gt_for_acc,
        pred=pred_for_acc,
        flow_loss_value=flow_loss,
        denormalize_fn=lambda x: _denormalize_flow(x, flow_mean, flow_std),
        contact_threshold=contact_threshold,
    )
    return acc.finalize()


def _projector_diagnostics(projector, tac_pre: torch.Tensor, tac_latent: torch.Tensor, view_idx: torch.Tensor) -> Dict:
    """Quantify how far the projector deviates from identity, batch-level.

    Reports projector_forward_mismatch first: ``|| P(tac_pre, vidx) - tac_latent || / || tac_latent ||``.
    This is the ground truth for whether our reproduced forward path
    (view_idx, modality_bias.bias, view_embed, alpha*MLP+LN, sum order) matches
    the projector used during training. Must be near zero (< 1e-3); if it is
    large, projector_inverse_error becomes uninterpretable.
    """
    p = tac_pre.float()
    q = tac_latent.float()
    p_norm = p.norm().clamp_min(1e-12)
    q_norm = q.norm().clamp_min(1e-12)
    proj_effect = ((q - p).norm() / p_norm).item()

    proj_dtype = next(projector.parameters()).dtype
    p_dt = p.to(proj_dtype)
    with torch.no_grad():
        proj_recon = projector(p_dt, view_idx)
    proj_forward_mismatch = (
        (proj_recon.float() - q).norm() / q_norm
    ).item()

    b, v, c, t, h, w = p_dt.shape

    view_bias = projector.view_embed(view_idx)  # (b, v, c)
    view_bias_full = view_bias.view(b, v, c, 1, 1, 1).expand_as(p_dt)
    mod_bias = projector.modality_bias.bias
    mod_bias_full = mod_bias.view(1, 1, c, 1, 1, 1).expand_as(p_dt)

    p_flat = rearrange(p_dt, "b v c t h w -> (b v t h w) c")
    residual = projector.mlp(projector.norm(p_flat))
    residual_full = rearrange(
        residual, "(b v t h w) c -> b v c t h w", b=b, v=v, t=t, h=h, w=w
    ).float()
    alpha = float(projector.alpha.detach().float().item())
    alpha_residual_rel = (alpha * residual_full).norm().float().item() / p_norm.item()

    view_bias_rel = view_bias_full.float().norm().item() / p_norm.item()
    mod_bias_rel = mod_bias_full.float().norm().item() / p_norm.item()

    return {
        "projector_forward_mismatch": float(proj_forward_mismatch),
        "alpha": alpha,
        "projector_effect_rel": float(proj_effect),
        "alpha_residual_rel": float(alpha_residual_rel),
        "view_bias_rel": float(view_bias_rel),
        "mod_bias_rel": float(mod_bias_rel),
    }


def _phase_a_batch(
    runner,
    batch: Dict,
    flow_mean: torch.Tensor,
    flow_std: torch.Tensor,
    contact_threshold: float,
    tactile_view_offset: int,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Phase A: decode-chain sanity, both with and without going through the projector.

    Computes:
      * phaseA_clean_recall: GT tactile -> adapter -> AuxFlowDecoder -> recall.
        Skips projector + inverse entirely; this is the true upper bound for
        the decode chain itself, and isolates "decoder works correctly" from
        "inverse projector is accurate".
      * phaseA_roundtrip_recall: GT tactile -> adapter -> projector -> 1st-order
        inverse -> AuxFlowDecoder -> recall. Same number as before; tests
        whether the inverse is good enough to use for Phase B.
      * projector_inverse_error: ||inv(projector(x)) - x|| / ||x||.
      * projector diagnostics: alpha, projector_effect_rel, alpha_residual_rel,
        view_bias_rel, mod_bias_rel.
    """
    device = runner.state.accelerator.device
    weight_dtype = runner.state.weight_dtype
    mem_size = int(runner.args.data["train"]["n_previous"])

    tactile = batch["tactile"].to(device, dtype=weight_dtype).contiguous()
    hand_pose = batch["hand_pose"].to(device, dtype=weight_dtype).contiguous() if "hand_pose" in batch else None
    b, v_hand = tactile.shape[0], tactile.shape[1]

    tac_latent, tac_pre, _ = runner._encode_tactile_split(
        tactile, mem_size, hand_pose=hand_pose, return_intermediates=True
    )  # (B, Vh, C, T_lat, 6, 8)

    vidx = _view_idx(b, v_hand, tac_latent.device, tactile_view_offset)
    logger.info(f"tactile view_idx (phaseA) = {vidx[0].tolist()}")

    diag = _projector_diagnostics(runner.projector, tac_pre, tac_latent, vidx)
    logger.info(
        "[phaseA] projector diag: "
        f"forward_mismatch={diag['projector_forward_mismatch']:.4e} "
        f"alpha={diag['alpha']:.4f} "
        f"proj_effect_rel={diag['projector_effect_rel']:.4f} "
        f"alpha_residual_rel={diag['alpha_residual_rel']:.4f} "
        f"view_bias_rel={diag['view_bias_rel']:.4f} "
        f"mod_bias_rel={diag['mod_bias_rel']:.4f}"
    )

    if "tactile_flow" not in batch:
        raise KeyError("Phase A requires batch['tactile_flow']; ensure R1-min/R2-min are active.")
    gt_flow = batch["tactile_flow"].to(device, dtype=torch.float32).contiguous()  # (B,Vh,5,T,24,32,3)
    gt_future = gt_flow[:, :, :, mem_size:, :, :, :]

    # ----- (A) CLEAN: skip projector + inverse entirely -------------------
    tac_pre_future = tac_pre[:, :, :, mem_size:, :, :]
    pred_flow_clean = _decode_flow(runner.tactile_vae, tac_pre_future).float()
    assert pred_flow_clean.shape[2] == gt_future.shape[2], (
        f"finger mismatch (clean): {pred_flow_clean.shape} vs {gt_future.shape}"
    )
    assert pred_flow_clean.shape[3] == gt_future.shape[3], (
        f"time mismatch (clean): {pred_flow_clean.shape} vs {gt_future.shape}"
    )
    assert tuple(pred_flow_clean.shape[-3:-1]) == (24, 32), (
        f"spatial mismatch (clean): {pred_flow_clean.shape}"
    )
    m_clean = _accumulate_recall(
        device, pred_flow_clean, gt_future, flow_mean, flow_std, contact_threshold
    )

    # ----- (B) ROUNDTRIP: through projector + 1st-order inverse -----------
    inv = _approximate_projector_inverse(runner.projector, tac_latent.float(), vidx)
    inv_err = ((inv - tac_pre.float()).norm() / tac_pre.float().norm().clamp_min(1e-12)).item()
    inv_future = inv[:, :, :, mem_size:, :, :]
    pred_flow_rt = _decode_flow(runner.tactile_vae, inv_future).float()
    assert pred_flow_rt.shape == pred_flow_clean.shape, (
        f"shape mismatch clean vs roundtrip: {pred_flow_clean.shape} vs {pred_flow_rt.shape}"
    )
    m_rt = _accumulate_recall(
        device, pred_flow_rt, gt_future, flow_mean, flow_std, contact_threshold
    )

    out = {
        "phaseA_clean_recall_upper_bound": float(m_clean["recall_mean"]),
        "phaseA_clean_precision_mean": float(m_clean["precision_mean"]),
        "phaseA_clean_recall_pf": m_clean["recall_pf"],
        "phaseA_roundtrip_recall_upper_bound": float(m_rt["recall_mean"]),
        "phaseA_roundtrip_precision_mean": float(m_rt["precision_mean"]),
        "phaseA_roundtrip_recall_pf": m_rt["recall_pf"],
        "projector_inverse_error": float(inv_err),
        "projector_diag": diag,
    }
    return out, tac_latent.detach()


def _phase_b_batch(
    runner,
    batch: Dict,
    sigma_mode: str,
) -> Dict[str, float]:
    """One-batch Phase B: projected-space x0 MSE (V1B primary metric).

    Full WM forward layout matches `_forward_loss_batch` (visual + tactile
    rows stacked, memory + future masks, text + state + action conditioning,
    same sigma sampler, same scheduler). After one DiT forward, compute:

      mse_x0    = MSE(x0_est_tactile_future, GT_projected_tactile_future)
      mse_noisy = MSE(noisy_tactile_future, GT_projected_tactile_future)
      denoise_ratio = mse_x0 / mse_noisy

    All metrics are restricted to FUTURE slots (>= mem_size in latent time)
    so memory/conditioning slots cannot inflate the ratio. MSE is computed in
    float32 from bf16 inputs to avoid catastrophic cancellation. Per-view
    breakdown is reported so we can spot one-hand collapses.

    This path intentionally does NOT call _approximate_projector_inverse or
    AuxFlowDecoder: with v0d projector geometry (alpha_residual_rel ~ 1.95),
    the 1st-order inverse is non-contractive and would pollute any
    downstream flow-space metric.
    """
    accelerator = runner.state.accelerator
    device = accelerator.device
    weight_dtype = runner.state.weight_dtype
    mem_size = int(runner.args.data["train"]["n_previous"])

    video = batch["video"].to(device, dtype=weight_dtype).contiguous()
    bsz, _, n_view_visual, _, raw_h, raw_w = video.shape
    video = rearrange(video, "b c v t h w -> (b v) c t h w")
    mem = video[:, :, :mem_size]
    future_video = video[:, :, mem_size:]

    if runner.args.return_action:
        future_video = future_video[:, :, :1].repeat(1, 1, runner.args.data["train"]["chunk"], 1, 1)

    latent_frames = future_video.shape[2] // runner.TEMPORAL_DOWN_RATIO + 1 + mem_size
    latent_height = raw_h // runner.SPATIAL_DOWN_RATIO
    latent_width = raw_w // runner.SPATIAL_DOWN_RATIO

    mem_latents, future_video_latents = get_latents(runner.vae, mem, future_video)
    mem_latents = rearrange(
        mem_latents, "(b v m) (h w) c -> (b v) c m h w",
        b=bsz, m=mem_size, h=latent_height
    )
    future_video_latents = rearrange(
        future_video_latents, "(b v) (f h w) c -> (b v) c f h w",
        b=bsz, h=latent_height, w=latent_width
    )
    latents = torch.cat((mem_latents, future_video_latents), dim=2)

    tactile = batch["tactile"].to(device, dtype=weight_dtype).contiguous()
    hand_pose = batch["hand_pose"].to(device, dtype=weight_dtype).contiguous() if "hand_pose" in batch else None
    n_view_tactile = tactile.shape[1]
    tac_full = runner._encode_tactile_split(tactile, mem_size, hand_pose=hand_pose)
    tac_full = rearrange(tac_full, "b v c t h w -> (b v) c t h w")
    tac_mem = tac_full[:, :, :mem_size]
    mem_latents = torch.cat([mem_latents, tac_mem], dim=0)
    latents = torch.cat([latents, tac_full], dim=0)
    n_view = n_view_visual + n_view_tactile

    latents = rearrange(latents, "bv c f h w -> bv (f h w) c")
    captions = batch["caption"]
    text_conds = get_text_conditions(runner.tokenizer, runner.text_encoder, captions)
    prompt_embeds = text_conds["prompt_embeds"]
    prompt_attention_mask = text_conds["prompt_attention_mask"]
    dropout_mask_prompt = torch.zeros(bsz, dtype=torch.bool, device=device).unsqueeze(1).unsqueeze(2)
    prompt_embeds = (
        runner.uncond_prompt_embeds.repeat(bsz, 1, 1) * dropout_mask_prompt
        + prompt_embeds * ~dropout_mask_prompt
    )

    scheduler_sigmas = runner.scheduler.sigmas.clone().to(device=device, dtype=weight_dtype)
    if sigma_mode == "random":
        weights = compute_density_for_timestep_sampling(
            weighting_scheme=runner.args.flow_weighting_scheme,
            batch_size=bsz,
            logit_mean=runner.args.flow_logit_mean,
            logit_std=runner.args.flow_logit_std,
            mode_scale=runner.args.flow_mode_scale,
        )
        weights = weights.unsqueeze(1).repeat(1, n_view)
        indices = (rearrange(weights, "b v -> (b v)") * runner.scheduler.config.num_train_timesteps).long()
        sigmas = scheduler_sigmas[indices]
    else:
        sigma_val = float(sigma_mode)
        sigmas = torch.full((bsz * n_view,), sigma_val, device=device, dtype=weight_dtype)
    timesteps = (sigmas * 1000.0).long()

    if runner.args.return_action:
        act_state = batch["state"]
        if act_state.shape[1] != 1:
            act_state = act_state[:, mem_size - 1:mem_size]
        act_state = act_state.to(device, dtype=weight_dtype).contiguous()
        actions = batch["actions"][:, -runner.args.data["train"]["action_chunk"]:].to(
            device, dtype=weight_dtype
        ).contiguous()
        noise_actions = randn_tensor(actions.shape, device=device, dtype=weight_dtype)
        action_timesteps = timesteps[:bsz].unsqueeze(-1).repeat(1, actions.shape[1])
        action_sigmas = sigmas[:bsz]
        action_ss = action_sigmas.reshape(-1, 1, 1).repeat(1, 1, actions.shape[-1])
        noisy_actions = (1.0 - action_ss) * actions + action_ss * noise_actions
    else:
        act_state = None
        action_timesteps = None
        noisy_actions = None

    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
        mem_latents, latent_frames, latent_height, latent_width,
        noise_to_condition_frames=runner.args.noise_to_first_frame,
    )
    if runner.args.pixel_wise_timestep:
        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
    else:
        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)
    ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
    noisy_latents = (1.0 - ss) * latents + ss * noise

    pred_all = forward_pass(
        model=runner.diffusion_model,
        timesteps=timesteps,
        noisy_latents=noisy_latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        num_frames=latent_frames,
        height=latent_height,
        width=latent_width,
        n_view=n_view,
        action_states=noisy_actions,
        action_timestep=action_timesteps,
        return_video=runner.args.return_video or runner.args.return_action,
        return_action=runner.args.return_action,
        video_attention_mask=None,
        history_action_state=act_state,
        condition_mask=conditioning_mask,
    )["latents"]
    v_pred = pred_all["video"]
    x0_est = noisy_latents - ss * v_pred  # scheduler-consistent for current trainer objective

    n_visual_rows = bsz * n_view_visual

    def _to_bvcthw(seq: torch.Tensor) -> torch.Tensor:
        # (b*v_tac, f*h*w, c) -> (b, v_tac, c, t, h, w) for tactile rows.
        return rearrange(
            seq, "(b v) (t h w) c -> b v c t h w",
            b=bsz, v=n_view_tactile, t=latent_frames, h=latent_height, w=latent_width,
        )

    tac_x0 = _to_bvcthw(x0_est[n_visual_rows:])
    tac_gt = _to_bvcthw(latents[n_visual_rows:])           # GT projected tactile latent
    tac_nz = _to_bvcthw(noisy_latents[n_visual_rows:])      # noisy input fed to DiT

    # Future-only: skip memory / conditioning slots so they cannot bias ratio.
    tac_x0_f = tac_x0[:, :, :, mem_size:, :, :].float()
    tac_gt_f = tac_gt[:, :, :, mem_size:, :, :].float()
    tac_nz_f = tac_nz[:, :, :, mem_size:, :, :].float()

    mse_x0 = ((tac_x0_f - tac_gt_f) ** 2).mean().item()
    mse_noisy = ((tac_nz_f - tac_gt_f) ** 2).mean().item()
    denoise_ratio = mse_x0 / max(mse_noisy, 1e-12)

    per_view: Dict[str, Dict[str, float]] = {}
    for v_idx in range(n_view_tactile):
        mv_x0 = ((tac_x0_f[:, v_idx] - tac_gt_f[:, v_idx]) ** 2).mean().item()
        mv_nz = ((tac_nz_f[:, v_idx] - tac_gt_f[:, v_idx]) ** 2).mean().item()
        per_view[f"view{v_idx}"] = {
            "mse_x0": float(mv_x0),
            "mse_noisy": float(mv_nz),
            "denoise_ratio": float(mv_x0 / max(mv_nz, 1e-12)),
            "rmse_x0": float(math.sqrt(max(mv_x0, 0.0))),
            "rmse_noisy": float(math.sqrt(max(mv_nz, 0.0))),
        }

    return {
        "phaseB_proj_mse_x0": float(mse_x0),
        "phaseB_proj_mse_noisy": float(mse_noisy),
        "phaseB_denoise_ratio": float(denoise_ratio),
        "phaseB_proj_rmse_x0": float(math.sqrt(max(mse_x0, 0.0))),
        "phaseB_proj_rmse_noisy": float(math.sqrt(max(mse_noisy, 0.0))),
        "phaseB_proj_per_view": per_view,
        "phaseB_future_latent_frames": int(latent_frames - mem_size),
        "phaseB_sigma_mode": sigma_mode,
        "phaseB_sigma_mean": float(sigmas.float().mean().item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate WM tactile decode-chain (Phase A) / projected-space x0 MSE (Phase B)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="step_N directory with DiT + projector")
    p.add_argument("--split", default="holdout_488")
    p.add_argument("--mode", choices=["phaseA", "phaseB"], required=True,
                   help="phaseA = decode-chain recall sanity (Phase A clean + roundtrip); "
                        "phaseB = projected-space x0 MSE (V1B primary WM-quality metric).")
    p.add_argument("--max_batches", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--sigma", default="random",
                   help="Phase B only: random | <float> (e.g. 0.1, 0.5). 'random' uses the trainer's flow-matching sampler.")
    p.add_argument("--flow_stats_path", default="data/stats/diverse_488/flow_stats.json")
    p.add_argument("--contact_threshold", type=float, default=0.5)
    p.add_argument("--tactile_view_offset", type=int, default=0)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    _ensure_dist_initialized()
    runner = _build_runner(args.config, args.flow_stats_path, args.batch_size)
    _load_ckpt(runner, args.checkpoint)

    if args.split not in runner.val_loaders:
        raise KeyError(f"split={args.split!r} not found. available={list(runner.val_loaders.keys())}")
    loader = runner.val_loaders[args.split]

    with open(args.flow_stats_path, "r") as f:
        fs = json.load(f)
    flow_mean = torch.tensor(fs["mean"], device=runner.state.accelerator.device, dtype=torch.float32)
    flow_std = torch.tensor(fs["std"], device=runner.state.accelerator.device, dtype=torch.float32)
    # _FLOW_CHANNELS_USED = (0, 1, 3) -> 3 channels (dx, dy, divergence).
    # If flow_stats.json stores all 4 channels (dx, dy, mag, div), select the
    # used channels here so _denormalize_flow's .view(...,3) is consistent
    # with the dataset's normalize path.
    if flow_mean.numel() == 4:
        idx = torch.tensor([0, 1, 3], device=flow_mean.device, dtype=torch.long)
        flow_mean = flow_mean.index_select(0, idx)
        flow_std = flow_std.index_select(0, idx)
    if flow_mean.numel() != 3 or flow_std.numel() != 3:
        raise ValueError(
            f"flow_stats.json must have mean/std of length 3 or 4; "
            f"got mean.numel={flow_mean.numel()} std.numel={flow_std.numel()}"
        )

    phase_a_clean_recalls: List[float] = []
    phase_a_rt_recalls: List[float] = []
    inv_errors: List[float] = []
    diags: List[Dict] = []
    pf_a_clean: List[List[float]] = []
    pf_a_rt: List[List[float]] = []

    phase_b_mse_x0: List[float] = []
    phase_b_mse_noisy: List[float] = []
    phase_b_ratio: List[float] = []
    phase_b_sigma_means: List[float] = []
    phase_b_per_view_acc: Dict[str, Dict[str, List[float]]] = {}
    phase_b_future_T: List[int] = []

    runner.diffusion_model.eval()
    if runner.projector is not None:
        runner.projector.eval()
    if runner.tactile_vae is not None:
        runner.tactile_vae.eval()

    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        out_a, _ = _phase_a_batch(
            runner, batch, flow_mean, flow_std,
            args.contact_threshold, args.tactile_view_offset,
        )
        phase_a_clean_recalls.append(out_a["phaseA_clean_recall_upper_bound"])
        phase_a_rt_recalls.append(out_a["phaseA_roundtrip_recall_upper_bound"])
        inv_errors.append(out_a["projector_inverse_error"])
        diags.append(out_a["projector_diag"])
        pf_a_clean.append(out_a["phaseA_clean_recall_pf"])
        pf_a_rt.append(out_a["phaseA_roundtrip_recall_pf"])

        if args.mode == "phaseA":
            logger.info(
                f"[phaseA] batch={i+1} clean={out_a['phaseA_clean_recall_upper_bound']:.6f} "
                f"roundtrip={out_a['phaseA_roundtrip_recall_upper_bound']:.6f} "
                f"inv_err={out_a['projector_inverse_error']:.6f}"
            )
        else:
            out_b = _phase_b_batch(runner, batch, args.sigma)
            phase_b_mse_x0.append(out_b["phaseB_proj_mse_x0"])
            phase_b_mse_noisy.append(out_b["phaseB_proj_mse_noisy"])
            phase_b_ratio.append(out_b["phaseB_denoise_ratio"])
            phase_b_sigma_means.append(out_b["phaseB_sigma_mean"])
            phase_b_future_T.append(out_b["phaseB_future_latent_frames"])
            for v_name, v_metrics in out_b["phaseB_proj_per_view"].items():
                bucket = phase_b_per_view_acc.setdefault(
                    v_name, {k: [] for k in v_metrics.keys()}
                )
                for k, vv in v_metrics.items():
                    bucket[k].append(float(vv))
            logger.info(
                f"[phaseB] batch={i+1} sigma_mean={out_b['phaseB_sigma_mean']:.4f} "
                f"mse_x0={out_b['phaseB_proj_mse_x0']:.4e} "
                f"mse_noisy={out_b['phaseB_proj_mse_noisy']:.4e} "
                f"denoise_ratio={out_b['phaseB_denoise_ratio']:.4f} "
                f"(phaseA_clean={out_a['phaseA_clean_recall_upper_bound']:.4f})"
            )

    def _mean_pf(stack):
        return torch.tensor(stack, dtype=torch.float32).mean(dim=0).tolist() if stack else []

    def _mean(xs):
        return float(sum(xs) / max(len(xs), 1))

    diag_mean = {}
    if diags:
        keys = diags[0].keys()
        diag_mean = {k: float(sum(d[k] for d in diags) / len(diags)) for k in keys}

    result = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_batches": args.max_batches,
        "sigma": args.sigma if args.mode == "phaseB" else None,
        "phaseA_clean_recall_upper_bound": _mean(phase_a_clean_recalls),
        "phaseA_roundtrip_recall_upper_bound": _mean(phase_a_rt_recalls),
        "projector_inverse_error": _mean(inv_errors),
        "projector_diag_mean": diag_mean,
        "phaseA_clean_recall_pf_mean": _mean_pf(pf_a_clean),
        "phaseA_roundtrip_recall_pf_mean": _mean_pf(pf_a_rt),
    }
    if args.mode == "phaseB":
        per_view_mean = {
            v: {k: _mean(vs) for k, vs in bucket.items()}
            for v, bucket in phase_b_per_view_acc.items()
        }
        mse_x0_mean = _mean(phase_b_mse_x0)
        mse_noisy_mean = _mean(phase_b_mse_noisy)
        result.update(
            {
                "phaseB_proj_mse_x0": mse_x0_mean,
                "phaseB_proj_mse_noisy": mse_noisy_mean,
                "phaseB_denoise_ratio": _mean(phase_b_ratio),
                "phaseB_denoise_ratio_from_means": (
                    mse_x0_mean / max(mse_noisy_mean, 1e-12)
                ),
                "phaseB_proj_rmse_x0": float(math.sqrt(max(mse_x0_mean, 0.0))),
                "phaseB_proj_rmse_noisy": float(math.sqrt(max(mse_noisy_mean, 0.0))),
                "phaseB_proj_per_view_mean": per_view_mean,
                "phaseB_sigma_mean": _mean(phase_b_sigma_means),
                "phaseB_future_latent_frames": (
                    int(phase_b_future_T[0]) if phase_b_future_T else 0
                ),
            }
        )

    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"saved report: {args.output_json}")

    # Cleanly tear down the dist process group. Without this, PyTorch
    # emits a non-fatal "destroy_process_group() was not called" warning
    # via sys.unraisablehook on interpreter shutdown, which is reported
    # by some PyTorch/NCCL builds as a non-zero process exit code and
    # breaks bash sweeps running with `set -e`.
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()

