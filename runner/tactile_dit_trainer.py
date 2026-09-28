# This file is an adaptation of Genie-Envisioner (AgibotTech) code that upstream
# licenses under CC BY-NC-SA 4.0, so the ShareAlike term applies and this file is
# distributed under the same licence rather than the repository's Apache 2.0:
# see LICENSES/CC-BY-NC-SA-4.0.txt. NonCommercial use only.
#
# Modified by the DexTacWAM Authors, 2026.

"""Stage 2 / Stage 3 tactile-into-DiT trainer (fork of ``runner/ge_trainer.py``).

Spec: ``stage_2_design_doc_bc5b3fc0.plan.md`` (LOCKED v2.2 with corrected
phase enum: ``video_only`` -> ``world_model_only``; Stage 3 phases
``action_only`` / ``action_full`` are first-class legal with or without
tactile views).

F8 decision (trainer-local hook):
    Visual + tactile latents are stacked along the ``(B*V)`` batch axis BEFORE
    ``forward_pass`` is called; ``n_view = V_rgb + V_hand``. NO modification to
    ``utils.model_utils.forward_pass`` or the DiT model itself. From the DiT's
    perspective, tactile views are indistinguishable from extra RGB views --
    a ``TactileProjector`` adds a learnable modality + per-view bias so the
    DiT can route them differently during attention.

Architecture::

    RGB:    batch['video']                                   (B, c, V_rgb, T, H, W)
            -> rearrange to (B*V_rgb, c, T, H, W)
            -> mem + future_video split
            -> visual VAE encode (frozen)
            -> latents_visual: (B*V_rgb, 128, T_lat, 6, 8)

    TAC:    batch['tactile']                                 (B, V_hand, F=5, T, H, W)
            -> mem-tactile + future-tactile split (mirrors visual mem/future)
            -> v0c-A.encode_per_hand (frozen, no_grad)
            -> latents_tactile: (B*V_hand, 128, T_lat, 6, 8)
            -> TactileProjector with view_idx=arange(V_hand)
            -> latents_tactile_projected (same shape)

    STACK:  latents = torch.cat([latents_visual, latents_tactile], dim=0)
                      shape (B*(V_rgb + V_hand), 128, T_lat, 6, 8)
            n_view = V_rgb + V_hand

    DENOISE: same noise schedule per-token; existing
             ``gen_noise_from_condition_frame_latent`` broadcasts over (B*V).

    PRED:   forward_pass(noisy_latents, n_view=V_rgb + V_hand) -> pred (BV, thw, c)

    LOSS (Stage 2 phases): split pred by view, MSE per-modality:
              loss = lambda_visual * mse(pred[:V_rgb], target[:V_rgb])
                   + lambda_tactile * mse(pred[V_rgb:], target[V_rgb:])

    LOSS (Stage 3 phases): action loss only.
              loss = action_loss_scale * mse(pred_action, target_action)
            ``lambda_visual`` / ``lambda_tactile`` are NO-OP in Stage 3 --
            tactile views serve as CONTEXT for the action head via DiT
            attention; they are NOT supervised by a denoise loss.

Phase enum (4 values, 2 stages):

  Stage 2 (world model with optional tactile co-denoise):
    ``tactile_projector_only`` -- Stage 2 phase 1 (~5k steps).
        Trainable: ``TactileProjector`` only. DiT FROZEN. Saves projector-only
        ckpt with provenance metadata.
        Mapping: ``args.train_mode = 'video_only'``.
    ``world_model_only``       -- Stage 2 phase 2 (~50k steps) OR strict GE
        world-model baseline.
        With ``use_tactile_views=true``: ``TactileProjector + DiT`` joint
        train (separate LR groups), loads phase-1 projector ckpt.
        With ``use_tactile_views=false``: strict GE baseline, no projector,
        no v0c-A, no tactile concat -- bit-for-bit GE pass-through.
        Mapping: ``args.train_mode = 'video_only'``.

  Stage 3 (action on top of world model):
    ``action_only``  -- only ``action_*`` DiT params trained; rest of DiT
        frozen. Action loss only.
        With ``use_tactile_views=true``: tactile views enter the DiT n_view
        stack as context. The action head reads the (tactile-enriched) DiT
        intermediate latent EXACTLY like GE's multi-view -- no architectural
        change to the action head. Projector loaded from a Stage-2 ckpt
        (``projector.warmstart_ckpt`` REQUIRED) and frozen by default; yaml
        knob ``projector.freeze_in_action_phase: false`` allows ablation.
        Mapping: ``args.train_mode = 'action_only'``.
    ``action_full``  -- every DiT param trained. Action loss only.
        Same tactile semantics as ``action_only`` (tactile is context, not
        denoise target). Projector default frozen, yaml-overridable.
        Mapping: ``args.train_mode = 'action_full'``.

Phase x use_tactile_views combo legality (only ONE invalid):

  +------------------------+------------------+--------------------+
  | phase                  | use_tactile=fals | use_tactile=true   |
  |                        | e                |                    |
  +------------------------+------------------+--------------------+
  | tactile_projector_only | INVALID          | OK (Stage 2 ph 1)  |
  | world_model_only       | OK (GE baseline) | OK (Stage 2 ph 2)  |
  | action_only            | OK (GE pass-thr) | OK (tactile ctx)   |
  | action_full            | OK (GE pass-thr) | OK (tactile ctx)   |
  +------------------------+------------------+--------------------+

The single invalid combo (``tactile_projector_only`` + ``use_tactile=false``)
is degenerate: there is no projector to learn against. All 7 other combos
are dispatched to a coherent training path.
"""

import os, random, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any, Dict, List, Optional

from datetime import datetime, timedelta
import argparse
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib

from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ---------------------------------------------------- Stage 2 additions
from models.tactile_models.projector import TactileProjector
from models.tactile_models.visual_vae_adapter import VisualVAEAdapterModel

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video

# ----------------------------------------------------
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

# ----------------------------------------------------
from utils.extra_utils import act_metric

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
logger = get_logger("wm_runner")
logger.setLevel(LOG_LEVEL)


# ---------------------------------------------------------------------------
# Phase enum (4 phases, 2 stages) + Stage / GE train_mode mappings
# ---------------------------------------------------------------------------

# All recognized trainer-public phase strings.
#
# Design: this trainer is a strict SUPERSET of GE's phase / train_mode space.
# Every GE-native phase (world_model_only / action_only / action_full) remains
# valid and runs with bit-for-bit GE behavior when use_tactile_views=false.
# Stage 2 adds ONE new phase ("tactile_projector_only") on top, plus an
# orthogonal tactile path that turns on via the use_tactile_views switch.
#
# Note: GE's INTERNAL `args.train_mode` still uses the legacy string
# "video_only" -- we mirror our public phase "world_model_only" onto it via
# `_PHASE_TO_GE_TRAIN_MODE` so the inherited GE train() loss dispatch keeps
# working bit-for-bit. The legacy name "video_only" is NOT exposed as a
# trainer-public phase value.
#
# Combo legality: every (phase, use_tactile_views) pair is legal except the
# degenerate `tactile_projector_only + use_tactile_views=false` (no projector
# to learn against). In particular, `action_only`/`action_full` with
# `use_tactile_views=true` is FIRST-CLASS legal: tactile views enter the DiT
# as additional views in the n_view stack, exactly like GE multi-view; the
# action head consumes the (tactile-enriched) DiT intermediate latent
# unchanged.
_VALID_PHASES = (
    "tactile_projector_only",   # Stage 2 phase 1; requires use_tactile_views=true
    "world_model_only",         # Stage 2 phase 2 (with tactile) OR GE world-model baseline (no tactile)
    "action_only",              # Stage 3: only action_* DiT params trained; tactile optional
    "action_full",              # Stage 3: every DiT param trained; tactile optional
)
_STAGE2_PHASES = ("tactile_projector_only", "world_model_only")
_STAGE3_PHASES = ("action_only", "action_full")
# Mapping from our trainer-public phase name to GE's INTERNAL `args.train_mode`
# string (used by the inherited GE train() loop's loss dispatch). Both Stage 2
# phases share the same GE train_mode `video_only` because they both supervise
# the video (co-denoise) loss; the difference is only WHICH params train.
_PHASE_TO_GE_TRAIN_MODE = {
    "tactile_projector_only": "video_only",
    "world_model_only":       "video_only",
    "action_only":            "action_only",
    "action_full":            "action_full",
}


def _load_v0c_a_frozen(
    vae,
    tactile_vae_config: Dict[str, Any],
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> VisualVAEAdapterModel:
    """Build v0c-A (or v0d) wrapper around ``vae`` and load its checkpoint.

    The returned model is in ``eval`` mode with ``requires_grad=False`` on all
    parameters; the Stage 2 trainer calls ``encode_per_hand`` under the
    method's own ``@torch.no_grad`` decorator so no gradients flow back into
    the adapter regardless of the surrounding autocast / accelerator config.

    v0d additions (pose injection, TimeSformer post-adapter, finger dropout)
    are opt-in via the ``adapter_use_*`` kwargs in ``tactile_vae_config``.
    Default values map to v0c-A behavior so existing cube configs are bit-
    for-bit identical to before. When any v0d flag is on, a ``[v0d-gate1]``
    log block is emitted with a positive proof of:
      * which v0d modules were instantiated (``use_pose_injection``,
        ``use_timesformer``, ``finger_dropout > 0``);
      * the presence and initial value of the alpha gates (``alpha_pose``,
        ``alpha_temp``);
      * the SHAPE of at least one v0d-specific weight loaded from the
        checkpoint, proving that the v0d portion of the ckpt actually
        landed (vs. silently dropped by ``strict=False``);
      * the sorted missing/unexpected key partitions, with explicit
        accounting for the v0c-A aux heads (``aux_*``) that we deliberately
        do NOT build into the Stage 2 wrapper.
    If a v0d flag is on but the corresponding weights are absent from the
    ckpt, a ``RuntimeError`` is raised here so we fail loudly BEFORE any
    training step instead of silently running a randomly-initialized v0d
    head against the rest of the pretrained network.

    Args:
        vae: pre-loaded ``AutoencoderKLLTXVideo`` instance to wrap. Sharing the
            visual VAE saves VRAM (no second full LTX VAE on the GPU); the VAE
            is already frozen by GE's ``prepare_models``.
        tactile_vae_config: dict from yaml ``tactile_vae.config``. All v0c-A
            knobs default to their published values; v0d knobs default to off
            so cube configs are unaffected. Supported keys:

              v0c-A (existing): ``latent_channels``, ``adapter_kind``,
              ``num_fingers``, ``num_heads``, ``spatial_h``, ``spatial_w``,
              ``use_finger_embed``, ``use_pos_query``, ``adapter_residual``,
              ``gray_to_rgb_init``, ``latent_mode``, ``adapter_n_layers``,
              ``adapter_ffn_dim``, ``adapter_dropout``.

              v0d (new): ``adapter_use_pose_injection`` (bool, default False),
              ``adapter_pose_dim`` (int, default 22),
              ``adapter_use_timesformer`` (bool, default False),
              ``adapter_timesformer_num_blocks`` (int, default 2),
              ``adapter_timesformer_num_heads`` (int, default 8),
              ``adapter_timesformer_ffn_dim`` (int, default 512),
              ``adapter_finger_dropout`` (float, default 0.0).
        model_path: path to a v0c-A / v0d ``.pt`` checkpoint. Loaded with
            ``strict=False`` so aux-head keys (``aux_*``) present in v0c-A
            ckpts are silently dropped (Stage 2 builds with
            ``enable_pre_fuse=False`` and never invokes aux heads). For v0d
            ckpts, the new keys (``adapter.pose_encoder.*``,
            ``adapter.timesformer_blocks.*``, ``adapter.alpha_pose``,
            ``adapter.alpha_temp``) MUST be present when the corresponding
            yaml flag is on -- otherwise this function raises.
        device: target device.
        dtype: target dtype (typically ``accelerator.weight_dtype``).

    Returns:
        Frozen ``VisualVAEAdapterModel`` ready for ``encode_per_hand``.

    Raises:
        RuntimeError: a v0d flag was enabled in the yaml but the corresponding
            v0d weights are absent from the loaded checkpoint -- the wrong
            ckpt path is configured.
    """
    cfg = tactile_vae_config or {}
    # ---- v0d flag extraction (used both for model construction and for the
    # Gate-1 logging block below) ----
    use_pose_injection = bool(cfg.get("adapter_use_pose_injection", False))
    use_timesformer = bool(cfg.get("adapter_use_timesformer", False))
    finger_dropout = float(cfg.get("adapter_finger_dropout", 0.0))
    pose_dim = int(cfg.get("adapter_pose_dim", 22))
    tf_num_blocks = int(cfg.get("adapter_timesformer_num_blocks", 2))
    tf_num_heads = int(cfg.get("adapter_timesformer_num_heads", 8))
    tf_ffn_dim = int(cfg.get("adapter_timesformer_ffn_dim", 512))
    any_v0d = use_pose_injection or use_timesformer or (finger_dropout > 0.0)

    model = VisualVAEAdapterModel(
        vae=vae,
        latent_channels=cfg.get("latent_channels", 128),
        adapter_kind=cfg.get("adapter_kind", "finger_set_transformer"),
        num_fingers=cfg.get("num_fingers", 5),
        num_heads=cfg.get("num_heads", 8),
        spatial_h=cfg.get("spatial_h", 6),
        spatial_w=cfg.get("spatial_w", 8),
        use_finger_embed=cfg.get("use_finger_embed", True),
        use_pos_query=cfg.get("use_pos_query", True),
        adapter_residual=cfg.get("adapter_residual", "weighted_mean"),
        gray_to_rgb_init=cfg.get("gray_to_rgb_init", "ones_repeat"),
        latent_mode=cfg.get("latent_mode", "mean"),
        # Stage 2 doesn't call aux heads; building with enable_pre_fuse=False
        # both saves params and makes the strict=False checkpoint load cleaner.
        enable_pre_fuse=False,
        adapter_n_layers=cfg.get("adapter_n_layers", 3),
        adapter_ffn_dim=cfg.get("adapter_ffn_dim", 1024),
        adapter_dropout=cfg.get("adapter_dropout", 0.0),
        # ---- v0d additions (all default off; only finger_set_transformer
        # adapter actually consumes these -- the VisualVAEAdapterModel
        # constructor raises ValueError if combined with a different
        # adapter_kind, so a misconfiguration fails loudly).
        adapter_use_pose_injection=use_pose_injection,
        adapter_pose_dim=pose_dim,
        adapter_use_timesformer=use_timesformer,
        adapter_timesformer_num_blocks=tf_num_blocks,
        adapter_timesformer_num_heads=tf_num_heads,
        adapter_timesformer_ffn_dim=tf_ffn_dim,
        adapter_finger_dropout=finger_dropout,
    )

    missing, unexpected = [], []
    sd_keys: List[str] = []
    if model_path:
        sd = torch.load(model_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd_keys = list(sd.keys())
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # `missing` is expected to include aux head keys (we built without them).
        # `unexpected` is a reverse case: ckpt has aux heads, model doesn't --
        # also expected and OK.
        if unexpected:
            aux_keys = [k for k in unexpected if k.startswith("aux_")]
            non_aux = [k for k in unexpected if not k.startswith("aux_")]
            if non_aux:
                logger.warning(
                    f"adapter ckpt has {len(non_aux)} unexpected non-aux keys "
                    f"(first 5: {non_aux[:5]}); proceeding but please verify."
                )
            logger.info(
                f"adapter ckpt: {len(aux_keys)} aux-head keys ignored (Stage 2 "
                f"doesn't use aux heads); {len(non_aux)} other unexpected."
            )
    else:
        logger.warning(
            "adapter model_path is empty -- using RANDOM weights. This is "
            "only valid for shape/structure smoke tests, NEVER for real "
            "training."
        )

    # =================================================================
    # Gate 1 -- positive proof that v0d modules actually loaded.
    # Always emitted (even for v0c-A configs) so the log carries a single
    # canonical record of which adapter variant the trainer is running.
    # =================================================================
    pose_enc_keys = [k for k in sd_keys if k.startswith("adapter.pose_encoder.")]
    timesformer_keys = [k for k in sd_keys if k.startswith("adapter.timesformer_blocks.")]
    has_alpha_pose_in_ckpt = "adapter.alpha_pose" in sd_keys
    has_alpha_temp_in_ckpt = "adapter.alpha_temp" in sd_keys

    # Model-side attribute presence + initial alpha values.
    # `alpha_pose` / `alpha_temp` are registered as buffers/params only when
    # the corresponding v0d flag is on; otherwise the wrapper exposes them
    # as `None` (still passes `hasattr`, but `.detach()` blows up). So we
    # also guard on `is not None` before reading the tensor value.
    inner_adapter = getattr(model, "adapter", None)
    model_use_pose_injection = bool(getattr(inner_adapter, "use_pose_injection", False))
    model_use_timesformer = bool(getattr(inner_adapter, "use_timesformer", False))
    model_alpha_pose_val: Optional[float] = None
    model_alpha_temp_val: Optional[float] = None
    alpha_pose_t = getattr(inner_adapter, "alpha_pose", None) if inner_adapter is not None else None
    alpha_temp_t = getattr(inner_adapter, "alpha_temp", None) if inner_adapter is not None else None
    if alpha_pose_t is not None:
        model_alpha_pose_val = float(alpha_pose_t.detach().cpu().item())
    if alpha_temp_t is not None:
        model_alpha_temp_val = float(alpha_temp_t.detach().cpu().item())

    # Pick a representative v0d weight to display the shape of (proof the
    # weight tensor actually got loaded, not silently dropped).
    pose_enc_repr_key = "adapter.pose_encoder.encoder.0.weight"
    pose_enc_repr_shape = None
    if pose_enc_repr_key in sd_keys:
        pose_enc_repr_shape = tuple(sd[pose_enc_repr_key].shape)
    tf_repr_key = next(
        (k for k in timesformer_keys if k.endswith(".proj.weight")),
        timesformer_keys[0] if timesformer_keys else None,
    )
    tf_repr_shape = tuple(sd[tf_repr_key].shape) if tf_repr_key else None

    # Sorted partitions for a deterministic log line.
    missing_sorted = sorted(missing)
    unexpected_sorted = sorted(unexpected)
    aux_unexpected = [k for k in unexpected_sorted if k.startswith("aux_")]
    nonaux_unexpected = [k for k in unexpected_sorted if not k.startswith("aux_")]

    logger.info(
        "[v0d-gate1] tactile adapter loaded:\n"
        f"  adapter_kind                 = {cfg.get('adapter_kind', 'finger_set_transformer')!r}\n"
        f"  use_pose_injection (yaml)    = {use_pose_injection}\n"
        f"  use_pose_injection (model)   = {model_use_pose_injection}\n"
        f"  use_timesformer (yaml)       = {use_timesformer}\n"
        f"  use_timesformer (model)      = {model_use_timesformer}\n"
        f"  timesformer_num_blocks       = {tf_num_blocks if use_timesformer else 0}\n"
        f"  timesformer_num_heads        = {tf_num_heads if use_timesformer else 0}\n"
        f"  finger_dropout               = {finger_dropout}\n"
        f"  pose_dim                     = {pose_dim if use_pose_injection else 0}\n"
        f"  alpha_pose in ckpt           = {has_alpha_pose_in_ckpt}  "
        f"(model value = {model_alpha_pose_val})\n"
        f"  alpha_temp in ckpt           = {has_alpha_temp_in_ckpt}  "
        f"(model value = {model_alpha_temp_val})\n"
        f"  pose_encoder weight in ckpt  = {pose_enc_repr_key if pose_enc_repr_shape else None}  "
        f"(shape = {pose_enc_repr_shape})\n"
        f"  timesformer weight in ckpt   = {tf_repr_key}  "
        f"(shape = {tf_repr_shape})\n"
        f"  total pose_encoder keys      = {len(pose_enc_keys)}\n"
        f"  total timesformer keys       = {len(timesformer_keys)}\n"
        f"  missing keys (total)         = {len(missing_sorted)}  "
        f"(first 5: {missing_sorted[:5]})\n"
        f"  unexpected aux keys          = {len(aux_unexpected)}  "
        f"(expected for v0c-A ckpts that include aux_pose / aux_flow_*)\n"
        f"  unexpected non-aux keys      = {len(nonaux_unexpected)}  "
        f"(first 5: {nonaux_unexpected[:5]})"
    )

    # ---- Hard fail when a v0d flag is on but the ckpt has none of the
    # corresponding weights. This catches the classic silent-failure mode:
    # yaml says pose_injection=true, model gets built with the new modules,
    # ckpt is v0c-A (no pose_encoder.*), strict=False drops nothing
    # because the v0d weights are MISSING from the ckpt (not unexpected),
    # and at training time we'd be training a randomly-initialized
    # pose_encoder against a fully-pretrained rest of the network.
    if any_v0d and model_path:
        if use_pose_injection and not pose_enc_keys:
            raise RuntimeError(
                "[v0d-gate1] FATAL: tactile_vae.config sets "
                "`adapter_use_pose_injection=True`, but the checkpoint at "
                f"{model_path!r} contains NO `adapter.pose_encoder.*` "
                "weights. This means the wrong ckpt path is set in the yaml "
                "(probably a v0c-A ckpt instead of v0d). Either fix the "
                "path to point at a v0d ckpt (e.g. "
                "`stage1_lite_v0d_full/.../best_recall_post/model.pt`), or "
                "set `adapter_use_pose_injection=false` to fall back to "
                "v0c-A behavior."
            )
        if use_timesformer and not timesformer_keys:
            raise RuntimeError(
                "[v0d-gate1] FATAL: tactile_vae.config sets "
                "`adapter_use_timesformer=True`, but the checkpoint at "
                f"{model_path!r} contains NO `adapter.timesformer_blocks.*` "
                "weights. The wrong ckpt is set in the yaml. Fix the path "
                "or disable timesformer."
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device=device, dtype=dtype)


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None



class _Stage2Bundle(torch.nn.Module):
    """Single-Module wrapper that lets DeepSpeed see DiT + projector as one model.

    HF accelerate's contract under the DeepSpeed backend allows EXACTLY ONE
    nn.Module per `Accelerator.prepare()` call (raises AssertionError
    "You can't use same `Accelerator()` instance with multiple models when
    using DeepSpeed" otherwise). Stage 2 has two trainable nn.Modules
    (`diffusion_model` for the DiT and `TactileProjector`); we register
    them as submodules of this container, hand the container to
    `prepare()`, then re-bind `self.diffusion_model` / `self.projector`
    to the inner submodules so the rest of the trainer code (which calls
    them directly) is unchanged.

    Why submodule access still gets DeepSpeed's gradient-reduction hooks:
        DeepSpeed registers hooks at the *parameter* level (via
        `register_post_accumulate_grad_hook` and friends), not at the
        engine's `forward()`. So a forward pass that calls
        `self.diffusion_model(x)` directly (bypassing the wrapping
        DeepSpeedEngine) still exercises the same parameters and the
        same hooks fire during backward.

    Why `forward()` raises:
        The trainer never calls `bundle.forward()`. Making it a hard
        error catches accidental misuse where someone wires
        `engine(batch)` thinking the engine is the DiT; they'd silently
        skip projector parameters in autograd otherwise.

    `accelerator.accumulate(...)`, `clip_grad_norm_`, and `.train()` /
    `.eval()` should be called on the *bundle* (or the wrapped engine
    that `prepare()` returns), not on the inner submodules. The trainer
    keeps a reference to the wrapped bundle as `self._prepared_model`
    for those callsites.
    """

    def __init__(self, diffusion_model: torch.nn.Module, projector: torch.nn.Module) -> None:
        super().__init__()
        self.diffusion_model = diffusion_model
        self.projector = projector

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "_Stage2Bundle.forward() should not be called directly. The "
            "trainer invokes `self.diffusion_model(...)` and "
            "`self.projector(...)` separately so DeepSpeed grad hooks "
            "fire per-parameter; calling the bundle's forward would "
            "skip one of the submodules' compute and silently break "
            "Stage 2 co-denoise."
        )


class TactileDiTTrainer:
    """Stage 2 / Stage 3 tactile-into-DiT trainer; fork of GE ``Trainer`` (see
    module docstring).

    Adds a phase-aware dispatcher (4 phases: ``tactile_projector_only`` /
    ``world_model_only`` for Stage 2, ``action_only`` / ``action_full`` for
    Stage 3) plus an orthogonal tactile path:
    v0c-A frozen encoder + ``TactileProjector`` + view-stack concat into the
    existing DiT ``n_view`` dimension.

    Strict superset of GE: every legal GE training config runs through this
    trainer with bit-for-bit identical behavior by setting
    ``use_tactile_views=false``. Tactile is opt-in via
    ``use_tactile_views=true`` and works in all phases except the degenerate
    ``tactile_projector_only + use_tactile_views=false`` combo.
    """

    def _validate_action_config(self) -> None:
        """Fail-fast on relative_eef_rot6d config invariants (no-op for absolute).

        Verifies the startup contract for the relative action mode so a config
        typo surfaces here instead of as an opaque shape mismatch mid-training.
        All widths come from ``data.train.arm_layout`` -- bimanual (rel 136 /
        state 90) or right_only (68 / 45):
          * tactile_inference.action_only_dim_override == rel_action_dim
          * diffusion_model action_in/out_channels == rel_action_dim + state_dim
            (226 bimanual, 113 right_only)
          * loss.action_dim_breakdown action dims sum to rel_action_dim and the
            state block is exactly [rel_action_dim:total]
        (the observed state_dim is cross-checked against arm_layout at runtime by
        the dataset.)
        """
        data_cfg = getattr(self.args, "data", None)
        train_cfg = data_cfg.get("train", {}) if isinstance(data_cfg, dict) else {}
        if train_cfg.get("action_type") != "relative_eef_rot6d":
            return
        # The relative-action dimension contract below (226-wide action expert,
        # action_dim_breakdown, action_only_dim_override) only applies to phases
        # that actually BUILD an action expert (action_only / action_full). A
        # world_model_only / tactile_projector_only config may legitimately set
        # action_type=relative_eef_rot6d so that a shared, action-agnostic WM lives
        # in the same *_eef_relative folder and reads the same stats file as the
        # paired action model: the dataset still relativizes the action window, but
        # this video-only phase (action_expert=false, return_action=false) DISCARDS
        # it, so the action-expert dim checks do not apply. (The dataset itself
        # still validates state_dim / relative-stats presence at runtime.)
        phase = getattr(self.args, "phase", None)
        if phase not in ("action_only", "action_full"):
            print(f"[relative_eef_rot6d] phase={phase!r}: no action expert -- the "
                  f"action window is relativized by the dataset but unused; "
                  f"skipping action-expert dim checks.")
            return
        from data.utils.relative_action import get_arm_layout
        # Same explicit key the dataset uses; default keeps existing bimanual configs
        # (which predate the key) validating exactly as before.
        L = get_arm_layout(train_cfg.get("arm_layout", "bimanual"))
        total = L.rel_action_dim + L.state_dim  # 226 bimanual, 113 right_only

        ti = getattr(self.args, "tactile_inference", {}) or {}
        override = ti.get("action_only_dim_override") if isinstance(ti, dict) else None
        assert override == L.rel_action_dim, (
            f"relative_eef_rot6d: tactile_inference.action_only_dim_override must "
            f"be {L.rel_action_dim}, got {override!r}")

        dm = getattr(self.args, "diffusion_model", {}) or {}
        dm_cfg = dm.get("config", {}) if isinstance(dm, dict) else {}
        assert dm_cfg.get("action_in_channels") == total and \
            dm_cfg.get("action_out_channels") == total, (
            f"relative_eef_rot6d: diffusion_model.config.action_in/out_channels "
            f"must both be {total} ({L.rel_action_dim} action + {L.state_dim} state "
            f"for arm_layout={L.name!r}), got "
            f"in={dm_cfg.get('action_in_channels')!r} out={dm_cfg.get('action_out_channels')!r}")

        loss_cfg = getattr(self.args, "loss", {}) or {}
        bd = loss_cfg.get("action_dim_breakdown", []) if isinstance(loss_cfg, dict) else []
        assert bd, "relative_eef_rot6d: loss.action_dim_breakdown is required"
        action_dims = sum(int(e[1]) - int(e[0]) for e in bd if e[2] != "state")
        assert action_dims == L.rel_action_dim, (
            f"relative_eef_rot6d: action_dim_breakdown action dims sum to "
            f"{action_dims}, expected {L.rel_action_dim}")
        state_entries = [e for e in bd if e[2] == "state"]
        assert len(state_entries) == 1 and (int(state_entries[0][0]), int(state_entries[0][1])) == (L.rel_action_dim, total), (
            f"relative_eef_rot6d: state block must be [{L.rel_action_dim}:{total}], "
            f"got {state_entries}")
        print(f"[relative_eef_rot6d] config validated: arm_layout={L.name!r}, "
              f"action_only_dim_override={override}, action_in/out={total}, "
              f"breakdown action dims={action_dims}")

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        self._validate_action_config()

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')
        self.state = State()

        # Gate-1 (v0d) one-shot flag: when True, the first call to
        # _encode_tactile_split has already printed the encode-side proof
        # block. Re-init here so successive train()/eval() runs from one
        # trainer instance re-log on the first encode of each phase.
        self._v0d_gate1_encode_logged = False

        # Gate-2 one-shot flags: prove train and val both call the SAME
        # `_forward_loss_batch` (same function id). One flag per branch
        # so each `training=True/False` call site emits exactly one log
        # line per trainer-instance lifetime.
        self._v0d_gate2_train_logged = False
        self._v0d_gate2_val_logged = False

        # A3 multi-split val: populated by `prepare_val_splits()` from
        # `self.args.data.get('val_splits', {})`. Empty `{}` means "no
        # multi-split val configured" -- the train loop's
        # `steps_to_val` hook then runs ONLY the legacy single-batch
        # `self.validate()` qualitative video sampling, and the new
        # multi-split block becomes a zero-iteration no-op. Initialised
        # here (before `prepare_dataset()` runs) so the attribute is
        # always defined regardless of which yaml the trainer is
        # dispatched against.
        self.val_loaders = {}

        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.unet = None
        self.vae = None
        self.scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if self.state.accelerator.is_main_process:

            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for k, v in args_dict.items():
                args_dict[k] = str(v)
            with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)
            
            if to_log:
                self.writer = SummaryWriter(log_dir=self.save_folder)
            else:
                self.writer = None

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

        init_logging(self.save_folder, rank=self.state.accelerator.process_index)


    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))  # 或 self.args.world_size
            grad_accum = self.args.gradient_accumulation_steps

            train_batch_size = per_device_bs * world_size * grad_accum
            self.args.deepspeed["train_batch_size"] = train_batch_size
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype


    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)
        

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir


    def prepare_dataset(self) -> None:

        logger.info(f"Training Dataset: {self.args.train_data_class}")
        local_rank = int(os.environ["LOCAL_RANK"])

        train_dataset_class = import_custom_class(
            self.args.train_data_class, self.args.train_data_class_path
        )
        self.train_dataset = train_dataset_class(**self.args.data['train'])

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.dataloader_num_workers,
            multiprocessing_context=None,
        )
        logger.info(f">>>>>>>>>>>>>Total Train Eps: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")


        if 'val' in self.args.data:
            self.prepare_val_dataset()

        # A3: independent of the legacy data.val single-batch Subset path.
        # Both can coexist; absence of val_splits keeps Stage-2/3 yamls
        # bit-identical.
        if 'val_splits' in self.args.data and self.args.data['val_splits']:
            self.prepare_val_splits()


    def prepare_val_splits(self) -> None:
        """A3 multi-split val: build self.val_loaders from data.val_splits.

        Each ``(name, kwargs)`` pair under ``args.data['val_splits']``
        is materialized into its own dataset + DataLoader, keyed by
        ``name``. The resulting ``self.val_loaders: Dict[str, DataLoader]``
        is consumed by the train loop at every ``steps_to_val`` interval:
        each split is fed independently to
        ``_compute_val_loss(loader, tag=name)`` so the val curves come
        out per-split (e.g. ``[val/holdout_488] ...``,
        ``[val/cube_val] ...``).

        Independence contract:

          * The legacy ``data.val`` single-batch Subset dataloader
            (built by ``prepare_val_dataset``) is UNTOUCHED and continues
            to feed ``self.validate()`` for qualitative video sampling.
          * The two pipelines never share state -- different dataset
            instances, different DataLoaders, different RNG snapshots
            (the A2 4-stream RNG fix in ``_compute_val_loss`` guarantees
            that even back-to-back multi-split calls do not pollute
            training RNG).
          * Order in the train loop: legacy validate (rank-0 only) ->
            barrier -> multi-split val (all ranks). The barrier is
            required because ``_compute_val_loss`` uses
            ``accelerator.reduce`` (NCCL allreduce) and would hang if
            non-main ranks entered while main was still inside
            ``self.validate()``.

        Backward-compatibility: when ``data.val_splits`` is absent or
        empty, this method is never called (gate in
        ``prepare_dataset``); ``self.val_loaders`` stays ``{}`` as
        initialized in ``__init__``. The train-loop multi-split block
        then iterates over zero splits and is a no-op. Every existing
        Stage-2 / Stage-3 yaml keeps its exact pre-A3 behavior.

        Dataset class resolution:
            Uses ``self.args.val_data_class`` / ``val_data_class_path``
            when defined (same convention as ``prepare_val_dataset``);
            otherwise falls back to the train dataset class. Each
            split's full kwargs dict is passed straight through to the
            dataset constructor (including the new ``episodes`` filter
            kwarg from A3.0). Construction failures (e.g. requested
            episode indices missing from the corpus) propagate up so
            yaml typos fail loud at trainer construction rather than
            silently producing an empty loader.
        """
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        if not hasattr(self.args, "val_data_class_path"):
            self.args.val_data_class_path = self.args.train_data_class_path
        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path,
        )

        splits_cfg = self.args.data['val_splits']
        if not isinstance(splits_cfg, dict):
            raise TypeError(
                f"args.data['val_splits'] must be a dict mapping split "
                f"names to dataset kwargs; got "
                f"{type(splits_cfg).__name__}."
            )

        self.val_loaders = {}
        for split_name, split_kwargs in splits_cfg.items():
            if not isinstance(split_kwargs, dict):
                raise TypeError(
                    f"args.data['val_splits'][{split_name!r}] must be a "
                    f"dict of dataset kwargs; got "
                    f"{type(split_kwargs).__name__}."
                )
            # Fail-loud on construction errors so a typo in the yaml
            # (e.g. wrong episode_index list, missing pose_stats_path)
            # surfaces at trainer-init time rather than silently
            # producing an empty loader and a misleading "all good"
            # smoke pass.
            ds = val_dataset_class(**split_kwargs)
            loader = torch.utils.data.DataLoader(
                dataset=ds,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=getattr(self.args, "pin_memory", False),
                persistent_workers=getattr(
                    self.args, "persistent_workers", False,
                ),
            )
            self.val_loaders[split_name] = loader
            # Per-split summary log. The dataset class already prints
            # `[DexVTAMDataset] episode filter active: kept ...` when
            # the filter is in use, so we don't duplicate that list
            # here; just confirm the split was built with the expected
            # sample count.
            kept = getattr(ds, "_kept_episode_indices", None)
            filter_str = (
                f" (episode filter: {len(kept)} eps)"
                if kept is not None
                else ""
            )
            logger.info(
                f"[val_splits] built '{split_name}': "
                f"{len(ds)} samples, "
                f"dataset_class={val_dataset_class.__name__}"
                f"{filter_str}"
            )


    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        self.val_dataset = val_dataset_class(**self.args.data['val'])

        self.val_index = []
        for _ in range(self.args.batch_size):
            self.val_index.append(random.randint(0, len(self.val_dataset)-1))
        if self.state.accelerator.is_main_process:
            with open(os.path.join(self.save_folder, 'idx.txt'), "w") as file:
                file.write(", ".join(map(str, self.val_index)))

        subset = torch.utils.data.Subset(self.val_dataset, self.val_index)
        self.val_dataloader = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size, shuffle=getattr(self.args, "val_shuffle", False)
        )
        logger.info(f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")


    def prepare_models(self):

        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=True
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        logger.info(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        logger.info(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        logger.info(f'Total parameters for transformer model:{total_params}')

        # Option A: dual cross-attn startup banner. Mirrors how `disable_projector`
        # logs a one-line provenance string above (see plan
        # cube_tactile_slowdown_ablation_1b64937f -> option-a-dual-crossattn_36ab43c4).
        # Emitting this AFTER the warmstart load so a future Probe N can grep
        # both the dual-attn flag AND the missing/unexpected key counts from
        # `load_checkpoints` to cross-check that `attn2_visual.*` /
        # `attn2_tactile.*` / `tactile_gate` keys actually landed as random
        # init (cube WM warmstart has no action_blocks at all, so they
        # always do for the Probe M setup).
        if getattr(self.diffusion_model, "dual_cross_attn", False):
            try:
                _tg = float(self.diffusion_model.action_blocks[0].tactile_gate.detach().cpu().item())
            except Exception:
                _tg = float("nan")
            logger.info(
                "[option-a-dual-crossattn] diffusion_model.dual_cross_attn=True. "
                "Action expert uses per-block (attn2_visual, attn2_tactile, "
                f"tactile_gate); tactile_gate[0]={_tg:.4f} "
                f"(sigmoid={torch.sigmoid(torch.tensor(_tg)).item():.4f}). "
                "Trainer will forward `n_view_visual` so the DiT can split "
                "the action K/V into visual vs tactile halves before the "
                "action-block call. D1b debug hook auto-disabled in this "
                "mode because it targets the legacy `.attn2` module."
            )
            # Probe N's `[probe-n-action-visual-only]` banner is also emitted
            # from inside `LTXVideoTransformer3DModel.__init__` via
            # `logging.getLogger(__name__)`, which (under the trainer's logger
            # config) often doesn't propagate to the main run log file. Re-emit
            # it here through the `wm_runner` logger so it co-locates with the
            # Option A banner and is grep-able from the same log file. We mirror
            # the same gating logic the model uses (action_visual_only AND
            # dual_cross_attn AND action_expert) so a false positive is
            # impossible.
            if bool(getattr(self.diffusion_model, "action_visual_only", False)):
                logger.info(
                    "[probe-n-action-visual-only] diffusion_model.action_visual_only=True. "
                    "tactile views still flow through transformer_blocks (WM body "
                    "sees both visual and tactile), but action_blocks consume "
                    "visual-only K/V (final_tactile forced to None in "
                    "_split_action_kv -> ActionTransformerBlock skips attn2_tactile "
                    "+ tactile_gate residual). tactile_gate is therefore frozen at "
                    f"init ({_tg:.4f}) for the whole run because it receives no "
                    "gradient. Compare to Probe M (same yaml minus this flag): if "
                    "Probe N loss curve matches Probe M, the slowdown is driven by "
                    "tactile pollution INSIDE the WM body, NOT by the action "
                    "expert's tactile cross-attn."
                )


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )

        ### -------------------------------------------------------------
        ### Stage 2 phase dispatcher + tactile path instantiation
        ### -------------------------------------------------------------
        self._stage2_prepare_tactile(device=device, dtype=dtype)


    def _stage2_prepare_tactile(self, device: torch.device, dtype: torch.dtype) -> None:
        """Dispatch on ``self.args.phase`` and ``self.args.use_tactile_views``.

        Constructs ``self.tactile_vae`` (frozen v0c-A) and ``self.projector``
        only when the tactile path is enabled. Per-phase rules:

          tactile_projector_only:
              DiT frozen, projector trainable. Requires ``use_tactile=true``.
          world_model_only + tactile=true:
              DiT trainable, projector trainable. Loads phase-1 projector
              ckpt with provenance check (unless ``projector.from_scratch=true``).
          world_model_only + tactile=false:
              GE world-model baseline. No projector, no v0c-A, no concat.
          action_only / action_full + tactile=true:
              Tactile views enter DiT as context (NOT denoise target). The
              projector is loaded from a Stage-2 ckpt (``warmstart_ckpt`` is
              REQUIRED -- hard-fail; no random fallback) and frozen by default
              (yaml knob ``projector.freeze_in_action_phase`` overrides).
              ``args.loss.lambda_visual`` / ``lambda_tactile`` are NO-OP in
              Stage 3 (warning logged if non-zero).
          action_only / action_full + tactile=false:
              Vanilla GE pass-through. No projector, no v0c-A.

        The only invalid combo is ``tactile_projector_only + use_tactile=false``
        (degenerate -- no projector to learn against).
        """
        phase = getattr(self.args, "phase", None)
        if phase is None:
            raise ValueError(
                "Trainer yaml must specify `phase` (one of "
                f"{list(_VALID_PHASES)})."
            )
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}; must be one of "
                f"{list(_VALID_PHASES)}."
            )

        use_tactile = bool(getattr(self.args, "use_tactile_views", False))

        # The single invalid combo: phase 1 (tactile_projector_only) needs
        # tactile views, since the projector has nothing to learn against
        # otherwise. All other 7 (phase, use_tactile_views) combos are legal.
        if phase == "tactile_projector_only" and not use_tactile:
            raise ValueError(
                "phase=tactile_projector_only requires use_tactile_views=true; "
                "there is nothing for the projector to learn against otherwise. "
                "If you want a strict GE world-model baseline run, use "
                "phase=world_model_only with use_tactile_views=false instead."
            )

        # Mirror our trainer-public `phase` onto GE's INTERNAL `args.train_mode`
        # field so the inherited GE train() loop's loss dispatch
        # (`if train_mode == 'video_only'` / `'action_only'` / `'action_full'`)
        # keeps working bit-for-bit. We never overwrite an explicit user value.
        if not hasattr(self.args, "train_mode") or self.args.train_mode is None:
            self.args.train_mode = _PHASE_TO_GE_TRAIN_MODE[phase]

        # Stage 3 lambda strictness: lambda_visual / lambda_tactile are NO-OP
        # in action phases (tactile is context, not denoise target). If the
        # yaml has them set, log a warning so users don't silently assume the
        # loss is wired -- Stage 3 computes action loss only.
        if phase in _STAGE3_PHASES:
            loss_cfg = self._stage2_dict_arg("loss")
            lambda_v = float(loss_cfg.get("lambda_visual", 0.0) or 0.0)
            lambda_t = float(loss_cfg.get("lambda_tactile", 0.0) or 0.0)
            if lambda_v != 0.0 or lambda_t != 0.0:
                logger.warning(
                    f"phase={phase!r} (Stage 3): "
                    f"loss.lambda_visual={lambda_v}, "
                    f"loss.lambda_tactile={lambda_t} are IGNORED. Stage 3 "
                    f"phases compute action loss only; denoise losses are "
                    f"not active. Tactile views (when use_tactile_views=true) "
                    f"serve as CONTEXT for the action head via DiT attention, "
                    f"not as a co-denoise target."
                )

        # ----- v0c-A + TactileProjector instantiation ------------------
        if use_tactile:
            tactile_vae_args = self._stage2_dict_arg("tactile_vae")
            projector_args = self._stage2_dict_arg("projector")

            # E_proj_identity falsification probe (cube tactile slowdown
            # ablation, plan cube_tactile_slowdown_ablation_1b64937f,
            # 2026-05-24 late). When set, `_encode_tactile_split` skips
            # the `self.projector(tac_latent_pre, view_idx)` call and
            # passes `tac_latent_pre` through with the same shape
            # contract. The projector module is still instantiated and
            # warmstart-loaded below (66k params, negligible) so the
            # Stage 3 + tactile validation path / _Stage2Bundle schema
            # are unchanged.
            #
            # Default False keeps every existing yaml byte-identical in
            # behaviour. Only the new
            # `action_model_diverse_488_tactile_v0d_proj_identity_dryrun.yaml`
            # sets it true. We surface a single-line WARNING (not INFO)
            # if the flag is on, so the bypass is unmistakable in a
            # green-field debugger glance.
            self._disable_projector_for_action = bool(
                (tactile_vae_args.get("config", {}) or {}).get(
                    "disable_projector", False,
                )
            )
            if self._disable_projector_for_action:
                logger.warning(
                    "[E_proj_identity] tactile_vae.config.disable_projector "
                    "= true. _encode_tactile_split will BYPASS the "
                    "TactileProjector and pass tac_latent_pre straight to "
                    "the DiT (tac_latent := tac_latent_pre). The projector "
                    "module is still constructed and warmstart-loaded for "
                    "schema compatibility but is never called in forward. "
                    "Expected at-val outcome: out.batch_std == in.batch_std "
                    "(~0.12 on holdout_488 v0d). This is a falsification "
                    "probe for the 'projector is the action-loss "
                    "bottleneck' hypothesis, not a production knob."
                )

            # Fail-fast: Stage 3 + tactile=true REQUIRES warmstart_ckpt
            # UNLESS the projector is being bypassed via
            # `tactile_vae.config.disable_projector: true`. In bypass mode
            # `_encode_tactile_split` passes `tac_latent_pre` straight to the
            # DiT and the projector module is never called in forward, so
            # whether it was warmstarted vs random-init makes ZERO difference
            # to the model's behavior -- requiring a warmstart_ckpt would be
            # vacuous. The projector is still instantiated (to keep the
            # _Stage2Bundle schema stable) but its state is irrelevant.
            #
            # Check this BEFORE loading the heavy v0c-A model, so a misrouted
            # yaml fails in milliseconds rather than after the VAE is on GPU.
            if phase in _STAGE3_PHASES:
                if not projector_args.get("warmstart_ckpt"):
                    if not self._disable_projector_for_action:
                        raise ValueError(
                            f"phase={phase!r} + use_tactile_views=true REQUIRES "
                            f"projector.warmstart_ckpt to be set (path to a "
                            f"Stage-2 projector ckpt). Random-init projector in "
                            f"Stage 3 is forbidden because action results would "
                            f"be ambiguous: cannot tell whether tactile is "
                            f"unhelpful or simply not warmstart'd. To explicitly "
                            f"override, either (a) set "
                            f"`tactile_vae.config.disable_projector: true` to "
                            f"bypass the projector in forward (typical for "
                            f"WM-bypass action runs), or (b) point "
                            f"warmstart_ckpt at a deliberately-empty placeholder "
                            f"ckpt (rare ablation)."
                        )
                    logger.warning(
                        "[E_proj_identity] Stage 3 + use_tactile_views=true + "
                        "disable_projector=true + warmstart_ckpt=null: "
                        "skipping the warmstart_ckpt requirement because the "
                        "projector is bypassed in forward (its state is "
                        "irrelevant to action behavior)."
                    )

            self.tactile_vae = _load_v0c_a_frozen(
                vae=self.vae,
                tactile_vae_config=tactile_vae_args.get("config", {}) or {},
                model_path=tactile_vae_args.get("model_path", ""),
                device=device,
                dtype=dtype,
            )

            self.projector = TactileProjector(
                latent_dim=int(projector_args.get("latent_dim", 128)),
                hidden_dim=int(projector_args.get("hidden_dim", 256)),
                num_views=int(projector_args.get("num_views", 2)),
            ).to(device=device, dtype=torch.float32)
            # Trainable params kept in fp32 master copy; autocast (set up by
            # accelerator) handles bf16/fp16 forward / backward as needed.
            # This matches GE's `cast_training_params(...)` discipline.

            if phase == "world_model_only":
                # Stage 2 phase 2: warmstart from phase-1 projector ckpt
                # (production path) UNLESS explicit opt-out for ablation.
                if not projector_args.get("from_scratch", False):
                    self._load_projector_warmstart(
                        projector_args, tactile_vae_args
                    )
                else:
                    logger.warning(
                        "phase=world_model_only + use_tactile_views=true + "
                        "projector.from_scratch=true: projector starts from "
                        "RANDOM init in phase 2. This is the opt-out escape "
                        "hatch (rare ablation), not the production path."
                    )

                # B2-style freeze: keep the warmstarted projector as a fixed
                # encoder while only the DiT learns world dynamics. Default
                # False preserves the original cube Phase-2 behavior (projector
                # jointly trainable).
                if projector_args.get("freeze_in_wm_phase", False):
                    self.projector.requires_grad_(False)
                    n_proj_trainable = sum(
                        p.numel()
                        for p in self.projector.parameters()
                        if p.requires_grad
                    )
                    logger.info(
                        "phase=world_model_only: projector FROZEN "
                        "(freeze_in_wm_phase=true). Only DiT params are "
                        f"trainable. projector trainable params after "
                        f"freeze: {n_proj_trainable}"
                    )

            elif phase in _STAGE3_PHASES:
                # warmstart_ckpt presence already verified above; this load
                # cannot raise on a missing path (only on schema/provenance
                # mismatch, which is correct fail-loud behavior).
                #
                # Bypass-mode exception: when `disable_projector=true`, the
                # fail-fast guard above lets `warmstart_ckpt=null` through
                # (the projector is never called in forward, so its state is
                # irrelevant). In that case we skip the load entirely -- the
                # projector remains at its torch-default random init, which
                # is fine because no forward path ever consults it.
                if projector_args.get("warmstart_ckpt"):
                    self._load_projector_warmstart(projector_args, tactile_vae_args)
                else:
                    assert self._disable_projector_for_action, (
                        "Stage 3 reached the warmstart-load block with "
                        "warmstart_ckpt=null but disable_projector=false; "
                        "this combination should have been caught by the "
                        "fail-fast guard above."
                    )
                    logger.warning(
                        "[E_proj_identity] Skipping _load_projector_warmstart "
                        "in Stage 3 because disable_projector=true. The "
                        "projector module is constructed for schema stability "
                        "but never called in forward."
                    )

                # Default: freeze projector to preserve Stage-2 tactile
                # alignment. yaml knob `projector.freeze_in_action_phase`
                # toggles for ablations.
                freeze_projector = bool(
                    projector_args.get("freeze_in_action_phase", True)
                )
                if freeze_projector:
                    self.projector.requires_grad_(False)
                    logger.info(
                        f"phase={phase!r} + use_tactile_views=true: projector "
                        f"FROZEN (default). Action loss does not update the "
                        f"tactile projector. Set "
                        f"projector.freeze_in_action_phase=false to override."
                    )
                else:
                    logger.info(
                        f"phase={phase!r} + use_tactile_views=true: projector "
                        f"TRAINABLE (yaml override). Action loss will update "
                        f"the tactile projector via DiT attention -> tactile "
                        f"view tokens path."
                    )
        else:
            self.tactile_vae = None
            self.projector = None
            # E_proj_identity: keep the attribute defined on the
            # no-tactile path so other code paths can read it safely.
            # The flag has no effect when use_tactile=False because
            # `_encode_tactile_split` is never reached.
            self._disable_projector_for_action = False

        # ----- Per-phase DiT-body global requires_grad ----------------
        # The exact per-name DiT filter (action_* in/out) is applied later in
        # prepare_optimizer. Here we only toggle the global requires_grad for
        # tactile_projector_only (DiT entirely frozen).
        if phase == "tactile_projector_only":
            self.diffusion_model.requires_grad_(False)
            logger.info(
                "phase=tactile_projector_only: DiT FROZEN, only "
                "TactileProjector is trainable."
            )
        else:
            # world_model_only / action_only / action_full: DiT params will be
            # selectively re-frozen in prepare_optimizer based on phase.
            self.diffusion_model.requires_grad_(True)

        logger.info(
            f"phase dispatch: phase={phase!r}, "
            f"use_tactile_views={use_tactile}, "
            f"projector_present={self.projector is not None}, "
            f"tactile_vae_present={self.tactile_vae is not None}, "
            f"args.train_mode={self.args.train_mode!r} (GE-internal)."
        )

    def _stage2_dict_arg(self, name: str) -> Dict[str, Any]:
        """Read a top-level dict argument from ``self.args``, returning ``{}``
        if it's missing or ``None``.

        ``argparse.Namespace`` doesn't enforce that nested yaml mappings stay
        as dicts; this helper canonicalizes to a dict so dispatch code can
        always do ``.get(...)``.
        """
        val = getattr(self.args, name, None)
        if val is None:
            return {}
        if not isinstance(val, dict):
            raise TypeError(
                f"args.{name} must be a dict (yaml mapping); got "
                f"{type(val).__name__}."
            )
        return val

    def _load_projector_warmstart(
        self,
        projector_args: Dict[str, Any],
        tactile_vae_args: Dict[str, Any],
    ) -> None:
        """Load a Stage-2-trained projector ckpt with provenance check.

        Used by:
          - ``world_model_only`` (Stage 2 phase 2): loads phase-1 projector
            (saved by ``_save_projector_ckpt``).
          - ``action_only`` / ``action_full`` (Stage 3): loads a projector
            from a Stage-2 run (typically a ``world_model_only`` run that
            ran the projector in joint training).

        The Stage-2 ckpt schema is::

            {
                "projector": projector.state_dict(),
                "tactile_vae_model_path": str,    # provenance
                "step": int,
                "phase": "tactile_projector_only" | "world_model_only",
            }

        Provenance check ensures the v0c-A path the projector was trained
        AGAINST matches the v0c-A path THIS yaml points at. Mismatch
        silently breaks the projector's learned latent distribution; we
        abort loudly unless ``projector.allow_v0c_a_mismatch=true`` is set
        explicitly (rare ablation).
        """
        ckpt_path = projector_args.get("warmstart_ckpt")
        if ckpt_path is None or not os.path.exists(ckpt_path):
            phase = getattr(self.args, "phase", "?")
            raise FileNotFoundError(
                f"phase={phase!r} with use_tactile_views=true requires a "
                f"valid projector.warmstart_ckpt path; got "
                f"warmstart_ckpt={ckpt_path!r}. (Stage 2 phase 2 may opt out "
                f"with projector.from_scratch=true; Stage 3 phases never "
                f"may.)"
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "projector" not in ckpt:
            raise ValueError(
                f"warmstart_ckpt {ckpt_path!r} does not contain a 'projector' "
                f"key; expected schema: "
                f"{{projector, tactile_vae_model_path, step, phase}}."
            )

        saved_phase = ckpt.get("phase")
        # Accept any Stage-2 phase as a valid warmstart source. Stage 3 is
        # forbidden (would be loading an action-tuned projector into another
        # action run -- not a "warmstart").
        if saved_phase not in _STAGE2_PHASES:
            raise ValueError(
                f"warmstart_ckpt must be from a Stage-2 run "
                f"(phase in {list(_STAGE2_PHASES)}); got phase={saved_phase!r} "
                f"at {ckpt_path!r}."
            )

        if not projector_args.get("allow_v0c_a_mismatch", False):
            saved_v0c = ckpt.get("tactile_vae_model_path")
            yaml_v0c = tactile_vae_args.get("model_path")
            if saved_v0c != yaml_v0c:
                raise ValueError(
                    f"warmstart projector was trained against v0c-A at\n"
                    f"  {saved_v0c!r}\n"
                    f"but the current yaml points at\n"
                    f"  {yaml_v0c!r}\n"
                    f"This silently breaks the projector's learned latent "
                    f"distribution. Set projector.allow_v0c_a_mismatch=true "
                    f"to override (rare; ablation only)."
                )

        # Load into the (still bare, unprepared) projector module. accelerator.prepare
        # is called later in prepare_for_training; loading before prepare keeps
        # state_dict keys clean (no DDP / DeepSpeed prefix).
        self.projector.load_state_dict(ckpt["projector"])
        logger.info(
            f"loaded projector warmstart: source_phase={saved_phase!r}, "
            f"step={ckpt.get('step', '?')} from {ckpt_path}"
        )


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        components_to_disable_grads = []
            
        for component in components_to_disable_grads:
            if component is not None:
                component.requires_grad_(False)

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True


    def prepare_optimizer(self):
        """Phase-aware optimizer (strict superset of GE).

        Phase semantics:
          tactile_projector_only:   ONLY projector parameters in optimizer;
                                    DiT fully frozen.
          world_model_only:         GE 'video_only' DiT filter (action_*
                                    excluded) + projector group when
                                    use_tactile_views=true; DiT-only when
                                    use_tactile_views=false (strict GE
                                    world-model baseline).
          action_only:              GE 'action_only' DiT filter (only
                                    action_* trained). With
                                    use_tactile_views=true, projector follows
                                    `freeze_in_action_phase` (default frozen,
                                    so no projector group; yaml override
                                    adds it).
          action_full:              GE 'all'/'action_full' filter (every DiT
                                    param trained). Same projector handling
                                    as action_only.

        LR config (under yaml top-level ``optim:``):
          - lr_projector: float    -- projector group LR
          - lr_dit: float          -- DiT group LR (read but ignored in phase 1)
          - train_steps: int       -- total optimizer steps for this launch

        Falls back to ``self.args.lr`` / ``self.args.train_steps`` when
        ``optim:`` is missing -- preserves bit-for-bit compatibility when this
        trainer is invoked with a vanilla GE yaml (sanity check 1 / GE
        action_* pass-through paths).
        """
        logger.info("Initializing optimizer and lr scheduler (phase-aware)")

        phase = self.args.phase
        optim_cfg = self._stage2_dict_arg("optim")
        # Backward-compat: pull lr from yaml's `lr` field if `optim:` is missing.
        # Allows running a vanilla GE yaml through this trainer for sanity
        # check 1 and for GE action_* pass-through experiments.
        lr_projector = float(optim_cfg.get("lr_projector", self.args.lr))
        lr_dit       = float(optim_cfg.get("lr_dit",       self.args.lr))

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = optim_cfg.get("train_steps", self.args.train_steps)

        # Make sure trainable params are fp32 master copies under fp16
        # autocast (matches GE base behavior; bf16 path is naturally handled).
        if self.args.mixed_precision == "fp16":
            casts = []
            if phase != "tactile_projector_only":
                casts.append(self.diffusion_model)
            if self.projector is not None:
                casts.append(self.projector)
            if casts:
                cast_training_params(casts, dtype=torch.float32)

        # ------ Build per-group trainable param lists ------------------
        # Per-phase DiT filter; mirrors GE's prepare_optimizer logic for the
        # action_* phases and adds the new tactile_projector_only branch.
        param_groups: List[Dict[str, Any]] = []
        diffusion_model_trainable_params: List[torch.nn.Parameter] = []

        if phase == "tactile_projector_only":
            # DiT fully frozen (already done in _stage2_prepare_tactile, but
            # re-assert here for defense in depth).
            for _name, param in self.diffusion_model.named_parameters():
                param.requires_grad = False
            primary_lr = lr_projector
        elif phase == "world_model_only":
            # GE 'video_only' filter: train everything EXCEPT action heads.
            # use_tactile_views=true adds the projector group below;
            # use_tactile_views=false yields strict GE world-model baseline.
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' not in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
            primary_lr = lr_dit
        elif phase == "action_only":
            # GE 'action_only' filter: train ONLY action_* params.
            # With use_tactile_views=true, projector is loaded from a Stage-2
            # ckpt and frozen by default (yaml-overridable in
            # _stage2_prepare_tactile). Tactile views feed the DiT as
            # additional views in the n_view stack; the action head reads
            # the DiT intermediate latent unchanged.
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
            primary_lr = lr_dit
        elif phase == "action_full":
            # GE 'all' / 'action_full' filter: every DiT param is trained.
            # Same tactile semantics as action_only (tactile is context, not
            # denoise target; projector default frozen).
            for _name, param in self.diffusion_model.named_parameters():
                param.requires_grad = True
                diffusion_model_trainable_params.append(param)
            primary_lr = lr_dit
        else:
            # Unreachable (validated in _stage2_prepare_tactile + _VALID_PHASES).
            raise NotImplementedError(
                f"prepare_optimizer cannot dispatch phase={phase!r}."
            )

        # `state.learning_rate` is used by GE training-loop logging + scale_lr.
        # We expose the "primary" group LR (the DiT group when present, else
        # projector LR) so existing GE logging code keeps working.
        self.state.learning_rate = primary_lr
        if self.args.scale_lr:
            scale = (
                self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )
            lr_projector = lr_projector * scale
            lr_dit = lr_dit * scale
            self.state.learning_rate = self.state.learning_rate * scale

        # DiT group (any phase that produced trainable DiT params).
        if diffusion_model_trainable_params:
            param_groups.append({
                "params": diffusion_model_trainable_params,
                "lr": lr_dit,
                "name": "dit",
            })

        # Projector group: ONLY add if it has any param with requires_grad=True.
        # In Stage 3 with default `freeze_in_action_phase=true`, the dispatcher
        # has already called `self.projector.requires_grad_(False)`, so the
        # comprehension below yields an empty list and the projector is NOT
        # added to the optimizer (correct: a frozen module has no business
        # in an optimizer's param groups).
        if self.projector is not None:
            projector_params = [p for p in self.projector.parameters() if p.requires_grad]
            if projector_params:
                param_groups.append({
                    "params": projector_params,
                    "lr": lr_projector,
                    "name": "projector",
                })

        if not param_groups:
            raise ValueError(
                f"prepare_optimizer produced ZERO trainable param groups for "
                f"phase={phase!r}, use_tactile_views="
                f"{getattr(self.args, 'use_tactile_views', False)}. This is "
                f"never a valid configuration."
            )

        num_trainable_params = sum(p.numel() for grp in param_groups for p in grp["params"])
        logger.info(
            f"trainable params (phase={phase}): {num_trainable_params:,} "
            f"across {len(param_groups)} group(s): "
            f"{[g['name'] for g in param_groups]}"
        )
        self.state.num_trainable_parameters = num_trainable_params

        optimizer = get_optimizer(
            params_to_optimize=param_groups,
            optimizer_name=self.args.optimizer,
            learning_rate=primary_lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        

    def prepare_for_training(self):
        # Stage 2: include projector in accelerator.prepare when present so
        # DDP / DeepSpeed wrapping is consistent across all trainable modules.
        #
        # Accelerate's DeepSpeed backend allows exactly one nn.Module per
        # prepare() call. With use_tactile_views=true we have two trainable
        # modules (DiT + projector), so we wrap them in `_Stage2Bundle` and
        # let the engine wrap the bundle. After prepare() we re-bind
        # `self.diffusion_model` / `self.projector` to the inner submodules
        # for the rest of the trainer (which calls them directly), AND we
        # stash a reference to the wrapped bundle as `self._prepared_model`
        # for the few callsites that need the engine itself
        # (`accelerator.accumulate(...)`, `.train()`, `clip_grad_norm_(...)`,
        # i.e. anything that needs DeepSpeed's gradient sync / mode toggle
        # at the engine level rather than per-submodule).
        #
        # Legacy path (use_tactile_views=false): identical to GE's original
        # 4-tuple `prepare()` and `self._prepared_model = self.diffusion_model`
        # because in that path the wrapped engine IS the DiT.
        if self.projector is not None:
            bundle = _Stage2Bundle(self.diffusion_model, self.projector)
            (
                bundle,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            ) = self.state.accelerator.prepare(
                bundle,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            )
            # accelerate wraps the bundle in DeepSpeedEngine (DeepSpeed) /
            # DistributedDataParallel (multi-GPU no-DeepSpeed) / no wrapper
            # (single-GPU no-DeepSpeed). Walk to the inner Module via
            # `.module` if a wrapper is present, otherwise the bundle is
            # already the bare nn.Module.
            inner = bundle.module if hasattr(bundle, "module") else bundle
            self.diffusion_model = inner.diffusion_model
            self.projector = inner.projector
            self._prepared_model = bundle
        else:
            (
                self.diffusion_model,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            ) = self.state.accelerator.prepare(
                self.diffusion_model,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            )
            self._prepared_model = self.diffusion_model

    # ------------------------------------------------------------------
    # Stage 2 helpers (called from train() in the Batch 3b commit)
    # ------------------------------------------------------------------

    def _save_projector_ckpt(self, save_path: str, global_step: int) -> None:
        """Save projector ckpt with provenance metadata (spec Section 6).

        Saved unconditionally whenever ``self.projector is not None`` (i.e.
        ``use_tactile_views=true``), MIRRORING GE's atomic-checkpoint
        philosophy: a deployable submodule is saved as a whole regardless
        of which params were actually updated this run. Concretely, in
        Stage 3 ``action_*`` phases the projector defaults to frozen and is
        re-saved here byte-identical to its warmstart source -- this is the
        correct behavior, since the action-phase ckpt directory must be
        self-contained for inference (an action policy with tactile views
        cannot be deployed without the projector).

        The saved ``phase`` field reflects ``self.args.phase`` so downstream
        consumers (``_load_projector_warmstart`` for ``world_model_only`` /
        Stage 3 phases) can validate that the warm-start source phase is
        legal. Currently legal source phases:

          - ``tactile_projector_only``: Stage 2 phase 1 (projector-only train).
          - ``world_model_only``: Stage 2 phase 2 (joint projector + DiT). The
            projector here is the most recent value AFTER joint refinement,
            which is the recommended warm-start for Stage 3 action phases.
          - ``action_only`` / ``action_full``: Stage 3 self-contained snapshot
            of the deployable projector. With the default frozen projector,
            this is identical to the source warmstart_ckpt; with the yaml
            unfreeze knob it captures the action-loss-tuned projector.

        Schema::

            {
                "projector":              projector.state_dict(),  # always saved
                "tactile_vae_model_path": str,                     # provenance
                "step":                   int,                     # snapshot step
                "phase":                  self.args.phase,         # source phase
                "source_projector_ckpt":  str | None,              # warmstart src
                "projector_frozen":       bool,                    # was it frozen
                # NOT saved: v0c-A weights (always loaded independently from
                #            tactile_vae.model_path; too large to bundle)
                # NOT saved: DiT weights (saved separately by GE save path
                #            via diffusion_model.save_pretrained)
                # NOT saved: optimizer state (each phase rebuilds a fresh
                #            optimizer with different param groups)
            }

        Caller is responsible for choosing ``save_path``; the trainer does
        not enforce a directory layout here.

        This helper is callable in any phase that has ``self.projector`` set
        (i.e. ``use_tactile_views=true``); raises ``RuntimeError`` otherwise.
        """
        accelerator = self.state.accelerator
        if self.projector is None:
            raise RuntimeError(
                "_save_projector_ckpt called but self.projector is None; "
                "this should never happen with use_tactile_views=true."
            )
        projector = unwrap_model(accelerator, self.projector)
        tactile_vae_args = self._stage2_dict_arg("tactile_vae")
        projector_args = self._stage2_dict_arg("projector")
        projector_frozen = not any(p.requires_grad for p in projector.parameters())
        ckpt = {
            "projector":              projector.state_dict(),
            "tactile_vae_model_path": tactile_vae_args.get("model_path", ""),
            "step":                   int(global_step),
            "phase":                  str(getattr(self.args, "phase", "")),
            "source_projector_ckpt":  projector_args.get("warmstart_ckpt", None),
            "projector_frozen":       bool(projector_frozen),
        }
        # Ensure parent directory exists; mkdir -p semantics.
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(ckpt, save_path)
        logger.info(
            f"saved projector ckpt to {save_path} "
            f"(phase={ckpt['phase']!r}, step={global_step}, frozen="
            f"{projector_frozen}, projector params="
            f"{sum(p.numel() for p in projector.parameters()):,})"
        )

    def _encode_tactile_split(
        self,
        tactile: torch.Tensor,
        mem_size: int,
        hand_pose: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Encode a tactile clip into per-hand fused latents matching the
        visual T_lat layout produced by ``utils.data_utils.get_latents``.

        Visual latent T axis is built from TWO different temporal modes:
          - mem path: each of ``mem_size`` frames encoded individually
                      (T_in=1 each) -> ``mem_size`` latent T-slots.
          - future path: full LTX VAE temporal compression on
                      ``chunk = T_total - mem_size`` frames -> ``chunk//8 + 1``
                      latent T-slots.
          - total: ``T_lat = mem_size + (chunk // 8 + 1)``.

        Naively calling ``encode_per_hand`` on the full T_total clip would
        produce ``T_total // 8 + 1`` latent slots, which does NOT match the
        visual T_lat. We mirror the mem/future split here so the resulting
        tactile latent layout is exactly co-aligned with visual along T,
        making the ``torch.cat([visual, tactile], dim=batch)`` step in
        ``train()`` valid without any reshape gymnastics.

        Projector application: the returned latent is ALREADY passed through
        ``self.projector`` (with ``view_idx = arange(V_hand)``) so callers
        get the DiT-ready tensor in one call.

        v0d pose-injection path: when the wrapper was constructed with
        ``adapter_use_pose_injection=True``, the caller must pass
        ``hand_pose`` of shape ``(B, V_hand, T_total, P)``. We slice it
        along the T axis exactly the same way as ``tactile`` (mem path
        gets the first ``mem_size`` frames flattened into batch as T=1
        each; future path gets the remaining ``chunk`` frames as one
        contiguous block). v0d's ``encode_per_hand`` then resamples each
        slice to its respective ``T_lat`` via ``_align_pose_to_lat``.

        Args:
            tactile: ``(B, V_hand, F=5, T_total, 192, 256)`` float32 in
                ``[-1, 1]`` -- output of the dataset's tactile field.
            mem_size: number of memory frames; must match the visual mem split.
            hand_pose: optional ``(B, V_hand, T_total, P)`` per-hand,
                per-frame pose trajectory at the dataset's temporal
                resolution. Required when the wrapper was built with
                ``adapter_use_pose_injection=True``; silently ignored
                otherwise. The first ``mem_size`` slots are passed
                per-frame (T_raw=1 each) to the mem encode; the remaining
                ``T_total - mem_size`` slots are passed as one block
                (T_raw=chunk) to the future encode.
            return_intermediates: if True, also return the pre-projector
                adapter output and the per-finger LTX VAE latent (before
                adapter fusion). Used by the val monitor to track
                distribution shift across the encoding chain.

        Returns:
            When ``return_intermediates=False`` (default):
                ``(B, V_hand, 128, T_lat, 6, 8)`` per-hand fused latent
                tensor, with ``T_lat = mem_size + (chunk // 8 + 1)``.
            When ``return_intermediates=True``:
                Tuple of ``(tac_latent, tac_latent_pre, tac_per_finger)``
                where ``tac_latent_pre`` is the adapter output before
                projector, and ``tac_per_finger`` is the concatenated
                per-finger LTX latent ``(B, V_hand, F, 128, T_lat, 6, 8)``.
        """
        if self.tactile_vae is None or self.projector is None:
            raise RuntimeError(
                "_encode_tactile_split called but tactile_vae/projector are "
                "None. This indicates use_tactile_views=false; the trainer "
                "should not be invoking the tactile path."
            )
        if tactile.ndim != 6:
            raise ValueError(
                f"_encode_tactile_split expects 6-D tactile (B, V_hand, F, T, "
                f"H, W); got shape {tuple(tactile.shape)}."
            )

        B, V_hand, F_finger, T_total, H, W = tactile.shape
        if T_total <= mem_size:
            raise ValueError(
                f"T_total ({T_total}) must be > mem_size ({mem_size}); "
                f"there must be at least 1 future frame to denoise."
            )

        # ---- v0d hand_pose validation + shape gating ----
        use_pose_injection = bool(getattr(self.tactile_vae, "use_pose_injection", False))
        if use_pose_injection:
            if hand_pose is None:
                raise ValueError(
                    "_encode_tactile_split: hand_pose is required when the "
                    "tactile adapter has use_pose_injection=True; got None. "
                    "Either set `read_hand_pose: true` in the dataset yaml "
                    "(adds `hand_pose` to each batch) or disable "
                    "`adapter_use_pose_injection` in `tactile_vae.config`."
                )
            if hand_pose.ndim != 4:
                raise ValueError(
                    f"_encode_tactile_split: hand_pose must be 4-D "
                    f"(B, V_hand, T_raw, P); got shape "
                    f"{tuple(hand_pose.shape)}."
                )
            expected_P = int(getattr(self.tactile_vae, "pose_dim", 22))
            if (
                hand_pose.shape[0] != B
                or hand_pose.shape[1] != V_hand
                or hand_pose.shape[2] != T_total
                or hand_pose.shape[3] != expected_P
            ):
                raise ValueError(
                    f"_encode_tactile_split: hand_pose shape mismatch -- "
                    f"expected (B={B}, V_hand={V_hand}, T_total={T_total}, "
                    f"P={expected_P}); got {tuple(hand_pose.shape)}."
                )
        elif hand_pose is not None:
            # Adapter has no pose head; silently drop to avoid surprising
            # the v0c-A code path. (Logged once in the Gate-1 first-encode
            # block below so the user is aware.)
            hand_pose = None

        # ----- v0d Gate-1 ENCODE log (one-shot) ----------------------
        # Emitted on the FIRST `_encode_tactile_split` call after a fresh
        # trainer init. Proves the dataset/adapter shape contracts are in
        # sync BEFORE any noise / loss math runs. Gated by a flag so the
        # log doesn't spam every step.
        if not getattr(self, "_v0d_gate1_encode_logged", False):
            mode = "v0d (pose-injection)" if use_pose_injection else "v0c-A"
            hp_shape = tuple(hand_pose.shape) if hand_pose is not None else None
            hp_dtype = hand_pose.dtype if hand_pose is not None else None
            logger.info(
                "[v0d-gate1-encode] first _encode_tactile_split call:\n"
                f"  mode                  = {mode}\n"
                f"  tactile.shape         = {tuple(tactile.shape)} "
                f"(B={B}, V_hand={V_hand}, F={F_finger}, T_total={T_total}, "
                f"H={H}, W={W})\n"
                f"  mem_size              = {mem_size}  (chunk={T_total - mem_size})\n"
                f"  hand_pose.shape       = {hp_shape}\n"
                f"  hand_pose.dtype       = {hp_dtype}\n"
                f"  use_pose_injection    = {use_pose_injection}"
            )
            self._v0d_gate1_encode_logged = True

        # ----- adapter encode (frozen, no_grad) -----------------------
        # The tactile_vae adapter is fully frozen in Stage 2/3 (loaded by
        # `_load_v0c_a_frozen` with `requires_grad=False` on every param and
        # `eval()`). Wrapping the encode calls in `torch.no_grad()` ensures
        # we don't retain the (potentially deep) per-frame VAE intermediate
        # activations in the autograd graph -- saves significant VRAM. The
        # projector application below is OUTSIDE this block so projector
        # grads flow normally during backward.
        with torch.no_grad():
            # Memory path: per-frame encoding (T_in=1 each).
            # Reshape so each mem frame becomes its own batch element with
            # T=1. encode_per_hand expects (B, V_hand, F, T, H, W); we
            # produce (B*mem_size, V_hand, F, 1, H, W).
            mem_tac = tactile[:, :, :, :mem_size]                          # (B, V_hand, F, mem_size, H, W)
            mem_tac = rearrange(mem_tac, "b vh f m h w -> (b m) vh f h w").unsqueeze(3)
            # -> (B*mem_size, V_hand, F, 1, H, W)
            # v0d: slice hand_pose along the same mem range and rearrange
            # so each mem frame becomes its own "batch with T_raw=1" entry,
            # matching the (B*mem_size, V_hand, F, 1, H, W) tactile layout.
            mem_pose = None
            if hand_pose is not None:
                mem_pose = hand_pose[:, :, :mem_size, :]                   # (B, V_hand, mem_size, P)
                mem_pose = rearrange(
                    mem_pose, "b vh m p -> (b m) vh p"
                ).unsqueeze(2)                                              # (B*mem_size, V_hand, 1, P)

            if return_intermediates:
                mem_lat, mem_pf = self.tactile_vae.encode_per_hand(
                    mem_tac, hand_pose=mem_pose, return_per_finger=True,
                )                                                           # mem_lat: (B*mem_size, V_hand, C, 1, h, w)
                                                                            # mem_pf:  (B*mem_size, V_hand, F, C, 1, h, w)
            else:
                mem_lat = self.tactile_vae.encode_per_hand(
                    mem_tac, hand_pose=mem_pose,
                )                                                           # (B*mem_size, V_hand, C, 1, h, w)
                mem_pf = None

            mem_lat = rearrange(
                mem_lat, "(b m) vh c t h w -> b vh c (m t) h w", b=B, m=mem_size
            )
            # -> (B, V_hand, 128, mem_size, 6, 8)

            # Future path: full LTX temporal compression.
            future_tac = tactile[:, :, :, mem_size:]                       # (B, V_hand, F, chunk, H, W)
            future_pose = None
            if hand_pose is not None:
                future_pose = hand_pose[:, :, mem_size:, :]                # (B, V_hand, chunk, P)

            if return_intermediates:
                future_lat, future_pf = self.tactile_vae.encode_per_hand(
                    future_tac, hand_pose=future_pose, return_per_finger=True,
                )                                                           # future_lat: (B, V_hand, C, chunk//8+1, h, w)
                                                                            # future_pf:  (B, V_hand, F, C, chunk//8+1, h, w)
            else:
                future_lat = self.tactile_vae.encode_per_hand(
                    future_tac, hand_pose=future_pose,
                )                                                           # (B, V_hand, C, chunk//8+1, h, w)
                future_pf = None

            # Concat along T axis -> matches visual T_lat.
            tac_latent_pre = torch.cat([mem_lat, future_lat], dim=3)       # (B, V_hand, 128, T_lat, 6, 8)

            # Per-finger concat along T axis (same split as per-hand).
            if return_intermediates:
                mem_pf = rearrange(
                    mem_pf, "(b m) vh f c t h w -> b vh f c (m t) h w",
                    b=B, m=mem_size,
                )
                tac_per_finger = torch.cat(
                    [mem_pf, future_pf], dim=4,
                )                                                           # (B, V_hand, F, 128, T_lat, 6, 8)
            else:
                tac_per_finger = None

        # We detach() defensively here as well: if the backbone ever held
        # any module-level state with `requires_grad=True` (e.g. user picks
        # a different tactile_vae config that forgets to freeze something),
        # the projector input is still cleanly grad-isolated from v0c-A.
        tac_latent_pre = tac_latent_pre.detach()

        # ----- Project (add modality + view markers + alpha-gated MLP) -
        # The projector is OUTSIDE the no_grad block above so projector
        # parameters can receive gradients via the downstream loss.
        #
        # E_proj_identity bypass (cube tactile slowdown ablation, plan
        # cube_tactile_slowdown_ablation_1b64937f, 2026-05-24 late):
        # when `tactile_vae.config.disable_projector = true`, skip the
        # projector call entirely and pass `tac_latent_pre` through
        # with the same shape. The `return_intermediates=True` tuple
        # then has `tac_latent IS tac_latent_pre` (same object), which
        # is exactly the signal the option_c diag block needs to
        # confirm bypass at val time (out.batch_std == in.batch_std,
        # out.total_std == in.total_std). The view_idx computation is
        # also skipped on the bypass path because the projector is the
        # only consumer; this avoids allocating a (B, V_hand) tensor
        # for no reason. See plan section "Implementation contract
        # (E_proj_identity)" for the falsification flow.
        if self._disable_projector_for_action:
            tac_latent = tac_latent_pre
        else:
            view_idx = torch.arange(
                V_hand, device=tac_latent_pre.device, dtype=torch.long
            ).unsqueeze(0).expand(B, V_hand).contiguous()                  # (B, V_hand)
            tac_latent = self.projector(tac_latent_pre, view_idx)

        if return_intermediates:
            return tac_latent, tac_latent_pre, tac_per_finger
        return tac_latent


    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)


    def _forward_loss_batch(
        self,
        batch: Dict[str, Any],
        *,
        training: bool,
    ) -> Dict[str, Any]:
        """Single forward + loss pass for one batch.

        This is the SHARED helper invoked by both :meth:`train` (with
        ``training=True``) and :meth:`_compute_val_loss` (with
        ``training=False``). Sharing the codepath at the FUNCTION level
        guarantees train and val losses are computed by an identical
        forward + loss recipe -- there is no way for one branch to drift
        from the other under future edits (a single change applies to
        both). This is the Stage 2 ``Gate 2`` requirement.

        Behavior differences between train and val are limited to two
        knobs and are gated explicitly inside this function:

          * ``apply_color_jitter_to_video``: applied only when
            ``training=True`` AND ``self.args.use_color_jitter`` is set.
          * ``caption dropout``: only when ``training=True``; val always
            uses the real caption (no unconditional dummies).

        Everything else -- VAE encode, tactile injection (incl. v0d
        ``hand_pose``), text conditioning, timestep / noise sampling,
        action conditioning, the forward pass, and the loss math
        (visual / tactile co-denoise + action) -- runs the same way for
        both train and val.

        Autograd / accumulate / backward / optimizer step are explicitly
        the CALLER's responsibility. The helper does NOT call
        ``accelerator.accumulate(...)``, does NOT call ``.backward(...)``,
        and does NOT call ``optimizer.step()``. Train wraps the call in
        ``accelerator.accumulate(...)`` and runs backward + step
        afterwards; val wraps the call in ``torch.no_grad()``.

        Args:
            batch: one batch dict from a DexVTAMDataset (or compatible)
                dataloader. Required keys: ``video``, ``caption``. Optional
                keys depending on yaml config: ``actions``, ``state``,
                ``tactile``, ``hand_pose``.
            training: True when called from the train loop, False when
                called from :meth:`_compute_val_loss`.

        Returns:
            A fresh dict (never mutated across calls) with the following
            entries -- 0-d torch tensors unless noted:

              * ``loss``: total scalar loss for backward (when training=True)
                or aggregation (when training=False).
              * ``loss_video``: per-modality video co-denoise loss
                (zero-tensor if ``train_mode`` is action-only).
              * ``loss_visual`` / ``loss_tactile``: Stage 2 modality split
                of ``loss_video``. When ``use_tactile_views=false`` or
                Stage 3, ``loss_visual`` mirrors ``loss_video`` and
                ``loss_tactile`` is a zero-tensor (so the caller can
                always reduce both unconditionally).
              * ``loss_action``: action-MSE loss (zero-tensor if
                ``train_mode`` is video-only).
              * ``n_view`` / ``n_view_visual`` / ``n_view_tactile``: view
                counts. Visual rows come FIRST in the batch axis,
                tactile rows come LAST -- this ordering is the contract
                for the Stage 2 loss split.
              * ``batch_size`` / ``mem_size``: python ints, the source
                batch's leading dim and memory-frame count respectively.
        """
        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype

        # ---- Gate-2 logging (one-shot per training=True / training=False) ----
        # Prints the SAME function id from both branches so a downstream
        # log audit can prove train and val share the exact codepath.
        _log_attr = (
            "_v0d_gate2_train_logged" if training else "_v0d_gate2_val_logged"
        )
        if not getattr(self, _log_attr, False):
            _fn = self._forward_loss_batch
            _fn_id = id(_fn.__func__ if hasattr(_fn, "__func__") else _fn)
            _use_tac = bool(getattr(self.args, "use_tactile_views", False))
            _use_pi = bool(
                self.tactile_vae is not None
                and getattr(self.tactile_vae, "use_pose_injection", False)
            )
            logger.info(
                "[v0d-gate2] first _forward_loss_batch call:\n"
                f"  training              = {training}\n"
                f"  function id           = {_fn_id}\n"
                f"  batch keys            = {sorted(batch.keys())}\n"
                f"  use_tactile_views     = {_use_tac}\n"
                f"  use_pose_injection    = {_use_pi}\n"
                f"  train_mode            = {self.args.train_mode!r}\n"
                f"  phase                 = {getattr(self.args, 'phase', None)!r}"
            )
            setattr(self, _log_attr, True)

        use_tactile_views = bool(getattr(self.args, "use_tactile_views", False))
        phase = getattr(self.args, "phase", None)
        is_stage2 = phase in _STAGE2_PHASES

        if is_stage2 and use_tactile_views:
            loss_cfg = self._stage2_dict_arg("loss")
            # Fail-loud: yaml `lambda_visual: null` -> float(None) -> TypeError;
            # bad string -> ValueError. Both surface immediately rather than
            # silently zeroing a modality.
            lambda_visual = float(loss_cfg.get("lambda_visual", 1.0))
            lambda_tactile = float(loss_cfg.get("lambda_tactile", 1.0))
        else:
            lambda_visual = 1.0
            lambda_tactile = 0.0

        # ---- Cache scheduler_sigmas on self (constant per scheduler) ----
        if not hasattr(self, "_cached_scheduler_sigmas"):
            self._cached_scheduler_sigmas = self.scheduler.sigmas.clone().to(
                device=accelerator.device, dtype=weight_dtype,
            )
        scheduler_sigmas = self._cached_scheduler_sigmas

        # ============ Visual VAE encode + mem/future split ============
        video = batch['video']
        video = video.to(accelerator.device, dtype=weight_dtype).contiguous()
        batch_size, c, n_view, _, h, w = video.shape
        video = rearrange(video, 'b c v t h w -> (b v) c t h w')

        # Color jitter ONLY during training. Val needs deterministic input.
        if training and self.args.use_color_jitter:
            video = apply_color_jitter_to_video(video)

        mem_size = self.args.data['train']['n_previous']
        mem = video[:, :, :mem_size]
        future_video = video[:, :, mem_size:]

        if self.args.return_action:
            future_video = future_video[:, :, :1].repeat(
                1, 1, self.args.data['train']['chunk'], 1, 1,
            )

        _, _, raw_frames, raw_height, raw_width = future_video.shape

        latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
        latent_height = raw_height // self.SPATIAL_DOWN_RATIO
        latent_width = raw_width // self.SPATIAL_DOWN_RATIO

        # Caption dropout ONLY during training. Val keeps every caption.
        if training:
            dropout_factor = torch.rand(batch_size).to(
                accelerator.device, dtype=weight_dtype,
            )
            dropout_mask_prompt = dropout_factor < self.args.caption_dropout_p
        else:
            dropout_mask_prompt = torch.zeros(
                batch_size, dtype=torch.bool, device=accelerator.device,
            )
        dropout_mask_prompt = dropout_mask_prompt.unsqueeze(1).unsqueeze(2)

        mem_latents, future_video_latents = get_latents(self.vae, mem, future_video)
        mem_latents = rearrange(
            mem_latents, '(b v m) (h w) c -> (b v) c m h w',
            b=batch_size, m=mem_size, h=latent_height,
        )
        future_video_latents = rearrange(
            future_video_latents, '(b v) (f h w) c -> (b v) c f h w',
            b=batch_size, h=latent_height, w=latent_width,
        )
        latents = torch.cat((mem_latents, future_video_latents), dim=2)

        # ============ Tactile injection (Batch 3b) ============
        n_view_visual = n_view
        n_view_tactile = 0
        if use_tactile_views:
            if 'tactile' not in batch:
                raise KeyError(
                    "use_tactile_views=true but batch has no 'tactile' "
                    "field. Wire the dataset's tactile column or set "
                    "use_tactile_views=false."
                )
            tactile = batch['tactile'].to(
                accelerator.device, dtype=weight_dtype,
            ).contiguous()
            if tactile.ndim != 6:
                raise ValueError(
                    f"batch['tactile'] expected (B, V_hand, F=5, T, H, W); "
                    f"got {tuple(tactile.shape)}."
                )
            n_view_tactile = tactile.shape[1]
            # v0d hand_pose pickup (gated on the loaded adapter's flag,
            # not on the yaml -- the load path already enforced they
            # match via Gate-1's abort-on-mismatch).
            hand_pose_t = None
            v0d_pose_on = bool(
                self.tactile_vae is not None
                and getattr(self.tactile_vae, "use_pose_injection", False)
            )
            if v0d_pose_on:
                if 'hand_pose' not in batch:
                    raise KeyError(
                        "tactile adapter has use_pose_injection=True but "
                        "batch has no 'hand_pose' field. Set "
                        "`read_hand_pose: true` and a valid "
                        "`pose_stats_path` in the dataset config, or "
                        "disable `adapter_use_pose_injection` in the "
                        "tactile_vae config."
                    )
                hand_pose_t = batch['hand_pose'].to(
                    accelerator.device, dtype=weight_dtype,
                ).contiguous()
                if hand_pose_t.ndim != 4:
                    raise ValueError(
                        f"batch['hand_pose'] expected (B, V_hand=2, "
                        f"T_total, 22); got {tuple(hand_pose_t.shape)}."
                    )
            # Mirror the visual return_action future-frame trick so
            # `_encode_tactile_split` produces a T_lat matching visual.
            if self.args.return_action:
                fut_chunk = self.args.data['train']['chunk']
                tac_mem_raw = tactile[:, :, :, :mem_size]
                tac_fut_raw = tactile[:, :, :, mem_size:mem_size + 1].repeat(
                    1, 1, 1, fut_chunk, 1, 1,
                )
                tactile = torch.cat([tac_mem_raw, tac_fut_raw], dim=3).contiguous()
                if hand_pose_t is not None:
                    hp_mem = hand_pose_t[:, :, :mem_size, :]
                    hp_fut = hand_pose_t[:, :, mem_size:mem_size + 1, :].repeat(
                        1, 1, fut_chunk, 1,
                    )
                    hand_pose_t = torch.cat([hp_mem, hp_fut], dim=2).contiguous()
            # D0 (cube tactile slowdown ablation, plan
            # cube_tactile_slowdown_ablation_1b64937f): demote the
            # `is_stage2` gate. The downstream distribution-chain monitor
            # below also gated on `is_stage2`, which silenced these stats
            # during Stage 3 (`train_mode in {action_only, action_full}`)
            # val even though Stage 3 still ships tactile views into the
            # DiT as context. We need vis_std / proj_in_std /
            # proj_out_std / per_finger_std in Stage 3 val to test the
            # "modality collapse by magnitude" hypothesis (Z/X, Y_pre/X,
            # Y_post/Y_pre ratios). Train path remains untouched: gated
            # on `not training`, val-only, and Stage 3's val cadence is
            # already throttled by `steps_to_val`.
            _want_intermediates = (
                not training and use_tactile_views
            )
            if _want_intermediates:
                tac_full, tac_full_pre, tac_per_finger = (
                    self._encode_tactile_split(
                        tactile, mem_size, hand_pose=hand_pose_t,
                        return_intermediates=True,
                    )
                )
            else:
                tac_full = self._encode_tactile_split(
                    tactile, mem_size, hand_pose=hand_pose_t,
                )
                tac_full_pre = None
                tac_per_finger = None
            # ----- Probe Q layout fix: interleave-before-flatten ---------
            # The original layout (`tac_full = rearrange("b v c f h w
            # -> (b v) c f h w"); latents = torch.cat([latents, tac_full],
            # dim=0); n_view = V_full`) produces row order
            # `[vis(b=0,v=*), vis(b=1,v=*), ..., tac(b=0,vh=*), ...]`.
            # The model's `(b v) l c -> b (v l) c, v=V_full` rearrange in
            # cross_view_attn ([transformer_ltx_multiview.py:104]) and in
            # `_split_action_kv` (line 540-541) assume each contiguous
            # V_full-row block belongs to ONE true_batch. The original
            # layout violates that for B>1: V_full-row blocks straddle
            # true_batches -> cross_view attention mixes tokens across
            # true_batches AND `_split_action_kv` mislabels visual/tactile
            # rows for at least one model_batch per step. Verified with
            # `/tmp/check_layout.py` (see plan probe-q-layout-fix_419db6e5).
            #
            # Fix: un-flatten `latents` / `mem_latents` back to 6D, cat
            # tactile on the view axis (per-batch interleave), re-flatten.
            # Output `latents` shape is identical (`(B*V_full, C, F, H, W)`),
            # only the row ORDER changes to canonical
            # `[vis(b=0), tac(b=0), vis(b=1), tac(b=1), ...]`. All downstream
            # sites that assumed the original `[all_vis, all_tac]` layout
            # (tac-chain val monitor at line 2289-2291, per-modality loss
            # split at line 2717-2719) are updated to use a view-axis
            # reshape in the same commit. B=1 is unaffected either way;
            # the fix is necessary AND sufficient for B>=1.
            vis_5d = rearrange(
                latents, "(b v) c f h w -> b v c f h w",
                b=batch_size, v=n_view_visual,
            )
            mem_vis_5d = rearrange(
                mem_latents, "(b v) c m h w -> b v c m h w",
                b=batch_size, v=n_view_visual,
            )
            tac_mem_5d = tac_full[:, :, :, :mem_size].contiguous()
            combined_5d = torch.cat([vis_5d, tac_full], dim=1)
            combined_mem_5d = torch.cat([mem_vis_5d, tac_mem_5d], dim=1)
            latents = rearrange(
                combined_5d, "b v c f h w -> (b v) c f h w",
            )
            mem_latents = rearrange(
                combined_mem_5d, "b v c m h w -> (b v) c m h w",
            )
            n_view = n_view_visual + n_view_tactile

            # One-time-per-process layout banner + runtime assertion.
            # The assertion builds a tagged synthetic tensor with the same
            # B/V_vis/V_hand and pushes it through the same interleave +
            # `(b v) -> b v` rearrange chain, then verifies each model_batch
            # slot contains rows from exactly one true_batch. Cost: 2 small
            # CPU tensor ops, once per process. Fails loud if a future
            # refactor regresses the layout contract -- see plan
            # probe-q-layout-fix_419db6e5 and /tmp/check_layout.py for the
            # offline equivalent.
            if not getattr(self, "_q_layout_banner_emitted", False):
                with torch.no_grad():
                    _v_vis = max(int(n_view_visual), 1)
                    _v_tac = max(int(n_view_tactile), 1)
                    _v_full = _v_vis + _v_tac
                    # tag = 100*true_b + (view_idx; tactile views offset +50)
                    _vis_tag = torch.tensor(
                        [[100 * b + v for v in range(_v_vis)] for b in range(2)]
                    )
                    _tac_tag = torch.tensor(
                        [[100 * b + 50 + v for v in range(_v_tac)] for b in range(2)]
                    )
                    _combined = torch.cat([_vis_tag, _tac_tag], dim=1)
                    _flat = rearrange(_combined, "b v -> (b v)")
                    _grouped = rearrange(_flat, "(b v) -> b v", v=_v_full)
                    for _b in range(2):
                        _seen = {int(x) // 100 for x in _grouped[_b].tolist()}
                        if _seen != {_b}:
                            raise RuntimeError(
                                f"[Q-layout-fix] Layout assertion FAILED on "
                                f"model_batch {_b}: true_batches present = "
                                f"{sorted(_seen)}, expected {{{_b}}}. The "
                                f"tactile interleave above must have regressed."
                            )
                logger.info(
                    "[Q-layout-fix] tactile injection interleaved per-batch: "
                    f"latents row layout = [vis(b=0,v=0..{n_view_visual - 1}), "
                    f"tac(b=0,vh=0..{n_view_tactile - 1}), vis(b=1,...), "
                    f"tac(b=1,...), ...]; B={batch_size}, V_full={n_view}. "
                    f"After the model's `(b v) l c -> b (v l) c, v={n_view}` "
                    f"rearrange each model_batch slot corresponds to exactly "
                    f"one true_batch (asserted on synthetic tagged tensor). "
                    f"See plan probe-q-layout-fix_419db6e5."
                )
                self._q_layout_banner_emitted = True
        # ----- end tactile injection -----

        video_attention_mask = None
        latents = rearrange(latents, 'bv c f h w -> bv (f h w) c')

        # ----- Distribution chain monitor (val-only) --------------------
        # Tracks mean + std at every stage of the tactile encoding chain:
        #   raw input -> per-finger LTX latent -> adapter out (proj_in)
        #   -> projector out -> visual latent (reference)
        # plus the projector's alpha gate value.
        #
        # "batch_std" = std computed along the batch axis then averaged
        # over all latent elements. Measures whether different samples
        # produce different representations (collapse = batch_std -> 0).
        #
        # "total_std" = std of the entire tensor. Measures the overall
        # spread of the latent distribution.
        #
        # Gated on `not training` to keep train-loop perf untouched.
        # Aggregation across val batches / DDP ranks is done by
        # `_compute_val_loss` using the same mean/reduce path as loss.
        #
        # D0 (cube tactile slowdown ablation, plan
        # cube_tactile_slowdown_ablation_1b64937f): the `is_stage2` gate
        # was demoted here so this block now fires in Stage 3 val too
        # (`action_only` / `action_full` with `use_tactile_views=true`).
        # Stage 3 has no co-denoise loss but still concatenates tactile
        # views into `latents` -- the distribution-chain stats are what
        # we need to test the magnitude-mismatch hypothesis (D1a:
        # measure Y_pre/X, Y_post/Y_pre, Z/X). Train path remains
        # unchanged: gated on `not training`, and `n_view_tactile > 0`
        # already keeps pure-visual configs (E1 / GE-on-cube style)
        # silent regardless of phase.
        #
        # Option C extension (2026-05-24, same plan): D1a showed Z/X
        # only in the MODERATE band (~0.5) while the tactile
        # batch_std/total_std collapsed to ~0.05 (visual ~0.61). This is
        # a content / sample-specific variation collapse, more critical
        # than the original magnitude hypothesis. We now also compute
        # `per_finger_batch_std`, `proj_in_batch_std`, and
        # `input_batch_std` to localize where in the encoding chain the
        # collapse first appears (raw input -> per-finger LTX latent ->
        # post-adapter -> post-projector). Same val-only gate, same
        # (b, V_hand)->row reduction convention as the existing
        # `proj_out_batch_std` so the numbers are directly comparable
        # across stages.
        input_tac_mean = 0.0
        input_tac_std = 0.0
        input_batch_std = 0.0
        per_finger_mean = 0.0
        per_finger_std = 0.0
        per_finger_batch_std = 0.0
        proj_in_mean = 0.0
        proj_in_std = 0.0
        proj_in_batch_std = 0.0
        proj_out_mean = 0.0
        proj_out_total_std = 0.0
        proj_out_batch_std = 0.0
        vis_mean = 0.0
        vis_total_std = 0.0
        vis_batch_std = 0.0
        proj_alpha = 0.0
        if (
            not training
            and use_tactile_views
            and n_view_tactile > 0
        ):
            with torch.no_grad():
                # Probe Q layout fix (2026-05): after interleave-before-
                # flatten, `latents` row order is
                # `[vis(b=0,v=*), tac(b=0,vh=*), vis(b=1,v=*), tac(b=1,vh=*), ...]`
                # so the original `[:B*V_rgb]` / `[B*V_rgb:]` slice no
                # longer extracts all-visual / all-tactile. Use a view-
                # axis reshape: split the BV axis into (B, V_full),
                # slice along the view axis, then flatten back.
                latents_bv = rearrange(
                    latents, "(b v) l c -> b v l c",
                    b=batch_size, v=n_view,
                )
                vis_rows = rearrange(
                    latents_bv[:, :n_view_visual],
                    "b v l c -> (b v) l c",
                ).float()
                tac_rows = rearrange(
                    latents_bv[:, n_view_visual:],
                    "b v l c -> (b v) l c",
                ).float()

                # -- Batch-wise std (collapse signal) --
                if vis_rows.shape[0] >= 2:
                    vis_batch_std = float(
                        vis_rows.std(dim=0).mean().item()
                    )
                if tac_rows.shape[0] >= 2:
                    proj_out_batch_std = float(
                        tac_rows.std(dim=0).mean().item()
                    )

                # -- Total mean/std (distribution shift signal) --
                vis_mean = float(vis_rows.mean().item())
                vis_total_std = float(vis_rows.std().item())
                proj_out_mean = float(tac_rows.mean().item())
                proj_out_total_std = float(tac_rows.std().item())

                # -- Adapter output / projector input --
                if tac_full_pre is not None:
                    pre = tac_full_pre.float()
                    proj_in_mean = float(pre.mean().item())
                    proj_in_std = float(pre.std().item())
                    # Option C extension (cube tactile slowdown ablation
                    # plan, 2026-05-24): batch_std at the post-adapter
                    # stage. Localizes batch-level collapse to before vs.
                    # after the 5->1 FingerSetTransformerAdapter fusion.
                    # Convention: treat each (b, V_hand) as one row to
                    # match `proj_out_batch_std`'s reduction.
                    pre_rows = rearrange(
                        pre, "b vh c t h w -> (b vh) c t h w"
                    )
                    if pre_rows.shape[0] >= 2:
                        proj_in_batch_std = float(
                            pre_rows.std(dim=0).mean().item()
                        )

                # -- Per-finger LTX latent (before adapter fusion) --
                if tac_per_finger is not None:
                    pf = tac_per_finger.float()
                    per_finger_mean = float(pf.mean().item())
                    per_finger_std = float(pf.std().item())
                    # Option C extension: batch_std at the per-finger LTX
                    # latent stage. If THIS is already collapsed, the
                    # frozen LTX VAE / GrayToRGB encoding is the main
                    # cause (LTX OOD on grayscale tactile). If finger is
                    # normal but proj_in (post-fusion) is low, the
                    # FingerSetTransformerAdapter 5->1 fusion is the
                    # cause. Same (b, V_hand)->row convention.
                    pf_rows = rearrange(
                        pf, "b vh f c t h w -> (b vh) f c t h w"
                    )
                    if pf_rows.shape[0] >= 2:
                        per_finger_batch_std = float(
                            pf_rows.std(dim=0).mean().item()
                        )

                # -- Raw tactile input (as seen by trainer, post-normalize) --
                if 'tactile' in batch:
                    raw = batch['tactile'].float()
                    input_tac_mean = float(raw.mean().item())
                    input_tac_std = float(raw.std().item())
                    # Option C extension: batch_std at the raw tactile
                    # input stage. A floor: if the dataset itself yields
                    # near-identical per-sample tactile signals, the
                    # downstream collapse is a data-side artefact (e.g.
                    # normalization stats, masking) rather than an
                    # encoder bug. Same (b, V_hand)->row convention.
                    raw_rows = rearrange(
                        raw, "b vh f t h w -> (b vh) f t h w"
                    )
                    if raw_rows.shape[0] >= 2:
                        input_batch_std = float(
                            raw_rows.std(dim=0).mean().item()
                        )

                # -- Projector alpha gate --
                proj_alpha = float(self.projector.alpha_value())

        captions = batch['caption']
        text_conds = get_text_conditions(self.tokenizer, self.text_encoder, captions)
        prompt_embeds = text_conds['prompt_embeds']
        prompt_attention_mask = text_conds['prompt_attention_mask']
        prompt_embeds = (
            self.uncond_prompt_embeds.repeat(batch_size, 1, 1) * dropout_mask_prompt
            + prompt_embeds * ~dropout_mask_prompt
        )

        # ============ Timestep sampling ============
        action_weights = compute_density_for_timestep_sampling(
            weighting_scheme=self.args.flow_weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.args.flow_logit_mean,
            logit_std=self.args.flow_logit_std,
            mode_scale=self.args.flow_mode_scale,
        )
        # 0-1, 0 -> most noisy, 1 -> almost clean
        action_indices = (
            action_weights * self.scheduler.config.num_train_timesteps
        ).long()
        action_sigmas = scheduler_sigmas[action_indices]
        action_timesteps = (action_sigmas * 1000.0).long()

        if self.args.return_action and self.args.noisy_video:
            weights = torch.full_like(action_weights, 0.0).unsqueeze(1).repeat(1, n_view)
        else:
            weights = action_weights.unsqueeze(1).repeat(1, n_view)

        weights = rearrange(weights, 'b v -> (b v)')
        indices = (weights * self.scheduler.config.num_train_timesteps).long()
        sigmas = scheduler_sigmas[indices]
        timesteps = (sigmas * 1000.0).long()

        # ============ Action conditioning ============
        if self.args.return_action:
            if getattr(self.args, "add_state", False):
                act_state = batch['state']
                if act_state.shape[1] != 1:
                    act_state = act_state[:, mem_size - 1:mem_size]
                act_state = act_state.to(
                    accelerator.device, dtype=weight_dtype,
                ).contiguous()
            else:
                act_state = None

            actions = batch['actions'][
                :, -self.args.data['train']['action_chunk']:
            ].to(accelerator.device, dtype=weight_dtype).contiguous()

            noise_actions = randn_tensor(
                actions.shape, device=accelerator.device, dtype=weight_dtype,
            )

            action_timesteps = action_timesteps.unsqueeze(-1).repeat(
                1, actions.shape[1],
            )
            action_ss = action_sigmas.reshape(-1, 1, 1).repeat(
                1, 1, actions.shape[-1],
            )

            noisy_actions = (1.0 - action_ss) * actions + action_ss * noise_actions

            action_weights = compute_loss_weighting_for_sd3(
                weighting_scheme=self.args.flow_weighting_scheme,
                sigmas=action_sigmas,
            ).reshape(-1, 1, 1).repeat(1, 1, actions.size(-1))
        else:
            actions = None
            action_timesteps = None
            noisy_actions = None
            act_state = None

        # ============ Noise from condition frame ============
        noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
            mem_latents, latent_frames, latent_height, latent_width,
            noise_to_condition_frames=self.args.noise_to_first_frame,
        )
        if self.args.pixel_wise_timestep:
            timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
        else:
            timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

        ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
        if self.args.return_action and self.args.noisy_video:
            ss = torch.full_like(ss, 1.0)

        noisy_latents = (1.0 - ss) * latents + ss * noise

        weights = compute_loss_weighting_for_sd3(
            weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas,
        ).reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))

        # ============ D1b K/V + QK content hook (val-only) ============
        # Option C part B (cube tactile slowdown ablation, plan
        # cube_tactile_slowdown_ablation_1b64937f, 2026-05-24).
        # Captures Q / K / V at ONE mid-stack action cross-attn block
        # (action_blocks[_d1b_block_idx].attn2) during the upcoming
        # forward_pass via forward-hooks on the to_q/to_k/to_v Linear
        # submodules. After forward_pass returns we slice K/V into
        # visual rows vs tactile rows along the (V*L) axis (visual
        # comes FIRST by construction, see the tactile-injection block
        # near line 2114) and compute:
        #   - ||K_vis||, ||K_tac||, ||V_vis||, ||V_tac|| (mean L2 row norm)
        #   - std(QK_vis), std(QK_tac), mean|QK|, p95|QK| (proxy logits)
        #   - cos(Q.mean, K_vis.mean) vs cos(Q.mean, K_tac.mean)
        #   - d1b_k_tac_batch_std / d1b_k_vis_batch_std: std of
        #     K.mean_over_rows along the BATCH axis -- the
        #     content-collapse-specific metric that distinguishes
        #     "low magnitude" from "low sample-specific information".
        #
        # Gated identically to the D1a diag block above. NEVER fires
        # in training. Hooks are removed in the `finally` so a forward
        # exception cannot leak them into later batches. Hooks are
        # cheap (no copy unless gate is on) and add ~30 MB peak val-only.
        #
        # Block-idx 14 was chosen as a mid-stack representative for
        # num_layers=28 (cube_v4 / diverse_488 config); the
        # `_d1b_safe_idx` clamp keeps the code valid if a future yaml
        # uses a shallower stack.
        _d1b_block_idx_request = 14
        _d1b_capture: Dict[str, Optional[torch.Tensor]] = {
            "q": None, "k": None, "v": None,
        }
        _d1b_handles: list = []
        # D1b hook captures a single `.attn2` per action block; the dual
        # cross-attn (Option A) path replaces that with `.attn2_visual` +
        # `.attn2_tactile`, so the K/V split convention used below
        # (visual-first concat) does not apply. Skip cleanly in dual mode
        # rather than crashing on the missing `.attn2` attribute.
        _d1b_enabled = (
            not training
            and use_tactile_views
            and n_view_tactile > 0
            and getattr(self.diffusion_model, "action_expert", False)
            and not getattr(self.diffusion_model, "dual_cross_attn", False)
            and hasattr(self.diffusion_model, "action_blocks")
            and len(self.diffusion_model.action_blocks) > 0
        )
        _d1b_safe_idx = -1
        if _d1b_enabled:
            _n_action_blocks = len(self.diffusion_model.action_blocks)
            _d1b_safe_idx = min(_d1b_block_idx_request, _n_action_blocks - 1)
            attn2 = self.diffusion_model.action_blocks[_d1b_safe_idx].attn2

            def _hook_q(_module, _inputs, output, _cap=_d1b_capture):
                _cap["q"] = output.detach().float()

            def _hook_k(_module, _inputs, output, _cap=_d1b_capture):
                _cap["k"] = output.detach().float()

            def _hook_v(_module, _inputs, output, _cap=_d1b_capture):
                _cap["v"] = output.detach().float()

            _d1b_handles.append(attn2.to_q.register_forward_hook(_hook_q))
            _d1b_handles.append(attn2.to_k.register_forward_hook(_hook_k))
            _d1b_handles.append(attn2.to_v.register_forward_hook(_hook_v))

        # ============ Forward pass ============
        # `n_view_visual` is forwarded so the DiT model can split the action
        # cross-attn K/V into visual vs tactile token sequences when
        # `dual_cross_attn=True` (Option A, plan option-a-dual-crossattn_36ab43c4).
        # When `dual_cross_attn=False` the model ignores this arg and the
        # legacy shared-softmax path is bit-for-bit unchanged.
        try:
            pred_all = forward_pass(
                model=self.diffusion_model,
                timesteps=timesteps,
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                num_frames=latent_frames,
                height=latent_height,
                width=latent_width,
                n_view=n_view,
                n_view_visual=n_view_visual,
                action_states=noisy_actions,
                action_timestep=action_timesteps,
                return_video=self.args.return_video or self.args.return_action,
                return_action=self.args.return_action,
                video_attention_mask=video_attention_mask,
                history_action_state=act_state,
                condition_mask=conditioning_mask,
            )['latents']
        finally:
            for _h in _d1b_handles:
                _h.remove()
            _d1b_handles.clear()

        # ============ D1b metric extraction (val-only) ============
        # All metrics default to 0.0 and stay 0.0 in train / pure-visual /
        # no-action-expert paths, keeping the return-dict schema stable.
        d1b_k_vis_norm = 0.0
        d1b_k_tac_norm = 0.0
        d1b_v_vis_norm = 0.0
        d1b_v_tac_norm = 0.0
        d1b_q_norm = 0.0
        d1b_qk_vis_std = 0.0
        d1b_qk_tac_std = 0.0
        d1b_qk_vis_meanabs = 0.0
        d1b_qk_tac_meanabs = 0.0
        d1b_qk_vis_p95 = 0.0
        d1b_qk_tac_p95 = 0.0
        d1b_cos_q_kvis = 0.0
        d1b_cos_q_ktac = 0.0
        d1b_k_vis_batch_std = 0.0
        d1b_k_tac_batch_std = 0.0
        d1b_block_idx_used = -1
        if (
            _d1b_enabled
            and _d1b_capture["q"] is not None
            and _d1b_capture["k"] is not None
            and _d1b_capture["v"] is not None
        ):
            with torch.no_grad():
                q = _d1b_capture["q"]  # (B, S_q, D_h)
                k = _d1b_capture["k"]  # (B, V*L, D_h)
                v = _d1b_capture["v"]  # (B, V*L, D_h)

                L_per_view = (
                    latent_frames * latent_height * latent_width
                )
                n_vis_rows = n_view_visual * L_per_view
                expected_total = (
                    n_view_visual + n_view_tactile
                ) * L_per_view
                # Sanity: encoder_hidden_states layout for action_blocks
                # is `rearrange(hidden, '(b v) l c -> b (v l) c')` (see
                # transformer_ltx_multiview.py near line 566), so the
                # K/V sequence axis must equal n_view*L_per_view. If a
                # future refactor breaks this contract we'd rather skip
                # D1b cleanly than emit misleading numbers.
                if k.shape[1] == expected_total and n_vis_rows > 0:
                    k_vis = k[:, :n_vis_rows, :]
                    k_tac = k[:, n_vis_rows:, :]
                    v_vis = v[:, :n_vis_rows, :]
                    v_tac = v[:, n_vis_rows:, :]
                    d1b_block_idx_used = int(_d1b_safe_idx)

                    # Mean per-row L2 norm (one scalar per group).
                    d1b_k_vis_norm = float(
                        k_vis.pow(2).sum(dim=-1).sqrt().mean().item()
                    )
                    d1b_k_tac_norm = float(
                        k_tac.pow(2).sum(dim=-1).sqrt().mean().item()
                    )
                    d1b_v_vis_norm = float(
                        v_vis.pow(2).sum(dim=-1).sqrt().mean().item()
                    )
                    d1b_v_tac_norm = float(
                        v_tac.pow(2).sum(dim=-1).sqrt().mean().item()
                    )
                    d1b_q_norm = float(
                        q.pow(2).sum(dim=-1).sqrt().mean().item()
                    )

                    # Proxy QK logits: Q @ K^T at the action_inner_dim
                    # level (no per-head split, no rope, no qk_norm).
                    # Scale by sqrt(D_h) so the magnitudes are
                    # comparable across configs. The RATIO between
                    # `vis` and `tac` is what matters for the
                    # content-collapse hypothesis; the absolute scale
                    # is a fixed multiplier that drops out.
                    d_h = q.shape[-1]
                    inv_sqrt_d = 1.0 / (d_h ** 0.5)
                    qk_vis = torch.bmm(
                        q, k_vis.transpose(1, 2)
                    ) * inv_sqrt_d
                    qk_tac = torch.bmm(
                        q, k_tac.transpose(1, 2)
                    ) * inv_sqrt_d

                    d1b_qk_vis_std = float(qk_vis.std().item())
                    d1b_qk_tac_std = float(qk_tac.std().item())
                    d1b_qk_vis_meanabs = float(qk_vis.abs().mean().item())
                    d1b_qk_tac_meanabs = float(qk_tac.abs().mean().item())
                    if qk_vis.numel() > 0:
                        d1b_qk_vis_p95 = float(
                            torch.quantile(
                                qk_vis.abs().flatten(), 0.95
                            ).item()
                        )
                    if qk_tac.numel() > 0:
                        d1b_qk_tac_p95 = float(
                            torch.quantile(
                                qk_tac.abs().flatten(), 0.95
                            ).item()
                        )

                    # Cosine alignment of mean-Q with mean-K (per group).
                    # Mean over the sequence axis collapses positional
                    # variation, leaving a per-sample "direction"
                    # vector. cos > 0 = aligned, cos ~= 0 = orthogonal.
                    q_mean = q.mean(dim=1)
                    k_vis_mean = k_vis.mean(dim=1)
                    k_tac_mean = k_tac.mean(dim=1)
                    eps = 1e-8
                    cos_q_kvis = (
                        (q_mean * k_vis_mean).sum(dim=-1)
                        / (
                            q_mean.norm(dim=-1)
                            * k_vis_mean.norm(dim=-1) + eps
                        )
                    )
                    cos_q_ktac = (
                        (q_mean * k_tac_mean).sum(dim=-1)
                        / (
                            q_mean.norm(dim=-1)
                            * k_tac_mean.norm(dim=-1) + eps
                        )
                    )
                    d1b_cos_q_kvis = float(cos_q_kvis.mean().item())
                    d1b_cos_q_ktac = float(cos_q_ktac.mean().item())

                    # CONTENT-COLLAPSE METRIC: how much does
                    # `K.mean_over_rows` vary across BATCH samples? If
                    # this is small while ||K_tac|| is normal, the
                    # action-attn K projection delivers nearly the same
                    # vector for every input -> per-sample tactile
                    # conditioning is zero regardless of attention mass.
                    if q.shape[0] >= 2:
                        d1b_k_vis_batch_std = float(
                            k_vis_mean.std(dim=0).mean().item()
                        )
                        d1b_k_tac_batch_std = float(
                            k_tac_mean.std(dim=0).mean().item()
                        )

        # ============ Loss computation ============
        if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
            pred = pred_all['video']
            target = noise - latents
            loss_video_raw = weights.float() * (pred.float() - target.float()).pow(2)
            loss_video_raw = loss_video_raw * (
                1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss_video_raw.size(-1))
            )
            # Per-row mean -> shape (B*n_view,). We keep this per-row form
            # so the (Stage 2) tactile path can split into visual and
            # tactile halves before the final scalar reduction. After
            # Probe Q layout fix (2026-05) the row order is
            # `[vis(b=0,v=*), tac(b=0,vh=*), vis(b=1,v=*), tac(b=1,vh=*), ...]`,
            # so split via view-axis reshape rather than the original
            # `[:B*V_rgb]` / `[B*V_rgb:]` head/tail slice.
            loss_video_per_row = loss_video_raw.mean(
                list(range(1, loss_video_raw.ndim))
            )
            if use_tactile_views and is_stage2 and n_view_tactile > 0:
                # Per-modality split. Split BV into (B, V_full) along the
                # view axis -- visual half is the first n_view_visual
                # views per true_batch, tactile half is the rest.
                loss_per_bv = rearrange(
                    loss_video_per_row, "(b v) -> b v",
                    b=batch_size, v=n_view,
                )
                loss_visual = loss_per_bv[:, :n_view_visual].mean()
                loss_tactile = loss_per_bv[:, n_view_visual:].mean()
                loss_video = (
                    lambda_visual * loss_visual
                    + lambda_tactile * loss_tactile
                )
            else:
                # GE world-model baseline (no tactile) OR
                # use_tactile_views=true reaching this branch via an
                # unexpected combo (defensive fallback). Stage 3
                # `action_*` phases never enter this branch because
                # their train_mode is `action_only` / `action_full`.
                loss_video = loss_video_per_row.mean()
                loss_visual = loss_video.detach()
                loss_tactile = pred.new_zeros(())
        else:
            loss_video = 0.
            loss_visual = 0.
            loss_tactile = 0.

        # Per-dim action-MSE breakdown (val-only diagnostic). Populated below
        # when train_mode is action_* AND yaml has `loss.action_dim_breakdown`.
        # Empty dict in train path; per-slice scalars at val time. The slices
        # use the SAME action_weights as the aggregate loss so per-dim numbers
        # are directly comparable (each is a sigma-weighted mean over the
        # (B, T_chunk, slice_width) sub-tensor).
        action_dim_breakdown_means: Dict[str, Any] = {}

        if (
            self.args.train_mode == 'all'
            or self.args.train_mode == 'action_only'
            or self.args.train_mode == 'action_full'
        ):
            target_action = noise_actions - actions
            loss_action_sq = action_weights.float() * (
                pred_all['action'].float() - target_action.float()
            ).pow(2)
            loss_action = loss_action_sq.mean()
            if not training:
                # Val-only: split the (B, T_chunk, D) per-element squared
                # error along feature dim D into named slices and emit each
                # slice's mean alongside `loss_action`. The yaml schema is:
                #   loss:
                #     action_dim_breakdown:
                #       - [ start, end, "name" ]
                #       - ...
                # Out-of-range slices are surfaced as zeros (not silently
                # dropped) so the val log line shape is yaml-stable across
                # ckpts with mismatched action_in_channels.
                loss_cfg = self._stage2_dict_arg("loss")
                breakdown_cfg = loss_cfg.get("action_dim_breakdown", []) or []
                if breakdown_cfg:
                    D = loss_action_sq.shape[-1]
                    for entry in breakdown_cfg:
                        if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
                            continue
                        start, end, name = int(entry[0]), int(entry[1]), str(entry[2])
                        key = f"loss_action_{name}"
                        if 0 <= start < end <= D:
                            action_dim_breakdown_means[key] = (
                                loss_action_sq[..., start:end].mean()
                            )
                        else:
                            action_dim_breakdown_means[key] = torch.tensor(
                                0.0,
                                device=loss_action_sq.device,
                                dtype=loss_action_sq.dtype,
                            )
        else:
            loss_action = 0.
        action_loss_scale = getattr(self.args, "action_loss_scale", 1.0)

        loss = loss_video + action_loss_scale * loss_action

        assert torch.isnan(loss) == False, "NaN loss detected"

        return {
            "loss": loss,
            "loss_video": loss_video,
            "loss_visual": loss_visual,
            "loss_tactile": loss_tactile,
            "loss_action": loss_action,
            "n_view": n_view,
            "n_view_visual": n_view_visual,
            "n_view_tactile": n_view_tactile,
            "batch_size": batch_size,
            "mem_size": mem_size,
            # Distribution chain diagnostics (val-only; zero in train).
            "input_tac_mean": input_tac_mean,
            "input_tac_std": input_tac_std,
            "input_batch_std": input_batch_std,
            "per_finger_mean": per_finger_mean,
            "per_finger_std": per_finger_std,
            "per_finger_batch_std": per_finger_batch_std,
            "proj_in_mean": proj_in_mean,
            "proj_in_std": proj_in_std,
            "proj_in_batch_std": proj_in_batch_std,
            "proj_out_mean": proj_out_mean,
            "proj_out_total_std": proj_out_total_std,
            "proj_out_batch_std": proj_out_batch_std,
            "vis_mean": vis_mean,
            "vis_total_std": vis_total_std,
            "vis_batch_std": vis_batch_std,
            "proj_alpha": proj_alpha,
            # D1b: action cross-attn K/V + QK content stats
            # (val-only; zero in train or when action_expert is off).
            "d1b_block_idx_used": float(d1b_block_idx_used),
            "d1b_q_norm": d1b_q_norm,
            "d1b_k_vis_norm": d1b_k_vis_norm,
            "d1b_k_tac_norm": d1b_k_tac_norm,
            "d1b_v_vis_norm": d1b_v_vis_norm,
            "d1b_v_tac_norm": d1b_v_tac_norm,
            "d1b_qk_vis_std": d1b_qk_vis_std,
            "d1b_qk_tac_std": d1b_qk_tac_std,
            "d1b_qk_vis_meanabs": d1b_qk_vis_meanabs,
            "d1b_qk_tac_meanabs": d1b_qk_tac_meanabs,
            "d1b_qk_vis_p95": d1b_qk_vis_p95,
            "d1b_qk_tac_p95": d1b_qk_tac_p95,
            "d1b_cos_q_kvis": d1b_cos_q_kvis,
            "d1b_cos_q_ktac": d1b_cos_q_ktac,
            "d1b_k_vis_batch_std": d1b_k_vis_batch_std,
            "d1b_k_tac_batch_std": d1b_k_tac_batch_std,
            # Val-only per-dim action MSE breakdown (empty in train path);
            # keys are `loss_action_<name>` for each entry under
            # args.loss.action_dim_breakdown.
            **action_dim_breakdown_means,
        }


    def train(self):
        # ------------------------------------------------------------------
        # Batch 3b: tactile injection wired into the loop. With
        # `use_tactile_views=true`, after the visual VAE encode we encode
        # tactile through frozen v0c-A + projector (`_encode_tactile_split`),
        # concat the tactile latent along the (B*V) batch axis, bump
        # `n_view` to `V_rgb + V_hand`, and let the existing noise / forward
        # / loss pipeline broadcast over the new view count. Per-modality
        # MSE split is applied in Stage 2 phases (`tactile_projector_only` /
        # `world_model_only`); Stage 3 phases (`action_only` / `action_full`)
        # let tactile views serve as DiT context for the action head and
        # only compute the action loss (no video / tactile co-denoise).
        #
        # With `use_tactile_views=false` the loop falls through bit-for-bit
        # to GE behavior -- this trainer is a strict superset of GE.
        # ------------------------------------------------------------------
        use_tactile_views = bool(getattr(self.args, "use_tactile_views", False))
        phase = getattr(self.args, "phase", None)
        is_stage2 = phase in _STAGE2_PHASES
        is_stage3 = phase in _STAGE3_PHASES

        # Per-modality loss weights (Stage 2 only). In Stage 3 the
        # dispatcher already logged a warning if non-zero; we hard-zero them
        # here so any future code path reading them sees the no-op state.
        if is_stage2 and use_tactile_views:
            loss_cfg = self._stage2_dict_arg("loss")
            # Fail-loud: yaml `lambda_visual: null` -> float(None) -> TypeError;
            # bad string -> ValueError. Both surface immediately rather than
            # silently zeroing a modality (a previous `or 0.0` fallback hid
            # this and could mute tactile loss without any warning).
            lambda_visual = float(loss_cfg.get("lambda_visual", 1.0))
            lambda_tactile = float(loss_cfg.get("lambda_tactile", 1.0))
        else:
            lambda_visual = 1.0
            lambda_tactile = 0.0

        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        
        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        # loss spikes
        anomalies = []

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            # `_prepared_model` is the wrapped engine (DeepSpeedEngine /
            # DDP / bare bundle); `.train()` recurses into both the DiT
            # and the projector when the bundle path is active.
            self._prepared_model.train()

            running_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                # `accelerator.accumulate(...)` expects the wrapped model
                # registered with the engine, not the inner submodule.
                with accelerator.accumulate([ self._prepared_model ]):
                    # ---- Shared forward + loss (Gate 2 contract) --------
                    # `_forward_loss_batch` is invoked verbatim by
                    # :meth:`_compute_val_loss`; the helper's Gate-2 log
                    # block prints the function id from BOTH call sites so
                    # a downstream audit can prove train and val numbers
                    # come from the exact same forward+loss codepath. This
                    # method is responsible only for the per-modality
                    # losses (see helper docstring); backward, gradient
                    # clipping, and optimizer step remain in this loop.
                    out = self._forward_loss_batch(batch, training=True)
                    loss = out["loss"]
                    loss_video = out["loss_video"]
                    loss_visual = out["loss_visual"]
                    loss_tactile = out["loss_tactile"]
                    loss_action = out["loss_action"]
                    n_view_visual = out["n_view_visual"]
                    n_view_tactile = out["n_view_tactile"]

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        # Clip across BOTH DiT + projector params (bundle
                        # path); legacy path's `_prepared_model` IS the DiT
                        # so this also matches GE's original behaviour.
                        # Frozen params have `param.grad is None` and are
                        # skipped by `clip_grad_norm_` automatically.
                        grad_norm = accelerator.clip_grad_norm_(self._prepared_model.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                

                loss = accelerator.reduce(loss.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                    loss_action = accelerator.reduce(loss_action.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                    loss_video = accelerator.reduce(loss_video.detach(), reduction='mean')
                    # Per-modality reductions (only meaningful when Stage 2 +
                    # use_tactile_views=true; otherwise loss_tactile is zeros
                    # and loss_visual mirrors loss_video).
                    if torch.is_tensor(loss_visual):
                        loss_visual = accelerator.reduce(
                            loss_visual.detach(), reduction='mean',
                        )
                    if torch.is_tensor(loss_tactile):
                        loss_tactile = accelerator.reduce(
                            loss_tactile.detach(), reduction='mean',
                        )

                running_loss += loss.item()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    logger.info(">>> max train step reached")
                    break

                if global_step % self.args.steps_to_log == 0:
                    if accelerator.is_main_process:
                        # Always-on stdout train log (grep-friendly in the
                        # detached training log). Independent of `report_to`
                        # / writer presence -- without this, runs with
                        # `report_to: None` (e.g. dryruns and the 488
                        # midtraining curve we want to inspect post-hoc)
                        # have ZERO per-step loss in stdout and we can only
                        # see the final tqdm postfix, which makes it
                        # impossible to spot mid-run NaN spikes or
                        # plateaus. Tag matches grep-conventions used by
                        # the val helper ("[val/...] over ... batches: ...").
                        log_parts = [
                            f"step={global_step}/{self.state.train_steps}",
                            f"loss={loss.item():.4f}",
                            f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}",
                        ]
                        if self.args.train_mode in (
                            'all', 'action_only', 'action_full'
                        ) and torch.is_tensor(loss_action):
                            log_parts.append(
                                f"action={loss_action.mean().item():.4f}"
                            )
                        if self.args.train_mode in ('all', 'video_only'):
                            if torch.is_tensor(loss_video):
                                log_parts.append(
                                    f"video={loss_video.item():.4f}"
                                )
                            if (
                                use_tactile_views
                                and is_stage2
                                and n_view_tactile > 0
                                and torch.is_tensor(loss_visual)
                                and torch.is_tensor(loss_tactile)
                            ):
                                log_parts.append(
                                    f"visual={loss_visual.item():.4f}"
                                )
                                log_parts.append(
                                    f"tactile={loss_tactile.item():.4f}"
                                )
                        # Option A: tactile_gate trajectory (one scalar per
                        # action block; init -3.0 -> sigmoid ~ 0.047).
                        # Logged every steps_to_log so we can watch the
                        # action expert opt in to tactile (gate growth)
                        # vs stay near init (architecture fine but tactile
                        # content weak at this operating point). Reports
                        # raw scalar (so we can see the parameter update
                        # directly) AND sigmoid (the actual residual
                        # multiplier) summarized as mean / min / max
                        # across all blocks plus per-block values at the
                        # first / middle / last layer. ZeRO-2 fully
                        # replicates parameter values across ranks so
                        # main-process read is consistent. Gated on
                        # dual_cross_attn so no overhead in legacy runs.
                        if getattr(self.diffusion_model, "dual_cross_attn", False) and hasattr(self.diffusion_model, "action_blocks"):
                            with torch.no_grad():
                                _gates = torch.stack([
                                    blk.tactile_gate.detach().float()
                                    for blk in self.diffusion_model.action_blocks
                                ]).cpu()
                                _sig = torch.sigmoid(_gates)
                                _n_blk = _gates.numel()
                                _i_mid = _n_blk // 2
                            log_parts.append(
                                f"gate=mean/min/max={_gates.mean().item():+.3f}/"
                                f"{_gates.min().item():+.3f}/{_gates.max().item():+.3f}"
                            )
                            log_parts.append(
                                f"gate_sig=mean/min/max={_sig.mean().item():.4f}/"
                                f"{_sig.min().item():.4f}/{_sig.max().item():.4f}"
                            )
                            log_parts.append(
                                f"gate_sig[0,{_i_mid},{_n_blk - 1}]="
                                f"{_sig[0].item():.4f}/"
                                f"{_sig[_i_mid].item():.4f}/"
                                f"{_sig[-1].item():.4f}"
                            )
                        logger.info("[train] " + "  ".join(log_parts))

                        if self.writer is not None:
                            self.writer.add_scalar("Training Loss", loss.item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                                self.writer.add_scalar("Action loss", loss_action.mean().item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                                self.writer.add_scalar("Video loss", loss_video.item(), global_step)
                                # Per-modality breakdown (Stage 2 + tactile only).
                                if (
                                    use_tactile_views
                                    and is_stage2
                                    and n_view_tactile > 0
                                    and torch.is_tensor(loss_visual)
                                    and torch.is_tensor(loss_tactile)
                                ):
                                    self.writer.add_scalar(
                                        "Visual loss", loss_visual.item(), global_step,
                                    )
                                    self.writer.add_scalar(
                                        "Tactile loss", loss_tactile.item(), global_step,
                                    )
                            # Option A: tactile_gate scalars. _gates / _sig
                            # are already populated above (same gate-on
                            # block) and we're still inside
                            # `accelerator.is_main_process`, so re-using
                            # them here is consistent and avoids a second
                            # parameter read. tag prefix `tactile_gate/`
                            # groups all 5 curves under one tensorboard
                            # heading for the M-vs-K-vs-J plot stack.
                            if getattr(self.diffusion_model, "dual_cross_attn", False) and hasattr(self.diffusion_model, "action_blocks"):
                                self.writer.add_scalar(
                                    "tactile_gate/sigmoid_mean", _sig.mean().item(), global_step,
                                )
                                self.writer.add_scalar(
                                    "tactile_gate/sigmoid_min", _sig.min().item(), global_step,
                                )
                                self.writer.add_scalar(
                                    "tactile_gate/sigmoid_max", _sig.max().item(), global_step,
                                )
                                self.writer.add_scalar(
                                    "tactile_gate/raw_mean", _gates.mean().item(), global_step,
                                )
                                self.writer.add_scalar(
                                    "tactile_gate/raw_max", _gates.max().item(), global_step,
                                )

                if global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(self.save_folder,f'Validation_step_{global_step}')
                        # validate() only consumes RGB views (no tactile in
                        # the val pipeline yet); pass n_view_visual so the
                        # gt_video rearrange shape matches `(c, V_rgb, T, H, W)`.
                        self.validate(accelerator, model_save_dir, global_step, n_view=n_view_visual, n_chunk=1)

                    # A3 multi-split val-loss path (ALL ranks). Each named
                    # split in self.val_loaders gets its own
                    # _compute_val_loss call; the helper is rank-aware
                    # (accelerator.reduce internally) and uses the A2
                    # 4-stream RNG snapshot/restore so back-to-back calls
                    # don't pollute training RNG.
                    #
                    # `tag=split_name` (no `val/` prefix); the helper
                    # internally formats `[val/{tag}] ...` which produces
                    # e.g. `[val/holdout_488] over N batches: loss=...`.
                    # Double-prefix `val/val/...` is avoided by NOT
                    # prepending here.
                    #
                    # No-op when self.val_loaders is {} (no
                    # data.val_splits yaml). Barrier first so non-main
                    # ranks (which skipped the legacy `self.validate()`
                    # block above) don't race into `_compute_val_loss`
                    # while main is still inside the pipeline call --
                    # `_compute_val_loss` uses NCCL allreduce internally
                    # and would deadlock if ranks entered out of order.
                    if self.val_loaders:
                        accelerator.wait_for_everyone()
                        for split_name, val_loader in self.val_loaders.items():
                            self._compute_val_loss(
                                val_loader, tag=split_name,
                            )

                
                if global_step % self.args.steps_to_save == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
                        # Save DiT for any phase EXCEPT tactile_projector_only,
                        # where the DiT is frozen and saving it just wastes
                        # disk space and is misleading. Stage 2 phase 1
                        # exclusively persists the projector ckpt.
                        if phase != "tactile_projector_only":
                            model_to_save = unwrap_model(accelerator, self.diffusion_model)
                            dtype = (
                                torch.float16
                                if self.args.mixed_precision == "fp16"
                                else torch.bfloat16
                                if self.args.mixed_precision == "bf16"
                                else torch.float32
                            )

                            model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                            del  model_to_save
                        # Save projector whenever it exists, mirroring GE's
                        # atomic-checkpoint philosophy: a deployable
                        # submodule is dumped as a whole regardless of
                        # which params were actually updated this step.
                        # In Stage 3 + tactile=true with the default frozen
                        # projector this re-writes a byte-identical copy of
                        # the warmstart projector each save_step (~270 KB,
                        # cheap), so the action-phase ckpt directory is
                        # self-contained for inference -- no need for the
                        # consumer to chase down `projector.warmstart_ckpt`.
                        if self.projector is not None:
                            os.makedirs(model_save_dir, exist_ok=True)
                            projector_path = os.path.join(model_save_dir, "projector.pt")
                            self._save_projector_ckpt(projector_path, global_step)
                        
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

            # GE-inherited bug fix: the inner step loop's `break` at
            # `if global_step >= self.state.train_steps` only exits the
            # current epoch's iteration of the dataloader; without this
            # outer check the for-epoch loop simply starts a new epoch
            # and runs ONE MORE wasted optimizer step (because the
            # max-step check is AFTER processing). With train_epochs set
            # to a large ceiling (e.g. 10000) and steps_per_epoch ~ 7,
            # the production phase 1 / phase 2 / Stage 3 yamls would
            # otherwise burn ~3x the intended train_steps in wasted
            # extra iterations after the optim budget is exhausted.
            if global_step >= self.state.train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else torch.bfloat16
                if self.args.mixed_precision == "bf16"
                else torch.float32
            )

            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
            # Mirror the per-step save policy: skip DiT save in phase 1
            # (DiT is frozen and 5 GB writing 25x to disk would be silly);
            # save projector whenever it exists (atomic-unit semantics --
            # see _save_projector_ckpt docstring).
            if phase != "tactile_projector_only":
                self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)
            if self.projector is not None:
                os.makedirs(model_save_dir, exist_ok=True)
                projector_path = os.path.join(model_save_dir, "projector.pt")
                self._save_projector_ckpt(projector_path, global_step)

        # Post-save barrier: rank 0 just did a multi-GB
        # `diffusion_model.save_pretrained` + projector dump inside the
        # `if accelerator.is_main_process:` block above. Without this
        # barrier, all OTHER ranks (which skipped the save block
        # entirely and have ~0 work between the pre-save barrier at the
        # top of this section and the del/cleanup below) race ahead,
        # tear down NCCL via `accelerator.end_training()`, and exit the
        # Python interpreter while rank 0 is still inside save I/O.
        # `torch.distributed.elastic` then reports a non-zero exitcode
        # (observed: 120 on rank 1+, with `error_file: <N/A>` and
        # `<NO_OTHER_FAILURES>`), turning a fully successful run --
        # ckpt files intact on disk, no NaN, training loop reached
        # max-step -- into a noisy "FAILED" return code that breaks
        # any wrapper script checking $?. This barrier costs ~one
        # ckpt-save's worth of wait on the non-main ranks but produces
        # a clean exit-0 across all ranks. See the in-line note on the
        # GC-timing fix below for the OTHER half of the exitcode-120
        # mitigation (engine reference release).
        accelerator.wait_for_everyone()

        # Release the wrapped engine reference BEFORE accelerator.end_training()
        # so the DeepSpeed engine's __del__ runs while the NCCL process group
        # is still alive, not during interpreter shutdown.
        #
        # Why this matters: with the _Stage2Bundle approach, self.diffusion_model
        # is the inner DiT submodule, NOT the engine -- the engine is held by
        # self._prepared_model, AND self.projector also strong-refs the inner
        # projector submodule (bundle.projector). Without explicitly clearing
        # both here, the engine survives `del self.diffusion_model` and is
        # GC'd later during Python interpreter shutdown, which races with
        # NCCL process-group teardown and produces a silent exitcode 120
        # on rank 1+ ("Exception ignored in sys.unraisablehook" with no
        # traceback). Single-rank runs don't expose this because the
        # process group is trivial; multi-rank does. The legacy no-tactile
        # path is unaffected (self._prepared_model aliases the wrapped
        # diffusion_model so the second nullify is idempotent).
        del self.diffusion_model, self.scheduler
        self._prepared_model = None
        self.projector = None

        # A3 / multi-loader exitcode-120 fix: explicitly release the
        # OTHER long-lived attributes that hold NCCL-aware or
        # multiprocessing-aware objects, so their __del__ runs while
        # the process group + main interpreter are still alive instead
        # of during interpreter shutdown.
        #
        # The earlier exitcode-120 fix above covers the DeepSpeed
        # engine reference (held by self._prepared_model). But after
        # the A3 patch added `self.val_loaders` (a dict of N extra
        # DataLoaders, each with its own pool of `dataloader_num_workers`
        # subprocess workers) on top of the existing
        # `self.train_dataloader`, the per-rank worker count went from
        # ~8 to ~16-24. Interpreter shutdown GCs these DataLoader
        # objects in non-deterministic order alongside `self.optimizer`
        # / `self.lr_scheduler` (DeepSpeed-wrapped, ProcessGroupNCCL-aware),
        # `self.writer` (TB SummaryWriter holds file handles + a flush
        # thread), and `self.tactile_vae` (frozen VAE holding bf16
        # CUDA tensors). When any of their `__del__`s fires AFTER NCCL
        # has already been torn down by `accelerator.end_training()`,
        # CPython prints "Exception ignored in sys.unraisablehook" and
        # exits the rank with code 120 -- producing exactly the
        # observed "FAILED ... rank: 1 ... exitcode: 120" pattern even
        # though training itself succeeded (all ckpts saved, ">>> max
        # train step reached" + "Memory after training end" both logged).
        #
        # Cleanup order matters: dataloaders first (kills worker pools
        # while CUDA + NCCL are still alive so workers' tensor cleanup
        # can use the GPU), then DeepSpeed-wrapped optim/sched, then
        # writer (close() flushes events + joins its background thread),
        # then frozen VAE. Each step is guarded so a partial trainer
        # state (e.g. failure before optimizer was built) doesn't
        # double-fault here.
        #
        # try/except is belt-and-suspenders: if any del raises, the
        # final NCCL teardown via end_training() must still run, so
        # we don't swap one cause of exitcode 120 for another.
        try:
            if hasattr(self, 'train_dataloader'):
                del self.train_dataloader
            if hasattr(self, 'val_loaders') and self.val_loaders:
                self.val_loaders.clear()
            if hasattr(self, 'optimizer'):
                del self.optimizer
            if hasattr(self, 'lr_scheduler'):
                del self.lr_scheduler
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.close()
                self.writer = None
            if hasattr(self, 'tactile_vae'):
                self.tactile_vae = None
        except Exception as cleanup_err:
            logger.warning(
                f"[teardown] non-fatal cleanup raised: "
                f"{type(cleanup_err).__name__}: {cleanup_err}"
            )

        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

        # Force-destroy NCCL process group WHILE the GIL + CUDA context are
        # still healthy, NOT during Python interpreter shutdown.
        #
        # Why this is needed even after accelerator.end_training():
        # accelerator.end_training() only finalizes trackers (TB, wandb);
        # it does NOT call torch.distributed.destroy_process_group(). The
        # default ProcessGroupNCCL therefore stays alive until interpreter
        # shutdown, when the runner object in main.py goes out of scope.
        # At that point the GC + atexit chain races with NCCL teardown,
        # and one or more __del__ methods (DeepSpeed atexit stats dump,
        # ProcessGroupNCCL.__del__, accelerator.__del__) raise after the
        # NCCL communicator is half-torn-down. CPython prints
        # "Exception ignored in sys.unraisablehook" on all ranks AND
        # exits the FIRST rank to finish finalization with code 120
        # (CPython's reserved code for "unhandled exception during
        # interpreter finalization"). torchrun then reaps that rank,
        # decides the run FAILED, SIGTERMs the other ranks, and reports
        # the whole job as a failure -- even though training completed
        # successfully (all ckpts on disk, >>> max train step reached
        # logged, Memory after training end logged on all ranks).
        #
        # Observed empirically on the B1 50-step dryrun: 4-rank run
        # consistently exits with exitcode 120 on rank 1 after both the
        # earlier teardown patches (engine ref nullify + A3 multi-loader
        # cleanup of train_dataloader/val_loaders/optimizer/lr_scheduler/
        # writer/tactile_vae). Those patches reduced the surface area
        # by clearing the heavyweight user-side refs, but the NCCL PG
        # itself is held by the framework and only gets dropped at
        # process exit -- which is the actual race.
        #
        # Fix: deterministically destroy the PG here. After this call,
        # `torch.distributed.is_initialized()` returns False on all
        # ranks. Any subsequent atexit / __del__ / GC callback that
        # tries a collective sees the no-init state, raises, and is
        # caught by Python's unraisable hook -- WITHOUT escalating to
        # exit 120, because the exception now happens BEFORE interpreter
        # finalization (during normal Python execution) where unraisable
        # hooks are properly absorbed.
        #
        # accelerator.wait_for_everyone() (not raw dist.barrier()) is
        # the final sync: it uses the correct device_ids internally and
        # avoids the noisy "No device id is provided via barrier"
        # UserWarning that we observed 4x in the previous dryrun logs.
        try:
            accelerator.wait_for_everyone()
        except Exception as sync_err:
            logger.warning(
                f"[teardown] final wait_for_everyone raised "
                f"(non-fatal, continuing to destroy_process_group): "
                f"{type(sync_err).__name__}: {sync_err}"
            )
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("[teardown] destroyed default NCCL process group")
        except Exception as pg_err:
            logger.warning(
                f"[teardown] destroy_process_group raised "
                f"(non-fatal, exit may still be 120): "
                f"{type(pg_err).__name__}: {pg_err}"
            )

        # Drop the trainer's strong ref to the accelerator so its
        # __del__ runs while we still hold the GIL and have a known
        # Python state. Without this, the accelerator survives until
        # main() returns and gets GC'd during interpreter shutdown,
        # where its destructor again races with whatever DeepSpeed /
        # torch.distributed have left running.
        try:
            self.state.accelerator = None
        except Exception:
            pass


    def _compute_val_loss(
        self,
        val_dataloader,
        *,
        max_batches: Optional[int] = None,
        tag: str = "val",
    ) -> Dict[str, float]:
        """Compute mean validation loss(es) over a held-out dataloader.

        This method exists for one reason: training loss alone is a poor
        signal once the model starts overfitting. For the Stage 2 488-
        episode midtraining comparison we need an apples-to-apples val
        number on the SAME forward+loss recipe as training -- otherwise
        any val/train gap could be an artifact of a forward path that
        differs from train, rather than an actual generalization gap.

        Implementation contract (Gate 2):

          * The per-batch math is delegated EXCLUSIVELY to
            :meth:`_forward_loss_batch` (training=False). There is NO
            duplicated loss code in this method. The Gate-2 log block
            in the helper prints the same function id from both
            ``training=True`` (train loop) and ``training=False`` (here)
            so a downstream audit can prove the codepath is shared.
          * Gradient tracking is disabled via ``torch.no_grad()``.
          * The model is put into ``eval()`` mode for the duration and
            restored to ``train()`` mode on the way out. The eval-mode
            scope is bounded by ``try/finally`` so an exception during
            val never leaves the model in eval.
          * Pre-val RNG state (CPU + per-CUDA-device) is snapshotted and
            restored on exit, then re-seeded with ``args.val_seed``
            (default 12345) inside the snapshot so val noise (timestep
            sampling, conditioning noise, prompt dropout if any) is
            REPRODUCIBLE across runs and BIT-INDEPENDENT of where the
            training loop's RNG sits at the moment we call val. Without
            this, training itself would become non-reproducible because
            every val pass would burn a different number of samples from
            the global RNG depending on dataloader timing.

        Args:
            val_dataloader: an iterable yielding batches with the same
                schema as the train dataloader (DexVTAMDataset emits
                ``video``, ``caption``, and optionally ``tactile``,
                ``hand_pose``, ``actions``, ``state``).
            max_batches: if not None, stop after this many batches. Useful
                for fast smoke runs.
            tag: label used in the log header for easier grep (e.g.
                ``"val_488"``, ``"val_cube"``).

        Returns:
            dict[str, float] with the mean (across batches AND across
            ranks for the DDP case) of each loss component:
                ``loss``, ``loss_video``, ``loss_visual``,
                ``loss_tactile``, ``loss_action``, plus ``num_batches``.
            All floats are Python floats on the main process (already
            scalar after ``accelerator.reduce``).
        """
        import random as _py_random
        import numpy as _np
        accelerator = self.state.accelerator
        was_training = self._prepared_model.training
        val_seed = int(getattr(self.args, "val_seed", 12345))

        # ---- Snapshot RNG state for deterministic val ---------------------
        # Why: many forward-path components consume from a GLOBAL RNG and
        # different libraries use DIFFERENT global RNGs. We have to handle
        # all four or val becomes non-deterministic across back-to-back
        # calls. Without a complete snapshot two things break:
        #   1) val numbers depend on what step training is at, so a
        #      val-curve plotted across steps mixes "real" gen gap with
        #      RNG drift.
        #   2) running val also bumps the training RNG forward by a
        #      val-dataloader-shaped amount, making training itself less
        #      reproducible.
        #
        # Concretely, the four streams we have to handle:
        #   * torch CPU RNG     -> timestep sampling, conditioning-frame
        #                          noise, sometimes attention dropout.
        #   * torch CUDA RNG    -> noise tensors produced on-device.
        #   * Python `random`   -> ``DexVTAMDataset.get_frame_indexes()``
        #                          calls ``random.randint`` to pick the
        #                          clip-end index, and ``__getitem__``
        #                          retries with ``random.randint`` on
        #                          dataset-level errors. The dataloader's
        #                          ``__iter__`` consumes these on every
        #                          iter() call.
        #   * numpy `np.random` -> same dataset path uses
        #                          ``np.random.choice`` to pick memory
        #                          frame indices.
        #
        # If we only snapshot torch (the v0d val-smoke determinism bug),
        # Call N+1's dataloader iter sees a Python/numpy RNG that has
        # already been advanced by Call N -> different ``chunk_end`` /
        # ``mem_indexes`` -> different batch -> ~20% loss drift across
        # back-to-back val invocations. The two extra snapshots below
        # close that hole.
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else None
        )
        py_rng = _py_random.getstate()
        np_rng = _np.random.get_state()

        sums = {
            "loss": 0.0,
            "loss_video": 0.0,
            "loss_visual": 0.0,
            "loss_tactile": 0.0,
            "loss_action": 0.0,
            # Distribution chain diagnostics (val-only). Aggregated
            # through the same mean/reduce path as loss so cross-rank
            # averages are comparable across runs.
            "input_tac_mean": 0.0,
            "input_tac_std": 0.0,
            "input_batch_std": 0.0,
            "per_finger_mean": 0.0,
            "per_finger_std": 0.0,
            "per_finger_batch_std": 0.0,
            "proj_in_mean": 0.0,
            "proj_in_std": 0.0,
            "proj_in_batch_std": 0.0,
            "proj_out_mean": 0.0,
            "proj_out_total_std": 0.0,
            "proj_out_batch_std": 0.0,
            "vis_mean": 0.0,
            "vis_total_std": 0.0,
            "vis_batch_std": 0.0,
            "proj_alpha": 0.0,
            # D1b action cross-attn K/V + QK content stats. Averaged
            # across val batches / DDP ranks the same way as the chain
            # stats above. `d1b_block_idx_used` averages a stable int
            # so the printed value reflects the (single) block we hooked.
            "d1b_block_idx_used": 0.0,
            "d1b_q_norm": 0.0,
            "d1b_k_vis_norm": 0.0,
            "d1b_k_tac_norm": 0.0,
            "d1b_v_vis_norm": 0.0,
            "d1b_v_tac_norm": 0.0,
            "d1b_qk_vis_std": 0.0,
            "d1b_qk_tac_std": 0.0,
            "d1b_qk_vis_meanabs": 0.0,
            "d1b_qk_tac_meanabs": 0.0,
            "d1b_qk_vis_p95": 0.0,
            "d1b_qk_tac_p95": 0.0,
            "d1b_cos_q_kvis": 0.0,
            "d1b_cos_q_ktac": 0.0,
            "d1b_k_vis_batch_std": 0.0,
            "d1b_k_tac_batch_std": 0.0,
        }
        num_batches = 0
        # Per-dim action-MSE breakdown keys: discovered from the first val
        # batch's `_forward_loss_batch` return dict (any key prefixed with
        # `loss_action_`, e.g. `loss_action_force` / `loss_action_arm_pose`).
        # Empty list when args.loss.action_dim_breakdown is unset (cube v4 /
        # B2 video-only runs). Discovery happens at runtime so this code
        # path is unconditional on action_in_channels.
        action_breakdown_keys: List[str] = []

        self._prepared_model.eval()
        try:
            # Re-seed every global RNG we touch above. The val_seed
            # convention matches torch.manual_seed exactly: same int
            # value across all four streams, so a downstream debugger
            # can reproduce the val batch by setting
            # `torch/np/random.seed(val_seed)` outside the trainer too.
            torch.manual_seed(val_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(val_seed)
            _py_random.seed(val_seed)
            _np.random.seed(val_seed)

            with torch.no_grad():
                for step, batch in enumerate(val_dataloader):
                    if max_batches is not None and step >= max_batches:
                        break
                    out = self._forward_loss_batch(batch, training=False)
                    # Discover per-dim action-MSE breakdown keys on the
                    # first batch and seed the sums dict so subsequent
                    # batches accumulate into the same slots.
                    if num_batches == 0:
                        for k in out:
                            if (
                                isinstance(k, str)
                                and k.startswith("loss_action_")
                                and k != "loss_action"
                                and k not in sums
                            ):
                                sums[k] = 0.0
                                action_breakdown_keys.append(k)
                    for k in sums:
                        v = out.get(k)
                        if torch.is_tensor(v):
                            sums[k] += float(v.detach().float().item())
                        elif isinstance(v, (int, float)):
                            sums[k] += float(v)
                        # Missing keys (or None) treated as 0; this only
                        # happens when train_mode skips a modality.
                    num_batches += 1
        finally:
            # Always restore EVERY RNG snapshotted above, even on
            # exception. Order doesn't matter (the streams are
            # independent), but we restore in the reverse of the
            # snapshot order for symmetry / readability.
            if was_training:
                self._prepared_model.train()
            _np.random.set_state(np_rng)
            _py_random.setstate(py_rng)
            if cuda_rng is not None:
                for d, state in enumerate(cuda_rng):
                    torch.cuda.set_rng_state(state, d)
            torch.set_rng_state(cpu_rng)

        # Per-rank means; then reduce across DDP ranks.
        if num_batches == 0:
            # Empty dataloader. Surface a warning but don't crash the
            # surrounding train loop -- the caller can decide.
            logger.warning(
                f"[val/{tag}] val_dataloader produced 0 batches; returning zeros."
            )
            means = {k: 0.0 for k in sums}
            means["num_batches"] = 0
            return means

        per_rank_means = {k: v / num_batches for k, v in sums.items()}

        # accelerator.reduce expects a tensor; pack/unpack for the mean.
        means: Dict[str, float] = {}
        for k, v in per_rank_means.items():
            t = torch.tensor([v], device=accelerator.device, dtype=torch.float32)
            t = accelerator.reduce(t, reduction="mean")
            means[k] = float(t.item())
        means["num_batches"] = int(num_batches)

        if accelerator.is_main_process:
            vbs = means.get("vis_batch_std", 0.0)
            pbs = means.get("proj_out_batch_std", 0.0)
            batch_ratio = (pbs / vbs) if vbs > 1e-8 else 0.0
            # Option C extension (2026-05-24): print batch_std at every
            # stage of the encoding chain so the reader can localize the
            # batch-level collapse (input -> finger -> in -> out).
            inbs = means.get("input_batch_std", 0.0)
            fbs = means.get("per_finger_batch_std", 0.0)
            pibs = means.get("proj_in_batch_std", 0.0)
            # Format per-dim action breakdown as a single space-separated
            # line under the main val summary. Skipped silently when the
            # yaml has no `loss.action_dim_breakdown` entries.
            if action_breakdown_keys:
                per_dim_str = "  ".join(
                    f"{k[len('loss_action_'):]}={means[k]:.6f}"
                    for k in action_breakdown_keys
                )
                per_dim_line = f"  action per-dim: {per_dim_str}\n"
            else:
                per_dim_line = ""
            # D1b summary line (Option C part B). All zero when
            # action_expert is off or val saw no tactile views.
            d1b_block_used = int(round(means.get("d1b_block_idx_used", -1.0)))
            d1b_kt_bs = means.get("d1b_k_tac_batch_std", 0.0)
            d1b_kv_bs = means.get("d1b_k_vis_batch_std", 0.0)
            d1b_kt_ratio = (
                (d1b_kt_bs / d1b_kv_bs) if d1b_kv_bs > 1e-8 else 0.0
            )
            if d1b_block_used >= 0:
                d1b_lines = (
                    f"  D1b @ action_blocks[{d1b_block_used}].attn2:\n"
                    f"    norms:  ||Q||={means.get('d1b_q_norm', 0.0):.4f}  "
                    f"||K_vis||={means.get('d1b_k_vis_norm', 0.0):.4f}  "
                    f"||K_tac||={means.get('d1b_k_tac_norm', 0.0):.4f}  "
                    f"||V_vis||={means.get('d1b_v_vis_norm', 0.0):.4f}  "
                    f"||V_tac||={means.get('d1b_v_tac_norm', 0.0):.4f}\n"
                    f"    QK_vis: std={means.get('d1b_qk_vis_std', 0.0):.4f}  "
                    f"meanabs={means.get('d1b_qk_vis_meanabs', 0.0):.4f}  "
                    f"p95={means.get('d1b_qk_vis_p95', 0.0):.4f}\n"
                    f"    QK_tac: std={means.get('d1b_qk_tac_std', 0.0):.4f}  "
                    f"meanabs={means.get('d1b_qk_tac_meanabs', 0.0):.4f}  "
                    f"p95={means.get('d1b_qk_tac_p95', 0.0):.4f}\n"
                    f"    cos(Q,K_vis)={means.get('d1b_cos_q_kvis', 0.0):+.4f}  "
                    f"cos(Q,K_tac)={means.get('d1b_cos_q_ktac', 0.0):+.4f}\n"
                    f"    K_vis batch_std={d1b_kv_bs:.4f}  "
                    f"K_tac batch_std={d1b_kt_bs:.4f}  "
                    f"K_tac/K_vis={d1b_kt_ratio:.3f}\n"
                )
            else:
                d1b_lines = ""
            logger.info(
                f"[val/{tag}] over {num_batches} batches: "
                f"loss={means['loss']:.6f}  "
                f"video={means['loss_video']:.6f}  "
                f"visual={means['loss_visual']:.6f}  "
                f"tactile={means['loss_tactile']:.6f}  "
                f"action={means['loss_action']:.6f}\n"
                f"{per_dim_line}"
                f"  alpha={means.get('proj_alpha', 0.0):.4f}\n"
                f"  input:  mean={means.get('input_tac_mean', 0.0):+.4f}  "
                f"std={means.get('input_tac_std', 0.0):.4f}  "
                f"batch_std={inbs:.4f}\n"
                f"  finger: mean={means.get('per_finger_mean', 0.0):+.4f}  "
                f"std={means.get('per_finger_std', 0.0):.4f}  "
                f"batch_std={fbs:.4f}\n"
                f"  in:     mean={means.get('proj_in_mean', 0.0):+.4f}  "
                f"std={means.get('proj_in_std', 0.0):.4f}  "
                f"batch_std={pibs:.4f}\n"
                f"  out:    mean={means.get('proj_out_mean', 0.0):+.4f}  "
                f"total_std={means.get('proj_out_total_std', 0.0):.4f}  "
                f"batch_std={pbs:.4f}\n"
                f"  vis:    mean={means.get('vis_mean', 0.0):+.4f}  "
                f"total_std={means.get('vis_total_std', 0.0):.4f}  "
                f"batch_std={vbs:.4f}\n"
                f"  batch_ratio={batch_ratio:.3f}\n"
                f"{d1b_lines}"
            )
        return means


    def validate(self, accelerator, model_save_dir, global_step, n_view=1, n_chunk=30, image=None, prompt=None, cap=None, path=None, gt_actions=None, to_log=True):

        os.makedirs(model_save_dir,exist_ok=True)

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer,
            unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model
        )

        batch = next(iter(self.val_dataloader))
        image = batch['video'][:,:,:,:self.args.data['train']['n_previous']].clone()  # shape b,c,v,t,h,w 
        prompt = batch['caption']
        gt_video = batch['video']
        b, c, v, t, h, w = image.shape
        negative_prompt = ''

        batch_size = 1

        image = image[:batch_size]

        image = rearrange(image, 'b c v t h w -> (b v) c t h w')
        num_denois_steps = self.args.num_inference_step

        if self.args.return_action and getattr(self.args, "add_state", False):
            history_action_state = batch['state'][:batch_size]
            if history_action_state.shape[1] > 1:
                history_action_state = history_action_state[:, self.args.data['train']['n_previous']-1:self.args.data['train']['n_previous'], :]
            history_action_state = history_action_state.contiguous()
        else:
            history_action_state = None

        preds = pipe.infer(
            image=image,
            prompt=prompt[:batch_size],
            negative_prompt=negative_prompt,
            num_inference_steps=num_denois_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=self.args.return_action,
            n_prev=self.args.data['train']['n_previous'],
            chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
            return_video=self.args.return_video,
            noise_seed=42,
            action_chunk=self.args.data['train']['action_chunk'],
            history_action_state = history_action_state,
            pixel_wise_timestep = self.args.pixel_wise_timestep,
            n_chunk=n_chunk,
            action_dim=self.args.diffusion_model["config"]["action_in_channels"] if self.args.return_action else None,
        )[0]

        cap = 'Validation'
        fps = int(getattr(self.args, "basic_fps", 30) / (self.args.data['train']['action_chunk'] // self.args.data['train']['chunk']))
        save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{cap}_gt.mp4'), fps=fps)

        if self.args.return_video:
            video = preds['video'].data.cpu()
            save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], os.path.join(model_save_dir, f'{cap}.mp4'), fps=fps)

        if to_log:
            self.writer.add_text(f'step_{global_step}/{cap} prompt:', prompt[0], global_step)

        if self.args.return_action:
            # shape t, c
            gt_actions = batch['actions'][:, -self.args.data['train']['action_chunk']:]
            action_dim = gt_actions.shape[-1]

            action_logs = act_metric(
                preds['action'][:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                gt_actions[:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                prefix=cap,
                start_stop_interval=[(0,1),(1,9),(9,25),(25,self.args.data['train']['action_chunk'])]
            )

            if to_log:
                for key, value in action_logs.items():
                    self.writer.add_scalar(key, value, global_step)

