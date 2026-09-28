"""Stage 1 lite trainer: frozen LTX visual VAE + tactile adapter probe.

Subclasses :class:`runner.tactile_vae_trainer.TactileVAETrainer` so all the
dataset / sampler / optimizer / wandb / checkpoint plumbing is reused
verbatim. Only the model construction, the loss computation, the validation
metric stream, and the train-step loss formula are overridden.

Architecture (see plan stage1_lite_visual_vae_adapter_probe)::

    gray tactile (B, 5, 1, T, 192, 256)
        -> GrayToRGB              (init weight=1, bias=0; trainable, ~6 params)
        -> FROZEN AutoencoderKLLTXVideo.encode (latent_dist.mode by default)
        -> per-finger latent (B, 5, 128, T_lat, 6, 8)
        -> AuxPreFuseFlowHead    (diagnostic; always trained when enabled)
        -> FingerAttentionAdapter
              hand_query + pos_embed[h,w] for Q
              z + finger_embed[5] for K/V
              softmax-weighted finger mean residual + alpha-gated attn
        -> per-hand latent (B, 128, T_lat, 6, 8)   (C5 contract enforced)
        -> ModalityEmbedding
        -> AuxFlowDecoder + AuxPoseDecoder

Loss::

    L_total = lambda_loc * L_flow_post
            + lambda_loc_pre * L_flow_pre
            + lambda_pose * L_pose
        (KL term dropped: VAE encoder is frozen, no posterior to regularize.)

Best-checkpoint metric: ``val_flow_mse_active_mean`` over the **post-fuse**
stream (the actual deployment latent for Stage 2 WM training).

The 1k / 2k early kill-or-keep gate reads ``train/loss_flow_post`` vs
``train/loss_flow_pre`` to attribute encoder-vs-adapter blame in real time.

Critical correctness invariant: VAE freeze is done via ``requires_grad_(False)``
ONLY (set inside ``VisualVAEAdapterModel.__init__``). The trainer does NOT
wrap ``vae.encode`` in ``torch.no_grad()``; gradients must flow back through
the VAE to ``GrayToRGB`` so the latter can learn its chromatic remapping.
The first training step asserts this invariant explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from tqdm import tqdm

from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from models.tactile_models.visual_vae_adapter import VisualVAEAdapterModel
from runner.tactile_vae_trainer import (
    TactileVAETrainer,
    _load_stats_json,
    _move_batch_to_device,
)
from utils.model_utils import load_latent_models, load_vae_models

LOG_LEVEL = "INFO"
logger = get_logger("visual_vae_adapter_trainer")
logger.setLevel(LOG_LEVEL)


_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


# ---------------------------------------------------------------------------
# v4b contact-aware accumulator (factored out so we can run it on the post-fuse
# AND pre-fuse streams independently in validate()).
# ---------------------------------------------------------------------------


class _V4BAccumulator:
    """Accumulates the v4b contact-aware metrics for one prediction stream.

    All sums kept in float64 on device for numerical stability across batches.
    """

    def __init__(self, device: torch.device):
        self.active_sq      = torch.zeros(5, device=device, dtype=torch.float64)
        self.inactive_sq    = torch.zeros(5, device=device, dtype=torch.float64)
        self.active_cnt     = torch.zeros(5, device=device, dtype=torch.float64)
        self.inactive_cnt   = torch.zeros(5, device=device, dtype=torch.float64)
        self.pred_active    = torch.zeros(5, device=device, dtype=torch.float64)
        self.tp             = torch.zeros(5, device=device, dtype=torch.float64)
        self.flow_loss_sum  = 0.0
        self.n_batches      = 0

    def update(
        self,
        gt: torch.Tensor,                # (B, 5, T, H, W, 3)
        pred: torch.Tensor,              # same shape
        flow_loss_value: float,
        denormalize_fn,
        contact_threshold: float,
    ) -> None:
        gt_f = gt.float()
        pred_f = pred.float()
        gt_phys = denormalize_fn(gt_f)
        pred_phys = denormalize_fn(pred_f)
        gt_mag = torch.linalg.norm(gt_phys, dim=-1)
        pred_mag = torch.linalg.norm(pred_phys, dim=-1)
        gt_active = gt_mag > contact_threshold
        pred_active = pred_mag > contact_threshold
        err_pix = (pred_f - gt_f).pow(2).mean(dim=-1)        # normalized-space MSE
        reduce_dims = [0, 2, 3, 4]                            # keep finger axis
        self.active_sq    = self.active_sq    + (err_pix * gt_active.float()).sum(dim=reduce_dims).double()
        self.inactive_sq  = self.inactive_sq  + (err_pix * (~gt_active).float()).sum(dim=reduce_dims).double()
        self.active_cnt   = self.active_cnt   + gt_active.float().sum(dim=reduce_dims).double()
        self.inactive_cnt = self.inactive_cnt + (~gt_active).float().sum(dim=reduce_dims).double()
        self.pred_active  = self.pred_active  + pred_active.float().sum(dim=reduce_dims).double()
        self.tp           = self.tp           + (pred_active & gt_active).float().sum(dim=reduce_dims).double()
        self.flow_loss_sum += flow_loss_value
        self.n_batches += 1

    def finalize(self) -> Dict:
        """Returns mean / per-finger metric dict (CPU floats)."""
        if self.n_batches == 0:
            return {}
        total_cnt = (self.active_cnt + self.inactive_cnt).clamp_min(1.0)
        flow_active_pf   = (self.active_sq / self.active_cnt.clamp_min(1.0)).cpu().tolist()
        flow_inactive_pf = (self.inactive_sq / self.inactive_cnt.clamp_min(1.0)).cpu().tolist()
        recall_pf        = (self.tp / self.active_cnt.clamp_min(1.0)).cpu().tolist()
        precision_pf     = (self.tp / self.pred_active.clamp_min(1.0)).cpu().tolist()
        contact_ratio_pf = (self.active_cnt / total_cnt).cpu().tolist()
        return {
            "flow_loss_mean":     self.flow_loss_sum / self.n_batches,
            "flow_active_pf":     flow_active_pf,
            "flow_inactive_pf":   flow_inactive_pf,
            "recall_pf":          recall_pf,
            "precision_pf":       precision_pf,
            "contact_ratio_pf":   contact_ratio_pf,
            "flow_active_mean":   sum(flow_active_pf)   / 5.0,
            "flow_inactive_mean": sum(flow_inactive_pf) / 5.0,
            "recall_mean":        sum(recall_pf)        / 5.0,
            "precision_mean":     sum(precision_pf)     / 5.0,
            "contact_ratio_mean": sum(contact_ratio_pf) / 5.0,
        }


class VisualVAEAdapterTrainer(TactileVAETrainer):
    """Stage 1 lite trainer (frozen visual VAE + tactile adapter).

    See module docstring for the architecture and loss. Inherits dataset /
    optimizer / wandb / checkpoint plumbing from :class:`TactileVAETrainer`;
    overrides:

        * ``prepare_models``           -- load frozen LTX VAE + adapter wrapper
        * ``_compute_losses``          -- drop KL; return per-stream flow + pose
        * ``validate``                 -- dual-stream v4b (post / pre) metrics
        * ``_save_flow_viz``           -- viz the post-fuse prediction
        * ``_select_best_metric_value`` -- prefer the _post stream
        * ``train``                    -- new total-loss formula + first-step grad assert
    """

    # ------------------------------------------------------------------
    # Stage 2 (in v4 trainer): model
    # ------------------------------------------------------------------

    def prepare_models(self):
        cfg = self.args.tactile_vae
        adapter_cfg: Dict = cfg["config"]

        visual_vae_path = getattr(self.args, "visual_vae_path", None)
        if not visual_vae_path:
            raise ValueError(
                "yaml: top-level `visual_vae_path` must be set for the lite "
                "trainer (path to the LTX root containing the `vae/` subfolder, "
                "OR -- when `visual_vae_path_kind: direct` -- to a directory "
                "containing config.json + diffusion_pytorch_model.safetensors)."
            )
        load_weights = bool(getattr(self.args, "visual_vae_load_weights", True))
        # `subfolder` (default): treat path as the LTX root and load the "vae"
        # subfolder via `load_latent_models` (matches ge_trainer / GE-Act).
        # `direct`: treat path as the VAE dir itself (config.json + safetensors)
        # and use the simpler `load_vae_models` helper. Useful when users
        # rsync only the VAE submodule to a custom path.
        path_kind = str(getattr(self.args, "visual_vae_path_kind", "subfolder"))

        # Pick VAE dtype to MATCH the trainer's mixed_precision setting. This
        # is the single biggest perf lever for the lite trainer: if VAE is in
        # fp32 and the rest of the model runs under bf16 autocast, every conv3d
        # in the LTX VAE incurs a fp32->bf16 cast PER OP (and bf16->fp32 on
        # backward). Loading the VAE directly in bf16 -- since it's frozen and
        # never updated by the optimizer -- avoids this entirely and matches
        # how runner.ge_trainer / runner.ge_inferencer load the same VAE.
        accelerator = self.state.accelerator
        if accelerator.mixed_precision == "bf16":
            vae_dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp16":
            vae_dtype = torch.float16
        else:
            vae_dtype = torch.float32
        # Allow yaml override (`visual_vae_dtype: fp32|bf16|fp16`) for users
        # who hit numerics issues with bf16 VAE encoding -- but the default
        # is to match accelerator's mixed_precision so things "just work".
        dtype_override = getattr(self.args, "visual_vae_dtype", None)
        if dtype_override is not None:
            _dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
            if str(dtype_override) not in _dtype_map:
                raise ValueError(
                    f"yaml: visual_vae_dtype={dtype_override!r} must be one of "
                    f"{list(_dtype_map)}."
                )
            vae_dtype = _dtype_map[str(dtype_override)]

        logger.info(
            f"Loading frozen LTX visual VAE: path={visual_vae_path} "
            f"kind={path_kind} dtype={vae_dtype} load_weights={load_weights}"
        )
        if path_kind == "subfolder":
            vae_full = load_latent_models(
                AutoencoderKLLTXVideo,
                visual_vae_path,
                vae_dtype=vae_dtype,
            )["vae"]
            if not load_weights:
                logger.warning(
                    "visual_vae_load_weights=false has no effect when "
                    "visual_vae_path_kind='subfolder' (HF from_pretrained "
                    "always loads weights). Use 'direct' kind if you need a "
                    "weights-less init for smoke testing."
                )
        elif path_kind == "direct":
            vae_full = load_vae_models(
                AutoencoderKLLTXVideo, visual_vae_path, load_weights=load_weights,
            )
            if vae_dtype != torch.float32:
                vae_full = vae_full.to(dtype=vae_dtype)
        else:
            raise ValueError(
                f"yaml: visual_vae_path_kind={path_kind!r} must be 'subfolder' "
                f"(default) or 'direct'."
            )

        # Build adapter wrapper. Defaults mirror the plan's v0 contract.
        # The adapter_n_layers / adapter_ffn_dim / adapter_dropout knobs are
        # only consumed by adapter_kind="finger_set_transformer" (v0c-A); they
        # are ignored otherwise so existing v0 / v0b yamls keep working.
        wrapper_kwargs = {
            "vae":              vae_full,
            "latent_channels":  int(adapter_cfg.get("latent_channels", 128)),
            "adapter_kind":     str(adapter_cfg.get("adapter_kind", "finger_attention")),
            "num_fingers":      int(adapter_cfg.get("num_fingers", 5)),
            "num_heads":        int(adapter_cfg.get("adapter_n_heads", 4)),
            "spatial_h":        int(adapter_cfg.get("spatial_h", 6)),
            "spatial_w":        int(adapter_cfg.get("spatial_w", 8)),
            "use_finger_embed": bool(adapter_cfg.get("adapter_use_finger_embed", True)),
            "use_pos_query":    bool(adapter_cfg.get("adapter_use_pos_query", True)),
            "adapter_residual": str(adapter_cfg.get("adapter_residual", "weighted_mean")),
            "gray_to_rgb_init": str(adapter_cfg.get("gray_to_rgb_init", "ones_repeat")),
            "latent_mode":      str(adapter_cfg.get("vae_latent_mode", "mean")),
            "enable_pre_fuse":  bool(adapter_cfg.get("enable_pre_fuse", True)),
            "adapter_n_layers": int(adapter_cfg.get("adapter_n_layers", 3)),
            "adapter_ffn_dim":  int(adapter_cfg.get("adapter_ffn_dim", 1024)),
            "adapter_dropout":  float(adapter_cfg.get("adapter_dropout", 0.0)),
            # ----- v0d kwargs (defaults reduce to v0c-A behavior) ---------
            "adapter_use_pose_injection":     bool(adapter_cfg.get("adapter_use_pose_injection", False)),
            "adapter_pose_dim":               int(adapter_cfg.get("adapter_pose_dim", 22)),
            "adapter_use_timesformer":        bool(adapter_cfg.get("adapter_use_timesformer", False)),
            "adapter_timesformer_num_blocks": int(adapter_cfg.get("adapter_timesformer_num_blocks", 2)),
            "adapter_timesformer_num_heads":  int(adapter_cfg.get("adapter_timesformer_num_heads", 8)),
            "adapter_timesformer_ffn_dim":    int(adapter_cfg.get("adapter_timesformer_ffn_dim", 512)),
            "adapter_finger_dropout":         float(adapter_cfg.get("adapter_finger_dropout", 0.0)),
        }
        self.vae: VisualVAEAdapterModel = VisualVAEAdapterModel(**wrapper_kwargs)

        # Cache v0d pose-injection flag on the trainer for downstream gating
        # (loss_pose skip, hand_pose plumbing). Reading from the YAML rather
        # than self.vae.use_pose_injection keeps the trainer decoupled from
        # the accelerator-wrap status of self.vae.
        self._use_pose_injection: bool = bool(
            adapter_cfg.get("adapter_use_pose_injection", False)
        )

        # Optional adapter-only checkpoint load (Stage 2 will use this; v0
        # probe doesn't need it but supporting it costs nothing).
        ckpt_path = cfg.get("model_path", None)
        if ckpt_path:
            logger.info(f"Loading VisualVAEAdapter checkpoint from: {ckpt_path}")
            sd = torch.load(ckpt_path, map_location="cpu")
            if "model" in sd:
                sd = sd["model"]
            missing, unexpected = self.vae.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(
                    f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            if unexpected:
                logger.warning(
                    f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
                )

        n_total = sum(p.numel() for p in self.vae.parameters())
        n_train = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        n_vae_frozen = sum(p.numel() for p in self.vae.vae.parameters())
        logger.info(f"VisualVAEAdapter total params       : {n_total/1e6:.2f} M")
        logger.info(f"VisualVAEAdapter trainable params   : {n_train/1e6:.2f} M")
        logger.info(f"VisualVAEAdapter frozen VAE params  : {n_vae_frozen/1e6:.2f} M")
        self.state.num_trainable_parameters = n_train

        # Per-submodule trainable breakdown for debuggability.
        for sub_name in (
            "gray_to_rgb", "adapter", "modality_embed",
            "aux_flow_post", "aux_pose", "aux_flow_pre",
        ):
            mod = getattr(self.vae, sub_name, None)
            if mod is None:
                continue
            n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            logger.info(f"  {sub_name:14s}: {n/1e6:7.3f} M")

        # Sanity: confirm the VAE is actually frozen end-to-end before training.
        n_vae_train = sum(p.numel() for p in self.vae.vae.parameters() if p.requires_grad)
        if n_vae_train > 0:
            raise RuntimeError(
                f"FROZEN VAE INVARIANT VIOLATED: {n_vae_train} VAE params have "
                f"requires_grad=True. This will cause WM-time drift / training "
                f"instability. Check VisualVAEAdapterModel.__init__ freezing logic."
            )

        # Sanity: confirm the VAE was loaded in the chosen dtype. If the dtype
        # silently fell back to fp32 while accelerator runs bf16 autocast, the
        # forward pass will pay a fp32->bf16 cast PER conv3d and tank throughput
        # (~250 s/step on an H200 -- this assert is the regression guard for
        # the perf fix that picks vae_dtype from accelerator.mixed_precision).
        bad_dtype: List[Tuple[str, torch.dtype]] = []
        for n, p in self.vae.vae.named_parameters():
            if p.dtype != vae_dtype:
                bad_dtype.append((n, p.dtype))
            if len(bad_dtype) >= 5:
                break
        if bad_dtype:
            raise RuntimeError(
                f"FROZEN VAE DTYPE INVARIANT VIOLATED: expected all VAE params "
                f"to be {vae_dtype} (matched to accelerator.mixed_precision="
                f"{accelerator.mixed_precision!r}); first offenders: {bad_dtype}. "
                f"Loading the frozen VAE in a dtype != accelerator.mixed_precision "
                f"causes per-op fp32<->bf16 casts in autocast and tanks throughput."
            )
        logger.info(
            f"frozen-VAE dtype invariant OK: all {n_vae_frozen/1e6:.1f}M params "
            f"are {vae_dtype} (mixed_precision={accelerator.mixed_precision!r})."
        )

        if getattr(self.args, "allow_tf32", True) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # Reuse v4 flow-stats caching for physical-unit denormalization (used
        # by both _compute_flow_loss and the v4b mask in validate()).
        flow_stats_path: Optional[str] = (
            self.cfg.get("data", {}).get("train", {}).get("flow_stats_path")
            or self.cfg.get("data", {}).get("val", {}).get("flow_stats_path")
        )
        flow_stats = _load_stats_json(flow_stats_path, expected_dim=3)
        device = self.state.accelerator.device
        if flow_stats is not None:
            self._flow_mean_t = (
                torch.from_numpy(flow_stats["mean"]).to(device).view(1, 1, 1, 1, 1, 3)
            )
            self._flow_std_t = (
                torch.from_numpy(flow_stats["std"]).clamp_min(1e-6).to(device).view(1, 1, 1, 1, 1, 3)
            )
        else:
            self._flow_mean_t = None
            self._flow_std_t = None
            logger.warning(
                "flow_stats not found via data.{train,val}.flow_stats_path; "
                "v4 magnitude-aware loss and v4b diagnostics will fall back to "
                "the normalized-space identity (still correct, just less "
                "comparable across runs with different normalization)."
            )

        # Track whether we've already run the first-step gradient-flow assert.
        self._grad_flow_asserted: bool = False

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        sample: bool = False,           # kept for base-class signature compat; ignored
    ) -> Dict[str, torch.Tensor]:
        """Run the wrapper and compute losses (post + pre + optional pose).

        ``sample`` is ignored; the wrapper's deterministic / stochastic mode
        is controlled by ``vae_latent_mode`` in the YAML (default ``mean``).

        v0d (``use_pose_injection=True``):
          - ``hand_pose`` is reshaped to ``(B, T_raw, P)`` (last-frame mode
            yields ``(B, 1, P)``, per-frame mode yields ``(B, T, P)``) and
            passed to ``self.vae`` as an input. The wrapper's
            ``_align_pose_to_lat`` resamples it to ``T_lat`` before the
            adapter consumes it.
          - ``pose_loss`` is set to a zero tensor (the model returns
            ``pose_pred=None`` because ``AuxPoseDecoder`` is dropped) so
            the total-loss accumulator skips it cleanly. The trainer's
            ``lambda_pose`` should also be 0.0 in v0d yamls (the gate is
            defensive even if it is not).
        """
        gt_flow = batch["tactile_flow"]
        gt_pose = batch["hand_pose"]

        if self._use_pose_injection:
            # Promote hand_pose to 3-D (B, T_raw, P). The dataset emits
            # (B, P) under pose_mode='last_frame' (broadcast to T_lat by
            # _align_pose_to_lat) and (B, T_raw, P) under
            # pose_mode='per_frame'. Either way the wrapper's
            # _align_pose_to_lat handles the rest.
            hand_pose_in = gt_pose
            if hand_pose_in.ndim == 2:
                hand_pose_in = hand_pose_in.unsqueeze(1)
            elif hand_pose_in.ndim != 3:
                raise RuntimeError(
                    f"hand_pose has unexpected ndim={hand_pose_in.ndim}; "
                    f"expected 2 (B, P) for last_frame mode or 3 "
                    f"(B, T_raw, P) for per_frame mode."
                )
            out = self.vae(batch["tactile"], hand_pose=hand_pose_in)
        else:
            out = self.vae(batch["tactile"])

        flow_loss_post = self._compute_flow_loss(out["flow_pred_post"], gt_flow)
        if out["flow_pred_pre"] is not None:
            flow_loss_pre = self._compute_flow_loss(out["flow_pred_pre"], gt_flow)
        else:
            flow_loss_pre = torch.zeros((), device=gt_flow.device, dtype=flow_loss_post.dtype)

        if out["pose_pred"] is None:
            # v0d Option A: AuxPoseDecoder dropped. Emit a zero scalar so the
            # train loop / validation accumulator still has a well-typed
            # tensor to log. The train loop additionally gates the addition
            # into total_loss on `out["pose_pred"] is not None`, so this
            # zero never actually flows through backward in v0d. We keep
            # it finite as a defensive belt-and-suspenders against a
            # misconfigured v0d yaml that sets lambda_pose != 0.0.
            pose_loss = torch.zeros((), device=gt_flow.device, dtype=flow_loss_post.dtype)
        else:
            # AuxPoseDecoder predicts a single per-clip pose vector of shape
            # (B, P). If the dataset is in per_frame mode (gt_pose shape
            # (B, T, P)), reduce to last-frame to match v0c-A semantics
            # before MSE -- otherwise broadcasting would silently inflate
            # the loss by replicating pose_pred T times against the wrong
            # ground truth. This guard only fires under the (cross-config)
            # combination pose_mode='per_frame' + use_pose_injection=False,
            # which the v0c-A and v0d official yamls never hit but is easy
            # to trip into via an ablation YAML.
            pose_pred = out["pose_pred"]
            if gt_pose.ndim == 3 and pose_pred.ndim == 2:
                gt_pose_aligned = gt_pose[:, -1, :]
            else:
                gt_pose_aligned = gt_pose
            pose_loss = F.mse_loss(pose_pred, gt_pose_aligned)

        return {
            "flow_loss_post": flow_loss_post,
            "flow_loss_pre":  flow_loss_pre,
            "pose_loss":      pose_loss,
            "out":            out,
        }

    # ------------------------------------------------------------------
    # Best-ckpt metric: prefer post-fuse stream
    # ------------------------------------------------------------------

    def _select_best_metric_value(
        self,
        metrics: Dict[str, float],
        metric_name: Optional[str] = None,
    ) -> Optional[float]:
        name = metric_name if metric_name is not None else self._best_ckpt_metric
        if name == "val_flow_mse_active_mean":
            # Lite trainer: the post-fuse stream is the "real" decision metric;
            # the pre-fuse stream is a diagnostic.
            actives = [v for k, v in metrics.items() if k.endswith("/flow_mse_active_post")]
            if not actives:
                # Fallback to legacy unsuffixed key for v3/v4 compat.
                actives = [v for k, v in metrics.items() if k.endswith("/flow_mse_active")]
            if not actives:
                return None
            return float(sum(actives) / len(actives))
        if name == "val_flow_mean":
            # Equivalent fallback for the lite trainer.
            flows = [v for k, v in metrics.items() if k.endswith("/flow_loss_post")]
            if not flows:
                flows = [v for k, v in metrics.items() if k.endswith("/flow_loss")]
            if not flows:
                return None
            return float(sum(flows) / len(flows))
        # `val_recall_post_mean` is keyed on `/active_recall_post` directly --
        # the lite trainer already emits that with the right suffix, so the
        # base implementation handles it. Same for any other metric not
        # specifically intercepted above.
        return super()._select_best_metric_value(metrics, metric_name)

    # ------------------------------------------------------------------
    # Validation: dual-stream v4b metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, global_step: int) -> Dict[str, float]:
        """Validate the post-fuse AND pre-fuse streams independently.

        Every metric the v4 trainer emits at ``val/T*/<key>`` is duplicated
        into two streams ``val/T*/<key>_post`` and ``val/T*/<key>_pre`` so
        the WandB dashboards can read off the encoder-vs-adapter attribution
        at a glance.
        """
        self.vae.eval()
        device = self.state.accelerator.device

        flow_cfg = self.args.tactile_vae["config"].get("flow_loss") or {}
        contact_threshold = float(flow_cfg.get("contact_threshold", 0.5))

        all_metrics: Dict[str, float] = {}
        for T, loader in self.val_dataloaders.items():
            acc_post = _V4BAccumulator(device)
            acc_pre = _V4BAccumulator(device)
            pose_loss_sum = 0.0
            n_batches = 0
            first_batch = None
            first_out = None

            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                losses = self._compute_losses(batch, sample=False)
                gt_flow = batch["tactile_flow"]
                out = losses["out"]

                acc_post.update(
                    gt=gt_flow,
                    pred=out["flow_pred_post"],
                    flow_loss_value=losses["flow_loss_post"].item(),
                    denormalize_fn=self._denormalize_flow,
                    contact_threshold=contact_threshold,
                )
                if out["flow_pred_pre"] is not None:
                    acc_pre.update(
                        gt=gt_flow,
                        pred=out["flow_pred_pre"],
                        flow_loss_value=losses["flow_loss_pre"].item(),
                        denormalize_fn=self._denormalize_flow,
                        contact_threshold=contact_threshold,
                    )
                pose_loss_sum += losses["pose_loss"].item()
                n_batches += 1
                if first_batch is None:
                    first_batch = batch
                    first_out = out

            if n_batches == 0:
                continue

            pose_loss_mean = pose_loss_sum / n_batches
            tag = f"val/T{T}"

            # Post-fuse stream (the "real" decision metric).
            r_post = acc_post.finalize()
            self._emit_v4b_metrics(tag, "post", r_post, global_step, all_metrics)

            # Pre-fuse stream (diagnostic, only if enabled).
            r_pre = acc_pre.finalize() if acc_pre.n_batches > 0 else {}
            if r_pre:
                self._emit_v4b_metrics(tag, "pre", r_pre, global_step, all_metrics)

            # Pose loss (single value -- shared across streams).
            self._log_scalars({f"{tag}/loss_pose": pose_loss_mean}, global_step)
            all_metrics[f"{tag}/pose_loss"] = pose_loss_mean

            logger.info(
                f"step {global_step}  {tag}: "
                f"flow_post={r_post['flow_loss_mean']:.4f}  "
                f"flow_pre={(r_pre['flow_loss_mean'] if r_pre else float('nan')):.4f}  "
                f"pose={pose_loss_mean:.4f}  "
                f"active_mse_post={r_post['flow_active_mean']:.4f}  "
                f"active_mse_pre={(r_pre['flow_active_mean'] if r_pre else float('nan')):.4f}  "
                f"recall_post={r_post['recall_mean']:.3f}  "
                f"recall_pre={(r_pre['recall_mean'] if r_pre else float('nan')):.3f}"
            )

            if first_batch is not None and self.save_folder is not None:
                self._save_flow_viz(first_batch, first_out, T=T, global_step=global_step)

        return all_metrics

    def _emit_v4b_metrics(
        self,
        tag: str,
        stream: str,                # "post" or "pre"
        r: Dict,
        global_step: int,
        all_metrics: Dict[str, float],
    ) -> None:
        """Push one stream's v4b metrics to TB / wandb / all_metrics dict."""
        scalars = {
            f"{tag}/loss_flow_{stream}":          r["flow_loss_mean"],
            f"{tag}/flow_mse_active_{stream}":    r["flow_active_mean"],
            f"{tag}/flow_mse_inactive_{stream}":  r["flow_inactive_mean"],
            f"{tag}/active_recall_{stream}":      r["recall_mean"],
            f"{tag}/active_precision_{stream}":   r["precision_mean"],
            f"{tag}/contact_ratio_mean_{stream}": r["contact_ratio_mean"],
        }
        self._log_scalars(scalars, global_step)
        all_metrics.update(scalars)
        for fi, name in enumerate(_FINGER_NAMES):
            per_finger = {
                f"{tag}/flow_mse_active_{name}_{stream}":    r["flow_active_pf"][fi],
                f"{tag}/flow_mse_inactive_{name}_{stream}":  r["flow_inactive_pf"][fi],
                f"{tag}/active_recall_{name}_{stream}":      r["recall_pf"][fi],
                f"{tag}/active_precision_{name}_{stream}":   r["precision_pf"][fi],
                f"{tag}/contact_ratio_{name}_{stream}":      r["contact_ratio_pf"][fi],
            }
            self._log_scalars(per_finger, global_step)
            all_metrics.update(per_finger)

    # ------------------------------------------------------------------
    # Visualization (post-fuse only -- the actual deployment latent)
    # ------------------------------------------------------------------

    def _save_flow_viz(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        T: int,
        global_step: int,
    ):
        """Save a 5x6 grid: 5 fingers x (3 GT chans + 3 post-fuse pred chans)."""
        import matplotlib.pyplot as plt
        viz_dir = os.path.join(self.save_folder, "val_viz", f"step_{global_step:08d}")
        os.makedirs(viz_dir, exist_ok=True)

        flow_gt = batch["tactile_flow"][0].detach().cpu().float().numpy()
        flow_pred = out["flow_pred_post"][0].detach().cpu().float().numpy()
        last_t = flow_gt.shape[1] - 1

        chan_names = ["dx", "dy", "div"]
        fig, axes = plt.subplots(5, 6, figsize=(14, 11))
        fig.suptitle(
            f"Val flow (post-fuse) @ step {global_step}, T={T}, frame={last_t}",
            fontsize=12,
        )
        for fi in range(5):
            for ci in range(3):
                axes[fi, ci].imshow(flow_gt[fi, last_t, :, :, ci], cmap="seismic")
                axes[fi, ci].set_title(f"GT {_FINGER_NAMES[fi]} {chan_names[ci]}", fontsize=8)
                axes[fi, ci].axis("off")
                axes[fi, 3 + ci].imshow(flow_pred[fi, last_t, :, :, ci], cmap="seismic")
                axes[fi, 3 + ci].set_title(
                    f"Pred {_FINGER_NAMES[fi]} {chan_names[ci]}", fontsize=8,
                )
                axes[fi, 3 + ci].axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(viz_dir, f"flow_T{T}.png")
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        self._log_image(f"val/T{T}/flow_viz_post", path, global_step)

    # ------------------------------------------------------------------
    # First-step gradient-flow assert
    # ------------------------------------------------------------------

    def _assert_grad_flow(self, global_step: int) -> None:
        """Mirror of the smoke test's grad-flow check, run on the very first
        train step to catch the silent ``torch.no_grad()`` bug at runtime
        rather than waiting for divergence to surface it indirectly.
        """
        accelerator = self.state.accelerator
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(self.vae)

        # GrayToRGB: must have non-None, non-zero grad.
        g_w = unwrapped.gray_to_rgb.conv.weight.grad
        if g_w is None or g_w.abs().sum().item() == 0.0:
            raise RuntimeError(
                f"GRAD-FLOW INVARIANT VIOLATED at step {global_step}: "
                f"gray_to_rgb.conv.weight.grad is "
                f"{'None' if g_w is None else 'all zero'}. The most likely cause "
                f"is that vae.encode is being wrapped in torch.no_grad() somewhere; "
                f"the VAE freeze should rely on requires_grad=False ONLY, never "
                f"torch.no_grad(), so the autograd graph stays alive for upstream "
                f"params like GrayToRGB."
            )

        # VAE: every param must have grad is None (frozen, not just zero).
        for n, p in unwrapped.vae.named_parameters():
            if p.grad is not None:
                raise RuntimeError(
                    f"FROZEN VAE PARAM RECEIVED GRAD at step {global_step}: "
                    f"vae.{n} (sum={p.grad.abs().sum().item():.3e}). "
                    f"This means VAE freeze got disabled somewhere; check "
                    f"VisualVAEAdapterModel.__init__ + that no code path "
                    f"un-freezes vae.parameters() between init and step 0."
                )

        n_train = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
        logger.info(
            f"grad-flow invariant OK at step {global_step}: "
            f"gray_to_rgb |grad|_sum={g_w.abs().sum().item():.3e}, "
            f"vae has 0 grad-bearing params, {n_train/1e6:.2f}M trainable."
        )

    # ------------------------------------------------------------------
    # Train loop (lite version: drops KL, adds lambda_loc_pre)
    # ------------------------------------------------------------------

    def train(self):
        accelerator = self.state.accelerator
        device = accelerator.device

        cfg = self.args.tactile_vae["config"]
        lambda_loc      = float(cfg.get("lambda_loc", 1.0))
        lambda_loc_pre  = float(cfg.get("lambda_loc_pre", 0.3))
        lambda_pose     = float(cfg.get("lambda_pose", 0.1))
        # The lite trainer ignores lambda_kl entirely (frozen VAE -> no posterior
        # to regularize). Warn loudly if a non-zero value sneaks in via yaml.
        lambda_kl_yaml = float(cfg.get("lambda_kl", 0.0))
        if lambda_kl_yaml != 0.0:
            logger.warning(
                f"yaml: tactile_vae.config.lambda_kl={lambda_kl_yaml} != 0; "
                f"the lite trainer ignores it (frozen VAE has no posterior). "
                f"Set lambda_kl: 0.0 in yaml to silence this."
            )

        steps_to_log = int(getattr(self.args, "steps_to_log", 50))
        steps_to_val = int(getattr(self.args, "steps_to_val", 1000))
        steps_to_save = int(getattr(self.args, "steps_to_save", 5000))
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 1.0))

        logger.info(
            f"Training (lite): train_steps={self.state.train_steps}  "
            f"steps_per_epoch={self._steps_per_epoch}  "
            f"epochs={self.state.train_epochs}  "
            f"batch_size={self.args.batch_size}  "
            f"world_size={accelerator.num_processes}  "
            f"lambda_loc={lambda_loc} lambda_loc_pre={lambda_loc_pre} "
            f"lambda_pose={lambda_pose}"
        )

        global_step = 0
        progress_bar = tqdm(
            range(self.state.train_steps),
            desc="train",
            disable=not accelerator.is_local_main_process,
        )

        # Initial validation pass to confirm pipeline + log starting point.
        if accelerator.is_main_process and self.val_dataloaders:
            init_metrics = self.validate(global_step)
            self._maybe_save_best(global_step, init_metrics)

        done = False
        for epoch in range(self.state.train_epochs):
            self._train_batch_sampler.set_epoch(epoch)
            self.vae.train()
            for step, batch in enumerate(self.train_dataloader):
                batch = _move_batch_to_device(batch, device)
                with accelerator.accumulate(self.vae):
                    losses = self._compute_losses(batch, sample=False)
                    flow_loss_post = losses["flow_loss_post"]
                    flow_loss_pre  = losses["flow_loss_pre"]
                    pose_loss      = losses["pose_loss"]

                    # v0d Option A: gate the pose term on `pose_pred is not
                    # None` so an accidentally non-zero lambda_pose in a v0d
                    # yaml can NEVER flow into total_loss when the model has
                    # no AuxPoseDecoder. In v0c-A this branch is identical
                    # to the previous lambda_pose * pose_loss formula
                    # (pose_pred is always a real tensor). Numerically the
                    # two forms agree in v0d only when lambda_pose=0.0; the
                    # gated form makes "no pose head -> no pose loss" the
                    # invariant rather than a numerical coincidence.
                    total_loss = (
                        lambda_loc     * flow_loss_post
                        + lambda_loc_pre * flow_loss_pre
                    )
                    if losses["out"]["pose_pred"] is not None:
                        total_loss = total_loss + lambda_pose * pose_loss

                    if not torch.isfinite(total_loss):
                        logger.warning(
                            f"Non-finite loss at step {global_step} ("
                            f"flow_post={flow_loss_post.item():.4f} "
                            f"flow_pre={flow_loss_pre.item():.4f} "
                            f"pose={pose_loss.item():.4f}); skipping."
                        )
                        self.optimizer.zero_grad()
                        continue

                    accelerator.backward(total_loss)

                    # First-step grad-flow assert: do this BEFORE clip/step/zero
                    # so the .grad tensors are populated and not yet zeroed.
                    if not self._grad_flow_asserted and accelerator.sync_gradients:
                        self._assert_grad_flow(global_step)
                        self._grad_flow_asserted = True

                    grad_norm = torch.tensor(0.0, device=device)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(
                            self.vae.parameters(), max_grad_norm
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                # Reduce per-step losses across ranks for logging.
                fl_post = accelerator.reduce(flow_loss_post.detach(), reduction="mean")
                fl_pre  = accelerator.reduce(flow_loss_pre.detach(), reduction="mean")
                pl      = accelerator.reduce(pose_loss.detach(), reduction="mean")
                tl      = accelerator.reduce(total_loss.detach(), reduction="mean")

                postfix = {
                    "T": int(batch["meta"]["T"][0].item()),
                    "loss": float(tl.item()),
                    "flow_post": float(fl_post.item()),
                    "flow_pre":  float(fl_pre.item()),
                    "pose": float(pl.item()),
                }
                progress_bar.set_postfix(postfix)

                if (
                    accelerator.is_main_process
                    and global_step > 0
                    and (global_step % steps_to_log == 0)
                ):
                    train_scalars = {
                        "train/loss_total":      tl.item(),
                        "train/loss_flow_post":  fl_post.item(),
                        "train/loss_flow_pre":   fl_pre.item(),
                        "train/loss_pose":       pl.item(),
                        "train/lambda_loc":      lambda_loc,
                        "train/lambda_loc_pre":  lambda_loc_pre,
                        "train/lambda_pose":     lambda_pose,
                        "train/lr":              float(self.lr_scheduler.get_last_lr()[0]),
                    }
                    if accelerator.sync_gradients:
                        train_scalars["train/grad_norm"] = float(grad_norm.item())
                    self._log_scalars(train_scalars, global_step)

                if (
                    global_step > 0
                    and global_step % steps_to_val == 0
                    and self.val_dataloaders
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        val_metrics = self.validate(global_step)
                        self._maybe_save_best(global_step, val_metrics)
                    self.vae.train()

                if (
                    global_step > 0
                    and global_step % steps_to_save == 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self.save_checkpoint(global_step)

                if global_step >= self.state.train_steps:
                    done = True
                    break

            if done:
                break

        # Final validation + checkpoint.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if self.val_dataloaders:
                final_metrics = self.validate(global_step)
                self._maybe_save_best(global_step, final_metrics)
            self.save_checkpoint(global_step, tag="final")

        progress_bar.close()
        logger.info("Training complete.")
        self._cleanup()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir.")
    args = parser.parse_args()

    trainer = VisualVAEAdapterTrainer(args.config, output_dir=args.output_dir)
    trainer.prepare_dataset()
    trainer.prepare_models()
    trainer.prepare_optimizer()
    trainer.prepare_for_training()
    trainer.train()


if __name__ == "__main__":
    main()
