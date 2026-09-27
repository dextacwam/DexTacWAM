"""Stage 1 Tactile VAE inferencer.

Single-process, eval-only counterpart to :mod:`runner.tactile_vae_trainer`.
Use this after Stage 1 training to:

* run quantitative evaluation per ``T`` regime (flow / pose / KL losses + grad-free metrics);
* render qualitative GT-vs-Pred flow grids for sanity-checking the reconstruction;
* dump ``mu`` latents over a split as a single ``.pt`` blob, ready to feed Stage 2 (TactileProjector training, distribution analysis, channel-utilization audits, etc.);
* probe channel utilization on the latent space (a key indicator of channel collapse before going to Stage 2 — see plan §10).

CLI examples::

    # Quick eval on val split using the dataset config baked into the ckpt yaml.
    python -m runner.tactile_vae_inferencer \\
        --ckpt outputs/.../checkpoints/final/model.pt \\
        --output_dir eval/run_001 --evaluate --visualize

    # Override dataset (e.g. evaluate on a different task) and dump latents.
    python -m runner.tactile_vae_inferencer \\
        --ckpt .../model.pt --output_dir eval/wipe_plate_dump \\
        --data_root data/wipe_plate --episodes 20 21 22 23 24 \\
        --dump_latents --num_dump_batches 50

Inferencer is intentionally NOT wrapped in ``Accelerator`` — single GPU is
sufficient for Stage 1 eval (the model is only ~33.5M params).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tactile_dataset import (
    FixedTBatchSampler,
    TactileDataset,
)
from models.tactile_models import TactileVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_stats_json(path: Optional[str], expected_dim: int) -> Optional[Dict[str, np.ndarray]]:
    """Load a flow/pose stats JSON; returns ``None`` if path is missing/None."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        obj = json.load(f)
    mean = np.asarray(obj["mean"], dtype=np.float32)
    std = np.asarray(obj["std"], dtype=np.float32)
    if mean.shape[0] != expected_dim or std.shape[0] != expected_dim:
        raise ValueError(
            f"Stats at {path} have dim {mean.shape[0]}/{std.shape[0]}, "
            f"expected {expected_dim}."
        )
    return {"mean": mean, "std": std}


def _kl_loss_per_sample_sum(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits_tau: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Mirror of the trainer's free-bits KL helper (kept identical for parity).

    Returns a dict ``{"loss", "raw", "floored"}``. With ``free_bits_tau == 0.0``
    behavior is bit-exact equivalent to the legacy v1 / v2 KL formulation, so
    older checkpoints continue to evaluate identically.
    """
    logvar_b = logvar.expand_as(mu)
    kl_per_elem = -0.5 * (1.0 + logvar_b - mu.pow(2) - logvar_b.exp())
    raw = kl_per_elem.flatten(1).sum(dim=-1).mean()
    if free_bits_tau <= 0.0:
        return {"loss": raw, "raw": raw, "floored": raw}
    kl_avg = kl_per_elem.mean(dim=(0, 3, 4, 5))                    # (F, C)
    kl_floored_avg = torch.clamp(kl_avg, min=float(free_bits_tau))
    group_size = mu.shape[3] * mu.shape[4] * mu.shape[5]            # T_lat * h * w
    floored = kl_floored_avg.sum() * group_size
    return {"loss": floored, "raw": raw, "floored": floored}


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------


class TactileVAEInferencer:
    """Eval-only wrapper around a trained :class:`TactileVAE`.

    Attributes:
        cfg: full yaml config dict (loaded from the checkpoint).
        model_cfg: just the ``tactile_vae.config`` sub-tree.
        vae: the underlying model on ``device`` in ``dtype``.
        device, dtype: torch device + dtype the model lives on.
        flow_stats, pose_stats: optional ``{"mean", "std"}`` arrays for
            denormalizing flow / pose predictions back to physical units.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        flow_stats_path: Optional[str] = None,
        pose_stats_path: Optional[str] = None,
    ):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" not in ckpt or "config" not in ckpt:
            raise ValueError(
                f"Checkpoint at {ckpt_path} is missing 'model' or 'config' "
                f"keys; was it produced by tactile_vae_trainer?"
            )
        self.cfg: Dict[str, Any] = ckpt["config"]
        self.model_cfg: Dict[str, Any] = self.cfg["tactile_vae"]["config"]
        self.global_step: int = int(ckpt.get("global_step", -1))

        # Strip loss-weight / loss-term keys before passing to TactileVAE __init__
        # (mirror of trainer logic — keeps yaml schema consistent).
        # ``free_bits_tau`` is a v3 loss-term hparam, not a model arch arg.
        init_cfg = {
            k: v for k, v in self.model_cfg.items()
            if k not in ("lambda_loc", "lambda_pose", "lambda_kl", "free_bits_tau")
        }
        self.vae = TactileVAE(**init_cfg)
        missing, unexpected = self.vae.load_state_dict(ckpt["model"], strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict mismatch: missing={missing[:3]}.., unexpected={unexpected[:3]}.."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.vae.to(device=self.device, dtype=self.dtype).eval()

        # Load stats. Prefer explicit args; fall back to dataset config for convenience.
        if flow_stats_path is None:
            flow_stats_path = self.cfg.get("data", {}).get("val", {}).get("flow_stats_path", None)
        if pose_stats_path is None:
            pose_stats_path = self.cfg.get("data", {}).get("val", {}).get("pose_stats_path", None)
        self.flow_stats = _load_stats_json(flow_stats_path, expected_dim=3)
        self.pose_stats = _load_stats_json(pose_stats_path, expected_dim=22)

        # Cache stats as torch tensors on device for fast denormalization.
        if self.flow_stats is not None:
            # Shape (1, 1, 1, 1, 1, 3) so it broadcasts to (B, F, T, H, W, 3).
            self._flow_mean_t = torch.from_numpy(self.flow_stats["mean"]).to(self.device).view(1, 1, 1, 1, 1, 3)
            self._flow_std_t = torch.from_numpy(self.flow_stats["std"]).clamp_min(1e-6).to(self.device).view(1, 1, 1, 1, 1, 3)
        else:
            self._flow_mean_t = self._flow_std_t = None
        if self.pose_stats is not None:
            self._pose_mean_t = torch.from_numpy(self.pose_stats["mean"]).to(self.device).view(1, 22)
            self._pose_std_t = torch.from_numpy(self.pose_stats["std"]).clamp_min(1e-6).to(self.device).view(1, 22)
        else:
            self._pose_mean_t = self._pose_std_t = None

        n_params = sum(p.numel() for p in self.vae.parameters())
        print(
            f"[TactileVAEInferencer] loaded ckpt @ step {self.global_step} "
            f"({n_params/1e6:.2f}M params, dtype={dtype}, device={self.device})"
        )
        if self.flow_stats is None:
            print("[TactileVAEInferencer] WARN: no flow stats — predictions stay in normalized space.")
        if self.pose_stats is None:
            print("[TactileVAEInferencer] WARN: no pose stats — predictions stay in normalized space.")

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, **kwargs) -> "TactileVAEInferencer":
        return cls(ckpt_path, **kwargs)

    # ------------------------------------------------------------------
    # Core encode / decode / reconstruct
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self, tactile: torch.Tensor, hand_pose: torch.Tensor, sample: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Encode a tactile clip into latent ``mu`` (and optionally a sampled ``z``).

        Args:
            tactile: ``(B, 5, 1, T, 192, 256)`` in ``[-1, 1]``.
            hand_pose: ``(B, 22)``, normalized.
            sample: if ``True``, also return ``z = mu + eps * std``.

        Returns:
            dict with ``mu``, ``logvar``, optionally ``z``.
        """
        tactile = tactile.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hand_pose = hand_pose.to(device=self.device, dtype=self.dtype, non_blocking=True)
        mu, logvar = self.vae.encode(tactile, hand_pose)
        out = {"mu": mu, "logvar": logvar}
        if sample:
            std = (0.5 * logvar).exp().expand_as(mu)
            out["z"] = mu + std * torch.randn_like(mu)
        return out

    @torch.no_grad()
    def decode(
        self, z: torch.Tensor, denormalize: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Decode latent ``z`` to ``flow_pred`` (B,5,T,24,32,3) and ``pose_pred`` (B,22).

        If ``denormalize=True`` and stats are loaded, predictions are mapped back
        from normalized space to raw physical units.
        """
        z = z.to(device=self.device, dtype=self.dtype, non_blocking=True)
        flow_pred, pose_pred = self.vae.decode(z)
        if denormalize:
            flow_pred = self.denormalize_flow(flow_pred)
            pose_pred = self.denormalize_pose(pose_pred)
        return {"flow_pred": flow_pred, "pose_pred": pose_pred}

    @torch.no_grad()
    def reconstruct(
        self,
        batch: Dict[str, torch.Tensor],
        sample: bool = False,
        denormalize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """End-to-end encode → (reparam or mu) → decode."""
        batch = _move_batch_to_device(batch, self.device)
        enc = self.encode(batch["tactile"], batch["hand_pose"], sample=sample)
        z = enc.get("z", enc["mu"])
        dec = self.decode(z, denormalize=denormalize)
        return {**enc, **dec}

    # ------------------------------------------------------------------
    # Denormalization helpers
    # ------------------------------------------------------------------

    def denormalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        if self._flow_mean_t is None:
            return flow
        return flow * self._flow_std_t.to(flow.dtype) + self._flow_mean_t.to(flow.dtype)

    def denormalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        if self._pose_mean_t is None:
            return pose
        return pose * self._pose_std_t.to(pose.dtype) + self._pose_mean_t.to(pose.dtype)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        sample: bool = False,
        compute_channel_stats: bool = True,
        contact_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Aggregate flow / pose / KL losses + (optionally) latent channel stats.

        Returns a dict like::

            {
                "n_batches": int, "n_samples": int,
                "flow_loss_mean": float, "pose_loss_mean": float, "kl_loss_mean": float,
                # v3 free-bits diagnostics: ``kl_raw_mean`` is the unfloored KL,
                # ``kl_floored_mean`` reflects the value after free-bits flooring
                # (== kl_raw_mean when ``free_bits_tau == 0``). ``free_bits_tau``
                # echoes the value read from the ckpt config.
                "kl_raw_mean": float, "kl_floored_mean": float, "free_bits_tau": float,
                "flow_loss_per_finger": [5 floats],
                # v4b contact-aware diagnostics: per-finger metrics that
                # disentangle "the model never saw a contact event" from
                # "the model failed on contact events" -- crucial because
                # ring/pinky have ~5-9% contact rates so full-frame MSE is
                # dominated by trivial near-zero predictions on non-contact
                # frames. ``contact_aware.threshold`` is in **physical** flow
                # units (px/frame magnitude; defaults to 0.5 unless
                # overridden). The mask is built by denormalizing both GT and
                # pred flow with ``flow_stats`` before thresholding so the
                # number is comparable across runs trained with different
                # normalization stats; the squared-error itself is still
                # computed in normalized space (matches training objective).
                "contact_aware": {
                    "threshold": float,
                    "contact_ratio_per_finger":     [5 floats],
                    "flow_mse_full_per_finger":     [5 floats],   # alias of flow_loss_per_finger
                    "flow_mse_active_per_finger":   [5 floats],
                    "flow_mse_inactive_per_finger": [5 floats],
                    "active_recall_per_finger":     [5 floats],
                    "active_precision_per_finger":  [5 floats],
                },
                # if compute_channel_stats:
                "channel_std": float[5, 128],          # std of mu over collected samples
                "n_collapsed_channels": int,           # std < 0.05 threshold
                "collapse_threshold": 0.05,
            }
        """
        flow_sum = pose_sum = kl_sum = kl_raw_sum = kl_floored_sum = 0.0
        flow_per_finger_sum = torch.zeros(5, device=self.device)
        n_batches = 0
        n_samples = 0
        mu_chunks: List[torch.Tensor] = []  # store mean over (T_lat, h, w) → (B, 5, 128)

        # v4b contact-aware accumulators (per finger). Use float64 to avoid
        # precision loss when summing across many batches and many pixels.
        active_sq_sum     = torch.zeros(5, device=self.device, dtype=torch.float64)
        inactive_sq_sum   = torch.zeros(5, device=self.device, dtype=torch.float64)
        active_count      = torch.zeros(5, device=self.device, dtype=torch.float64)
        inactive_count    = torch.zeros(5, device=self.device, dtype=torch.float64)
        pred_active_count = torch.zeros(5, device=self.device, dtype=torch.float64)
        tp_count          = torch.zeros(5, device=self.device, dtype=torch.float64)

        # v3: report KL with the same free-bits floor that was used at training
        # time, plus the raw (unfloored) KL for diagnostic purposes.
        free_bits_tau = float(self.model_cfg.get("free_bits_tau", 0.0))

        for bi, batch in enumerate(tqdm(dataloader, desc="evaluate", leave=False)):
            if max_batches is not None and bi >= max_batches:
                break
            batch = _move_batch_to_device(batch, self.device)
            out = self.vae(batch["tactile"], batch["hand_pose"], sample=sample)
            flow_loss = F.mse_loss(out["flow_pred"], batch["tactile_flow"])
            pose_loss = F.mse_loss(out["pose_pred"], batch["hand_pose"])
            kl = _kl_loss_per_sample_sum(
                out["mu"], out["logvar"], free_bits_tau=free_bits_tau
            )

            # Per-finger flow MSE: reduce all dims except finger axis.
            # Flow tensors are (B, 5, T, 24, 32, 3) with the last axis = (dx, dy, divergence).
            ff = (out["flow_pred"] - batch["tactile_flow"]).pow(2)
            flow_per_finger_sum = flow_per_finger_sum + ff.mean(dim=[0, 2, 3, 4, 5])

            # v4b contact-aware accumulation (per finger).
            # Threshold is applied in PHYSICAL flow units so it is comparable
            # across runs trained with different normalization stats. The
            # squared error itself stays in normalized space to match the
            # training objective.
            gt_flow_f   = batch["tactile_flow"].float()
            pred_flow_f = out["flow_pred"].float()
            gt_flow_phys   = self.denormalize_flow(gt_flow_f)
            pred_flow_phys = self.denormalize_flow(pred_flow_f)
            gt_mag   = torch.linalg.norm(gt_flow_phys,   dim=-1)           # (B, 5, T, H, W)
            pred_mag = torch.linalg.norm(pred_flow_phys, dim=-1)           # (B, 5, T, H, W)
            gt_active   = gt_mag   > contact_threshold
            pred_active = pred_mag > contact_threshold
            ff_pix = (pred_flow_f - gt_flow_f).pow(2).mean(dim=-1)         # (B, 5, T, H, W)

            reduce_dims = [0, 2, 3, 4]
            active_sq_sum     = active_sq_sum     + (ff_pix * gt_active.float()).sum(dim=reduce_dims).double()
            inactive_sq_sum   = inactive_sq_sum   + (ff_pix * (~gt_active).float()).sum(dim=reduce_dims).double()
            active_count      = active_count      + gt_active.float().sum(dim=reduce_dims).double()
            inactive_count    = inactive_count    + (~gt_active).float().sum(dim=reduce_dims).double()
            pred_active_count = pred_active_count + pred_active.float().sum(dim=reduce_dims).double()
            tp_count          = tp_count          + (pred_active & gt_active).float().sum(dim=reduce_dims).double()

            flow_sum += flow_loss.item()
            pose_sum += pose_loss.item()
            kl_sum += kl["loss"].item()
            kl_raw_sum += kl["raw"].item()
            kl_floored_sum += kl["floored"].item()
            n_batches += 1
            n_samples += batch["tactile"].shape[0]

            if compute_channel_stats:
                # mu: (B, 5, 128, T_lat, h, w) → mean over (T_lat, h, w) → (B, 5, 128)
                mu_chunks.append(out["mu"].float().mean(dim=[3, 4, 5]).cpu())

        if n_batches == 0:
            return {"n_batches": 0, "n_samples": 0}

        # v4b contact-aware metrics: divide accumulated sums by counts. clamp_min
        # avoids div-by-zero on splits where a finger never goes active.
        total_count_per_finger = (active_count + inactive_count).clamp_min(1.0)
        flow_loss_per_finger_list = (flow_per_finger_sum / n_batches).cpu().tolist()
        contact_aware = {
            "threshold": contact_threshold,
            "contact_ratio_per_finger":     (active_count / total_count_per_finger).cpu().tolist(),
            "flow_mse_full_per_finger":     flow_loss_per_finger_list,
            "flow_mse_active_per_finger":   (active_sq_sum   / active_count.clamp_min(1.0)).cpu().tolist(),
            "flow_mse_inactive_per_finger": (inactive_sq_sum / inactive_count.clamp_min(1.0)).cpu().tolist(),
            "active_recall_per_finger":     (tp_count / active_count.clamp_min(1.0)).cpu().tolist(),
            "active_precision_per_finger":  (tp_count / pred_active_count.clamp_min(1.0)).cpu().tolist(),
        }

        result = {
            "n_batches": n_batches,
            "n_samples": n_samples,
            "flow_loss_mean": flow_sum / n_batches,
            "pose_loss_mean": pose_sum / n_batches,
            "kl_loss_mean": kl_sum / n_batches,
            "kl_raw_mean": kl_raw_sum / n_batches,
            "kl_floored_mean": kl_floored_sum / n_batches,
            "free_bits_tau": free_bits_tau,
            "flow_loss_per_finger": flow_loss_per_finger_list,
            "contact_aware": contact_aware,
        }

        if compute_channel_stats and mu_chunks:
            mu_all = torch.cat(mu_chunks, dim=0)            # (N, 5, 128)
            channel_std = mu_all.std(dim=0)                  # (5, 128)
            collapse_threshold = 0.05
            n_collapsed = int((channel_std < collapse_threshold).sum().item())
            result.update({
                "channel_std": channel_std.tolist(),
                "n_collapsed_channels": n_collapsed,
                "collapse_threshold": collapse_threshold,
                "channel_std_summary": {
                    "min": float(channel_std.min().item()),
                    "max": float(channel_std.max().item()),
                    "mean": float(channel_std.mean().item()),
                    "median": float(channel_std.median().item()),
                },
            })
        return result

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualize(
        self,
        batch: Dict[str, torch.Tensor],
        save_path: str,
        sample_idx: int = 0,
        title_extra: str = "",
    ):
        """Save a 5×6 grid (5 fingers × {GT dx, GT dy, GT div, Pred dx, Pred dy, Pred div})
        for the last frame of one batch sample. Mirror of trainer's viz so they
        are visually comparable.
        """
        batch = _move_batch_to_device(batch, self.device)
        out = self.vae(batch["tactile"], batch["hand_pose"], sample=False)

        flow_gt = batch["tactile_flow"][sample_idx].cpu().float().numpy()
        flow_pred = out["flow_pred"][sample_idx].cpu().float().numpy()
        T_meta = int(batch["meta"]["T"][sample_idx])
        last_t = flow_gt.shape[1] - 1

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        chan_names = ["dx", "dy", "div"]
        fig, axes = plt.subplots(5, 6, figsize=(14, 11))
        title = f"Reconstruction T={T_meta} frame={last_t}"
        if title_extra:
            title = f"{title} | {title_extra}"
        fig.suptitle(title, fontsize=12)

        for fi in range(5):
            for ci in range(3):
                axes[fi, ci].imshow(flow_gt[fi, last_t, :, :, ci], cmap="seismic")
                axes[fi, ci].set_title(f"GT {finger_names[fi]} {chan_names[ci]}", fontsize=8)
                axes[fi, ci].axis("off")
                axes[fi, 3 + ci].imshow(flow_pred[fi, last_t, :, :, ci], cmap="seismic")
                axes[fi, 3 + ci].set_title(f"Pred {finger_names[fi]} {chan_names[ci]}", fontsize=8)
                axes[fi, 3 + ci].axis("off")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_path, dpi=80, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def plot_channel_utilization(
        self,
        channel_std: List[List[float]],
        save_path: str,
        threshold: float = 0.05,
    ):
        """Bar plot of latent-channel std per finger (5 subplots × 128 bars each).

        Channels with std < ``threshold`` are highlighted red (collapse candidates).
        """
        std_arr = np.asarray(channel_std)  # (5, 128)
        n_fingers, n_channels = std_arr.shape
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]

        fig, axes = plt.subplots(n_fingers, 1, figsize=(12, 2.0 * n_fingers), sharex=True)
        if n_fingers == 1:
            axes = [axes]
        for fi in range(n_fingers):
            ax = axes[fi]
            stds = std_arr[fi]
            colors = ["#d62728" if s < threshold else "#1f77b4" for s in stds]
            ax.bar(np.arange(n_channels), stds, color=colors, width=1.0)
            ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8)
            ax.set_ylabel(f"{finger_names[fi]}\nstd")
            n_collapsed = int((stds < threshold).sum())
            ax.set_title(
                f"finger {fi} ({finger_names[fi]}): "
                f"{n_collapsed}/{n_channels} channels under threshold {threshold}",
                fontsize=9,
            )
        axes[-1].set_xlabel("latent channel index")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=80, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Latent dump (for Stage 2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dump_latents(
        self,
        dataloader: DataLoader,
        save_path: str,
        max_batches: Optional[int] = None,
        store_logvar: bool = False,
    ) -> Dict[str, Any]:
        """Encode an entire dataloader and save the ``mu`` (and optional ``logvar``)
        latents alongside their metadata to a single ``.pt`` file.

        Output dict layout::

            {
                "mu":          float32 tensor (N, 5, 128, T_lat_max, h, w)  — padded if T mixes
                "T_lat":       int32 tensor (N,)   — actual T_lat per sample (since T_lat varies)
                "logvar":      float32 tensor (N, 5, 1, T_lat_max, h, w)    — if store_logvar
                "episode_index": int32 (N,)
                "hand":          int32 (N,)
                "start_frame":   int32 (N,)
                "T":             int32 (N,)
                "model_step":    int (the trained step recovered from the ckpt)
            }
        """
        # Group batches by T to keep tensors dense — Stage 2 typically wants
        # one tensor per T regime.
        per_T: Dict[int, Dict[str, list]] = {}

        for bi, batch in enumerate(tqdm(dataloader, desc="dump_latents", leave=False)):
            if max_batches is not None and bi >= max_batches:
                break
            batch = _move_batch_to_device(batch, self.device)
            mu, logvar = self.vae.encode(batch["tactile"], batch["hand_pose"])
            T = int(batch["meta"]["T"][0].item())
            slot = per_T.setdefault(T, {"mu": [], "logvar": [], "ep": [], "hand": [], "frame": []})
            slot["mu"].append(mu.float().cpu())
            if store_logvar:
                slot["logvar"].append(logvar.float().cpu())
            slot["ep"].append(batch["meta"]["episode_index"].cpu())
            slot["hand"].append(batch["meta"]["hand"].cpu())
            slot["frame"].append(batch["meta"]["start_frame"].cpu())

        out: Dict[str, Any] = {"model_step": self.global_step, "by_T": {}}
        total_n = 0
        for T, slot in per_T.items():
            mu_all = torch.cat(slot["mu"], dim=0)
            entry = {
                "mu": mu_all,
                "episode_index": torch.cat(slot["ep"], dim=0),
                "hand": torch.cat(slot["hand"], dim=0),
                "start_frame": torch.cat(slot["frame"], dim=0),
                "T": int(T),
                "T_lat": int(mu_all.shape[3]),
                "n_samples": int(mu_all.shape[0]),
            }
            if store_logvar:
                entry["logvar"] = torch.cat(slot["logvar"], dim=0)
            out["by_T"][int(T)] = entry
            total_n += entry["n_samples"]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(out, save_path)
        print(
            f"[dump_latents] saved {total_n} samples across T={list(out['by_T'].keys())} "
            f"to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)"
        )
        return {"path": save_path, "n_samples": total_n, "T_keys": list(out["by_T"].keys())}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def make_val_dataset(
        self,
        data_root: Optional[str] = None,
        episodes: Optional[Sequence[int]] = None,
        T_choices: Optional[Sequence[int]] = None,
    ) -> TactileDataset:
        """Reconstruct the val dataset using the ckpt's yaml config, with optional overrides.

        If the ckpt yaml uses multi-root ``datasets:`` form (v2+) AND the caller
        overrides with single-root ``data_root``/``episodes``, the ``datasets``
        key is dropped so the dataset sees exactly one form. This is the normal
        eval pattern: "load ckpt trained on combined data, but evaluate on one
        specific task/split at a time".
        """
        data_cfg = dict(self.cfg.get("data", {}).get("val", {}))
        if not data_cfg:
            raise ValueError("Checkpoint config has no data.val section; pass dataset args explicitly.")
        # Single-root override: strip any inherited multi-root key so the two
        # forms don't clash in TactileDataset.__init__.
        if data_root is not None or episodes is not None:
            data_cfg.pop("datasets", None)
        if data_root is not None:
            data_cfg["data_root"] = data_root
        if episodes is not None:
            data_cfg["episodes"] = list(episodes)
        if T_choices is not None:
            data_cfg["T_choices"] = list(T_choices)
        return TactileDataset(**data_cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_dataloader(
    dataset: TactileDataset, T: int, batch_size: int, num_batches: Optional[int]
) -> DataLoader:
    sampler = FixedTBatchSampler(
        dataset=dataset,
        T=T,
        batch_size=batch_size,
        num_batches=num_batches,
        seed=0,
        shuffle=True,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model.pt produced by tactile_vae_trainer.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])

    # Dataset overrides (otherwise inherit from ckpt yaml).
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--T_choices", type=int, nargs="+", default=None)
    parser.add_argument("--flow_stats_path", type=str, default=None)
    parser.add_argument("--pose_stats_path", type=str, default=None)

    # Modes (set at least one).
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--dump_latents", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_eval_batches", type=int, default=20, help="Per-T batches for --evaluate.")
    parser.add_argument("--num_dump_batches", type=int, default=None, help="Per-T batches for --dump_latents (None = full).")
    parser.add_argument("--num_viz_batches", type=int, default=1, help="Per-T batches to visualize.")
    parser.add_argument("--store_logvar", action="store_true", help="Also dump logvar in --dump_latents.")
    parser.add_argument("--no_channel_stats", action="store_true", help="Skip channel utilization in --evaluate.")
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=0.5,
        help=(
            "v4b contact-aware metrics: threshold on ||flow||_2 in PHYSICAL "
            "flow units (px/frame) to define an active/contact pixel. GT and "
            "pred flow are denormalized with --flow_stats_path before "
            "thresholding so the value is comparable across runs trained "
            "with different normalization stats. Default 0.5."
        ),
    )

    args = parser.parse_args()

    if not (args.evaluate or args.visualize or args.dump_latents):
        parser.error("specify at least one of --evaluate, --visualize, --dump_latents")

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    inf = TactileVAEInferencer(
        ckpt_path=args.ckpt,
        device=args.device,
        dtype=dtype,
        flow_stats_path=args.flow_stats_path,
        pose_stats_path=args.pose_stats_path,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inferencer] outputs → {output_dir}")

    # Build val dataset.
    dataset = inf.make_val_dataset(
        data_root=args.data_root,
        episodes=args.episodes,
        T_choices=args.T_choices,
    )
    T_choices = tuple(args.T_choices) if args.T_choices else dataset.T_choices
    print(f"[inferencer] val dataset: {len(dataset)} samples, T_choices={T_choices}")

    overall: Dict[str, Any] = {
        "ckpt": args.ckpt,
        "model_step": inf.global_step,
        "T_choices": list(T_choices),
        "by_T": {},
    }

    for T in T_choices:
        if T not in dataset.indices_by_T or not dataset.indices_by_T[T]:
            print(f"[inferencer] T={T} has no samples in dataset, skipping.")
            continue
        per_T: Dict[str, Any] = {}

        if args.evaluate:
            loader = _build_dataloader(dataset, T, args.batch_size, args.num_eval_batches)
            metrics = inf.evaluate(
                loader,
                max_batches=args.num_eval_batches,
                compute_channel_stats=not args.no_channel_stats,
                contact_threshold=args.contact_threshold,
            )
            per_T["evaluate"] = {k: v for k, v in metrics.items() if k != "channel_std"}
            ca = metrics.get("contact_aware", {})
            ca_active = ca.get("flow_mse_active_per_finger") or []
            ca_inactive = ca.get("flow_mse_inactive_per_finger") or []
            ca_recall = ca.get("active_recall_per_finger") or []
            ca_precision = ca.get("active_precision_per_finger") or []

            def _avg(xs):
                return sum(xs) / max(len(xs), 1) if xs else float("nan")

            print(
                f"[inferencer] T={T}  evaluate: "
                f"flow={metrics['flow_loss_mean']:.4f}  "
                f"pose={metrics['pose_loss_mean']:.4f}  "
                f"kl={metrics['kl_loss_mean']:.2f}  "
                f"kl_raw={metrics['kl_raw_mean']:.2f}  "
                f"kl_floored={metrics['kl_floored_mean']:.2f}  "
                f"tau={metrics['free_bits_tau']}  "
                f"collapsed_ch={metrics.get('n_collapsed_channels', 'n/a')}"
            )
            print(
                f"[inferencer] T={T}  contact_aware (tau_c={args.contact_threshold:g}): "
                f"flow_active={_avg(ca_active):.4f}  "
                f"flow_inactive={_avg(ca_inactive):.4f}  "
                f"recall={_avg(ca_recall):.3f}  "
                f"precision={_avg(ca_precision):.3f}"
            )
            if "channel_std" in metrics and not args.no_channel_stats:
                inf.plot_channel_utilization(
                    metrics["channel_std"],
                    save_path=str(output_dir / f"channel_util_T{T}.png"),
                    threshold=metrics["collapse_threshold"],
                )

        if args.visualize:
            loader = _build_dataloader(dataset, T, args.batch_size, args.num_viz_batches)
            for vi, batch in enumerate(loader):
                if vi >= args.num_viz_batches:
                    break
                inf.visualize(
                    batch,
                    save_path=str(output_dir / f"recon_T{T}_batch{vi:02d}.png"),
                    sample_idx=0,
                    title_extra=f"step={inf.global_step}",
                )
            per_T["visualize"] = {"num_batches": args.num_viz_batches}
            print(f"[inferencer] T={T}  visualize: saved {args.num_viz_batches} grid(s).")

        if args.dump_latents:
            loader = _build_dataloader(dataset, T, args.batch_size, args.num_dump_batches)
            dump_info = inf.dump_latents(
                loader,
                save_path=str(output_dir / f"latents_T{T}.pt"),
                max_batches=args.num_dump_batches,
                store_logvar=args.store_logvar,
            )
            per_T["dump_latents"] = dump_info

        overall["by_T"][int(T)] = per_T

    # Persist overall JSON summary.
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, sort_keys=False, default=str)
    print(f"[inferencer] summary → {summary_path}")


if __name__ == "__main__":
    main()
