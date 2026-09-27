"""Stage 1 Tactile VAE trainer.

Standalone trainer for :class:`models.tactile_models.TactileVAE`. Modeled on
:mod:`runner.ge_trainer` but stripped down to one trainable model + three loss
terms (no DiT, no text encoder, no diffusion scheduler).

Loss (plan §4)::

    total = lambda_loc  * L_rec  + lambda_pose * L_pose + lambda_kl * L_KL
          = lambda_loc  * MSE(flow_pred, flow_gt)
          + lambda_pose * MSE(pose_pred, pose_gt)
          + lambda_kl   * KL(N(mu, sigma) || N(0, I))

Mixed-T schedule (plan §7) is enforced by :class:`MixedTBatchSampler`: each
training batch contains samples with one homogeneous ``T`` (1 or 9), drawn
according to ``T_sample_ratio``. Validation runs ``T=1`` and ``T=9`` separately
so each regime is reported independently.

Single-process or single-node multi-GPU (DDP) supported via ``Accelerate``.
For DDP the per-rank seed of the sampler is offset by rank so each rank sees
a different stream of batches.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yaml import Loader, load

from data.tactile_dataset import (
    FixedTBatchSampler,
    MixedTBatchSampler,
    TactileDataset,
)
from models.tactile_models import TactileVAE, kl_divergence
from utils import import_custom_class, init_logging
from utils.optimizer_utils import get_optimizer

LOG_LEVEL = "INFO"
logger = get_logger("tactile_vae_trainer")
logger.setLevel(LOG_LEVEL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = "/") -> Dict:
    """Flatten a nested dict for JSON dumping (used in config snapshot)."""
    out = {}
    for k, v in d.items():
        full = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full, sep=sep))
        else:
            out[full] = v if not isinstance(v, Path) else str(v)
    return out


def _load_stats_json(path: Optional[str], expected_dim: int) -> Optional[Dict[str, np.ndarray]]:
    """Load a flow / pose normalization stats JSON. Returns ``None`` if path is missing.

    Mirrors the helper in :mod:`runner.tactile_vae_inferencer` so the trainer
    can build the same physical-unit denormalization mapping at training time
    that v4b inference uses for the contact mask. Kept private + duplicated to
    avoid any cross-module dependency between trainer and inferencer.
    """
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


def _avg_pool_T(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Centered moving average over the T axis (dim=2) of a ``(B, F, T, H, W, 1)``
    tensor. Returns the same shape as input. ``count_include_pad=False`` so
    boundary frames are not artificially diluted by zero pads.

    Used by the v4 ``temporal_smooth_window`` hook on the GT-flow magnitude
    when single-frame magnitude outliers are suspected to inflate the
    sqrt-smoothed weight. Defaulted off via ``window == 1`` short-circuit.
    """
    if kernel <= 1:
        return x
    B, NF, T, H, W, C = x.shape
    if T < kernel:
        # Not enough timesteps to apply the window cleanly; skip silently
        # (e.g. T=1 batches will always hit this when window > 1).
        return x
    pad = kernel // 2
    x_flat = x.reshape(B * NF, T, H * W * C).transpose(1, 2)            # (B*F, H*W*C, T)
    x_smooth = F.avg_pool1d(
        x_flat, kernel_size=kernel, stride=1, padding=pad, count_include_pad=False,
    )
    # avg_pool1d with stride=1 + padding=k//2 yields T_out = T (k odd) or
    # T+1 (k even); trim back to the original T.
    x_smooth = x_smooth[..., :T]
    return x_smooth.transpose(1, 2).contiguous().view(B, NF, T, H, W, C)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tactile/flow/pose tensors to device.

    Tensors are kept in their native dtype (fp32). Accelerator's autocast
    handles bf16/fp16 conversion inside the model forward — manually casting
    targets here causes mixed-dtype backward errors with bf16 mixed precision.
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _kl_loss_per_sample_sum(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits_tau: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Plan §4 KL with optional free-bits flooring (v3 ablation).

    ``logvar`` (B, F, 1, T_lat, h, w) broadcasts to ``mu`` (B, F, 128, T_lat, h, w).

    Returns a dict::

        {
            "loss":    scalar tensor used for backprop (raw if tau==0, else floored),
            "raw":     unfloored KL = per-sample sum, batch-mean (always reported),
            "floored": floored KL    (== raw when tau == 0.0).
        }

    Free-bits semantics (when ``free_bits_tau > 0``): per-(finger, channel) KL is
    averaged over the batch dim and over (T_lat, h, w) spatial-temporal positions
    to obtain a (F, C) tensor of per-element nats. Each (F, C) entry is then
    floored at ``tau`` before being re-summed (and rescaled by group_size = T_lat
    * h * w) to match the original "per-sample sum, batch-mean" magnitude. With
    ``tau == 0`` this is bit-exact equivalent to the v1 / v2 formulation.
    """
    logvar_b = logvar.expand_as(mu)
    kl_per_elem = -0.5 * (1.0 + logvar_b - mu.pow(2) - logvar_b.exp())  # (B,F,C,T_lat,h,w)
    raw = kl_per_elem.flatten(1).sum(dim=-1).mean()
    if free_bits_tau <= 0.0:
        return {"loss": raw, "raw": raw, "floored": raw}

    # Per-(F, C) average over batch + spatial-temporal positions -> (F, C)
    kl_avg = kl_per_elem.mean(dim=(0, 3, 4, 5))
    kl_floored_avg = torch.clamp(kl_avg, min=float(free_bits_tau))
    group_size = mu.shape[3] * mu.shape[4] * mu.shape[5]   # T_lat * h * w
    floored = kl_floored_avg.sum() * group_size
    return {"loss": floored, "raw": raw, "floored": floored}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class _State:
    seed: Optional[int] = None
    accelerator: Optional[Accelerator] = None
    weight_dtype: torch.dtype = torch.float32
    train_steps: Optional[int] = None
    train_epochs: Optional[int] = None
    learning_rate: float = 0.0
    num_trainable_parameters: int = 0
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TactileVAETrainer:
    """Stage 1 Tactile VAE trainer.

    Public entry points (called in order by the launcher script):

    1. ``__init__(config_file, output_dir=None)`` — parse yaml, init accelerator.
    2. ``prepare_dataset()`` — build train + val datasets and samplers.
    3. ``prepare_models()`` — instantiate ``TactileVAE`` (optionally load ckpt).
    4. ``prepare_optimizer()`` — build optimizer + LR scheduler.
    5. ``prepare_for_training()`` — ``accelerator.prepare(model, optim, lr_sched)``.
    6. ``train()`` — main loop with periodic validation + checkpointing.
    """

    def __init__(self, config_file: str, output_dir: Optional[str] = None):
        with open(config_file) as f:
            cfg = load(f, Loader=Loader)
        self.cfg = cfg
        self.args = argparse.Namespace(**cfg)

        if output_dir is not None:
            self.args.output_dir = output_dir

        self.state = _State()

        # Best-checkpoint tracking. Lower is better for all supported metrics
        # (recall-style "higher is better" metrics negate internally so the
        # `cur < best` comparison works uniformly).
        #
        # Two YAML schemas are supported:
        #   * legacy single-best: ``best_ckpt_metric: <name>`` -- one ckpt
        #     written to ``checkpoints/best/`` (the v0 / v0_full path).
        #   * v0c-A multi-best: ``best_ckpt_metrics: [<name1>, <name2>, ...]``
        #     -- one ckpt per metric written to ``checkpoints/best_<alias>/``
        #     where ``alias`` is the canonical short tag for the metric (see
        #     ``_metric_alias`` below). The legacy ``checkpoints/best/`` is
        #     NOT written in multi-best mode to avoid ambiguity.
        # When both keys appear in the YAML, ``best_ckpt_metrics`` wins.
        multi_metrics = getattr(self.args, "best_ckpt_metrics", None) or []
        if multi_metrics:
            self._best_ckpt_metric_names: List[str] = [str(m) for m in multi_metrics]
            self._best_multi_mode: bool = True
        else:
            self._best_ckpt_metric_names = [
                str(getattr(self.args, "best_ckpt_metric", "val_flow_mean"))
            ]
            self._best_multi_mode = False
        self._best_metric_values: Dict[str, Optional[float]] = {
            n: None for n in self._best_ckpt_metric_names
        }
        self._best_steps: Dict[str, Optional[int]] = {
            n: None for n in self._best_ckpt_metric_names
        }
        # Legacy single-best aliases preserved for any caller / log line that
        # still reads them. They reflect the FIRST configured metric so
        # legacy single-best mode is byte-identical to the previous behavior.
        self._best_ckpt_metric: str = self._best_ckpt_metric_names[0]
        self._best_metric_value: Optional[float] = None
        self._best_step: Optional[int] = None

        self._init_distributed()
        self._init_save_folder_and_writer()
        self._init_logging()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, getattr(self.args, "logging_dir", "logs"))
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        ipg_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=getattr(self.args, "nccl_timeout", 1800))
        )
        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=getattr(self.args, "gradient_accumulation_steps", 1),
            mixed_precision=getattr(self.args, "mixed_precision", "no"),
            log_with=None,
            kwargs_handlers=[ddp_kwargs, ipg_kwargs],
        )
        self.state.accelerator = accelerator

        if getattr(self.args, "seed", None) is not None:
            self.state.seed = int(self.args.seed)
            # Deterministic per-rank seeding.
            set_seed(self.state.seed + accelerator.process_index)

        if accelerator.mixed_precision == "fp16":
            self.state.weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            self.state.weight_dtype = torch.bfloat16
        else:
            self.state.weight_dtype = torch.float32

    def _init_save_folder_and_writer(self):
        accelerator = self.state.accelerator
        if accelerator.is_main_process:
            ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            sub = getattr(self.args, "sub_folder", None)
            if sub:
                self.save_folder = os.path.join(self.args.output_dir, sub)
            else:
                self.save_folder = os.path.join(self.args.output_dir, ts)
            os.makedirs(self.save_folder, exist_ok=True)
            os.makedirs(os.path.join(self.save_folder, "checkpoints"), exist_ok=True)

            # Snapshot config.
            with open(os.path.join(self.save_folder, "config.yaml"), "w") as f:
                import yaml as _yaml
                _yaml.safe_dump(self.cfg, f, sort_keys=False)

            self.writer = SummaryWriter(log_dir=self.save_folder)
            self._init_wandb()
        else:
            self.save_folder = None
            self.writer = None
            self._wandb_enabled = False
            self._wandb_run = None

        # Broadcast save_folder path so all ranks know where to log.
        if accelerator.num_processes > 1:
            from torch import distributed as dist

            if accelerator.is_main_process:
                payload = self.save_folder.encode()
                length = torch.tensor([len(payload)], device=accelerator.device)
            else:
                length = torch.zeros(1, dtype=torch.long, device=accelerator.device)
            dist.broadcast(length, src=0)
            if accelerator.is_main_process:
                buf = torch.ByteTensor(list(payload)).to(accelerator.device)
            else:
                buf = torch.empty(int(length.item()), dtype=torch.uint8, device=accelerator.device)
            dist.broadcast(buf, src=0)
            if not accelerator.is_main_process:
                self.save_folder = bytes(buf.tolist()).decode()

        self.state.output_dir = self.save_folder

    def _init_wandb(self):
        """Initialize a wandb run on the main rank if configured.

        YAML schema (top-level ``wandb`` block, all fields optional except
        ``enabled``)::

            wandb:
              enabled: true|false      # default false
              project: <str>           # default "dex_vtam_tactile_vae_stage1"
              entity:  <str|null>      # default null (use account default)
              name:    <str|null>      # default basename(save_folder)
              tags:    [<str>, ...]    # default []
              notes:   <str|null>      # default null
              mode:    online|offline|disabled  # default "online"

        TB writer is unaffected — wandb is a parallel sink, so existing
        TB-based tooling keeps working.
        """
        self._wandb_enabled = False
        self._wandb_run = None

        cfg = getattr(self.args, "wandb", None) or {}
        if not bool(cfg.get("enabled", False)):
            logger.info("wandb logging: disabled (yaml: wandb.enabled=false or unset)")
            return

        try:
            import wandb
        except ImportError:
            logger.warning(
                "wandb requested via yaml but the package is not installed; "
                "falling back to TensorBoard only."
            )
            return

        project = cfg.get("project", "dex_vtam_tactile_vae_stage1")
        entity = cfg.get("entity", None)
        run_name = cfg.get("name") or os.path.basename(self.save_folder)
        tags = list(cfg.get("tags", []) or [])
        notes = cfg.get("notes", None)
        mode = cfg.get("mode", "online")

        # Keep wandb's local cache under the run dir so it's preserved with
        # the run + doesn't pollute $HOME on shared clusters. Pass save_folder
        # as `dir` directly -- wandb auto-creates a `wandb/` subdir inside it,
        # so passing `save_folder/wandb` would yield save_folder/wandb/wandb/.
        try:
            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                notes=notes,
                mode=mode,
                dir=self.save_folder,
                config=self.cfg,
                resume="allow",
                # `quiet` here (via Settings) replaces the deprecated
                # `wandb.finish(quiet=True)` we used to call in _cleanup.
                settings=wandb.Settings(quiet=True),
            )
            self._wandb_enabled = True
            logger.info(
                f"wandb logging enabled: project={project} entity={entity} "
                f"run={run_name} mode={mode} dir={self.save_folder}"
            )
        except Exception as e:
            logger.warning(
                f"wandb.init failed ({type(e).__name__}: {e}); falling back to TB only."
            )
            self._wandb_enabled = False
            self._wandb_run = None

    def _log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        """Fan-out scalar logging to TensorBoard + wandb (main rank only)."""
        if not self.state.accelerator.is_main_process:
            return
        if self.writer is not None:
            for tag, val in scalars.items():
                self.writer.add_scalar(tag, val, step)
        if getattr(self, "_wandb_enabled", False):
            try:
                import wandb
                wandb.log(scalars, step=step)
            except Exception as e:
                logger.warning(f"wandb.log failed at step {step}: {e}")

    def _log_image(self, tag: str, image_path: str, step: int) -> None:
        """Log an on-disk PNG to wandb. TB image logging is intentionally not
        mirrored (the run dir already keeps the PNG under val_viz/)."""
        if not self.state.accelerator.is_main_process:
            return
        if not getattr(self, "_wandb_enabled", False):
            return
        if not os.path.isfile(image_path):
            return
        try:
            import wandb
            wandb.log({tag: wandb.Image(image_path)}, step=step)
        except Exception as e:
            logger.warning(f"wandb.log image failed at step {step} ({tag}): {e}")

    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%m/%d %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.save_folder is not None:
            init_logging(self.save_folder, rank=self.state.accelerator.process_index)
        logger.info(f"output: {self.save_folder}")
        logger.info(f"distributed state: {self.state.accelerator.state}", main_process_only=False)

    # ------------------------------------------------------------------
    # Stage 1: dataset
    # ------------------------------------------------------------------

    def _build_dataset(self, split_cfg: Dict) -> TactileDataset:
        dataset_class_path = getattr(self.args, "train_data_class_path", None)
        if dataset_class_path is not None:
            cls = import_custom_class(self.args.train_data_class, dataset_class_path)
        else:
            cls = TactileDataset
        return cls(**split_cfg)

    def prepare_dataset(self):
        accelerator = self.state.accelerator

        data_cfg = self.args.data
        T_choices = tuple(data_cfg.get("T_choices", (1, 9)))
        T_sample_ratio = tuple(data_cfg.get("T_sample_ratio", (0.5, 0.5)))

        # Train dataset.
        self.train_dataset = self._build_dataset(data_cfg["train"])
        logger.info(
            f"Train dataset: {len(self.train_dataset)} samples "
            f"(per T: {[(t, len(self.train_dataset.indices_by_T[t])) for t in T_choices]})"
        )

        # Train batch sampler — per-rank seed offset for distributed.
        per_device_bs = int(self.args.batch_size)
        rank_seed = int(getattr(self.args, "seed", 0)) + accelerator.process_index * 1000
        num_train_batches = data_cfg.get("num_train_batches_per_epoch", None)
        if num_train_batches is None:
            # Default: full epoch over train_dataset / batch_size.
            num_train_batches = len(self.train_dataset) // per_device_bs

        self._train_batch_sampler = MixedTBatchSampler(
            dataset=self.train_dataset,
            batch_size=per_device_bs,
            T_sample_ratio=T_sample_ratio,
            num_batches=num_train_batches,
            seed=rank_seed,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self._train_batch_sampler,
            num_workers=int(getattr(self.args, "dataloader_num_workers", 0)),
            persistent_workers=bool(getattr(self.args, "persistent_workers", False))
            and int(getattr(self.args, "dataloader_num_workers", 0)) > 0,
            pin_memory=bool(getattr(self.args, "pin_memory", True)),
        )
        self._steps_per_epoch = len(self._train_batch_sampler)

        # Validation datasets — one DataLoader per T value for separate logging.
        if "val" in data_cfg:
            self.val_dataset = self._build_dataset(data_cfg["val"])
            logger.info(f"Val dataset: {len(self.val_dataset)} samples")

            val_batch_size = int(data_cfg.get("val_batch_size", per_device_bs))
            num_val_batches = int(data_cfg.get("num_val_batches", 8))
            # Reuse the train num_workers setting for val by default. The old
            # `num_workers=0` was fine for tiny corpora but becomes a hard
            # bottleneck on large multi-task datasets (488+ episodes), where
            # `_deep_stack` deserialization in the main process can stall the
            # initial validation pass for tens of minutes. Allow override via
            # `data.val.num_workers` in the YAML if you want a different value
            # for validation specifically.
            # Val DataLoader: default to num_workers=0 because every other
            # combination we tried on the 488-episode corpus deadlocked the
            # very first iteration of `validate()`:
            #
            #   * num_workers=4 + pin_memory=True + persistent_workers=True
            #       -> hung 7 min, 4 workers idle on futex_wait, GPU 0%, no
            #       forward, no traceback. Suspected pytorch/pytorch#52928.
            #   * num_workers=4 + pin_memory=True + persistent_workers=False
            #       -> SAME 4-min hang pattern. Same 2/4 workers in R, 2/4
            #       in futex_wait, main futex_wait, GPU memory frozen at
            #       initial allocation (no forward ever happened).
            #
            # The exact mechanism is unclear (could be pin_memory thread,
            # could be fork-inherited CUDA context corruption in workers,
            # could be parquet/pyarrow contention with 488 episodes worth
            # of file handles in flight). num_workers=0 sidesteps all
            # multi-process concerns and simply does `_deep_stack` in the
            # main process. It's slower (~5-10 min per val pass on the 488
            # corpus, vs ~1-2 min if multi-worker had worked) but reliable.
            #
            # If you want to override (e.g. on a smaller corpus where the
            # deadlock hasn't been observed), set `data.val.num_workers: N`
            # in the YAML.
            train_num_workers = int(getattr(self.args, "dataloader_num_workers", 0))
            val_num_workers_default = 0  # see comment above; do NOT inherit train default
            val_num_workers = int(data_cfg.get("val", {}).get("num_workers", val_num_workers_default))
            logger.info(
                f"Val DataLoader: num_workers={val_num_workers} "
                f"(train uses {train_num_workers}; val pinned to 0 by default to avoid "
                f"the 488-corpus pin_memory/_deep_stack deadlock observed 2026-05-14)"
            )
            self.val_dataloaders: Dict[int, DataLoader] = {}
            for T in T_choices:
                sampler = FixedTBatchSampler(
                    dataset=self.val_dataset,
                    T=T,
                    batch_size=val_batch_size,
                    num_batches=num_val_batches,
                    seed=int(getattr(self.args, "seed", 0)),
                    shuffle=True,
                )
                self.val_dataloaders[T] = DataLoader(
                    self.val_dataset,
                    batch_sampler=sampler,
                    num_workers=val_num_workers,
                    persistent_workers=False,
                    pin_memory=True,
                )
        else:
            self.val_dataset = None
            self.val_dataloaders = {}

    # ------------------------------------------------------------------
    # Stage 2: model
    # ------------------------------------------------------------------

    def prepare_models(self):
        cfg = self.args.tactile_vae
        # Loss weights / loss-term hparams live under tactile_vae.config for
        # plan parity, but they are NOT TactileVAE __init__ kwargs — strip
        # them before constructing. ``free_bits_tau`` is a v3 loss-term hparam;
        # ``flow_loss`` is the v4 magnitude-aware-loss hparam dict.
        _LOSS_ONLY_KEYS = (
            "lambda_loc", "lambda_pose", "lambda_kl",
            "free_bits_tau",
            "flow_loss",
        )
        model_cfg = {
            k: v for k, v in cfg["config"].items() if k not in _LOSS_ONLY_KEYS
        }
        self.vae: TactileVAE = TactileVAE(**model_cfg)

        # Optional checkpoint load.
        ckpt_path = cfg.get("model_path", None)
        if ckpt_path:
            logger.info(f"Loading TactileVAE checkpoint from: {ckpt_path}")
            sd = torch.load(ckpt_path, map_location="cpu")
            if "model" in sd:
                sd = sd["model"]
            missing, unexpected = self.vae.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"Missing keys when loading: {missing[:5]}{'...' if len(missing)>5 else ''}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

        n_params = sum(p.numel() for p in self.vae.parameters())
        n_trainable = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        logger.info(f"TactileVAE total params    : {n_params/1e6:.2f} M")
        logger.info(f"TactileVAE trainable params: {n_trainable/1e6:.2f} M")
        self.state.num_trainable_parameters = n_trainable

        if getattr(self.args, "gradient_checkpointing", False):
            logger.info("Enabling gradient checkpointing on the encoder backbone.")
            self.vae.set_gradient_checkpointing(True)

        if getattr(self.args, "allow_tf32", True) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # ----- v4: cache flow normalization stats for physical-unit weighting ---
        # The v4 magnitude-aware flow loss and the v4b validation diagnostics
        # both threshold / weight on ||flow||_2 in PHYSICAL flow units (px /
        # frame). To keep this comparable across runs trained with different
        # normalization stats, we denormalize gt / pred flow at training time
        # the same way the inferencer does at eval time.
        flow_stats_path: Optional[str] = (
            self.cfg.get("data", {}).get("train", {}).get("flow_stats_path")
            or self.cfg.get("data", {}).get("val", {}).get("flow_stats_path")
        )
        flow_stats = _load_stats_json(flow_stats_path, expected_dim=3)
        device = self.state.accelerator.device
        if flow_stats is not None:
            # Shape (1, 1, 1, 1, 1, 3) so it broadcasts to (B, F, T, H, W, 3).
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

    def _denormalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """Map normalized flow back to physical (px / frame) units.

        Identity if ``flow_stats`` weren't loadable, matching the inferencer's
        :meth:`TactileVAEInferencer.denormalize_flow` behavior.
        """
        if self._flow_mean_t is None:
            return flow
        return (
            flow * self._flow_std_t.to(device=flow.device, dtype=flow.dtype)
            + self._flow_mean_t.to(device=flow.device, dtype=flow.dtype)
        )

    # ------------------------------------------------------------------
    # Stage 3: optimizer + LR scheduler
    # ------------------------------------------------------------------

    def prepare_optimizer(self):
        accelerator = self.state.accelerator

        self.state.train_epochs = int(getattr(self.args, "train_epochs", 50))
        self.state.train_steps = getattr(self.args, "train_steps", None)
        if self.state.train_steps is None:
            steps_per_epoch = math.ceil(self._steps_per_epoch / max(1, getattr(self.args, "gradient_accumulation_steps", 1)))
            self.state.train_steps = self.state.train_epochs * steps_per_epoch
        else:
            self.state.train_steps = int(self.state.train_steps)

        self.state.learning_rate = float(self.args.lr)
        if getattr(self.args, "scale_lr", False):
            self.state.learning_rate *= (
                getattr(self.args, "gradient_accumulation_steps", 1)
                * self.args.batch_size
                * accelerator.num_processes
            )

        params_to_optimize = [
            {"params": list(p for p in self.vae.parameters() if p.requires_grad),
             "lr": self.state.learning_rate}
        ]
        self.optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=getattr(self.args, "optimizer", "adamw"),
            learning_rate=self.state.learning_rate,
            beta1=float(getattr(self.args, "beta1", 0.9)),
            beta2=float(getattr(self.args, "beta2", 0.95)),
            beta3=float(getattr(self.args, "beta3", 0.98)),
            epsilon=float(getattr(self.args, "epsilon", 1e-8)),
            weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
            use_8bit=bool(getattr(self.args, "optimizer_8bit", False)),
            use_torchao=bool(getattr(self.args, "optimizer_torchao", False)),
        )

        self.lr_scheduler = get_scheduler(
            name=getattr(self.args, "lr_scheduler", "constant_with_warmup"),
            optimizer=self.optimizer,
            num_warmup_steps=int(getattr(self.args, "lr_warmup_steps", 500)) * accelerator.num_processes,
            num_training_steps=self.state.train_steps * accelerator.num_processes,
            num_cycles=int(getattr(self.args, "lr_num_cycles", 1)),
            power=float(getattr(self.args, "lr_power", 1.0)),
        )

    # ------------------------------------------------------------------
    # Stage 4: prepare for training (Accelerate wrap)
    # ------------------------------------------------------------------

    def prepare_for_training(self):
        # NOTE: we deliberately do *not* pass the dataloader to `accelerate.prepare()`
        # because that would replace our custom `MixedTBatchSampler` with a
        # DistributedSampler. We seed the batch sampler per-rank instead.
        self.vae, self.optimizer, self.lr_scheduler = self.state.accelerator.prepare(
            self.vae, self.optimizer, self.lr_scheduler
        )

    # ------------------------------------------------------------------
    # KL annealing
    # ------------------------------------------------------------------

    def _current_lambda_kl(self, global_step: int) -> float:
        base = float(self.args.tactile_vae["config"].get("lambda_kl", 1e-4))
        warmup = int(getattr(self.args, "lambda_kl_warmup_steps", 0))
        if warmup <= 0:
            return base
        return base * min(1.0, global_step / warmup)

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        sample: bool,
    ) -> Dict[str, torch.Tensor]:
        out = self.vae(batch["tactile"], batch["hand_pose"], sample=sample)
        gt = batch["tactile_flow"]
        pred = out["flow_pred"]
        flow_loss = self._compute_flow_loss(pred, gt)
        pose_loss = F.mse_loss(out["pose_pred"], batch["hand_pose"])
        free_bits_tau = float(self.args.tactile_vae["config"].get("free_bits_tau", 0.0))
        kl = _kl_loss_per_sample_sum(
            out["mu"], out["logvar"], free_bits_tau=free_bits_tau
        )
        return {
            "flow_loss": flow_loss,
            "pose_loss": pose_loss,
            "kl_loss": kl["loss"],
            "kl_raw": kl["raw"],
            "kl_floored": kl["floored"],
            "out": out,
        }

    def _compute_flow_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Flow MSE with optional v4 magnitude-aware sqrt-smoothed reweighting.

        When ``tactile_vae.config.flow_loss.enabled`` is true (v4+), this is a
        normalized weighted mean (form b):

            rel    = ||gt_phys|| / mean(||gt_phys||)            # per (B, F)
            weight = clamp(1 + lambda_w * sqrt(rel), max=w_max)
            loss   = sum(weight * err_sq) / (sum(weight) * 3)

        Form (b) keeps the loss numerically comparable to plain MSE, so
        ``lambda_loc / lambda_pose / lambda_kl`` ratios stay calibrated and v3
        vs v4 ``train/flow_loss`` curves are directly overlayable on WandB.

        When the ``flow_loss`` block is absent or disabled (v1 / v2 / v3
        configs), this reduces to the legacy ``F.mse_loss`` so older yamls
        keep working unchanged.
        """
        flow_cfg = self.args.tactile_vae["config"].get("flow_loss") or {}
        if not flow_cfg.get("enabled", False):
            return F.mse_loss(pred, gt)

        eps = float(flow_cfg.get("eps", 1.0e-6))
        lambda_w = float(flow_cfg.get("lambda_w", 2.0))
        w_max = float(flow_cfg.get("w_max", 5.0))
        window = int(flow_cfg.get("temporal_smooth_window", 1))

        # Build the per-pixel weight from GT motion magnitude in PHYSICAL flow
        # units. ``mag_phys`` is computed under no_grad because the weight is
        # a non-differentiable function of the GT and we only want gradients
        # to flow through ``err_sq``. (PyTorch will already not backprop
        # through gt -> mag because gt has requires_grad=False, but we make
        # this explicit to keep the intent visible.)
        with torch.no_grad():
            gt_phys = self._denormalize_flow(gt)
            mag_phys = torch.linalg.norm(gt_phys, dim=-1, keepdim=True)     # (B, F, T, H, W, 1)
            if window > 1:
                mag_phys = _avg_pool_T(mag_phys, kernel=window)
            # Per-(sample, finger) clip-mean over (T, H, W). A global-batch
            # mean would push pinky's relative weight DOWN because thumb has
            # higher baseline raw motion; per (B, F) keeps "rel = 1 means
            # average for THIS finger in THIS clip" semantics. This matters
            # for tasks like wrap_adhesive where thumb dominates raw
            # magnitude (see [tactile_vae_v4_plan]).
            mean_mag = mag_phys.mean(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
            rel = mag_phys / mean_mag
            weight = (1.0 + lambda_w * rel.sqrt()).clamp(max=w_max)         # (B, F, T, H, W, 1)

        err_sq = (pred - gt).pow(2)                                          # (B, F, T, H, W, 3)
        # Form (b): true weighted mean. Denominator * 3 because err_sq has 3
        # flow channels and weight broadcasts over them; this preserves
        # ``loss = E[w * err^2] / E[w]`` semantics. In the all-inactive case
        # (rel = 0 everywhere -> weight = 1), the loss reduces exactly to
        # plain MSE, giving a clean fallback boundary.
        denom = (weight.sum() * err_sq.shape[-1]).clamp_min(eps)
        return (err_sq * weight).sum() / denom

    # ------------------------------------------------------------------
    # Stage 5: training loop
    # ------------------------------------------------------------------

    def train(self):
        accelerator = self.state.accelerator
        device = accelerator.device

        cfg = self.args.tactile_vae["config"]
        lambda_loc = float(cfg.get("lambda_loc", 1.0))
        lambda_pose = float(cfg.get("lambda_pose", 0.1))

        steps_to_log = int(getattr(self.args, "steps_to_log", 50))
        steps_to_val = int(getattr(self.args, "steps_to_val", 1000))
        steps_to_save = int(getattr(self.args, "steps_to_save", 5000))
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 1.0))

        logger.info(
            f"Training: train_steps={self.state.train_steps}  "
            f"steps_per_epoch={self._steps_per_epoch}  "
            f"epochs={self.state.train_epochs}  "
            f"batch_size={self.args.batch_size}  "
            f"world_size={accelerator.num_processes}"
        )

        global_step = 0
        progress_bar = tqdm(
            range(self.state.train_steps),
            desc="train",
            disable=not accelerator.is_local_main_process,
        )

        # If we have validation, run an initial pass at step 0 to confirm pipeline.
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
                    losses = self._compute_losses(batch, sample=True)
                    flow_loss = losses["flow_loss"]
                    pose_loss = losses["pose_loss"]
                    kl_loss = losses["kl_loss"]
                    kl_raw = losses["kl_raw"]
                    kl_floored = losses["kl_floored"]

                    lambda_kl = self._current_lambda_kl(global_step)
                    total_loss = (
                        lambda_loc * flow_loss
                        + lambda_pose * pose_loss
                        + lambda_kl * kl_loss
                    )

                    if not torch.isfinite(total_loss):
                        logger.warning(
                            f"Non-finite loss at step {global_step} (flow={flow_loss.item():.4f} "
                            f"pose={pose_loss.item():.4f} kl={kl_loss.item():.4f}); skipping."
                        )
                        self.optimizer.zero_grad()
                        continue

                    accelerator.backward(total_loss)

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
                flow_l = accelerator.reduce(flow_loss.detach(), reduction="mean")
                pose_l = accelerator.reduce(pose_loss.detach(), reduction="mean")
                kl_l = accelerator.reduce(kl_loss.detach(), reduction="mean")
                kl_raw_l = accelerator.reduce(kl_raw.detach(), reduction="mean")
                kl_floored_l = accelerator.reduce(kl_floored.detach(), reduction="mean")
                total_l = accelerator.reduce(total_loss.detach(), reduction="mean")

                postfix = {
                    "T": int(batch["meta"]["T"][0].item()),
                    "loss": float(total_l.item()),
                    "flow": float(flow_l.item()),
                    "pose": float(pose_l.item()),
                    "kl": float(kl_l.item()),
                }
                progress_bar.set_postfix(postfix)

                if (
                    accelerator.is_main_process
                    and global_step > 0
                    and (global_step % steps_to_log == 0)
                ):
                    train_scalars = {
                        "train/loss_total": total_l.item(),
                        "train/loss_flow":  flow_l.item(),
                        "train/loss_pose":  pose_l.item(),
                        "train/loss_kl":    kl_l.item(),
                        # v3: surface raw vs floored KL so we can verify the
                        # free-bits floor is engaged (floored >= raw when active).
                        "train/kl_loss_raw":     kl_raw_l.item(),
                        "train/kl_loss_floored": kl_floored_l.item(),
                        "train/lambda_kl":       lambda_kl,
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

        # Final checkpoint + validation.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if self.val_dataloaders:
                final_metrics = self.validate(global_step)
                self._maybe_save_best(global_step, final_metrics)
            self.save_checkpoint(global_step, tag="final")

        progress_bar.close()
        logger.info("Training complete.")
        self._cleanup()

    def _cleanup(self):
        """Explicit tear-down to avoid noisy ``sys.unraisablehook`` chatter at exit.

        ``utils.init_logging`` swaps ``sys.stderr``/``sys.stdout`` for a ``Tee``
        that writes to a file. After the atexit hook closes the file, late
        writes from CUDA/DataLoader cleanup threads explode in the Tee and
        bubble up as "Exception ignored in sys.unraisablehook" with a non-zero
        exit code. Restoring the originals keeps the process exit clean.
        """
        if getattr(self, "writer", None) is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            self.writer = None
        # Finalize wandb run (flushes and uploads remaining metrics/images).
        # `quiet` is set at init via wandb.Settings(quiet=True), so we must
        # NOT pass it here -- wandb 0.21+ deprecated the kwarg on finish().
        if getattr(self, "_wandb_enabled", False):
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
            self._wandb_enabled = False
            self._wandb_run = None
        try:
            plt.close("all")
        except Exception:
            pass
        # Help DataLoader worker threads exit cleanly.
        for attr in ("train_dataloader",):
            loader = getattr(self, attr, None)
            if loader is not None and hasattr(loader, "_iterator") and loader._iterator is not None:
                try:
                    loader._iterator._shutdown_workers()
                except Exception:
                    pass
        # Restore stdout/stderr so post-exit writes don't crash through Tee.
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, global_step: int) -> Dict[str, float]:
        """Run validation for each T separately. Logs to TB and saves a flow viz.

        Only called on the main process.
        """
        self.vae.eval()
        device = self.state.accelerator.device

        # v4b contact-aware mask threshold. Read from flow_loss block; default
        # 0.5 px / frame so v3 / v2 yamls (which don't have the block) still
        # get the same diagnostic metrics. Squared error itself is computed
        # in normalized space (matches the optimization signal); the mask
        # only decides which pixels are aggregated into active vs inactive.
        flow_cfg = self.args.tactile_vae["config"].get("flow_loss") or {}
        contact_threshold = float(flow_cfg.get("contact_threshold", 0.5))

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]

        all_metrics: Dict[str, float] = {}
        for T, loader in self.val_dataloaders.items():
            agg = {
                "flow_loss": 0.0,
                "pose_loss": 0.0,
                "kl_loss": 0.0,
                "kl_raw": 0.0,
                "kl_floored": 0.0,
            }
            # v4b accumulators (per finger). float64 on device for precision
            # when summing across many batches and many pixels.
            ca_active_sq      = torch.zeros(5, device=device, dtype=torch.float64)
            ca_inactive_sq    = torch.zeros(5, device=device, dtype=torch.float64)
            ca_active_cnt     = torch.zeros(5, device=device, dtype=torch.float64)
            ca_inactive_cnt   = torch.zeros(5, device=device, dtype=torch.float64)
            ca_pred_active    = torch.zeros(5, device=device, dtype=torch.float64)
            ca_tp             = torch.zeros(5, device=device, dtype=torch.float64)

            n_batches = 0
            first_batch = None
            first_out = None
            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                losses = self._compute_losses(batch, sample=False)
                agg["flow_loss"] += losses["flow_loss"].item()
                agg["pose_loss"] += losses["pose_loss"].item()
                agg["kl_loss"] += losses["kl_loss"].item()
                agg["kl_raw"] += losses["kl_raw"].item()
                agg["kl_floored"] += losses["kl_floored"].item()
                n_batches += 1
                if first_batch is None:
                    first_batch = batch
                    first_out = losses["out"]

                # v4b mirror: same physical-unit threshold semantics as the
                # inferencer. err_pix is in normalized space (matches what the
                # trainer optimizes); the mask is in physical space (matches
                # the inferencer / WandB / cross-run comparability).
                gt_f = batch["tactile_flow"].float()
                pred_f = losses["out"]["flow_pred"].float()
                gt_phys = self._denormalize_flow(gt_f)
                pred_phys = self._denormalize_flow(pred_f)
                gt_mag = torch.linalg.norm(gt_phys, dim=-1)                     # (B, 5, T, H, W)
                pred_mag = torch.linalg.norm(pred_phys, dim=-1)
                gt_active = gt_mag > contact_threshold
                pred_active = pred_mag > contact_threshold
                err_pix = (pred_f - gt_f).pow(2).mean(dim=-1)                   # (B, 5, T, H, W)
                reduce_dims = [0, 2, 3, 4]                                      # keep finger axis
                ca_active_sq    = ca_active_sq    + (err_pix * gt_active.float()).sum(dim=reduce_dims).double()
                ca_inactive_sq  = ca_inactive_sq  + (err_pix * (~gt_active).float()).sum(dim=reduce_dims).double()
                ca_active_cnt   = ca_active_cnt   + gt_active.float().sum(dim=reduce_dims).double()
                ca_inactive_cnt = ca_inactive_cnt + (~gt_active).float().sum(dim=reduce_dims).double()
                ca_pred_active  = ca_pred_active  + pred_active.float().sum(dim=reduce_dims).double()
                ca_tp           = ca_tp           + (pred_active & gt_active).float().sum(dim=reduce_dims).double()

            if n_batches == 0:
                continue
            for k in agg:
                agg[k] /= n_batches

            # v4b per-finger derived metrics (5 floats each).
            total_cnt = (ca_active_cnt + ca_inactive_cnt).clamp_min(1.0)
            flow_active_pf   = (ca_active_sq / ca_active_cnt.clamp_min(1.0)).cpu().tolist()
            flow_inactive_pf = (ca_inactive_sq / ca_inactive_cnt.clamp_min(1.0)).cpu().tolist()
            recall_pf        = (ca_tp / ca_active_cnt.clamp_min(1.0)).cpu().tolist()
            precision_pf     = (ca_tp / ca_pred_active.clamp_min(1.0)).cpu().tolist()
            contact_ratio_pf = (ca_active_cnt / total_cnt).cpu().tolist()

            flow_active_mean   = sum(flow_active_pf)   / 5.0
            flow_inactive_mean = sum(flow_inactive_pf) / 5.0
            recall_mean        = sum(recall_pf)        / 5.0
            precision_mean     = sum(precision_pf)     / 5.0
            contact_ratio_mean = sum(contact_ratio_pf) / 5.0

            tag = f"val/T{T}"
            logger.info(
                f"step {global_step}  {tag}: "
                f"flow={agg['flow_loss']:.4f}  "
                f"pose={agg['pose_loss']:.4f}  "
                f"kl={agg['kl_loss']:.4f}  "
                f"kl_raw={agg['kl_raw']:.4f}  "
                f"kl_floored={agg['kl_floored']:.4f}  "
                f"flow_active={flow_active_mean:.4f}  "
                f"flow_inactive={flow_inactive_mean:.4f}  "
                f"recall={recall_mean:.3f}  "
                f"precision={precision_mean:.3f}"
            )
            self._log_scalars(
                {
                    f"{tag}/loss_flow":        agg["flow_loss"],
                    f"{tag}/loss_pose":        agg["pose_loss"],
                    f"{tag}/loss_kl":          agg["kl_loss"],
                    f"{tag}/kl_loss_raw":      agg["kl_raw"],
                    f"{tag}/kl_loss_floored":  agg["kl_floored"],
                    f"{tag}/flow_mse_active":    flow_active_mean,
                    f"{tag}/flow_mse_inactive":  flow_inactive_mean,
                    f"{tag}/active_recall":      recall_mean,
                    f"{tag}/active_precision":   precision_mean,
                    f"{tag}/contact_ratio_mean": contact_ratio_mean,
                },
                global_step,
            )
            # Per-finger TB / WandB scalars for in-flight imbalance debugging.
            for fi, name in enumerate(finger_names):
                self._log_scalars(
                    {
                        f"{tag}/flow_mse_active_{name}":    flow_active_pf[fi],
                        f"{tag}/flow_mse_inactive_{name}":  flow_inactive_pf[fi],
                        f"{tag}/active_recall_{name}":      recall_pf[fi],
                        f"{tag}/active_precision_{name}":   precision_pf[fi],
                        f"{tag}/contact_ratio_{name}":      contact_ratio_pf[fi],
                    },
                    global_step,
                )

            # Legacy key (preserved for any downstream consumer keyed on val/T{T}).
            all_metrics[tag] = agg["flow_loss"]
            # Explicit per-T per-metric keys (used by best-ckpt selector + future tooling).
            all_metrics[f"{tag}/flow_loss"] = agg["flow_loss"]
            all_metrics[f"{tag}/pose_loss"] = agg["pose_loss"]
            all_metrics[f"{tag}/kl_loss"] = agg["kl_loss"]
            all_metrics[f"{tag}/kl_raw"] = agg["kl_raw"]
            all_metrics[f"{tag}/kl_floored"] = agg["kl_floored"]
            # v4b: feeds the val_flow_mse_active_mean best-ckpt selector and any
            # future tooling that wants the per-T contact-aware metrics.
            all_metrics[f"{tag}/flow_mse_active"]    = flow_active_mean
            all_metrics[f"{tag}/flow_mse_inactive"]  = flow_inactive_mean
            all_metrics[f"{tag}/active_recall"]      = recall_mean
            all_metrics[f"{tag}/active_precision"]   = precision_mean
            all_metrics[f"{tag}/contact_ratio_mean"] = contact_ratio_mean

            # Save a flow visualization for the FIRST sample of the FIRST val batch.
            if first_batch is not None and self.save_folder is not None:
                self._save_flow_viz(
                    first_batch, first_out, T=T, global_step=global_step
                )

        return all_metrics

    def _save_flow_viz(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        T: int,
        global_step: int,
    ):
        """Save a 5×6 grid: 5 fingers × (3 GT channels + 3 pred channels) at the last frame."""
        viz_dir = os.path.join(self.save_folder, "val_viz", f"step_{global_step:08d}")
        os.makedirs(viz_dir, exist_ok=True)

        flow_gt = batch["tactile_flow"][0].detach().cpu().float().numpy()    # (5, T, 24, 32, 3)
        flow_pred = out["flow_pred"][0].detach().cpu().float().numpy()        # (5, T, 24, 32, 3)
        last_t = flow_gt.shape[1] - 1

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        chan_names = ["dx", "dy", "div"]
        fig, axes = plt.subplots(5, 6, figsize=(14, 11))
        fig.suptitle(f"Val flow @ step {global_step}, T={T}, frame={last_t}", fontsize=12)

        for fi in range(5):
            for ci in range(3):
                axes[fi, ci].imshow(flow_gt[fi, last_t, :, :, ci], cmap="seismic")
                axes[fi, ci].set_title(f"GT {finger_names[fi]} {chan_names[ci]}", fontsize=8)
                axes[fi, ci].axis("off")
                axes[fi, 3 + ci].imshow(flow_pred[fi, last_t, :, :, ci], cmap="seismic")
                axes[fi, 3 + ci].set_title(f"Pred {finger_names[fi]} {chan_names[ci]}", fontsize=8)
                axes[fi, 3 + ci].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(viz_dir, f"flow_T{T}.png")
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)

        # Mirror to wandb (no-op if wandb is disabled). The PNG remains on disk
        # under val_viz/ regardless, so this is purely for convenience in the UI.
        self._log_image(f"val/T{T}/flow_viz", path, global_step)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, global_step: int, tag: Optional[str] = None):
        if not self.state.accelerator.is_main_process:
            return
        if tag is None:
            tag = f"step_{global_step:08d}"
        ckpt_dir = os.path.join(self.save_folder, "checkpoints", tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        unwrapped = self.state.accelerator.unwrap_model(self.vae)
        torch.save(
            {
                "model": unwrapped.state_dict(),
                "global_step": global_step,
                "config": self.cfg,
            },
            os.path.join(ckpt_dir, "model.pt"),
        )
        with open(os.path.join(ckpt_dir, "step.txt"), "w") as f:
            f.write(str(global_step))
        # Also overwrite "latest" pointer.
        latest = os.path.join(self.save_folder, "checkpoints", "latest")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(tag, latest)
        except OSError:
            pass
        logger.info(f"Saved checkpoint to: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Best-checkpoint tracking (v3)
    # ------------------------------------------------------------------

    # Map full metric name -> short alias used for ``checkpoints/best_<alias>/``.
    # Names not in this map fall back to a sanitized version of the name itself.
    _BEST_METRIC_ALIASES: Dict[str, str] = {
        "val_flow_mean":              "flow_mean",
        "val_T1_flow":                "T1_flow",
        "val_T9_flow":                "T9_flow",
        "val_flow_mse_active_mean":   "active_mse",
        "val_recall_post_mean":       "recall_post",
    }
    # Metrics where the underlying scalar is "higher is better"; we negate the
    # selector value internally so ``cur < best`` keeps working uniformly.
    _BEST_METRIC_NEGATED: set = {"val_recall_post_mean"}

    def _metric_alias(self, name: str) -> str:
        if name in self._BEST_METRIC_ALIASES:
            return self._BEST_METRIC_ALIASES[name]
        # Sanitize: strip "val_" prefix, replace separators.
        alias = name
        if alias.startswith("val_"):
            alias = alias[len("val_"):]
        return alias.replace("/", "_").replace(" ", "_")

    def _select_best_metric_value(
        self,
        metrics: Dict[str, float],
        metric_name: Optional[str] = None,
    ) -> Optional[float]:
        """Pick the scalar metric used for "best" comparison.

        ``metric_name`` controls the choice (or falls back to
        ``self._best_ckpt_metric`` for legacy callers):
          * ``val_flow_mean`` (default): mean over ``val/T*/flow_loss``
          * ``val_T1_flow``  : ``val/T1/flow_loss``
          * ``val_T9_flow``  : ``val/T9/flow_loss``
          * ``val_flow_mse_active_mean`` (v4): mean over
            ``val/T*/flow_mse_active`` (or ``/flow_mse_active_post`` for the
            v0c lite trainer; the trainer subclass overrides this method to
            prefer the post-fuse stream).
          * ``val_recall_post_mean`` (v0c): mean over
            ``val/T*/active_recall_post``. Higher is better, so negated
            internally.
          * any explicit metric key already present in ``metrics``
        """
        name = metric_name if metric_name is not None else self._best_ckpt_metric
        if name == "val_flow_mean":
            flows = [v for k, v in metrics.items() if k.endswith("/flow_loss")]
            if not flows:
                return None
            return float(sum(flows) / len(flows))
        if name == "val_flow_mse_active_mean":
            actives = [v for k, v in metrics.items() if k.endswith("/flow_mse_active")]
            if not actives:
                return None
            return float(sum(actives) / len(actives))
        if name == "val_recall_post_mean":
            # Higher recall is better; negate so the existing `cur < best`
            # comparison works.
            posts = [
                v for k, v in metrics.items()
                if k.endswith("/active_recall_post")
            ]
            if not posts:
                return None
            return -float(sum(posts) / len(posts))
        if name == "val_T1_flow":
            return metrics.get("val/T1/flow_loss")
        if name == "val_T9_flow":
            return metrics.get("val/T9/flow_loss")
        return metrics.get(name)

    def _maybe_save_best(self, global_step: int, metrics: Dict[str, float]) -> None:
        """Update + persist the best checkpoint(s) if any tracked metric improved.

        Called after every ``validate(...)`` invocation. In multi-best mode
        each tracked metric writes its own ``checkpoints/best_<alias>/`` ckpt
        independently. Only the main rank persists state.
        """
        accelerator = self.state.accelerator
        if not accelerator.is_main_process:
            return
        if not metrics:
            return

        for metric_name in self._best_ckpt_metric_names:
            cur = self._select_best_metric_value(metrics, metric_name)
            if cur is None or not math.isfinite(cur):
                continue

            prev = self._best_metric_values.get(metric_name)
            if prev is None or cur < prev:
                self._best_metric_values[metric_name] = cur
                self._best_steps[metric_name] = global_step
                # Display value: un-negate for "higher is better" metrics.
                display_value = (
                    -cur if metric_name in self._BEST_METRIC_NEGATED else cur
                )
                display_prev = (
                    None if prev is None else
                    (-prev if metric_name in self._BEST_METRIC_NEGATED else prev)
                )
                logger.info(
                    f"NEW BEST @ step {global_step}: "
                    f"{metric_name}={display_value:.6f} (prev={display_prev})"
                )
                # Mirror the legacy single-best fields (first metric wins).
                if metric_name == self._best_ckpt_metric_names[0]:
                    self._best_metric_value = cur
                    self._best_step = global_step
                self._save_best_checkpoint(
                    global_step, metric_name, cur, metrics,
                )

    def _save_best_checkpoint(
        self,
        global_step: int,
        metric_name: str,
        metric_value: float,
        metrics: Dict[str, float],
    ) -> None:
        """Persist ``checkpoints/best{,_<alias>}/{model.pt, best_metrics.json}``.

        In legacy single-best mode the directory is ``checkpoints/best/`` (the
        existing v0 / v0_full path -- byte-identical to the previous trainer).
        In multi-best mode each tracked metric writes to its own
        ``checkpoints/best_<alias>/`` directory.
        """
        if not self.state.accelerator.is_main_process:
            return
        if self._best_multi_mode:
            ckpt_dir = os.path.join(
                self.save_folder, "checkpoints",
                f"best_{self._metric_alias(metric_name)}",
            )
        else:
            ckpt_dir = os.path.join(self.save_folder, "checkpoints", "best")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Display value: un-negate for "higher is better" metrics so probe
        # tooling sees the natural number.
        display_value = (
            -metric_value if metric_name in self._BEST_METRIC_NEGATED
            else metric_value
        )

        unwrapped = self.state.accelerator.unwrap_model(self.vae)
        torch.save(
            {
                "model": unwrapped.state_dict(),
                "global_step": global_step,
                "config": self.cfg,
                "best_metric_name": metric_name,
                # Stored value matches the comparison key (negated for higher-
                # is-better metrics). Tools that want the natural value should
                # read ``best_metrics.json`` (key: ``metric_value_display``)
                # OR look at ``all_val_metrics`` for the per-T raw scalars.
                "best_metric_value": metric_value,
                "best_metric_value_display": display_value,
            },
            os.path.join(ckpt_dir, "model.pt"),
        )
        with open(os.path.join(ckpt_dir, "best_metrics.json"), "w") as f:
            json.dump(
                {
                    "global_step": global_step,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "metric_value_display": display_value,
                    "all_val_metrics": {
                        k: float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Saved BEST checkpoint to: {ckpt_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir.")
    args = parser.parse_args()

    trainer = TactileVAETrainer(args.config, output_dir=args.output_dir)
    trainer.prepare_dataset()
    trainer.prepare_models()
    trainer.prepare_optimizer()
    trainer.prepare_for_training()
    trainer.train()


if __name__ == "__main__":
    main()
