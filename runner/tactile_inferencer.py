# This file is an adaptation of Genie-Envisioner (AgibotTech) code that upstream
# licenses under CC BY-NC-SA 4.0, so the ShareAlike term applies and this file is
# distributed under the same licence rather than the repository's Apache 2.0:
# see LICENSES/CC-BY-NC-SA-4.0.txt. NonCommercial use only.
#
# Modified by the DexTacWAM Authors, 2026.

import os, random, math
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
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video, _normalize_latents

from data.utils.statistics import StatisticInfo

# ----------------------------------------------------
# Tactile-aware additions (Stage 3 inference). These mirror the trainer's
# tactile injection path so we can evaluate the action expert on the exact
# same view stack it saw during training. Kept import-side so the non-tactile
# branch in `validate` (yaml `use_tactile_views=false`) does not pay any
# additional dependency cost beyond ge_inferencer's own deps.
from models.tactile_models.projector import TactileProjector
from models.pipeline.custom_pipeline import calculate_shift, retrieve_timesteps
from runner.tactile_dit_trainer import _load_v0c_a_frozen
from diffusers import FlowMatchEulerDiscreteScheduler



class TactileInferencer:

    def __init__(self, config_file, output_dir=None, weight_dtype=torch.bfloat16, device="cuda:0", action_norm_type="meanstd") -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')

        # Tokenizers
        self.tokenizer = None

        # Text encoders
        self.text_encoder = None

        # Denoisers
        self.diffusion_model = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        # Tactile-aware additions: lazily filled by prepare_models() when
        # yaml `use_tactile_views=true`; remain None otherwise so the
        # non-tactile branch in validate() works identically to ge_inferencer.
        self.tactile_vae = None
        self.projector = None
        self.scheduler_action = None
        # action_only_dim splits the 194-dim action vector into
        # [0:action_only_dim]=action_dims and [action_only_dim:]=state_dims
        # for the action-vs-state MSE diagnostic. Populated lazily inside
        # _validate_with_tactile by probing the first val batch. Yaml override
        # via `tactile_inference.action_only_dim_override` short-circuits the
        # probe (use for sanity / ablation runs where you want to force a
        # specific split point).
        self.action_only_dim = None

        self.args.output_dir = Path(self.args.output_dir)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_folder = os.path.join(self.args.output_dir, start_time)
        if getattr(self.args, "sub_folder", False):
            self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
        os.makedirs(self.save_folder, exist_ok=True)

        args_dict = vars(deepcopy(self.args))
        for k, v in args_dict.items():
            args_dict[k] = str(v)
        with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
            json.dump(args_dict, file, indent=4, sort_keys=False)
        
        self.weight_dtype = weight_dtype
        self.device = device


        self.StatisticInfo = StatisticInfo
        if self.args.data['val'].get('stat_file', None) is not None:
            with open(self.args.data['val']['stat_file'], "r") as f:
                self.StatisticInfo = json.load(f)

        self.action_norm_type = action_norm_type

    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        print(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        
        self.args.data['val'].update({"fix_epiidx": 0, "fix_sidx":0, "fix_mem_idx":[0,0,0,0]})

        self.val_dataset = val_dataset_class(**self.args.data['val'])
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
        )


    def prepare_models(self,):

        print("Initializing models")
        # Bootstrap accelerate state BEFORE any code path that touches the
        # trainer-shared helpers (`_load_v0c_a_frozen` -> uses
        # `accelerate.logging.get_logger("wm_runner").info(...)`, which
        # raises RuntimeError unless either `PartialState()` or
        # `Accelerator()` has been called first).
        #
        # In the trainer path this is implicit (the Accelerator() constructor
        # runs inside `prepare_for_training`); the inferencer path never
        # builds an Accelerator (no DDP / DeepSpeed / autocast scaler
        # needed), so we initialize the lightweight PartialState singleton
        # explicitly here. It is idempotent: subsequent PartialState() /
        # Accelerator() calls re-use the same registered state.
        #
        # PartialState reads RANK / WORLD_SIZE / LOCAL_RANK / MASTER_PORT
        # from the torchrun-set env vars (scripts/infer_tactile.sh +
        # scripts/eval_action_open_loop_488_bypass_shared.sh both invoke
        # torchrun --nproc_per_node=1 so the env is well-formed).
        from accelerate.state import PartialState
        PartialState()

        device = self.device
        dtype = self.weight_dtype

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
            load_weights=self.args.load_weights
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
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


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
        self.diffusion_model.eval().requires_grad_(False)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')


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

        ### Tactile-aware additions (Stage 3 ckpts) ----------------------
        # The pipeline class above never sees tactile inputs (its `infer`
        # signature has no tactile_* params). For yaml `use_tactile_views=
        # true`, _validate_with_tactile bypasses pipe.infer and runs an
        # in-house denoise loop that needs `self.tactile_vae`, `self.
        # projector`, and a separate `self.scheduler_action` (the action
        # scheduler is independent of the video scheduler in the trainer's
        # design; see custom_pipeline.py:185).
        if getattr(self.args, "use_tactile_views", False):
            tactile_vae_args = getattr(self.args, "tactile_vae", {}) or {}
            projector_args = getattr(self.args, "projector", {}) or {}

            # Frozen v0c-A wrapping the shared visual VAE. We reuse the same
            # helper the trainer uses so the weight loading semantics
            # (strict=False, aux-head tolerance) match byte-for-byte.
            self.tactile_vae = _load_v0c_a_frozen(
                vae=self.vae,
                tactile_vae_config=tactile_vae_args.get("config", {}) or {},
                model_path=tactile_vae_args.get("model_path", ""),
                device=torch.device(device),
                dtype=dtype,
            )
            self.tactile_vae.eval().requires_grad_(False)

            # E_proj_identity bypass mirror (see runner/tactile_dit_trainer.py
            # :1110-1127). When yaml `tactile_vae.config.disable_projector =
            # true`, the trainer constructs + loads the projector for schema
            # stability but skips the projector call in forward, passing
            # `tac_latent_pre` straight to the DiT. The inferencer MUST honor
            # the same flag or the eval would feed a tactile latent
            # distribution the DiT was never trained against (random-init +
            # frozen projector on this run), silently producing wrong action
            # MSE numbers. Drift guard at the end of this block re-derives
            # the flag from yaml independently and asserts they match, so
            # any future code path that mutates the attribute will fail loud.
            self._disable_projector_for_action = bool(
                (tactile_vae_args.get("config", {}) or {}).get("disable_projector", False)
            )
            if self._disable_projector_for_action:
                print(
                    "[tactile_inferencer] [E_proj_identity] disable_projector=true: "
                    "_encode_tactile_split will BYPASS the TactileProjector and pass "
                    "tac_latent_pre straight to the DiT (matches training-time "
                    "behavior). projector.pt is loaded for schema stability but "
                    "never called in forward."
                )

            # Projector: built from yaml hparams, then loaded from the Stage-3
            # ckpt dir (NOT from yaml.projector.warmstart_ckpt, which is the
            # Stage-2 source). `main.py:45` has already rewritten
            # `args.diffusion_model['model_path']` to the Stage-3 step_NNNN
            # dir, so projector.pt sits right next to the DiT safetensors.
            #
            # We DELIBERATELY do NOT call `_load_projector_warmstart` here:
            # that method gates legal source-phases (Stage-2 only when caller
            # is `world_model_only`), which would reject a Stage-3 ckpt
            # loading its own projector. Plain torch.load is correct because
            # the schema is well-known (see `_save_projector_ckpt` docstring).
            self.projector = TactileProjector(
                latent_dim=int(projector_args.get("latent_dim", 128)),
                hidden_dim=int(projector_args.get("hidden_dim", 256)),
                num_views=int(projector_args.get("num_views", 2)),
            ).to(device=device, dtype=torch.float32)

            projector_path = os.path.join(
                self.args.diffusion_model['model_path'], 'projector.pt'
            )
            if not os.path.isfile(projector_path):
                raise FileNotFoundError(
                    f"use_tactile_views=true but {projector_path} not found. "
                    f"Stage-3 ckpts must save projector.pt alongside the DiT "
                    f"safetensors; see trainer._save_projector_ckpt."
                )
            proj_ckpt = torch.load(projector_path, map_location='cpu')
            proj_sd = proj_ckpt['projector'] if isinstance(proj_ckpt, dict) and 'projector' in proj_ckpt else proj_ckpt
            missing, unexpected = self.projector.load_state_dict(proj_sd, strict=False)
            src_phase = proj_ckpt.get('phase', '<unknown>') if isinstance(proj_ckpt, dict) else '<raw>'
            src_step = proj_ckpt.get('step', '<unknown>') if isinstance(proj_ckpt, dict) else '<raw>'
            print(
                f"[tactile_inferencer] projector loaded from {projector_path}: "
                f"source_phase={src_phase!r}, step={src_step}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            self.projector.eval().requires_grad_(False)

            # Action scheduler -- separate instance from self.scheduler. The
            # trainer's pipeline keeps these decoupled because the action and
            # video denoise schedules can differ across training phases (see
            # custom_pipeline.py:185 default arg).
            self.scheduler_action = diffusion_scheduler_class(
                **self.args.diffusion_scheduler_args
            ) if hasattr(self.args, "diffusion_scheduler_args") else diffusion_scheduler_class()

            # R3 drift guard: recompute disable_projector independently from
            # the yaml dict and assert it agrees with
            # self._disable_projector_for_action. This catches "flag set
            # from wrong source / shadowed by an earlier override" -- the
            # specific drift class that would cause silently-wrong evals
            # (eval uses a tactile-latent code path the trainer never saw).
            tactile_vae_args_guard = getattr(self.args, "tactile_vae", {}) or {}
            _yaml_disable = bool(
                (tactile_vae_args_guard.get("config", {}) or {}).get("disable_projector", False)
            )
            assert self._disable_projector_for_action == _yaml_disable, (
                f"_disable_projector_for_action ({self._disable_projector_for_action}) "
                f"is out of sync with yaml tactile_vae.config.disable_projector "
                f"({_yaml_disable}). Refusing to evaluate -- the model would be "
                f"fed a tactile latent distribution different from what the "
                f"trainer saw."
            )

    def validate(
        self, model_save_dir, global_step, n_view=1, n_chunk_video=1, n_chunk_action=10, n_validation=1, domain_name="agibotworld",
    ):

        os.makedirs(model_save_dir,exist_ok=True)

        # Tactile-aware branch (Stage 3 + use_tactile_views=true): tactile
        # views are not a parameter of CustomPipeline.infer (the pipeline
        # signature on custom_pipeline.py:564 has no tactile_* args), so we
        # bypass pipe.infer and run an in-house denoise loop that mirrors
        # the trainer's tactile injection path. The non-tactile branch
        # below stays unchanged (identical to ge_inferencer.validate).
        if getattr(self.args, "use_tactile_views", False):
            return self._validate_with_tactile(
                model_save_dir=model_save_dir,
                global_step=global_step,
                n_view=n_view,
                n_chunk_action=n_chunk_action,
                n_validation=n_validation,
                domain_name=domain_name,
            )

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

        assert(self.args.return_action | self.args.return_video)


        if self.args.return_action:
            n_chunk_video = 1
            action_type = self.args.data["train"]["action_type"]
            action_space = self.args.data["train"]["action_space"]
            

        if self.args.return_video:
            n_chunk_action = 1


        for i_validation in range(n_validation):
            
            self.val_dataloader.dataset.fix_epiidx = i_validation

            if self.args.return_action:
                self.val_dataloader.dataset.fix_sidx = 0
                self.val_dataloader.dataset.fix_mem_idx = [1 for _ in range(self.args.data['train']['n_previous'])]

                pd_actions_arr_all = None
                gt_actions_arr_all = None

            if self.args.return_action:
                # The previous hardcoded plt.subplots(10, 2, figsize=(20, 28))
                # was here. Removed because (a) it raised IndexError on
                # high-dim action vectors (handover: 150 action + 90 state =
                # 240) and (b) the post-loop block below now either delegates
                # to _dump_tactile_diagnostics (when add_state=true) or
                # auto-sizes its own grid -- both compute the figure layout
                # from gt_actions_arr_all.shape[-1], which is only known
                # *after* the chunk loop.
                total_steps = n_chunk_action * self.args.data["train"]["action_chunk"]

            for i_chunk_action in range(n_chunk_action):

                batch = next(iter(self.val_dataloader))
                image = batch['video'][:,:,:,:self.args.data['train']['n_previous']]  # shape b,c,v,t,h,w 
                prompt = batch['caption']
                gt_video = batch['video']

                b, c, v, t, h, w = image.shape

                negative_prompt = ''

                batch_size = 1

                image = image[:batch_size]

                image = rearrange(image, 'b c v t h w -> (b v) c t h w')

                if getattr(self.args, "add_state", False):
                    history_action_state = batch["state"][:batch_size] 
                    if history_action_state.shape[1] > 1:
                        history_action_state = history_action_state[:, self.args.data['train']['n_previous']-1:self.args.data['train']['n_previous'], :]
                    history_action_state = history_action_state.contiguous() ### B, 1, C
                else:
                    history_action_state = None

                preds = pipe.infer(
                    image=image,
                    prompt=prompt[:batch_size],
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.args.num_inference_step,
                    decode_timestep=0.03,
                    decode_noise_scale=0.025,
                    guidance_scale=1.0,
                    height=h,
                    width=w,
                    n_view=v,
                    n_view_visual=v,
                    n_view_tactile=0,
                    return_action=self.args.return_action,
                    n_prev=self.args.data['train']['n_previous'],
                    chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
                    return_video=self.args.return_video,
                    noise_seed=42,
                    action_chunk=self.args.data['train']['action_chunk'],
                    history_action_state = history_action_state,
                    pixel_wise_timestep = self.args.pixel_wise_timestep,
                    n_chunk=n_chunk_video,
                    action_dim=self.args.diffusion_model["config"]["action_in_channels"] if self.args.return_action else None,
                )[0]

                save_cap = f'Validation_{i_validation}'

                if self.args.return_video:
                    
                    video = preds['video'].data.cpu()

                    save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{save_cap}_gt.mp4'), fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)

                    save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], os.path.join(model_save_dir, f'{save_cap}.mp4'), fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)


                if self.args.return_action:

                    gt_actions = batch['actions'][:,self.args.data['train']['n_previous']:]

                    # shape t, c
                    pd_actions_arr = preds['action'][0].data.cpu().to(torch.float).numpy()
                    gt_actions_arr = gt_actions[0].data.cpu().to(torch.float).numpy()

                    # n_dim = pd_actions_arr.shape[-1]
                    
                    if pd_actions_arr_all is None:
                        pd_actions_arr_all = pd_actions_arr
                    else:
                        pd_actions_arr_all = np.concatenate((pd_actions_arr_all, pd_actions_arr), axis=0)
                    
                    if gt_actions_arr_all is None:
                        gt_actions_arr_all = gt_actions_arr
                    else:
                        gt_actions_arr_all = np.concatenate((gt_actions_arr_all, gt_actions_arr), axis=0)


                image = None

                ### prepare for next chunk action prediction
                self.val_dataloader.dataset.fix_sidx += self.args.data['train']['action_chunk']
                self.val_dataloader.dataset.fix_mem_idx = x = (np.linspace(0, self.val_dataloader.dataset.fix_sidx-1, self.args.data['train']['n_previous']).round().astype(np.int16)).tolist()


            if self.args.return_action:
                num_dims = gt_actions_arr_all.shape[-1]

                # Resolve action_only_dim via the same helper the tactile
                # path uses so this branch picks up
                # tactile_inference.action_only_dim_override from the yaml.
                # The override is REQUIRED for corpora where `actions` and
                # `state` are both already action-aligned (e.g.
                # correctaction layout used by handover: actions=240=
                # 150 action + 90 state echo, state=240 with the state
                # block right-padded into the action vector). A naive
                # state.shape[-1] probe degenerates to 240 here and the
                # subtraction yields action_only_dim=0 -- which is exactly
                # what the original tactile path probe guards against in
                # _resolve_action_only_dim. Reusing the helper means the
                # visual-only path and the tactile path produce the same
                # split.
                try:
                    action_only_dim = self._resolve_action_only_dim(batch)
                except (RuntimeError, ValueError) as e:
                    print(
                        f"[tactile_inferencer/legacy] val{i_validation} "
                        f"_resolve_action_only_dim failed ({type(e).__name__}: {e}). "
                        f"Falling back to plot-only (no loss_split.json). "
                        f"To enable per-group MSE, set "
                        f"`tactile_inference.action_only_dim_override` in the yaml."
                    )
                    action_only_dim = None

                if action_only_dim is not None and 0 < action_only_dim < num_dims:
                    self._dump_tactile_diagnostics(
                        pd_all=pd_actions_arr_all,
                        gt_all=gt_actions_arr_all,
                        i_validation=i_validation,
                        action_chunk=self.args.data["train"]["action_chunk"],
                        n_chunk_action=n_chunk_action,
                        action_only_dim=action_only_dim,
                        action_dim=num_dims,
                        model_save_dir=model_save_dir,
                    )
                else:
                    # Fallback: add_state=false (no state in batch) -- can't
                    # split action vs state, so just emit a single auto-sized
                    # grid plot. No loss_split.json dump in this branch.
                    x_axis = np.arange(gt_actions_arr_all.shape[0])
                    n_cols = 8
                    n_rows = (num_dims + n_cols - 1) // n_cols
                    fig, axes = plt.subplots(
                        n_rows, n_cols,
                        figsize=(4 * n_cols, 2.5 * n_rows),
                        sharex=True,
                    )
                    axes = np.atleast_1d(axes).flatten()
                    start_indices = np.arange(0, gt_actions_arr_all.shape[0], self.args.data["train"]["action_chunk"])
                    for dim_idx in range(num_dims):
                        ax = axes[dim_idx]
                        ax.plot(x_axis, gt_actions_arr_all[:, dim_idx], label='Ground Truth', color='cornflowerblue', alpha=0.9)
                        ax.plot(x_axis, pd_actions_arr_all[:, dim_idx], label='Inferred', color='tomato', linestyle='--', alpha=0.9)
                        ax.scatter(start_indices, gt_actions_arr_all[start_indices, dim_idx], c='blue', marker='o', s=40, zorder=5, label='GT Start')
                        ax.scatter(start_indices, pd_actions_arr_all[start_indices, dim_idx], c='darkred', marker='x', s=40, zorder=5, label='Inferred Start')
                        ax.set_title(f"Dimension- {dim_idx}")
                        ax.grid(True, linestyle=':', alpha=0.6)
                    for slot in range(num_dims, len(axes)):
                        axes[slot].axis('off')
                    axes[num_dims - 1].legend(loc='best', fontsize=8)
                    axes[num_dims - 1].set_ylabel('Value')
                    fig.supxlabel(f'Continuous Timestep (across {n_chunk_action} inferences)')
                    plt.tight_layout(rect=[0, 0, 1, 0.98])
                    fig.suptitle(f'Comparison of Ground Truth and Inferred Actions', fontsize=18)
                    plt.savefig(f'{self.save_folder}/openloop_evaluation_val{i_validation}.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)


    # ------------------------------------------------------------------
    # Tactile-aware helpers (Stage 3 path; called only when yaml
    # `use_tactile_views=true`). Kept below the original `validate` so the
    # diff against ge_inferencer is purely additive.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_tactile_split(
        self,
        tactile: torch.Tensor,
        mem_size: int,
        hand_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standalone copy of TactileDiTTrainer._encode_tactile_split.

        Inlined here (rather than re-using the trainer method) so the
        inferencer is not coupled to a TactileDiTTrainer instance: the
        trainer method reads `self.tactile_vae` / `self.projector` /
        `self._disable_projector_for_action` from the trainer's own
        attribute namespace, and the inferencer has those same attribute
        names populated by `prepare_models` above.

        Mirror logic (see runner/tactile_dit_trainer.py:1747-1995):
          - mem path:   per-frame encode (T_in=1 each)  -> mem_size T-slots
          - future path: full LTX temporal compress     -> chunk//8 + 1 slots
          - concat along T -> matches visual T_lat layout exactly.
        Caller is responsible for the `return_action` future-frame-repeat
        trick on `tactile` BEFORE calling this method (mirror
        `tactile_dit_trainer.py:2217-2229` at the call site). The caller
        MUST apply the same future-frame-repeat trick to `hand_pose` in
        v0d mode so the two tensors have a co-aligned T axis.

        v0d pose-injection: when the wrapper has `use_pose_injection=
        True`, the caller MUST pass `hand_pose` of shape
        `(B, V_hand, T_total, P=22)`. Mem and future slices are passed
        to `encode_per_hand(hand_pose=...)` exactly the way the trainer
        does it.

        E_proj_identity bypass: when `self._disable_projector_for_action`
        is True (yaml `tactile_vae.config.disable_projector = true`),
        the projector call is skipped and `tac_latent_pre` is returned
        verbatim, matching trainer behavior at
        `runner/tactile_dit_trainer.py:1985-1991`.

        Args:
            tactile: (B, V_hand, F=5, T_total, H, W) float in [-1,1].
            mem_size: number of memory frames; must match the visual split.
            hand_pose: optional (B, V_hand, T_total, P) per-hand,
                per-frame pose trajectory. Required when the adapter
                has `use_pose_injection=True`; silently dropped
                otherwise (matches trainer line 1860-1864).

        Returns:
            (B, V_hand, 128, T_lat, 6, 8) per-hand latent.
        """
        if self.tactile_vae is None or self.projector is None:
            raise RuntimeError(
                "_encode_tactile_split called but tactile_vae/projector are "
                "None. prepare_models() should have populated them when "
                "use_tactile_views=true."
            )
        if tactile.ndim != 6:
            raise ValueError(
                f"_encode_tactile_split expects 6-D tactile (B,V,F,T,H,W); "
                f"got {tuple(tactile.shape)}."
            )

        B, V_hand, F_finger, T_total, H, W = tactile.shape
        if T_total <= mem_size:
            raise ValueError(
                f"T_total ({T_total}) must be > mem_size ({mem_size})."
            )

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
            # Adapter has no pose head; silently drop (matches trainer
            # line 1860-1864). Logged once below in the Gate-1 block.
            hand_pose = None

        # Memory path: per-frame.
        mem_tac = tactile[:, :, :, :mem_size]
        mem_tac = rearrange(mem_tac, "b vh f m h w -> (b m) vh f h w").unsqueeze(3)
        mem_pose = None
        if hand_pose is not None:
            mem_pose = hand_pose[:, :, :mem_size, :]
            mem_pose = rearrange(
                mem_pose, "b vh m p -> (b m) vh p",
            ).unsqueeze(2)
        mem_lat = self.tactile_vae.encode_per_hand(mem_tac, hand_pose=mem_pose) if use_pose_injection \
            else self.tactile_vae.encode_per_hand(mem_tac)
        mem_lat = rearrange(
            mem_lat, "(b m) vh c t h w -> b vh c (m t) h w", b=B, m=mem_size
        )

        # Future path: full LTX temporal compression.
        future_tac = tactile[:, :, :, mem_size:]
        future_pose = None
        if hand_pose is not None:
            future_pose = hand_pose[:, :, mem_size:, :]
        future_lat = self.tactile_vae.encode_per_hand(future_tac, hand_pose=future_pose) if use_pose_injection \
            else self.tactile_vae.encode_per_hand(future_tac)

        tac_latent_pre = torch.cat([mem_lat, future_lat], dim=3)
        tac_latent_pre = tac_latent_pre.detach()

        # E_proj_identity bypass: mirror trainer 1985-1991. When the
        # yaml disables the projector, return tac_latent_pre verbatim
        # (no view_idx computation, no projector call). Shape is
        # identical because TactileProjector is shape-preserving.
        if getattr(self, "_disable_projector_for_action", False):
            tac_latent = tac_latent_pre
        else:
            view_idx = torch.arange(
                V_hand, device=tac_latent_pre.device, dtype=torch.long
            ).unsqueeze(0).expand(B, V_hand).contiguous()
            tac_latent = self.projector(tac_latent_pre, view_idx)

        # R2: one-shot [v0d-gate1-encode] log proving the bypass /
        # non-bypass branch behaved as expected. Mirrors trainer
        # line 1866-1886 plus tac_latent_pre.shape + tac_latent.shape
        # so when disable_projector=True we can visually confirm the
        # two shapes are identical (alias semantics).
        if not getattr(self, "_v0d_gate1_encode_logged", False):
            mode = "v0d (pose-injection)" if use_pose_injection else "v0c-A"
            hp_shape = tuple(hand_pose.shape) if hand_pose is not None else None
            hp_dtype = hand_pose.dtype if hand_pose is not None else None
            print(
                "[tactile_inferencer] [v0d-gate1-encode] first _encode_tactile_split call:\n"
                f"  mode                  = {mode}\n"
                f"  tactile.shape         = {tuple(tactile.shape)} "
                f"(B={B}, V_hand={V_hand}, F={F_finger}, T_total={T_total}, "
                f"H={H}, W={W})\n"
                f"  hand_pose.shape       = {hp_shape}\n"
                f"  hand_pose.dtype       = {hp_dtype}\n"
                f"  mem_size              = {mem_size}  (chunk={T_total - mem_size})\n"
                f"  tac_latent_pre.shape  = {tuple(tac_latent_pre.shape)}\n"
                f"  tac_latent.shape      = {tuple(tac_latent.shape)}\n"
                f"  disable_projector     = {getattr(self, '_disable_projector_for_action', False)}\n"
                f"  use_pose_injection    = {use_pose_injection}"
            )
            self._v0d_gate1_encode_logged = True

        return tac_latent

    def _resolve_action_only_dim(self, probe_batch: Dict[str, torch.Tensor]) -> int:
        """Resolve `self.action_only_dim` from yaml override or probe batch.

        Precedence (highest first):
          1. yaml `tactile_inference.action_only_dim_override` -- explicit
             integer; bypasses the probe entirely. Useful when state dims
             are absent from the batch (ablation / debug runs).
          2. probe: action_only_dim = actions.shape[-1] - state.shape[-1].
             Production tactile yaml has add_state=true so state is
             always present; we assert action_dim - state_dim > 0 so a
             mis-configured run fails loud rather than silently producing
             garbage action/state split MSE.

        Args:
            probe_batch: a single batch from val_dataloader.
        Returns:
            action_only_dim int in (0, action_dim).
        """
        ti_args = getattr(self.args, "tactile_inference", {}) or {}
        override = ti_args.get("action_only_dim_override", None)
        if override is not None:
            print(
                f"[tactile_inferencer] action_only_dim override active: "
                f"action_only_dim={override}"
            )
            return int(override)

        actions = probe_batch.get("actions", None)
        state = probe_batch.get("state", None)
        if actions is None:
            raise RuntimeError(
                "probe batch has no 'actions' key; cannot derive "
                "action_only_dim. Set tactile_inference.action_only_dim_"
                "override in yaml as a fallback."
            )
        action_dim = int(actions.shape[-1])
        if state is None:
            raise RuntimeError(
                "probe batch has no 'state' key but add_state was presumed "
                "true. Either enable add_state in yaml or set "
                "tactile_inference.action_only_dim_override."
            )
        state_dim = int(state.shape[-1])
        action_only_dim = action_dim - state_dim
        if action_only_dim <= 0 or action_only_dim >= action_dim:
            raise ValueError(
                f"probe-derived action_only_dim={action_only_dim} is "
                f"degenerate (action_dim={action_dim}, state_dim="
                f"{state_dim}). Check your dataset's action/state "
                f"layout (expected action_dim = action_only + state)."
            )
        print(
            f"[tactile_inferencer] action_only_dim derived from probe: "
            f"action_dim={action_dim}, state_dim={state_dim} -> "
            f"action_only_dim={action_only_dim}"
        )
        return action_only_dim

    @torch.no_grad()
    def _validate_with_tactile(
        self,
        model_save_dir: str,
        global_step: int,
        n_view: int,
        n_chunk_action: int,
        n_validation: int,
        domain_name: str,
    ) -> None:
        """In-house denoise loop with tactile-view injection.

        Mirrors CustomPipeline.infer's action-only path (custom_pipeline.py
        :812-913) but:
          - encodes tactile via `_encode_tactile_split` + the trainer's
            return_action future-frame-repeat trick (tactile_dit_trainer.py
            :1568-1576);
          - cats tactile into the mem latents along the batch axis BEFORE
            `gen_noise_from_condition_frame_latent`, so n_view bumps to
            V_rgb + V_hand exactly as during training;
          - skips the video decode entirely (return_action only);
          - dumps action-vs-state split MSE (`loss_split.json`) plus two
            openloop PNGs so the operator can eyeball whether the
            loss=0.3 plateau is state-dragged or genuinely stuck.

        No DeepSpeed / accelerate / DDP -- single GPU. guidance_scale=1.0
        (no CFG; matches Stage-3 training).
        """
        if not (self.args.return_action and not self.args.return_video):
            raise NotImplementedError(
                "Tactile-aware inferencer currently supports action-only mode "
                "(return_action=True, return_video=False). Got "
                f"return_action={self.args.return_action}, "
                f"return_video={self.args.return_video}."
            )

        device = torch.device(self.device)
        dtype = self.weight_dtype

        stat_file = self.args.data['val'].get('stat_file', None)
        joint_key = f"{domain_name}_joint"
        state_key = f"{domain_name}_state_joint"
        print("[action-norm]")
        print(f"  domain_name            = {domain_name}")
        print(f"  stat_file              = {stat_file}")
        try:
            _stat_keys = list(self.StatisticInfo.keys())
        except Exception as _e:
            _stat_keys = f"<err: {_e}>"
        print(f"  available stat keys    = {_stat_keys}")
        for k in [joint_key, state_key]:
            if k in self.StatisticInfo:
                s = self.StatisticInfo[k]
                print(f"  {k}: fields = {list(s.keys())}")
                for stat_name in ('q01', 'q99', 'mean', 'std'):
                    v = s.get(stat_name, None)
                    if v is not None:
                        print(f"    {stat_name}[:8] = "
                              f"{[round(float(x), 4) for x in v[:8]]}  "
                              f"(len={len(v)})")
            else:
                print(f"  {k}: MISSING from stats file")

        train_cfg = self.args.data['train']
        mem_size = int(train_cfg['n_previous'])
        chunk = int(train_cfg['chunk'])
        action_chunk = int(train_cfg['action_chunk'])
        sample_h, sample_w = train_cfg['sample_size']
        # latent_frames matches trainer's tactile_dit_trainer.py:1508
        # (chunk frames -> chunk//TEMPORAL_DOWN_RATIO + 1 latent slots).
        latent_frames = chunk // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
        latent_height = sample_h // self.SPATIAL_DOWN_RATIO
        latent_width = sample_w // self.SPATIAL_DOWN_RATIO
        action_dim = int(self.args.diffusion_model['config']['action_in_channels'])
        num_inference_steps = int(getattr(self.args, "num_inference_step", 10))
        noise_seed = int(getattr(self.args, "seed", 42))

        # Encode the unconditional ("") prompt ONCE (no CFG, but the DiT
        # still expects an encoder_hidden_states tensor). The prompt is
        # taken from the batch's 'caption' field per-chunk below.
        text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        uncond_embed = text_uncond['prompt_embeds']
        uncond_mask = text_uncond['prompt_attention_mask']

        # rope_interpolation_scale mirrors custom_pipeline.py:804-809.
        frame_rate = 30
        latent_frame_rate = frame_rate / self.TEMPORAL_DOWN_RATIO
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            self.SPATIAL_DOWN_RATIO,
            self.SPATIAL_DOWN_RATIO,
        )

        # Resolve action_only_dim once before the validation loop. We probe
        # the FIRST possible batch (epi 0, sidx 0) without disturbing the
        # later per-validation reset of fix_sidx / fix_mem_idx.
        self.val_dataloader.dataset.fix_epiidx = 0
        self.val_dataloader.dataset.fix_sidx = 0
        self.val_dataloader.dataset.fix_mem_idx = [1 for _ in range(mem_size)]
        probe_batch = next(iter(self.val_dataloader))
        self.action_only_dim = self._resolve_action_only_dim(probe_batch)
        action_only_dim = self.action_only_dim
        assert action_dim == probe_batch['actions'].shape[-1], (
            f"yaml action_in_channels={action_dim} disagrees with dataset "
            f"actions.shape[-1]={probe_batch['actions'].shape[-1]}; refusing "
            f"to run with split MSE on mismatched layout."
        )

        loss_cfg = getattr(self.args, "loss", {}) or {}
        try:
            _loss_dim_breakdown = loss_cfg.get('action_dim_breakdown', None)
        except AttributeError:
            _loss_dim_breakdown = getattr(loss_cfg, 'action_dim_breakdown', None)
        print("[action-layout]")
        print(f"  action_in_channels (yaml)      = {action_dim}")
        print(f"  action_only_dim (resolved)     = {action_only_dim}")
        print(f"  action_dim_breakdown (yaml)    = {_loss_dim_breakdown}")
        print(f"  action_chunk (yaml)            = {action_chunk}")
        print(f"  state_key / action_key (yaml)  = "
              f"{self.args.data['val'].get('state_key', None)} / "
              f"{self.args.data['val'].get('action_key', None)}")
        print(f"  probe_batch['actions'].shape   = {tuple(probe_batch['actions'].shape)}")
        print(f"  probe_batch['state'].shape     = "
              f"{tuple(probe_batch['state'].shape) if 'state' in probe_batch else None}")
        print(f"  self.action_norm_type (dead?)  = {self.action_norm_type}")

        scheduler_dtype = uncond_embed.dtype

        for i_validation in range(n_validation):
            self.val_dataloader.dataset.fix_epiidx = i_validation
            self.val_dataloader.dataset.fix_sidx = 0
            self.val_dataloader.dataset.fix_mem_idx = [1 for _ in range(mem_size)]

            pd_actions_arr_all = None
            gt_actions_arr_all = None

            for i_chunk_action in range(n_chunk_action):
                cur_sidx = self.val_dataloader.dataset.fix_sidx
                cur_mem_idx = list(self.val_dataloader.dataset.fix_mem_idx)
                print(
                    f"[tactile_inferencer] val{i_validation} "
                    f"chunk{i_chunk_action}: fix_sidx={cur_sidx}, "
                    f"fix_mem_idx={cur_mem_idx}"
                )

                # NOTE: each next(iter(...)) re-triggers __getitem__, which
                # reads the mutated fix_sidx (data/dex_vtam_dataset.py:394-
                # 400). Replacing this with `for batch in val_dataloader`
                # would break chunk advancement -- see plan discussion.
                batch = next(iter(self.val_dataloader))

                # ---- Visual mem encode (per-frame; same as get_latents
                #      mem path, but mem-only since action-mode discards
                #      future video in favor of pure noise) ----
                image = batch['video'][:, :, :, :mem_size].to(device, dtype=dtype)
                # image: (B, C, V_rgb, mem, H, W) -> (B*V_rgb, C, mem, H, W)
                image = rearrange(image, 'b c v t h w -> (b v) c t h w')
                batch_size = image.shape[0] // n_view  # =1 in production (val batch=1)

                video_generator = torch.Generator(device=device).manual_seed(noise_seed)
                # Per-frame VAE encode: matches utils.data_utils.get_latents
                # mem path so the latent ordering is bit-equivalent to the
                # trainer's view.
                image_perframe = rearrange(image, 'bv c t h w -> (bv t) c h w').unsqueeze(2)
                init_latents = self.vae.encode(image_perframe).latent_dist.sample(generator=video_generator)
                init_latents = init_latents.to(dtype=dtype)
                init_latents = _normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
                # (B*V_rgb*mem, C, 1, h, w) -> (B*V_rgb, C, mem, h, w)
                init_latents = rearrange(
                    init_latents, '(bv t) c f h w -> bv c (t f) h w', t=mem_size
                )

                # ---- Tactile mem encode (mirrors training-time path) ----
                tactile = batch['tactile'].to(device, dtype=dtype).contiguous()
                n_view_tactile = tactile.shape[1]

                # v0d hand_pose pickup (mirrors trainer line 2189-2214).
                # Gated on the adapter's flag (NOT the yaml) because the
                # _load_v0c_a_frozen path already enforced they agree.
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
                            "`pose_stats_path` in the dataset val config, or "
                            "disable `adapter_use_pose_injection` in the "
                            "tactile_vae config."
                        )
                    hand_pose_t = batch['hand_pose'].to(
                        device, dtype=dtype,
                    ).contiguous()
                    if hand_pose_t.ndim != 4:
                        raise ValueError(
                            f"batch['hand_pose'] expected (B, V_hand=2, "
                            f"T_total, 22); got {tuple(hand_pose_t.shape)}."
                        )

                # Trainer's future-frame-repeat trick (tactile_dit_trainer
                # .py:2217-2229): in action mode, the dataset's real future
                # tactile is discarded and the 1st future frame is repeated
                # `chunk` times so the tactile latent T_lat matches the
                # visual latent T_lat exactly. The same repeat MUST be
                # applied to `hand_pose_t` so the two tensors keep a
                # co-aligned T axis -- the v0d adapter's
                # `_align_pose_to_lat` consumes them per-slice (mem
                # vs future) and would silently mis-align otherwise.
                tac_mem_raw = tactile[:, :, :, :mem_size]
                tac_fut_raw = tactile[:, :, :, mem_size:mem_size + 1].repeat(
                    1, 1, 1, chunk, 1, 1
                )
                tactile_synth = torch.cat([tac_mem_raw, tac_fut_raw], dim=3).contiguous()
                if hand_pose_t is not None:
                    hp_mem = hand_pose_t[:, :, :mem_size, :]
                    hp_fut = hand_pose_t[:, :, mem_size:mem_size + 1, :].repeat(
                        1, 1, chunk, 1,
                    )
                    hand_pose_t = torch.cat([hp_mem, hp_fut], dim=2).contiguous()
                    # R1: hand_pose T MUST equal tactile T after the
                    # repeat (= mem_size + chunk). If a future change
                    # ever drops one of the two repeats, this assert
                    # would fire instead of silently shipping mis-aligned
                    # latents to the DiT.
                    assert tactile_synth.shape[3] == hand_pose_t.shape[2], (
                        f"tactile T={tactile_synth.shape[3]} != hand_pose "
                        f"T={hand_pose_t.shape[2]}; future-frame-repeat "
                        f"trick was applied inconsistently between the two."
                    )

                tac_full = self._encode_tactile_split(
                    tactile_synth, mem_size, hand_pose=hand_pose_t,
                )
                # (B, V_hand, C, T_lat, 6, 8) -> (B*V_hand, C, T_lat, 6, 8)
                tac_full = rearrange(tac_full, 'b v c f h w -> (b v) c f h w')
                # Only the mem prefix participates in the conditioning-mask
                # path (gen_noise_from_condition_frame_latent uses the last
                # mem frame for init, noises everything else); tac_full's
                # future slots are overwritten by random noise anyway when
                # noisy_video=true (trainer line 1669-1670, equivalent in
                # inference's pure-noise init), so we cat only tac_mem.
                tac_mem = tac_full[:, :, :mem_size]

                # ---- View-axis cat: visual rows FIRST, tactile rows LAST
                #      (matches trainer line 1567-1568; preserves the row
                #       ordering the action expert was trained against). ----
                mem_latents_all = torch.cat([init_latents, tac_mem], dim=0)
                n_view_total = n_view + n_view_tactile

                # ---- Build noisy latents via gen_noise (mem clean +
                #      future noise; equivalent to trainer's noisy_latents
                #      when ss=1.0). ----
                latents, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                    mem_latents_all, latent_frames, latent_height, latent_width,
                    generator=video_generator, noise_to_condition_frames=0,
                )

                # First-chunk debug print of all relevant shapes (per
                # plan refinement #3); cheap one-time sanity check.
                if i_validation == 0 and i_chunk_action == 0:
                    print(
                        f"[tactile_inferencer] shapes: video={tuple(batch['video'].shape)}, "
                        f"tactile={tuple(batch['tactile'].shape)}, "
                        f"init_latents={tuple(init_latents.shape)}, "
                        f"tac_full={tuple(tac_full.shape)}, "
                        f"tac_mem={tuple(tac_mem.shape)}, "
                        f"mem_latents_all={tuple(mem_latents_all.shape)}, "
                        f"latents={tuple(latents.shape)}, "
                        f"conditioning_mask={tuple(conditioning_mask.shape)}, "
                        f"n_view={n_view_total} (visual={n_view}, tactile={n_view_tactile})"
                    )
                    assert n_view_total == n_view + n_view_tactile

                # ---- Text conditioning per-chunk (caption may differ per
                #      sample / episode, but in practice the dataset's
                #      caption is static). ----
                prompt = batch['caption']
                if isinstance(prompt, (list, tuple)):
                    prompt = list(prompt)[:batch_size]
                text_cond = get_text_conditions(self.tokenizer, self.text_encoder, prompt=prompt)
                prompt_embeds = text_cond['prompt_embeds'].to(device, dtype=scheduler_dtype)
                prompt_attention_mask = text_cond['prompt_attention_mask']

                # ---- history_action_state (state at the boundary frame),
                #      mirrors ge_inferencer.validate lines 253-259. ----
                if getattr(self.args, "add_state", False):
                    history_action_state = batch['state'][:batch_size]
                    if history_action_state.shape[1] > 1:
                        history_action_state = history_action_state[
                            :, mem_size - 1:mem_size, :
                        ]
                    history_action_state = history_action_state.contiguous().to(
                        device=device, dtype=scheduler_dtype
                    )
                else:
                    history_action_state = None

                # ---- Action noise init ----
                action_generator = torch.Generator(device=device).manual_seed(noise_seed)
                actions = randn_tensor(
                    (batch_size, action_chunk, action_dim),
                    device=device, dtype=scheduler_dtype,
                    generator=action_generator,
                )

                if i_validation == 0 and i_chunk_action == 0:
                    print(f"[action-eval] chunk0 init: "
                          f"actions.shape={tuple(actions.shape)}, "
                          f"amin={actions.amin().item():.4f}, "
                          f"amax={actions.amax().item():.4f}, "
                          f"mean={actions.mean().item():.4f}, "
                          f"std={actions.std().item():.4f}")

                # Snapshot the initial noise sample and the NORMALIZED clean
                # GT for this chunk. These are the two inputs the trainer
                # uses at every sigma to synthesize
                #     noisy_train = (1 - sigma) * clean + sigma * eps
                #     target_v   = eps - clean             (FlowMatch v-pred)
                # We reuse them inside the denoise loop to run a training-
                # equivalent local-loss probe at sigma in {1.0, 0.5, 0.1}.
                # See tactile_dit_trainer.py:2561-2572,2898.
                eps_init = actions.clone()
                gt_clean_norm = batch['actions'][:, mem_size:].to(
                    device=device, dtype=scheduler_dtype,
                ).contiguous()

                # ---- Timesteps (mirror custom_pipeline.py:771-795) ----
                video_sequence_length = latent_frames * latent_height * latent_width
                sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                mu = calculate_shift(
                    video_sequence_length,
                    self.scheduler.config.base_image_seq_len,
                    self.scheduler.config.max_image_seq_len,
                    self.scheduler.config.base_shift,
                    self.scheduler.config.max_shift,
                )
                timesteps, _ = retrieve_timesteps(
                    self.scheduler, num_inference_steps, device, None,
                    sigmas=sigmas, mu=mu,
                )
                _, _ = retrieve_timesteps(
                    self.scheduler_action, num_inference_steps, device, None,
                    sigmas=sigmas, mu=mu,
                )

                if i_validation == 0 and i_chunk_action == 0:
                    print("[scheduler-debug]")
                    print(f"  num_inference_steps        = {num_inference_steps}")
                    print(f"  video_sequence_length      = {video_sequence_length}")
                    print(f"  mu (computed)              = {mu:.6f}")
                    print(f"  user sigmas (pre-shift)    = "
                          f"{[round(float(s), 6) for s in sigmas]}")
                    print(f"  self.scheduler.sigmas        = "
                          f"{[round(float(s), 6) for s in self.scheduler.sigmas.cpu().tolist()]}")
                    print(f"  self.scheduler.timesteps     = "
                          f"{[round(float(t), 4) for t in self.scheduler.timesteps.cpu().tolist()]}")
                    print(f"  self.scheduler.config.final_sigmas_type = "
                          f"{getattr(self.scheduler.config, 'final_sigmas_type', None)!r}")
                    print(f"  self.scheduler_action.sigmas    = "
                          f"{[round(float(s), 6) for s in self.scheduler_action.sigmas.cpu().tolist()]}")
                    print(f"  self.scheduler_action.timesteps = "
                          f"{[round(float(t), 4) for t in self.scheduler_action.timesteps.cpu().tolist()]}")
                    print(f"  self.scheduler_action.config.final_sigmas_type = "
                          f"{getattr(self.scheduler_action.config, 'final_sigmas_type', None)!r}")
                    print(f"  inference scheduler.sigmas len        = "
                          f"{len(self.scheduler.sigmas)}")
                    print(f"  inference scheduler_action.sigmas len = "
                          f"{len(self.scheduler_action.sigmas)}")

                # ---- Denoise loop (no CFG, return_video=False) ----
                # We MUST mirror custom_pipeline.py:901,922-923,976-977,984-985
                # for the video-states cache: on step 0 we compute the WM body
                # forward (return_video=True) AND store its per-block hidden
                # states into `video_states_buffer`; on step 1+ we pass that
                # buffer back and the WM body is SKIPPED (transformer reuses
                # the cached hidden states block-by-block, see
                # transformer_ltx_multiview.py:818). Without this, step 1+
                # hits the `assert store_buffer or return_video` at line 780
                # (return_action=True + video_states_buffer is None +
                # return_video=False is illegal).
                latents = latents.to(scheduler_dtype)
                video_states_buffer = None
                for i_step, t in enumerate(timesteps):
                    latent_model_input = latents.clone()
                    action_timesteps = t.unsqueeze(-1).repeat(
                        actions.shape[0], actions.shape[1]
                    )

                    # pixel_wise_timestep True (production yaml line 194)
                    timestep_per_pixel = t.expand(latent_model_input.shape[0])
                    if getattr(self.args, "pixel_wise_timestep", True):
                        timestep_per_pixel = timestep_per_pixel.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        timestep_per_pixel = timestep_per_pixel.unsqueeze(-1) * (1 - cond_indicator)

                    # In action-only mode (outer return_video=False), step 0
                    # runs the WM body + caches; step 1+ skips the WM body
                    # entirely and reads from the cache.
                    compute_video = (i_step == 0)
                    store_buffer = (i_step == 0)

                    noise_pred = self.diffusion_model(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep_per_pixel,
                        encoder_attention_mask=prompt_attention_mask,
                        num_frames=latent_frames,
                        height=latent_height,
                        width=latent_width,
                        rope_interpolation_scale=rope_interpolation_scale,
                        return_dict=False,
                        action_states=actions.to(scheduler_dtype),
                        action_timestep=action_timesteps,
                        return_video=compute_video,
                        return_action=True,
                        n_view=n_view_total,
                        n_view_visual=n_view,
                        video_states_buffer=video_states_buffer,
                        store_buffer=store_buffer,
                        video_attention_mask=None,
                        history_action_state=history_action_state,
                        condition_mask=conditioning_mask,
                    )[0]

                    if store_buffer:
                        video_states_buffer = noise_pred["video_states_buffer"]

                    action_noise_pred = noise_pred['action'].float()

                    # Multi-sigma diagnostic block. Fires at the first, middle
                    # and last denoise step so we can see the WHOLE sensitivity
                    # curve and decisively localize the train-vs-infer gap.
                    #
                    # (A) sensitivity-probe: perturb the rollout x_t by a small
                    #     delta and measure ||pred_diff||/||delta|| + cosine.
                    #     Analytical reference for healthy v=noise-data with
                    #     Var(data) ~ 0.2:
                    #         sigma=1.0 -> sensitivity ~ 1.00, cos ~ +1.0
                    #         sigma=0.5 -> sensitivity ~ 1.33, cos ~ +1.0
                    #         sigma=0.1 -> sensitivity ~ 0.47, cos ~ -1.0
                    #     (cosine sign flips at low sigma because the optimal
                    #      slope of pred w.r.t. x_t turns negative).
                    #
                    # (B) local-loss probe: synthesize the EXACT input the
                    #     trainer would build at this sigma,
                    #         noisy_train = (1 - sigma) * clean + sigma * eps,
                    #     run one model forward (reusing the cached video
                    #     buffer), and compute MSE against every plausible
                    #     target convention. The one whose MSE matches the
                    #     reported training loss (~0.0135 at step 20k) IS the
                    #     convention the model has actually learned.
                    probe_steps = (
                        0,
                        num_inference_steps // 2,
                        num_inference_steps - 1,
                    )
                    if (
                        i_validation == 0
                        and i_chunk_action == 0
                        and i_step in probe_steps
                    ):
                        sigma_now = float(self.scheduler_action.sigmas[i_step])

                        # ---- (A) Rollout-x_t sensitivity probe ----
                        delta = (0.1 * torch.randn_like(actions)).to(scheduler_dtype)
                        actions_perturbed = (actions + delta).to(scheduler_dtype)
                        noise_pred_probe = self.diffusion_model(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep_per_pixel,
                            encoder_attention_mask=prompt_attention_mask,
                            num_frames=latent_frames,
                            height=latent_height,
                            width=latent_width,
                            rope_interpolation_scale=rope_interpolation_scale,
                            return_dict=False,
                            action_states=actions_perturbed,
                            action_timestep=action_timesteps,
                            return_video=False,
                            return_action=True,
                            n_view=n_view_total,
                            n_view_visual=n_view,
                            video_states_buffer=video_states_buffer,
                            store_buffer=False,
                            video_attention_mask=None,
                            history_action_state=history_action_state,
                            condition_mask=conditioning_mask,
                        )[0]
                        pred_probe = noise_pred_probe['action'].float()
                        diff = pred_probe - action_noise_pred
                        delta_f = delta.float()
                        delta_norm = delta_f.norm()
                        diff_norm = diff.norm()
                        sensitivity = (diff_norm / (delta_norm + 1e-8)).item()
                        cos_sd = ((diff.flatten() * delta_f.flatten()).sum()
                                  / (diff_norm * delta_norm + 1e-8)).item()
                        print(
                            f"[sensitivity-probe] i_step={i_step} "
                            f"sigma={sigma_now:.3f}: sensitivity={sensitivity:.4f}, "
                            f"cosine(pred_diff, delta)={cos_sd:.4f}, "
                            f"||delta||={delta_norm.item():.3f}, "
                            f"||pred_diff||={diff_norm.item():.3f}"
                        )

                        # ---- (B) Training-equivalent local-loss probe ----
                        noisy_train = (
                            (1.0 - sigma_now) * gt_clean_norm
                            + sigma_now * eps_init
                        ).to(scheduler_dtype)
                        noise_pred_train = self.diffusion_model(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep_per_pixel,
                            encoder_attention_mask=prompt_attention_mask,
                            num_frames=latent_frames,
                            height=latent_height,
                            width=latent_width,
                            rope_interpolation_scale=rope_interpolation_scale,
                            return_dict=False,
                            action_states=noisy_train,
                            action_timestep=action_timesteps,
                            return_video=False,
                            return_action=True,
                            n_view=n_view_total,
                            n_view_visual=n_view,
                            video_states_buffer=video_states_buffer,
                            store_buffer=False,
                            video_attention_mask=None,
                            history_action_state=history_action_state,
                            condition_mask=conditioning_mask,
                        )[0]
                        pred_train = noise_pred_train['action'].float()
                        eps_f = eps_init.float()
                        clean_f = gt_clean_norm.float()
                        noisy_train_f = noisy_train.float()
                        candidate_targets = [
                            ("v = noise - clean  (TRAINING)",
                             eps_f - clean_f),
                            ("v_neg = clean - noise (sign flip)",
                             clean_f - eps_f),
                            ("x_t - clean",
                             noisy_train_f - clean_f),
                            ("clean - x_t",
                             clean_f - noisy_train_f),
                            ("clean  (x0-pred)",
                             clean_f),
                            ("noise  (eps-pred)",
                             eps_f),
                        ]
                        pred_norm = pred_train.norm().item()
                        print(
                            f"[local-loss] i_step={i_step} "
                            f"sigma={sigma_now:.3f}: ||pred||={pred_norm:.3f}, "
                            f"pred(amin/amax/mean/std)="
                            f"{pred_train.amin().item():.3f}/"
                            f"{pred_train.amax().item():.3f}/"
                            f"{pred_train.mean().item():.3f}/"
                            f"{pred_train.std().item():.3f}"
                        )
                        for name, tgt in candidate_targets:
                            mse = (pred_train - tgt).pow(2).mean().item()
                            tnorm = tgt.norm().item()
                            cos_pt = (
                                (pred_train.flatten() * tgt.flatten()).sum()
                                / (pred_train.norm() * tgt.norm() + 1e-8)
                            ).item()
                            print(
                                f"[local-loss]   target='{name}': "
                                f"MSE={mse:.4f}, ||target||={tnorm:.3f}, "
                                f"cos(pred,target)={cos_pt:+.4f}"
                            )

                    actions = self.scheduler_action.step(
                        action_noise_pred, t, actions, return_dict=False
                    )[0]

                    if (
                        i_validation == 0
                        and i_chunk_action == 0
                        and i_step in (
                            0,
                            num_inference_steps // 2,
                            num_inference_steps - 1,
                        )
                    ):
                        print(
                            f"[action-eval] chunk0 step{i_step}: "
                            f"t={float(t):.4f}, "
                            f"pred(amin/amax/mean/std)="
                            f"{action_noise_pred.amin().item():.4f}/"
                            f"{action_noise_pred.amax().item():.4f}/"
                            f"{action_noise_pred.mean().item():.4f}/"
                            f"{action_noise_pred.std().item():.4f}, "
                            f"actions_after_step(amin/amax/mean/std)="
                            f"{actions.amin().item():.4f}/"
                            f"{actions.amax().item():.4f}/"
                            f"{actions.mean().item():.4f}/"
                            f"{actions.std().item():.4f}"
                        )

                # ---- Collect predicted vs GT actions for this chunk ----
                pd_actions_arr = actions[0].detach().cpu().float().numpy()
                gt_actions = batch['actions'][:, mem_size:]
                gt_actions_arr = gt_actions[0].detach().cpu().float().numpy()

                if i_validation == 0 and i_chunk_action == 0:
                    pd_arr = pd_actions_arr
                    gt_arr = gt_actions_arr
                    D = pd_arr.shape[-1]
                    assert gt_arr.shape[-1] == D, (
                        f"pd dim {D} != gt dim {gt_arr.shape[-1]} -- "
                        f"layout mismatch between model output and dataset GT."
                    )
                    print(
                        f"[action-eval] chunk0 final: dim={D}, "
                        f"pd(amin/amax/mean/std)="
                        f"{pd_arr.min():.4f}/{pd_arr.max():.4f}/"
                        f"{pd_arr.mean():.4f}/{pd_arr.std():.4f}, "
                        f"gt(amin/amax/mean/std)="
                        f"{gt_arr.min():.4f}/{gt_arr.max():.4f}/"
                        f"{gt_arr.mean():.4f}/{gt_arr.std():.4f}"
                    )
                    sq = (pd_arr - gt_arr) ** 2
                    if D >= 194:
                        print(
                            f"[action-eval] chunk0 per-block MSE: "
                            f"force[0:60]={sq[:, 0:60].mean():.4f}, "
                            f"arm[60:92]={sq[:, 60:92].mean():.4f}, "
                            f"hand[92:136]={sq[:, 92:136].mean():.4f}, "
                            f"state[136:194]={sq[:, 136:194].mean():.4f}"
                        )
                    elif D >= 136:
                        print(
                            f"[action-eval] chunk0 per-block MSE "
                            f"(action-only D={D}): "
                            f"force[0:60]={sq[:, 0:60].mean():.4f}, "
                            f"arm[60:92]={sq[:, 60:92].mean():.4f}, "
                            f"hand[92:136]={sq[:, 92:136].mean():.4f}"
                        )
                    else:
                        print(
                            f"[action-eval] chunk0 per-block MSE: "
                            f"total[0:{D}]={sq.mean():.4f}"
                        )

                if pd_actions_arr_all is None:
                    pd_actions_arr_all = pd_actions_arr
                    gt_actions_arr_all = gt_actions_arr
                else:
                    pd_actions_arr_all = np.concatenate(
                        (pd_actions_arr_all, pd_actions_arr), axis=0
                    )
                    gt_actions_arr_all = np.concatenate(
                        (gt_actions_arr_all, gt_actions_arr), axis=0
                    )

                # Advance dataset state for the next chunk (mirrors
                # ge_inferencer.validate:319-320).
                self.val_dataloader.dataset.fix_sidx += action_chunk
                self.val_dataloader.dataset.fix_mem_idx = (
                    np.linspace(
                        0, self.val_dataloader.dataset.fix_sidx - 1, mem_size
                    ).round().astype(np.int16)
                ).tolist()

            # End of episode: dump diagnostics.
            self._dump_tactile_diagnostics(
                pd_all=pd_actions_arr_all,
                gt_all=gt_actions_arr_all,
                i_validation=i_validation,
                action_chunk=action_chunk,
                n_chunk_action=n_chunk_action,
                action_only_dim=action_only_dim,
                action_dim=action_dim,
                model_save_dir=model_save_dir,
            )

    def _dump_tactile_diagnostics(
        self,
        pd_all: np.ndarray,
        gt_all: np.ndarray,
        i_validation: int,
        action_chunk: int,
        n_chunk_action: int,
        action_only_dim: int,
        action_dim: int,
        model_save_dir: str,
    ) -> None:
        """Compute split MSE + dump preds/gts/JSON + two openloop PNGs."""
        val_dir = os.path.join(model_save_dir, f"val{i_validation}")
        os.makedirs(val_dir, exist_ok=True)

        # ---- Save raw arrays for downstream tooling. ----
        np.save(os.path.join(val_dir, "preds.npy"), pd_all)
        np.save(os.path.join(val_dir, "gts.npy"), gt_all)

        # ---- Split MSE (action dims vs state dims). ----
        sq_err = (pd_all - gt_all).astype(np.float64) ** 2
        per_dim_mse = sq_err.mean(axis=0)  # (action_dim,)
        mse_all = float(per_dim_mse.mean())
        mse_action_dims = float(per_dim_mse[:action_only_dim].mean())
        mse_state_dims = float(per_dim_mse[action_only_dim:].mean())

        # Top-10 worst dims by per-dim MSE.
        worst_idx = np.argsort(-per_dim_mse)[:10]
        top_10_worst = [
            {
                "dim": int(d),
                "mse": float(per_dim_mse[d]),
                "group": "action" if d < action_only_dim else "state",
            }
            for d in worst_idx
        ]

        loss_split = {
            "n_total_dims": int(action_dim),
            "action_dims": [0, int(action_only_dim)],
            "state_dims": [int(action_only_dim), int(action_dim)],
            "mse_all": mse_all,
            "mse_action_dims": mse_action_dims,
            "mse_state_dims": mse_state_dims,
            "top_10_worst_dims": top_10_worst,
            "n_chunks": int(n_chunk_action),
            "action_chunk": int(action_chunk),
            "n_steps_collected": int(pd_all.shape[0]),
        }
        with open(os.path.join(val_dir, "loss_split.json"), "w") as f:
            json.dump(loss_split, f, indent=4, sort_keys=False)

        print(
            f"[tactile_inferencer] val{i_validation} MSE summary: "
            f"all={mse_all:.4f}, action[0:{action_only_dim}]={mse_action_dims:.4f}, "
            f"state[{action_only_dim}:{action_dim}]={mse_state_dims:.4f}"
        )
        print(f"[tactile_inferencer] val{i_validation} top-10 worst dims:")
        for entry in top_10_worst:
            print(
                f"    dim={entry['dim']:>3d} ({entry['group']}): "
                f"mse={entry['mse']:.4f}"
            )

        # ---- Two openloop PNGs (action vs state). Reuses the ge_inferencer
        #      plotting style (GT blue solid, Pred red dashed, chunk-start
        #      markers). ----
        self._plot_openloop_grid(
            pd_all=pd_all,
            gt_all=gt_all,
            dim_range=(0, action_only_dim),
            title=f"Validation {i_validation}: action dims [0:{action_only_dim}]",
            n_chunk_action=n_chunk_action,
            action_chunk=action_chunk,
            save_path=os.path.join(val_dir, "openloop_action_dims.png"),
        )
        self._plot_openloop_grid(
            pd_all=pd_all,
            gt_all=gt_all,
            dim_range=(action_only_dim, action_dim),
            title=f"Validation {i_validation}: state dims [{action_only_dim}:{action_dim}]",
            n_chunk_action=n_chunk_action,
            action_chunk=action_chunk,
            save_path=os.path.join(val_dir, "openloop_state_dims.png"),
        )

    def _plot_openloop_grid(
        self,
        pd_all: np.ndarray,
        gt_all: np.ndarray,
        dim_range,
        title: str,
        n_chunk_action: int,
        action_chunk: int,
        save_path: str,
    ) -> None:
        """One PNG grid per dim group (action / state). Auto-sizes the grid
        so we never get an out-of-bounds axis on the smaller (state) group.
        """
        d_lo, d_hi = dim_range
        n_dims = d_hi - d_lo
        if n_dims <= 0:
            return
        n_cols = 8
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), sharex=True
        )
        axes = np.atleast_1d(axes).flatten()
        x_axis = np.arange(gt_all.shape[0])
        start_indices = np.arange(0, gt_all.shape[0], action_chunk)

        for slot, d in enumerate(range(d_lo, d_hi)):
            ax = axes[slot]
            ax.plot(x_axis, gt_all[:, d], color='cornflowerblue', alpha=0.9, label='GT')
            ax.plot(x_axis, pd_all[:, d], color='tomato', linestyle='--', alpha=0.9, label='Pred')
            ax.scatter(start_indices, gt_all[start_indices, d], c='blue', marker='o', s=20, zorder=5)
            ax.scatter(start_indices, pd_all[start_indices, d], c='darkred', marker='x', s=20, zorder=5)
            ax.set_title(f"dim {d}", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
        for slot in range(n_dims, len(axes)):
            axes[slot].axis('off')
        fig.suptitle(title, fontsize=14)
        fig.supxlabel(f"Continuous timestep (across {n_chunk_action} chunks)")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)


    def infer(self, n_chunk_action=4, n_chunk_video=1, n_validation=10, global_step=0, domain_name="agibotworld"):
        model_save_dir = os.path.join(self.save_folder,f'Inference')
        self.validate(
            model_save_dir, global_step,
            n_view=len(self.args.data["train"]["valid_cam"]),
            n_chunk_video=n_chunk_video,
            n_chunk_action=n_chunk_action,
            n_validation=n_validation,
            domain_name=domain_name,
        )
