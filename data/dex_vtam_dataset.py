# This file is an adaptation of Genie-Envisioner (AgibotTech) code that upstream
# licenses under CC BY-NC-SA 4.0, so the ShareAlike term applies and this file is
# distributed under the same licence rather than the repository's Apache 2.0:
# see LICENSES/CC-BY-NC-SA-4.0.txt. NonCommercial use only.
#
# Modified by the DexTacWAM Authors, 2026.


import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import traceback
import json
import random
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from PIL import Image

# from data.utils.domain_table import DomainTable
from data.utils.statistics import StatisticInfo
# from data.utils.get_actions import parse_h5

from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform
# Reused so Stage 2 dataset preprocessing of the LeRobot 'tactile' column
# is bit-for-bit consistent with v0c-A's training-time preprocessing.
from data.tactile_dataset import _deep_stack, _FLOW_CHANNELS_USED
from data.utils.relative_action import (
    assert_layout_dims,
    build_relative_action_from_window,
    get_arm_layout,
)


def load_jsonl(jsonl_path):
    """
    load jsonl file
    """
    data = []
    with open(jsonl_path, 'r', encoding='UTF-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data



class DexVTAMDataset(Dataset):
    # Opt-in provenance for offline analysis; see __getitem__. Set on the
    # instance (``ds.return_meta = True``) rather than via the constructor so
    # no config or trainer call site has to change. Left False here so
    # training sees exactly the sample dict it always has.
    return_meta = False

    def __init__(self,
        data_roots,
        domains,
        task_recap_file = None,
        step_recap_file = None,
        sample_size=(192, 256), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        dataset_info_cache_path = None,
        action_type = "absolute",
        action_space = "joint",
        arm_layout = "bimanual",
        ignore_seek = False,
        train_dataset=True,
        action_key = "action",
        state_key = "observation.state",
        use_unified_prompt = False,
        unified_prompt = "best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx = None,
        fix_sidx = None,
        fix_mem_idx = None,
        stat_file = None,
        extra_parquet_index = False,
        valid_act_dim = None,
        valid_sta_dim = None,
        read_tactile = True,
        tactile_key = "tactile",
        repeat_dataset: int = 1,
        cache_dir = None,
        read_tactile_flow: bool = False,
        flow_stats_path = None,
        read_hand_pose: bool = False,
        pose_stats_path = None,
        pose_mode: str = "per_frame",
        episodes = None,
    ):
        """
        data_roots:              directory of LeRoBot dataset
        domains:                 name of your dataset, used to index different statistics
        task_recap_file:         json file of augmented task captions:
                                 {
                                    'ori_task_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_task_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        step_recap_file:         json file of augmented step captions:
                                 {
                                    'ori_step_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_step_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        sample_size:             video frame size
        sample_n_frames:         number of frames used to randomly or uniformly select memories
        preprocess:              frame preprocessing strategy, resize or center_crop_resize
        valid_cam:               list of cam names 
        chunk:                   number of video frames to predict
        action_chunk:            number of actions to predict, action_chunk should be an integer multiple of chunk.
        n_previous:              number of memory frames
        previous_pick_mode:      how to select memories
        random_crop:             randomly crop images
        dataset_info_cache_path: path to save dataset meta information cache
        action_type:             action space to use in this dataset
                                    'absolute': norm(act_t)
                                    'delta':    norm(act_t - act_{t-1})
                                    'relative': norm(act_t) - norm(state)
        action_space:            joint or eef, which is used to determinate the statistics values only in this dataset
        arm_layout:              which corpus layout the flat state/action vectors follow,
                                 'bimanual' (action 150 / state 90 / rel 136) or
                                 'right_only' (75 / 45 / 68). Named EXPLICITLY rather than
                                 inferred from the observed widths, so a stats-vs-parquet
                                 mismatch fails loudly on the first batch. See
                                 ``data/utils/relative_action.py``.
        ignore_seek:             if True, load the first furture frame only
        use_unified_prompt:      if set all prompt the same
        unified_prompt:          unified prompt
        fix_epiidx:              used in validation stage only, set episode index to fix_epiidx
        fix_sidx:                used in validation stage only, set start index to fix_sidx
        fix_mem_idx:             used in validation stage only, set memory indexes to fix_mem_idx
        stat_file:               used to specific statistics
        extra_parquet_index:     when extra_parquet_index=True, the shape of the action/state arrary saved in .parquet files should be [T,1,C]; when extra_parquet_index=False, the shape of the action/state arrary saved in .parquet files should be [T,C].
        valid_act_dim:           when valid_act_dim is not None, only the first $valid_act_dim dimenssions of actions will be used.
        valid_sta_dim:           when valid_sta_dim is not None, only the first $valid_sta_dim dimenssions of actions will be used.
        read_tactile:            if True (default), reads the parquet `tactile_key` column and adds a
                                 `tactile: (V_hand=2, F=5, T, 192, 256) float32 in [-1, 1]` field to
                                 each sample. Set False to short-circuit tactile I/O for the strict
                                 GE baseline (`use_tactile_views=false`); the trainer must then not
                                 assume `tactile` is present in the sample dict.
        tactile_key:             parquet column name for the tactile clip (default 'tactile').
        read_hand_pose:          if True, extract per-frame hand pose from the raw `state` column
                                 BEFORE state-normalization and add a
                                 `hand_pose: (V_hand, T, 22) float32` field to each sample
                                 (V_hand=2 bimanual, 1 right-only).
                                 Required by v0d Stage-1 adapter (pose injection). Per-hand slicing
                                 dispatches on ``state_dim`` (see ``_extract_hand_pose``):
                                 ``state_dim==58`` (cube/488 legacy, [L_arm, L_hand, R_arm, R_hand])
                                 uses left=state[:, 7:29], right=state[:, 36:58]; any other width
                                 takes the slices from ``arm_layout.state_hands``. v0d's downstream
                                 ``_align_pose_to_lat`` sees the same per-frame trajectory format
                                 Stage-1 trained on. Default False keeps the cube-trained
                                 sample-dict identical for the v0c-A code path.
        pose_stats_path:         JSON file with `{"mean": [...22], "std": [...22]}` for per-joint
                                 pose normalization. Same format Stage-1 used (see
                                 `data/stats/diverse_488/pose_stats.json`). When provided, applies
                                 `(hand_pose - mean) / std` so the trainer-time pose distribution
                                 matches the Stage-1 train-time distribution exactly. When None or
                                 missing, raw (un-normalized) pose is emitted. Only consulted when
                                 `read_hand_pose=True`.
        pose_mode:               "per_frame" (default; matches v0d Stage-1 schedule) emits the full
                                 (V_hand, T, 22) trajectory. "last_frame" emits (V_hand, 1, 22) only.
                                 Only consulted when `read_hand_pose=True`.
        episodes:                Optional[list[int]] whitelist of episode_index values to keep.
                                 ``None`` (default) preserves legacy "load all episodes" behavior.
                                 When a list is provided, only episodes whose ``episode_index``
                                 (the field in ``meta/episodes.jsonl``, GLOBAL numbering across
                                 the corpus) is in the list are retained; all others are dropped.
                                 The filter applies AFTER the JSONL walk / cache load and AFTER
                                 the optional cache save, so ``dataset_info_cache_path`` is the
                                 unfiltered superset and is safe to reuse across different
                                 filters.

                                 Failure modes (fail loud, never silent partial filter):
                                   - ``episodes`` is an empty list -> ValueError (use ``None``
                                     to disable instead).
                                   - ZERO requested ids matched the corpus -> ValueError,
                                     showing the first 10 requested ids (usually means
                                     wrong corpus or global-vs-local index confusion).
                                   - SOME requested ids missing from corpus -> ValueError
                                     listing the missing ids (typo / out-of-range / parquet
                                     file missing for that episode).

                                 Stage-2 A3 use: each ``data.val_splits.<name>`` config passes
                                 an ``episodes`` list to materialize one named val dataset
                                 (e.g. ``holdout_488: episodes: [464..487]`` reuses Stage-1
                                 v0d's val split; ``cube_val: episodes: [90..99]`` holds out
                                 the last 10 of the cube 100 corpus). The attribute
                                 ``self._kept_episode_indices: set[int]`` is exposed for
                                 smoke-test introspection (and is bit-equal to ``set(episodes)``
                                 by construction post-validation).
        repeat_dataset:          virtual length multiplier; ``self.dataset`` is replicated
                                 ``repeat_dataset`` times (default 1 = no repeat). For short
                                 corpora (e.g. 100 episodes) set to 1000 so each epoch contains
                                 enough batches to amortize DataLoader worker startup and to
                                 align with GE's ``steps_to_log`` / ``steps_to_save`` cadence.
                                 Each ``__getitem__`` already samples a fresh random window
                                 inside the episode, so repeats don't yield identical batches.
        cache_dir:               optional path to an offline preprocessing cache produced by
                                 ``scripts/preprocess_dex_vtam_cache.py``. When set, ``get_batch``
                                 reads pre-decoded ``video.npy`` / ``tactile.npy`` /
                                 ``action.npy`` / ``state.npy`` per episode via
                                 ``np.load(mmap_mode='r')``, skipping the per-getitem
                                 ``pd.read_parquet`` + PIL decode + ``_deep_stack`` + resize.
                                 OS pagecache shares the bytes across DataLoader workers.
                                 The cache schema (``schema.json``) is validated against
                                 the dataset's data-shaping kwargs at construction time;
                                 any mismatch (sample_size / preprocess / valid_cam /
                                 tactile_key / action_key / state_key /
                                 extra_parquet_index) raises ``ValueError``. Incompatible
                                 with ``random_crop=True`` (cache is post-resize fixed).
                                 Default ``None`` keeps the live decode path identical to
                                 pre-cache behavior, byte-for-byte.

        """
        
        zero_rank_print(f"loading annotations...")

        assert(action_type in ["delta", "absolute", "relative", "relative_eef_rot6d"])
        self.action_type = action_type
        assert(action_space in ["eef", "joint"])
        self.action_space = action_space
        # Which corpus layout the flat state/actions vectors follow. EXPLICIT by name
        # (never inferred from width, which would mask a stats/cache mismatch); the
        # observed widths are cross-checked against it on the first batch.
        self.arm_layout_name = arm_layout
        self.arm_layout = get_arm_layout(arm_layout)
        self._layout_checked = False
        # relative_eef_rot6d: the arm_target_pose block (4x4 per arm, e.g. bimanual
        # action [118:150]) is relativized to per-arm rel9 against a single anchor
        # (state row n_previous-1), shrinking the action 150 -> 136 (bimanual) or
        # 75 -> 68 (right_only). State stays absolute. valid_act_dim/valid_sta_dim
        # reslicing is incompatible with the fixed relative layout
        # (see data/utils/relative_action.py).
        if action_type == "relative_eef_rot6d":
            assert valid_act_dim is None and valid_sta_dim is None, (
                "relative_eef_rot6d does not support valid_act_dim/valid_sta_dim")



        self.action_key = action_key
        self.state_key = state_key
        self.extra_parquet_index = extra_parquet_index
        self.valid_act_dim = valid_act_dim
        self.valid_sta_dim = valid_sta_dim
        self.read_tactile = read_tactile
        self.tactile_key = tactile_key
        self.read_tactile_flow = bool(read_tactile_flow)
        self.flow_stats_path = flow_stats_path
        self._flow_mean = None
        self._flow_std = None
        if self.read_tactile_flow:
            if not flow_stats_path or not os.path.isfile(flow_stats_path):
                raise ValueError(
                    "DexVTAMDataset: read_tactile_flow=True requires a valid "
                    f"flow_stats_path; got {flow_stats_path!r}."
                )
            with open(flow_stats_path) as _f:
                _stats = json.load(_f)
            self._flow_mean = np.asarray(_stats["mean"], dtype=np.float32)  # (3,)
            self._flow_std = np.asarray(_stats["std"], dtype=np.float32)
            if self._flow_mean.shape != (len(_FLOW_CHANNELS_USED),) or self._flow_std.shape != (len(_FLOW_CHANNELS_USED),):
                raise ValueError(
                    f"flow_stats_path {flow_stats_path!r}: mean/std must each "
                    f"be ({len(_FLOW_CHANNELS_USED)},); got mean={self._flow_mean.shape}, "
                    f"std={self._flow_std.shape}."
                )

        # ---- hand_pose (v0d Stage-1 pose-injection adapter) -------------
        # Off by default so the v0c-A code path through cube configs is
        # unchanged. Enabled by the 488-midtrain yaml + any future v0d-aware
        # config. Slice indices dispatch on state_dim (see _extract_hand_pose):
        #   state_dim==58 (cube/488): left=state[:, 7:29], right=state[:, 36:58]
        #   otherwise:                arm_layout.state_hands
        # Each yields (T, 22) per arm; unrecognized widths are rejected.
        if pose_mode not in ("per_frame", "last_frame"):
            raise ValueError(
                f"pose_mode must be 'per_frame' or 'last_frame'; got "
                f"{pose_mode!r}."
            )
        self.read_hand_pose = bool(read_hand_pose)
        self.pose_mode = pose_mode
        self.pose_stats_path = pose_stats_path
        self._pose_mean = None
        self._pose_std = None
        if self.read_hand_pose and pose_stats_path and os.path.isfile(pose_stats_path):
            with open(pose_stats_path) as _f:
                _stats = json.load(_f)
            self._pose_mean = np.asarray(_stats["mean"], dtype=np.float32)  # (22,)
            self._pose_std = np.asarray(_stats["std"], dtype=np.float32)
            if self._pose_mean.shape != (22,) or self._pose_std.shape != (22,):
                raise ValueError(
                    f"pose_stats_path {pose_stats_path!r}: mean/std must each "
                    f"be (22,); got mean={self._pose_mean.shape}, "
                    f"std={self._pose_std.shape}."
                )
        elif self.read_hand_pose and pose_stats_path:
            zero_rank_print(
                f"[DexVTAMDataset] read_hand_pose=True but pose_stats_path "
                f"{pose_stats_path!r} does not exist; emitting raw "
                f"(un-normalized) hand_pose. v0d adapter training will see a "
                f"different distribution than Stage-1 -- verify this is "
                f"intentional."
            )

        # ---- Episode-level filter (Stage-2 A3 multi-split val) ----------
        # Pre-validate the `episodes` kwarg shape now. The actual filtering
        # happens AFTER self.dataset has been populated (post-cache-save)
        # so the cache stays filter-agnostic and is reusable across
        # multiple val_splits configs that re-target the same corpus.
        # `self._kept_episode_indices` is overwritten with the actually-
        # kept ids by the post-filter validator (see further down in
        # __init__); it stays None when no filter was requested.
        self._kept_episode_indices = None
        if episodes is not None:
            if not isinstance(episodes, (list, tuple)):
                raise TypeError(
                    f"DexVTAMDataset: `episodes` must be a list/tuple of "
                    f"ints (or None); got {type(episodes).__name__}."
                )
            _requested = {int(e) for e in episodes}
            if len(_requested) == 0:
                raise ValueError(
                    "DexVTAMDataset: `episodes` was passed as an empty "
                    "list. Pass None to disable filtering, or a non-empty "
                    "list of episode_index values."
                )
            # Stash the REQUESTED set; the post-filter validator below
            # will overwrite with the actually-kept set (after asserting
            # the two are equal).
            self._kept_episode_indices = _requested

        self.random_crop = random_crop
        
        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam, ]
        self.valid_cam = valid_cam
        if len(data_roots) == 1 and len(domains) > 1:
            data_roots = data_roots * len(domains)
        self.data_roots = data_roots
        self.dataset = []
        
        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            # construct the dataset_info
            for _data_root, _domain_name in zip(self.data_roots, domains):

                print(f"Loading {_domain_name} data from {_data_root}")
                
                # into the meta folder
                if os.path.exists(os.path.join(_data_root, _domain_name, "meta", "tasks.jsonl")):
                    meta_folder = os.path.join(_data_root, _domain_name, "meta")
                    data_folder = os.path.join(_data_root, _domain_name, "data")
                    video_folder = os.path.join(_data_root, _domain_name, "videos")
                else:
                    meta_folder = os.path.join(_data_root, "meta")
                    data_folder = os.path.join(_data_root, "data")
                    video_folder = os.path.join(_data_root, "videos")
                    
                tasks_jsonl = os.path.join(meta_folder, "tasks.jsonl")
                task_index_task_str = load_jsonl(tasks_jsonl)
                task_index_task_str_dict = {}
                for item in task_index_task_str:
                    task_index_task_str_dict[item['task_index']] = item['task']


                with open(os.path.join(meta_folder, "info.json"), "r") as f:
                    metainfo = json.load(f)
                    total_chunks = metainfo["total_chunks"]
                    chunks_size = metainfo["chunks_size"]

                episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
                epiosdes_data = load_jsonl(episodes_jsonl) # episode_index  tasks  length


                for episode_data in tqdm(epiosdes_data):

                    episode_index = episode_data['episode_index']
                    tasks = episode_data['tasks']
                    if len(tasks) > 1:
                        task = random.choice(tasks)
                    else:
                        task = tasks[0]
                    length = episode_data['length']
                    
                    episode_chunk = int(episode_index//chunks_size)

                    parquet_path = os.path.join(data_folder, f"chunk-{episode_chunk:03d}", f"episode_{episode_index:06d}.parquet")
                    if not os.path.exists(parquet_path):
                        zero_rank_print(f"parquet file not found: {parquet_path}")
                        continue

                    video_path = os.path.join(video_folder, f"chunk-{episode_chunk:03d}", "{}", f"episode_{episode_index:06d}.mp4")
                    
                    info = [
                        video_path,
                        None, # no need for camera_info
                        parquet_path,
                        _domain_name, "", # DomainTable[_domain_name],
                        None, task, # no task_info
                        length,
                    ]
                    self.dataset.append(info)

        if dataset_info_cache_path is not None and not(os.path.exists(dataset_info_cache_path)):
            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "w") as f:
                json.dump(self.dataset, f)

        # ---- Episode-level filter: apply now (post-cache-save) ----------
        # Filter is applied on the populated self.dataset (whether sourced
        # from live JSONL walk or a pre-existing cache JSON), and BEFORE
        # `repeat_dataset` multiplication. Episode_index is re-derived from
        # the parquet basename of each info tuple (same logic as
        # _open_cache_for_episode); this is necessary because the info
        # tuple schema does not include episode_index as a field.
        if self._kept_episode_indices is not None:
            requested = set(self._kept_episode_indices)
            filtered_dataset = []
            actually_kept = set()
            corpus_size_before = len(self.dataset)
            for info in self.dataset:
                parquet_path = info[2]
                ep_basename = os.path.basename(parquet_path)
                if not (
                    ep_basename.startswith("episode_")
                    and ep_basename.endswith(".parquet")
                ):
                    continue
                try:
                    ep_idx = int(
                        ep_basename[len("episode_"):-len(".parquet")]
                    )
                except ValueError:
                    continue
                if ep_idx in requested:
                    filtered_dataset.append(info)
                    actually_kept.add(ep_idx)
            # Fail-loud validation (never silently produce a partial filter).
            if len(actually_kept) == 0:
                raise ValueError(
                    f"DexVTAMDataset: episode filter requested "
                    f"{len(requested)} episodes but ZERO matched the "
                    f"corpus at data_roots={self.data_roots!r} "
                    f"(corpus had {corpus_size_before} episode info "
                    f"tuples post-walk). requested ids (first 10) = "
                    f"{sorted(requested)[:10]}. Verify data_roots / "
                    f"domain / episode-index numbering convention; this "
                    f"often happens when the requested ids are "
                    f"episode_index from a DIFFERENT corpus (e.g. cube "
                    f"indices [90..99] vs 488 indices [464..487])."
                )
            missing = requested - actually_kept
            if missing:
                raise ValueError(
                    f"DexVTAMDataset: episode filter requested "
                    f"{len(requested)} episodes but {len(missing)} are "
                    f"missing from the corpus at "
                    f"data_roots={self.data_roots!r}. missing ids = "
                    f"{sorted(missing)}. This means either (a) those "
                    f"episode_index values do not exist in "
                    f"episodes.jsonl, or (b) their per-episode parquet "
                    f"files are missing on disk."
                )
            # Pretty grep-friendly log of the actually-kept ids.
            kept_sorted = sorted(actually_kept)
            if len(kept_sorted) <= 12:
                kept_str = str(kept_sorted)
            else:
                kept_str = (
                    f"[{kept_sorted[0]}, {kept_sorted[1]}, ..., "
                    f"{kept_sorted[-2]}, {kept_sorted[-1]}] "
                    f"(N={len(kept_sorted)})"
                )
            zero_rank_print(
                f"[DexVTAMDataset] episode filter active: kept "
                f"{len(actually_kept)} of {corpus_size_before} corpus "
                f"episodes; kept = {kept_str}"
            )
            self.dataset = filtered_dataset
            # Replace the stashed REQUESTED set with the ACTUALLY-KEPT
            # set so downstream introspection (smoke tests) reads the
            # post-validated value. By construction these two sets are
            # equal here (the `missing` check above would have raised),
            # but using the actually-kept value defends against future
            # code paths that might dedupe / reorder requested.
            self._kept_episode_indices = actually_kept

        # `repeat_dataset` virtually inflates the dataset length by N (mirrors
        # WORLD-MODEL-TOUCH/data/libero_dataset.py). Useful for short corpora
        # (e.g. 100-episode pick-cube): without this each "epoch" is only a
        # handful of batches, which thrashes the DataLoader workers and makes
        # the GE log/save cadence (steps_to_log / steps_to_save) misalign with
        # epoch boundaries. Cache JSON above is written BEFORE the multiply so
        # the cache stays the unique-episode list, not 100k duplicates.
        if repeat_dataset is None:
            repeat_dataset = 1
        repeat_dataset = int(repeat_dataset)
        if repeat_dataset > 1:
            self.dataset = self.dataset * repeat_dataset

        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length} (repeat_dataset={repeat_dataset})")

        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert(self.chunk * self.video_temporal_stride == self.action_chunk)

        self.sample_n_frames = sample_n_frames
        
        self.sample_size = sample_size

        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),  # the size of shape (1,) means the smaller edge will be resized to it and the img will keep the h-w ratio.
                transforms.CenterCrop(sample_size),
            ])
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        if n_previous > 1:
            self.n_previous = int(n_previous)
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = int(self.sample_n_frames - self.chunk)
            self.previous_pick_mode = 'uniform'

        if task_recap_file is not None:
            with open(task_recap_file, 'r', encoding='UTF-8') as f:
                self.task_recap_map = json.load(f)
        else:
            self.task_recap_map = None

        if step_recap_file is not None:
            with open(step_recap_file, 'r', encoding='UTF-8') as f:
                self.step_recap_map = json.load(f)
        else:
            self.step_recap_map = None

        self.use_unified_prompt = use_unified_prompt

        ### validation only
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx

        ### load stat_file if provided
        self.StatisticInfo = StatisticInfo
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)

        self.ignore_seek = ignore_seek

        # ----- Optional offline preprocessing cache -----
        # If ``cache_dir`` is set, ``get_batch`` will route through
        # :meth:`_get_batch_from_cache`, reading mmap-friendly per-episode
        # ``.npy`` files produced by ``scripts/preprocess_dex_vtam_cache.py``
        # instead of doing the per-getitem PIL decode + ``_deep_stack`` +
        # resize. Schema is validated up front so a stale cache fails loud.
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            if self.random_crop:
                raise ValueError(
                    "DexVTAMDataset: cache_dir is incompatible with random_crop=True. "
                    "The cache is generated post-resize and cannot reproduce per-sample "
                    "random crops. Set random_crop=False (the production yaml already "
                    "does), or remove cache_dir."
                )
            self._validate_cache_schema(self.cache_dir)

    def _validate_cache_schema(self, cache_dir):
        """Ensure ``<cache_dir>/schema.json`` matches the dataset's data-shaping
        config. Any drift raises ``ValueError`` so a stale cache cannot
        silently corrupt training tensors."""
        schema_path = os.path.join(cache_dir, "schema.json")
        if not os.path.isfile(schema_path):
            raise FileNotFoundError(
                f"DexVTAMDataset.cache_dir = {cache_dir!r} is missing 'schema.json'. "
                f"Run scripts/preprocess_dex_vtam_cache.py to generate the cache."
            )
        with open(schema_path) as f:
            schema = json.load(f)
        expected = {
            "schema_version": "v1",
            "sample_size": list(self.sample_size),
            "preprocess": self.preprocess,
            "valid_cam": list(self.valid_cam),
            "tactile_key": self.tactile_key,
            "action_key": self.action_key,
            "state_key": self.state_key,
            "extra_parquet_index": bool(self.extra_parquet_index),
        }
        mismatches = []
        for key, want in expected.items():
            got = schema.get(key)
            # Lists vs tuples: compare normalized.
            if isinstance(want, list):
                got_norm = list(got) if got is not None else None
                if got_norm != want:
                    mismatches.append(f"  {key}: cache={got_norm!r}  dataset={want!r}")
            else:
                if got != want:
                    mismatches.append(f"  {key}: cache={got!r}  dataset={want!r}")
        if mismatches:
            raise ValueError(
                f"DexVTAMDataset cache schema mismatch with dataset config "
                f"(cache_dir={cache_dir!r}):\n"
                + "\n".join(mismatches)
                + "\nFix: regenerate the cache with matching kwargs, OR pick a "
                  "different cache_dir, OR update the yaml."
            )

    def get_frame_indexes(self, total_frames, ):
        """
        select self.n_previous memory frames and self.action_chunk prediction frmaes
        1. randomly select the end frame
        2. take frames from {end-action_chunk} to {end} as the prediction frames
        3. uniformly/randomly select memory frames from {end-self.sample_n_frames} to {end-action_chunk}
        """

        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx+self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            action_indexes = np.clip(action_indexes, a_min=0, a_max=total_frames-1).tolist()
            frame_indexes = np.clip(frame_indexes, a_min=0, a_max=total_frames-1).tolist()
            fix_mem_idx = np.clip(self.fix_mem_idx, a_min=0, a_max=total_frames-1).tolist()
            self._stash_meta_phase(fix_mem_idx + frame_indexes, total_frames)
            return fix_mem_idx + frame_indexes, fix_mem_idx + action_indexes

        chunk_end = random.randint(self.action_chunk, total_frames+self.action_chunk)
        indexes = np.array(list(range(max(-100, chunk_end-self.sample_n_frames), chunk_end)))
        indexes = np.clip(indexes, a_min=0, a_max=total_frames-1).tolist()
        video_end = indexes[-self.action_chunk:]
        # mem_candidates = [
        #     indexes[int(i)] for i in range(0, self.sample_n_frames-self.action_chunk-1)
        # ]
        mem_candidates = indexes[:-self.action_chunk]
        if len(mem_candidates)<self.n_previous-1:
            mem_candidates = [1,]*(self.n_previous-1) + mem_candidates

        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates)-1, self.n_previous).tolist()]

        elif self.previous_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]

        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")       
        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_temporal_stride-1::self.video_temporal_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]

        action_indexes = mem_indexes + video_end

        self._stash_meta_phase(frame_indexes, total_frames)
        return frame_indexes, action_indexes

    def _stash_meta_phase(self, frame_indexes, total_frames):
        """Record where in the episode the clip just sampled ends.

        Both ``get_batch`` and ``_get_batch_from_cache`` route through
        ``get_frame_indexes``, so stashing here covers every path. Reading it
        back in ``__getitem__`` is safe because each DataLoader worker holds
        its own dataset copy and calls ``__getitem__`` serially.

        ``frame_indexes[-1]`` is already clipped into ``[0, total_frames-1]``,
        so the ratio is a normalized "how far into the episode" in [0, 1] and
        is comparable across episodes of different lengths.
        """
        self._meta_phase = float(frame_indexes[-1]) / float(max(total_frames - 1, 1))


    def get_action_bias_std(self, domain_name):
        return torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['mean']).unsqueeze(0), torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['std']).unsqueeze(0)+1e-6


    def get_action_q01_q99(self, domain_name):
        return torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['q01']).unsqueeze(0), torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['q99']).unsqueeze(0)


    def _get_relative_q01_q99(self, domain_name):
        """q01/q99 for the relative_eef_rot6d action block ({domain}_relative_{space}).

        HARD-FAILS (KeyError) if the relative block is missing -- never silently
        falls back to the absolute stats (that would mis-normalize the relative
        action). Regenerate with ``get_statistics.py --relative``.

        Also checks the block's WIDTH against ``arm_layout``: a bimanual (136-D)
        stats file paired with a right-only dataset would otherwise broadcast
        against a 68-D action and silently mis-scale every dim.
        """
        key = f"{domain_name}_relative_{self.action_space}"
        if key not in self.StatisticInfo:
            raise KeyError(
                f"action_type='relative_eef_rot6d' requires stats key {key!r}, "
                f"but it is missing from the stat_file. Regenerate with "
                f"scripts/get_statistics.py --relative (see "
                f"scripts/calc_stats_relative_eef_rot6d.sh). "
                f"Available keys: {list(self.StatisticInfo)}")
        sub = self.StatisticInfo[key]
        assert_layout_dims(self.arm_layout, rel_action_dim=len(sub['q01']),
                           where=f"stat_file key {key!r}")
        return (torch.tensor(sub['q01']).unsqueeze(0),
                torch.tensor(sub['q99']).unsqueeze(0))


    def _check_layout(self, abs_action_dim, state_dim):
        """Cross-check the OBSERVED parquet widths against the configured ``arm_layout``.

        Runs once (first window): the widths are fixed by the parquet schema, so a
        wrong ``arm_layout`` is a config error that should surface on the first batch
        rather than as a mis-sliced pose thousands of steps in.
        """
        if self._layout_checked:
            return
        assert_layout_dims(self.arm_layout, abs_action_dim=abs_action_dim,
                           state_dim=state_dim, where="dataset parquet")
        self._layout_checked = True

    def _build_action_target(self, action_full_raw, indexes, state_win_raw,
                             domain_name, action_min, action_max):
        """Build the normalized action-only target for the configured action_type.

        action_full_raw : (T_total, abs_action_dim) RAW numpy action array.
        indexes         : the action window indexes (len n_previous+action_chunk).
        state_win_raw   : (T_win, state_dim) RAW (pre-normalization) torch state
                          window == state[indexes]; supplies the relative anchor.
        Returns a (T_win, action_only_dim) torch tensor normalized to [-1, 1]
        (``abs_action_dim`` for absolute, ``rel_action_dim`` for relative_eef_rot6d:
        150/136 bimanual, 75/68 right_only).
        """
        if self.action_type == "absolute":
            action = action_full_raw[indexes].astype(np.float32)
            action = torch.FloatTensor(action)
            action = (action - action_min) / (action_max - action_min + 1e-6)
            return action * 2.0 - 1.0
        if self.action_type == "relative_eef_rot6d":
            action_win_raw = action_full_raw[indexes].astype(np.float32)   # (T, abs_action_dim)
            state_win_raw_np = np.asarray(state_win_raw, dtype=np.float32)  # (T, state_dim)
            self._check_layout(action_win_raw.shape[1], state_win_raw_np.shape[1])
            rel_raw = build_relative_action_from_window(
                action_win_raw, state_win_raw_np, self.n_previous,
                self.arm_layout)                                            # (T, rel_action_dim)
            rel_min, rel_max = self._get_relative_q01_q99(domain_name)
            action = torch.FloatTensor(rel_raw)
            action = (action - rel_min) / (rel_max - rel_min + 1e-6)
            return action * 2.0 - 1.0
        raise NotImplementedError(
            f"action_type={self.action_type!r} is not supported in this dataset "
            f"(expected 'absolute' or 'relative_eef_rot6d').")


    def seek_mp4(self, video_path, cam_name_list, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,v,t,h,w)
        """
        video_list = []
        for cam_name in cam_name_list:
            video_reader = VideoFileClip(video_path.format(cam_name))
            fps = video_reader.fps
            video = []
            for idx in slices:
                video.append(video_reader.get_frame(float(idx)/fps))
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float()/255.
            video_reader.close()
            video_list.append(video)
        video_list = torch.stack(video_list, dim=1)
        return video_list



    def transform_video(self, videos, specific_transforms_resize, intrinsics, sample_size):
        """
        crop (optional) and resize the videos, and modify the intrinsic accordingly
        """
        c, v, t, h, w = videos.shape
        new_videos = []
        new_intrinsics = []
        for iv in range(v):
            video = videos[:, iv]
            if self.random_crop:
                h_start, w_start, h_crop, w_crop = gen_crop_config(video)
                video = video[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
                if intrinsics is not None:
                    intrinsic = intrin_crop_transform(intrinsics[iv], h_start, w_start)
                
                h, w = h_crop, w_crop
            if intrinsics is not None:
                intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
                new_intrinsics.append(intrinsic)
                
            video = specific_transforms_resize(video)
            new_videos.append(video)
        new_videos = torch.stack(new_videos, dim=1)
        if len(new_intrinsics) > 0:
            new_intrinsics = torch.stack(new_intrinsics, dim=0)
        else:
            new_intrinsics = None
        return new_videos, None


    def normalize_video(self, video, specific_transforms_norm):
        """
        input video should have shape (c,v,t,h,w)
        """
        c,v,t,h,w = video.shape
        video = specific_transforms_norm(video.permute(1,2,0,3,4).reshape(-1,c,h,w)).reshape(v,t,c,h,w).permute(2,0,1,3,4)
        return video


    def get_transform(self, ):
        sample_size = self.sample_size
        specific_transforms_resize = self.pixel_transforms_resize
        specific_transforms_norm = self.pixel_transforms_norm
        return sample_size, specific_transforms_resize, specific_transforms_norm


    def get_long_recaption(self, step_captions, task_caption):
        newcap = []
        # find = []
        for step_caption in step_captions:
            if self.step_recap_map is not None:
                recap_list = self.step_recap_map.get(step_caption,[])
                recap_list.append(step_caption)
                step_caption = np.random.choice(recap_list,1)
                newcap.append(str(step_caption[0]))
            else:
                newcap.append(step_caption)

        newcap = ", ".join(newcap)
        newcap = newcap.replace(" the "," ")
        if self.task_recap_map is not None:
            task_recap_list = self.task_recap_map.get(task_caption,[])
            task_recap_list.append(task_caption)
            task_newcap = np.random.choice(task_recap_list,1)
            task_newcap = str(task_newcap[0])
            fullcap = task_newcap + ": " + newcap
        else:
            task_newcap = task_caption
            fullcap = task_caption + ": " + newcap
        cap_type = random.randint(0,2)
        allcap = [fullcap, task_newcap, newcap]
        recap = allcap[cap_type]
        return recap



    # ------------------------------------------------------------------
    # Hand pose extraction (v0d pose-injection adapter)
    # ------------------------------------------------------------------

    def _extract_hand_pose(self, state_raw: np.ndarray) -> torch.Tensor:
        """Extract per-hand, per-frame hand pose from raw (pre-normalize) state.

        Shared between the cached and uncached parquet paths so the two
        branches emit byte-identical hand_pose. Called BEFORE the
        ``state * 2 - 1`` normalize because hand pose has its own (Stage-1)
        normalization stats; mixing the two normalizations would shift the
        distribution v0d's pose_encoder was trained on.

        Args:
            state_raw: ``(T, state_dim)`` float32 ndarray; ``state_dim`` must be
                either 58 (legacy cube corpus) or the configured ``arm_layout``'s
                ``state_dim``, which supplies the per-arm hand slices.

        Returns:
            ``(V_hand, T, 22)`` float32 torch tensor, one entry per arm of the
            configured ``arm_layout`` in its ``arms`` order -- ``[left, right]`` for
            bimanual (matching ``HANDS_LEFT=0, HANDS_RIGHT=1`` in
            :class:`data.tactile_dataset.TactileDataset`), ``[right]`` alone for a
            right-only corpus, so V_hand matches the ``(T, 1, 5, ...)`` tactile stack.
            When ``pose_mode="last_frame"``, the T axis is collapsed to length 1.
        """
        if state_raw.ndim != 2:
            raise ValueError(
                f"_extract_hand_pose: expected (T, state_dim); got "
                f"{state_raw.shape!r}."
            )
        # Layouts, dispatched on state_dim (bit-stable per corpus by parquet schema):
        #
        #   58-dim (cube/488 legacy):  [L_arm(7), L_hand(22), R_arm(7), R_hand(22)]
        #     -> L_hand [7:29], R_hand [36:58]. Hardcoded: no pose blocks, so it is
        #        not expressible as a RelativeArmLayout.
        #
        #   otherwise: the configured arm_layout's state_hands, i.e.
        #     bimanual  90-D [L_arm, R_arm, L_hand, R_hand, L_pose, R_pose]
        #     right_only 45-D [R_arm, R_hand, R_pose]
        state_dim = state_raw.shape[1]
        if state_dim == 58:
            # legacy cube/488 corpus: predates arm_layout and has no pose blocks,
            # so its hand slices are not derivable from a RelativeArmLayout.
            hand_slices = ((7, 29), (36, 58))
        elif state_dim == self.arm_layout.state_dim:
            # 90-D bimanual -> ((14,36),(36,58)); 45-D right_only -> ((7,29),).
            # Note 45-D's R_hand [7:29] collides with the legacy layout's L_hand,
            # which is why this keys on the declared layout and not on the slice.
            hand_slices = self.arm_layout.state_hands
        else:
            raise ValueError(
                f"_extract_hand_pose: state_dim={state_dim} matches neither the 58-D "
                f"legacy cube layout ([L_arm, L_hand, R_arm, R_hand]) nor the "
                f"configured arm_layout={self.arm_layout_name!r} "
                f"(state_dim={self.arm_layout.state_dim}). The current dataset's "
                f"`state_key={self.state_key!r}` does not have a supported layout. "
                f"Fix `arm_layout`, disable `read_hand_pose`, or regenerate the parquet."
            )
        hands = [state_raw[:, lo:hi].astype(np.float32, copy=True) for lo, hi in hand_slices]
        if self._pose_mean is not None:
            # _pose_mean / _pose_std are (22,); broadcast over T.
            hands = [(h - self._pose_mean) / self._pose_std for h in hands]
        if self.pose_mode == "last_frame":
            hands = [h[-1:, :] for h in hands]
        hand_pose = np.stack(hands, axis=0)  # (V_hand, T, 22)
        return torch.from_numpy(np.ascontiguousarray(hand_pose))

    # ------------------------------------------------------------------
    # Offline preprocessing cache fast path
    # ------------------------------------------------------------------

    def _open_cache_for_episode(self, parquet_path):
        """Open (and per-worker memoize) the four mmap views for an episode.

        The cache is keyed by ``parquet_path`` so two ``self.dataset`` entries
        that point to the same parquet (e.g. via ``repeat_dataset``) share one
        set of ``np.memmap`` handles. Returned dict is::

            {"video": memmap, "tactile": memmap or None, "action": memmap, "state": memmap}

        Each DataLoader worker has its own forked ``self``, so the dict is
        per-worker; OS pagecache shares the underlying file pages across all
        workers automatically.
        """
        if not hasattr(self, "_mmap_cache"):
            self._mmap_cache = {}
        ep_basename = os.path.basename(parquet_path)
        if not (ep_basename.startswith("episode_") and ep_basename.endswith(".parquet")):
            raise ValueError(
                f"_open_cache_for_episode: cannot derive episode_index from "
                f"unexpected parquet basename {ep_basename!r}; expected "
                f"'episode_NNNNNN.parquet'."
            )
        ep_idx = int(ep_basename[len("episode_"):-len(".parquet")])
        ep_dir = os.path.join(self.cache_dir, f"episode_{ep_idx:06d}")
        cached = self._mmap_cache.get(ep_dir)
        if cached is not None:
            return cached
        if not os.path.isdir(ep_dir):
            raise FileNotFoundError(
                f"_open_cache_for_episode: episode cache missing at {ep_dir!r}; "
                f"run preprocess_dex_vtam_cache.py to generate it."
            )
        cached = {
            "video":   np.load(os.path.join(ep_dir, "video.npy"),   mmap_mode="r"),
            "tactile": (np.load(os.path.join(ep_dir, "tactile.npy"), mmap_mode="r")
                        if self.read_tactile else None),
            "tactile_flow": (
                np.load(os.path.join(ep_dir, "tactile_flow.npy"), mmap_mode="r")
                if self.read_tactile_flow else None
            ),
            "action":  np.load(os.path.join(ep_dir, "action.npy"),  mmap_mode="r"),
            "state":   np.load(os.path.join(ep_dir, "state.npy"),   mmap_mode="r"),
        }
        self._mmap_cache[ep_dir] = cached
        return cached

    def _get_batch_from_cache(self, idx):
        """Cache-backed equivalent of :meth:`get_batch`.

        Mirrors the live ``get_batch`` ordering and tensor ops exactly so that
        ``sample_with_cache[i] == sample_without_cache[i]`` for the same RNG
        state and ``vid_indexes`` (``scripts/test_dex_vtam_cache_consistency.py``
        enforces this). The only step removed is the per-getitem
        ``pd.read_parquet`` + PIL decode + ``_deep_stack`` + resize -- those
        are baked into the mmap'd ``.npy`` files.
        """
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        caption = self.dataset[idx][6]
        total_frames = self.dataset[idx][7]

        sample_size, _, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames)

        cached = self._open_cache_for_episode(parquet_path)

        action_min, action_max = self.get_action_q01_q99(domain_name)
        state_min, state_max = self.get_action_q01_q99(domain_name + "_state")

        # ---- action / state (mirror live get_batch lines 489-533) ----
        # Cache stores RAW (pre-normalize) so changes to stat_file don't
        # invalidate the cache. The full T_total array is small so we just
        # materialize it once (np.asarray copies out of mmap into RAM).
        action = np.asarray(cached["action"]).astype(np.float32)   # (T_total, action_dim)
        state = np.asarray(cached["state"]).astype(np.float32)     # (T_total, state_dim)

        # ---- hand_pose extraction (v0d) ----
        # hand_pose must be temporally co-aligned with TACTILE (which is
        # sliced by `vid_indexes`). Using `indexes` here is wrong when
        # `action_chunk != chunk * video_temporal_stride`: cube Stage-2
        # has action_chunk=54, chunk=9, stride=undef -> vid_indexes has
        # `n_previous + chunk = 13` elements, while `indexes` (=action_indexes)
        # has `n_previous + action_chunk = 58`. The v0d adapter's
        # `_align_pose_to_lat` aligns hand_pose against the tactile T-axis,
        # so the input must match tactile's length (here 13), not the
        # action window's length (58). Extraction happens BEFORE state's
        # [-1,1] normalization because hand_pose carries its own pose_stats
        # (mixing normalizations changes the distribution v0d's
        # pose_encoder was trained on).
        if self.read_hand_pose:
            state_raw_indexed = state[vid_indexes]  # (T_tactile, state_dim) np.float32
            hand_pose_tensor = self._extract_hand_pose(state_raw_indexed)
        else:
            hand_pose_tensor = None

        state = torch.FloatTensor(state)[indexes]
        # RAW (pre-normalization) state window. relative_eef_rot6d reads its
        # single anchor from row n_previous-1 here, before the [-1,1] norm below.
        state_win_raw = state.clone()

        if self.valid_act_dim is not None:
            action = action[:, :self.valid_act_dim]
            action_min = action_min[:, :self.valid_act_dim]
            action_max = action_max[:, :self.valid_act_dim]
        if self.valid_sta_dim is not None:
            state = state[:, :self.valid_sta_dim]
            state_min = state_min[:, :self.valid_sta_dim]
            state_max = state_max[:, :self.valid_sta_dim]

        state = (state - state_min) / (state_max - state_min + 1e-6)
        state = state * 2.0 - 1.0

        action = self._build_action_target(
            action, indexes, state_win_raw, domain_name, action_min, action_max)

        ori_act_dim = action.shape[1]
        action = torch.cat((action, state), dim=1)
        state = torch.cat((torch.zeros([1, ori_act_dim]), state[self.n_previous-1:self.n_previous]), dim=1)

        # ---- video (mirror live get_batch lines 537-551) ----
        # cache video layout: (T_total, V_rgb, H, W, 3) uint8 (post-resize).
        # Live layout post-pipeline: (3, V_rgb, T_clip, H, W) float32 in [-1,1].
        video_np = np.array(cached["video"][vid_indexes])              # (T_clip, V, H, W, 3) uint8 -- writable copy
        videos = torch.from_numpy(video_np).float() / 255.0            # (T_clip, V, H, W, 3)
        videos = videos.permute(4, 1, 0, 2, 3).contiguous()            # (3, V, T_clip, H, W)
        # transform_video is skipped: cache is already post-resize, and our
        # configs use random_crop=False (enforced by __init__ validation).
        videos = self.normalize_video(videos, specific_transforms_norm)

        # ---- tactile (mirror live get_batch lines 562-568) ----
        if self.read_tactile:
            tac_np = np.array(cached["tactile"][vid_indexes])          # (T_clip, V_hand, F, H, W) uint8
            tactile = tac_np.astype(np.float32) / 127.5 - 1.0
            tactile = np.transpose(tactile, (1, 2, 0, 3, 4))           # (V_hand, F, T_clip, H, W)
            tactile = torch.from_numpy(np.ascontiguousarray(tactile))
        else:
            tactile = None

        if self.read_tactile_flow:
            flow_np = np.array(cached["tactile_flow"][vid_indexes])     # (T_clip, V_hand, F, 24, 32, 4)
            flow_np = flow_np[..., list(_FLOW_CHANNELS_USED)]           # (T_clip, V_hand, F, 24, 32, 3)
            flow_np = (flow_np - self._flow_mean.reshape(1, 1, 1, 1, 1, 3)) / (
                self._flow_std.reshape(1, 1, 1, 1, 1, 3) + 1e-6
            )
            flow_np = np.transpose(flow_np, (1, 2, 0, 3, 4, 5))         # (V_hand, F, T_clip, 24, 32, 3)
            tactile_flow = torch.from_numpy(np.ascontiguousarray(flow_np.astype(np.float32)))
        else:
            tactile_flow = None

        # hand_pose_tensor: (V_hand=2, T, 22) float32 if read_hand_pose else None.
        # Sanity gate: T-axis of pose MUST equal tactile's T-axis. Both
        # are sliced by `vid_indexes` upstream, so this should always
        # hold; a mismatch here indicates either a regression in the
        # slicing code above or a yaml that uses different windows for
        # the two modalities (which v0d's `_align_pose_to_lat` does not
        # support).
        if hand_pose_tensor is not None and tactile is not None:
            if hand_pose_tensor.shape[1] != tactile.shape[2]:
                raise ValueError(
                    f"hand_pose T-axis ({hand_pose_tensor.shape[1]}) != "
                    f"tactile T-axis ({tactile.shape[2]}); index mismatch."
                )

        return videos, action, caption, state, tactile, tactile_flow, hand_pose_tensor

    def get_batch(self, idx):

        # Offline preprocessing cache fast path (skips PIL decode +
        # _deep_stack + resize). See ``_get_batch_from_cache`` and
        # ``_validate_cache_schema``. The cache is byte-identical to the
        # live path when ``transforms.Resize(sample_size)`` is a no-op
        # (i.e. parquet frames are already at sample_size); otherwise the
        # cached video carries at most 2/255 max abs error per pixel after
        # the downstream Normalize(0.5, 0.5) due to one uint8 quantization
        # round-trip.
        if self.cache_dir is not None:
            return self._get_batch_from_cache(idx)

        video_path = self.dataset[idx][0]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        # domain_id = self.dataset[idx][4]
        caption = self.dataset[idx][6]
        total_frames = self.dataset[idx][7]
        
        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames, )
        
        data = pd.read_parquet(parquet_path)


        action_min, action_max = self.get_action_q01_q99(domain_name)
        state_min, state_max = self.get_action_q01_q99(domain_name + "_state")
        
        ###
        ### example data
        ### data[self.action_key] with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        ### data[self.state_key]  with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        # Catch only the schema-mismatch exceptions and chain the original
        # error so the actual cause (KeyError on a wrong action_key /
        # state_key, IndexError on extra_parquet_index mismatch, etc.) is
        # preserved in the traceback. The previous bare `except:` swallowed
        # KeyboardInterrupt too, which made runaway DataLoader workers
        # impossible to Ctrl+C-out of.
        try:
            if self.extra_parquet_index:
                action = np.stack([data[self.action_key][i][0] for i in range(data[self.action_key].shape[0])])
                state = np.stack([data[self.state_key][i][0] for i in range(data[self.state_key].shape[0])])
            else:
                action = np.stack([data[self.action_key][i] for i in range(data[self.action_key].shape[0])])
                state = np.stack([data[self.state_key][i] for i in range(data[self.state_key].shape[0])])
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            raise ValueError(
                f"Failed to read action/state from parquet. "
                f"action_key={self.action_key!r}, state_key={self.state_key!r}, "
                f"extra_parquet_index={self.extra_parquet_index}. "
                f"Expected per-row T*C arrays. Underlying error: {e!r}"
            ) from e

        action = action.astype(np.float32)
        state = state.astype(np.float32)

        # ---- hand_pose extraction (v0d) ----
        # Same logic as `_get_batch_from_cache`: hand_pose must be
        # co-aligned with TACTILE (vid_indexes, length=n_previous+chunk),
        # NOT with the action window (indexes, length=n_previous+
        # action_chunk). See the longer comment in
        # `_get_batch_from_cache`. Extracted BEFORE state's [-1,1]
        # normalization so v0d's pose_encoder sees the same distribution
        # Stage-1 was trained on.
        if self.read_hand_pose:
            state_raw_indexed = state[vid_indexes]  # (T_tactile, state_dim) np.float32
            hand_pose_tensor = self._extract_hand_pose(state_raw_indexed)
        else:
            hand_pose_tensor = None

        state = torch.FloatTensor(state)[indexes]
        # RAW (pre-normalization) state window. relative_eef_rot6d reads its
        # single anchor from row n_previous-1 here, before the [-1,1] norm below.
        state_win_raw = state.clone()

        if self.valid_act_dim is not None:
            action = action[:, :self.valid_act_dim]
            action_min = action_min[:, :self.valid_act_dim]
            action_max = action_max[:, :self.valid_act_dim]

        if self.valid_sta_dim is not None:
            state = state[:, :self.valid_sta_dim]
            state_min = state_min[:, :self.valid_sta_dim]
            state_max = state_max[:, :self.valid_sta_dim]

        state = (state - state_min) / (state_max - state_min + 1e-6)
        state = state * 2.0 - 1.0

        ### act = norm(act)
        action = self._build_action_target(
            action, indexes, state_win_raw, domain_name, action_min, action_max)

        ori_act_dim = action.shape[1]

        action = torch.cat((action, state), dim=1)
        state = torch.cat((torch.zeros([1,ori_act_dim]), state[self.n_previous-1:self.n_previous]), dim=1)

        # videos = self.seek_mp4(video_path, self.valid_cam, vid_indexes)

        video_list = []
        for cam in self.valid_cam:
            cam_img_bytes = data[cam].to_list()
            video = []
            for index in vid_indexes:
                img = Image.open(io.BytesIO(cam_img_bytes[index]["bytes"]))
                video.append(img)
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float()/255.
            video_list.append(video)
        videos = torch.stack(video_list, dim=1) 
        videos, _ = self.transform_video(
            videos, specific_transforms_resize, None, sample_size
        )
        videos = self.normalize_video(videos, specific_transforms_norm)

        # ----- Tactile column read (Stage 2) -----
        # LeRobot stores the per-frame tactile blob as a nested-object array:
        # each row of `data[self.tactile_key]` unpacks via `_deep_stack` to a
        # dense ``(V_hand=2, F=5, 192, 256) uint8`` array (left, right hand).
        # We slice with the same vid_indexes used for video so tactile and
        # video stay temporally aligned, transpose to v0c-A's expected layout
        # ``(V_hand, F, T, H, W)``, and normalize to ``[-1, 1]`` to match
        # Stage 1's tactile preprocessing exactly. This guarantees the frozen
        # v0c-A encoder sees the same input distribution it was trained on.
        if self.read_tactile:
            tactile_rows = data[self.tactile_key].to_list()
            tactile_clips_per_t = [_deep_stack(tactile_rows[i]) for i in vid_indexes]
            tactile = np.stack(tactile_clips_per_t, axis=0)              # (T, V_hand=2, F=5, 192, 256) uint8
            tactile = tactile.astype(np.float32) / 127.5 - 1.0           # [-1, 1]
            tactile = np.transpose(tactile, (1, 2, 0, 3, 4))             # (V_hand, F, T, 192, 256)
            tactile = torch.from_numpy(np.ascontiguousarray(tactile))
        else:
            tactile = None

        if self.read_tactile_flow:
            if "tactile_flow" not in data:
                raise KeyError(
                    "DexVTAMDataset: read_tactile_flow=True but parquet has no "
                    "'tactile_flow' column."
                )
            flow_rows = data["tactile_flow"].to_list()
            flow_clips_per_t = [_deep_stack(flow_rows[i]) for i in vid_indexes]
            flow = np.stack(flow_clips_per_t, axis=0).astype(np.float32)  # (T, V_hand, F, 24, 32, 4)
            flow = flow[..., list(_FLOW_CHANNELS_USED)]                    # (T, V_hand, F, 24, 32, 3)
            flow = (flow - self._flow_mean.reshape(1, 1, 1, 1, 1, 3)) / (
                self._flow_std.reshape(1, 1, 1, 1, 1, 3) + 1e-6
            )
            flow = np.transpose(flow, (1, 2, 0, 3, 4, 5))                # (V_hand, F, T, 24, 32, 3)
            tactile_flow = torch.from_numpy(np.ascontiguousarray(flow.astype(np.float32)))
        else:
            tactile_flow = None

        # hand_pose / tactile T-axis sanity (matches `_get_batch_from_cache`).
        if hand_pose_tensor is not None and tactile is not None:
            if hand_pose_tensor.shape[1] != tactile.shape[2]:
                raise ValueError(
                    f"hand_pose T-axis ({hand_pose_tensor.shape[1]}) != "
                    f"tactile T-axis ({tactile.shape[2]}); index mismatch."
                )

        return videos, action, caption, state, tactile, tactile_flow, hand_pose_tensor



    def __len__(self):
        return self.length



    def __getitem__(self, idx):        
        
        # video, actions, caption, state, tactile, tactile_flow, hand_pose = self.get_batch(idx)

        if self.fix_epiidx is not None:
            video, actions, caption, state, tactile, tactile_flow, hand_pose = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, actions, caption, state, tactile, tactile_flow, hand_pose = self.get_batch(idx)
                    break
                except:
                    ### print error information to debug
                    traceback.print_exc()
                    ### 
                    idx = random.randint(0, self.length-1)
                    
        sample = dict(
            video=video,
            actions=actions,
            caption=caption,
            state=state,
        )
        # Only include tactile when actually populated. The strict GE baseline
        # (read_tactile=False) keeps the sample dict identical to libero's
        # output for byte-level reproducibility against vanilla GE training.
        if tactile is not None:
            sample["tactile"] = tactile
        if tactile_flow is not None:
            sample["tactile_flow"] = tactile_flow
        # Only include hand_pose when read_hand_pose=True. Trainer code paths
        # gate on `self.tactile_vae.use_pose_injection` to consume it; for
        # v0c-A configs the key is simply absent (no-op).
        if hand_pose is not None:
            sample["hand_pose"] = hand_pose
        # Provenance for offline analysis (t-SNE colouring by episode / by how
        # far into the episode a clip sits). Off by default so the sample dict
        # handed to the trainer is byte-identical to before.
        if self.return_meta:
            sample["meta_episode"] = int(
                self.fix_epiidx if self.fix_epiidx is not None else idx)
            sample["meta_phase"] = float(getattr(self, "_meta_phase", -1.0))
        return sample

