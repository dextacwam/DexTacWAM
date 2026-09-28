"""Stage 1 Tactile VAE dataset.

Loads LeRobot-format parquet episodes and yields per-hand tactile clips suitable
for the :class:`models.tactile_models.TactileVAE` forward signature::

    sample = {
        "tactile":      (5, 1, T, 192, 256)  float32 in [-1, 1],
        "tactile_flow": (5, T, 24, 32, 3)    float32, normalized (dx, dy, divergence),
        "hand_pose":    (22,)                float32, normalized,
        "meta":         {"episode_index": int, "hand": int, "start_frame": int, "T": int},
    }

The mixed ``T ∈ {1, 9}`` schedule from plan §5–§7 is *not* implemented inside
``__getitem__`` (which would force dynamic shapes within a batch). Instead, ``T``
is part of the sample identity and ``MixedTBatchSampler`` groups indices so each
batch is homogeneous in ``T``.

Data layout (verified against ``meta/info.json`` and ``vtam_data_scripts``):

* Each parquet row = one frame.
* ``tactile`` is a nested ``(2,)`` object array → ``(2, 5, 192, 256)`` uint8.
* ``tactile_flow`` is a nested ``(2,)`` object array → ``(2, 5, 24, 32, 4)`` float32
  with channels ``(dx, dy, magnitude, divergence)``. We use ``[0, 1, 3]``.
* ``state`` is ``(58,)`` ``[left_arm (7), left_hand (22), right_arm (7), right_hand (22)]``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Channel order for tactile_flow as stored in the parquet:
#   c0 = dx, c1 = dy, c2 = magnitude (= sqrt(dx^2+dy^2), redundant), c3 = divergence
# Stage 1 uses (dx, dy, divergence) only.
_FLOW_CHANNELS_USED = (0, 1, 3)

# Hand-pose slices into the 58-dim state vector.
_LEFT_HAND_SLICE = slice(7, 29)     # 22 dims
_RIGHT_HAND_SLICE = slice(36, 58)   # 22 dims

# Columns we actually need from the parquet (skip head_img / actions to cut IO).
_PARQUET_COLUMNS = ["tactile", "tactile_flow", "state", "frame_index", "episode_index"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_stack(arr: np.ndarray) -> np.ndarray:
    """Recursively stack a nested-object numpy array into a single dense array.

    LeRobot stores multi-dim arrays as nested ``object`` arrays (each axis a
    Python ``ndarray`` of ``ndarray``). This rebuilds the dense form in one
    pass; uses ``np.asarray`` on the leaves and ``np.stack`` on the way up.
    """
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return np.stack([_deep_stack(x) for x in arr])
    return np.asarray(arr)


def _load_stats(path: Optional[str], expected_dim: int, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load ``{"mean": [...], "std": [...]}`` JSON; return (mean, std) as float32."""
    if path is None:
        return None
    if not os.path.exists(path):
        # Stats files are produced by chunk G; if they are missing we silently
        # fall through so the dataset is still smoke-testable end-to-end. The
        # trainer (chunk D) is responsible for failing loudly in that case.
        return None
    with open(path) as f:
        d = json.load(f)
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
        raise ValueError(
            f"{name} stats at {path} have shapes {mean.shape}/{std.shape}; "
            f"expected ({expected_dim},)/({expected_dim},)."
        )
    return mean, std


# ---------------------------------------------------------------------------
# Sample identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SampleSpec:
    """A single sample's identity. T is part of the spec so the BatchSampler
    can group homogeneous batches.

    ``dataset_idx`` namespaces ``episode_index`` so two source datasets can
    reuse the same numeric episode id without collision (multi-root mode).
    For single-root usage it is always ``0``.
    """

    episode_index: int
    hand: int                # 0 = left, 1 = right
    start_frame: int         # absolute frame index in the episode (raw 30 Hz)
    T: int                   # clip length (1 or 9)
    dataset_idx: int = 0     # which entry in ``TactileDataset.datasets`` (0 in single-root mode)


@dataclass(frozen=True)
class _DatasetEntry:
    """One source dataset's (data_root, episodes) pair after argument normalization."""

    data_root: str
    episodes: Tuple[int, ...]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TactileDataset(Dataset):
    """Per-hand tactile clip dataset for Stage 1 VAE training.

    Two equivalent constructor forms are supported:

    1. **Single-root (legacy / v1)**::

           TactileDataset(data_root="/path/to/lerobot_root", episodes=[0..7], ...)

    2. **Multi-root (v2 data-scaling)** -- mix two or more LeRobot dataset
       roots into one logical dataset. Each entry contributes its own
       ``data_root`` and ``episodes`` list; samples are namespaced by
       ``dataset_idx`` so episode ids do not collide across roots::

           TactileDataset(
               datasets=[
                   {"data_root": "/data/.../wipe_plate",    "episodes": [0..19]},
                   {"data_root": "/data/.../wrap_adhesive", "episodes": [0..34]},
               ],
               ...
           )

    The legacy form is internally normalized to a single-entry list.
    ``MixedTBatchSampler`` and ``FixedTBatchSampler`` are unaware of the
    multi-root split -- they only see the flat ``indices_by_T`` lists, so a
    batch can contain samples drawn from any combination of source datasets.

    Args:
        data_root: LeRobot dataset root (single-root form). Mutually exclusive
            with ``datasets``.
        episodes: episode indices to include (single-root form). Mutually
            exclusive with ``datasets``.
        datasets: list of ``{"data_root": str, "episodes": Sequence[int]}``
            dicts (multi-root form). Mutually exclusive with
            ``data_root`` / ``episodes``.
        T_choices: clip lengths to materialize as samples. The default
            ``(1, 9)`` matches the plan's mixed-T schedule.
        temporal_stride: spacing (in raw 30 Hz frames) between *frames inside a
            clip*. ``stride=6`` gives a ~5 Hz effective rate.
        clip_start_stride: spacing between consecutive *clip start frames*.
            Default ``=temporal_stride`` so successive samples are
            non-overlapping at the chosen rate. Set to 1 for max coverage
            (heavy overlap between samples).
        hands: ``"left"``, ``"right"``, or ``"both"``.
        flow_stats_path / pose_stats_path: optional JSON files with
            ``{"mean": [...], "std": [...]}`` for per-channel / per-joint
            normalization. If ``None`` or missing, no normalization is applied
            (caller's responsibility to pre-normalize during chunk-G stats run).
            In multi-root mode the same stats are applied to every dataset --
            run ``compute_tactile_stats.py`` over the *combined* train corpus
            before training.
        cache_episodes_in_memory: if True (default), each worker keeps the
            recovered ``(num_frames, ...)`` arrays of every accessed episode in
            memory (~660 MB / episode). Turn off if you have many workers and
            limited RAM.
        pose_mode: ``"last_frame"`` (default, v0c-A behavior) returns
            ``hand_pose`` as ``(22,)`` -- the pose at the LAST frame of the
            clip, matching the AuxPoseDecoder cycle-consistency target.
            ``"per_frame"`` (v0d) returns ``hand_pose`` as ``(T, 22)`` --
            the full per-frame pose trajectory of the clip, suitable for
            the v0d pose-injection adapter which resamples to ``T_lat`` via
            ``VisualVAEAdapterModel._align_pose_to_lat``. v0d's
            ``_align_pose_to_lat`` is a strict information superset: for
            T=9, T_lat=2 it picks indices ``[0, 8]`` so slot 1 is bit-
            identical to ``last_frame`` mode's value.
    """

    HANDS_LEFT = 0
    HANDS_RIGHT = 1

    _POSE_MODES = ("last_frame", "per_frame")

    def __init__(
        self,
        data_root: Optional[str] = None,
        episodes: Optional[Sequence[int]] = None,
        T_choices: Sequence[int] = (1, 9),
        temporal_stride: int = 6,
        clip_start_stride: Optional[int] = None,
        hands: str = "both",
        flow_stats_path: Optional[str] = None,
        pose_stats_path: Optional[str] = None,
        cache_episodes_in_memory: bool = True,
        datasets: Optional[Sequence[Dict]] = None,
        pose_mode: str = "last_frame",
    ):
        super().__init__()

        if hands not in ("left", "right", "both"):
            raise ValueError(f"hands must be 'left', 'right', or 'both'; got {hands!r}.")
        if any(t < 1 for t in T_choices):
            raise ValueError(f"T_choices must be positive; got {T_choices}.")
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be >= 1; got {temporal_stride}.")
        if pose_mode not in self._POSE_MODES:
            raise ValueError(
                f"pose_mode must be one of {self._POSE_MODES}; got "
                f"{pose_mode!r}."
            )
        self.pose_mode = pose_mode

        # Normalize the (data_root, episodes) vs datasets= argument forms into a
        # single tuple of _DatasetEntry so the rest of __init__ is shape-agnostic.
        self.datasets: Tuple[_DatasetEntry, ...] = self._normalize_dataset_args(
            data_root=data_root, episodes=episodes, datasets=datasets,
        )

        self.T_choices = tuple(T_choices)
        self.temporal_stride = int(temporal_stride)
        self.clip_start_stride = (
            int(clip_start_stride) if clip_start_stride is not None else self.temporal_stride
        )
        self.hands_setting = hands
        self.cache_episodes_in_memory = bool(cache_episodes_in_memory)

        # Resolve which hand indices to enumerate.
        if hands == "left":
            self._active_hands = (self.HANDS_LEFT,)
        elif hands == "right":
            self._active_hands = (self.HANDS_RIGHT,)
        else:
            self._active_hands = (self.HANDS_LEFT, self.HANDS_RIGHT)

        # Read episode lengths from each root's meta/episodes.jsonl so we do
        # not open every parquet up-front. Keys are namespaced (dataset_idx,
        # episode_index) so two roots with the same episode id do not collide.
        self._episode_lengths: Dict[Tuple[int, int], int] = {}
        for ds_idx, entry in enumerate(self.datasets):
            ep_meta_path = os.path.join(entry.data_root, "meta", "episodes.jsonl")
            ep_lengths_one_root: Dict[int, int] = {}
            with open(ep_meta_path) as f:
                for line in f:
                    row = json.loads(line)
                    ep_lengths_one_root[int(row["episode_index"])] = int(row["length"])
            for ep in entry.episodes:
                if ep not in ep_lengths_one_root:
                    raise ValueError(
                        f"Episode {ep} not present in {ep_meta_path} "
                        f"(dataset_idx={ds_idx}, root={entry.data_root})."
                    )
                self._episode_lengths[(ds_idx, ep)] = ep_lengths_one_root[ep]

        # Build the flat index of all (dataset_idx, episode, hand, start, T)
        # samples. T is part of the spec so MixedTBatchSampler can group
        # homogeneous-T batches.
        self._samples: List[_SampleSpec] = []
        # Parallel structure: indices grouped by T for the BatchSampler.
        self._indices_by_T: Dict[int, List[int]] = {t: [] for t in self.T_choices}
        for ds_idx, entry in enumerate(self.datasets):
            for ep in entry.episodes:
                length = self._episode_lengths[(ds_idx, ep)]
                for T in self.T_choices:
                    # A clip [start, start+stride, ..., start+(T-1)*stride]
                    # needs the last frame to be within [0, length-1].
                    last_offset = (T - 1) * self.temporal_stride
                    if last_offset >= length:
                        continue
                    max_start = length - 1 - last_offset
                    for start in range(0, max_start + 1, self.clip_start_stride):
                        for hand in self._active_hands:
                            flat_idx = len(self._samples)
                            self._samples.append(
                                _SampleSpec(
                                    episode_index=ep,
                                    hand=hand,
                                    start_frame=start,
                                    T=T,
                                    dataset_idx=ds_idx,
                                )
                            )
                            self._indices_by_T[T].append(flat_idx)

        if not self._samples:
            raise RuntimeError(
                "No samples generated. Check episode lengths vs T_choices and stride."
            )

        # Backward-compat aliases for callers that read these attributes
        # directly. In multi-root mode they reflect the FIRST entry only -- new
        # code should use ``self.datasets`` instead.
        self.data_root = self.datasets[0].data_root
        self.episodes = list(self.datasets[0].episodes)

        # Normalization stats (optional).
        flow_stats = _load_stats(flow_stats_path, expected_dim=len(_FLOW_CHANNELS_USED), name="flow")
        pose_stats = _load_stats(pose_stats_path, expected_dim=22, name="pose")
        if flow_stats is not None:
            mean, std = flow_stats
            # Broadcast against (5, T, 24, 32, 3) — channel last.
            self._flow_mean = mean.reshape(1, 1, 1, 1, -1)
            self._flow_std = np.maximum(std, 1e-6).reshape(1, 1, 1, 1, -1)
        else:
            self._flow_mean = None
            self._flow_std = None
        if pose_stats is not None:
            mean, std = pose_stats
            self._pose_mean = mean
            self._pose_std = np.maximum(std, 1e-6)
        else:
            self._pose_mean = None
            self._pose_std = None

        # Per-worker cache of recovered episode arrays. Initialised lazily.
        # NOTE: pickled at worker fork; entries fill in as workers touch episodes.
        # Keys are (dataset_idx, episode_index) to namespace across multiple roots.
        self._episode_cache: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Argument normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_dataset_args(
        data_root: Optional[str],
        episodes: Optional[Sequence[int]],
        datasets: Optional[Sequence[Dict]],
    ) -> Tuple[_DatasetEntry, ...]:
        """Validate and convert the dual-form constructor args into a tuple of
        :class:`_DatasetEntry`. Exactly one of ``(data_root, episodes)`` or
        ``datasets`` must be provided."""
        single_form = data_root is not None or episodes is not None
        multi_form = datasets is not None
        if single_form and multi_form:
            raise ValueError(
                "Pass either (data_root, episodes) for single-root mode OR "
                "datasets=[{...}, ...] for multi-root mode, not both."
            )
        if not single_form and not multi_form:
            raise ValueError(
                "Must pass either (data_root, episodes) or datasets=[{...}, ...]."
            )

        if single_form:
            if data_root is None or episodes is None:
                raise ValueError(
                    "Single-root mode requires both data_root and episodes."
                )
            return (_DatasetEntry(data_root=str(data_root), episodes=tuple(int(e) for e in episodes)),)

        if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
            raise ValueError("datasets must be a non-empty list of dicts.")

        entries = []
        for i, entry in enumerate(datasets):
            if not isinstance(entry, dict):
                raise ValueError(f"datasets[{i}] must be a dict; got {type(entry).__name__}.")
            try:
                root_i = entry["data_root"]
                eps_i = entry["episodes"]
            except KeyError as e:
                raise ValueError(
                    f"datasets[{i}] missing required key {e!s}; expected "
                    f"keys 'data_root' and 'episodes'."
                ) from None
            if not isinstance(eps_i, (list, tuple, range)):
                raise ValueError(
                    f"datasets[{i}].episodes must be list/tuple/range; got {type(eps_i).__name__}."
                )
            entries.append(_DatasetEntry(data_root=str(root_i), episodes=tuple(int(e) for e in eps_i)))
        return tuple(entries)

    # ------------------------------------------------------------------
    # Public read-only API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def indices_by_T(self) -> Dict[int, List[int]]:
        """Mapping ``T -> list of dataset indices with that T``. Used by
        :class:`MixedTBatchSampler`."""
        return self._indices_by_T

    @property
    def sample_specs(self) -> List[_SampleSpec]:
        return self._samples

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec = self._samples[idx]
        episode_data = self._get_episode(spec.dataset_idx, spec.episode_index)

        frame_idxs = (
            spec.start_frame
            + np.arange(spec.T, dtype=np.int64) * self.temporal_stride
        )

        # Tactile clip: (T, 2, 5, 192, 256) -> (5, 1, T, 192, 256) for selected hand.
        tactile_clip = episode_data["tactile"][frame_idxs, spec.hand]  # (T, 5, 192, 256) uint8
        tactile_clip = tactile_clip.astype(np.float32) / 127.5 - 1.0    # [-1, 1]
        # (T, 5, 192, 256) -> (5, 1, T, 192, 256) — add a channel axis.
        tactile_clip = np.transpose(tactile_clip, (1, 0, 2, 3))[:, None, :, :, :]

        # Flow clip: (T, 2, 5, 24, 32, 4) -> select hand & 3 channels -> (5, T, 24, 32, 3).
        flow_clip_raw = episode_data["tactile_flow"][frame_idxs, spec.hand]
        flow_clip = flow_clip_raw[..., list(_FLOW_CHANNELS_USED)]       # (T, 5, 24, 32, 3)
        flow_clip = np.transpose(flow_clip, (1, 0, 2, 3, 4))            # (5, T, 24, 32, 3)
        if self._flow_mean is not None:
            flow_clip = (flow_clip - self._flow_mean) / self._flow_std

        # Hand pose:
        #   pose_mode="last_frame" (v0c-A default): return (22,) -- the most
        #     recent observation the encoder sees, matching the AuxPoseDecoder
        #     cycle-consistency target.
        #   pose_mode="per_frame"  (v0d): return (T, 22) -- the full per-frame
        #     trajectory of the clip, suitable for the v0d pose-injection
        #     adapter which resamples to T_lat via
        #     VisualVAEAdapterModel._align_pose_to_lat.
        hand_slice = (
            _LEFT_HAND_SLICE if spec.hand == self.HANDS_LEFT else _RIGHT_HAND_SLICE
        )
        if self.pose_mode == "last_frame":
            last_frame = int(frame_idxs[-1])
            state_row = episode_data["state"][last_frame]    # (58,)
            hand_pose = state_row[hand_slice]                # (22,)
            hand_pose = hand_pose.astype(np.float32, copy=True)
            if self._pose_mean is not None:
                hand_pose = (hand_pose - self._pose_mean) / self._pose_std
        else:  # "per_frame"
            state_clip = episode_data["state"][frame_idxs]   # (T, 58)
            hand_pose = state_clip[:, hand_slice]            # (T, 22)
            hand_pose = hand_pose.astype(np.float32, copy=True)
            if self._pose_mean is not None:
                # _pose_mean shape (22,) broadcasts to (T, 22).
                hand_pose = (hand_pose - self._pose_mean) / self._pose_std

        return {
            "tactile": torch.from_numpy(tactile_clip),
            "tactile_flow": torch.from_numpy(np.ascontiguousarray(flow_clip)),
            "hand_pose": torch.from_numpy(np.ascontiguousarray(hand_pose)),
            "meta": {
                "dataset_idx": spec.dataset_idx,
                "episode_index": spec.episode_index,
                "hand": spec.hand,
                "start_frame": spec.start_frame,
                "T": spec.T,
            },
        }

    # ------------------------------------------------------------------
    # Episode loading + cache
    # ------------------------------------------------------------------

    def _get_episode(self, dataset_idx: int, episode_index: int) -> Dict[str, np.ndarray]:
        """Return the recovered (num_frames, ...) arrays for one episode.

        ``dataset_idx`` selects which entry in :attr:`datasets` to resolve the
        parquet path against, and namespaces the per-worker cache so two
        roots can reuse the same numeric ``episode_index``.
        """
        cache_key = (dataset_idx, episode_index)
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            return cached

        data_root = self.datasets[dataset_idx].data_root
        # episode_chunk = episode_index // chunks_size (chunks_size=1000 in info.json)
        episode_chunk = episode_index // 1000
        parquet_path = os.path.join(
            data_root,
            "data",
            f"chunk-{episode_chunk:03d}",
            f"episode_{episode_index:06d}.parquet",
        )
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path, columns=_PARQUET_COLUMNS)
        # Recover dense arrays from nested object arrays.
        tactile = np.stack([_deep_stack(v) for v in df["tactile"].values])           # (N, 2, 5, 192, 256) uint8
        tactile_flow = np.stack([_deep_stack(v) for v in df["tactile_flow"].values]) # (N, 2, 5, 24, 32, 4) float32
        state = np.stack([_deep_stack(v) for v in df["state"].values]).astype(np.float32)  # (N, 58)

        if tactile.dtype != np.uint8:
            tactile = tactile.astype(np.uint8)
        if tactile_flow.dtype != np.float32:
            tactile_flow = tactile_flow.astype(np.float32)

        recovered = {
            "tactile": tactile,
            "tactile_flow": tactile_flow,
            "state": state,
        }
        if self.cache_episodes_in_memory:
            self._episode_cache[cache_key] = recovered
        return recovered


# ---------------------------------------------------------------------------
# Mixed-T batch sampler
# ---------------------------------------------------------------------------


class MixedTBatchSampler(Sampler[List[int]]):
    """Yields batches that are *homogeneous in T* drawn from a TactileDataset.

    For each batch we (i) sample a target ``T`` from ``T_choices`` with
    ``T_sample_ratio`` probabilities, then (ii) draw ``batch_size`` indices
    from the dataset's pool of samples with that ``T``.

    Why homogeneous-T batches: a batch with mixed ``T`` would force a dynamic
    shape inside the model and break vectorization across the batch axis.
    Plan §7 calls this out explicitly.

    Args:
        dataset: a :class:`TactileDataset` (used for its ``indices_by_T`` map
            and ``T_choices``).
        batch_size: number of samples per batch.
        T_sample_ratio: per-T probability. Must align with
            ``dataset.T_choices`` order.
        num_batches: total batches per epoch. Defaults to
            ``len(dataset) // batch_size``.
        seed: RNG seed (per-replica, per-epoch random state can be set via
            :meth:`set_epoch`).
        drop_last: ignored (we draw exactly ``num_batches`` × ``batch_size``
            samples; the last "incomplete" batch concept does not apply when
            we sample with replacement within a T pool).
    """

    def __init__(
        self,
        dataset: TactileDataset,
        batch_size: int,
        T_sample_ratio: Sequence[float],
        num_batches: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = True,  # noqa: ARG002 — kept for API parity
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        if len(T_sample_ratio) != len(dataset.T_choices):
            raise ValueError(
                f"T_sample_ratio (len {len(T_sample_ratio)}) must align with "
                f"dataset.T_choices (len {len(dataset.T_choices)})."
            )
        ratio_sum = float(sum(T_sample_ratio))
        if ratio_sum <= 0:
            raise ValueError("T_sample_ratio must sum to a positive value.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.T_choices = tuple(dataset.T_choices)
        self.T_sample_probs = np.asarray(T_sample_ratio, dtype=np.float64) / ratio_sum
        self.num_batches = (
            int(num_batches) if num_batches is not None else len(dataset) // batch_size
        )
        if self.num_batches <= 0:
            raise ValueError("num_batches must be > 0.")

        self.seed = int(seed)
        self._epoch = 0

        # Sanity: every T must have at least batch_size samples available.
        for T in self.T_choices:
            n = len(dataset.indices_by_T[T])
            if n < self.batch_size:
                raise RuntimeError(
                    f"T={T} pool has only {n} samples (< batch_size={self.batch_size}); "
                    "increase data, decrease batch_size, or drop this T from T_choices."
                )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic per-epoch shuffling."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        for _ in range(self.num_batches):
            T = int(rng.choice(self.T_choices, p=self.T_sample_probs))
            pool = self.dataset.indices_by_T[T]
            # Sample WITHOUT replacement within a batch (keeps diversity), but
            # WITH replacement across batches (pool is small for low-T cases on
            # tiny datasets).
            batch = rng.choice(len(pool), size=self.batch_size, replace=False)
            yield [pool[i] for i in batch]

    def __len__(self) -> int:
        return self.num_batches


# ---------------------------------------------------------------------------
# Fixed-T batch sampler  (used by validation, where each T is reported separately)
# ---------------------------------------------------------------------------


class FixedTBatchSampler(Sampler[List[int]]):
    """Yields batches with a single fixed ``T`` value.

    Used by validation so we can report ``val/T1/*`` and ``val/T9/*`` losses
    separately without conflating the two regimes.

    Args:
        dataset: a :class:`TactileDataset`.
        T: the fixed ``T`` to draw samples for. Must be in ``dataset.T_choices``.
        batch_size: number of samples per batch.
        num_batches: how many batches to yield per ``__iter__``. Defaults to
            ``len(pool) // batch_size`` (one full pass without replacement).
        seed: RNG seed (per-epoch shuffle controlled via :meth:`set_epoch`).
        shuffle: if True, shuffle the pool before chunking.
    """

    def __init__(
        self,
        dataset: TactileDataset,
        T: int,
        batch_size: int,
        num_batches: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = True,
    ):
        super().__init__()
        if T not in dataset.T_choices:
            raise ValueError(f"T={T} is not in dataset.T_choices={dataset.T_choices}.")
        self.dataset = dataset
        self.T = int(T)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

        self._pool = list(dataset.indices_by_T[T])
        if not self._pool:
            raise RuntimeError(f"No samples with T={T} in dataset.")
        max_full_batches = len(self._pool) // self.batch_size
        if num_batches is None:
            self.num_batches = max(1, max_full_batches)
        else:
            self.num_batches = int(num_batches)
            if self.num_batches > max_full_batches:
                # Allow over-subscription (we'll re-shuffle the pool).
                pass

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        order = np.arange(len(self._pool))
        if self.shuffle:
            rng.shuffle(order)
        for i in range(self.num_batches):
            start = (i * self.batch_size) % len(self._pool)
            end = start + self.batch_size
            if end <= len(self._pool):
                chunk = order[start:end]
            else:
                # Wrap around if the pool is smaller than num_batches * batch_size.
                chunk = np.concatenate([order[start:], order[: end - len(self._pool)]])
            yield [self._pool[j] for j in chunk]

    def __len__(self) -> int:
        return self.num_batches
