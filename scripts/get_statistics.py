# Modified by the DexTacWAM Authors, 2026.
# Originally from Genie-Envisioner (AgibotTech) at commit d54425c4.

import os
import sys
import numpy as np
import pandas as pd
import tqdm
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils.relative_action import (  # noqa: E402
    LAYOUTS,
    assert_layout_dims,
    build_relative_action_from_window,
    get_arm_layout,
)


# ---------------------------------------------------------------------------
# Shared-force-range groups
# ---------------------------------------------------------------------------
# A "shared force range" group says: dims [start:end] of the named block
# should all use a single (shared) q01/q99 value, instead of per-dim values.
# The shared lower bound is the most-negative q01 in the group, the shared
# upper bound is the most-positive q99 in the group.
#
# Why: for force-sensor blocks where many dims are pure noise / never
# activate (e.g. LEFT tactile force in a right-hand-only task), per-dim
# normalization stretches the noise to fill [-1, 1], which the model then
# has to chase. Sharing the range across the whole sensor group lets the
# noise dims collapse to ~constant in normalized space, while the active
# dims keep their full dynamic range. See
# ``vtam_data_scripts/patch_stats_shared_force_range.py`` for a standalone
# post-hoc patcher that does the same thing on an already-emitted stats JSON.
#
# CLI format: ``--shared-force-range <which>:<start>:<end>`` where <which> ∈
# {joint, delta_joint, state_joint}. Repeatable.
#
# The implementation is dim-agnostic: it works for any contiguous block of
# dims, not just force. The name reflects the intended use case.

_WHICH_TO_SUFFIX = {
    "joint":          "",           # data_name + "_" + data_type
    "delta_joint":    "_delta",     # data_name + "_delta_" + data_type
    "state_joint":    "_state",     # data_name + "_state_" + data_type
    "relative_joint": "_relative",  # data_name + "_relative_" + data_type (relative_eef_rot6d)
}


def load_data(data_path, key="action"):
    # NOTE: pass columns=[key] so pyarrow only materializes the requested
    # column. Parquet files in lerobot datasets can carry large image / tactile
    # tensors per row (~700MB / file); loading them just to read a 1-D action
    # array would dominate the runtime.
    data = pd.read_parquet(data_path, columns=[key])
    data = np.stack([data[key][i] for i in range(data[key].shape[0])])
    return data 


def cal_statistic(data, _filter=True):
    q99 = np.percentile(data, 99, axis=0)
    q01 = np.percentile(data,  1, axis=0)
    if _filter:
        data_mask = (data>=q01) & (data <= q99)
        data_mask = data_mask.min(axis=1)
        data = data[data_mask, :]
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return means, stds, q99, q01


def parse_shared_force_range_spec(spec: str) -> tuple[str, int, int]:
    """Parse '<which>:<start>:<end>' -> (which, start, end), validated.

    Implementation is generic (works for any dim group), but the public name
    reflects the dominant use case: a contiguous block of force-sensor dims
    that should share one q01/q99 across the group.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"bad --shared-force-range spec {spec!r}; expected '<which>:<start>:<end>'")
    which, start_s, end_s = parts
    if which not in _WHICH_TO_SUFFIX:
        raise argparse.ArgumentTypeError(
            f"--shared-force-range '{which}' must be one of {sorted(_WHICH_TO_SUFFIX)}")
    try:
        start, end = int(start_s), int(end_s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"non-integer start/end in {spec!r}") from e
    if not (0 <= start < end):
        raise argparse.ArgumentTypeError(f"need 0 <= start < end in {spec!r}")
    return which, start, end


def apply_shared_force_range(statistics_info: dict, data_name: str, data_type: str,
                             which: str, start: int, end: int) -> None:
    """Mutate ``statistics_info`` so dims [start:end] of the (data_name, which)
    block share a single q01 (group min) and q99 (group max).

    Per-dim ``mean`` / ``std`` are left untouched because the dataloader uses
    only q01 / q99 for normalization; mean / std remain accurate documentary
    statistics of the raw signal.
    """
    suffix = _WHICH_TO_SUFFIX[which]
    key = f"{data_name}{suffix}_{data_type}"
    if key not in statistics_info:
        raise KeyError(f"stats has no key {key!r}; available: {list(statistics_info)}")
    sub = statistics_info[key]
    for field in ("q01", "q99"):
        arr = np.asarray(sub[field], dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{key!r}[{field!r}] expected 1-D, got {arr.shape}")
        if end > arr.shape[0]:
            raise ValueError(f"end {end} > dim count {arr.shape[0]} for {key!r}[{field!r}]")
        # Snapshot pre-mutation range for diagnostics (arr[start:end] is a
        # view, so reading it after the write below would show the shared
        # value instead of the original per-dim spread).
        seg = arr[start:end]
        seg_min, seg_max = float(seg.min()), float(seg.max())
        shared = seg_min if field == "q01" else seg_max
        arr[start:end] = shared
        sub[field] = arr.tolist()
        print(f"  [{key}] {field} dims [{start}:{end}]: "
              f"before per-dim range = [{seg_min:+.4f}, {seg_max:+.4f}]  "
              f"-> shared {field} = {shared:+.4f}")


def get_statistics(data_root, data_name, data_type, save_path,
                   action_key="action", state_key="observation.state",
                   nrnd=50000, _filter=True, shared_force_ranges=None,
                   relative=False, n_previous=None, action_chunk=None, min_std=1e-6,
                   arm_layout="bimanual"):
    """
    shared_force_ranges : optional list of (which, start, end) tuples produced
        by ``parse_shared_force_range_spec``. After per-dim stats are computed,
        each tuple is applied via ``apply_shared_force_range`` so the listed
        dim slice within the named block (joint / delta_joint / state_joint /
        relative_joint) shares a single q01 / q99 across all its dims. Designed
        for force sensor groups where many dims are pure noise, but the
        implementation is dim-agnostic and can be used for any contiguous block.

    arm_layout : EXPLICIT name of the corpus layout (``bimanual`` 150/90/136 or
        ``right_only`` 75/45/68). Never inferred from the parquet width: the observed
        widths are cross-checked against this name instead, so pointing the stats pass
        at the wrong dataset fails loudly rather than emitting a stats file whose dims
        silently disagree with the config that will consume it.

    relative : if True, ALSO compute the ``{data_name}_relative_{data_type}``
        (``rel_action_dim``-D) block for the relative_eef_rot6d action mode, in the SAME pass
        (the whole stats file is regenerated fresh -- nothing is copied from or
        appended to a pre-existing file). The relative action replaces the
        absolute bimanual arm_target_pose (L/R 4x4 = 32-D) with per-arm
        ``rel9 = [xyz(3), rot6d(6)]`` relativized against a single anchor (the
        last observed state row = window row ``n_previous-1``). Its distribution
        is offset-dependent, so it is computed via the SHARED builder over
        STRIDE-1 windows of length ``W = n_previous + action_chunk`` (anchor at
        row ``n_previous-1``), which captures the contiguous future offsets
        +1..+action_chunk exactly. Requires ``n_previous`` and ``action_chunk``.
    """
    assert(data_type in ["joint", "eef"])
    layout = get_arm_layout(arm_layout)
    if relative:
        assert n_previous is not None and action_chunk is not None, \
            "relative=True requires n_previous and action_chunk"
        assert n_previous >= 1 and action_chunk >= 1
        window = n_previous + action_chunk

    data_path_list = os.listdir(data_root)
    data_path_list.sort()
    if nrnd <= len(data_path_list):
        data_path_list = np.random.choice(data_path_list, nrnd)

    data_list = []
    state_list = []
    delta_data_list = []
    rel_rows = [] if relative else None
    n_rel_skipped = 0
    for data_path in tqdm.tqdm(data_path_list):
        p = os.path.join(data_root, data_path)
        data = load_data(p, action_key)
        state = load_data(p, state_key)
        data_list.append(data)
        delta_data = data[1:] - data[:-1]
        delta_data_list.append(delta_data)
        state_list.append(state)
        if relative:
            a_ep = data.astype(np.float32)
            s_ep = state.astype(np.float32)
            assert_layout_dims(layout, abs_action_dim=a_ep.shape[1],
                               state_dim=s_ep.shape[1], where=data_path)
            T = a_ep.shape[0]
            if T < window:
                n_rel_skipped += 1
            else:
                for s0 in range(0, T - window + 1):
                    rel_rows.append(build_relative_action_from_window(
                        a_ep[s0:s0 + window], s_ep[s0:s0 + window], n_previous, layout))

    data_list = np.concatenate(data_list, axis=0)
    assert(len(data_list.shape)==2)
    means, stds, q99, q01 = cal_statistic(data_list, _filter=_filter)

    delta_data_list = np.concatenate(delta_data_list, axis=0)
    assert(len(delta_data_list.shape)==2)
    delta_means, delta_stds, delta_q99, delta_q01 = cal_statistic(delta_data_list, _filter=_filter)

    state_list = np.concatenate(state_list, axis=0)
    assert(len(state_list.shape)==2)
    state_means, state_stds, state_q99, state_q01 = cal_statistic(state_list, _filter=_filter)

    ### example:
    ### data_name=agibotworld, data_type="joint"/"eef"
    ### 
    ### StatisticInfo = {
    ###     "agibotworld_joint": {
    ###         "mean": [
    ###             ...
    ###         ]
    ###         "std": [
    ###             ...
    ###         ]
    ###     "agibotworld_delta_joint": {
    ###         "mean": [
    ###             ...
    ###         ]
    ###         "std": [
    ###             ...
    ###         ]
    ### }
    ###     "agibotworld_state_joint": {
    ###         "mean": [
    ###             ...
    ###         ]
    ###         "std": [
    ###             ...
    ###         ]
    ### }

    statistics_info = dict({
        data_name+"_"+data_type:dict({
            "mean": means.tolist(),
            "std": stds.tolist(),
            "q99": q99.tolist(),
            "q01": q01.tolist(),
        }),
        data_name+"_delta_"+data_type:dict({
            "mean": delta_means.tolist(),
            "std": delta_stds.tolist(),
            "q99": delta_q99.tolist(),
            "q01": delta_q01.tolist(),
        }),
        data_name+"_state_"+data_type:dict({
            "mean": state_means.tolist(),
            "std": state_stds.tolist(),
            "q99": state_q99.tolist(),
            "q01": state_q01.tolist(),
        }),
    })

    # if os.path.exists(save_path):
    #     with open(save_path, "r") as f:
    #         exist_info = json.load(f)
    # else:
    #     exist_info = dict()
    # for k in statistics_info.keys():
    #     assert k not in exist_info

    # relative_eef_rot6d: add the 136-D {data_name}_relative_{data_type} block.
    if relative:
        if n_rel_skipped:
            print(f"  [relative] skipped {n_rel_skipped} episodes shorter than window {window}")
        if not rel_rows:
            raise RuntimeError(
                f"no episode >= window {window} (n_previous={n_previous} + "
                f"action_chunk={action_chunk}); cannot compute relative block.")
        rel_all = np.concatenate(rel_rows, axis=0)
        assert_layout_dims(layout, rel_action_dim=rel_all.shape[1], where="relative block")
        rel_means, rel_stds, rel_q99, rel_q01 = cal_statistic(rel_all, _filter=_filter)
        for nm, arr in (("mean", rel_means), ("std", rel_stds), ("q99", rel_q99), ("q01", rel_q01)):
            assert arr.shape == (layout.rel_action_dim,), f"relative {nm} shape {arr.shape}"
            assert np.all(np.isfinite(arr)), f"relative {nm} has non-finite values"
        # the 9-per-arm relative arm-pose dims MUST vary; a near-constant one signals
        # future-state leakage / a degenerate anchor (rel ~= identity everywhere).
        #
        # The guard inspects the RAW (unfiltered) std, NOT the stored/filtered
        # rel_stds above. cal_statistic(_filter=True) keeps only rows whose EVERY
        # one of the 136 dims lies inside its own [q01,q99]; that central filter
        # collapses bounded-near-1 rot6d diagonal entries (R00=dim121, R11=dim125)
        # to ~constant for a stabilizing arm (e.g. the LEFT arm in a right-hand
        # task), even though the raw signal varies (recon is lossless). Real
        # leakage / identity-collapse would still show ~0 RAW std, so this keeps
        # the guard's intent while dropping the filter false-positive. The stored
        # std stays filtered (consistent with every other block); normalization
        # uses q01/q99 only (see _build_action_target), so std is documentary.
        _, rel_stds_raw, _, _ = cal_statistic(rel_all, _filter=False)
        pose_lo = layout.rel_poses[0][0]
        bad = np.where(rel_stds_raw[pose_lo:layout.rel_action_dim] < min_std)[0] + pose_lo
        if bad.size:
            raise RuntimeError(
                f"relative arm-pose dims {bad.tolist()} have RAW std < {min_std} "
                f"(rel near-constant -> suspect future-state leakage or bad anchor)")
        statistics_info[data_name + "_relative_" + data_type] = {
            "mean": rel_means.tolist(),
            "std": rel_stds.tolist(),
            "q99": rel_q99.tolist(),
            "q01": rel_q01.tolist(),
        }
        print(f"  [relative] added key {data_name + '_relative_' + data_type!r} "
              f"({layout.rel_action_dim}-D, arm_layout={layout.name!r}) "
              f"from {rel_all.shape[0]} rows")

    exist_info = dict()
    exist_info.update(statistics_info)

    # Apply requested shared-force-range groups (in-place on q01/q99).
    if shared_force_ranges:
        print("\napplying shared-force-range groups:")
        for which, start, end in shared_force_ranges:
            apply_shared_force_range(exist_info, data_name, data_type, which, start, end)

    with open(save_path, "w") as f:
        json.dump(exist_info, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="PATH/TO/YOUR/DATASET")
    parser.add_argument('--data_name', default="YOUR_CUSTOM_DATASET")
    parser.add_argument('--data_type', default="joints")
    parser.add_argument('--action_key', default="action")
    parser.add_argument('--state_key', default="observation.state")
    parser.add_argument('--save_path', default="PATH/OF/JSON/FILE")
    parser.add_argument(
        '--shared-force-range', dest='shared_force_range',
        action='append', type=parse_shared_force_range_spec, default=[],
        help="Repeatable. Format '<which>:<start>:<end>' where <which> ∈ "
             "{joint, delta_joint, state_joint, relative_joint}. Dims [start:end] "
             "within that block will share a single q01/q99 across the group, "
             "instead of per-dim values. Designed for force sensor groups where "
             "many dims are pure noise (e.g. LEFT tactile force in a right-hand "
             "task): '--shared-force-range joint:0:60 --shared-force-range "
             "delta_joint:0:60'. (Works for any contiguous block of dims, not "
             "just force.)",
    )
    parser.add_argument(
        '--relative', action='store_true',
        help="relative_eef_rot6d mode: ALSO compute the {data_name}_relative_"
             "{data_type} block in the SAME fresh pass (whole stats file "
             "regenerated; nothing copied/appended). Requires --n-previous and "
             "--action-chunk to match the training config.",
    )
    parser.add_argument(
        '--arm-layout', dest='arm_layout', default='bimanual', choices=sorted(LAYOUTS),
        help="EXPLICIT corpus layout, must match the training config's arm_layout: "
             "'bimanual' = 150/90 -> 136 (erase/handover/chip/unscrew), 'right_only' = "
             "75/45 -> 68 (right-hand-only tong/bowl). The observed parquet widths are "
             "cross-checked against it; never inferred from them.",
    )
    parser.add_argument('--n-previous', dest='n_previous', type=int, default=None,
                        help="(relative) number of memory frames; anchor = window row n_previous-1.")
    parser.add_argument('--action-chunk', dest='action_chunk', type=int, default=None,
                        help="(relative) number of future action frames per window.")

    args = parser.parse_args()

    if args.relative and (args.n_previous is None or args.action_chunk is None):
        parser.error("--relative requires --n-previous and --action-chunk")

    get_statistics(
        args.data_root, args.data_name, args.data_type, args.save_path,
        action_key=args.action_key, state_key=args.state_key,
        shared_force_ranges=args.shared_force_range,
        relative=args.relative, n_previous=args.n_previous,
        action_chunk=args.action_chunk, arm_layout=args.arm_layout,
    )
