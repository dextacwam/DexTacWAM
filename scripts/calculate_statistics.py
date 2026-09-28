import os
import numpy as np
import pandas as pd
import tqdm
import json
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def load_data(data_path, key="action"):
    """Load a single (T, C) column from a LeRobot-style parquet file.

    Only the requested column is read; this is critical for corpora where
    other columns are bulky (e.g. tactile uint8 (2,5,192,256), tactile_flow
    float32 (2,5,24,32,4), or PNG-encoded image columns) -- loading those
    eagerly from every worker easily exhausts memory and triggers
    BrokenProcessPool on a multi-worker pool.
    """
    data = pd.read_parquet(data_path, columns=[key])
    data = np.stack([data[key][i] for i in range(data[key].shape[0])])
    return data

def unwrap_angles(data, angle_dims=[3, 4, 5]):
    """
    Apply unwrap to angle dimensions (Euler angles) to remove 2π discontinuities.
    
    Args:
        data: (T, C) numpy array
        angle_dims: list of dimension indices to unwrap (default: [3,4,5] for roll, pitch, yaw)
    
    Returns:
        data with unwrapped angles
    """
    T, C = data.shape
    if T > 0 and C >= 6:
        for d in angle_dims:
            if d < C:
                data[:, d] = np.unwrap(data[:, d])
    return data

def process_single_file(parquet_path, action_key, state_key, is_calvin_eef, data_type, check_jumps=False, pos_threshold=50.0, ori_threshold=1.0, joint_threshold=1.0, gripper_threshold=50.0, gripper_dim_index=None, force_dims=None, force_threshold=10.0, unwrap_eef_angles=False, extract_7d_state=False):
    """
    Process a single parquet file and return action, delta_action, and state.
    This function is designed to be called in parallel.

    Args:
        check_jumps: if True, check for large jumps in action trajectory
        pos_threshold: threshold for position jump (meters, default: 0.1m = 10cm)
        ori_threshold: threshold for orientation jump (radians, default: 1.0 rad ≈ 57°)
        gripper_dim_index: which action dimension is the gripper (opt-in). When
            ``None`` (the default) no dimension is treated as gripper -- joint
            and force thresholds cover every dim. Pass an explicit index for
            CALVIN / agibot / any corpus where the last action dim is a gripper
            command. Tactile / no-gripper datasets should leave this unset.
        gripper_threshold: threshold for gripper jump (dataset units), only
            consulted when ``gripper_dim_index is not None``.

    Returns:
        tuple: (action, delta_action, state, warnings) or None if error
    """
    try:
        # Load action data
        action = load_data(parquet_path, action_key)
        
        # Apply unwrap to EEF angles if needed
        if (is_calvin_eef or unwrap_eef_angles) and data_type == "eef":
            action = unwrap_angles(action, angle_dims=[3, 4, 5])
        
        # Compute delta within this episode (not across episodes!)
        delta_action = None
        if len(action) > 1:
            delta_action = action[1:] - action[:-1]
        
        # Check for large jumps if requested
        warnings = []
        if check_jumps and delta_action is not None and len(delta_action) > 0:
            T, C = delta_action.shape
            if data_type == "eef":
                # Check first 6 dimensions (EEF: pos_x, pos_y, pos_z, roll, pitch, yaw)
                if C >= 6:
                    for t in range(T):
                        # Check position (x, y, z) in mm
                        pos_delta = delta_action[t, :3]
                        pos_jump = np.linalg.norm(pos_delta)
                        if pos_jump > pos_threshold:
                            warnings.append({
                                "episode": parquet_path,
                                "timestep": t + 1,  # +1 because delta is between t and t+1
                                "type": "position",
                                "jump_magnitude": float(pos_jump),
                                "threshold": pos_threshold,
                                "delta": pos_delta.tolist()
                            })
                        
                        # Check orientation (roll, pitch, yaw) in rad
                        ori_delta = delta_action[t, 3:6]
                        for i, angle_name in enumerate(['roll', 'pitch', 'yaw']):
                            angle_jump = abs(ori_delta[i])
                            if angle_jump > ori_threshold:
                                warnings.append({
                                    "episode": parquet_path,
                                    "timestep": t + 1,
                                    "type": f"orientation_{angle_name}",
                                    "jump_magnitude": float(angle_jump),
                                    "threshold": ori_threshold,
                                    "delta": float(ori_delta[i])
                                })
            else:
                # Joint space: all dims are in radians (optionally one gripper
                # dim + a few force dims). Gripper detection is opt-in: when
                # gripper_dim_index is None we use a sentinel that never
                # matches any valid action index, so every dim falls through
                # to the joint / force branches. This makes the script correct
                # for tactile / no-gripper datasets without any flag changes.
                gripper_dim = gripper_dim_index if gripper_dim_index is not None else -1
                for t in range(T):
                    for i in range(C):
                        angle_jump = abs(delta_action[t, i])
                        if i == gripper_dim:
                            if angle_jump > gripper_threshold:
                                warnings.append({
                                    "episode": parquet_path,
                                    "timestep": t + 1,
                                    "type": "gripper",
                                    "jump_magnitude": float(angle_jump),
                                    "threshold": gripper_threshold,
                                    "delta": float(delta_action[t, i])
                                })
                        elif force_dims and i in force_dims:
                            if angle_jump > force_threshold:
                                warnings.append({
                                    "episode": parquet_path,
                                    "timestep": t + 1,
                                    "type": f"force_{i}",
                                    "jump_magnitude": float(angle_jump),
                                    "threshold": force_threshold,
                                    "delta": float(delta_action[t, i])
                                })
                        else:
                            if angle_jump > joint_threshold:
                                warnings.append({
                                    "episode": parquet_path,
                                    "timestep": t + 1,
                                    "type": f"joint_{i}",
                                    "jump_magnitude": float(angle_jump),
                                    "threshold": joint_threshold,
                                    "delta": float(delta_action[t, i])
                                })
        
        # Load state data
        state = load_data(parquet_path, state_key)
        
        # Apply unwrap to state angles if needed
        if (is_calvin_eef or unwrap_eef_angles) and data_type == "eef":
            state = unwrap_angles(state, angle_dims=[3, 4, 5])
        
        # Optionally extract 7D EEF state from 15D CALVIN state
        if (is_calvin_eef or extract_7d_state) and data_type == "eef":
            if state.shape[-1] == 15:
                pos_ori = state[:, :6]      # EE position + orientation
                grip_act = state[:, -1:]    # gripper action (last dim)
                state = np.concatenate([pos_ori, grip_act], axis=-1)
        
        return (action, delta_action, state, warnings)
        
    except Exception as e:
        print(f"\nError processing {parquet_path}: {e}")
        return None

def cal_statistic(data, _filter=True):
    """
    Calculate statistics.

    - mean/std are used for zscore-style normalization.
    - q01/q99 are used for min-max scaling in some dataloaders (e.g., the
      upstream Genie-Envisioner ones).
    - min/max are also kept for backward compatibility / debugging.
    """
    q99 = np.percentile(data, 99, axis=0)
    q01 = np.percentile(data, 1, axis=0)

    if _filter:
        data_mask = (data >= q01) & (data <= q99)
        data_mask = data_mask.min(axis=1)
        data = data[data_mask, :]

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    return means, stds, mins, maxs, q01, q99

def get_statistics(
    data_root, 
    data_name, 
    data_type, 
    save_path, 
    action_key="action", 
    state_key="observation.state", 
    nrnd=50000, 
    _filter=True,
    is_calvin_eef=False,
    num_workers=32,
    check_jumps=True,
    pos_threshold=50.0,
    ori_threshold=1.0,
    joint_threshold=1.0,
    gripper_threshold=50.0,
    fix_gripper_dim_action=None,
    fix_gripper_dim_state=None,
    fix_gripper_min=0.0,
    fix_gripper_max=850.0,
    gripper_dim_index=None,
    force_dims=None,
    force_threshold=10.0,
    unwrap_eef_angles=False,
    extract_7d_state=False,
):
    """
    Compute dataset statistics for actions, delta actions, and states.

    Gripper handling is opt-in. By default (no ``--gripper_dim_index`` /
    ``--fix_gripper_dim_*``) every action dim is treated as a joint and the
    output stats contain no gripper-specific override. This is the correct
    mode for tactile / no-gripper datasets such as the DexTacWAM corpus. For
    CALVIN / agibot pass ``--gripper_dim_index 6`` (and optionally the
    ``--fix_gripper_*`` knobs) to restore the legacy behavior.

    Args:
        data_root: Root directory containing parquet files
        data_name: Dataset name (e.g., "ABC_lerobot" or "D_lerobot")
        data_type: Action space type ("joint" or "eef")
        save_path: Path to save JSON statistics
        action_key: Key for action data in parquet
        state_key: Key for state data in parquet
        nrnd: Maximum number of episodes to sample
        _filter: Whether to filter outliers (1st-99th percentile)
        is_calvin_eef: If True, apply unwrap to EEF angles AND extract 7D state (legacy flag)
        unwrap_eef_angles: If True, unwrap Euler angles (dims 3,4,5) for EEF data (no state extraction)
        extract_7d_state: If True, extract 7D state from 15D CALVIN state
        num_workers: Number of parallel workers for processing files
        check_jumps: If True, check for large jumps in action trajectory
        pos_threshold: Position jump threshold in mm (default: 50mm)
        ori_threshold: Orientation jump threshold in radians (default: 1.0 rad ≈ 57°)
        joint_threshold: Joint jump threshold in radians (default: 1.0 rad)
        gripper_threshold: Gripper jump threshold (only used when
            ``gripper_dim_index`` is not None).
        gripper_dim_index: Action index of the gripper dim, or None (default)
            to disable gripper detection entirely.
    """
    
    assert data_type in ["joint", "eef"], f"data_type must be 'joint' or 'eef', got {data_type}"
    
    # Get all parquet files
    if not os.path.isdir(data_root):
        raise ValueError(f"data_root must be a directory: {data_root}")

    parquet_files = []

    # Case A: plain LeRobot layout: <data_root>/data/chunk-*/episode_*.parquet
    direct_data_dir = os.path.join(data_root, "data")
    if os.path.isdir(direct_data_dir):
        parquet_files.extend(glob.glob(os.path.join(direct_data_dir, "chunk-*", "*.parquet")))

    # Case B: domain/task subdir layout: <data_root>/<domain>/data/chunk-*/*.parquet
    if len(parquet_files) == 0:
        for task_dir in os.listdir(data_root):
            task_path = os.path.join(data_root, task_dir)
            if os.path.isdir(task_path):
                data_dir = os.path.join(task_path, "data")
                if os.path.exists(data_dir):
                    parquet_files.extend(glob.glob(os.path.join(data_dir, "chunk-*", "*.parquet")))
        
    # Case C: fallback: parquet directly under root
        if len(parquet_files) == 0:
            parquet_files = glob.glob(os.path.join(data_root, "*.parquet"))
    
    parquet_files.sort()
    print(f"Found {len(parquet_files)} parquet files")
    if len(parquet_files) == 0:
        raise RuntimeError(
            f"No parquet files found under '{data_root}'. Expected LeRobot layout like '{data_root}/data/chunk-*/episode_*.parquet'."
        )
    
    # Randomly sample if needed
    if nrnd > 0 and nrnd < len(parquet_files):
        parquet_files = list(np.random.choice(parquet_files, nrnd, replace=False))
        print(f"Randomly sampled {nrnd} files")
    
    data_list = []
    state_list = []
    delta_data_list = []
    all_warnings = []
    
    print(f"Processing episodes with {num_workers} workers...")
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_single_file,
        action_key=action_key,
        state_key=state_key,
        is_calvin_eef=is_calvin_eef,
        data_type=data_type,
        check_jumps=check_jumps,
        pos_threshold=pos_threshold,
        ori_threshold=ori_threshold,
        joint_threshold=joint_threshold,
        gripper_threshold=gripper_threshold,
        gripper_dim_index=gripper_dim_index,
        force_dims=force_dims,
        force_threshold=force_threshold,
        unwrap_eef_angles=unwrap_eef_angles,
        extract_7d_state=extract_7d_state,
    )
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_func, path): path for path in parquet_files}
        
        # Collect results with progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(parquet_files)):
            result = future.result()
            if result is not None:
                action, delta_action, state, warnings = result
                data_list.append(action)
                if delta_action is not None:
                    delta_data_list.append(delta_action)
                state_list.append(state)
                if warnings:
                    all_warnings.extend(warnings)
    
    # Concatenate all data
    print("\nConcatenating data...")
    data_list = np.concatenate(data_list, axis=0)
    assert len(data_list.shape) == 2, f"Expected 2D array, got shape {data_list.shape}"
    print(f"Action data shape: {data_list.shape}")
    
    delta_data_list = np.concatenate(delta_data_list, axis=0) if delta_data_list else np.zeros((0, data_list.shape[1]))
    assert len(delta_data_list.shape) == 2, f"Expected 2D array, got shape {delta_data_list.shape}"
    print(f"Delta action data shape: {delta_data_list.shape}")
    
    state_list = np.concatenate(state_list, axis=0)
    assert len(state_list.shape) == 2, f"Expected 2D array, got shape {state_list.shape}"
    print(f"State data shape: {state_list.shape}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    means, stds, mins, maxs, q01, q99 = cal_statistic(data_list, _filter=_filter)
    delta_means, delta_stds, delta_mins, delta_maxs, delta_q01, delta_q99 = cal_statistic(
        delta_data_list, _filter=_filter
    )
    state_means, state_stds, state_mins, state_maxs, state_q01, state_q99 = cal_statistic(
        state_list, _filter=_filter
    )
    
    gmin, gmax = fix_gripper_min, fix_gripper_max
    gmean = (gmin + gmax) / 2.0
    gstd = (gmax - gmin) / 4.0

    # Override action gripper dimension with fixed range
    if fix_gripper_dim_action is not None:
        d = fix_gripper_dim_action
        print(f"\nFixing action gripper dim {d}: min={gmin}, max={gmax}")

        if d < len(means):
            means[d], stds[d] = gmean, gstd
            mins[d], maxs[d] = gmin, gmax
            q01[d], q99[d] = gmin, gmax
            print(f"   Action dim {d} overridden")

        if d < len(delta_means):
            delta_means[d], delta_stds[d] = gmean, gstd
            delta_mins[d], delta_maxs[d] = gmin, gmax
            delta_q01[d], delta_q99[d] = gmin, gmax
            print(f"   Delta action dim {d} overridden")

    # Override state gripper dimension with fixed range
    if fix_gripper_dim_state is not None:
        d = fix_gripper_dim_state
        print(f"\nFixing state gripper dim {d}: min={gmin}, max={gmax}")

        if d < len(state_means):
            state_means[d], state_stds[d] = gmean, gstd
            state_mins[d], state_maxs[d] = gmin, gmax
            state_q01[d], state_q99[d] = gmin, gmax
            print(f"   State dim {d} overridden")

    # Build statistics dictionary
    statistics_info = {
        f"{data_name}_{data_type}": {
            "mean": means.tolist(),
            "std": stds.tolist(),
            "min": mins.tolist(),
            "max": maxs.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "normalize": "mean_std"
        },
        f"{data_name}_delta_{data_type}": {
            "mean": delta_means.tolist(),
            "std": delta_stds.tolist(),
            "min": delta_mins.tolist(),
            "max": delta_maxs.tolist(),
            "q01": delta_q01.tolist(),
            "q99": delta_q99.tolist(),
            "normalize": "mean_std"
        },
        f"{data_name}_state_{data_type}": {
            "mean": state_means.tolist(),
            "std": state_stds.tolist(),
            "min": state_mins.tolist(),
            "max": state_maxs.tolist(),
            "q01": state_q01.tolist(),
            "q99": state_q99.tolist(),
            "normalize": "mean_std"
        },
    }
    
    # Save to JSON
    print(f"\nSaving statistics to {save_path}...")
    with open(save_path, "w") as f:
        json.dump(statistics_info, f, indent=4)
    
    # Report warnings if any
    if all_warnings:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: Found {len(all_warnings)} large action jumps!")
        print(f"{'='*60}")
        
        # Group warnings by type
        pos_warnings = [w for w in all_warnings if w['type'] == 'position']
        ori_warnings = [w for w in all_warnings if w['type'].startswith('orientation')]
        gripper_warnings = [w for w in all_warnings if w['type'] == 'gripper']
        force_warnings = [w for w in all_warnings if w['type'].startswith('force_')]
        joint_warnings = [w for w in all_warnings if w['type'].startswith('joint_')]
        
        if pos_warnings:
            print(f"\n📍 Position jumps (> {pos_threshold}m): {len(pos_warnings)} occurrences")
            print(f"   Top 5 largest jumps:")
            sorted_pos = sorted(pos_warnings, key=lambda x: x['jump_magnitude'], reverse=True)[:5]
            for w in sorted_pos:
                print(f"   - {w['episode']}")
                print(f"     Timestep {w['timestep']}: {w['jump_magnitude']:.4f}m (delta: {w['delta']})")
        
        if ori_warnings:
            print(f"\n🔄 Orientation jumps (> {ori_threshold} rad): {len(ori_warnings)} occurrences")
            print(f"   Top 5 largest jumps:")
            sorted_ori = sorted(ori_warnings, key=lambda x: x['jump_magnitude'], reverse=True)[:5]
            for w in sorted_ori:
                print(f"   - {w['episode']}")
                print(f"     Timestep {w['timestep']}: {w['type']} = {w['jump_magnitude']:.4f} rad ({np.degrees(w['jump_magnitude']):.1f}°)")
        
        if gripper_warnings:
            print(f"\n🤏 Gripper jumps (> {gripper_threshold}): {len(gripper_warnings)} occurrences")
            print(f"   Top 5 largest jumps:")
            sorted_gripper = sorted(gripper_warnings, key=lambda x: x['jump_magnitude'], reverse=True)[:5]
            for w in sorted_gripper:
                print(f"   - {w['episode']}")
                print(f"     Timestep {w['timestep']}: {w['jump_magnitude']:.4f} (delta: {w['delta']})")

        if force_warnings:
            print(f"\n💪 Force jumps (> {force_threshold}): {len(force_warnings)} occurrences")
            print(f"   Top 5 largest jumps:")
            sorted_force = sorted(force_warnings, key=lambda x: x['jump_magnitude'], reverse=True)[:5]
            for w in sorted_force:
                print(f"   - {w['episode']}")
                print(f"     Timestep {w['timestep']}: {w['type']} = {w['jump_magnitude']:.4f} (delta: {w['delta']})")

        if joint_warnings:
            print(f"\n🔧 Joint jumps (> {joint_threshold} rad): {len(joint_warnings)} occurrences")
            print(f"   Top 5 largest jumps:")
            sorted_joint = sorted(joint_warnings, key=lambda x: x['jump_magnitude'], reverse=True)[:5]
            for w in sorted_joint:
                print(f"   - {w['episode']}")
                print(f"     Timestep {w['timestep']}: {w['type']} = {w['jump_magnitude']:.4f} rad")

        # Save warnings to a separate file
        warnings_path = save_path.replace('.json', '_warnings.json')
        with open(warnings_path, 'w') as f:
            json.dump(all_warnings, f, indent=2)
        print(f"\n💾 Full warning details saved to: {warnings_path}")
    else:
        print(f"\n✅ No large action jumps detected (pos_threshold={pos_threshold}m, ori_threshold={ori_threshold} rad)")
    
    print("Done!")
    print(f"\nStatistics keys generated:")
    for key in statistics_info.keys():
        print(f"  - {key}")
    
    return statistics_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset statistics with proper angle unwrapping (parallelized)")
    parser.add_argument('--data_root', required=True, help="Root directory of the dataset")
    parser.add_argument('--data_name', required=True, help="Dataset name (e.g., ABC_lerobot, D_lerobot)")
    parser.add_argument('--data_type', required=True, choices=["joint", "eef"], help="Action space type")
    parser.add_argument('--action_key', default="action", help="Key for action data in parquet")
    parser.add_argument('--state_key', default="observation.state", help="Key for state data in parquet")
    parser.add_argument('--save_path', required=True, help="Path to save JSON statistics")
    parser.add_argument('--nrnd', type=int, default=0, help="Number of episodes to randomly sample (0 = use all)")
    parser.add_argument('--filter', action='store_true', default=True, help="Filter outliers (1-99 percentile)")
    parser.add_argument('--is_calvin_eef', action='store_true', help="Legacy: apply unwrap angles AND extract 7D state")
    parser.add_argument('--unwrap_eef_angles', action='store_true', help="Unwrap Euler angles (dims 3,4,5) for EEF data without extracting state")
    parser.add_argument('--extract_7d_state', action='store_true', help="Extract 7D state from 15D CALVIN state")
    parser.add_argument('--num_workers', type=int, default=32, help="Number of parallel workers (default: 32)")
    parser.add_argument('--check_jumps', action='store_true', default=True, help="Check for large jumps in action trajectory")
    parser.add_argument('--pos_threshold', type=float, default=50.0, help="Position jump threshold in mm (default: 50)")
    parser.add_argument('--ori_threshold', type=float, default=1.0, help="Orientation jump threshold in radians (default: 1.0)")
    parser.add_argument('--joint_threshold', type=float, default=1.0, help="Joint jump threshold in radians (default: 1.0)")
    parser.add_argument('--gripper_threshold', type=float, default=50.0, help="Gripper jump threshold (only used when --gripper_dim_index is set)")
    parser.add_argument('--gripper_dim_index', type=int, default=None, help="Index of the gripper dimension for jump detection. Opt-in: when omitted, no dim is treated as gripper (correct for tactile / no-gripper datasets). Pass e.g. 6 for CALVIN/agibot.")
    parser.add_argument('--fix_gripper_dim_action', type=int, default=None, help="Override this ACTION dimension with fixed gripper range (opt-in; leave unset for no-gripper datasets)")
    parser.add_argument('--fix_gripper_dim_state', type=int, default=None, help="Override this STATE dimension with fixed gripper range (opt-in; leave unset for no-gripper datasets)")
    parser.add_argument('--fix_gripper_min', type=float, default=0.0, help="Fixed gripper min, shared by action and state (only used with --fix_gripper_dim_*)")
    parser.add_argument('--fix_gripper_max', type=float, default=850.0, help="Fixed gripper max, shared by action and state (only used with --fix_gripper_dim_*)")
    parser.add_argument('--force_dims', type=int, nargs='+', default=None, help="Action dimensions that are force (e.g., --force_dims 7 8 9)")
    parser.add_argument('--force_threshold', type=float, default=10.0, help="Force jump threshold (default: 10.0)")
    
    args = parser.parse_args()
    
    get_statistics(
        data_root=args.data_root,
        data_name=args.data_name,
        data_type=args.data_type,
        save_path=args.save_path,
        action_key=args.action_key,
        state_key=args.state_key,
        nrnd=args.nrnd,
        _filter=args.filter,
        is_calvin_eef=args.is_calvin_eef,
        num_workers=args.num_workers,
        check_jumps=args.check_jumps,
        pos_threshold=args.pos_threshold,
        ori_threshold=args.ori_threshold,
        joint_threshold=args.joint_threshold,
        gripper_threshold=args.gripper_threshold,
        fix_gripper_dim_action=args.fix_gripper_dim_action,
        fix_gripper_dim_state=args.fix_gripper_dim_state,
        fix_gripper_min=args.fix_gripper_min,
        fix_gripper_max=args.fix_gripper_max,
        gripper_dim_index=args.gripper_dim_index,
        force_dims=args.force_dims,
        force_threshold=args.force_threshold,
        unwrap_eef_angles=args.unwrap_eef_angles,
        extract_7d_state=args.extract_7d_state,
    )
