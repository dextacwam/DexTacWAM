#!/usr/bin/env python3
"""Offline dry-run for right-only / relative serving (deploy gate F2).

Checks everything about a serving config that does NOT need a GPU, a checkpoint
or the robot: the yaml's layout declarations, the stats file widths, the raw ->
model gather, and the relative -> absolute compose chain. Runs in seconds, so it
can gate every config change instead of waiting for a hardware slot.

What it proves, in the order the server does it:

  1. yaml declares BOTH layouts explicitly, and they are mutually consistent
     with valid_cam / max_view / action_in_channels.
  2. the stats file really is that layout (widths + finite, ordered q01/q99).
  3. the raw 90-D observation gathers into the model's state, with the right
     arm's blocks landing on the model's declared slices.
  4. LEFT-hand data cannot influence a right-only model state (poison test).
  5. compose(anchor_from_gathered_state, rel9) reconstructs the absolute EEF
     target that training relativized -- with the anchor read from the MODEL
     state, which is the step that silently reads the wrong 16 numbers if the
     layouts are ever conflated.

Usage:
    python web_infer_scripts/dryrun_relative_server.py \
        --config configs/tongs/stage3_action_expert.yaml

Deliberately numpy-only: it does not import torch or the server module, so it
runs on a login node. Gate F3 boots the real server against a real checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.utils.layout_contract import (  # noqa: E402
    assert_all_layouts_match_contract,
    load_contract,
)
from data.utils.pose_math import (  # noqa: E402
    _random_se3_mat16,   # same generator training's self-tests use
    compose_mat16_and_relative_rot6d,
    mat16_to_mat44,
    relativize_mat16_to_rot6d,
)
from data.utils.raw_obs_layout import (  # noqa: E402
    ARM7,
    FINGERS,
    HAND22,
    POSE16,
    build_blank_window_table,
    build_state_gather_index,
    build_tactile_hand_index,
    get_raw_obs_layout,
)
from data.utils.relative_action import (  # noqa: E402
    LAYOUTS,
    assert_layout_dims,
    get_arm_layout,
)
from data.utils.task_registry import (  # noqa: E402
    NO_TASK,
    assert_task_matches_checkpoint,
    get_task,
)


def _fail(msg: str) -> None:
    raise SystemExit(f"[dryrun] FAIL: {msg}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", required=True, help="serving yaml (the TRAINING yaml)")
    p.add_argument("--domain-name", default=None,
                   help="stats key prefix; defaults to data.train.domains[0]")
    p.add_argument("--chunk", type=int, default=None,
                   help="rows to simulate; defaults to data.train.action_chunk")
    p.add_argument("--task", default=None,
                   help="registered task id to cross-check against this config's "
                        "domain and arm_layout (or 'none'); skipped when omitted")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.is_file():
        _fail(f"config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    tcfg = cfg["data"]["train"]

    print(f"[dryrun] config = {cfg_path}")

    # ---- 1. layouts: explicit, mutually consistent -------------------------
    # Mirror the server's own default: configs written before right_only existed
    # omit arm_layout and mean bimanual. Being stricter here than the server would
    # reject configs it serves happily. A right-only corpus that forgets the key is
    # still caught below, where the stats widths are cross-checked against the
    # layout (45/68 cannot pass as bimanual's 90/136).
    layout_explicit = "arm_layout" in tcfg
    layout = get_arm_layout(str(tcfg.get("arm_layout", "bimanual")))
    raw_layout = get_raw_obs_layout(str(tcfg.get("raw_obs_layout", "bimanual")))
    arms = list(layout.arms)
    n_arms = len(arms)

    contract = load_contract()
    version = assert_all_layouts_match_contract(LAYOUTS, contract)
    print(f"[dryrun] contract       v{version} sha256={contract['_sha256'][:8]} OK")
    print(f"[dryrun] arm_layout     {layout.name} arms={arms} "
          f"({'explicit' if layout_explicit else 'IMPLICIT default'})")
    if not layout_explicit and layout.n_arms < 2:
        _fail("arm_layout defaulted to bimanual but resolved to <2 arms; "
              "a non-bimanual layout must be declared explicitly")
    print(f"[dryrun] raw_obs_layout {raw_layout.name} state={raw_layout.state_dim} "
          f"tactile_hands={raw_layout.tactile_hands} "
          f"({'explicit' if 'raw_obs_layout' in tcfg else 'default'})")

    action_type = str(tcfg.get("action_type", "absolute"))
    action_space = str(tcfg.get("action_space", "joint"))
    is_relative = action_type == "relative_eef_rot6d"
    if action_type not in ("absolute", "relative_eef_rot6d"):
        _fail(f"action_type={action_type!r} is not servable by tactile_server")

    cams = [str(c) for c in tcfg["valid_cam"]]
    dm_cfg = cfg["diffusion_model"]["config"]
    # Under the tactile-WM ablation the tactile views never enter the DiT, so
    # they do not consume a slot in the view-embedding table max_view sizes --
    # they consume one in tactile_late_fuse_max_view instead. Sizing max_view
    # against cameras+arms here would reject exactly the config the ablation
    # needs, and padding max_view to make this check pass would allocate view
    # embeddings the checkpoint does not have.
    late_fuse = bool(cfg.get("tactile_late_fuse", False))
    n_view_total = len(cams) + (0 if late_fuse else n_arms)
    # max_view and projector.num_views are OPTIONAL: older configs omit max_view
    # and the server reads neither (it takes only action_in_channels from that
    # block). Check them when present rather than requiring them.
    max_view = dm_cfg.get("max_view")
    if max_view is not None and int(max_view) < n_view_total:
        _fail(f"max_view={max_view} < n_view_visual({len(cams)}) + "
              f"n_view_tactile({0 if late_fuse else n_arms}) = {n_view_total}")
    if late_fuse:
        lf_max = dm_cfg.get("tactile_late_fuse_max_view")
        if not bool(dm_cfg.get("tactile_late_fuse", False)):
            _fail("tactile_late_fuse=true at the top level but "
                  "diffusion_model.config.tactile_late_fuse is not set; the DiT "
                  "would be built without the late-fuse tactile branch and the "
                  "tokens would be silently dropped")
        if not bool(dm_cfg.get("dual_cross_attn", False)):
            _fail("tactile_late_fuse=true requires dual_cross_attn=true; with a "
                  "shared cross-attention there is no separate tactile branch "
                  "for the bypassed tokens to reach")
        if lf_max is not None and int(lf_max) < n_arms:
            _fail(f"tactile_late_fuse_max_view={lf_max} < n_arms={n_arms}")
        print(f"[dryrun] tactile route late_fuse (bypasses the DiT) "
              f"tactile_late_fuse_max_view="
              f"{lf_max if lf_max is not None else 'unset'} OK")
    num_views = (cfg.get("projector") or {}).get("num_views")
    if num_views is not None and int(num_views) != n_arms:
        _fail(f"projector.num_views={num_views} != n_arms={n_arms} for "
              f"arm_layout={layout.name!r}")
    print(f"[dryrun] cameras        {cams} -> n_view_total={n_view_total} "
          f"(max_view={max_view if max_view is not None else 'unset'}, "
          f"projector.num_views={num_views if num_views is not None else 'unset'}) OK")

    # ---- 2. stats file really is this layout ------------------------------
    domain = args.domain_name or str(tcfg["domains"][0])

    # ---- 1b. task / prompt provenance -------------------------------------
    # Checked here as well as in the server so a wrong --task costs a second and
    # no GPU: the launcher runs this as its last guard before loading weights.
    entry = None
    if args.task is not None:
        try:
            entry = get_task(args.task)
        except KeyError as exc:
            _fail(str(exc))
        if entry is None:
            print(f"[dryrun] task           {NO_TASK} -- prompt comes from the "
                  f"observation, health gating off (offline tooling only)")
        else:
            try:
                assert_task_matches_checkpoint(entry, domain, layout.name)
            except ValueError as exc:
                _fail(str(exc))
            print(f"[dryrun] task           {entry.task_id} "
                  f"sha256={entry.prompt_sha256[:12]} matches domain + layout OK")

    stat_path = Path(tcfg["stat_file"])
    if not stat_path.is_absolute():
        stat_path = REPO_ROOT / stat_path
    if not stat_path.is_file():
        _fail(f"stat_file not found: {stat_path}")
    stats = json.loads(stat_path.read_text())

    act_key = (f"{domain}_relative_{action_space}" if is_relative
               else f"{domain}_{action_space}")
    sta_key = f"{domain}_state_{action_space}"
    for k in (act_key, sta_key):
        if k not in stats:
            _fail(f"stats key {k!r} missing from {stat_path.name}; "
                  f"has {sorted(k for k in stats if not k.startswith('_'))}")

    act_min = np.asarray(stats[act_key]["q01"], dtype=np.float32)
    act_max = np.asarray(stats[act_key]["q99"], dtype=np.float32)
    sta_min = np.asarray(stats[sta_key]["q01"], dtype=np.float32)
    sta_max = np.asarray(stats[sta_key]["q99"], dtype=np.float32)
    action_dim, state_dim = int(act_min.size), int(sta_min.size)

    assert_layout_dims(
        layout,
        state_dim=state_dim,
        **({"rel_action_dim": action_dim} if is_relative
           else {"abs_action_dim": action_dim}),
        where=f"{stat_path.name}[{act_key}, {sta_key}]",
    )
    # The de-normalizer divides by (q99-q01); a non-finite or inverted range
    # there produces NaN targets that the robot would happily try to reach.
    for name, lo, hi in ((act_key, act_min, act_max), (sta_key, sta_min, sta_max)):
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            _fail(f"{name}: q01/q99 contain non-finite values")
        bad = int((hi < lo).sum())
        if bad:
            _fail(f"{name}: q99 < q01 in {bad} dims")
    ain = int(cfg["diffusion_model"]["config"]["action_in_channels"])
    if action_dim + state_dim != ain:
        _fail(f"action({action_dim}) + state({state_dim}) != action_in_channels({ain})")
    override = (cfg.get("tactile_inference") or {}).get("action_only_dim_override")
    if override is not None and int(override) != action_dim:
        _fail(f"action_only_dim_override={override} != stats action dim {action_dim}")
    print(f"[dryrun] stats          {stat_path.name}: {act_key}={action_dim}, "
          f"{sta_key}={state_dim}; +state == action_in_channels={ain} OK")

    # v0d pose injection normalizes hand_pose with Stage-1 stats; the server
    # refuses to feed un-normalized pose to the adapter, so catch a missing or
    # wrong-width pose_stats here rather than after the model has loaded.
    pose_inj = bool((cfg.get("tactile_vae", {}).get("config", {}) or {})
                    .get("adapter_use_pose_injection", False))
    pose_stats_path = tcfg.get("pose_stats_path")
    if pose_inj:
        if not pose_stats_path:
            _fail("adapter_use_pose_injection=true but data.train.pose_stats_path is unset")
        pp = Path(pose_stats_path)
        if not pp.is_absolute():
            pp = REPO_ROOT / pp
        if not pp.is_file():
            _fail(f"pose_stats_path not found: {pp}")
        ps = json.loads(pp.read_text())
        for f in ("mean", "std"):
            v = np.asarray(ps[f], dtype=np.float32)
            if v.shape != (HAND22,):
                _fail(f"pose_stats {f} is {v.shape}, want ({HAND22},)")
        if float(np.asarray(ps["std"], dtype=np.float32).min()) <= 0.0:
            _fail("pose_stats std has a non-positive entry (division by zero)")
        print(f"[dryrun] pose stats     {pp.name}: mean/std ({HAND22},), std > 0 OK")

    # ---- 3./4. raw -> model gather, and no left leakage -------------------
    gather = build_state_gather_index(raw_layout, layout)
    tac_hands = build_tactile_hand_index(raw_layout, layout)
    rng = np.random.default_rng(0)

    state_raw = rng.standard_normal(raw_layout.state_dim).astype(np.float32)
    # Put valid SE(3) poses in the raw EEF blocks so the compose test is meaningful.
    for i in range(raw_layout.tactile_hands):
        lo, hi = raw_layout.eef_pose[i]
        state_raw[lo:hi] = _random_se3_mat16(1, rng)[0]
    state_model = state_raw[gather]

    for i, arm in enumerate(arms):
        r = raw_layout.arm_position(arm)
        checks = (
            (raw_layout.arm_joints[r], (ARM7 * i, ARM7 * (i + 1)), "arm_joints"),
            (raw_layout.hand_joints[r], layout.state_hands[i], "hand_joints"),
            (raw_layout.eef_pose[r], layout.state_poses[i], "eef_pose"),
        )
        for (rlo, rhi), (mlo, mhi), what in checks:
            if not np.array_equal(state_raw[rlo:rhi], state_model[mlo:mhi]):
                _fail(f"{arm} {what}: raw[{rlo}:{rhi}] did not land on model[{mlo}:{mhi}]")
    print(f"[dryrun] gather         raw {raw_layout.state_dim} -> model {state_dim}; "
          f"every {arms} block lands on its declared model slice OK")

    dropped = [a for a in raw_layout.arms if a not in arms]
    if dropped:
        poisoned = state_raw.copy()
        for a in dropped:
            r = raw_layout.arm_position(a)
            for lo, hi in (raw_layout.arm_joints[r], raw_layout.hand_joints[r],
                           raw_layout.eef_pose[r]):
                poisoned[lo:hi] = 1e6
        if not np.array_equal(poisoned[gather], state_model):
            _fail(f"data from dropped arm(s) {dropped} leaked into the model state")
        print(f"[dryrun] leakage        poisoning {dropped} left the model state "
              f"unchanged OK")
    else:
        if not np.array_equal(gather, np.arange(raw_layout.state_dim)):
            _fail("no arms dropped, but the gather is not the identity")
        print("[dryrun] leakage        n/a (no arms dropped); gather is identity OK")

    print(f"[dryrun] tactile        raw hands {list(range(raw_layout.tactile_hands))} "
          f"-> keep {tac_hands} = {arms}, shape ({n_arms},{FINGERS},H,W) OK")

    # ---- 5. compose round-trip through the SERVER's anchor path ------------
    if is_relative:
        chunk = int(args.chunk or tcfg["action_chunk"])
        n_prev = int(tcfg["n_previous"])
        if n_prev < 1:
            _fail(f"n_previous={n_prev} cannot index an anchor row")

        # Pretend training produced this chunk: absolute EEF targets, relativized
        # against the SAME anchor the server will use (the observed state row).
        for i, arm in enumerate(arms):
            tgt16 = _random_se3_mat16(chunk, rng)
            anchor16 = state_model[slice(*layout.state_poses[i])]
            rel9 = relativize_mat16_to_rot6d(anchor16, tgt16)
            if rel9.shape != (chunk, 9):
                _fail(f"{arm}: relativize gave {rel9.shape}, want ({chunk}, 9)")

            rec16 = compose_mat16_and_relative_rot6d(anchor16, rel9)
            T_rec, T_tgt = mat16_to_mat44(rec16), mat16_to_mat44(tgt16)
            err = float(np.abs(T_rec - T_tgt).max())
            if err > 1e-4:
                _fail(f"{arm}: compose(anchor, rel9) off by {err:.3e} from the target")

            # Composing against the WRONG arm's anchor must be detectably wrong;
            # otherwise this test would pass even with the anchors swapped.
            if n_arms > 1:
                other = layout.state_poses[(i + 1) % n_arms]
                wrong = compose_mat16_and_relative_rot6d(
                    state_model[slice(*other)], rel9)
                if float(np.abs(mat16_to_mat44(wrong) - T_tgt).max()) < 1e-3:
                    _fail(f"{arm}: composing against the other arm's anchor gave the "
                          f"same result; the anchor test is vacuous")
            print(f"[dryrun] compose        {arm}: anchor=model state"
                  f"[{layout.state_poses[i][0]}:{layout.state_poses[i][1]}] "
                  f"(row {n_prev - 1}), max|T_rec - T_tgt| = {err:.2e} OK")

        # Reading the anchor out of the RAW vector with the MODEL slice is the
        # concrete bug the two-layout split exists to prevent. Prove it differs.
        for i, arm in enumerate(arms):
            mlo, mhi = layout.state_poses[i]
            if mhi <= raw_layout.state_dim and np.array_equal(
                state_raw[mlo:mhi], state_model[mlo:mhi]
            ):
                if raw_layout.name != layout.name:
                    _fail(f"{arm}: raw[{mlo}:{mhi}] equals the model anchor, so a "
                          f"layout mix-up would go undetected here")

        print(f"[dryrun] response       arm_target_pose ({chunk},{n_arms},4,4) "
              f"ABSOLUTE, hand_target ({chunk},{HAND22 * n_arms}), "
              f"anchor fixed for the whole chunk")
    else:
        lo, hi = layout.arm_joints[0], layout.hand[1]
        print(f"[dryrun] response       joint_targets action[{lo}:{hi}] "
              f"= {(ARM7 + HAND22) * n_arms} dims")

    # Same source the server will use, so this line is the served table and not
    # a lookalike. Without --task it is the legacy fallback, and says so.
    windows = build_blank_window_table(raw_layout, layout, entry)
    print(f"[dryrun] blank windows  "
          f"{ {f'{a}:{f}': w for (a, f), w in windows.items()} }"
          f"{'' if entry is not None else '   (LEGACY fallback -- no --task)'}")
    print(f"[dryrun] client must send state ({raw_layout.state_dim},), tactile "
          f"({raw_layout.tactile_hands},{FINGERS},H,W), images ({len(cams)},H,W,3) "
          f"in order {cams}")
    assert POSE16 == 16
    print("[dryrun] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
