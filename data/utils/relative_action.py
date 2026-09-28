"""Shared builder for the DexVTAM ``relative_eef_rot6d`` action mode.

Single source of truth used by BOTH ``dex_vtam_dataset.get_batch`` /
``_get_batch_from_cache`` AND the offline stats pass
(``scripts/get_statistics.py``), so the relative action they build can never
drift.

The ONLY block that becomes relative is ``arm_target_pose`` (16-D row-major 4x4
per arm), replaced by ``rel9 = [xyz(3), rot6d(6)]`` per arm. Everything else
(force, arm_target_joints, hand_target) stays absolute and byte-identical.

Two layouts exist, selected EXPLICITLY by name via ``get_arm_layout`` -- never
inferred from tensor width, because a width only says which layout the data
*looks* like and would mask a config/stats/cache mismatch. Callers additionally
cross-check the observed dims with ``assert_layout_dims``.

``bimanual`` (erase / handover / chip / unscrew) -- 150 -> 136:

    absolute action (150):
        [  0: 60] force            (L+R tactile_f6)
        [ 60: 74] arm_target_joints (L7 + R7)
        [ 74:118] hand_target       (L22 + R22)
        [118:134] arm_target_pose L (row-major 4x4)
        [134:150] arm_target_pose R (row-major 4x4)

    absolute state (90, arms-first):
        [  0:  7] L_arm7   [  7: 14] R_arm7
        [ 14: 36] L_hand22 [ 36: 58] R_hand22
        [ 58: 74] L_pose16 (row-major 4x4)   <- L anchor
        [ 74: 90] R_pose16 (row-major 4x4)   <- R anchor

    relative action (136):
        [  0:118] force + arm_target_joints + hand_target (unchanged)
        [118:127] rel9 L = [xyz(3), rot6d(6)]
        [127:136] rel9 R = [xyz(3), rot6d(6)]

``right_only`` (tong / bowl right-hand-only datasets) -- 75 -> 68:

    absolute action (75):
        [  0: 30] force            (R tactile_f6)
        [ 30: 37] arm_target_joints (R7)
        [ 37: 59] hand_target       (R22)
        [ 59: 75] arm_target_pose R (row-major 4x4)

    absolute state (45):
        [  0:  7] R_arm7   [  7: 29] R_hand22
        [ 29: 45] R_pose16 (row-major 4x4)   <- R anchor

    relative action (68):
        [  0: 59] force + arm_target_joints + hand_target (unchanged)
        [ 59: 68] rel9 R = [xyz(3), rot6d(6)]

Bimanual layout is VERIFIED (2026-07) from the erase/handover stats blocks and
``vtam_data_scripts/convert_to_vtam.py`` (row-major mat16); right-only follows
from the same ``vtam_data_scripts/layout.py`` field order with the left blocks
dropped (state 45, action 75).

The single anchor is the LAST OBSERVED state row (window row ``n_previous-1``),
matching VTAM's ``base6 = sel_state[n_previous-1]``. All predicted rows in the
window are relativized against this one fixed anchor (no per-timestep / future
state, no recursive composition).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.utils.pose_math import relativize_mat16_to_rot6d


POSE16, REL9, HAND22, ARM7 = 16, 9, 22, 7


@dataclass(frozen=True)
class RelativeArmLayout:
    """Index layout of one action/state corpus.

    Per-arm pose slices are ordered tuples (one entry per entry of ``arms``, in the
    same order the blocks appear in the flat vector), so a 1-arm corpus is expressed
    by the same type as a 2-arm one instead of needing L/R fields to both exist.
    """
    name: str
    arms: tuple[str, ...]
    # absolute action
    abs_action_dim: int
    force: tuple[int, int]
    arm_joints: tuple[int, int]
    hand: tuple[int, int]
    action_poses: tuple[tuple[int, int], ...]
    # absolute state
    state_dim: int
    state_hands: tuple[tuple[int, int], ...]   # 22-D hand_joint block per arm
    state_poses: tuple[tuple[int, int], ...]
    # relative action
    rel_action_dim: int
    rel_poses: tuple[tuple[int, int], ...]

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def validate(self) -> None:
        n = len(self.arms)
        assert n >= 1, f"{self.name}: needs at least one arm"
        assert len(self.action_poses) == len(self.state_poses) == len(self.rel_poses) \
            == len(self.state_hands) == n, (
            f"{self.name}: arms={self.arms} but {len(self.action_poses)}/"
            f"{len(self.state_poses)}/{len(self.rel_poses)}/{len(self.state_hands)} slices"
        )
        for slc in self.action_poses + self.state_poses:
            assert slc[1] - slc[0] == POSE16, f"{self.name}: {slc} is not a 16-D mat16"
        for slc in self.rel_poses:
            assert slc[1] - slc[0] == REL9, f"{self.name}: {slc} is not a 9-D rel9"
        # absolute action: force | arm_joints | hand | pose per arm, contiguous from 0
        assert self.force[0] == 0, f"{self.name}: action must start with force"
        assert self.arm_joints[0] == self.force[1], f"{self.name}: gap before arm_joints"
        assert self.hand[0] == self.arm_joints[1], f"{self.name}: gap before hand"
        prefix_end = self.hand[1]
        for i, slc in enumerate(self.action_poses):
            want = prefix_end if i == 0 else self.action_poses[i - 1][1]
            assert slc[0] == want, f"{self.name}: action pose {i} starts at {slc[0]}, want {want}"
        assert self.action_poses[-1][1] == self.abs_action_dim, \
            f"{self.name}: action poses end at {self.action_poses[-1][1]} != {self.abs_action_dim}"
        # absolute state: arm_joints per arm | hand_joint per arm | pose per arm
        for slc in self.state_hands:
            assert slc[1] - slc[0] == HAND22, f"{self.name}: {slc} is not a 22-D hand_joint"
        for i, slc in enumerate(self.state_hands[1:], start=1):
            assert slc[0] == self.state_hands[i - 1][1], f"{self.name}: gap in state hands"
        assert self.state_hands[0][0] == ARM7 * n, \
            f"{self.name}: state hands start at {self.state_hands[0][0]}, want {ARM7 * n}"
        assert self.state_hands[-1][1] == self.state_poses[0][0], \
            f"{self.name}: state hands end at {self.state_hands[-1][1]}, " \
            f"but poses start at {self.state_poses[0][0]}"
        for i, slc in enumerate(self.state_poses[1:], start=1):
            assert slc[0] == self.state_poses[i - 1][1], f"{self.name}: gap in state poses"
        assert self.state_poses[-1][1] == self.state_dim, \
            f"{self.name}: state poses end at {self.state_poses[-1][1]} != {self.state_dim}"
        # relative = the SAME absolute prefix, then rel9 per arm
        for i, slc in enumerate(self.rel_poses):
            want = prefix_end if i == 0 else self.rel_poses[i - 1][1]
            assert slc[0] == want, f"{self.name}: rel pose {i} starts at {slc[0]}, want {want}"
        assert self.rel_poses[-1][1] == self.rel_action_dim, \
            f"{self.name}: rel poses end at {self.rel_poses[-1][1]} != {self.rel_action_dim}"
        assert self.rel_action_dim == prefix_end + REL9 * n, \
            f"{self.name}: rel_action_dim {self.rel_action_dim} != {prefix_end} + {REL9}*{n}"

    @property
    def prefix_end(self) -> int:
        """End of the block copied verbatim into the relative action (== hand[1])."""
        return self.hand[1]


BIMANUAL_LAYOUT = RelativeArmLayout(
    name="bimanual",
    arms=("left", "right"),
    abs_action_dim=150,
    force=(0, 60),
    arm_joints=(60, 74),
    hand=(74, 118),
    action_poses=((118, 134), (134, 150)),
    state_dim=90,
    state_hands=((14, 36), (36, 58)),
    state_poses=((58, 74), (74, 90)),
    rel_action_dim=136,
    rel_poses=((118, 127), (127, 136)),
)

# Right-hand-only corpora (tong / bowl): the left blocks are absent, not zeroed, so
# every offset shifts. Derived from vtam_data_scripts/layout.py field order.
RIGHT_ONLY_LAYOUT = RelativeArmLayout(
    name="right_only",
    arms=("right",),
    abs_action_dim=75,
    force=(0, 30),
    arm_joints=(30, 37),
    hand=(37, 59),
    action_poses=((59, 75),),
    state_dim=45,
    state_hands=((7, 29),),
    state_poses=((29, 45),),
    rel_action_dim=68,
    rel_poses=((59, 68),),
)

LAYOUTS: dict[str, RelativeArmLayout] = {
    BIMANUAL_LAYOUT.name: BIMANUAL_LAYOUT,
    RIGHT_ONLY_LAYOUT.name: RIGHT_ONLY_LAYOUT,
}

for _lay in LAYOUTS.values():
    _lay.validate()

# Kept so existing bimanual call sites (erase / handover / chip / unscrew) are untouched.
DEFAULT_LAYOUT = BIMANUAL_LAYOUT


def get_arm_layout(name: str) -> RelativeArmLayout:
    """Look up a layout by EXPLICIT name (yaml ``arm_layout``). No width inference,
    no fallback: an unknown name is a hard error rather than a silent bimanual default."""
    if not isinstance(name, str) or name not in LAYOUTS:
        raise ValueError(f"Unknown arm_layout={name!r}; expected one of {sorted(LAYOUTS)}")
    return LAYOUTS[name]


def assert_layout_dims(
    layout: RelativeArmLayout,
    *,
    abs_action_dim: int | None = None,
    state_dim: int | None = None,
    rel_action_dim: int | None = None,
    where: str = "",
) -> None:
    """Cross-check OBSERVED widths against the explicitly selected layout.

    The yaml names the intent; this proves the data (and the stats / cache built from
    it) actually match, so a wrong ``arm_layout`` fails immediately at load / startup
    instead of surviving into a reshape somewhere downstream.
    """
    tag = f"{where}: " if where else ""
    for got, want, what in ((abs_action_dim, layout.abs_action_dim, "abs_action_dim"),
                            (state_dim, layout.state_dim, "state_dim"),
                            (rel_action_dim, layout.rel_action_dim, "rel_action_dim")):
        if got is not None and int(got) != want:
            raise AssertionError(
                f"{tag}observed {what}={int(got)} does not match arm_layout="
                f"{layout.name!r} ({what}={want}). Wrong arm_layout, or stale "
                f"stats/cache from a different hand selection."
            )


def build_relative_arm_pose_action(
    action_window_raw: np.ndarray,
    anchor_state_row_raw: np.ndarray,
    layout: RelativeArmLayout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Relativize the arm_target_pose block against a SINGLE explicit anchor.

    action_window_raw:    (T, abs_action_dim) RAW (pre-normalization) absolute rows.
    anchor_state_row_raw: (state_dim,)        RAW absolute state row = anchor frame.
    returns:              (T, rel_action_dim) RAW relative action (arm pose -> rel9).

    Pure function with an EXPLICIT anchor -- no ``indexes`` / ``n_previous``
    resolution happens here (see ``resolve_anchor_row`` /
    ``build_relative_action_from_window``). One rel9 per arm, emitted in
    ``layout.arms`` order, so bimanual is [L, R] exactly as before.
    """
    a = np.asarray(action_window_raw, dtype=np.float32)
    anchor = np.asarray(anchor_state_row_raw, dtype=np.float32)
    assert a.ndim == 2 and a.shape[1] == layout.abs_action_dim, \
        f"action_window must be (T,{layout.abs_action_dim}) for arm_layout=" \
        f"{layout.name!r}; got {a.shape}"
    assert anchor.ndim == 1 and anchor.shape[0] == layout.state_dim, \
        f"anchor_state_row must be ({layout.state_dim},) for arm_layout=" \
        f"{layout.name!r}; got {anchor.shape}"

    rel_blocks = [
        relativize_mat16_to_rot6d(anchor[slice(*st)], a[:, slice(*act)])   # (T, 9)
        for st, act in zip(layout.state_poses, layout.action_poses)
    ]
    prefix = a[:, : layout.prefix_end]        # absolute, unchanged
    out = np.concatenate([prefix, *rel_blocks], axis=1).astype(np.float32)
    assert out.shape[1] == layout.rel_action_dim
    return out


def resolve_anchor_row(
    state_window_raw: np.ndarray,
    n_previous: int,
    layout: RelativeArmLayout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Return the single anchor state row = last observed frame (n_previous-1).

    state_window_raw: (T, 90) RAW absolute state rows over the SAME window
                      (``state[indexes]``) the action uses.
    Asserts the anchor precedes every predicted (future) row so we never
    relativize against a future observation.
    """
    s = np.asarray(state_window_raw, dtype=np.float32)
    assert s.ndim == 2 and s.shape[1] == layout.state_dim, \
        f"state_window must be (T,{layout.state_dim}); got {s.shape}"
    assert n_previous >= 1, f"n_previous must be >=1 for a valid anchor; got {n_previous}"
    anchor_idx = n_previous - 1
    assert anchor_idx < s.shape[0], \
        f"anchor idx {anchor_idx} out of range for window T={s.shape[0]}"
    # anchor is the last OBSERVED row; every future/predicted row (>= n_previous)
    # comes strictly after it.
    assert anchor_idx <= s.shape[0] - 1
    return s[anchor_idx]


def build_relative_action_from_window(
    action_window_raw: np.ndarray,
    state_window_raw: np.ndarray,
    n_previous: int,
    layout: RelativeArmLayout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Convenience wrapper used by BOTH the dataset and the stats pass.

    action_window_raw: (T, abs_action_dim) RAW action[indexes].
    state_window_raw:  (T, state_dim)      RAW state[indexes].
    n_previous:        number of memory frames (anchor = row n_previous-1).
    returns:           (T, rel_action_dim) RAW relative action.
    """
    anchor = resolve_anchor_row(state_window_raw, n_previous, layout)
    return build_relative_arm_pose_action(action_window_raw, anchor, layout)


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def _test_layout(L: RelativeArmLayout, rng: np.random.Generator) -> None:
    from data.utils.pose_math import (
        _random_se3_mat16,
        compose_mat16_and_relative_rot6d,
        mat16_to_mat44,
    )
    T, n_prev = 12, 4

    # build a fake raw window: random absolute prefix + valid SE(3) poses
    action = rng.standard_normal((T, L.abs_action_dim)).astype(np.float32)
    state = rng.standard_normal((T, L.state_dim)).astype(np.float32)
    for act_slc, st_slc in zip(L.action_poses, L.state_poses):
        action[:, slice(*act_slc)] = _random_se3_mat16(T, rng)
        state[:, slice(*st_slc)] = _random_se3_mat16(T, rng)

    rel = build_relative_action_from_window(action, state, n_prev, L)
    assert rel.shape == (T, L.rel_action_dim)

    # (a) absolute prefix carried over byte-identical
    assert np.array_equal(rel[:, : L.prefix_end], action[:, : L.prefix_end]), \
        f"{L.name}: absolute prefix (force/arm_joints/hand) was altered"

    # (b) compose(anchor, rel) reconstructs the absolute target pose exactly
    anchor = state[n_prev - 1]
    for rel_slc, act_slc, st_slc in zip(L.rel_poses, L.action_poses, L.state_poses):
        rec16 = compose_mat16_and_relative_rot6d(anchor[slice(*st_slc)], rel[:, slice(*rel_slc)])
        T_rec = mat16_to_mat44(rec16)
        T_tgt = mat16_to_mat44(action[:, slice(*act_slc)])
        assert np.allclose(T_rec, T_tgt, atol=1e-4), \
            f"{L.name}: compose did not reconstruct target pose"

    # (c) when target pose == anchor pose, rel row is identity [0,0,0,1,0,0,0,1,0]
    action_id = action.copy()
    for act_slc, st_slc in zip(L.action_poses, L.state_poses):
        action_id[:, slice(*act_slc)] = anchor[slice(*st_slc)]
    rel_id = build_relative_action_from_window(action_id, state, n_prev, L)
    id9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    for rel_slc in L.rel_poses:
        assert np.allclose(rel_id[:, slice(*rel_slc)], id9, atol=1e-4), \
            f"{L.name}: anchor==target did not give the identity rel9"

    # (d) each arm is relativized against ITS OWN anchor: perturbing one arm's anchor
    #     must move only that arm's rel9 block.
    if L.n_arms > 1:
        for i, st_slc in enumerate(L.state_poses):
            bumped = state.copy()
            bumped[:, slice(*st_slc)] = _random_se3_mat16(T, rng)
            rel_b = build_relative_action_from_window(action, bumped, n_prev, L)
            for j, rel_slc in enumerate(L.rel_poses):
                same = np.allclose(rel_b[:, slice(*rel_slc)], rel[:, slice(*rel_slc)], atol=1e-6)
                assert same == (j != i), \
                    f"{L.name}: anchor {L.arms[i]} changed rel block {L.arms[j]} unexpectedly"


def _run_tests() -> None:
    rng = np.random.default_rng(3)

    # The bimanual numbers are a published contract: erase / handover / chip / unscrew
    # checkpoints, stats and caches were all built against them. Pin them so a future
    # edit to the layout table cannot silently re-interpret existing 136-D data.
    B = BIMANUAL_LAYOUT
    assert (B.abs_action_dim, B.state_dim, B.rel_action_dim) == (150, 90, 136)
    assert B.action_poses == ((118, 134), (134, 150)) and B.state_poses == ((58, 74), (74, 90))
    assert B.rel_poses == ((118, 127), (127, 136)) and B.arms == ("left", "right")
    assert B.state_hands == ((14, 36), (36, 58))
    R = RIGHT_ONLY_LAYOUT
    assert (R.abs_action_dim, R.state_dim, R.rel_action_dim) == (75, 45, 68)
    assert R.action_poses == ((59, 75),) and R.state_poses == ((29, 45),)
    assert R.rel_poses == ((59, 68),) and R.arms == ("right",)
    # right-only R_hand lands at [7:29] -- the SAME slice the legacy 58-D layout uses
    # for the LEFT hand, so hand-pose extraction must never key on the slice alone.
    assert R.state_hands == ((7, 29),)

    for name in sorted(LAYOUTS):
        _test_layout(get_arm_layout(name), rng)

    # explicit selection only: unknown / non-string names must not fall back to bimanual
    for bad in ("both", "right", "", None, 45):
        try:
            get_arm_layout(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"get_arm_layout({bad!r}) should have raised")

    # observed-dim cross-check catches a config/data mismatch in either direction
    assert_layout_dims(R, abs_action_dim=75, state_dim=45, rel_action_dim=68, where="test")
    for kw in ({"abs_action_dim": 150}, {"state_dim": 90}, {"rel_action_dim": 136}):
        try:
            assert_layout_dims(R, where="test", **kw)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"assert_layout_dims should have rejected {kw} for right_only")

    # The deploy server and the dataset converter slice these same vectors from
    # separate checkouts, so the layouts must equal the shared declaration.
    from data.utils.layout_contract import assert_all_layouts_match_contract, load_contract
    contract = load_contract()
    version = assert_all_layouts_match_contract(LAYOUTS, contract)

    print(f"relative_action: self-tests passed for {sorted(LAYOUTS)} "
          f"(pinned dims, prefix, compose round-trip, identity, per-arm anchors, "
          f"explicit selection, dim cross-check, contract v{version} "
          f"sha256={contract['_sha256'][:8]})")


if __name__ == "__main__":
    _run_tests()
