"""Raw client-observation layout, kept deliberately separate from model layouts.

Two different coordinate systems meet in the policy server and they must never
share a type:

* ``RelativeArmLayout`` (``data/utils/relative_action.py``) describes MODEL-space
  slices -- offsets into the vectors the checkpoint was trained on. For
  ``right_only`` the state blocks are ``arm [0:7]``, ``hand [7:29]``,
  ``pose [29:45]``.
* ``RawObservationLayout`` (here) describes what the CLIENT physically sends --
  offsets into the raw bimanual observation. The right arm's blocks live at
  ``arm [7:14]``, ``hand [36:58]``, ``pose [74:90]``.

Those are unrelated numbers describing the same physical arm. Fusing them into
one object is how ``[29:45]`` eventually gets applied to a raw 90-D vector and
silently reads the wrong 16 numbers -- so the raw source mapping is declared
here and only here, and the two layouts are always passed as separate arguments.

Note in particular that right-only ``state_hands == (7, 29)`` is the slice the
legacy bimanual state uses for the LEFT arm's joints. Nothing about a slice
alone identifies which arm it refers to; only the (layout, arm) pair does.

The server configures BOTH explicitly (``raw_obs_layout: bimanual`` +
``arm_layout: right_only``) and never infers either from an observed width: a
45-D input must not be able to masquerade as a legal raw observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per-arm block widths in the raw observation. Pinned against relative_action's
# own constants by the self-test below.
ARM7 = 7
HAND22 = 22
POSE16 = 16

# Fingers per tactile hand, ordered (thumb, index, middle, ring, pinky).
FINGERS = 5

# LEGACY fallback for the ``--task none`` offline tooling path only. These are
# tong/bowl's conversion policy, which was the only policy that existed when the
# online filter was written. It is NOT a default that generalises: a task whose
# run_convert.sh passes no --operating-fingers was converted with every finger
# strict, and serving it these numbers would tolerate blank runs twice as long
# as its training data ever contained. The per-task policy now lives on
# TaskEntry, next to the prompt, for exactly that reason.
OPERATING_FINGERS_GLOBAL = (5, 6, 7)
STRICT_BLANK_WINDOW = 5
LENIENT_BLANK_WINDOW = 10


@dataclass(frozen=True)
class RawObservationLayout:
    """Flat slices of the raw observation vector the client transmits.

    Per-arm tuples are indexed positionally against ``arms``, mirroring
    ``RelativeArmLayout`` so both layouts read the same way at a call site.
    """

    name: str
    state_dim: int
    arms: tuple[str, ...]
    # All per-arm, ordered as `arms`.
    arm_joints: tuple[tuple[int, int], ...]
    hand_joints: tuple[tuple[int, int], ...]
    eef_pose: tuple[tuple[int, int], ...]
    # Index along the raw tactile hand axis, ordered as `arms`.
    tactile_hand_index: tuple[int, ...]

    @property
    def tactile_hands(self) -> int:
        return len(self.arms)

    def arm_position(self, arm: str) -> int:
        if arm not in self.arms:
            raise KeyError(
                f"raw_obs_layout={self.name!r} has no arm {arm!r}; has {self.arms}. "
                f"The model layout asks for an arm the client never sends."
            )
        return self.arms.index(arm)

    def validate(self) -> None:
        n = len(self.arms)
        assert n >= 1, f"{self.name}: no arms"
        assert len(set(self.arms)) == n, f"{self.name}: duplicate arm names in {self.arms}"
        for field, width in (
            ("arm_joints", ARM7),
            ("hand_joints", HAND22),
            ("eef_pose", POSE16),
        ):
            blocks = getattr(self, field)
            assert len(blocks) == n, f"{self.name}: {field} has {len(blocks)} blocks, want {n}"
            for lo, hi in blocks:
                assert hi - lo == width, f"{self.name}: {field} block ({lo},{hi}) width != {width}"
        assert len(self.tactile_hand_index) == n, f"{self.name}: tactile_hand_index != n arms"
        assert sorted(self.tactile_hand_index) == list(range(n)), (
            f"{self.name}: tactile_hand_index {self.tactile_hand_index} is not a "
            f"permutation of range({n})"
        )

        # Block-major: every arm's arm_joints, then every hand, then every pose --
        # the same grouping the model-space state uses.
        expect = 0
        for field in ("arm_joints", "hand_joints", "eef_pose"):
            for lo, hi in getattr(self, field):
                assert lo == expect, (
                    f"{self.name}: {field} block starts at {lo}, expected {expect} "
                    f"(raw state must tile with no gaps or overlaps)"
                )
                expect = hi
        assert expect == self.state_dim, (
            f"{self.name}: blocks end at {expect} != state_dim {self.state_dim}"
        )


RAW_BIMANUAL_OBS_LAYOUT = RawObservationLayout(
    name="bimanual",
    state_dim=90,
    arms=("left", "right"),
    arm_joints=((0, 7), (7, 14)),
    hand_joints=((14, 36), (36, 58)),
    eef_pose=((58, 74), (74, 90)),
    tactile_hand_index=(0, 1),
)

RAW_LAYOUTS: dict[str, RawObservationLayout] = {
    RAW_BIMANUAL_OBS_LAYOUT.name: RAW_BIMANUAL_OBS_LAYOUT,
}

for _raw in RAW_LAYOUTS.values():
    _raw.validate()

# The hardware always reports both arms, whatever the policy consumes.
DEFAULT_RAW_OBS_LAYOUT = RAW_BIMANUAL_OBS_LAYOUT


def get_raw_obs_layout(name: str) -> RawObservationLayout:
    """Look up a raw layout by EXPLICIT name (yaml ``raw_obs_layout``).

    No width inference and no fallback: this names what the client is wired to
    send, which cannot be recovered from a tensor after the fact.
    """
    if not isinstance(name, str) or name not in RAW_LAYOUTS:
        raise ValueError(
            f"Unknown raw_obs_layout={name!r}; expected one of {sorted(RAW_LAYOUTS)}. "
            f"Note the model-space names ('right_only') are NOT raw layouts -- the "
            f"robot reports both arms even when the policy drives one."
        )
    return RAW_LAYOUTS[name]


def build_state_gather_index(
    raw_layout: RawObservationLayout,
    model_layout,
) -> np.ndarray:
    """Source indices that turn a raw state vector into a model state vector.

    ``state_model = state_raw[idx]``, with ``len(idx) == model_layout.state_dim``.
    When both layouts describe the same arms this is the identity permutation,
    constructed the same way -- so bimanual serving exercises the very code path
    right-only serving depends on, instead of a pass-through branch that hides it.
    """
    idx: list[int] = []
    for field in ("arm_joints", "hand_joints", "eef_pose"):
        for arm in model_layout.arms:
            lo, hi = getattr(raw_layout, field)[raw_layout.arm_position(arm)]
            idx.extend(range(lo, hi))

    out = np.asarray(idx, dtype=np.int64)
    assert out.size == model_layout.state_dim, (
        f"gather index has {out.size} entries but arm_layout={model_layout.name!r} "
        f"state_dim is {model_layout.state_dim}"
    )
    assert np.unique(out).size == out.size, "gather index repeats a source index"
    assert int(out.max()) < raw_layout.state_dim, (
        f"gather index reads index {int(out.max())} beyond raw state_dim "
        f"{raw_layout.state_dim}"
    )

    # The DESTINATION must tile the model layout's own declared blocks. A length
    # check passes on a block-order slip; this does not.
    n = len(model_layout.arms)
    dst = ARM7 * n
    for slc in model_layout.state_hands:
        assert slc == (dst, dst + HAND22), (
            f"{model_layout.name}: state_hands {slc} does not follow the arm-joint "
            f"prefix at {dst}; gather order and model layout disagree"
        )
        dst += HAND22
    for slc in model_layout.state_poses:
        assert slc == (dst, dst + POSE16), (
            f"{model_layout.name}: state_poses {slc} out of order at {dst}"
        )
        dst += POSE16
    assert dst == model_layout.state_dim
    return out


def build_tactile_hand_index(
    raw_layout: RawObservationLayout,
    model_layout,
) -> list[int]:
    """Raw tactile hand-axis indices to keep, ordered as ``model_layout.arms``."""
    return [
        raw_layout.tactile_hand_index[raw_layout.arm_position(arm)]
        for arm in model_layout.arms
    ]


def build_blank_window_table(
    raw_layout: RawObservationLayout,
    model_layout,
    task=None,
) -> dict[tuple[str, int], int]:
    """Per-(arm, model-space finger) online blank-fill window, in frames.

    Keys use the finger index WITHIN that hand's tactile view (0-4): after the
    right-only trim the server holds ``(V_hand=1, F=5, ...)`` and no longer sees
    global indices. Values resolve through the global index, so the
    operating-finger policy stays correct if the layout or hand selection changes.

    ``task`` is a ``TaskEntry`` (duck-typed: anything with ``blank_window(int)``),
    and supplies the window policy that task's corpus was actually converted
    with. It is optional only for the ``--task none`` offline tooling path, which
    predates the registry and whose byte-parity captures must keep the legacy
    constants. Passing None for a served checkpoint would hand tong/bowl's split
    policy to whatever task is loaded -- correct for them, wrong for a corpus
    converted uniformly, and wrong in the lenient direction: it would carry a
    dropout forward past anything training contained instead of holding.
    """
    table: dict[tuple[str, int], int] = {}
    for arm in model_layout.arms:
        base = raw_layout.tactile_hand_index[raw_layout.arm_position(arm)] * FINGERS
        for finger in range(FINGERS):
            g = base + finger
            table[(arm, finger)] = (
                task.blank_window(g) if task is not None
                else (STRICT_BLANK_WINDOW if g in OPERATING_FINGERS_GLOBAL
                      else LENIENT_BLANK_WINDOW)
            )
    return table


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def _run_tests() -> None:
    from data.utils.relative_action import (
        ARM7 as M_ARM7,
        BIMANUAL_LAYOUT,
        HAND22 as M_HAND22,
        POSE16 as M_POSE16,
        RIGHT_ONLY_LAYOUT,
    )

    raw = RAW_BIMANUAL_OBS_LAYOUT

    # Widths are a shared fact with the model-space layout, not an independent one.
    assert (ARM7, HAND22, POSE16) == (M_ARM7, M_HAND22, M_POSE16)
    assert raw.state_dim == BIMANUAL_LAYOUT.state_dim == 90

    # bimanual -> bimanual is the identity permutation, built by the same code.
    ident = build_state_gather_index(raw, BIMANUAL_LAYOUT)
    assert np.array_equal(ident, np.arange(90)), ident

    # bimanual -> right_only picks the right arm's three blocks, in model order.
    ro = build_state_gather_index(raw, RIGHT_ONLY_LAYOUT)
    want = np.concatenate([np.arange(7, 14), np.arange(36, 58), np.arange(74, 90)])
    assert np.array_equal(ro, want), ro
    assert ro.size == RIGHT_ONLY_LAYOUT.state_dim == 45

    # A gathered row must land exactly on the model layout's declared blocks.
    state_raw = np.arange(90, dtype=np.float32)
    state_model = state_raw[ro]
    assert np.array_equal(state_model[0:7], np.arange(7, 14)), "R arm joints misplaced"
    lo, hi = RIGHT_ONLY_LAYOUT.state_hands[0]
    assert np.array_equal(state_model[lo:hi], np.arange(36, 58)), "R hand misplaced"
    lo, hi = RIGHT_ONLY_LAYOUT.state_poses[0]
    assert np.array_equal(state_model[lo:hi], np.arange(74, 90)), "R anchor pose misplaced"

    # The anchor the server composes against must be the RIGHT arm's pose. Reading
    # the model slice [29:45] out of the raw vector would grab raw [29:45] -- part
    # of the right hand's joints -- so pin that the two differ.
    assert not np.array_equal(state_raw[lo:hi], state_model[lo:hi]), (
        "test is vacuous: model pose slice happens to equal the raw slice"
    )

    # Poisoning only LEFT blocks must not perturb the gathered model state. This is
    # the invariant the deploy dry-run then asserts end-to-end on live shapes.
    poisoned = state_raw.copy()
    for lo, hi in (raw.arm_joints[0], raw.hand_joints[0], raw.eef_pose[0]):
        poisoned[lo:hi] = -999.0
    assert np.array_equal(poisoned[ro], state_model), "left data leaked into right-only state"

    # Tactile: right_only keeps hand axis 1; bimanual keeps both, in order.
    assert build_tactile_hand_index(raw, RIGHT_ONLY_LAYOUT) == [1]
    assert build_tactile_hand_index(raw, BIMANUAL_LAYOUT) == [0, 1]

    # Blank windows: right fingers 0,1,2 are global 5,6,7 -> strict; 3,4 lenient.
    windows = build_blank_window_table(raw, RIGHT_ONLY_LAYOUT)
    assert windows == {
        ("right", 0): STRICT_BLANK_WINDOW,
        ("right", 1): STRICT_BLANK_WINDOW,
        ("right", 2): STRICT_BLANK_WINDOW,
        ("right", 3): LENIENT_BLANK_WINDOW,
        ("right", 4): LENIENT_BLANK_WINDOW,
    }, windows
    bi = build_blank_window_table(raw, BIMANUAL_LAYOUT)
    assert len(bi) == 10
    assert all(w == LENIENT_BLANK_WINDOW for (arm, _), w in bi.items() if arm == "left"), (
        "left fingers are global 0-4, none of which is an operating finger"
    )

    # A model-layout name must not resolve as a raw layout, and vice versa.
    for bad in ("right_only", "right", "", None, 90):
        try:
            get_raw_obs_layout(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"get_raw_obs_layout({bad!r}) should have raised")

    # An arm the client does not send must fail loudly, not gather garbage.
    one_arm = RawObservationLayout(
        name="_test_right_hw", state_dim=45, arms=("right",),
        arm_joints=((0, 7),), hand_joints=((7, 29),), eef_pose=((29, 45),),
        tactile_hand_index=(0,),
    )
    one_arm.validate()
    try:
        build_state_gather_index(one_arm, BIMANUAL_LAYOUT)
    except KeyError:
        pass
    else:
        raise AssertionError("gathering 'left' from a right-only raw layout must raise")

    # Malformed layouts must not construct silently.
    for kwargs, why in (
        (dict(state_dim=91), "state_dim disagrees with the blocks"),
        (dict(hand_joints=((14, 36), (36, 57))), "hand block is 21-D"),
        (dict(eef_pose=((59, 75), (75, 91))), "gap after the hand blocks"),
        (dict(tactile_hand_index=(1, 1)), "tactile index is not a permutation"),
    ):
        base = dict(
            name="_bad", state_dim=90, arms=("left", "right"),
            arm_joints=((0, 7), (7, 14)), hand_joints=((14, 36), (36, 58)),
            eef_pose=((58, 74), (74, 90)), tactile_hand_index=(0, 1),
        )
        base.update(kwargs)
        try:
            RawObservationLayout(**base).validate()
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validate() should have rejected: {why}")

    print(
        f"raw_obs_layout: self-tests passed for {sorted(RAW_LAYOUTS)} "
        f"(block tiling, identity + right_only gather, model-block destinations, "
        f"left-poison invariance, tactile hand index, blank windows, "
        f"explicit selection, malformed layouts)"
    )


if __name__ == "__main__":
    _run_tests()
