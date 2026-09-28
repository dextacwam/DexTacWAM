#!/usr/bin/env python3
"""
task_registry.py
================
The served language prompt's single source of truth.

Why this exists
---------------
The model is language-conditioned, and until now the prompt reached it by
whatever string the client happened to send: ``eval_genie.py`` hardcoded the
erase instruction and the server used ``obs.get("prompt", "")`` verbatim,
unvalidated, defaulting to empty. Serving bowl or tong from the robot therefore
conditioned a right-only pick policy on whiteboard erasing, and a client that
omitted the field conditioned it on nothing at all. Neither raises: the policy
still emits smooth, plausible-looking motion, which makes it the most expensive
class of bug to diagnose on hardware.

So the prompt is treated like every other piece of train/serve provenance -- the
layout contract, the stats file, the per-finger blank windows. It is named at
startup, cross-checked against the checkpoint's own domain, published in ping,
and hashed so both ends can prove they agree.

Interim status
--------------
The right long-term home for the prompt is the checkpoint's stats ``_metadata``,
next to the windows and the fill regime, so that one artifact answers every
"what was this trained with" question. This registry is the interim: the wire
interface (``task_id`` / ``task_prompt`` / ``task_prompt_sha256`` in ping, hash
echoed per step) is already the long-term one, and only the server's *source*
for the string changes later.

What is deliberately NOT here
-----------------------------
Arms. A task entry carries ``expected_arm_layout`` and the server asserts it,
but nothing ever *reads* it to build an arm list or slice a vector -- those come
from the layout contract alone. A registry that participates in constructing the
layout is exactly the two-sources-of-truth pattern that produced the blank-window
bug, where a hardcoded right-only finger policy silently applied to bimanual
checkpoints.

Prompts are copied BYTE-EXACT from the ``run_convert.sh`` branch that generated
each dataset. A paraphrase would be worse than the current hardcode: it would
produce a train/serve mismatch that looks deliberate and validated. Run

    python -m data.utils.task_registry --task bowl --verify-dataset-root <root>

to diff an entry against the dataset's own ``meta/tasks.jsonl``, which is what
training actually consumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Sentinel for offline tooling (parity replay, byte-parity capture) that must
# keep the pre-registry behaviour: prompt taken verbatim from the observation,
# no health gating, ping publishes task_id=None. The hardware client refuses a
# server in this mode, so it cannot reach the robot by accident.
NO_TASK = "none"


@dataclass(frozen=True)
class TaskEntry:
    task_id: str
    prompt: str
    domain: str
    # ASSERTION ONLY -- never a source of arms or slices. See module docstring.
    expected_arm_layout: str

    # The conversion's per-finger dropout policy, copied from this task's
    # run_convert.sh branch, in the converter's own three variables so the two
    # can be diffed line by line. Global finger indices: left 0-4, right 5-9.
    #
    # This lives here for the same reason the prompt does. The online blank-fill
    # window has to be the window each finger's TRAINING data was built with: a
    # window longer than the conversion's carries a dropout forward for frames
    # the model never saw filled, instead of holding the loop. Before this was
    # per-task it was one module constant carrying tong/bowl's split policy,
    # which is right for them and wrong for every task converted uniformly.
    #
    # Defaults are the converter's defaults (convert_to_vtam.py: all fingers
    # operating, short_run_max = nonoperating_short_run_max = 5), so a task whose
    # run_convert.sh passes no --operating-fingers needs to say nothing here.
    operating_fingers: tuple[int, ...] = tuple(range(10))
    short_run_max: int = 5
    nonoperating_short_run_max: int = 5

    @property
    def prompt_sha256(self) -> str:
        return prompt_sha256(self.prompt)

    def blank_window(self, global_finger: int) -> int:
        """Online blank-fill window for a GLOBAL finger index (left 0-4, right 5-9)."""
        return (
            self.short_run_max
            if global_finger in self.operating_fingers
            else self.nonoperating_short_run_max
        )


TASK_REGISTRY: Dict[str, TaskEntry] = {
    "tong": TaskEntry(
        task_id="tong",
        prompt=(
            "Pick up the tongs with the right hand, grasp and hold the cherry "
            "tomato steadily, move it over the plate, then release it."
        ),
        domain="20260725_pick_cherry_tomato_with_tong_right_only",
        expected_arm_layout="right_only",
        # run_convert.sh: --operating-fingers 5 6 7 --nonoperating-short-run-max 10
        operating_fingers=(5, 6, 7),
        short_run_max=5,
        nonoperating_short_run_max=10,
    ),
    "bowl": TaskEntry(
        task_id="bowl",
        prompt=(
            "Hold the lower bowls with the right thumb, lift the edge of the top "
            "bowl with the index or middle finger, slide the thumb over, then "
            "pinch and lift the top bowl."
        ),
        domain="20260725_pinch_from_bowl_with_fingers_right_only",
        expected_arm_layout="right_only",
        # run_convert.sh: --operating-fingers 5 6 7 --nonoperating-short-run-max 10
        operating_fingers=(5, 6, 7),
        short_run_max=5,
        nonoperating_short_run_max=10,
    ),
    # Same rig and same right-only policy as `tong`, but the tongs start already
    # held, so the demo omits the pick-up phase and the prompt must not describe
    # it. Serving tong's prompt here would condition the policy on an approach it
    # never performs -- and both prompts are plausible for either corpus, so the
    # domain assertion below is what actually keeps them apart.
    "placed_tong": TaskEntry(
        task_id="placed_tong",
        prompt=(
            "Grasp the cherry tomato with the tongs, move it over the plate, "
            "then release it."
        ),
        domain="20260801_placed_tong_right_only",
        expected_arm_layout="right_only",
        # run_convert.sh: --operating-fingers 5 6 7 --nonoperating-short-run-max 10
        operating_fingers=(5, 6, 7),
        short_run_max=5,
        nonoperating_short_run_max=10,
    ),
    # The first BIMANUAL entry, and the id carries the corpus version because a
    # v1 corpus (20260724_unscrew_bottle_cap) also exists: an id of plain
    # "unscrew" would leave the two distinguishable only by the domain check,
    # which fires after someone has already typed the wrong one.
    # run_convert.sh calls this task `bottlecap`; the prompt below is that
    # branch's string, and it has been checked byte-for-byte against the
    # corpus's own meta/tasks.jsonl on the training host (--verify-dataset-root, 2026-08-01).
    "unscrew_v2": TaskEntry(
        task_id="unscrew_v2",
        prompt="Unscrew the cap off the bottle and set it down.",
        domain="20260724_unscrew_bottle_cap_v2",
        expected_arm_layout="bimanual",
        # run_convert.sh `bottlecap` passes NO finger flags, so every finger was
        # converted at the strict window -- these are the converter defaults,
        # restated because an omission here is indistinguishable from an oversight.
        operating_fingers=tuple(range(10)),
        short_run_max=5,
        nonoperating_short_run_max=5,
    ),
    # `with_wrist` is in the id because 0530_pick_chip_no_wrist serves the SAME
    # domain from a different checkpoint (absolute/joint, 240 channels, 1 view).
    # So `assert_task_matches_checkpoint` below CANNOT tell those two apart --
    # unlike every other entry here, the domain check is not the thing standing
    # between you and the wrong weights. What is: action_type and guard 2's
    # action_in_channels, both of which fire before the model loads.
    #
    # Verified byte-for-byte against the corpus's meta/tasks.jsonl on the training host
    # (--verify-dataset-root, 2026-08-02), which is the only way this string can
    # be trusted: transcribing it by eye from a terminal produced a copy one
    # space short, and that copy looked entirely reasonable.
    "chip_with_wrist": TaskEntry(
        task_id="chip_with_wrist",
        prompt=(
            "Pick up the chip with your right hand, move your left hand in the air and "
            "turn the palm facing up, place the chip onto the left palm with your right "
            "hand, and then slip the chip off onto the paper plate on the table with "
            "your left hand."
        ),
        domain="0530_pick_chip_100episodes",
        expected_arm_layout="bimanual",
        # UNLIKE the entries above, these are NOT read off a run_convert.sh branch.
        # This corpus was converted in May, before the converter grew per-finger
        # dropout flags, so no such branch exists to copy. The converter defaults
        # are used by decision, matching unscrew_v2. Treat a blank-tactile hold on
        # this task as unproven rather than validated: the window is not known to
        # be what its training data was built with, only known to be no more
        # lenient than any window this pipeline has ever used.
        operating_fingers=tuple(range(10)),
        short_run_max=5,
        nonoperating_short_run_max=5,
    ),
    # ---- the 080x corpora ---------------------------------------------------
    # Each of the three below is served by TWO checkpoints: the full visuo-tactile
    # B3, and the tactile-world-model ablation whose tactile bypasses the DiT and
    # reaches the action expert directly. Both train on the SAME corpus and the
    # SAME prompt, so they share one entry here -- what separates them is the
    # config the launcher resolves (VARIANT=ablation), not the task id. Giving the
    # ablation its own id would put a second copy of the prompt in this file, and
    # two copies of a string that must be byte-exact is one copy too many.
    "cube_handover": TaskEntry(
        task_id="cube_handover",
        prompt="Pick up the cube with the right hand and hand it over to the left hand.",
        domain="20260806_cube_handover",
        expected_arm_layout="bimanual",
        # run_convert.sh `cube_handover` passes NO finger flags: uniform strict
        # window on all ten fingers, same as bottlecap. Restated rather than
        # omitted, because an omission here reads the same as an oversight.
        operating_fingers=tuple(range(10)),
        short_run_max=5,
        nonoperating_short_run_max=5,
    ),
    "wipe_white_board": TaskEntry(
        task_id="wipe_white_board",
        prompt=(
            "Hold the white board with the left hand and wipe it clean with the "
            "eraser in the right hand."
        ),
        domain="20260808_wipe_white_board",
        expected_arm_layout="bimanual",
        # run_convert.sh `wipe_white_board`: no finger flags, converter defaults.
        operating_fingers=tuple(range(10)),
        short_run_max=5,
        nonoperating_short_run_max=5,
    ),
    # HEAD CAMERA ONLY, by design rather than by omission: the right wrist camera
    # was dropped so the policy has to survive the occlusion of its own hand. That
    # is a valid_cam/max_view property of the config, not of this entry -- named
    # here only because "right_only with one camera" otherwise looks like a typo.
    "pick_place_cube": TaskEntry(
        task_id="pick_place_cube",
        prompt=(
            "Pick up the cube from the right side of the table with the right "
            "hand and place it in the middle."
        ),
        domain="20260809_pick_place_cube",
        expected_arm_layout="right_only",
        # run_convert.sh: --operating-fingers 5 6 7 --nonoperating-short-run-max 10
        operating_fingers=(5, 6, 7),
        short_run_max=5,
        nonoperating_short_run_max=10,
    ),
}


def prompt_sha256(prompt: str) -> str:
    """Hash of the exact UTF-8 bytes fed to the text encoder."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_task(task_id: str) -> Optional[TaskEntry]:
    """Resolve a ``--task`` value. ``None`` means the explicit NO_TASK sentinel.

    An unknown id raises rather than falling back: serving an unregistered task
    is the failure this module exists to prevent.
    """
    if task_id == NO_TASK:
        return None
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"unknown --task {task_id!r}. Registered: "
            f"{sorted(TASK_REGISTRY)} (or {NO_TASK!r} for offline tooling). "
            f"A new task needs a byte-exact prompt from its dataset's "
            f"meta/tasks.jsonl before it can be served."
        )
    entry = TASK_REGISTRY[task_id]
    if not entry.prompt.strip():
        raise ValueError(f"task {task_id!r} has an empty prompt")
    return entry


def read_dataset_prompt(dataset_root: str | Path) -> str:
    """The prompt training consumed, read from ``meta/tasks.jsonl``.

    Requires exactly one task: a multi-task dataset would make "the" prompt
    ambiguous, and silently picking row 0 is how the wrong one gets served.
    """
    path = Path(dataset_root) / "meta" / "tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(
            f"{path} has {len(rows)} tasks; the registry assumes exactly one "
            f"prompt per dataset"
        )
    return str(rows[0]["task"])


def verify_against_dataset(entry: TaskEntry, dataset_root: str | Path) -> None:
    """Byte-exact diff of a registry entry against the dataset it came from.

    No whitespace normalization and no case folding -- the text encoder sees the
    bytes, so anything short of equality is a train/serve mismatch.
    """
    actual = read_dataset_prompt(dataset_root)
    if actual != entry.prompt:
        raise AssertionError(
            f"prompt mismatch for task {entry.task_id!r}\n"
            f"  registry ({len(entry.prompt)} chars, sha {prompt_sha256(entry.prompt)[:12]}):\n"
            f"    {entry.prompt!r}\n"
            f"  dataset  ({len(actual)} chars, sha {prompt_sha256(actual)[:12]}):\n"
            f"    {actual!r}"
        )


def assert_task_matches_checkpoint(
    entry: TaskEntry, domain_name: str, arm_layout_name: str
) -> None:
    """Refuse a task/checkpoint pairing that disagrees, before the model loads.

    The hash echo between client and server proves only that both ends are
    talking about the same *served* prompt; it cannot notice that the server was
    launched with tong's ``--task`` against bowl's weights. This is the check
    that can, because the domain is derived from the served yaml rather than
    typed alongside the task.
    """
    if entry.domain != domain_name:
        raise ValueError(
            f"--task {entry.task_id!r} expects domain {entry.domain!r} but this "
            f"checkpoint serves {domain_name!r}. Refusing to start: the served "
            f"prompt would not be the one this checkpoint trained on."
        )
    if entry.expected_arm_layout != arm_layout_name:
        raise ValueError(
            f"--task {entry.task_id!r} expects arm_layout "
            f"{entry.expected_arm_layout!r} but the config resolves to "
            f"{arm_layout_name!r}."
        )


def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Task registry inspection / verification")
    p.add_argument("--task", help=f"task id, or {NO_TASK!r}")
    p.add_argument("--verify-dataset-root",
                   help="dataset root whose meta/tasks.jsonl must match byte-exactly")
    p.add_argument("--list", action="store_true", help="print every entry and its hash")
    args = p.parse_args()

    if args.list or not args.task:
        width = max(len(t) for t in TASK_REGISTRY)
        for tid, e in sorted(TASK_REGISTRY.items()):
            pad = " " * width
            print(f"{tid:{width}s} sha256={e.prompt_sha256}")
            print(f"{pad} domain={e.domain}")
            print(f"{pad} layout={e.expected_arm_layout}")
            print(f"{pad} prompt={e.prompt!r}")
        return 0

    entry = get_task(args.task)
    if entry is None:
        print(f"{NO_TASK}: no registry entry (offline tooling mode)")
        return 0
    print(f"task   : {entry.task_id}")
    print(f"domain : {entry.domain}")
    print(f"layout : {entry.expected_arm_layout}")
    print(f"sha256 : {entry.prompt_sha256}")
    print(f"prompt : {entry.prompt!r}")
    if args.verify_dataset_root:
        verify_against_dataset(entry, args.verify_dataset_root)
        print(f"PASS: byte-identical to {args.verify_dataset_root}/meta/tasks.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
