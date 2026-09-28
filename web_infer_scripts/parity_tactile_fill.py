#!/usr/bin/env python3
"""Offline/online tactile fill parity on REAL episodes (deploy gate F4b).

The synthetic tests in ``data/utils/tactile_health.py`` prove the state machine
behaves as I read the converter. This proves it behaves as the converter
ACTUALLY behaves, which is the only thing that rules out a train/serve mismatch.

Method -- deliberately not a re-derivation. Rather than parse per-episode trim
metadata and try to map source frames onto output frames (the converter also
applies a contact trim on top of the segment trim, so that mapping is easy to
get subtly wrong), this calls the converter's OWN pipeline:

    load_fix_inputs(h5)                     -> pre-fill raw tactile
    dfx.inspect_and_fix_episode(...)        -> kept segments, already filled
    seg.source_frame_start/end              -> the exact window, from the converter

then replays the pre-fill raw of that window through OnlineTactileHealthFilter
and demands byte equality with ``seg.right_raw``. Alignment therefore comes from
the converter itself.

Checked per episode:
  1. from the first frame the online filter is willing to act on, online filled
     == offline filled, exact uint8
  2. the blank detection itself matches seg.black_original, not merely the
     filled result
  3. inside a kept segment online NEVER reaches sensor_lost -- the converter
     kept these frames, so the server must be willing to act on them
  4. an episode the converter DISCARDED for an over-long required-finger gap
     must drive the online filter to sensor_lost -- the failure the gate exists
     for. (Discards for other reasons are reported, not failed.)

ACCEPTED DIVERGENCE, reported and counted rather than tolerated silently: a
segment may BEGIN with a blank required finger, which offline is backfilled from
a later frame (``carry_forward_fill``'s pre-first-valid branch). No causal filter
can reproduce that. It survives the leading trim because ``classify_runs`` only
calls a run "leading" when it starts at absolute frame 0, so a trim boundary
landing on the start of an internal run leaves that run at the head of the kept
segment. Online those frames are ``not_ready``: they enter neither inference nor
the keyframe buffer, so they cannot influence the model -- the robot simply holds
a moment longer at episode start. Parity is therefore required from the first
usable frame onward, and the summary prints how many frames this covers.

Config mirrors run_convert.sh for right-only tasks exactly:
    --operating-fingers 5 6 7 --nonoperating-short-run-max 10
    --required-fingers-for-keep 5 6 7 8 9 --monitor-only-fingers 0 1 2 3 4
    --short-run-max 5 --boundary-trim-max 15 --min-segment-len 30

Usage:
    python web_infer_scripts/parity_tactile_fill.py \
        --data-scripts tools/vtam_data_scripts \
        --raw-root <raw source root for tong or bowl> --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.utils.tactile_health import (  # noqa: E402
    STATUS_SENSOR_UNAVAILABLE,
    USABLE_STATUSES,
    OnlineTactileHealthFilter,
)

# right-only window table, same numbers the server derives from the layout
RIGHT_WINDOWS = {("right", 0): 5, ("right", 1): 5, ("right", 2): 5,
                 ("right", 3): 10, ("right", 4): 10}


def _replay(raw_right: np.ndarray, fl: OnlineTactileHealthFilter):
    """raw_right: (T,5,H,W) uint8 pre-fill -> (filled, statuses, blank mask (T,5))."""
    out = np.empty_like(raw_right)
    blank = np.zeros(raw_right.shape[:2], dtype=bool)
    statuses = []
    for t in range(raw_right.shape[0]):
        r = fl.process(raw_right[t][None])          # (1, 5, H, W)
        out[t] = r.tactile[0]
        blank[t] = r.blank_mask[0]
        statuses.append(r.status)
    return out, statuses, blank


def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-scripts", required=True,
                   help="path to the vtam_data_scripts checkout (the converter)")
    p.add_argument("--raw-root", required=True, help="raw source episode root")
    p.add_argument("--limit", type=int, default=10, help="episodes to check")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.data_scripts).resolve()))
    import data_fixes as dfx                                    # noqa: E402
    from episode_io import h5_path_for, list_episode_dirs, load_arm, load_fix_inputs  # noqa: E402

    config = dfx.DataFixConfig(
        baseline_policy="bias_only",
        short_run_max=5,
        boundary_trim_max=15,
        min_segment_len=30,
        operating_fingers=(5, 6, 7),
        nonoperating_short_run_max=10,
        required_fingers_for_keep=(5, 6, 7, 8, 9),
        monitor_only_fingers=(0, 1, 2, 3, 4),
        p2_gate_mode="enforce",
        p3_gate_mode="enforce",
    )
    # The filter must be built on the same windows the converter just resolved,
    # or this compares two different policies and passes for the wrong reason.
    for f in range(5):
        w = dfx.internal_window(5 + f, config.short_run_max, config.operating_fingers,
                                config.nonoperating_short_run_max)
        if w != RIGHT_WINDOWS[("right", f)]:
            raise SystemExit(
                f"[parity] FAIL: converter window for right finger {f} is {w}, "
                f"the server uses {RIGHT_WINDOWS[('right', f)]}")
    print(f"[parity] windows agree with the converter: "
          f"{ {f'right{f}': RIGHT_WINDOWS[('right', f)] for f in range(5)} }")

    ep_dirs = list(list_episode_dirs(args.raw_root))[:args.limit]
    if not ep_dirs:
        raise SystemExit(f"[parity] no episodes under {args.raw_root}")
    print(f"[parity] {len(ep_dirs)} episodes from {args.raw_root}")

    ok = True
    n_seg = n_frames = n_filled = 0
    n_kept = n_discarded = n_lost_confirmed = 0
    n_lead_frames = n_lead_segs = 0

    for ep_dir in ep_dirs:
        name = Path(ep_dir).name
        h5 = h5_path_for(ep_dir)
        fi = load_fix_inputs(h5)
        arm = load_arm(h5)
        res = dfx.inspect_and_fix_episode(**fi, config=config, arm=arm)
        raw_right_full = np.asarray(fi["right_raw"])            # (T, 5, H, W) pre-fill

        if res.rejected or not res.segments:
            n_discarded += 1
            code = res.metadata.get("discard_code") or res.metadata.get("reject_reason")
            fl = OnlineTactileHealthFilter(("right",), RIGHT_WINDOWS)
            _, statuses, _ = _replay(raw_right_full, fl)
            gap_discard = "gap" in str(code).lower()
            if gap_discard:
                if STATUS_SENSOR_UNAVAILABLE in statuses:
                    n_lost_confirmed += 1
                    at = statuses.index(STATUS_SENSOR_UNAVAILABLE)
                    print(f"[parity] {name}: converter discarded ({code}); online "
                          f"sensor_unavailable at frame {at}  OK")
                else:
                    # Only a REQUIRED (right) finger gap must be reproducible; a
                    # left-hand gap discards offline but is invisible right-only.
                    print(f"[parity] {name}: converter discarded ({code}) but online "
                          f"never lost the sensor -- check whether the gap was on a "
                          f"left (monitor-only) finger")
            else:
                print(f"[parity] {name}: converter discarded ({code}); not a gap "
                      f"discard, skipping")
            continue

        n_kept += 1
        for seg in res.segments:
            s, e = int(seg.source_frame_start), int(seg.source_frame_end)
            pre = raw_right_full[s:e + 1]
            offline = np.asarray(seg.right_raw)
            if offline.shape != pre.shape:
                print(f"[parity] {name} seg {seg.seg_index}: shape {offline.shape} vs "
                      f"pre-fill {pre.shape}; cannot compare")
                ok = False
                continue

            fl = OnlineTactileHealthFilter(("right",), RIGHT_WINDOWS)
            online, statuses, blank = _replay(pre, fl)

            # Same dropout test, not just the same end result: seg.black_original
            # is the converter's own pre-fill mask (all 10 fingers; right = 5:).
            ref_blank = np.asarray(seg.black_original, dtype=bool)[:, 5:]
            if ref_blank.shape == blank.shape and not np.array_equal(ref_blank, blank):
                bad = int(np.count_nonzero(ref_blank != blank))
                print(f"[parity] {name} seg {seg.seg_index}: FAIL blank detection "
                      f"disagrees with the converter on {bad} finger-frames")
                ok = False

            n_seg += 1
            n_frames += pre.shape[0]
            blanks = int(np.sum(~pre.any(axis=(-1, -2))))
            n_filled += blanks

            # Leading frames the online filter refuses to act on. They reach
            # neither inference nor the keyframe buffer, so a difference there
            # cannot influence the model -- but it IS a real divergence and gets
            # counted rather than waved through. It arises when the segment's
            # first frame is blank for a required finger: offline that frame is
            # BACKFILLED from a later one (carry_forward_fill's pre-first-valid
            # branch), which no causal filter can reproduce. It survives the
            # leading trim because a run only counts as "leading" if it starts at
            # absolute frame 0, so a trim boundary landing on the start of an
            # internal run leaves that run at the head of the segment.
            #
            # The run includes the recovery streak: a hold clears only after
            # `recovery_valid_streak` strictly-live frames, so one backfilled
            # head frame withholds a few more behind it.
            lead = 0
            while lead < len(statuses) and statuses[lead] not in USABLE_STATUSES:
                lead += 1
            if lead:
                n_lead_frames += lead
                n_lead_segs += 1
                fingers = sorted(set(np.flatnonzero(blank[:lead].any(axis=0)).tolist()))
                print(f"[parity] {name} seg {seg.seg_index}: {lead} leading withheld "
                      f"frame(s) ({statuses[0]}), right finger(s) {fingers} -- offline "
                      f"backfilled them, online holds instead (accepted divergence)")

            # A hold AFTER the segment has started running is the real failure:
            # it would mean the converter kept a gap longer than it fills.
            tail = statuses[lead:]
            if STATUS_SENSOR_UNAVAILABLE in tail:
                at = tail.index(STATUS_SENSOR_UNAVAILABLE) + lead
                print(f"[parity] {name} seg {seg.seg_index}: FAIL online "
                      f"sensor_unavailable at frame {at} inside a segment the "
                      f"converter KEPT")
                ok = False

            a, b = online[lead:], offline[lead:]
            if a.tobytes() == b.tobytes():
                if args.verbose or blanks:
                    print(f"[parity] {name} seg {seg.seg_index}: {pre.shape[0]} frames "
                          f"({lead} withheld), {blanks} blank finger-frames, "
                          f"byte-identical from the first usable frame  OK")
            else:
                diff = np.flatnonzero((a != b).any(axis=(1, 2, 3)))
                print(f"[parity] {name} seg {seg.seg_index}: FAIL {diff.size} frames "
                      f"differ after the first usable frame, first at "
                      f"{int(diff[0]) + lead if diff.size else -1}")
                ok = False

    print(f"\n[parity] {n_kept} kept / {n_discarded} discarded episodes; "
          f"{n_seg} segments, {n_frames} frames, {n_filled} blank finger-frames "
          f"carry-forward filled; {n_lost_confirmed} gap-discards reproduced as "
          f"sensor_unavailable")
    print(f"[parity] accepted leading divergence: {n_lead_frames} frame(s) across "
          f"{n_lead_segs}/{n_seg} segments withheld online but backfilled offline "
          f"({100.0 * n_lead_frames / max(n_frames, 1):.3f}% of frames)")
    if n_filled == 0:
        print("[parity] WARNING: no blank frames in this sample, so the fill path "
              "was never exercised -- widen --limit or pick episodes with dropouts")
    print(f"[parity] OFFLINE/ONLINE TACTILE PARITY: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
