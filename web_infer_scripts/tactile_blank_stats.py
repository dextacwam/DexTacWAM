#!/usr/bin/env python3
"""Right-hand tactile dropout statistics on the REAL corpus, for choosing a
`degraded` policy instead of guessing one.

Motivating correction to the obvious reading of the parity numbers: the model
consumes FIVE right fingers, so a blank has to be judged against 5, not 10, and
"1.4% of finger-frames" is not the rate at which the robot would stop. What
stops the robot is a control tick on which ANY required finger is blank, and
ticks are not independent of each other.

The second correction is architectural, and it is the reason this script
measures two cadences. The client executes `--threshold` rows per chunk before
it sends another observation, so today the SERVER-side filter runs once per
chunk (1.8 s at 54 rows / 30 Hz), not once per frame. Every window in
`tactile_health.py` is expressed in 30 Hz converter frames, so at that cadence a
carry-forward does not reproduce the converter's 33 ms fill -- it inserts a
frame one whole chunk old. This script reports:

  observed   -- the 30 Hz statistics, the cadence the converter filled at
  sampled    -- what the server actually sees, every `--threshold`-th frame,
                averaged over all chunk phases

Emitted (per the review):
  1. tick-level degraded rate: frames with >= 1 right finger blank
  2. simultaneous-blank histogram, 0..5 fingers
  3. per-finger blank rate, run count, run-length p50/p90/p95/p99/max, split
     operating (0,1,2 -> W=5) vs monitor (3,4 -> W=10)
  4. degraded run lengths in frames and ms
  5. blanks vs contact/action phase: |F| and hand-target speed at blank frames
     against the segment as a whole
  6. carry-forward confidence: |last_good - first_good_after| across each gap,
     against the same finger's frame-to-frame change (the ratio is the honest
     figure -- an absolute grey-level delta means nothing on its own)
  7. hold rate under the current any-blank policy, and under age<=1/2/3
     permissive variants, at both cadences

Usage:
    python web_infer_scripts/tactile_blank_stats.py \
        --data-scripts tools/vtam_data_scripts_qc \
        --raw-root <tong or bowl raw success root> --limit 100 \
        --json /tmp/tong_blank_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RIGHT_WINDOWS = {0: 5, 1: 5, 2: 5, 3: 10, 4: 10}
OPERATING = (0, 1, 2)      # source fingers 5,6,7
MONITOR = (3, 4)           # source fingers 8,9
PCTS = (50, 90, 95, 99)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a 1-D bool array as (start, length)."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def _pct_line(name: str, v: list[int], hz: float) -> str:
    if not v:
        return f"    {name:<26s} none"
    a = np.asarray(v)
    q = np.percentile(a, PCTS)
    return (f"    {name:<26s} n={a.size:<6d} "
            + " ".join(f"p{p}={x:.0f}" for p, x in zip(PCTS, q))
            + f" max={a.max():d}  ({1000.0 * a.max() / hz:.0f} ms max)")


def _sample_hold_rates(blank: np.ndarray, threshold: int) -> dict:
    """Hold rate at the server's real cadence, averaged over every chunk phase.

    `blank` is (T,5). Returns the fraction of sampled ticks that would be
    withheld under the current any-blank policy, plus the staleness a
    carry-forward would carry at that cadence.
    """
    T = blank.shape[0]
    any_blank = blank.any(axis=1)
    rates, stale = [], []
    for phase in range(min(threshold, T)):
        idx = np.arange(phase, T, threshold)
        if idx.size == 0:
            continue
        rates.append(float(any_blank[idx].mean()))
        # If a sampled tick is blank, the newest frame the SERVER holds is the
        # previous sampled tick -- one whole chunk back, not one frame back.
        stale.append(int(np.count_nonzero(any_blank[idx])))
    return {"n_phase": len(rates),
            "hold_rate_mean": float(np.mean(rates)) if rates else 0.0,
            "hold_rate_min": float(np.min(rates)) if rates else 0.0,
            "hold_rate_max": float(np.max(rates)) if rates else 0.0,
            "n_blank_ticks": int(np.sum(stale))}


def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-scripts", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--threshold", type=int, default=54,
                   help="rows executed per chunk = server observation period")
    p.add_argument("--command-hz", type=float, default=30.0)
    p.add_argument("--json", default=None, help="write raw aggregates here")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.data_scripts).resolve()))
    import data_fixes as dfx                                        # noqa: E402
    from episode_io import (h5_path_for, list_episode_dirs, load_arm,  # noqa: E402
                            load_fix_inputs)

    config = dfx.DataFixConfig(
        baseline_policy="bias_only", short_run_max=5, boundary_trim_max=15,
        min_segment_len=30, operating_fingers=(5, 6, 7),
        nonoperating_short_run_max=10, required_fingers_for_keep=(5, 6, 7, 8, 9),
        monitor_only_fingers=(0, 1, 2, 3, 4),
        p2_gate_mode="enforce", p3_gate_mode="enforce")

    hz = args.command_hz
    ep_dirs = list(list_episode_dirs(args.raw_root))[:args.limit]
    if not ep_dirs:
        raise SystemExit(f"[stats] no episodes under {args.raw_root}")
    print(f"[stats] {len(ep_dirs)} episodes from {args.raw_root}; "
          f"chunk={args.threshold} rows = {args.threshold / hz:.2f} s at {hz:g} Hz")

    n_seg = n_frames = 0
    per_finger_blank = np.zeros(5, dtype=np.int64)
    simult = np.zeros(6, dtype=np.int64)              # 0..5 fingers blank
    finger_runs: dict[int, list[int]] = {f: [] for f in range(5)}
    tick_runs: list[int] = []
    unfillable = 0                                    # run longer than the window
    gap_delta: list[float] = []                       # |last_good - first_good_after|
    gap_ratio: list[float] = []                       # ...over the frame-to-frame change
    f_blank: list[float] = []                         # |F| at blank frames
    f_all: list[float] = []
    v_blank: list[float] = []                         # hand-target speed at blank frames
    v_all: list[float] = []
    sampled = {"hold": [], "ticks": 0, "blank_ticks": 0}

    for ep_dir in ep_dirs:
        name = Path(ep_dir).name
        h5 = h5_path_for(ep_dir)
        fi = load_fix_inputs(h5)
        res = dfx.inspect_and_fix_episode(**fi, config=config, arm=load_arm(h5))
        if res.rejected or not res.segments:
            continue
        raw_full = np.asarray(fi["right_raw"])
        f6_full = np.asarray(fi["right_f6"], dtype=np.float64)
        tgt_full = np.asarray(fi["right_hand_target"], dtype=np.float64)

        for seg in res.segments:
            s, e = int(seg.source_frame_start), int(seg.source_frame_end)
            pre = raw_full[s:e + 1]                              # (T,5,H,W) pre-fill
            T = pre.shape[0]
            blank = ~pre.any(axis=(-1, -2))                      # (T,5)
            n_seg += 1
            n_frames += T
            per_finger_blank += blank.sum(axis=0)
            np.add.at(simult, blank.sum(axis=1), 1)

            tick_blank = blank.any(axis=1)
            tick_runs.extend(L for _, L in _runs(tick_blank))
            for f in range(5):
                for st, L in _runs(blank[:, f]):
                    finger_runs[f].append(L)
                    if L > RIGHT_WINDOWS[f]:
                        unfillable += 1
                    # Carry-forward confidence: how much did this finger's image
                    # actually change across the gap it was filled through?
                    if st > 0 and st + L < T:
                        a = pre[st - 1, f].astype(np.float32)
                        b = pre[st + L, f].astype(np.float32)
                        d = float(np.abs(a - b).mean())
                        gap_delta.append(d)
                        good = ~blank[:, f]
                        pair = good[:-1] & good[1:]
                        if pair.any():
                            i = np.flatnonzero(pair)
                            step = float(np.abs(
                                pre[i + 1, f].astype(np.float32)
                                - pre[i, f].astype(np.float32)).mean())
                            if step > 1e-6:
                                gap_ratio.append(d / step)

            # Contact / motion phase. |F| is the 3-axis force magnitude summed
            # over the right fingers; speed is the L2 step of the 22-D hand
            # target. Both at the segment's own frames, so the comparison is
            # blank-vs-rest within the same episode.
            f6 = f6_full[s:e + 1]
            fmag = np.linalg.norm(f6[..., :3], axis=-1).sum(axis=-1) \
                if f6.ndim == 3 else np.abs(f6).sum(axis=-1)
            tgt = tgt_full[s:e + 1]
            spd = np.concatenate(([0.0], np.linalg.norm(np.diff(tgt, axis=0), axis=-1)))
            f_all.extend(fmag.tolist())
            v_all.extend(spd.tolist())
            if tick_blank.any():
                f_blank.extend(fmag[tick_blank].tolist())
                v_blank.extend(spd[tick_blank].tolist())

            sr = _sample_hold_rates(blank, args.threshold)
            sampled["hold"].append(sr["hold_rate_mean"])
            sampled["ticks"] += int(np.ceil(T / args.threshold))
            sampled["blank_ticks"] += sr["n_blank_ticks"] // max(sr["n_phase"], 1)

    if n_frames == 0:
        raise SystemExit("[stats] no kept segments")

    tick_blank_frames = int(sum(tick_runs))
    print(f"\n[stats] {n_seg} segments, {n_frames} frames "
          f"({n_frames / hz / 60:.1f} min at {hz:g} Hz)\n")

    print("  1. tick-level degraded rate (>=1 right finger blank)")
    print(f"     {tick_blank_frames} / {n_frames} frames = "
          f"{100.0 * tick_blank_frames / n_frames:.3f}%")
    print(f"     finger-frames: {int(per_finger_blank.sum())} / {n_frames * 5} = "
          f"{100.0 * per_finger_blank.sum() / (n_frames * 5):.3f}%\n")

    print("  2. simultaneous blank fingers per frame")
    for k in range(6):
        if simult[k]:
            print(f"     {k} blank: {simult[k]:>8d}  {100.0 * simult[k] / n_frames:6.3f}%")
    print()

    print("  3. per-finger (0,1,2 operating W=5; 3,4 monitor W=10)")
    for f in range(5):
        role = "op " if f in OPERATING else "mon"
        rate = 100.0 * per_finger_blank[f] / n_frames
        print(f"     finger {f} [{role} W={RIGHT_WINDOWS[f]:>2d}] blank {rate:6.3f}%")
        print(_pct_line("run length (frames)", finger_runs[f], hz))
    print(f"     runs longer than their window (would be sensor_lost): {unfillable}\n")

    print("  4. degraded runs (consecutive frames with any blank)")
    print(_pct_line("run length (frames)", tick_runs, hz))
    if tick_runs:
        a = np.asarray(tick_runs)
        for L in (1, 2, 3, 5):
            print(f"     <= {L} frame(s): {100.0 * (a <= L).mean():5.1f}% of runs "
                  f"({1000.0 * L / hz:.0f} ms)")
    print()

    print("  5. blank vs contact / action phase (median, blank frames vs all)")
    if f_blank:
        print(f"     |F| sum over fingers  blank {np.median(f_blank):9.3f}   "
              f"all {np.median(f_all):9.3f}")
        print(f"     hand-target step L2   blank {np.median(v_blank):9.5f}   "
              f"all {np.median(v_all):9.5f}")
    print()

    print("  6. carry-forward confidence across a gap")
    if gap_delta:
        print(f"     |last_good - first_good_after|  median "
              f"{np.median(gap_delta):.2f} grey levels "
              f"(p90 {np.percentile(gap_delta, 90):.2f})")
    if gap_ratio:
        print(f"     ...as a multiple of the same finger's frame-to-frame change: "
              f"median {np.median(gap_ratio):.2f}x "
              f"(p90 {np.percentile(gap_ratio, 90):.2f}x)")
        print("     ~1x means the gap costs no more than one ordinary frame step")
    print()

    obs_rate = 100.0 * tick_blank_frames / n_frames
    samp = 100.0 * float(np.mean(sampled["hold"])) if sampled["hold"] else 0.0
    print("  7. hold rate under the current any-blank policy")
    print(f"     at 30 Hz (every frame judged):        {obs_rate:6.3f}% of frames")
    print(f"     at the server's real cadence:         {samp:6.3f}% of chunks "
          f"(~{sampled['ticks']} observation ticks)")
    print(f"     each hold costs one tick + one inference, then re-observes; "
          f"{100.0 * (np.asarray(tick_runs) <= 2).mean() if tick_runs else 0:.0f}% "
          f"of degraded runs are over within 2 frames (67 ms)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "frames": n_frames, "segments": n_seg,
            "per_finger_blank": per_finger_blank.tolist(),
            "simultaneous": simult.tolist(),
            "finger_runs": {str(k): v for k, v in finger_runs.items()},
            "tick_runs": tick_runs, "unfillable_runs": unfillable,
            "gap_delta": gap_delta, "gap_ratio": gap_ratio,
            "tick_blank_frames": tick_blank_frames,
            "sampled_hold_rate": samp / 100.0, "threshold": args.threshold,
        }, indent=2))
        print(f"\n[stats] wrote {args.json}")


if __name__ == "__main__":
    main()
