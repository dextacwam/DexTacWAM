#!/usr/bin/env python3
"""scripts/make_raw_tactile_figure.py

A figure showing what the raw Sharpa visuo-tactile input actually looks like,
contrasting a pre-contact instant with a contact instant.

The layout is two blocks side by side, one per instant. Each block stacks a
camera row (head plus both wrists) over one tactile row per hand, so that all
ten fingertips are visible and the before/during comparison stays horizontal.
Panels are drawn at the size they occupy in the paper, so label point sizes in
the PDF are the point sizes the reader sees.

Frames come from the native teleop episode: camera frames from the MP4s and raw
gel images from the HDF5, both at full resolution. Frame indices are shared
because the LeRobot conversion for this episode applied no contact trim.

Example:
    python scripts/make_raw_tactile_figure.py \
        --episode-dir data/datasets/20260808_wipe_white_board/success/episode_0004 \
        --before 100 --during 400 \
        --width-in 7.03 --font-size 6.8 \
        --out figures/raw_tactile_observations
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
# IEEE PDF eXpress rejects Type 3 fonts; 42 embeds TrueType instead.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import h5py
import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CAMERAS = [("head_left_rgb", "Head"), ("left_wrist", "Left wrist"),
           ("right_wrist", "Right wrist")]


def read_frames(mp4: str, indices) -> dict:
    """Decode the requested frames from an MP4 at native resolution."""
    import decord

    vr = decord.VideoReader(mp4)
    idx = sorted(set(indices))
    batch = vr.get_batch(idx).asnumpy()
    return {i: batch[k] for k, i in enumerate(idx)}


def parse_crop(spec: str):
    x0, y0, w, h = (int(v) for v in spec.split(","))
    return lambda im: im[y0:y0 + h, x0:x0 + w]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--before", type=int, required=True)
    ap.add_argument("--during", type=int, required=True)
    ap.add_argument("--fingers", default=",".join(FINGERS),
                    help="Comma-separated fingertips shown for each hand.")
    ap.add_argument("--fps", type=float, default=30.0)
    # The cameras are 16:9 while the gel images are 4:3. Cropping to 4:3 keeps
    # every panel undistorted and zooms the head view in on the hands.
    ap.add_argument("--head-crop", default="90,0,400,300")
    ap.add_argument("--wrist-crop", default="80,0,480,360")
    ap.add_argument("--before-note", default="both hands in free space")
    ap.add_argument("--during-note",
                    default="left hand stabilizes board; right hand loads eraser")
    ap.add_argument("--width-in", type=float, default=7.03,
                    help="Rendered width in the paper, so fonts are 1:1.")
    ap.add_argument("--font-size", type=float, default=6.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ep = args.episode_dir.rstrip("/")
    name = ep.rsplit("/", 1)[-1]

    shown = [f.strip() for f in args.fingers.split(",")]
    bad = [f for f in shown if f not in FINGERS]
    if bad:
        raise SystemExit(f"unknown finger(s) {bad}; choose from {FINGERS}")
    idx = [FINGERS.index(f) for f in shown]
    instants = (args.before, args.during)

    with h5py.File(f"{ep}/{name}.h5", "r") as f:
        gel = {hand: {t: f[f"{hand}_hand_tactile_raw"][t][idx] for t in instants}
               for hand in ("left", "right")}

    crops = [parse_crop(args.head_crop)] + [parse_crop(args.wrist_crop)] * 2
    cams = []
    for (stream, _), crop in zip(CAMERAS, crops):
        frames = read_frames(f"{ep}/{name}_{stream}.mp4", instants)
        cams.append({t: crop(im) for t, im in frames.items()})

    # Every gel panel shares one grayscale range so the blocks are directly
    # comparable and contact cannot be faked by per-panel rescaling.
    stack = np.stack([g for hand in gel.values() for g in hand.values()])
    vmin, vmax = float(stack.min()), float(stack.max())

    nf = len(shown)
    W, fs = args.width_in, args.font_size

    label_w = 0.098 * fs      # left margin holding the row labels
    group_gap = 0.17          # between the before and during blocks
    pg = 0.018                # gap between neighbouring panels

    panel_w = (W - label_w - group_gap) / (2 * nf)
    group_w = nf * panel_w
    tac_h = panel_w * 0.75
    cam_w = group_w / 3
    cam_h = cam_w * 0.75

    band = 0.020 + fs / 72.0  # a single line of text plus breathing room
    row_gap = 0.020
    fig_h = band + band + cam_h + band + tac_h + row_gap + tac_h + band

    fig = plt.figure(figsize=(W, fig_h))

    def place(x0: float, y_top: float, w: float, h: float):
        """Add an axes positioned from the top-left corner, in inches."""
        return fig.add_axes([x0 / W, 1 - (y_top + h) / fig_h, w / W, h / fig_h])

    def show(ax, img):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="nearest", aspect="auto")
        else:
            ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.35")

    y_cam = 2 * band
    y_left = y_cam + cam_h + band
    y_right = y_left + tac_h + row_gap

    blocks = [(args.before, "Before contact", args.before_note),
              (args.during, "During contact", args.during_note)]

    for gi, (t, title, note) in enumerate(blocks):
        gx = label_w + gi * (group_w + group_gap)

        fig.text((gx + group_w / 2) / W, 1 - 0.006,
                 f"{title} ($t={t / args.fps:.1f}$ s)",
                 ha="center", va="top", fontsize=fs)
        # A rule under the title shows how far each block extends.
        fig.add_artist(plt.Line2D([gx / W, (gx + group_w) / W],
                                  [1 - (band - 0.006) / fig_h] * 2,
                                  lw=0.5, color="0.35"))

        for k, (cam, (_, cam_name)) in enumerate(zip(cams, CAMERAS)):
            ax = place(gx + k * cam_w + pg / 2, y_cam, cam_w - pg, cam_h)
            show(ax, cam[t])
            ax.set_title(cam_name, fontsize=fs, pad=1.8)

        for hand, y in (("left", y_left), ("right", y_right)):
            for k, img in enumerate(gel[hand][t]):
                ax = place(gx + k * panel_w + pg / 2, y, panel_w - pg, tac_h)
                show(ax, img)
                if hand == "left":
                    ax.set_title(shown[k].capitalize(), fontsize=fs, pad=1.8)

        fig.text((gx + group_w / 2) / W, 0.010, note, ha="center", va="bottom",
                 fontsize=fs, style="italic", color="0.25")

    for y, h, text in ((y_cam, cam_h, "Cameras"),
                       (y_left, tac_h, "Left hand"),
                       (y_right, tac_h, "Right hand")):
        fig.text((label_w - 0.035) / W, 1 - (y + h / 2) / fig_h, text,
                 ha="right", va="center", fontsize=fs)

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=400)
    print(f"wrote {args.out}.pdf / .png  ({W:.2f} x {fig_h:.2f} in), "
          f"tactile panel {panel_w:.2f}in, gel range [{vmin:.0f}, {vmax:.0f}]")


if __name__ == "__main__":
    main()
