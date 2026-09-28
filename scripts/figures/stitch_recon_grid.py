"""Stitch existing Tactile-VAE inferencer reconstruction PNGs into a side-by-side
comparison grid (rows = models, columns = batches).

Each input ``recon_T{T}_batch{idx:02d}.png`` is already a self-contained 5x6
finger x (GT/Pred per channel) panel produced by ``TactileVAEInferencer.visualize``.
This script just composes a row of such panels per model so v1/v2/v3 sit on
top of each other for visual triage.

Example::

    python scripts/stitch_recon_grid.py \\
        --eval_root outputs/eval \\
        --task wipe_plate \\
        --models v1:wipe_plate_v1 v2:wipe_plate_v2_final v3-best:wipe_plate_v3_best \\
        --T 1 --batches 0 1 \\
        --output_path outputs/eval/three_way_comparison/wipe_plate_T1.png

Inputs are expected at::

    {eval_root}/{eval_dir}/recon_T{T}_batch{idx:02d}.png

If a particular file is missing, the corresponding cell is filled with a
"missing" placeholder so the rest of the grid still renders.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def _parse_model_spec(spec: str) -> Tuple[str, str]:
    """Parse ``"label:eval_dir"`` into ``(label, eval_dir)``."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--models entry must be 'label:eval_dir', got {spec!r}"
        )
    label, sub = spec.split(":", 1)
    label, sub = label.strip(), sub.strip()
    if not label or not sub:
        raise argparse.ArgumentTypeError(
            f"--models entry has empty label or eval_dir: {spec!r}"
        )
    return label, sub


def _load_or_placeholder(path: Path, size: Tuple[int, int]) -> Image.Image:
    if path.is_file():
        return Image.open(path).convert("RGB")
    placeholder = Image.new("RGB", size, color=(245, 245, 245))
    draw = ImageDraw.Draw(placeholder)
    msg = f"missing\n{path.name}"
    draw.text((10, 10), msg, fill=(120, 120, 120))
    return placeholder


def _get_font(size: int = 28) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _build_grid(
    models: Sequence[Tuple[str, str]],
    eval_root: Path,
    task: str,
    T: int,
    batches: Sequence[int],
    label_col_w: int = 220,
    pad: int = 12,
    title: Optional[str] = None,
) -> Image.Image:
    cells: List[List[Image.Image]] = []
    cell_size: Optional[Tuple[int, int]] = None

    for _, sub in models:
        row: List[Image.Image] = []
        for bi in batches:
            path = eval_root / sub / f"recon_T{T}_batch{bi:02d}.png"
            if path.is_file() and cell_size is None:
                with Image.open(path) as im:
                    cell_size = im.size
            placeholder_size = cell_size if cell_size is not None else (1024, 768)
            row.append(_load_or_placeholder(path, placeholder_size))
        cells.append(row)

    if cell_size is None:
        cell_size = cells[0][0].size

    cw, ch = cell_size

    n_rows = len(models)
    n_cols = len(batches)
    title_h = 56 if title else 0

    total_w = label_col_w + n_cols * cw + (n_cols + 1) * pad
    total_h = title_h + n_rows * ch + (n_rows + 1) * pad + 36  # +36 for column headers

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _get_font(size=32)
    label_font = _get_font(size=26)
    head_font = _get_font(size=22)

    if title:
        draw.text((label_col_w + pad, 10), title, fill=(20, 20, 20), font=title_font)

    header_y = title_h + 4
    for ci, bi in enumerate(batches):
        x = label_col_w + pad + ci * (cw + pad)
        draw.text(
            (x + cw // 2 - 60, header_y),
            f"batch {bi:02d}",
            fill=(60, 60, 60),
            font=head_font,
        )

    for ri, (label, _) in enumerate(models):
        y = title_h + 36 + pad + ri * (ch + pad)
        draw.text(
            (12, y + ch // 2 - 18),
            label,
            fill=(0, 0, 0),
            font=label_font,
        )
        for ci in range(n_cols):
            x = label_col_w + pad + ci * (cw + pad)
            canvas.paste(cells[ri][ci], (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_root", type=str, required=True,
                        help="Root containing one sub-directory per (model, task) eval.")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name used in the title (informational; not used to locate files).")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="One or more 'label:eval_dir' specs. eval_dir is relative to --eval_root.")
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--title", type=str, default=None,
                        help="Optional override; default is auto from --task / --T.")
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    if not eval_root.is_dir():
        sys.exit(f"--eval_root not found: {eval_root}")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = [_parse_model_spec(s) for s in args.models]
    title = args.title or f"Three-way reconstruction: task={args.task}  T={args.T}"

    canvas = _build_grid(
        models=models,
        eval_root=eval_root,
        task=args.task,
        T=args.T,
        batches=args.batches,
        title=title,
    )
    canvas.save(output_path)
    print(f"[stitch] wrote {output_path}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
