#!/usr/bin/env python
"""Publish a stage-3 action checkpoint to a Hugging Face model repo.

Weights only. The config yaml and the normalization stats stay in git (the
deployment checkout reads them from `configs/ltx_model/<domain>/`), so what
goes to the Hub is the part git should not carry. The generated README pins
the git commit and the repo-relative paths of both, because a checkpoint
without its stat_file is not merely incomplete -- loading it against the
wrong normalization produces plausible-looking actions at the wrong scale
and never raises.

Everything in the checkpoint dir that is NOT uploaded is listed before the
upload starts, so optimizer/DeepSpeed shards can't be shipped by accident
and nothing needed can be dropped silently.

Usage
-----
    # inspect only, no network
    python scripts/publish_action_ckpt_to_hf.py \\
        --ckpt-dir outputs/stage3_action_full_.../step_20000 \\
        --config configs/ltx_model/0729_tong_right_only_eef_relative/action_model_...yaml \\
        --repo-id JensenYuan/DexTacWAM_tong \\
        --task-title "Pick up a cherry tomato with tongs (right arm)" \\
        --eval-md eval_artifacts/.../aggregate.md \\
        --dry-run

    # same command without --dry-run performs the upload

Auth comes from the ambient `hf auth login` token.
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

# Files a consumer needs to instantiate the model. Anything else in the
# checkpoint dir (optimizer state, DeepSpeed shards, rng state) is training
# scaffolding and is deliberately left behind.
ALLOW_PATTERNS = [
    "*.safetensors",
    "projector.pt",
    "config.json",
    "*.index.json",
]


def _git(*args, cwd):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return "<unavailable>"


def _matches_allow(name):
    from fnmatch import fnmatch
    return any(fnmatch(name, pat) for pat in ALLOW_PATTERNS)


def _human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024


FILE_ROLES = {
    "diffusion_pytorch_model.safetensors": "DiT action model weights",
    "config.json": "diffusers model config (architecture); required by `from_pretrained`",
    "projector.pt": ("tactile projector -- constructed for schema compatibility and "
                     "never called in forward under `proj_bypass`"),
}


def build_readme(args, cfg, repo_root, commit, included):
    tr = cfg["data"]["train"]
    domain = tr["domains"][0]
    stat_file = tr["stat_file"]
    warmstart = cfg["diffusion_model"]["model_path"]
    layout = tr.get("arm_layout", "bimanual")
    tac = cfg.get("tactile_inference", {}) or {}
    action_only = tac.get("action_only_dim_override", cfg.get("action_only_dim_override"))
    action_in = (cfg.get("diffusion_model", {}).get("config", {}) or {}).get("action_in_channels")

    eval_section = ""
    if args.eval_md and os.path.isfile(args.eval_md):
        with open(args.eval_md) as f:
            eval_section = "\n## Open-loop evaluation\n\n" + f.read().strip() + "\n"

    notes = f"\n{args.notes.strip()}\n" if args.notes else ""

    contents = "\n".join(
        f"| `{name}` | {_human(size)} | {FILE_ROLES.get(name, 'see the code repo')} |"
        for name, size in included
    )

    return f"""---
license: apache-2.0
tags:
  - robotics
  - manipulation
  - tactile
  - diffusion-policy
---

# {args.repo_id.split('/')[-1]}

{args.task_title}

Stage-3 action model (B3) from DexTacWAM. This repository holds **weights
only**. The config and the normalization statistics live in the code repo and
are required to use these weights.

## Contents

| File | Size | Purpose |
|------|------|---------|
{contents}

## What you also need

| Artifact | Where |
|----------|-------|
| Config | [`{args.config}`](https://github.com/dextacwam/DexTacWAM/blob/{commit}/{args.config}) |
| Normalization stats | [`{stat_file}`](https://github.com/dextacwam/DexTacWAM/blob/{commit}/{stat_file}) |
| Pose stats | `{tr.get('pose_stats_path')}` |
| Code | `dextacwam/DexTacWAM` @ `{commit}` |

The stats file is not optional. It carries the q01/q99 used to de-normalize
the predicted action; pairing these weights with a different corpus's stats
yields well-formed but wrongly-scaled actions and raises no error.

## Model card

| | |
|---|---|
| Corpus | `{domain}` |
| Train / val episodes | `[{min(tr['episodes'])}..{max(tr['episodes'])}]` / `{cfg['data']['val']['episodes']}` |
| Arm layout | `{layout}` |
| Action type | `{tr['action_type']}` / `{tr['action_space']}` |
| Action vector | `{action_in}`-D = `{action_only}` relative action + state echo |
| Checkpoint step | `{args.step}` |
| WM warmstart | `{warmstart}` |
{eval_section}{notes}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="step_N directory to publish")
    p.add_argument("--config", required=True, help="repo-relative action_model yaml")
    p.add_argument("--repo-id", required=True, help="e.g. JensenYuan/DexTacWAM_tong")
    p.add_argument("--task-title", default="", help="one-line description for the README")
    p.add_argument("--eval-md", default=None, help="aggregate.md to embed verbatim")
    p.add_argument("--notes", default="", help="extra markdown appended to the README")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="inspect and print, do not upload")
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    if not os.path.isdir(args.ckpt_dir):
        sys.exit(f"ERROR: ckpt dir not found: {args.ckpt_dir}")
    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    args.step = os.path.basename(args.ckpt_dir.rstrip("/"))
    cfg = yaml.safe_load(open(args.config))
    commit = _git("rev-parse", "HEAD", cwd=repo_root)

    if "FILL_ME" in json.dumps(cfg):
        sys.exit("ERROR: config still contains the FILL_ME sentinel; refusing to publish.")

    # ---- classify every file in the checkpoint dir ------------------------
    included, excluded = [], []
    for name in sorted(os.listdir(args.ckpt_dir)):
        path = os.path.join(args.ckpt_dir, name)
        size = os.path.getsize(path) if os.path.isfile(path) else -1
        (included if _matches_allow(name) else excluded).append((name, size))

    print(f"==== {args.repo_id}")
    print(f"  ckpt   : {args.ckpt_dir}")
    print(f"  config : {args.config}")
    print(f"  commit : {commit}")
    print("  UPLOAD:")
    for name, size in included:
        print(f"    + {name:50s} {_human(size) if size >= 0 else '<dir>'}")
    print("  SKIP:")
    for name, size in excluded:
        print(f"    - {name:50s} {_human(size) if size >= 0 else '<dir>'}")
    if not any(n.endswith(".safetensors") for n, _ in included):
        sys.exit("ERROR: no *.safetensors in the checkpoint dir -- wrong path?")

    readme = build_readme(args, cfg, repo_root, commit, included)
    print("  README:")
    print("\n".join("    | " + line for line in readme.splitlines()))

    if args.dry_run:
        print("\n  --dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    api.upload_folder(folder_path=args.ckpt_dir, repo_id=args.repo_id,
                      repo_type="model", allow_patterns=ALLOW_PATTERNS,
                      commit_message=f"Add {args.step} weights")
    print(f"\n  done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
