#!/usr/bin/env python3
"""
offline_client_vision_sharpa_dexmate.py
=======================================
Offline server check for ``vision_server_sharpa_dexmate.py`` (V=3, no
tactile). Replays a recorded LeRobot episode through the wire interface,
compares de-normalized predictions to GT actions in RAW units, prints
per-block MAE + dumps a GT-vs-pred plot. Vision-only sibling of
``offline_client_tactile_sharpa_dexmate.py``.

Usage
-----
    # terminal 1: start the server
    bash web_infer_scripts/run_server_vision_sharpa_dexmate.sh

    # terminal 2: this check
    python web_infer_scripts/offline_client_vision_sharpa_dexmate.py --port 5010 \\
        --data-root /home/zekai/dex-vtam/data/lerobot_dataset/lerobot_0514night_pick_cube_handover_sdh_100_episodes \\
        --episode 90 --n-chunks 4
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import socket
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same blocks as the tactile offline client (action layout is still 150-D for
# the with-wrist visual-only ckpt; f6 dims just won't be well-predicted).
ACTION_BLOCKS = [
    (0, 60, "force"),
    (60, 74, "arm_target"),
    (74, 118, "hand_target"),
    (118, 150, "arm_target_pose"),
]


def _send(sock, payload):
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _recv(sock):
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("server disconnected")
        header += chunk
    n = struct.unpack(">I", header)[0]
    body = bytearray()
    while len(body) < n:
        chunk = sock.recv(min(65536, n - len(body)))
        if not chunk:
            raise ConnectionError("server disconnected")
        body.extend(chunk)
    return pickle.loads(bytes(body))


def _rpc(sock, request):
    _send(sock, request)
    resp = _recv(sock)
    if not resp.get("ok", False):
        raise RuntimeError(
            f"server error for op={request.get('op')!r}:\n"
            f"{resp.get('traceback', resp.get('error', resp))}"
        )
    return resp


def load_episode(data_root: Path, episode: int, cam_columns):
    chunk = episode // 1000
    parquet = data_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    df = pd.read_parquet(parquet)
    n = len(df)

    # Decode each cam separately, then stack along V axis at access time:
    # cams: (V, T, H, W, 3) uint8.
    cams_decoded = {}
    for cam in cam_columns:
        cams_decoded[cam] = np.stack([
            np.array(Image.open(io.BytesIO(df[cam].iloc[i]["bytes"])).convert("RGB"))
            for i in range(n)
        ])
    states = np.stack([df["state"].iloc[i] for i in range(n)]).astype(np.float32)
    actions = np.stack([df["actions"].iloc[i] for i in range(n)]).astype(np.float32)

    prompt = None
    ep_jsonl = data_root / "meta" / "episodes.jsonl"
    if ep_jsonl.is_file():
        for line in ep_jsonl.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("episode_index") == episode:
                tasks = rec.get("tasks", [])
                if tasks:
                    prompt = tasks[0]
                break
    if prompt is None:
        tasks_jsonl = data_root / "meta" / "tasks.jsonl"
        prompt = json.loads(tasks_jsonl.read_text().splitlines()[0])["task"]

    return cams_decoded, states, actions, prompt


def main():
    p = argparse.ArgumentParser(description="Offline vision-only V=3 server check")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5010)
    p.add_argument("--data-root", required=True)
    p.add_argument("--episode", type=int, default=90)
    p.add_argument("--n-chunks", type=int, default=4)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    data_root = Path(args.data_root)

    sock = socket.create_connection((args.host, args.port))
    info = _rpc(sock, {"op": "ping"})["info"]
    action_chunk = int(info["action_chunk"])
    action_dim = int(info["action_dim"])
    cam_columns = info.get("valid_cam") or ["head_img", "left_wrist_img", "right_wrist_img"]
    print(f"[client] server: action_chunk={action_chunk}, action_dim={action_dim}, "
          f"V={info['n_view_total']}, valid_cam={cam_columns}")

    cams, states, gt_actions, prompt = load_episode(data_root, args.episode, cam_columns)
    T = len(states)
    print(f"[client] episode {args.episode}: {T} frames, cams=({len(cams)} x {cams[cam_columns[0]].shape[1:]}), "
          f"state={states.shape[1]}, action={gt_actions.shape[1]}, prompt={prompt!r}")

    n_chunks = min(args.n_chunks, T // action_chunk)
    if n_chunks == 0:
        raise RuntimeError(f"episode too short ({T} frames) for one chunk ({action_chunk}).")

    _rpc(sock, {"op": "reset"})
    pred_chunks = []
    for i in range(n_chunks):
        t = i * action_chunk
        # build (V, H, W, 3) uint8 at frame t, ordered to match valid_cam
        images = np.stack([cams[cam][t] for cam in cam_columns], axis=0)
        obs = {
            "images": images,                  # (V_rgb, H, W, 3) uint8
            "state": states[t],                # (90,) raw
            "prompt": prompt,
            "execution_step": action_chunk,
        }
        resp = _rpc(sock, {"op": "step", "obs": obs})
        action = np.asarray(resp["action"], dtype=np.float32)
        pred_chunks.append(action)
        print(f"[client] chunk {i}: t={t:4d}  pred shape={action.shape}")

    _send(sock, {"op": "shutdown"})
    sock.close()

    pred = np.concatenate(pred_chunks, axis=0)
    L = pred.shape[0]
    gt = gt_actions[:L]

    err = np.abs(pred - gt)
    mae_per_dim = err.mean(axis=0)
    print(f"\n[client] compared {L} steps over {n_chunks} chunks (RAW units)")
    print(f"[client] overall MAE = {err.mean():.5f}")
    for lo, hi, name in ACTION_BLOCKS:
        print(f"[client]   {name:16s} [{lo:3d}:{hi:3d}] MAE = {mae_per_dim[lo:hi].mean():.5f}")
    print(f"[client]   joint_targets    [ 60:118] MAE = {mae_per_dim[60:118].mean():.5f}  (what robot executes)")
    worst = np.argsort(mae_per_dim)[::-1][:10]
    print("[client] worst 10 dims (dim: MAE):")
    for d in worst:
        block = next(nm for lo, hi, nm in ACTION_BLOCKS if lo <= d < hi)
        print(f"           dim {d:3d} ({block}): {mae_per_dim[d]:.5f}")

    out = args.output or f"offline_check_vision_ep{args.episode}.png"
    ndim = pred.shape[1]
    ncols = 8
    nrows = (ndim + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    x = np.arange(L)
    starts = np.arange(0, L, action_chunk)
    for d in range(ndim):
        ax = axes[d]
        ax.plot(x, gt[:, d], color="cornflowerblue", label="ground truth", alpha=0.9)
        ax.plot(x, pred[:, d], color="tomato", linestyle="--", label="server pred", alpha=0.9)
        ax.scatter(starts, gt[starts, d], c="blue", marker="o", s=18, zorder=5)
        ax.set_title(f"dim {d} (MAE {mae_per_dim[d]:.3f})", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
    for d in range(ndim, len(axes)):
        axes[d].axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"dexmate+sharpa VISION-ONLY V=3 server offline check -- episode {args.episode}, "
        f"{n_chunks} chunks, overall MAE {err.mean():.4f}",
        fontsize=14,
    )
    fig.supxlabel("timestep")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[client] saved plot -> {out}")


if __name__ == "__main__":
    main()
