#!/usr/bin/env python3
"""
offline_client_tactile_sharpa_dexmate.py
========================================
Offline inference check for ``tactile_server_sharpa_dexmate.py``, driven
entirely through the server's wire interface — exactly how the real robot
client talks to it, but replaying a recorded LeRobot episode instead of a
live robot. Tactile sibling of VTAM's ``offline_client_sharpa_dexmate.py``.

What it does
------------
1. Connects to a running ``tactile_server_sharpa_dexmate.py``.
2. Loads one episode from the correctaction LeRobot dataset: head images,
   per-hand tactile, raw 90-D states, raw 150-D ground-truth actions, prompt.
3. Replays it as a real client would: every ``action_chunk`` steps it sends
   ``{images, tactile, state, prompt, execution_step}`` and receives a
   predicted action chunk. The client sends ONLY raw values — the server does
   all normalization (state / tactile / hand_pose) and de-normalization.
4. Concatenates the predicted chunks and compares them against the recorded
   ground-truth actions in RAW units: prints overall + per-block MAE
   (force / arm_target / hand_target / pose) and saves a GT-vs-pred plot.

This validates the *server* end-to-end. NOTE: it is NOT bit-identical to the
offline ``infer``-mode eval (runner/tactile_inferencer.py): that eval reads
real linspace-sampled mem frames and compares in NORMALIZED space, while this
client feeds one keyframe per chunk into the server's rolling buffer and
compares de-normalized RAW actions. The point here is to confirm the server's
normalization / prefix / de-normalization plumbing is correct end-to-end.

Usage
-----
    # terminal 1: start the server
    bash web_infer_scripts/run_server_tactile_sharpa_dexmate.sh

    # terminal 2: run this check
    python web_infer_scripts/offline_client_tactile_sharpa_dexmate.py --port 5008 \\
        --data-root /home/zekai/dex-vtam/data/lerobot_dataset/lerobot_0501_pick_cube_sdh_100_episodes_no_wrist_vtam_correctaction \\
        --episode 90 --n-chunks 6
"""

from __future__ import annotations

import argparse
import io
import json
import os
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

# repo root so we can reuse the dataset's exact tactile-unpacking helper.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.tactile_dataset import _deep_stack  # noqa: E402

# correctaction action_dim_breakdown (matches the yaml loss.action_dim_breakdown).
ACTION_BLOCKS = [
    (0, 60, "force"),
    (60, 74, "arm_target"),
    (74, 118, "hand_target"),
    (118, 150, "arm_target_pose"),
]


# ---------------------------- wire helpers ---------------------------------

def _send(sock: socket.socket, payload: dict) -> None:
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _recv(sock: socket.socket) -> dict:
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


def _rpc(sock: socket.socket, request: dict) -> dict:
    _send(sock, request)
    resp = _recv(sock)
    if not resp.get("ok", False):
        raise RuntimeError(
            f"server error for op={request.get('op')!r}:\n"
            f"{resp.get('traceback', resp.get('error', resp))}"
        )
    return resp


# ---------------------------- data loading ---------------------------------

def load_episode(data_root: Path, episode: int):
    """Return (images[T,H,W,3] uint8, tactile[T,2,5,H,W] uint8,
               states[T,90], gt_actions[T,150], prompt)."""
    chunk = episode // 1000
    parquet = data_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    df = pd.read_parquet(parquet)
    n = len(df)

    images = np.stack([
        np.array(Image.open(io.BytesIO(df["head_img"].iloc[i]["bytes"])).convert("RGB"))
        for i in range(n)
    ])
    # tactile: each row unpacks (via _deep_stack) to (V_hand=2, F=5, H, W) uint8.
    tactile_rows = df["tactile"].to_list()
    tactile = np.stack([_deep_stack(tactile_rows[i]) for i in range(n)]).astype(np.uint8)
    states = np.stack([df["state"].iloc[i] for i in range(n)]).astype(np.float32)
    actions = np.stack([df["actions"].iloc[i] for i in range(n)]).astype(np.float32)

    # prompt: episode-specific if available, else the dataset-wide task.
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

    return images, tactile, states, actions, prompt


# ------------------------------- main --------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Offline tactile server check (dexmate + sharpa)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5008)
    p.add_argument("--data-root", required=True, help="correctaction LeRobot dataset root")
    p.add_argument("--episode", type=int, default=90)
    p.add_argument("--n-chunks", type=int, default=6, help="how many chunks to replay")
    p.add_argument("--output", default=None, help="plot path (default: ./offline_check_tactile_ep<N>.png)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    images, tactile, states, gt_actions, prompt = load_episode(data_root, args.episode)
    T = len(images)
    print(f"[client] episode {args.episode}: {T} frames, tactile {tactile.shape[1:]}, "
          f"state_dim={states.shape[1]}, action_dim={gt_actions.shape[1]}, prompt={prompt!r}")

    # ---- connect + handshake ----------------------------------------------
    sock = socket.create_connection((args.host, args.port))
    info = _rpc(sock, {"op": "ping"})["info"]
    action_chunk = int(info["action_chunk"])
    action_dim = int(info["action_dim"])
    print(f"[client] server: action_chunk={action_chunk}, action_dim={action_dim}, "
          f"joint_targets=action{info.get('joint_target_slice')}")

    n_chunks = min(args.n_chunks, T // action_chunk)
    if n_chunks == 0:
        raise RuntimeError(
            f"episode too short ({T} frames) for one chunk ({action_chunk})."
        )

    # ---- replay through the wire interface --------------------------------
    _rpc(sock, {"op": "reset"})
    pred_chunks = []
    for i in range(n_chunks):
        t = i * action_chunk                       # current time of this inference
        obs = {
            "images": images[t][None],             # (V_rgb=1, H, W, 3) uint8
            "tactile": tactile[t],                 # (V_hand=2, F=5, H, W) uint8
            "state": states[t],                    # (90,) raw
            "prompt": prompt,
            "execution_step": action_chunk,        # consumed a full chunk last time
        }
        resp = _rpc(sock, {"op": "step", "obs": obs})
        action = np.asarray(resp["action"], dtype=np.float32)  # (action_chunk, 150)
        pred_chunks.append(action)
        print(f"[client] chunk {i}: t={t:4d}  pred shape={action.shape}")

    _send(sock, {"op": "shutdown"})
    sock.close()

    pred = np.concatenate(pred_chunks, axis=0)     # (n_chunks*chunk, 150)
    L = pred.shape[0]
    gt = gt_actions[:L]                            # (L, 150)

    # ---- metrics ----------------------------------------------------------
    err = np.abs(pred - gt)
    mae_per_dim = err.mean(axis=0)
    print(f"\n[client] compared {L} steps over {n_chunks} chunks (RAW units)")
    print(f"[client] overall MAE = {err.mean():.5f}")
    for lo, hi, name in ACTION_BLOCKS:
        print(f"[client]   {name:16s} [{lo:3d}:{hi:3d}] MAE = {mae_per_dim[lo:hi].mean():.5f}")
    print(f"[client]   joint_targets    [ 60:118] MAE = {mae_per_dim[60:118].mean():.5f}  "
          f"(what the robot executes)")
    worst = np.argsort(mae_per_dim)[::-1][:10]
    print("[client] worst 10 dims (dim: MAE):")
    for d in worst:
        block = next(nm for lo, hi, nm in ACTION_BLOCKS if lo <= d < hi)
        print(f"           dim {d:3d} ({block}): {mae_per_dim[d]:.5f}")

    # ---- plot: GT vs prediction, per dimension (8-col grid for 150 dims) --
    out = args.output or f"offline_check_tactile_ep{args.episode}.png"
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
        f"dexmate+sharpa TACTILE server offline check — episode {args.episode}, "
        f"{n_chunks} chunks, overall MAE {err.mean():.4f}",
        fontsize=14,
    )
    fig.supxlabel("timestep")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[client] saved plot -> {out}")


if __name__ == "__main__":
    main()
