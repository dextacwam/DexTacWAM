#!/usr/bin/env python3
"""Record a policy server's exact responses to a fixed observation sequence.

Half of the bimanual byte-parity regression: the layout refactor replaced every
hardcoded bimanual slice in the server with layout lookups, and F1/F2/F3 only
prove the RIGHT-ONLY path. This script pins down whether erase/chip bimanual
serving still produces the identical bytes it did before.

Run it twice against the SAME checkpoint -- once with the pre-refactor server
up, once with the refactored one -- then diff with compare_server_captures.py.

The observations come from a real episode, so they are byte-identical across
runs by construction; the capture records a sha256 of every observation it sent
so the comparison can PROVE both servers saw the same input rather than assume
it. A parity pass on differing inputs would be meaningless.

This client is deliberately tolerant of the server's ping: the pre-refactor
server does not report arms / camera_names / raw_obs_layout, so nothing here may
require them. It records whatever ping returns and moves on.

Usage:
    # with the OLD server listening
    python web_infer_scripts/capture_server_responses.py -c <cfg> \
        --data-root <ds> --episode 90 --n-chunks 3 \
        --label pre-refactor --out /tmp/cap_pre.pkl
    # then the NEW server on the same port, same GPU, same checkpoint
    python web_infer_scripts/capture_server_responses.py -c <cfg> \
        --data-root <ds> --episode 90 --n-chunks 3 \
        --label refactored --out /tmp/cap_new.pkl

IMPORTANT: run both servers on the SAME GPU. Different GPU models can produce
different floating-point results for identical code, which would break byte
parity for reasons that have nothing to do with the refactor.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.utils.raw_obs_layout import get_raw_obs_layout  # noqa: E402
from data.utils.relative_action import get_arm_layout  # noqa: E402
from offline_client_relative import (  # noqa: E402
    _rpc,
    _send,
    embed_state,
    embed_tactile,
    load_episode,
)


def _obs_sha256(obs: dict) -> str:
    h = hashlib.sha256()
    for key in ("images", "tactile", "state"):
        a = np.ascontiguousarray(obs[key])
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(str(obs["prompt"]).encode())
    h.update(str(obs["execution_step"]).encode())
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--episode", type=int, default=90)
    p.add_argument("--n-chunks", type=int, default=3)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5008)
    p.add_argument("--label", required=True, help="e.g. pre-refactor / refactored")
    p.add_argument("--out", required=True, help="output .pkl")
    p.add_argument("--shutdown-server", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())
    tcfg = cfg["data"]["train"]
    layout = get_arm_layout(str(tcfg.get("arm_layout", "bimanual")))
    raw_layout = get_raw_obs_layout(str(tcfg.get("raw_obs_layout", "bimanual")))
    cams = [str(c) for c in tcfg["valid_cam"]]

    images, tactile, states, gt_actions, prompt = load_episode(
        Path(args.data_root), args.episode, cams)
    T = len(images)
    print(f"[capture] {args.label}: episode {args.episode}, {T} frames, "
          f"layout={layout.name}, cams={cams}")

    sock = socket.create_connection((args.host, args.port))
    ping = _rpc(sock, {"op": "ping"})["info"]
    action_chunk = int(ping["action_chunk"])
    print(f"[capture] ping: action_chunk={action_chunk}, "
          f"action_mode={ping.get('action_mode', ping.get('action_type'))}, "
          f"arm_layout={ping.get('arm_layout', '<absent: pre-refactor server>')}")

    n_chunks = min(args.n_chunks, T // action_chunk)
    if n_chunks == 0:
        raise SystemExit(f"[capture] episode too short ({T}) for one chunk ({action_chunk})")

    _rpc(sock, {"op": "reset"})
    chunks, obs_hashes = [], []
    for i in range(n_chunks):
        t = i * action_chunk
        obs = {
            "images": images[t],
            "tactile": embed_tactile(tactile[t], raw_layout, layout),
            "state": embed_state(states[t], raw_layout, layout, 0.0),
            "prompt": prompt,
            "execution_step": action_chunk,
        }
        obs_hashes.append(_obs_sha256(obs))
        resp = _rpc(sock, {"op": "step", "obs": obs})
        rec = {k: v for k, v in resp.items() if k != "ok"}
        chunks.append(rec)
        shapes = {k: getattr(v, "shape", type(v).__name__) for k, v in rec.items()}
        print(f"[capture] chunk {i}: obs {obs_hashes[-1][:12]} -> {shapes}")

    if args.shutdown_server:
        _send(sock, {"op": "shutdown"})
    sock.close()

    try:
        client_commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        client_commit = "unknown"

    out = {
        "label": args.label,
        "config": str(cfg_path),
        "data_root": str(args.data_root),
        "episode": args.episode,
        "n_chunks": n_chunks,
        "arm_layout": layout.name,
        "cameras": cams,
        "client_commit": client_commit,
        "ping": ping,
        "obs_sha256": obs_hashes,
        "chunks": chunks,
    }
    Path(args.out).write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"[capture] wrote {args.out} ({n_chunks} chunks)")


if __name__ == "__main__":
    main()
