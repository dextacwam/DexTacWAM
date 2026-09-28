#!/usr/bin/env python3
"""
robot_client_tactile_sharpa_dexmate.py
======================================
Minimal real-robot client for the DexVTAM TACTILE deployment server
(``tactile_server_sharpa_dexmate.py``).

This is the tactile sibling of VTAM's ``robot_client_sharpa_dexmate.py``: same
pickle/raw-socket protocol and receding-horizon loop, but each step also sends
the per-hand tactile frame. The client sends ONLY raw values; the server does
all normalization (state, tactile, hand_pose) and de-normalization.

This is a TEMPLATE. The wire protocol and control loop are complete. Four
functions are robot-specific — implement them for your hardware (marked
ROBOT-SPECIFIC below):

    read_camera()        -> (1, H, W, 3) uint8 RGB head-camera frame
    read_tactile()       -> (2, 5, Ht, Wt) uint8 per-hand,per-finger tactile
    read_robot_state()   -> (90,) float32 raw [L_arm,R_arm,L_hand,R_hand,L_pose,R_pose]
    send_joint_targets() -> push one 58-D joint target to the controller

Usage:
    python web_infer_scripts/robot_client_tactile_sharpa_dexmate.py --host <ip> --port 5008
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time

import numpy as np

# --------------------------------------------------------------------------
# Deployment constants
# --------------------------------------------------------------------------
PROMPT = "pick up the green cube with your right hand"   # must match training
CONTROL_HZ = 30          # match the 30 fps training data
EXEC_HORIZON = 16        # steps executed from each chunk before re-planning

# Joint limits (radians) for the 58-D JOINT-TARGET command the server returns:
#   joint_targets layout = [L_arm(7), R_arm(7), L_hand(22), R_hand(22)]
# TODO: replace with your robot's real limits — this clamp is the last guard
# against a runaway prediction reaching the hardware.
JOINT_MIN = np.full(58, -np.pi, dtype=np.float32)
JOINT_MAX = np.full(58, np.pi, dtype=np.float32)


# --------------------------------------------------------------------------
# Wire protocol (must match tactile_server_sharpa_dexmate.py)
# --------------------------------------------------------------------------

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


# ==========================================================================
# ROBOT-SPECIFIC — implement these four for your hardware
# ==========================================================================

def read_camera() -> np.ndarray:
    """Return the current head-camera frame as (1, H, W, 3) uint8 RGB.

    The server resizes to 192x256 internally, so any resolution is fine, but
    it must be the same camera/viewpoint the model was trained on.
    """
    raise NotImplementedError("ROBOT-SPECIFIC: return (1, H, W, 3) uint8 RGB")


def read_tactile() -> np.ndarray:
    """Return the current tactile frame as (2, 5, Ht, Wt) uint8.

    Axis order MUST be (hand, finger, H, W) with
        hand   = (left, right)
        finger = (thumb, index, middle, ring, pinky)
    matching convert_to_vtam_no_wrist.py. Single-channel (grayscale) per finger;
    the server resizes to 192x256 and normalizes to [-1, 1].
    """
    raise NotImplementedError("ROBOT-SPECIFIC: return (2, 5, Ht, Wt) uint8 tactile")


def read_robot_state() -> np.ndarray:
    """Return current state as (90,) float32, RAW (un-normalized).

    Layout MUST be the correctaction 90-D arms-first layout:
        [L_arm(7), R_arm(7), L_hand(22), R_hand(22), L_cur_pose(16), R_cur_pose(16)]
    (convert_to_vtam_no_wrist.py). The server derives hand_pose from the hand
    slices [14:36] (left) and [36:58] (right), so the hand joints must be in
    the same order/units as training.
    """
    raise NotImplementedError("ROBOT-SPECIFIC: return (90,) float32 state")


def send_joint_targets(targets: np.ndarray) -> None:
    """Send one (58,) joint-position target to the low-level controller.

    `targets` = [L_arm(7), R_arm(7), L_hand(22), R_hand(22)], already clamped to
    JOINT_MIN/MAX by the caller.
    """
    raise NotImplementedError("ROBOT-SPECIFIC: push (58,) joint targets to controller")

# ==========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description="dexmate + sharpa tactile robot client")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5008)
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after N executed steps (0 = run until Ctrl-C)")
    args = p.parse_args()

    sock = socket.create_connection((args.host, args.port))
    info = _rpc(sock, {"op": "ping"})["info"]
    action_chunk = int(info["action_chunk"])
    lo, hi = info["joint_target_slice"]
    print(f"[client] connected: action_chunk={action_chunk}, "
          f"action_dim={info['action_dim']}, joint_targets=action[{lo}:{hi}]")
    _rpc(sock, {"op": "reset"})

    period = 1.0 / CONTROL_HZ
    execution_step = 0          # first step: server ignores this while it fills its buffer
    total_executed = 0

    try:
        while True:
            # ---- observe + infer --------------------------------------
            obs = {
                "images": read_camera(),            # (1, H, W, 3) uint8
                "tactile": read_tactile(),          # (2, 5, Ht, Wt) uint8
                "state": read_robot_state(),        # (90,) float32 raw
                "prompt": PROMPT,
                "execution_step": execution_step,
            }
            resp = _rpc(sock, {"op": "step", "obs": obs})
            # server returns the 58-D joint targets directly (action[60:118]);
            # full 150-D action also available under resp["action"] if needed.
            chunk = np.asarray(resp["joint_targets"], dtype=np.float32)  # (chunk, 58)

            # ---- execute a prefix of the chunk (receding horizon) -----
            k = min(EXEC_HORIZON, len(chunk))
            for i in range(k):
                target = np.clip(chunk[i], JOINT_MIN, JOINT_MAX)
                send_joint_targets(target)
                time.sleep(period)

            execution_step = k
            total_executed += k
            if args.max_steps and total_executed >= args.max_steps:
                print(f"[client] reached max-steps ({args.max_steps}); stopping.")
                break
    except KeyboardInterrupt:
        print("\n[client] interrupted by user.")
    finally:
        sock.close()
        print(f"[client] done — executed {total_executed} steps.")


if __name__ == "__main__":
    main()
