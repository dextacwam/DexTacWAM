#!/usr/bin/env python3
"""
robot_client_vision_sharpa_dexmate.py
=====================================
Vision-only V=3 sibling of ``robot_client_tactile_sharpa_dexmate.py``. Same
pickle/raw-socket protocol; the obs dict drops `tactile`, and `images` is
(3, H, W, 3) covering head + L_wrist + R_wrist (in that order).

Three ROBOT-SPECIFIC functions to implement for your hardware.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time

import numpy as np

PROMPT = "pick up the cyan cube on the table with your left hand, hand it over to the right hand, and place the cube down on the table with your right hand"
CONTROL_HZ = 30
EXEC_HORIZON = 16

# 58-D joint-target layout = [L_arm(7), R_arm(7), L_hand(22), R_hand(22)].
JOINT_MIN = np.full(58, -np.pi, dtype=np.float32)
JOINT_MAX = np.full(58, np.pi, dtype=np.float32)


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


# ==========================================================================
# ROBOT-SPECIFIC -- implement for your hardware
# ==========================================================================

def read_cameras() -> np.ndarray:
    """Return (3, H, W, 3) uint8 RGB stacked in order
    [head_img, left_wrist_img, right_wrist_img]. Order MUST match the yaml's
    `valid_cam:`; the server enforces (V_rgb == n_view), but cannot detect
    swapped cams."""
    raise NotImplementedError("ROBOT-SPECIFIC: return (3, H, W, 3) uint8")


def read_robot_state() -> np.ndarray:
    """Return (90,) float32 RAW state:
    [L_arm(7), R_arm(7), L_hand(22), R_hand(22), L_cur_pose(16), R_cur_pose(16)]."""
    raise NotImplementedError("ROBOT-SPECIFIC: return (90,) float32 state")


def send_joint_targets(targets: np.ndarray) -> None:
    """Send (58,) joint-target = [L_arm(7), R_arm(7), L_hand(22), R_hand(22)]
    to your low-level controller. Already clamped to JOINT_MIN/MAX."""
    raise NotImplementedError("ROBOT-SPECIFIC: push (58,) joint targets")

# ==========================================================================


def main():
    p = argparse.ArgumentParser(description="dexmate + sharpa vision-only V=3 robot client")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5010)
    p.add_argument("--max-steps", type=int, default=0)
    args = p.parse_args()

    sock = socket.create_connection((args.host, args.port))
    info = _rpc(sock, {"op": "ping"})["info"]
    action_chunk = int(info["action_chunk"])
    lo, hi = info["joint_target_slice"]
    print(f"[client] connected: action_chunk={action_chunk}, "
          f"V={info['n_view_total']}, joint_targets=action[{lo}:{hi}], "
          f"valid_cam={info.get('valid_cam')}")
    _rpc(sock, {"op": "reset"})

    period = 1.0 / CONTROL_HZ
    execution_step = 0
    total_executed = 0
    try:
        while True:
            obs = {
                "images": read_cameras(),          # (3, H, W, 3) uint8
                "state": read_robot_state(),       # (90,) float32 raw
                "prompt": PROMPT,
                "execution_step": execution_step,
            }
            resp = _rpc(sock, {"op": "step", "obs": obs})
            chunk = np.asarray(resp["joint_targets"], dtype=np.float32)
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
        print("\n[client] interrupted.")
    finally:
        sock.close()
        print(f"[client] done -- executed {total_executed} steps.")


if __name__ == "__main__":
    main()
