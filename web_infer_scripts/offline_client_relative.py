#!/usr/bin/env python3
"""Offline replay check for a relative_eef_rot6d server (deploy gate F3).

Sibling of offline_client_tactile_sharpa_dexmate.py, which cannot be pointed at a
right-only corpus: that dataset does not merely use narrower vectors, it has NO
LEFT ARM IN IT, so there is nothing to rebuild the raw 90-D bimanual observation
from. This client therefore EMBEDS the dataset's rows back into the raw bimanual
wire format the client hardware would send:

    state   (45,)        -> (90,)          right blocks into the raw right slots
    tactile (1, 5, H, W) -> (2, 5, H, W)   dataset hand into raw hand index 1

The left filler defaults to NaN on purpose. The server gathers only the right
blocks, so NaN can never reach the model -- and if it ever did, every output
would be NaN instead of subtly wrong. That turns the left-poison invariance from
something the unit tests assert into something the live server demonstrates.

For a BIMANUAL corpus the same code is an identity embed: both arms' blocks
cover all 90 raw dims and both tactile hands are filled, so no filler survives.
Every check below reads its widths from the layout, so nothing here is
right-only apart from that one asymmetry in what the filler is for.

Two classes of check, deliberately separated:

  ALGEBRAIC (hard PASS/FAIL, independent of model quality -- valid even on a
  10k-step checkpoint):
    * compose(anchor_we_sent, action[:, rel_poses]) == arm_target_pose
      The server composed with the anchor from OUR observation and wrote it into
      the slice it claims. This is the check that would catch reading the anchor
      from the wrong 16 numbers, or re-anchoring per row.
    * the reverse direction, relativize(anchor, arm_target_pose), split by block:
      translation must match action[:, rel_poses][:3] exactly, but rotation only
      matches UP TO Gram-Schmidt -- the model emits 6 unconstrained numbers and
      rot6d_to_mat orthonormalizes them, so demanding raw equality would fail on
      every checkpoint. The residual is reported as a diagnostic instead.
    * hand_target == action[:, hand]
    * every returned pose is a valid SE(3) with a clean homogeneous row
    * shapes match ping

  ACCURACY (informational): predicted absolute EEF target vs the episode's GT
  absolute arm_target_pose, in mm and degrees, plus hand joint MAE. Expect this
  to be poor on an early checkpoint; it is not a gate.

Usage:
    # server first (see run_server_tactile_relative.sh), then:
    python web_infer_scripts/offline_client_relative.py \
        --config configs/bowl_unstack/stage3_action_expert.yaml \
        --data-root data/datasets_lerobot/20260725_pinch_from_bowl_with_fingers_right_only \
        --episode 90 --port 5008
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
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.tactile_dataset import _deep_stack  # noqa: E402
from data.utils.pose_math import (  # noqa: E402
    compose_mat16_and_relative_rot6d,
    mat16_to_mat44,
    mat_to_rot6d,
    relativize_mat16_to_rot6d,
    rot6d_to_mat,
)
from data.utils.raw_obs_layout import FINGERS, get_raw_obs_layout  # noqa: E402
from data.utils.relative_action import get_arm_layout  # noqa: E402


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

def load_episode(data_root: Path, episode: int, camera_names: list[str]):
    """Load one episode's model-space rows plus the requested camera views."""
    chunk = episode // 1000
    parquet = data_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    df = pd.read_parquet(parquet)
    n = len(df)

    for cam in camera_names:
        if cam not in df.columns:
            raise KeyError(
                f"camera {cam!r} not in the parquet; has "
                f"{[c for c in df.columns if c.endswith('_img')]}"
            )
    images = np.stack([
        np.stack([
            np.array(Image.open(io.BytesIO(df[cam].iloc[i]["bytes"])).convert("RGB"))
            for cam in camera_names
        ])
        for i in range(n)
    ])                                                    # (T, V, H, W, 3)

    tactile = np.stack([_deep_stack(r) for r in df["tactile"].to_list()]).astype(np.uint8)
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

    return images, tactile, states, actions, prompt


# ------------------------- model -> raw embedding --------------------------

def embed_state(state_model, raw_layout, layout, left_fill):
    """Place the model's per-arm blocks back into a full raw observation.

    The inverse of the server's state gather. Model state begins with one 7-D
    arm-joint block per arm, which RelativeArmLayout does not slice per arm; the
    coverage assert below is what keeps that assumption honest.
    """
    raw = np.full((raw_layout.state_dim,), left_fill, dtype=np.float32)
    covered = 0
    for i, arm in enumerate(layout.arms):
        r = raw_layout.arm_position(arm)
        pairs = (
            ((7 * i, 7 * (i + 1)), raw_layout.arm_joints[r]),
            (layout.state_hands[i], raw_layout.hand_joints[r]),
            (layout.state_poses[i], raw_layout.eef_pose[r]),
        )
        for (mlo, mhi), (rlo, rhi) in pairs:
            assert mhi - mlo == rhi - rlo, f"{arm}: model {mlo}:{mhi} vs raw {rlo}:{rhi}"
            raw[rlo:rhi] = state_model[mlo:mhi]
            covered += mhi - mlo
    assert covered == layout.state_dim, \
        f"embedded {covered} of {layout.state_dim} model state dims -- layout changed"
    return raw


def embed_tactile(tac_model, raw_layout, layout):
    """Place the model's tactile hands back onto the raw hand axis (rest zeroed)."""
    assert tac_model.shape[0] == len(layout.arms), tac_model.shape
    raw = np.zeros((raw_layout.tactile_hands,) + tac_model.shape[1:], dtype=np.uint8)
    for i, arm in enumerate(layout.arms):
        raw[raw_layout.tactile_hand_index[raw_layout.arm_position(arm)]] = tac_model[i]
    return raw


def _pose_err(T_a, T_b):
    """(translation metres, rotation degrees) between two 4x4 transforms."""
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    R = T_a[:3, :3] @ T_b[:3, :3].T
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return dt, float(np.degrees(np.arccos(cos)))


# ------------------------------- main --------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", required=True, help="the yaml the server is serving")
    p.add_argument("--data-root", required=True, help="right-only LeRobot dataset root")
    p.add_argument("--episode", type=int, default=90)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5008)
    p.add_argument("--n-chunks", type=int, default=3)
    p.add_argument("--left-fill", choices=("nan", "zero"), default="nan",
                   help="value for the arm(s) the policy does not consume; nan proves "
                        "the server cannot read them (default)")
    p.add_argument("--atol", type=float, default=2e-4,
                   help="tolerance for the algebraic compose round-trip")
    p.add_argument("--shutdown-server", action="store_true",
                   help="tell the server to exit when done (default: leave it up so "
                        "the gate can be re-run without another model load)")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())
    tcfg = cfg["data"]["train"]
    layout = get_arm_layout(str(tcfg.get("arm_layout", "bimanual")))
    raw_layout = get_raw_obs_layout(str(tcfg.get("raw_obs_layout", "bimanual")))
    cams = [str(c) for c in tcfg["valid_cam"]]
    left_fill = float("nan") if args.left_fill == "nan" else 0.0

    images, tactile, states, gt_actions, prompt = load_episode(
        Path(args.data_root), args.episode, cams)
    T = len(images)
    print(f"[client] episode {args.episode}: {T} frames, images {images.shape[1:]} "
          f"({cams}), tactile {tactile.shape[1:]}, state {states.shape[1]}, "
          f"action {gt_actions.shape[1]}")
    print(f"[client] prompt: {prompt!r}")
    assert states.shape[1] == layout.state_dim, \
        f"dataset state {states.shape[1]} != arm_layout {layout.name} {layout.state_dim}"
    assert gt_actions.shape[1] == layout.abs_action_dim, \
        f"dataset action {gt_actions.shape[1]} != {layout.abs_action_dim}"
    assert tactile.shape[1:3] == (len(layout.arms), FINGERS), tactile.shape

    # ---- connect + handshake ----------------------------------------------
    sock = socket.create_connection((args.host, args.port))
    info = _rpc(sock, {"op": "ping"})["info"]
    action_chunk = int(info["action_chunk"])
    arms = [str(a) for a in info["arms"]]
    n_arms = len(arms)
    print(f"[client] server: arm_layout={info.get('arm_layout')} arms={arms}, "
          f"action_chunk={action_chunk}, action_mode={info.get('action_mode')}, "
          f"contract v{info.get('layout_contract_version')} "
          f"sha256={str(info.get('layout_contract_sha256'))[:8]}")
    if str(info.get("action_mode")) != "relative_eef_rot6d":
        raise SystemExit(f"[client] FAIL: server is {info.get('action_mode')!r}, "
                         f"expected relative_eef_rot6d")
    if arms != list(layout.arms):
        raise SystemExit(f"[client] FAIL: server arms {arms} != config layout "
                         f"{list(layout.arms)}")
    if [str(c) for c in info["camera_names"]] != cams:
        raise SystemExit(f"[client] FAIL: server cameras {info['camera_names']} != "
                         f"config valid_cam {cams}")

    # ---- task / prompt: the server's, and it must be the dataset's ----------
    # This replays the very episodes the model trained on, so the registry entry
    # and the dataset's own tasks.jsonl must agree byte-for-byte. Checking it
    # here means F3 also gates prompt provenance, not just the compose math.
    task_id = info.get("task_id")
    if task_id is None:
        print("[client] WARN: server started with --task none; sending the "
              "dataset prompt verbatim and skipping the tactile health block")
    else:
        server_prompt = str(info.get("task_prompt", ""))
        if server_prompt != prompt:
            raise SystemExit(
                f"[client] FAIL: server task {task_id!r} prompt differs from the "
                f"dataset's\n  server : {server_prompt!r}\n  dataset: {prompt!r}")
        print(f"[client] task {task_id}: prompt byte-identical to the dataset "
              f"(sha {str(info.get('task_prompt_sha256'))[:12]})")
    health_block = None
    if info.get("tactile_health_required_in_obs"):
        # Dataset tactile is post-converter, i.e. already carry-forward filled,
        # so "ok" is the truthful claim. The server re-checks it for all-zero
        # fingers, which would mean the cache was built from unfilled frames.
        health_block = {
            "status": "ok",
            "contract_version": info.get("tactile_health_contract_version"),
            "sha256": info.get("tactile_health_sha256"),
            "fill_age": [[0] * 5 for _ in arms],
            "valid_streak": 0,
        }

    n_chunks = min(args.n_chunks, T // action_chunk)
    if n_chunks == 0:
        raise SystemExit(f"[client] episode too short ({T}) for one chunk ({action_chunk})")

    # ---- replay -----------------------------------------------------------
    _rpc(sock, {"op": "reset"})
    ok = True
    acc_rows = []            # (dt_m, dr_deg, hand_mae) per compared row
    for i in range(n_chunks):
        t = i * action_chunk
        state_raw = embed_state(states[t], raw_layout, layout, left_fill)
        obs = {
            "images": images[t],                                  # (V, H, W, 3)
            "tactile": embed_tactile(tactile[t], raw_layout, layout),
            "state": state_raw,
            "prompt": prompt,
            "execution_step": action_chunk,
        }
        if health_block is not None:
            obs["tactile_health"] = health_block
        resp = _rpc(sock, {"op": "step", "obs": obs})
        pose = np.asarray(resp["arm_target_pose"], dtype=np.float64)
        hand = np.asarray(resp["hand_target"], dtype=np.float64)
        act = np.asarray(resp["action"], dtype=np.float64)

        # ---- shapes ------------------------------------------------------
        assert pose.shape == (action_chunk, n_arms, 4, 4), pose.shape
        assert hand.shape == (action_chunk, 22 * n_arms), hand.shape
        assert act.shape == (action_chunk, layout.rel_action_dim), act.shape

        # ---- no NaN escaped from the unused arm --------------------------
        if not np.isfinite(pose).all() or not np.isfinite(hand).all():
            print(f"[client] FAIL chunk {i}: non-finite output -- the server read the "
                  f"arm(s) it should have dropped (left filler was {args.left_fill})")
            ok = False
            break

        # ---- algebraic: compose used OUR anchor, in the slice it claims ---
        for arm_i, arm in enumerate(arms):
            lo, hi = layout.rel_poses[arm_i]
            anchor16 = states[t][slice(*layout.state_poses[arm_i])]  # what we sent
            rel_claim = act[:, lo:hi]                               # (chunk, 9)
            pose16 = pose[:, arm_i].reshape(action_chunk, 16)

            # (a) PRIMARY: recompose the claimed rel9 with the anchor WE sent and
            # compare 4x4s. A wrong anchor (other arm, raw slice instead of model
            # slice, or re-anchored per row) cannot survive this.
            expect = mat16_to_mat44(
                compose_mat16_and_relative_rot6d(anchor16, rel_claim)).astype(np.float64)
            d_pose = float(np.abs(expect - pose[:, arm_i]).max())

            # (b) translation round-trips EXACTLY: T_rel[:3,3] is rel9[:3] verbatim,
            # with no normalization in the path.
            rel_back = relativize_mat16_to_rot6d(anchor16, pose16)   # (chunk, 9)
            d_xyz = float(np.abs(rel_back[:, :3] - rel_claim[:, :3]).max())

            # (c) rotation round-trips only UP TO Gram-Schmidt. The model emits 6
            # unconstrained numbers; rot6d_to_mat orthonormalizes them, so
            # relativizing the composed pose returns the NORMALIZED rot6d. Comparing
            # it to the raw predicted 6-D vector is guaranteed to differ by however
            # non-orthonormal the head's output happens to be -- that is a property
            # of the checkpoint, not a server bug. Compare against the normalized
            # form, and report the gap separately as a checkpoint diagnostic.
            rel_norm = mat_to_rot6d(rot6d_to_mat(rel_claim[:, 3:9])).astype(np.float64)
            d_rot = float(np.abs(rel_back[:, 3:9] - rel_norm).max())
            gs_gap = float(np.abs(rel_claim[:, 3:9] - rel_norm).max())

            bad = max(d_pose, d_xyz, d_rot) > args.atol
            ok = ok and not bad
            print(f"[client] chunk {i} {arm}: compose(anchor, action[{lo}:{hi}]) vs "
                  f"arm_target_pose {d_pose:.2e} | rel xyz {d_xyz:.2e} | rel rot6d "
                  f"(mod Gram-Schmidt) {d_rot:.2e} -> {'FAIL' if bad else 'OK'}"
                  f"   [rot6d non-orthonormality {gs_gap:.3f}]")

            # SE(3) validity of what the client would hand to IK
            for j in (0, action_chunk // 2, action_chunk - 1):
                Tj = pose[j, arm_i]
                R = Tj[:3, :3]
                if not (np.allclose(R @ R.T, np.eye(3), atol=1e-3)
                        and abs(np.linalg.det(R) - 1.0) < 1e-3
                        and np.allclose(Tj[3], (0, 0, 0, 1), atol=1e-4)):
                    print(f"[client] FAIL chunk {i} {arm} row {j}: not a valid SE(3)")
                    ok = False

        # ---- algebraic: hand_target is the action's hand block -----------
        dev_hand = float(np.abs(hand - act[:, slice(*layout.hand)]).max())
        if dev_hand != 0.0:
            print(f"[client] FAIL chunk {i}: hand_target != action[{layout.hand}] "
                  f"(max dev {dev_hand:.2e})")
            ok = False

        # ---- accuracy vs GT absolute targets (informational) -------------
        for j in range(action_chunk):
            k = t + j
            if k >= T:
                break
            for arm_i in range(n_arms):
                gt16 = gt_actions[k][slice(*layout.action_poses[arm_i])]
                dt, dr = _pose_err(pose[j, arm_i], mat16_to_mat44(gt16.astype(np.float64)))
                gt_hand = gt_actions[k][slice(*layout.hand)]
                mae = float(np.abs(hand[j] - gt_hand).mean())
                acc_rows.append((dt, dr, mae))

    # Default is to LEAVE THE SERVER UP: this gate gets re-run while iterating on
    # the checks, and a restart costs a full model load. The sibling offline client
    # shuts down on exit; opt into that explicitly.
    if args.shutdown_server:
        _send(sock, {"op": "shutdown"})
        print("[client] sent shutdown; the server will exit")
    sock.close()

    if acc_rows:
        a = np.asarray(acc_rows)
        print(f"\n[client] accuracy vs GT over {len(a)} rows (INFORMATIONAL -- an early "
              f"checkpoint is expected to be poor; this is not a gate):")
        print(f"    EEF translation : mean {a[:, 0].mean() * 1000:8.2f} mm   "
              f"median {np.median(a[:, 0]) * 1000:8.2f} mm   max {a[:, 0].max() * 1000:8.2f} mm")
        print(f"    EEF rotation    : mean {a[:, 1].mean():8.2f} deg  "
              f"median {np.median(a[:, 1]):8.2f} deg  max {a[:, 1].max():8.2f} deg")
        print(f"    hand joint MAE  : mean {a[:, 2].mean():8.4f} rad")

    print(f"\n[client] GATE F3: {'PASS' if ok else 'FAIL'} "
          f"(shapes, no-leak, compose round-trip, SE(3) validity, hand passthrough)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
