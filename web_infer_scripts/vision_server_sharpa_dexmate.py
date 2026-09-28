#!/usr/bin/env python3
"""
vision_server_sharpa_dexmate.py
===============================
DexVTAM (VISION-ONLY, V=3) deployment server for the dexmate + sharpa
bimanual robot. Vision-only sibling of
``web_infer_scripts/tactile_server_sharpa_dexmate.py``:

  * same pickle/raw-socket wire protocol (so the existing robot client
    works with only the ``tactile`` obs field removed)
  * V_rgb = 3 cameras (head + L_wrist + R_wrist), no tactile views, no
    hand_pose, no v0d adapter, no projector
  * uses ``pipe.infer(...)`` directly (the non-tactile / single-modality
    code path) instead of an in-house denoise loop -- much simpler than
    the tactile server

Expects a yaml with ``use_tactile_views: false``, ``add_state: true``,
``valid_cam: ['head_img', 'left_wrist_img', 'right_wrist_img']``.

State / action layout (convert_to_vtam.py with wrist, e.g. handover-with-wrist):
    state  (90)  = [L_arm(7), R_arm(7), L_hand(22), R_hand(22),
                    L_cur_pose(16), R_cur_pose(16)]
    action (150) = [L_f6(30), R_f6(30), L_arm_tgt(7), R_arm_tgt(7),
                    L_hand_tgt(22), R_hand_tgt(22),
                    L_arm_tgt_pose(16), R_arm_tgt_pose(16)]
Joint commands the robot executes = ``action[:, 60:118]`` (58-D).
The visual-only model still PREDICTS the 150-D action (f6 + commands +
pose) since training targets are unchanged, but it can't predict f6
well without tactile inputs -- ignore the f6 block in deployment.

I/O contract
------------
  step request obs dict:
    {
      "images":         np.ndarray (V_rgb=3, H, W, 3) uint8 RGB
                         # order matches yaml valid_cam:
                         #   [head_img, left_wrist_img, right_wrist_img]
      "state":          np.ndarray (90,) float32 raw
      "prompt":         str
      "execution_step": int   # steps consumed since last call
    }
  step response:
    {
      "ok": True,
      "action":        np.ndarray (action_chunk, 150) float32 raw
      "joint_targets": np.ndarray (action_chunk, 58)  float32 raw
                                  [L_arm, R_arm, L_hand, R_hand]
    }

Wire: 4-byte big-endian length prefix + pickle. Ops: ping / reset / step /
shutdown. Identical to tactile server modulo the missing `tactile` obs key.

Usage
-----
    python web_infer_scripts/vision_server_sharpa_dexmate.py \\
        --config  <visual-only yaml> \\
        --weight  <step_NNNN dir with diffusion_pytorch_model.safetensors> \\
        --domain-name 0514_pick_cube_handover_100episodes \\
        --host 0.0.0.0 --port 5010

NOTE: --weight is the step_NNNN DIRECTORY (only the safetensors is needed
for visual-only; projector.pt is irrelevant and not required).
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from einops import rearrange  # noqa: E402

from runner.tactile_inferencer import TactileInferencer  # noqa: E402

TARGET_H, TARGET_W = 192, 256


# ============================ wire helpers =================================

def _send(sock: socket.socket, payload: dict) -> None:
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _recv(sock: socket.socket) -> dict:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("peer disconnected")
        header += chunk
    n = struct.unpack(">I", header)[0]
    body = bytearray()
    while len(body) < n:
        chunk = sock.recv(min(65536, n - len(body)))
        if not chunk:
            raise ConnectionError("peer disconnected")
        body.extend(chunk)
    return pickle.loads(bytes(body))


# =================== vision-only V=3 inference engine =====================

class VisionBimanualInference:
    """Loads the DexVTAM Stage-3 visual-only action model and runs one
    receding-horizon inference via ``pipe.infer``.

    Model loading is delegated to ``TactileInferencer.prepare_models``,
    which (with ``use_tactile_views: false``) skips the tactile_vae /
    projector blocks. Inference uses the pipeline's ``infer(...)`` single-
    modality code path (``n_view_visual = n_view, n_view_tactile = 0``).
    """

    def __init__(
        self,
        config_file: str,
        weight_dir: str,
        domain_name: str,
        denoise_steps: int,
        threshold: Optional[int] = None,
        device: str = "cuda:0",
    ) -> None:
        self.domain_name = domain_name

        if not os.path.isdir(weight_dir):
            raise NotADirectoryError(
                f"--weight must be a step_NNNN DIRECTORY holding "
                f"diffusion_pytorch_model.safetensors; got {weight_dir!r}."
            )
        if not os.path.isfile(os.path.join(weight_dir, "diffusion_pytorch_model.safetensors")):
            raise FileNotFoundError(
                f"{weight_dir} has no diffusion_pytorch_model.safetensors."
            )

        tmp_out = tempfile.mkdtemp(prefix="dexvtam_vision_server_")
        self.inf = TactileInferencer(config_file, output_dir=tmp_out, device=device)
        ns = self.inf.args
        if bool(getattr(ns, "use_tactile_views", False)):
            raise RuntimeError(
                "config has use_tactile_views=True; this VISION-ONLY server "
                "requires use_tactile_views=false. Use tactile_server_sharpa_"
                "dexmate.py for tactile yamls."
            )
        if not bool(getattr(ns, "add_state", False)):
            raise RuntimeError(
                "config has add_state=False; this server requires add_state=True "
                "(the [zeros(150), state] history-prefix protocol)."
            )
        if isinstance(ns.diffusion_model, dict):
            ns.diffusion_model["model_path"] = weight_dir
        ns.num_inference_step = int(denoise_steps)
        self.device = torch.device(self.inf.device)
        self.dtype = self.inf.weight_dtype

        self.inf.prepare_models()

        # Build the pipeline (mirrors validate() non-tactile branch).
        self.pipe = self.inf.pipeline_class(
            self.inf.scheduler,
            self.inf.vae,
            self.inf.text_encoder,
            self.inf.tokenizer,
            self.inf.diffusion_model,
        )

        # ---- config-derived constants ------------------------------------
        tcfg = ns.data["train"]
        self.mem_size = int(tcfg["n_previous"])
        self.chunk_raw = int(tcfg["chunk"])
        self.action_chunk = int(tcfg["action_chunk"])
        self.sample_size = tuple(tcfg["sample_size"])
        self.valid_cam = list(tcfg["valid_cam"])
        self.n_view = len(self.valid_cam)
        if str(tcfg.get("action_type", "absolute")) != "absolute":
            raise NotImplementedError("this server only supports action_type='absolute'.")
        self.action_space = str(tcfg.get("action_space", "joint"))
        self.action_in_channels = int(ns.diffusion_model["config"]["action_in_channels"])
        self.num_inference_steps = int(getattr(ns, "num_inference_step", 10))
        self.pixel_wise_timestep = bool(getattr(ns, "pixel_wise_timestep", True))

        self.TEMPORAL_DOWN = int(self.inf.TEMPORAL_DOWN_RATIO)
        self.chunk_lat = (self.chunk_raw - 1) // self.TEMPORAL_DOWN + 1

        # ---- stats (q01/q99 min-max) --------------------------------------
        stats = self.inf.StatisticInfo
        act_key = f"{domain_name}_{self.action_space}"
        sta_key = f"{domain_name}_state_{self.action_space}"
        for k in (act_key, sta_key):
            if k not in stats:
                raise KeyError(
                    f"stats key {k!r} not in stat_file. Available: {list(stats.keys())}"
                )
        self.act_min = np.asarray(stats[act_key]["q01"], dtype=np.float32)[None, :]
        self.act_max = np.asarray(stats[act_key]["q99"], dtype=np.float32)[None, :]
        self.sta_min = np.asarray(stats[sta_key]["q01"], dtype=np.float32)[None, :]
        self.sta_max = np.asarray(stats[sta_key]["q99"], dtype=np.float32)[None, :]
        self.basic_action_dim = int(self.act_min.shape[1])
        self.state_dim = int(self.sta_min.shape[1])

        ti = getattr(ns, "tactile_inference", {}) or {}
        override = ti.get("action_only_dim_override", None)
        self.action_only_dim = int(override) if override is not None \
            else self.basic_action_dim
        assert self.basic_action_dim + self.state_dim == self.action_in_channels, (
            f"action_dim({self.basic_action_dim}) + state_dim({self.state_dim}) != "
            f"action_in_channels({self.action_in_channels}); stats not paired with ckpt."
        )

        self._resize_rgb = transforms.Resize((TARGET_H, TARGET_W), antialias=True)
        self._norm_rgb = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False
        )
        self.threshold = int(threshold) if threshold else self.action_chunk

        print(
            f"[vision_server] domain        = {domain_name}\n"
            f"[vision_server] valid_cam     = {self.valid_cam}  (V_rgb = {self.n_view})\n"
            f"[vision_server] action key    = {act_key}  (dim {self.basic_action_dim})\n"
            f"[vision_server] state  key    = {sta_key}  (dim {self.state_dim})\n"
            f"[vision_server] action_chunk  = {self.action_chunk}, n_prev = {self.mem_size}\n"
            f"[vision_server] action_only   = [0:{self.action_only_dim}]; joint cmds = action[60:118]\n"
            f"[vision_server] denoise steps = {self.num_inference_steps}, threshold = {self.threshold}",
            flush=True,
        )
        self.reset()

    # ---- per-episode rolling state ----------------------------------------
    def reset(self) -> None:
        # each frame: tensor (V_rgb, 3, H, W) in [-1, 1]
        self.frames: List[torch.Tensor] = []
        self.commit: Optional[torch.Tensor] = None
        self.count = 0

    # ---- preprocessing ----------------------------------------------------
    def _preprocess_rgb(self, images_np: np.ndarray) -> torch.Tensor:
        images_np = np.asarray(images_np)
        if images_np.ndim != 4 or images_np.shape[-1] != 3 or images_np.shape[0] != self.n_view:
            raise ValueError(
                f"images must be (V_rgb={self.n_view}, H, W, 3) uint8; got {images_np.shape}. "
                f"Cam order MUST match yaml valid_cam = {self.valid_cam}."
            )
        obs = torch.from_numpy(images_np.copy()).permute(0, 3, 1, 2).float() / 255.0
        obs = self._resize_rgb(obs)
        obs = self._norm_rgb(obs)
        return obs.to(self.device, dtype=self.dtype)

    def _normalize_state(self, state_np: np.ndarray) -> np.ndarray:
        sn = (state_np[None, :] - self.sta_min) / (self.sta_max - self.sta_min + 1e-6)
        return (sn * 2.0 - 1.0).astype(np.float32)

    # ---- one inference step -----------------------------------------------
    @torch.no_grad()
    def step(
        self,
        images_np: np.ndarray,
        state_np: np.ndarray,
        prompt: str,
        execution_step: int,
    ) -> Dict[str, np.ndarray]:
        state_np = np.asarray(state_np, dtype=np.float32).reshape(-1)
        if state_np.shape[0] != self.state_dim:
            raise ValueError(
                f"state has {state_np.shape[0]} dims; model expects {self.state_dim}."
            )

        frame = self._preprocess_rgb(images_np)
        state_norm = self._normalize_state(state_np)

        # ---- rolling keyframe buffer (mirror tactile / vtam servers) -----
        if not self.frames:
            self.frames = [frame] * self.mem_size
            self.count = self.threshold - 1
            self.commit = self.frames[-1]
        else:
            self.count += int(execution_step)
            if self.count >= self.threshold:
                self.count = 0
                self.frames.pop(0)
                self.frames[-1] = self.commit
                self.frames.append(frame)
            else:
                self.frames[-1] = frame
            self.commit = self.frames[-1]

        # ---- build (V_rgb, C, mem, H, W) image stack -> (V_rgb, C, mem, H, W)
        # exactly the layout pipe.infer's non-tactile path expects (B=1).
        obs_tensor = torch.stack(self.frames, dim=1)            # (V_rgb, mem, C, H, W)
        obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w").unsqueeze(0)
        obs_tensor = rearrange(obs_tensor, "b c v t h w -> (b v) c t h w")

        # ---- history_action_state = [zeros(action), state_norm(state)] ----
        state_in = np.concatenate(
            [np.zeros((1, self.basic_action_dim), dtype=np.float32), state_norm], axis=1
        )
        history_action_state = torch.from_numpy(state_in).unsqueeze(1).to(
            device=self.device, dtype=self.dtype
        )

        # ---- pipe.infer (mirrors validate() non-tactile branch) ----------
        preds = self.pipe.infer(
            image=obs_tensor,
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=self.num_inference_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=TARGET_H,
            width=TARGET_W,
            n_view=self.n_view,
            n_view_visual=self.n_view,
            n_view_tactile=0,
            return_action=True,
            n_prev=self.mem_size,
            chunk=self.chunk_lat,
            return_video=False,
            noise_seed=42,
            action_chunk=self.action_chunk,
            history_action_state=history_action_state,
            pixel_wise_timestep=self.pixel_wise_timestep,
            n_chunk=1,
            action_dim=self.action_in_channels,
        )[0]

        # (action_chunk, action_in_channels=240); keep [0:action_only_dim]
        pred = preds["action"][0].detach().float().cpu().numpy()
        act_norm = pred[:, : self.action_only_dim]
        act_raw = (act_norm + 1.0) / 2.0 * (
            self.act_max - self.act_min + 1e-6
        ) + self.act_min
        act_raw = act_raw.astype(np.float32, copy=False)
        joint_targets = act_raw[:, 60:118].copy()
        return {"action": act_raw, "joint_targets": joint_targets}


# ============================== server =====================================

class VisionSharpaDexmateServer:
    def __init__(self, engine: VisionBimanualInference, host: str, port: int) -> None:
        self.engine = engine
        self.host = host
        self.port = port

    def _handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "ping":
            e = self.engine
            return {
                "ok": True,
                "info": {
                    "action_chunk": e.action_chunk,
                    "action_dim": e.basic_action_dim,
                    "action_only_dim": e.action_only_dim,
                    "state_dim": e.state_dim,
                    "n_view_total": e.n_view,
                    "valid_cam": e.valid_cam,
                    "joint_target_slice": [60, 118],
                    "action_type": "absolute",
                    "tactile_expected": False,
                },
            }
        if op == "reset":
            self.engine.reset()
            return {"ok": True}
        if op == "step":
            obs = request["obs"]
            out = self.engine.step(
                images_np=obs["images"],
                state_np=obs["state"],
                prompt=obs.get("prompt", ""),
                execution_step=int(obs.get("execution_step", 1)),
            )
            return {"ok": True, **out}
        if op == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"unknown op: {op}"}

    def serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[vision_server] listening on {self.host}:{self.port} — ready.", flush=True)
        while True:
            conn, addr = srv.accept()
            print(f"[vision_server] connection from {addr}", flush=True)
            try:
                while True:
                    request = _recv(conn)
                    try:
                        response = self._handle(request)
                    except Exception:
                        response = {
                            "ok": False,
                            "error": "server exception",
                            "traceback": traceback.format_exc(),
                        }
                    _send(conn, response)
                    if request.get("op") == "shutdown":
                        print("[vision_server] shutdown requested.", flush=True)
                        conn.close()
                        return
            except (ConnectionError, OSError) as e:
                print(f"[vision_server] {addr} disconnected: {e}", flush=True)
            except Exception:
                print(f"[vision_server] error:\n{traceback.format_exc()}", flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass


def main() -> None:
    p = argparse.ArgumentParser(description="DexVTAM vision-only V=3 server (dexmate + sharpa)")
    p.add_argument("-c", "--config", required=True, help="visual-only yaml (use_tactile_views: false)")
    p.add_argument("-w", "--weight", required=True,
                   help="step_NNNN DIR with diffusion_pytorch_model.safetensors")
    p.add_argument("--domain-name", required=True,
                   help="stats key prefix (e.g. 0514_pick_cube_handover_100episodes)")
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--threshold", type=int, default=0,
                   help="executed steps between keyframe commits (0 => action_chunk)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default=os.environ.get("VTAM_SERVER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("VTAM_SERVER_PORT", "5010")))
    args = p.parse_args()

    print("[vision_server] loading model ...", flush=True)
    engine = VisionBimanualInference(
        config_file=args.config,
        weight_dir=args.weight,
        domain_name=args.domain_name,
        denoise_steps=args.denoise_steps,
        threshold=(args.threshold or None),
        device=args.device,
    )
    print("[vision_server] model loaded.", flush=True)
    VisionSharpaDexmateServer(engine, args.host, args.port).serve()


if __name__ == "__main__":
    main()
