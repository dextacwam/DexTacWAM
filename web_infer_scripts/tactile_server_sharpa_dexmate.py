#!/usr/bin/env python3
"""
tactile_server_sharpa_dexmate.py
================================
DexVTAM (TACTILE-aware) deployment server for the dexmate + sharpa bimanual
robot. This is the Stage-3 action_full sibling of VTAM's
``web_infer_scripts/vtam_server_sharpa_dexmate.py``: same pickle/raw-socket
wire protocol, same receding-horizon rolling-buffer control, but the inference
engine injects TACTILE views (+ v0d hand-pose) into the DiT exactly the way the
offline open-loop eval does (``runner/tactile_inferencer.py::_validate_with_tactile``).

Why a new file instead of reusing vtam_server: the DexVTAM action model is a
tactile world-model + action-expert. Its ``CustomPipeline.infer`` has NO
tactile_* args, so the offline eval bypasses the pipeline and runs an in-house
denoise loop that (a) per-frame VAE-encodes the visual mem frames, (b) encodes
per-hand tactile through the frozen v0d adapter (FingerSetTransformer + pose
injection + TimeSformer, projector BYPASSED), (c) cats the tactile rows onto the
visual view stack so ``n_view`` bumps 1 -> 3, (d) denoises the action tokens with
the WM video-states cached after step 0. This server reuses that exact code path
by composing a ``TactileInferencer`` for model loading + ``_encode_tactile_split``.

Two layouts, always distinct
---------------------------
The robot ALWAYS reports both arms, whatever the policy consumes, so the raw
wire format and the model's index layout are separate objects (yaml
``raw_obs_layout`` and ``arm_layout``; both explicit, never width-inferred):

    raw_obs_layout=bimanual  state (90) = [L_arm(7), R_arm(7),
                                           L_hand(22), R_hand(22),
                                           L_cur_pose(16), R_cur_pose(16)]
                             tactile (V_hand=2, F=5, H, W)

    arm_layout=bimanual      state 90, abs action 150, rel action 136
    arm_layout=right_only    state 45, abs action  75, rel action  68
                             tactile (V_hand=1, F=5, H, W)

For ``right_only`` the server GATHERS the right arm's blocks out of the raw
90-D vector (raw arm[7:14], hand[36:58], pose[74:90]) into the model's 45-D
order, and selects raw tactile hand axis 1. Note the model's right-arm pose
slice is [29:45], which in the RAW vector is part of the right hand's joints --
applying a model slice to a raw vector reads plausible garbage rather than
failing, which is exactly why the two layouts never share a type.

    bimanual abs action (150) = [L_f6(30), R_f6(30),          predicted tactile force
                                 L_arm_tgt(7), R_arm_tgt(7),   target arm joints
                                 L_hand_tgt(22), R_hand_tgt(22),
                                 L_arm_tgt_pose(16), R_arm_tgt_pose(16)]

The model internally operates on an ``[action, state]`` tensor (add_state=True;
240 = 150+90 bimanual absolute, 113 = 68+45 right-only relative) and its action
head emits that same width, of which we keep the leading action block. For the
absolute mode the ROBOT JOINT COMMANDS are the ``arm_joints..hand`` span
(bimanual [60:118] = 58 joint targets), taken from the layout rather than
written out here. f6 is predicted tactile force, not a command.

I/O contract
------------
The client always sends the FULL RAW bimanual observation; the server does ALL of:
  * raw -> model trim/gather        (state gather, tactile hand select)
  * state min-max normalization     (q01/q99 -> [-1, 1], mirrors training)
  * tactile resize + [-1, 1] normalize
  * hand_pose extraction from raw state + Stage-1 pose-stats normalization
  * the [zeros(action), state_norm] history-prefix
  * action de-normalization         (inverse min-max -> raw targets)
  * relative -> ABSOLUTE EEF compose

Trimming lives on the server because the client must stay layout-agnostic: it
sends what the hardware reports and reads the shapes it should expect back from
``ping``, so a right-only checkpoint needs no client reconfiguration.

Tactile health lives on the CLIENT, for the opposite reason
-----------------------------------------------------------
The converter's dropout windows are frame counts at the 30 Hz recording rate
(W=5 is 167 ms). This server is only called once per action chunk -- with
num_action_execute=54 that is every 1.8 s -- so a fill decided here would carry
a frame up to nine seconds old and commit it to the keyframe buffer. The filter
therefore runs in the client's 30 Hz fetch thread, which sends an observation
only when it is usable, and this server:

  * NEVER instantiates ``OnlineTactileHealthFilter``;
  * verifies the claim statelessly (contract sha, status, and that no finger
    actually arrived all-zero) and refuses on disagreement;
  * logs what the client reported alongside what arrived.

Refusal is a backstop for a contract bug, not a control path: a correct client
holds locally and never asks. Health verification is enabled by ``--task``
(a hardware rollout) and off for ``--task none`` (offline tooling replaying
dataset frames that are already filled).

  step request obs dict (identical for every arm_layout):
    {
      "images":         np.ndarray (V_rgb, H, W, 3) uint8 RGB
                        # order MUST match ping's camera_names (yaml valid_cam)
      "tactile":        np.ndarray (V_hand=2, F=5, Ht, Wt) uint8 # per-hand,per-finger gray
                        # order: (left,right) x (thumb,index,middle,ring,pinky)
                        # ALREADY carry-forward filled by the client's 30 Hz filter
      "tactile_health": {"status": "ok"|"degraded_safe",  # required unless --task none
                         "contract_version": int, "sha256": str,
                         "fill_age": [[int]*5]*V_hand, "valid_streak": int}
      "state":          np.ndarray (90,) float32  RAW joint positions + cur poses
      "prompt":         str   # ECHO of ping's task_prompt, not an input: the
      "prompt_sha256":  str   # server conditions on its own registry string and
                              # REFUSES the step if either echo disagrees
      "execution_step": int   # steps consumed from the previous chunk
    }
  step response (action_type='absolute'):
    {
      "ok": True,
      "action_mode":   "absolute",
      "action":        np.ndarray (action_chunk, abs_action_dim) float32  RAW
      "joint_targets": np.ndarray (action_chunk, 29*n_arms) float32 RAW
                       # [arm7 per arm..., hand22 per arm...], arms in ping's arm order
    }
  step response (action_type='relative_eef_rot6d'):
    The server denorms the relative action and composes each arm's rel9
    ([xyz(3), rot6d(6)]) with that arm's EEF anchor from the RAW observed state
    that produced the chunk, T_abs = T_anchor @ T_rel, so the client just runs IK
    on absolute poses (no client-side compose / re-anchor).
    {
      "ok": True,
      "action_mode":     "relative_eef_rot6d",
      "action":          np.ndarray (action_chunk, rel_action_dim) float32 RAW (logging)
      "arm_target_pose": np.ndarray (action_chunk, n_arms, 4, 4) float32
                         # ABSOLUTE homogeneous; row i is ping's arms[i]
                         # bimanual -> (chunk,2,4,4) [L,R]; right_only -> (chunk,1,4,4) [R]
      "hand_target":     np.ndarray (action_chunk, 22*n_arms) float32 RAW
    }

  The anchor is held FIXED for every row of the chunk (training relativizes a
  whole window against the single row n_previous-1), so the client must not
  re-anchor per executed step.

Wire format: 4-byte big-endian length prefix + pickle payload.
Ops: ping / reset / step / shutdown.   (identical to vtam_server_sharpa_dexmate)

Usage
-----
    python web_infer_scripts/tactile_server_sharpa_dexmate.py \\
        --config  configs/cube_place/stage3_action_expert.yaml \\
        --weight  outputs/stage3_action_full_right_hand_pick_cube_correctaction_cubewm_bypass_shared_rmsnorm_long50000/<TS>/step_20000 \\
        --domain-name lerobot_0501_pick_cube_sdh_100_episodes_no_wrist_vtam_correctaction \\
        --task bowl \\
        --host 0.0.0.0 --port 5008

``--task`` is required and selects the served prompt from ``data/utils/task_registry``;
it is cross-checked against the config's own domain, so a task/checkpoint mismatch
is a startup error. Offline tooling passes ``--task none``.

NOTE: --weight is the step_NNNN DIRECTORY (must hold both
diffusion_pytorch_model.safetensors AND projector.pt), not a single file.
"""

from __future__ import annotations

import argparse
import json
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

# --- locate DexVTAM repo root (this file lives in DexVTAM/web_infer_scripts/) -
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from einops import rearrange  # noqa: E402

from runner.tactile_inferencer import TactileInferencer  # noqa: E402
from utils.data_utils import (  # noqa: E402
    get_text_conditions,
    gen_noise_from_condition_frame_latent,
    randn_tensor,
    _normalize_latents,
)
from models.pipeline.custom_pipeline import calculate_shift, retrieve_timesteps  # noqa: E402
# Shared, unit-tested SE(3)/rot6d helpers -- SAME math the dataset + stats pass
# use, so the deploy-side relative->absolute compose can never drift from training.
from data.utils.pose_math import (  # noqa: E402
    compose_mat16_and_relative_rot6d,
    mat16_to_mat44,
)
# MODEL-space index layout (what the checkpoint was trained on) and the RAW
# client-observation layout (what the robot reports). These are two different
# coordinate systems over the same physical arms -- see raw_obs_layout's header.
# Both are selected by EXPLICIT yaml name; neither is inferred from a width.
from data.utils.relative_action import assert_layout_dims, get_arm_layout  # noqa: E402
# Only CONSTANTS and the sha are imported: the fill state machine runs in the
# client's 30 Hz fetch thread, and this server must never instantiate one. See
# "Tactile health" in the header.
from data.utils.tactile_health import (  # noqa: E402
    CONTRACT_VERSION as TACTILE_HEALTH_CONTRACT_VERSION,
    USABLE_STATUSES,
    module_sha256 as tactile_health_sha256,
)
# The served prompt is provenance, not a client-supplied string: named by
# --task, cross-checked against this checkpoint's own domain, published in ping.
from data.utils.task_registry import (  # noqa: E402
    NO_TASK,
    assert_task_matches_checkpoint,
    get_task,
    prompt_sha256,
)
from data.utils.raw_obs_layout import (  # noqa: E402
    ARM7,
    FINGERS,
    HAND22,
    build_blank_window_table,
    build_state_gather_index,
    build_tactile_hand_index,
    get_raw_obs_layout,
)

# Visual + tactile target resolution must match training (convert_to_vtam_no_wrist.py).
TARGET_H, TARGET_W = 192, 256


class TactileHealthContractError(ValueError):
    """The client's health claim disagrees with the frames it sent, or with this
    server's copy of the filter. Distinct from a shape/dtype error so the two
    never get reported as the same failure."""


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


# ====================== inference engine (tactile) =========================

class TactileBimanualInference:
    """Loads the DexVTAM Stage-3 action model (+ frozen v0d tactile adapter)
    and runs one receding-horizon inference step with TACTILE injection.

    Model loading is delegated to ``TactileInferencer.prepare_models`` so the
    DiT / VAE / tactile_vae / projector / schedulers are byte-identical to the
    offline eval. The per-chunk denoise math mirrors
    ``TactileInferencer._validate_with_tactile`` (diagnostics stripped), but on
    LIVE observations + a rolling keyframe buffer instead of a dataset batch.
    """

    def __init__(
        self,
        config_file: str,
        weight_dir: str,
        domain_name: str,
        denoise_steps: int,
        task_id: str,
        threshold: Optional[int] = None,
        device: str = "cuda:0",
    ) -> None:
        self.domain_name = domain_name
        # Resolved before anything expensive: an unknown task is a typo, and a
        # typo must not cost a checkpoint load. The domain/layout cross-check
        # happens once the yaml is parsed, still before prepare_models().
        self.task_id = str(task_id)
        self.task = get_task(self.task_id)

        if not os.path.isdir(weight_dir):
            raise NotADirectoryError(
                f"--weight must be a step_NNNN DIRECTORY (holding "
                f"diffusion_pytorch_model.safetensors + projector.pt); got "
                f"{weight_dir!r}."
            )
        for need in ("diffusion_pytorch_model.safetensors", "projector.pt"):
            if not os.path.isfile(os.path.join(weight_dir, need)):
                raise FileNotFoundError(
                    f"{weight_dir} is missing {need}; not a valid Stage-3 "
                    f"tactile action checkpoint dir."
                )

        # ---- model loading via TactileInferencer (byte-identical to eval) --
        tmp_out = tempfile.mkdtemp(prefix="dexvtam_tactile_server_")
        self.inf = TactileInferencer(
            config_file, output_dir=tmp_out, device=device,
        )
        ns = self.inf.args
        # Two yamls reach the action expert with tactile, and this server serves
        # both. `use_tactile_views` routes tactile through the DiT as extra views;
        # `tactile_late_fuse` (the tactile-world-model ablation) skips the DiT and
        # feeds the action expert's tactile cross-attention directly. What the
        # server cares about is neither flag on its own but whether the policy
        # consumes tactile at all -- everything downstream here (the health
        # contract, the blank-fill windows, the `tac` field the client must send)
        # is identical under both. Refusing on `use_tactile_views` alone would
        # reject an ablation checkpoint that needs tactile just as much.
        self.use_tactile_views = bool(getattr(ns, "use_tactile_views", False))
        self.tactile_late_fuse = bool(getattr(ns, "tactile_late_fuse", False))
        if self.use_tactile_views and self.tactile_late_fuse:
            raise RuntimeError(
                "config sets BOTH use_tactile_views and tactile_late_fuse. They "
                "are mutually exclusive routings of the same tokens; the trainer "
                "refuses this pairing, so no checkpoint can have been trained "
                "with it."
            )
        if not (self.use_tactile_views or self.tactile_late_fuse):
            raise RuntimeError(
                "config has use_tactile_views=False and tactile_late_fuse=False; "
                "this server requires a TACTILE Stage-3 action_full yaml (either "
                "use_tactile_views=true, or tactile_late_fuse=true for the "
                "tactile-WM ablation). A vision-only policy has no tactile path "
                "for the `tac` field this server demands from every client step."
            )
        if not bool(getattr(ns, "add_state", False)):
            raise RuntimeError(
                "config has add_state=False; this server requires add_state=True "
                "(the [zeros(action), state] history-prefix protocol)."
            )

        # ---- layouts: model-space (trained) vs raw wire (reported) ---------
        # Resolved BEFORE prepare_models() so a layout / contract mistake costs a
        # second instead of a full checkpoint load. Defaults keep every existing
        # bimanual yaml working unchanged; a right-only checkpoint says so in its
        # own yaml, because no observed width can distinguish the two.
        tcfg = ns.data["train"]
        self.layout = get_arm_layout(str(tcfg.get("arm_layout", "bimanual")))
        self.raw_layout = get_raw_obs_layout(str(tcfg.get("raw_obs_layout", "bimanual")))
        self.arms = list(self.layout.arms)
        self.n_arms = len(self.arms)
        self.raw_state_dim = self.raw_layout.state_dim

        # ---- task: prompt provenance + checkpoint pairing ------------------
        # domain_name comes from the served yaml's domains[0] (the launcher reads
        # it rather than accepting a hand-typed value), so this catches "bowl
        # weights launched with --task tong" -- which the client-side hash echo
        # structurally cannot, since both ends would agree on the wrong prompt.
        if self.task is not None:
            assert_task_matches_checkpoint(
                self.task, domain_name, self.layout.name
            )
            self.task_prompt = self.task.prompt
            self.task_prompt_sha256 = self.task.prompt_sha256
        else:
            # Offline tooling only (parity replay, byte-parity capture): the
            # pre-registry path, prompt taken verbatim from the observation.
            # ping publishes task_id=None and the hardware client refuses it.
            self.task_prompt = None
            self.task_prompt_sha256 = None
        # Precomputed raw->model maps; building them here means an arm the client
        # never sends aborts at load, not on the first live observation.
        self._state_gather = build_state_gather_index(self.raw_layout, self.layout)
        self._tactile_hands = build_tactile_hand_index(self.raw_layout, self.layout)
        # self.task carries the window policy its corpus was converted with;
        # None only for --task none, which falls back to the legacy constants so
        # the offline byte-parity captures stay comparable.
        self.blank_windows = build_blank_window_table(
            self.raw_layout, self.layout, self.task)
        # Published in ping so the client builds its filter from THIS table
        # rather than a second copy of the numbers.
        self.health_log_dir = Path(
            os.environ.get("DEXVTAM_HEALTH_LOG_DIR", "/tmp/dexvtam_health_logs"))
        self._health_fh = None
        self._episode_index = -1
        self._health_step = -1
        self._health_sha = tactile_health_sha256()
        self._health_counts: Dict[str, int] = {}
        self._health_events: List[dict] = []
        self._prev_status: Optional[str] = None
        # Health gating rides on the task registry: a registered task is a
        # hardware rollout and must present a verified filter, while --task none
        # is offline tooling replaying already-filled dataset frames.
        self.tactile_health_mode = "client_30hz" if self.task is not None else "disabled"

        # The dataset converter, the trainer and this server slice these same
        # vectors from separate checkouts, so prove the layout tables are the
        # shared declaration rather than three lookalikes that have drifted.
        from data.utils.layout_contract import (
            assert_all_layouts_match_contract, load_contract,
        )
        from data.utils.relative_action import LAYOUTS as _MODEL_LAYOUTS
        self._contract = load_contract()
        self._contract_version = assert_all_layouts_match_contract(
            _MODEL_LAYOUTS, self._contract
        )

        # main.py rewrites diffusion_model.model_path to the step_NNNN ckpt dir
        # before prepare_models(); replicate so the DiT safetensors AND
        # projector.pt both load from the deployed checkpoint.
        if isinstance(ns.diffusion_model, dict):
            ns.diffusion_model["model_path"] = weight_dir
        ns.num_inference_step = int(denoise_steps)
        self.device = torch.device(self.inf.device)
        self.dtype = self.inf.weight_dtype

        self.inf.prepare_models()

        # ---- pull config-derived constants (tcfg read above) --------------
        self.mem_size = int(tcfg["n_previous"])
        self.chunk_raw = int(tcfg["chunk"])
        self.action_chunk = int(tcfg["action_chunk"])
        sample_h, sample_w = tcfg["sample_size"]
        self.camera_names = [str(c) for c in tcfg["valid_cam"]]
        self.n_view = len(self.camera_names)                 # visual views
        self.action_mode = str(tcfg.get("action_type", "absolute"))
        if self.action_mode not in ("absolute", "relative_eef_rot6d"):
            raise NotImplementedError(
                f"unsupported action_type={self.action_mode!r}; this server "
                "supports 'absolute' and 'relative_eef_rot6d'."
            )
        self.is_relative = self.action_mode == "relative_eef_rot6d"
        self.action_space = str(tcfg.get("action_space", "joint"))

        self.SPATIAL_DOWN = int(self.inf.SPATIAL_DOWN_RATIO)
        self.TEMPORAL_DOWN = int(self.inf.TEMPORAL_DOWN_RATIO)
        self.action_in_channels = int(ns.diffusion_model["config"]["action_in_channels"])
        self.num_inference_steps = int(getattr(ns, "num_inference_step", 10))
        self.noise_seed = int(getattr(ns, "seed", 42))
        self.pixel_wise_timestep = bool(getattr(ns, "pixel_wise_timestep", True))

        # latent grid (mirror _validate_with_tactile lines 829-833).
        self.latent_frames = self.chunk_raw // self.TEMPORAL_DOWN + 1 + self.mem_size
        self.latent_height = sample_h // self.SPATIAL_DOWN
        self.latent_width = sample_w // self.SPATIAL_DOWN

        # ---- action_only_dim (yaml override or action-minus-state) --------
        ti = getattr(ns, "tactile_inference", {}) or {}
        override = ti.get("action_only_dim_override", None)

        # ---- stats: min-max q01/q99 for state-norm + action de-norm -------
        # absolute      -> {domain}_{space}        (150-D) + {domain}_state_{space} (90)
        # relative_eef  -> {domain}_relative_{space} (136-D) + {domain}_state_{space} (90)
        # (the relative block is the ONLY thing that changes width; state stays 90-D
        #  absolute and is still the state-echo target -- see relative_eef_rot6d_design.md)
        stats = self.inf.StatisticInfo
        act_key = (f"{domain_name}_relative_{self.action_space}" if self.is_relative
                   else f"{domain_name}_{self.action_space}")     # *_relative_eef(136) / *_joint(150)
        sta_key = f"{domain_name}_state_{self.action_space}"      # *_state_* (90)
        for k in (act_key, sta_key):
            if k not in stats:
                raise KeyError(
                    f"stats key {k!r} not in stat_file. Available: "
                    f"{list(stats.keys())}"
                )
        self.act_min = np.asarray(stats[act_key]["q01"], dtype=np.float32)[None, :]
        self.act_max = np.asarray(stats[act_key]["q99"], dtype=np.float32)[None, :]
        self.sta_min = np.asarray(stats[sta_key]["q01"], dtype=np.float32)[None, :]
        self.sta_max = np.asarray(stats[sta_key]["q99"], dtype=np.float32)[None, :]
        self.basic_action_dim = int(self.act_min.shape[1])       # 150 / 136 / 75 / 68
        self.state_dim = int(self.sta_min.shape[1])              # 90 / 45  (MODEL space)
        self.action_only_dim = int(override) if override is not None \
            else self.basic_action_dim
        assert self.basic_action_dim + self.state_dim == self.action_in_channels, (
            f"action_dim({self.basic_action_dim}) + state_dim({self.state_dim}) "
            f"!= action_in_channels({self.action_in_channels}); stats not paired "
            f"with this checkpoint."
        )
        # The yaml NAMES the layout; this proves the stats file actually is that
        # layout. Without it a bimanual stats file under arm_layout=right_only
        # would only surface as a confusing reshape much later.
        assert_layout_dims(
            self.layout,
            state_dim=self.state_dim,
            **({"rel_action_dim": self.basic_action_dim} if self.is_relative
               else {"abs_action_dim": self.basic_action_dim}),
            where=f"stat_file[{act_key}, {sta_key}]",
        )

        # ---- pose stats (Stage-1 v0d hand-pose normalization) -------------
        # hand_pose is extracted from RAW state, then (pose - mean) / std with
        # the SAME pose_stats v0d was trained on (yaml data.train.pose_stats_path).
        pose_stats_path = tcfg.get("pose_stats_path", None)
        self._pose_mean = None
        self._pose_std = None
        if pose_stats_path:
            if not os.path.isabs(pose_stats_path):
                pose_stats_path = str(REPO_ROOT / pose_stats_path)
            with open(pose_stats_path, "r") as f:
                ps = json.load(f)
            self._pose_mean = np.asarray(ps["mean"], dtype=np.float32)  # (22,)
            self._pose_std = np.asarray(ps["std"], dtype=np.float32)
            assert self._pose_mean.shape == (22,) and self._pose_std.shape == (22,)
        self.v0d_pose_on = bool(
            self.inf.tactile_vae is not None
            and getattr(self.inf.tactile_vae, "use_pose_injection", False)
        )
        if self.v0d_pose_on and self._pose_mean is None:
            raise RuntimeError(
                "tactile adapter has use_pose_injection=True but yaml has no "
                "pose_stats_path; refusing to feed un-normalized pose to v0d."
            )

        # ---- image transforms (mirror dataset normalize_video) ------------
        self._resize_rgb = transforms.Resize((TARGET_H, TARGET_W), antialias=True)
        self._norm_rgb = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False
        )

        # rolling-buffer threshold: commit a new keyframe every `threshold`
        # executed steps (mirror vtam_server / MVActor; default = action_chunk).
        self.threshold = int(threshold) if threshold else self.action_chunk

        # Startup manifest: everything needed to tell, from the log alone, which
        # layout/stats/checkpoint combination actually got served.
        L = self.layout
        g = self._state_gather
        exec_desc = (
            f"absolute joint cmds action[{L.arm_joints[0]}:{L.hand[1]}]"
            if not self.is_relative else
            f"compose(rel9)->ABS EEF 4x4 (arm, client IK) + hand action"
            f"[{L.hand[0]}:{L.hand[1]}]"
        )
        task_desc = (
            f"{self.task_id}  prompt_sha256={self.task_prompt_sha256[:12]}\n"
            f"[tactile_server] prompt        = {self.task_prompt!r}"
            if self.task is not None else
            f"{NO_TASK}  (OFFLINE TOOLING: prompt taken verbatim from the "
            f"observation, health gating off; the hardware client refuses this)"
        )
        print(
            f"[tactile_server] task          = {task_desc}\n"
            f"[tactile_server] domain        = {domain_name}\n"
            f"[tactile_server] arm_layout    = {L.name} arms={self.arms} "
            f"(contract v{self._contract_version} "
            f"sha256={self._contract['_sha256'][:8]})\n"
            f"[tactile_server] raw_obs_layout= {self.raw_layout.name} "
            f"state {self.raw_state_dim} -> model {self.state_dim}"
            + ("  (identity)\n" if np.array_equal(g, np.arange(g.size))
               else f"  gather[0:4]={g[:4].tolist()}...[-4:]={g[-4:].tolist()}\n")
            + f"[tactile_server] action key    = {act_key}  (dim {self.basic_action_dim})\n"
            f"[tactile_server] state  key    = {sta_key}  (dim {self.state_dim})\n"
            f"[tactile_server] action_chunk  = {self.action_chunk}, n_prev = "
            f"{self.mem_size} (anchor row {self.mem_size - 1})\n"
            f"[tactile_server] cameras       = {self.camera_names} "
            f"(client MUST send images in this order)\n"
            f"[tactile_server] n_view total  = {self.n_view + self.n_arms} "
            f"(visual={self.n_view}, tactile={self.n_arms})\n"
            f"[tactile_server] tactile hands = raw {self._tactile_hands} -> "
            f"{self.arms}; blank windows "
            f"{ {f'{a}{f}': w for (a, f), w in self.blank_windows.items()} }\n"
            f"[tactile_server] tactile health= {self.tactile_health_mode} "
            f"(filter runs CLIENT-side at 30 Hz; contract "
            f"v{TACTILE_HEALTH_CONTRACT_VERSION} sha {self._health_sha[:8]}, "
            f"this server only verifies)\n"
            f"[tactile_server] action_mode   = {self.action_mode}\n"
            f"[tactile_server] action_only   = [0:{self.action_only_dim}]; exec = {exec_desc}\n"
            f"[tactile_server] denoise steps = {self.num_inference_steps}, threshold = {self.threshold}\n"
            f"[tactile_server] v0d pose inj  = {self.v0d_pose_on}",
            flush=True,
        )
        self.reset()

    # ---- per-episode rolling state ----------------------------------------
    def reset(self) -> None:
        # each frame: {"rgb":(V_rgb,3,H,W), "tac":(V_hand,F,H,W), "pose":(V_hand,22)}
        self.frames: List[dict] = []
        self.commit: Optional[dict] = None
        self.count = 0
        self._roll_health_log()

    # ---- tactile health logging (JSONL, one file per episode) --------------
    def _roll_health_log(self) -> None:
        if self._health_fh is not None:
            self._write_health_summary()
            self._health_fh.close()
        self._episode_index += 1
        self._health_step = -1
        self._health_counts: Dict[str, int] = {}
        self._health_events: List[dict] = []
        self._prev_status: Optional[str] = None
        try:
            self.health_log_dir.mkdir(parents=True, exist_ok=True)
            path = self.health_log_dir / f"episode_{self._episode_index:04d}.jsonl"
            self._health_fh = path.open("w", buffering=1)
            print(f"[tactile_server] health log -> {path}", flush=True)
        except OSError as exc:
            # Never let logging take the policy down.
            print(f"[tactile_server] WARN: health log disabled ({exc})", flush=True)
            self._health_fh = None

    def _log_client_health(
        self, health: Optional[dict], blank: np.ndarray, execution_step: int
    ) -> None:
        """Record what the CLIENT reported, next to what actually arrived.

        The server no longer decides tactile health, but it is the process with
        a log file, and a status that disagrees with the frames on the wire is
        the one thing worth having both halves of on disk.
        """
        status = str((health or {}).get("status", "unreported"))
        self._health_counts[status] = self._health_counts.get(status, 0) + 1
        transition = self._prev_status is not None and status != self._prev_status
        if transition:
            self._health_events.append(
                {"step": self._health_step, "from": self._prev_status, "to": status}
            )
            print(f"[tactile_server] client tactile health {self._prev_status} -> "
                  f"{status}", flush=True)
        self._prev_status = status
        if self._health_fh is None:
            return
        rec = {
            "episode": self._episode_index,
            "step": self._health_step,
            "client_status": status,
            "execution_step": int(execution_step),
            "blank_on_arrival": [[bool(v) for v in row] for row in blank],
            "client_fill_age": (health or {}).get("fill_age"),
            "client_valid_streak": (health or {}).get("valid_streak"),
        }
        if transition:
            rec["event"] = "status_change"
        try:
            self._health_fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _write_health_summary(self) -> None:
        if self._health_fh is None:
            return
        try:
            self._health_fh.write(json.dumps({
                "event": "episode_summary",
                "episode": self._episode_index,
                "steps": self._health_step + 1,
                "client_status_counts": self._health_counts,
                "transitions": self._health_events,
                "hands": list(self.arms),
                "windows": {f"{a}{f}": w for (a, f), w in self.blank_windows.items()},
                "tactile_health_mode": self.tactile_health_mode,
            }) + "\n")
        except OSError:
            pass

    # ---- stateless verification of the client's health claim ---------------
    def _verify_client_health(
        self, health: Optional[dict], tac_model: np.ndarray
    ) -> np.ndarray:
        """Check the client's claim against the frames that arrived. Returns the
        per-(hand,finger) blank mask.

        This is a BACKSTOP, not the control path. A correct client never sends an
        unusable observation, because the decision is made in its 30 Hz thread
        where the window arithmetic is meaningful; anything reaching here that
        fails these checks means the two ends disagree about the contract, and
        guessing which one is right is how a stale frame gets executed.

        Deliberately stateless. A per-request filter here would run at chunk
        cadence (~0.56 Hz), where a W=5 window spans nine seconds instead of
        167 ms -- the bug this architecture exists to remove.
        """
        blank = ~np.asarray(tac_model).any(axis=(-1, -2))       # (n_arms, F)
        if self.tactile_health_mode == "disabled":
            return blank

        if health is None:
            raise TactileHealthContractError(
                "obs has no 'tactile_health' block. This server serves task "
                f"{self.task_id!r} on hardware and requires the client-side 30 Hz "
                "filter; an unfiltered client would feed blank tactile straight "
                "to a model that never saw one in training."
            )
        version = health.get("contract_version")
        sha = health.get("sha256")
        if version != TACTILE_HEALTH_CONTRACT_VERSION or sha != self._health_sha:
            raise TactileHealthContractError(
                f"tactile_health contract mismatch: client v{version} "
                f"sha {str(sha)[:12]}, server v{TACTILE_HEALTH_CONTRACT_VERSION} "
                f"sha {self._health_sha[:12]}. The two ends would fill "
                f"differently; re-vendor the client's copy."
            )
        status = str(health.get("status", ""))
        if status not in USABLE_STATUSES:
            raise TactileHealthContractError(
                f"client sent an observation it had already judged {status!r}. "
                f"Usable statuses are {list(USABLE_STATUSES)}; the client must "
                f"hold locally instead of asking the server to refuse."
            )
        if blank.any():
            bad = [f"{self.arms[h]}{f}" for h, f in zip(*np.nonzero(blank))]
            raise TactileHealthContractError(
                f"client reported {status!r} but fingers {bad} arrived all-zero. "
                f"Either the filter ran on a different buffer than the one sent, "
                f"or the fill was dropped between them."
            )
        return blank

    # ---- preprocessing (RAW client values -> model-space tensors) ---------
    def _preprocess_rgb(self, images_np: np.ndarray) -> torch.Tensor:
        images_np = np.asarray(images_np)
        if images_np.ndim != 4 or images_np.shape[-1] != 3:
            raise ValueError(f"images must be (V_rgb,H,W,3); got {images_np.shape}")
        obs = torch.from_numpy(images_np.copy()).permute(0, 3, 1, 2).float() / 255.0
        obs = self._resize_rgb(obs)
        obs = self._norm_rgb(obs)                          # (V_rgb, 3, H, W) in [-1,1]
        return obs.to(self.device, dtype=self.dtype)

    # ---- raw (wire) -> model space ----------------------------------------
    # Deliberately separate from the _preprocess_* helpers below: trimming picks
    # WHICH numbers the model sees and happens first, on raw uint8/float values,
    # while preprocessing decides how they are scaled. Interleaving the two is
    # how a resize ends up applied to a left-hand view that is about to be
    # dropped, or a gather ends up applied to already-normalized data.

    def _trim_state(self, state_raw: np.ndarray) -> np.ndarray:
        """Gather the model's state out of the RAW observation. Identity when the
        two layouts describe the same arms."""
        assert state_raw.shape == (self.raw_state_dim,), (
            f"raw state must be ({self.raw_state_dim},) for raw_obs_layout="
            f"{self.raw_layout.name!r}; got {state_raw.shape}"
        )
        out = state_raw[self._state_gather]
        assert out.shape == (self.state_dim,), (
            f"gathered state is {out.shape}, expected ({self.state_dim},)"
        )
        return np.ascontiguousarray(out, dtype=np.float32)

    def _trim_tactile(self, tac_np: np.ndarray) -> np.ndarray:
        """Select the model's tactile hands out of the RAW (V_hand, F, H, W) stack.

        Runs on raw uint8 so the dropped views are never resized or normalized.
        """
        tac_np = np.asarray(tac_np)
        if tac_np.ndim != 4:
            raise ValueError(
                f"tactile must be (V_hand={self.raw_layout.tactile_hands}, "
                f"F={FINGERS}, H, W) uint8; got {tac_np.shape}"
            )
        if tac_np.shape[0] != self.raw_layout.tactile_hands or tac_np.shape[1] != FINGERS:
            raise ValueError(
                f"tactile must be (V_hand={self.raw_layout.tactile_hands}, "
                f"F={FINGERS}, H, W) for raw_obs_layout={self.raw_layout.name!r}; "
                f"got {tac_np.shape}. The client sends the FULL raw stack; the "
                f"server trims to arm_layout={self.layout.name!r}."
            )
        out = tac_np[self._tactile_hands]
        assert out.shape[:2] == (self.n_arms, FINGERS)
        return out

    def _preprocess_tactile(self, tac_np: np.ndarray) -> torch.Tensor:
        """tac_np is ALREADY trimmed to (V_hand=n_arms, F, H, W)."""
        tac_np = np.asarray(tac_np)
        if tac_np.ndim != 4 or tac_np.shape[0] != self.n_arms:
            raise ValueError(
                f"tactile must be (V_hand={self.n_arms}, F={FINGERS}, H, W) uint8 "
                f"after trimming; got {tac_np.shape}"
            )
        vh, ff, h, w = tac_np.shape
        # resize_tactile equivalent: bilinear+antialias -> round to uint8 (to
        # match the cached parquet), then /127.5 - 1 (dataset normalization).
        t = torch.from_numpy(tac_np.copy()).reshape(vh * ff, 1, h, w).float()
        t = F.interpolate(
            t, size=(TARGET_H, TARGET_W), mode="bilinear",
            align_corners=False, antialias=True,
        )
        t = t.clamp_(0, 255).to(torch.uint8).float()
        t = t.reshape(vh, ff, TARGET_H, TARGET_W) / 127.5 - 1.0
        return t.to(self.device, dtype=self.dtype)         # (V_hand, F, H, W)

    def _extract_hand_pose(self, state_raw: np.ndarray) -> torch.Tensor:
        """Per-hand 22-D joint pose for the v0d adapter, read from the RAW state.

        Read through the RAW layout and emitted in ``self.arms`` order, so pose
        row i belongs to the same hand as tactile view i after trimming.
        """
        rows = []
        for arm in self.arms:
            lo, hi = self.raw_layout.hand_joints[self.raw_layout.arm_position(arm)]
            p = state_raw[lo:hi].astype(np.float32, copy=True)
            if self._pose_mean is not None:
                p = (p - self._pose_mean) / self._pose_std
            rows.append(p)
        pose = np.stack(rows, axis=0)                      # (V_hand=n_arms, 22)
        assert pose.shape == (self.n_arms, HAND22), pose.shape
        return torch.from_numpy(np.ascontiguousarray(pose)).to(
            self.device, dtype=self.dtype
        )

    def _normalize_state(self, state_np: np.ndarray) -> np.ndarray:
        sn = (state_np[None, :] - self.sta_min) / (self.sta_max - self.sta_min + 1e-6)
        return (sn * 2.0 - 1.0).astype(np.float32)         # (1, state_dim)

    # ---- one inference step -----------------------------------------------
    @torch.no_grad()
    def step(
        self,
        images_np: np.ndarray,
        tactile_np: np.ndarray,
        state_np: np.ndarray,
        prompt: str,
        execution_step: int,
        client_health: Optional[dict] = None,
    ) -> Dict[str, np.ndarray]:
        state_np = np.asarray(state_np, dtype=np.float32).reshape(-1)
        if state_np.shape[0] != self.raw_state_dim:
            raise ValueError(
                f"state has {state_np.shape[0]} dims; raw_obs_layout="
                f"{self.raw_layout.name!r} expects {self.raw_state_dim} "
                f"(the FULL bimanual observation, even when arm_layout="
                f"{self.layout.name!r} consumes {self.state_dim})."
            )
        images_np = np.asarray(images_np)
        if images_np.ndim != 4 or images_np.shape[0] != self.n_view:
            raise ValueError(
                f"images must be ({self.n_view},H,W,3) in camera order "
                f"{self.camera_names}; got {images_np.shape}"
            )

        # RAW -> MODEL space happens once, up front, before any scaling.
        state_model = self._trim_state(state_np)

        # ---- tactile health: verify, never decide ---------------------------
        # The frames arrive ALREADY filled by the client's 30 Hz filter, which is
        # the only place the converter's frame-count windows mean what they say.
        # Raises on a contract violation, which _handle turns into ok=False.
        tac_model = self._trim_tactile(tactile_np)
        blank = self._verify_client_health(client_health, tac_model)
        self._health_step += 1
        self._log_client_health(client_health, blank, execution_step)

        frame = {
            "rgb": self._preprocess_rgb(images_np),
            "tac": self._preprocess_tactile(tac_model),
            "pose": self._extract_hand_pose(state_np),      # RAW state, raw layout
        }
        state_norm = self._normalize_state(state_model)     # (1, state_dim)

        # ---- rolling keyframe buffer (mirror vtam_server / MVActor) -------
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

        act_norm = self._infer_chunk(self.frames, state_norm, prompt)  # (chunk, D)
        act_raw = (act_norm + 1.0) / 2.0 * (
            self.act_max - self.act_min + 1e-6
        ) + self.act_min
        act_raw = act_raw.astype(np.float32, copy=False)

        L = self.layout
        if not self.is_relative:
            # Joint commands are the arm_joints..hand span (bimanual [60:118]).
            joint_targets = act_raw[:, L.arm_joints[0]:L.hand[1]].copy()
            assert joint_targets.shape[1] == (ARM7 + HAND22) * self.n_arms
            # No "status" key: it was Phase 4b's server-side health verdict, and
            # health is the client's now. Request-level success is "ok" at the
            # envelope. Dropping it also restores the pre-Phase-4b response shape.
            return {
                "action": act_raw,
                "action_mode": "absolute",
                "joint_targets": joint_targets,
            }

        # ---- relative_eef_rot6d: compose rel9 -> ABSOLUTE 4x4 EEF poses ----
        # Anchor = each arm's EEF pose in the RAW observed state that PRODUCED this
        # chunk, held fixed for every row: T_abs[t] = T_anchor @ T_rel[t]. Mirrors
        # training's single chunk-shared anchor -- do NOT re-anchor per executed step.
        # Read the anchor through the MODEL state (state_model), whose pose slices
        # are the ones the rel9 blocks were trained against.
        poses = []
        for i, arm in enumerate(self.arms):
            anchor16 = state_model[slice(*L.state_poses[i])]          # (16,) raw mat16
            rel9 = act_raw[:, slice(*L.rel_poses[i])]                 # (chunk, 9)
            tgt16 = compose_mat16_and_relative_rot6d(anchor16, rel9)  # (chunk, 16)
            poses.append(mat16_to_mat44(tgt16))
        arm_target_pose = np.stack(poses, axis=1).astype(np.float32)
        assert arm_target_pose.shape == (act_raw.shape[0], self.n_arms, 4, 4), (
            f"arm_target_pose {arm_target_pose.shape} != "
            f"({act_raw.shape[0]}, {self.n_arms}, 4, 4)"
        )
        hand_target = act_raw[:, slice(*L.hand)].copy()   # (chunk, 22*n_arms)
        assert hand_target.shape[1] == HAND22 * self.n_arms
        return {
            "action": act_raw,                             # raw, for logging/back-compat
            "action_mode": "relative_eef_rot6d",
            "arm_target_pose": arm_target_pose,            # ABSOLUTE homogeneous, ready for IK
            "hand_target": hand_target,
        }

    @torch.no_grad()
    def _infer_chunk(
        self, frames: List[dict], state_norm: np.ndarray, prompt
    ) -> np.ndarray:
        """Single-chunk tactile denoise. Mirrors
        TactileInferencer._validate_with_tactile (per-chunk body, no diagnostics).
        Returns NORMALIZED predicted action [0:action_only_dim], (action_chunk, D).
        """
        inf = self.inf
        device, dtype = self.device, self.dtype
        mem = self.mem_size
        scheduler_dtype = inf.uncond_prompt_embeds.dtype

        gen = torch.Generator(device=device).manual_seed(self.noise_seed)

        # ---- visual mem latents (per-frame VAE encode) -------------------
        rgb = torch.stack([f["rgb"] for f in frames], dim=2)   # (V_rgb, 3, mem, H, W)
        image_pf = rearrange(rgb, "v c t h w -> (v t) c h w").unsqueeze(2)
        init_latents = inf.vae.encode(image_pf).latent_dist.sample(generator=gen)
        init_latents = init_latents.to(dtype=dtype)
        init_latents = _normalize_latents(
            init_latents, inf.vae.latents_mean, inf.vae.latents_std
        )
        init_latents = rearrange(
            init_latents, "(v t) c f h w -> v c (t f) h w", t=mem
        )                                                      # (V_rgb, C, mem, h, w)

        # ---- tactile mem latents (future seed = last mem frame, ignore_seek) ---
        tac = torch.stack([f["tac"] for f in frames], dim=2)   # (V_hand, F, mem, H, W)
        tactile = tac.unsqueeze(0)                             # (1, V_hand, F, mem, H, W)
        tac_fut = tactile[:, :, :, mem - 1:mem].repeat(1, 1, 1, self.chunk_raw, 1, 1)
        tactile_synth = torch.cat([tactile, tac_fut], dim=3).contiguous()

        hand_pose_synth = None
        if self.v0d_pose_on:
            pose = torch.stack([f["pose"] for f in frames], dim=1)  # (V_hand, mem, 22)
            hand_pose = pose.unsqueeze(0)                           # (1, V_hand, mem, 22)
            hp_fut = hand_pose[:, :, mem - 1:mem, :].repeat(1, 1, self.chunk_raw, 1)
            hand_pose_synth = torch.cat([hand_pose, hp_fut], dim=2).contiguous()

        tac_full_bv = inf._encode_tactile_split(
            tactile_synth, mem, hand_pose=hand_pose_synth
        )
        tac_full = rearrange(tac_full_bv, "b v c f h w -> (b v) c f h w")
        n_view_tactile = tac_full.shape[0]
        assert n_view_tactile == self.n_arms, (
            f"tactile adapter produced {n_view_tactile} views but arm_layout="
            f"{self.layout.name!r} has {self.n_arms} arm(s); the view-axis cat "
            f"below would hand the DiT a different n_view than training."
        )

        if self.tactile_late_fuse:
            # ---- ablation: tactile bypasses the world model ----------------
            # Same encoder and the same projector bypass as the view path; only
            # the destination changes. Mirrors
            # TactileInferencer._validate_with_tactile's late-fuse branch, which
            # is what produced this checkpoint's open-loop numbers -- the server
            # and the offline eval must not disagree about where tactile enters.
            #
            # These tokens are deliberately NOT noised and NOT truncated to the
            # mem prefix: they are cross-attention context over the full T_lat,
            # as in training, where they never went through flow matching.
            tactile_ca_latents = rearrange(
                tac_full_bv, "b v c f h w -> (b v) (f h w) c"
            )
            # The hand count travels alongside the tokens rather than being
            # recoverable from them: the rows are (b*V_tac), so the DiT cannot
            # tell one hand's tokens from two hands' without being told.
            n_view_tactile_late = n_view_tactile
            mem_latents_all = init_latents
            n_view_total = self.n_view
        else:
            # ---- view-axis cat: visual rows FIRST, tactile rows LAST -------
            tactile_ca_latents = None
            n_view_tactile_late = 0
            tac_mem = tac_full[:, :, :mem]
            mem_latents_all = torch.cat([init_latents, tac_mem], dim=0)
            n_view_total = self.n_view + n_view_tactile

        latents, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
            mem_latents_all, self.latent_frames, self.latent_height, self.latent_width,
            generator=gen, noise_to_condition_frames=0,
        )

        # ---- text conditioning -------------------------------------------
        if isinstance(prompt, (list, tuple)):
            prompt = list(prompt)[:1]
        text_cond = get_text_conditions(inf.tokenizer, inf.text_encoder, prompt=prompt)
        prompt_embeds = text_cond["prompt_embeds"].to(device, dtype=scheduler_dtype)
        prompt_attention_mask = text_cond["prompt_attention_mask"]

        # ---- history_action_state = [zeros(action_dim), state_norm(state_dim)] ---
        state_in = np.concatenate(
            [np.zeros((1, self.basic_action_dim), dtype=np.float32), state_norm], axis=1
        )
        history_action_state = torch.from_numpy(state_in).unsqueeze(1).to(
            device=device, dtype=scheduler_dtype
        )                                          # (1, 1, action_in_channels)

        # ---- action noise init -------------------------------------------
        action_gen = torch.Generator(device=device).manual_seed(self.noise_seed)
        actions = randn_tensor(
            (1, self.action_chunk, self.action_in_channels),
            device=device, dtype=scheduler_dtype, generator=action_gen,
        )

        # ---- timesteps (mirror custom_pipeline.py:771-795) ---------------
        video_sequence_length = self.latent_frames * self.latent_height * self.latent_width
        sigmas = np.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            inf.scheduler.config.base_image_seq_len,
            inf.scheduler.config.max_image_seq_len,
            inf.scheduler.config.base_shift,
            inf.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            inf.scheduler, self.num_inference_steps, device, None, sigmas=sigmas, mu=mu,
        )
        retrieve_timesteps(
            inf.scheduler_action, self.num_inference_steps, device, None, sigmas=sigmas, mu=mu,
        )

        frame_rate = 30
        latent_frame_rate = frame_rate / self.TEMPORAL_DOWN
        rope_interpolation_scale = (
            1 / latent_frame_rate, self.SPATIAL_DOWN, self.SPATIAL_DOWN,
        )

        # ---- denoise loop (WM body cached after step 0; no CFG) ----------
        latents = latents.to(scheduler_dtype)
        video_states_buffer = None
        for i_step, t in enumerate(timesteps):
            latent_model_input = latents.clone()
            action_timesteps = t.unsqueeze(-1).repeat(actions.shape[0], actions.shape[1])
            timestep_per_pixel = t.expand(latent_model_input.shape[0]).unsqueeze(-1)
            if self.pixel_wise_timestep:
                timestep_per_pixel = timestep_per_pixel * (1 - conditioning_mask)
            else:
                timestep_per_pixel = timestep_per_pixel * (1 - cond_indicator)

            compute_video = (i_step == 0)
            noise_pred = inf.diffusion_model(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_per_pixel,
                encoder_attention_mask=prompt_attention_mask,
                num_frames=self.latent_frames,
                height=self.latent_height,
                width=self.latent_width,
                rope_interpolation_scale=rope_interpolation_scale,
                return_dict=False,
                action_states=actions.to(scheduler_dtype),
                action_timestep=action_timesteps,
                return_video=compute_video,
                return_action=True,
                n_view=n_view_total,
                n_view_visual=self.n_view,
                tactile_late_fuse_latents=tactile_ca_latents,
                n_view_tactile_late=n_view_tactile_late,
                video_states_buffer=video_states_buffer,
                store_buffer=compute_video,
                video_attention_mask=None,
                history_action_state=history_action_state,
                condition_mask=conditioning_mask,
            )[0]
            if compute_video:
                video_states_buffer = noise_pred["video_states_buffer"]
            action_noise_pred = noise_pred["action"].float()
            actions = inf.scheduler_action.step(
                action_noise_pred, t, actions, return_dict=False
            )[0]

        pred = actions[0].detach().float().cpu().numpy()        # (action_chunk, action_in_channels)
        return pred[:, : self.action_only_dim]  # (action_chunk, 150 abs / 136 relative)


# ============================== server =====================================

class TactileSharpaDexmateServer:
    def __init__(self, engine: TactileBimanualInference, host: str, port: int) -> None:
        self.engine = engine
        self.host = host
        self.port = port

    def _resolve_prompt(self, obs: dict) -> str:
        """The prompt the model is conditioned on, decided by the SERVER.

        With a registered task the registry string wins outright. Anything the
        client sends is treated as a claim to be checked, not as input: a stale
        client still carrying its own hardcoded prompt must fail loudly, because
        silently overriding it would hide exactly the train/serve mismatch this
        is here to catch.
        """
        e = self.engine
        if e.task is None:
            return str(obs.get("prompt", ""))
        claims = []
        if obs.get("prompt") is not None:
            claims.append(("prompt", prompt_sha256(str(obs["prompt"]))))
        if obs.get("prompt_sha256") is not None:
            claims.append(("prompt_sha256", str(obs["prompt_sha256"])))
        for field, sha in claims:
            if sha != e.task_prompt_sha256:
                raise ValueError(
                    f"client {field} does not match the served task "
                    f"{e.task_id!r}: client sha {sha[:12]}, server sha "
                    f"{e.task_prompt_sha256[:12]}. Refusing to infer -- the "
                    f"client is out of date or pointed at the wrong task."
                )
        return e.task_prompt

    def _handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "ping":
            # The client drives its send/receive shapes from THIS dict rather than
            # from its own constants, so a right-only checkpoint needs no client
            # edit -- and a mismatch is a startup error instead of a silent
            # left/right swap on the hardware.
            e = self.engine
            L = e.layout
            info = {
                # ---- task / prompt provenance -------------------------------
                # The client takes the prompt from HERE rather than holding its
                # own copy, verifies the hash, and refuses a null task_id.
                "task_id": e.task_id if e.task is not None else None,
                "task_prompt": e.task_prompt,
                "task_prompt_sha256": e.task_prompt_sha256,
                "action_chunk": e.action_chunk,
                "action_dim": e.basic_action_dim,
                "action_only_dim": e.action_only_dim,
                "action_mode": e.action_mode,
                "action_type": e.action_mode,      # back-compat alias
                # ---- layouts -------------------------------------------------
                "arm_layout": L.name,
                "arms": list(e.arms),              # response row order
                "raw_obs_layout": e.raw_layout.name,
                "layout_contract_version": e._contract_version,
                "layout_contract_sha256": e._contract["_sha256"],
                # ---- what to SEND (raw) vs what the model consumes -----------
                "raw_state_dim": e.raw_state_dim,      # client sends this
                "state_dim": e.state_dim,              # model consumes this
                "raw_tactile_shape": [e.raw_layout.tactile_hands, FINGERS],
                "model_tactile_shape": [e.n_arms, FINGERS],
                "camera_names": list(e.camera_names),  # required image order
                "n_view_visual": e.n_view,
                # The DiT's view stack, which under the tactile-WM ablation is
                # purely visual: tactile still arrives (the client sends the same
                # `tac` field either way) but reaches the action expert's own
                # cross-attention instead of passing through any DiT block.
                "n_view_total": e.n_view + (0 if e.tactile_late_fuse else e.n_arms),
                "tactile_routing": (
                    "late_fuse" if e.tactile_late_fuse else "world_model_views"
                ),
                # ---- anchoring ----------------------------------------------
                "n_previous": e.mem_size,
                "anchor_row": e.mem_size - 1,
                "anchor_fixed_per_chunk": True,
                # ---- per-(arm,finger) online blank-fill windows --------------
                "tactile_blank_windows": {
                    f"{arm}:{finger}": w
                    for (arm, finger), w in e.blank_windows.items()
                },
                # ---- tactile health contract --------------------------------
                # The CLIENT owns the filter: it runs at 30 Hz, where these
                # windows are the frame counts the converter used, and it simply
                # does not send an observation it has judged unusable. The server
                # verifies the claim and refuses on disagreement, but that path
                # is a backstop -- it should never fire in a healthy rollout.
                "tactile_health_mode": e.tactile_health_mode,
                "tactile_health_contract_version": TACTILE_HEALTH_CONTRACT_VERSION,
                "tactile_health_sha256": e._health_sha,
                "tactile_health_usable_statuses": list(USABLE_STATUSES),
                "tactile_health_required_in_obs": e.tactile_health_mode != "disabled",
                "channel_breakdown": {
                    "force": list(L.force),
                    "arm_joints": list(L.arm_joints),
                    "hand": list(L.hand),
                    "action_poses": [list(s) for s in L.action_poses],
                    "rel_poses": [list(s) for s in L.rel_poses],
                    "state_hands": [list(s) for s in L.state_hands],
                    "state_poses": [list(s) for s in L.state_poses],
                },
            }
            if e.is_relative:
                # arm comes from ABSOLUTE 4x4 EEF poses (client IK); hand executed direct.
                info["arm_target_pose_shape"] = [e.action_chunk, e.n_arms, 4, 4]
                info["hand_target_dim"] = HAND22 * e.n_arms
            else:
                info["joint_target_slice"] = [L.arm_joints[0], L.hand[1]]
                info["joint_target_dim"] = (ARM7 + HAND22) * e.n_arms
            return {"ok": True, "info": info}
        if op == "reset":
            self.engine.reset()
            return {"ok": True}
        if op == "step":
            obs = request["obs"]
            try:
                prompt = self._resolve_prompt(obs)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            try:
                out = self.engine.step(
                    images_np=obs["images"],
                    tactile_np=obs["tactile"],
                    state_np=obs["state"],
                    prompt=prompt,
                    execution_step=int(obs.get("execution_step", 1)),
                    client_health=obs.get("tactile_health"),
                )
            except TactileHealthContractError as exc:
                # Health-claim violations are the client's to fix; refuse rather
                # than infer on data one end has already called unusable. Narrow
                # on purpose -- a shape error must stay a shape error.
                return {"ok": False, "error": str(exc),
                        "error_kind": "tactile_health_contract"}
            return {"ok": True, **out}
        if op == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"unknown op: {op}"}

    def serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[tactile_server] listening on {self.host}:{self.port} — ready.", flush=True)
        while True:
            conn, addr = srv.accept()
            print(f"[tactile_server] connection from {addr}", flush=True)
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
                        print("[tactile_server] shutdown requested.", flush=True)
                        conn.close()
                        return
            except (ConnectionError, OSError) as e:
                print(f"[tactile_server] {addr} disconnected: {e}", flush=True)
            except Exception:
                print(f"[tactile_server] error:\n{traceback.format_exc()}", flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass


def main() -> None:
    p = argparse.ArgumentParser(description="DexVTAM tactile bimanual server (dexmate + sharpa)")
    p.add_argument("-c", "--config", required=True, help="Stage-3 action-model YAML (use_tactile_views=true)")
    p.add_argument("-w", "--weight", required=True,
                   help="step_NNNN DIRECTORY (holds diffusion_pytorch_model.safetensors + projector.pt)")
    p.add_argument("--domain-name", required=True,
                   help="stats key prefix; must match the stat_file (e.g. "
                        "lerobot_0501_pick_cube_sdh_100_episodes_no_wrist_vtam_correctaction)")
    p.add_argument("--task", required=True,
                   help="registered task id (selects the served prompt and is "
                        "cross-checked against the config's domain), or "
                        f"{NO_TASK!r} for offline tooling only")
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--threshold", type=int, default=0,
                   help="executed steps between keyframe commits (0 => action_chunk)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default=os.environ.get("VTAM_SERVER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("VTAM_SERVER_PORT", "5008")))
    args = p.parse_args()

    print("[tactile_server] loading model ...", flush=True)
    engine = TactileBimanualInference(
        config_file=args.config,
        weight_dir=args.weight,
        domain_name=args.domain_name,
        denoise_steps=args.denoise_steps,
        task_id=args.task,
        threshold=(args.threshold or None),
        device=args.device,
    )
    print("[tactile_server] model loaded.", flush=True)
    TactileSharpaDexmateServer(engine, args.host, args.port).serve()


if __name__ == "__main__":
    main()
