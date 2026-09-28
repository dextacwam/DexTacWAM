#!/usr/bin/env bash
# =============================================================================
# Start the DexVTAM TACTILE deployment server (dexmate + sharpa, Stage-3
# action_full + tactile). The port only opens once the model is fully loaded
# -- a successful connect from the robot client means "ready".
#
# DexVTAM sibling of VTAM/web_infer_scripts/run_server_sharpa_dexmate.sh, with:
#   * uv venv activation (DexVTAM runs in .venv, not conda)
#   * --weight points at a step_NNNN DIRECTORY (DiT safetensors + projector.pt)
#   * defensive single-process accelerate env (TactileInferencer.prepare_models
#     calls accelerate PartialState(); set RANK/WORLD_SIZE so it inits cleanly
#     when launched as plain python, no torchrun)
#   * /tmp cache dirs (cthulhu2 has no writable /data/local; avoids triton-on-NFS)
#
# Usage:
#     bash web_infer_scripts/run_server_tactile_sharpa_dexmate.sh           # default GPU pick
#     GPU=4 bash web_infer_scripts/run_server_tactile_sharpa_dexmate.sh     # force GPU
#     STEP=10000 bash web_infer_scripts/run_server_tactile_sharpa_dexmate.sh
# =============================================================================

set -uo pipefail

# ---- repo + venv ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# ---- what to serve (edit these) --------------------------------------------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5008}"
DOMAIN_NAME="${DOMAIN_NAME:-lerobot_0501_pick_cube_sdh_100_episodes_no_wrist_vtam_correctaction}"
CONFIG="${CONFIG:-configs/cube_place/stage3_action_expert.yaml}"
# WEIGHT is the ckpt DIRECTORY (holds diffusion_pytorch_model.safetensors +
# projector.pt). Default = the flat exported ckpt dir on this box. Set WEIGHT=
# directly to point elsewhere, or WEIGHT="" to auto-resolve the timestamped
# RUN_ROOT_BASE/RUN_TS/step_STEP layout instead.
WEIGHT="${WEIGHT-$REPO_ROOT/outputs/stage3_action_full_right_hand_pick_cube_correctaction_cubewm_bypass_shared_rmsnorm_long50000_step20000}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$REPO_ROOT/outputs/stage3_action_full_right_hand_pick_cube_correctaction_cubewm_bypass_shared_rmsnorm_long50000}"
RUN_TS="${RUN_TS:-}"          # empty => auto-pick latest run dir (TS-subdir layout)
STEP="${STEP:-20000}"
DENOISE_STEPS="${DENOISE_STEPS:-10}"
# Legacy bimanual launcher (erase / chip / cube). Those tasks have no verified
# prompt provenance yet, so they serve as TASK=none: the prompt is taken verbatim
# from the observation, exactly as before the registry existed, and the hardware
# client refuses to connect. This is what keeps the byte-parity capture runnable.
TASK="${TASK:-none}"

# ---- resolve the step_NNNN ckpt DIRECTORY ----------------------------------
# WEIGHT= can be set directly to bypass resolution entirely. Otherwise we try
# the FLAT layout ($RUN_ROOT_BASE/step_NNNN, e.g. the erase run) first, then
# fall back to the TS-subdir layout ($RUN_ROOT_BASE/<TS>/step_NNNN).
WEIGHT="${WEIGHT:-}"
if [[ -z "$WEIGHT" ]]; then
    if [[ -d "$RUN_ROOT_BASE/step_${STEP}" ]]; then
        WEIGHT="$RUN_ROOT_BASE/step_${STEP}"
        echo "[run_tactile_server] flat layout: WEIGHT=$WEIGHT"
    else
        if [[ -z "$RUN_TS" ]]; then
            RUN_TS=$(ls -td "$RUN_ROOT_BASE"/*/ 2>/dev/null | head -1 | xargs -I{} basename {})
            [[ -z "$RUN_TS" ]] && { echo "[run_tactile_server] FATAL: no run dirs under $RUN_ROOT_BASE" >&2; exit 3; }
            echo "[run_tactile_server] auto-picked RUN_TS=$RUN_TS"
        fi
        WEIGHT="$RUN_ROOT_BASE/$RUN_TS/step_${STEP}"
    fi
fi
if [[ ! -f "$WEIGHT/diffusion_pytorch_model.safetensors" || ! -f "$WEIGHT/projector.pt" ]]; then
    echo "[run_tactile_server] FATAL: $WEIGHT is not a complete tactile ckpt dir" >&2
    echo "    (needs diffusion_pytorch_model.safetensors AND projector.pt)" >&2
    exit 4
fi

# ---- GPU pick (most-free with >= MIN_FREE_MB, unless GPU= forces one) -------
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
if [[ -n "${GPU:-}" ]]; then
    echo "[run_tactile_server] using user-specified GPU: $GPU"
else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' -v MIN="$MIN_FREE_MB" '{gsub(/ /,"");if($2+0>=MIN+0)printf "%010d %s\n",(1000000-$2),$1}' \
        | sort -n | head -1 | awk '{print $2}')
    [[ -z "$GPU" ]] && { echo "[run_tactile_server] FATAL: no GPU has >= ${MIN_FREE_MB} MiB free." >&2; nvidia-smi >&2; exit 5; }
    echo "[run_tactile_server] auto-picked GPU: $GPU"
fi

# ---- single-process accelerate env (prepare_models calls PartialState()) ---
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 500))}"

# ---- local-fast cache dirs (avoid NFS / read-only /data/local) -------------
_CACHE_BASE="${EVAL_CACHE_BASE:-/tmp/dexvtam_server_cache_${USER}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$_CACHE_BASE/triton}"
export HF_HOME="${HF_HOME:-$_CACHE_BASE/hf}"
export TORCH_HOME="${TORCH_HOME:-$_CACHE_BASE/torch}"
export TMPDIR="${TMPDIR:-$_CACHE_BASE/tmp}"
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" 2>/dev/null || true

echo "================================================================="
echo " DexVTAM tactile policy server (dexmate + sharpa)"
echo "   Config  : $CONFIG"
echo "   Weight  : $WEIGHT"
echo "   Domain  : $DOMAIN_NAME"
echo "   Task    : $TASK"
echo "   GPU     : $GPU"
echo "   Listen  : $HOST:$PORT   (denoise_steps=$DENOISE_STEPS)"
echo "================================================================="

CUDA_VISIBLE_DEVICES="$GPU" python3 web_infer_scripts/tactile_server_sharpa_dexmate.py \
    -c "$CONFIG" \
    -w "$WEIGHT" \
    --domain-name "$DOMAIN_NAME" \
    --task "$TASK" \
    --denoise-steps "$DENOISE_STEPS" \
    --device "cuda:0" \
    --host "$HOST" \
    --port "$PORT"
