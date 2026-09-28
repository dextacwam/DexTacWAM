#!/usr/bin/env bash
# =============================================================================
# DexVTAM VISION-ONLY V=3 deployment server (dexmate + sharpa). Sibling of
# run_server_tactile_sharpa_dexmate.sh, but for visual-only (no tactile)
# checkpoints with head + 2 wrist cameras.
#
# Defaults target the handover-with-wrist visual-only step_20000 ckpt + its
# trainset_eval_local yaml (the one we made for local paths). Override as needed.
#
# Usage:
#     bash web_infer_scripts/run_server_vision_sharpa_dexmate.sh           # default GPU pick
#     GPU=2 bash web_infer_scripts/run_server_vision_sharpa_dexmate.sh
#     WEIGHT=/path/to/your/step_20000 bash ...                              # direct weight override
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5010}"
DOMAIN_NAME="${DOMAIN_NAME:-0514_pick_cube_handover_100episodes}"
CONFIG="${CONFIG:-configs/cube_handover/stage3_action_expert.yaml}"
# WEIGHT must be a step_NNNN DIR with diffusion_pytorch_model.safetensors.
# Default = the local handover-with-wrist visual-only step_20000 ckpt.
WEIGHT="${WEIGHT:-$REPO_ROOT/outputs/0514_pick_cube_handover_with_wrist_handoverwm_visual_only_rmsnorm_step20000}"
DENOISE_STEPS="${DENOISE_STEPS:-10}"

if [[ ! -f "$CONFIG" ]]; then
    echo "[run_vision_server] FATAL: config not found: $CONFIG" >&2; exit 2
fi
if [[ ! -f "$WEIGHT/diffusion_pytorch_model.safetensors" ]]; then
    echo "[run_vision_server] FATAL: $WEIGHT has no diffusion_pytorch_model.safetensors" >&2; exit 3
fi

# ---- GPU pick (most-free >= MIN_FREE_MB, unless GPU= forces one) ------------
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
if [[ -n "${GPU:-}" ]]; then
    echo "[run_vision_server] using user-specified GPU: $GPU"
else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' -v MIN="$MIN_FREE_MB" '{gsub(/ /,"");if($2+0>=MIN+0)printf "%010d %s\n",(1000000-$2),$1}' \
        | sort -n | head -1 | awk '{print $2}')
    [[ -z "$GPU" ]] && { echo "[run_vision_server] FATAL: no GPU has >= ${MIN_FREE_MB} MiB free." >&2; nvidia-smi >&2; exit 5; }
    echo "[run_vision_server] auto-picked GPU: $GPU"
fi

# Single-process accelerate env (TactileInferencer.prepare_models calls PartialState())
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 500))}"

_CACHE_BASE="${EVAL_CACHE_BASE:-/tmp/dexvtam_server_cache_${USER}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$_CACHE_BASE/triton}"
export HF_HOME="${HF_HOME:-$_CACHE_BASE/hf}"
export TORCH_HOME="${TORCH_HOME:-$_CACHE_BASE/torch}"
export TMPDIR="${TMPDIR:-$_CACHE_BASE/tmp}"
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" 2>/dev/null || true

echo "================================================================="
echo " DexVTAM vision-only V=3 policy server (dexmate + sharpa)"
echo "   Config  : $CONFIG"
echo "   Weight  : $WEIGHT"
echo "   Domain  : $DOMAIN_NAME"
echo "   GPU     : $GPU"
echo "   Listen  : $HOST:$PORT   (denoise_steps=$DENOISE_STEPS)"
echo "================================================================="

CUDA_VISIBLE_DEVICES="$GPU" python3 web_infer_scripts/vision_server_sharpa_dexmate.py \
    -c "$CONFIG" \
    -w "$WEIGHT" \
    --domain-name "$DOMAIN_NAME" \
    --denoise-steps "$DENOISE_STEPS" \
    --device "cuda:0" \
    --host "$HOST" \
    --port "$PORT"
