#!/usr/bin/env bash
# =============================================================================
# Start the DexVTAM TACTILE policy server for a relative_eef_rot6d checkpoint,
# right-only (tong / bowl) or bimanual (unscrew_v2). Sibling of
# run_server_tactile_sharpa_dexmate.sh; the port only opens once the model is
# fully loaded, so a successful connect from the robot client means "ready".
#
# Nothing here branches on the layout. TASK picks a yaml; the yaml declares
# arm_layout; the server and all four guards derive every width, camera and
# slice from that one declaration. Adding a task is a row in the case below.
#
# Everything that can be derived is derived, because the failure modes here are
# silent rather than loud:
#   * DOMAIN_NAME comes from the yaml's data.train.domains[0] -- serving bowl's
#     checkpoint with tong's stats keys would otherwise just de-normalize every
#     action with the wrong q01/q99 and produce plausible, wrong targets.
#   * CONFIG comes from TASK, so the config and the checkpoint cannot be paired
#     by hand incorrectly.
#
# Four guards run BEFORE the model loads (each one catches a mismatch that is
# otherwise invisible until the arm moves):
#   1. the ckpt dir is complete (DiT safetensors + projector.pt)
#   2. the ckpt's config.json agrees with the yaml on action_in_channels and
#      max_view -- this is what rejects a layout mismatch in either direction
#      (bimanual is 226 / max_view 5, right-only is 113 / 3)
#   3. the local v0d tactile adapter is the SAME snapshot training used
#      (sha256); it loads with strict=False, so a wrong snapshot loads without
#      error and only shows up as degraded contact behaviour
#   4. the offline dry-run gate (layouts, stats widths, gather, compose)
#
# WHICH checkpoint: the B3 ACTION run (stage3_action_full_...), not the B2 world
# model it warmstarted from. bowl's only B3 ckpt so far is step_10000.
#
# The config variant (_local for the robot box, _remote for remote) is picked by
# checking which one's pretrained dir exists here; CONFIG overrides it.
#
# Usage (robot-side box):
#     TASK=bowl STEP=10000 bash web_infer_scripts/run_server_tactile_relative.sh
#     TASK=bowl WEIGHT=/abs/path/to/stage3_.../<TS>/step_10000 bash ...
#     TASK=bowl bash ...    # no STEP => lists the available step_* dirs and exits
#     GPU=4 TASK=bowl STEP=10000 ... (force a GPU)
#     VARIANT=ablation TASK=wipe_white_board STEP=10000 ...   # tactile-WM ablation
#
# Usage (remote, for the offline F1/F3 gates, conda already activated). The
# checkpoints live outside this checkout, so give WEIGHT or RUN_ROOT_BASE:
#     GPU=3 TASK=bowl \
#     WEIGHT=outputs/stage3_action_full_0729_bowl_right_only_eef_relative_wm_bypass_shared_rmsnorm_long50000/2026_07_29_22_53_51/step_10000 \
#     bash web_infer_scripts/run_server_tactile_relative.sh
# =============================================================================

set -uo pipefail

# ---- repo + venv ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# The robot-side box runs uv (.venv); remote runs conda. Activate the venv when it
# exists, otherwise assume the caller already activated an environment -- so the
# same launcher can be used for the offline F1/F3 gates on a GPU cluster.
ACTIVATE="${ACTIVATE:-$REPO_ROOT/.venv/bin/activate}"
if [[ -f "$ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$ACTIVATE"
else
    echo "[run_relative] no venv at $ACTIVATE; using the current environment" \
         "($(command -v python3 || echo 'python3 NOT FOUND'))"
fi

# ---- which task ------------------------------------------------------------
# TASK is the registry id, so the string typed here is the same one whose prompt
# gets served and hashed. Nothing below branches on layout: right-only and
# bimanual differ only in which yaml this resolves to, and the guards re-derive
# every width from that yaml.
#
# VARIANT picks WHICH checkpoint family for that task. Two corpora are served by
# two policies each: `full`, the production visuo-tactile B3, and `ablation`, the
# tactile-world-model ablation whose tactile skips the DiT and enters the action
# expert directly. They share the corpus, the prompt and the registry id -- so
# the ablation is a VARIANT here rather than a second TASK, which keeps one copy
# of the prompt in the registry and lets the domain assertion still fire.
TASK="${TASK:-}"
VARIANT="${VARIANT:-full}"
case "$TASK" in
    bowl)             CFG="configs/bowl_unstack/stage3_action_expert.yaml" ;;
    placed_tong)      CFG="configs/tongs/stage3_action_expert.yaml" ;;
    unscrew_v2)       CFG="configs/bottle_cap/stage3_action_expert.yaml" ;;
    cube_handover)    CFG="configs/cube_handover/stage3_action_expert.yaml" ;;
    wipe_white_board) CFG="configs/wipe_whiteboard/stage3_action_expert.yaml" ;;
    # HEAD CAMERA ONLY (max_view 2), unlike every other right-only task here,
    # which also sends the right wrist. Guard 2 is what catches serving this
    # against a bowl/tong-shaped checkpoint: the widths agree at 113 but the
    # view count does not.
    pick_place_cube)  CFG="configs/cube_place/stage3_action_expert.yaml" ;;
    *)          echo "[run_relative] FATAL: set TASK=bowl|placed_tong|unscrew_v2|cube_handover|wipe_white_board|pick_place_cube (got '${TASK}')" >&2
                echo "    TASK must be a task_registry id; see" >&2
                echo "      python3 -m data.utils.task_registry --list" >&2
                exit 2 ;;
esac

# ---- which checkpoint family for that task ---------------------------------
case "$VARIANT" in
    full)  ;;
    ablation)
        echo "[run_relative] FATAL: the tactile-WM ablation policies are not part" >&2
        echo "    of this release. configs/ablations/ holds the stage-2 visual-only" >&2
        echo "    world models, which are not servable as action policies." >&2
        echo "    Train an ablation stage 3 yourself and pass CONFIG=<path>." >&2
        exit 2 ;;
    *)  echo "[run_relative] FATAL: set VARIANT=full|ablation (got '${VARIANT}')" >&2
        exit 2 ;;
esac
# An explicit CONFIG always wins; otherwise serve the task's released config.
CONFIG="${CONFIG:-$REPO_ROOT/$CFG}"
[[ -f "$CONFIG" ]] || { echo "[run_relative] FATAL: config not found: $CONFIG" >&2; exit 2; }

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5008}"
DENOISE_STEPS="${DENOISE_STEPS:-10}"
# No default STEP on purpose. What we serve is the B3 ACTION checkpoint under
# stage3_action_full_...; the numbers that appear in the yaml (e.g. step_30000)
# belong to the B2 WORLD MODEL it warmstarted from, and a stale default here
# would quietly serve the wrong weights.
STEP="${STEP:-}"
# Read the run dir straight out of the config's output_dir rather than rebuilding
# it from a slug, so the two cannot drift apart.
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$REPO_ROOT/$(python3 -c "
import yaml,sys
print(yaml.safe_load(open('$CONFIG')).get('output_dir') or '')
")}"
RUN_TS="${RUN_TS:-}"

# The v0d Stage-1 adapter every relative run trained against -- right-only and
# bimanual alike share this one snapshot
# (remote: outputs/stage1_lite_v0d_full/2026_05_20_06_52_55/checkpoints/best_recall_post/model.pt)
EXPECTED_V0D_SHA256="${EXPECTED_V0D_SHA256:-031d5e05984471ade8d2530c61a2fd073085c98eb989866cfc2ae21cb78d8df5}"

# ---- derive DOMAIN_NAME from the yaml (never hand-typed) --------------------
DOMAIN_NAME="${DOMAIN_NAME:-$(python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg['data']['train']['domains'][0])
")}"
[[ -n "$DOMAIN_NAME" ]] || { echo "[run_relative] FATAL: could not read domains[0] from $CONFIG" >&2; exit 2; }

# Banner only -- the server derives its own layout from the same yaml. Printed
# because "which layout am I actually serving" is the question an operator has
# at exactly this moment, and the launcher's name no longer answers it.
ARM_LAYOUT="$(python3 -c "
import yaml
print(yaml.safe_load(open('$CONFIG'))['data']['train'].get('arm_layout', 'bimanual'))
" 2>/dev/null || echo unknown)"

# ---- resolve the step_NNNN ckpt DIRECTORY ----------------------------------
WEIGHT="${WEIGHT:-}"
if [[ -z "$WEIGHT" ]]; then
    # Resolve the run dir: flat layout ($BASE/step_NNNN) if present, else the
    # newest timestamped subdir ($BASE/<TS>/step_NNNN).
    RUN_DIR="$RUN_ROOT_BASE"
    if ! compgen -G "$RUN_ROOT_BASE/step_*" > /dev/null; then
        if [[ -z "$RUN_TS" ]]; then
            RUN_TS=$(ls -td "$RUN_ROOT_BASE"/*/ 2>/dev/null | head -1 | xargs -I{} basename {})
            [[ -z "$RUN_TS" ]] && { echo "[run_relative] FATAL: no run dirs under $RUN_ROOT_BASE" >&2; exit 3; }
            echo "[run_relative] auto-picked RUN_TS=$RUN_TS"
        fi
        RUN_DIR="$RUN_ROOT_BASE/$RUN_TS"
    fi
    if [[ -z "$STEP" ]]; then
        echo "[run_relative] FATAL: set STEP=<n> (or WEIGHT=<dir>)." >&2
        echo "    This is the B3 ACTION checkpoint under stage3_action_full_...," >&2
        echo "    NOT the B2 world model it warmstarted from." >&2
        echo "    Available under $RUN_DIR:" >&2
        if compgen -G "$RUN_DIR/step_*" > /dev/null; then
            for d in "$RUN_DIR"/step_*; do echo "      $(basename "$d")" >&2; done
        else
            echo "      (none -- B3 has not saved a checkpoint here yet)" >&2
        fi
        exit 2
    fi
    WEIGHT="$RUN_DIR/step_${STEP}"
    echo "[run_relative] resolved WEIGHT=$WEIGHT"
fi

# ---- guard 1: complete checkpoint dir --------------------------------------
if [[ ! -f "$WEIGHT/diffusion_pytorch_model.safetensors" || ! -f "$WEIGHT/projector.pt" ]]; then
    echo "[run_relative] FATAL: $WEIGHT is not a complete tactile ckpt dir" >&2
    echo "    (needs diffusion_pytorch_model.safetensors AND projector.pt)" >&2
    exit 4
fi

# ---- guard 2: ckpt config.json vs served yaml ------------------------------
# A bimanual ckpt has action_in_channels=226 / max_view=5; a right-only one has
# 113 / 3. Loading the wrong pair would fail deep inside the DiT (or worse, not
# fail at all for max_view), so compare here.
if [[ -f "$WEIGHT/config.json" ]]; then
    python3 -c "
import json, sys, yaml
cfg = yaml.safe_load(open('$CONFIG'))['diffusion_model']['config']
ck  = json.load(open('$WEIGHT/config.json'))
bad = []
for k in ('action_in_channels', 'action_out_channels', 'max_view'):
    # Compare only where BOTH declare the key: older configs omit max_view, and
    # a missing key is not a mismatch.
    if k in ck and k in cfg and int(ck[k]) != int(cfg[k]):
        bad.append(f'{k}: ckpt={ck[k]} yaml={cfg[k]}')
if bad:
    print('[run_relative] FATAL: checkpoint does not match the served config:', file=sys.stderr)
    for b in bad:
        print('    ' + b, file=sys.stderr)
    print('    -> the ckpt and the yaml are different layouts (bimanual is', file=sys.stderr)
    print('       226 / max_view 5, right-only is 113 / 3); check TASK', file=sys.stderr)
    sys.exit(6)
print(f\"[run_relative] guard 2 OK: ckpt config.json matches yaml \"
      f\"(action_in_channels={cfg['action_in_channels']}, \"
      f\"max_view={cfg.get('max_view', 'unset')})\")
" || exit 6
else
    echo "[run_relative] WARN: $WEIGHT/config.json missing; cannot cross-check the ckpt shape" >&2
fi

# ---- guard 3: v0d tactile adapter identity ---------------------------------
V0D_PATH="$(python3 -c "
import yaml
print(yaml.safe_load(open('$CONFIG'))['tactile_vae']['model_path'])
")"
if [[ ! -f "$V0D_PATH" ]]; then
    echo "[run_relative] FATAL: tactile_vae.model_path not found: $V0D_PATH" >&2
    exit 7
fi
V0D_SHA="$(sha256sum "$V0D_PATH" | awk '{print $1}')"
if [[ "$V0D_SHA" != "$EXPECTED_V0D_SHA256" ]]; then
    echo "[run_relative] FATAL: v0d tactile adapter is NOT the snapshot training used." >&2
    echo "    path     : $V0D_PATH" >&2
    echo "    got      : $V0D_SHA" >&2
    echo "    expected : $EXPECTED_V0D_SHA256" >&2
    echo "    The adapter loads with strict=False, so this would NOT raise at load" >&2
    echo "    time -- the per-finger tactile latents would just be mis-aligned with" >&2
    echo "    what the world-model body learned. Copy the right snapshot, or set" >&2
    echo "    EXPECTED_V0D_SHA256=<sha> if you have deliberately changed adapters." >&2
    exit 7
fi
echo "[run_relative] guard 3 OK: v0d adapter sha256 ${V0D_SHA:0:8} matches training"

# ---- guard 4: offline dry-run (layouts / stats / gather / compose) ----------
python3 web_infer_scripts/dryrun_relative_server.py -c "$CONFIG" --task "$TASK" || {
    echo "[run_relative] FATAL: dry-run gate failed; refusing to serve." >&2
    exit 8
}

# ---- GPU pick (most-free with >= MIN_FREE_MB, unless GPU= forces one) -------
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
if [[ -n "${GPU:-}" ]]; then
    echo "[run_relative] using user-specified GPU: $GPU"
else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' -v MIN="$MIN_FREE_MB" '{gsub(/ /,"");if($2+0>=MIN+0)printf "%010d %s\n",(1000000-$2),$1}' \
        | sort -n | head -1 | awk '{print $2}')
    [[ -z "$GPU" ]] && { echo "[run_relative] FATAL: no GPU has >= ${MIN_FREE_MB} MiB free." >&2; nvidia-smi >&2; exit 5; }
    echo "[run_relative] auto-picked GPU: $GPU"
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

# ---- manifest: enough to identify this exact serve from the log alone -------
echo "================================================================="
echo " DexVTAM tactile policy server -- relative_eef_rot6d ($ARM_LAYOUT)"
echo "   Task    : $TASK   (variant=$VARIANT)"
echo "   Config  : $CONFIG"
echo "   Weight  : $WEIGHT"
echo "   Domain  : $DOMAIN_NAME   (from yaml domains[0])"
echo "   v0d     : $V0D_PATH  (sha256 ${V0D_SHA:0:8})"
echo "   Git     : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "   GPU     : $GPU"
echo "   Listen  : $HOST:$PORT   (denoise_steps=$DENOISE_STEPS)"
echo "================================================================="
echo " The server prints its own layout manifest next (arm_layout, raw->model"
echo " gather, cameras, contract sha). The client must send images in the"
echo " camera order it reports."
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
