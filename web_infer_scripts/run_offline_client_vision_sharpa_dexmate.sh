#!/usr/bin/env bash
# Offline check for the DexVTAM vision-only V=3 server (dexmate + sharpa).
# Sibling of run_offline_client_tactile_sharpa_dexmate.sh; defaults to the
# handover-with-wrist 100-episodes local corpus, val episode 90.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

HOST="${HOST:-localhost}"
PORT="${PORT:-5010}"
DATA_ROOT="${DATA_ROOT:-/home/zekai/dex-vtam/data/lerobot_dataset/lerobot_0514night_pick_cube_handover_sdh_100_episodes}"
EPISODE="${EPISODE:-90}"
N_CHUNKS="${N_CHUNKS:-4}"

python3 web_infer_scripts/offline_client_vision_sharpa_dexmate.py \
    --host "$HOST" \
    --port "$PORT" \
    --data-root "$DATA_ROOT" \
    --episode "$EPISODE" \
    --n-chunks "$N_CHUNKS"
