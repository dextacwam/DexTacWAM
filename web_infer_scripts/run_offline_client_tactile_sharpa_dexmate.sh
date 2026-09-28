#!/usr/bin/env bash
# =============================================================================
# Offline check for the DexVTAM TACTILE server: replays a recorded
# correctaction LeRobot episode through the wire interface and compares the
# server's de-normalized predictions against the recorded ground-truth actions.
#
# Start run_server_tactile_sharpa_dexmate.sh first (in another terminal), then
# run this. Tactile sibling of VTAM/web_infer_scripts/run_offline_client_sharpa_dexmate.sh.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

HOST="${HOST:-localhost}"
PORT="${PORT:-5008}"
DATA_ROOT="${DATA_ROOT:-/home/zekai/dex-vtam/data/lerobot_dataset/lerobot_0501_pick_cube_sdh_100_episodes_no_wrist_vtam_correctaction}"
EPISODE="${EPISODE:-90}"     # 90..99 are the val episodes for this corpus
N_CHUNKS="${N_CHUNKS:-6}"

python3 web_infer_scripts/offline_client_tactile_sharpa_dexmate.py \
    --host "$HOST" \
    --port "$PORT" \
    --data-root "$DATA_ROOT" \
    --episode "$EPISODE" \
    --n-chunks "$N_CHUNKS"
