#!/usr/bin/env bash
# Assemble a runnable tree from this repository plus Genie-Envisioner-V1.
#
# Genie-Envisioner-V1 carries no licence, so its code is not vendored here. You
# clone it yourself at the commit DexTacWAM was developed against, this script
# applies our patch to the eight upstream files we changed, and overlays the
# modules under src/ that keep upstream-relative paths so imports resolve.
set -euo pipefail

UPSTREAM_URL="https://github.com/AgibotTech/Genie-Envisioner-V1.git"
UPSTREAM_SHA="d54425c4d0ba9d56b33adc4bfe5e5187e5335603"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/build}"

if [[ -e "$DEST" ]]; then
  echo "error: $DEST already exists; remove it or pass another path." >&2
  exit 1
fi

echo "==> cloning Genie-Envisioner-V1 into $DEST"
git clone --quiet "$UPSTREAM_URL" "$DEST"
git -C "$DEST" checkout --quiet "$UPSTREAM_SHA"

echo "==> applying patches/genie-envisioner-${UPSTREAM_SHA:0:8}.patch"
git -C "$DEST" apply "$HERE/patches/genie-envisioner-${UPSTREAM_SHA:0:8}.patch"

echo "==> overlaying DexTacWAM modules"
cp -r "$HERE/src/." "$DEST/"
ln -s "$HERE/configs" "$DEST/configs/dextacwam"
ln -s "$HERE/tests" "$DEST/tests_dextacwam"

cat <<EOF

Done. The assembled tree is at:
  $DEST

Expected data layout, relative to that directory:
  pretrained_models/ltx_video/
  pretrained_models/genie_envisioner/GE_base_fast_v0.1.safetensors
  data/datasets_lerobot/<task>/
  data/cache/<task>/
  outputs/

Next: see README.md for the three training stages.
EOF
