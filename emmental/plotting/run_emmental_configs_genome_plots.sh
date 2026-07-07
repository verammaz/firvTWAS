#!/bin/bash
# Emmental config comparison (full / no_rhog / no_wg / no_wg_rhog) — genome-wide all genes.

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TRAIN_COMMON01_ROOT:-/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01}"

PY="${PYTHON:-python3}"
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate 2>/dev/null || true
PY=python

"$PY" "$SCRIPT_DIR/plot_emmental_configs_pergene_r2.py" \
    --train_common01_root "$ROOT" \
    --out_dir "$ROOT/plots" \
    --tag genome
