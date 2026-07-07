#!/bin/bash
# Baseline vs Emmental R² proportion plots (thresholds 0.01 and 0.1).
#
# 1) Post-joint + top200_BRR genes (fixed list)
# 2) Post-pergene + genome-wide gene intersection (all methods have R²)
#
# Outputs under {config}/plots/post_joint_top200/ and post_pergene_intersection/

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TRAIN_COMMON01_ROOT:-/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01}"
GENE_LIST="${GENE_LIST:-/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt}"
BASELINE="${BASELINE_ROOT:-/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01}"
THRESHOLDS="0.01 0.1"

PY="${PYTHON:-python3}"
if ! "$PY" -c "import matplotlib" 2>/dev/null; then
    source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate 2>/dev/null || true
    PY=python
fi

echo "=== Post-joint vs baselines (top200) ==="
"$PY" "$SCRIPT_DIR/plot_postjoint_r2_comparison.py" \
    --train_common01_root "$ROOT" \
    --baseline_root "$BASELINE" \
    --emmental-source post_joint \
    --gene-set fixed_list \
    --gene_list "$GENE_LIST" \
    --plots-subdir post_joint_top200 \
    --thresholds $THRESHOLDS

echo "=== Post-pergene vs baselines (genome intersection) ==="
"$PY" "$SCRIPT_DIR/plot_postjoint_r2_comparison.py" \
    --train_common01_root "$ROOT" \
    --baseline_root "$BASELINE" \
    --emmental-source post_pergene \
    --gene-set intersection \
    --plots-subdir post_pergene_intersection \
    --thresholds $THRESHOLDS

echo "Done."
