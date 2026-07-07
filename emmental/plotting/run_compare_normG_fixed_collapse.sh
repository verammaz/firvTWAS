#!/bin/bash
# Compare joint global parameters: normG_collapse (collapsed) vs normG_fixed (full).
set -euo pipefail

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/plotting"
COLLAPSE_ROOT="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01_normG_collapse"
FIXED_ROOT="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01_normG_fixed"
OUT_DIR="${OUT_DIR:-${COLLAPSE_ROOT}/param_compare_normG_fixed}"

PY="${PYTHON:-python3}"
if ! "$PY" -c "import matplotlib" 2>/dev/null; then
    source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate 2>/dev/null || true
    PY=python
fi

mkdir -p "$OUT_DIR"

echo "=== τ / T boxplots across refits ==="
"$PY" "$SCRIPT_DIR/plot_tau_refit_compare.py" \
    --joint_dir_a "${COLLAPSE_ROOT}/joint" \
    --joint_dir_b "${FIXED_ROOT}/joint" \
    --label_a "normG_collapse" \
    --label_b "normG_fixed" \
    --out_dir "$OUT_DIR"

echo "=== Mean τ / T / per-gene param correlations ==="
"$PY" "$SCRIPT_DIR/compare_joint_learned_params.py" \
    --root_a "$COLLAPSE_ROOT" \
    --root_b "$FIXED_ROOT" \
    --label_a "normG_collapse" \
    --label_b "normG_fixed" \
    --out_dir "$OUT_DIR"

echo "Done -> $OUT_DIR"
