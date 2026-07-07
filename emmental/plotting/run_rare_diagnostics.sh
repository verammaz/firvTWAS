#!/bin/bash
# Post-train rare-variant diagnostics (R² decomposition, μ/λ/MAF, T-gate comparison).
#
# 1) Post-pergene genome-wide (variant-level fast; R²+gate per chr via array)
# 2) Post-joint top200 (full diagnostics on fixed gene list)
#
# Outputs: {config}/diagnostics/post_pergene/ and post_joint/

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TRAIN_COMMON01_ROOT:-/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01}"
GENE_LIST="${GENE_LIST:-/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt}"
BASELINE="${BASELINE_ROOT:-/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01}"
CONFIG="${CONFIG:-full}"

PY="${PYTHON:-python3}"
if ! "$PY" -c "import matplotlib" 2>/dev/null; then
    source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate 2>/dev/null || true
    PY=python
fi

CONFIG_DIR="${CONFIG_DIR:-$ROOT/$CONFIG}"

echo "=== Post-pergene: variant-level diagnostics (all chr, pooled) ==="
"$PY" "$SCRIPT_DIR/diagnose_posttrain_rare.py" \
    --config_dir "$CONFIG_DIR" \
    --post_source post_pergene \
    --baseline_root "$BASELINE" \
    --skip_r2 \
    --skip_gate

echo "=== Post-joint top200: full diagnostics ==="
"$PY" "$SCRIPT_DIR/diagnose_posttrain_rare.py" \
    --config_dir "$CONFIG_DIR" \
    --post_source post_joint \
    --gene_list "$GENE_LIST" \
    --baseline_root "$BASELINE"

echo "Done. See $CONFIG_DIR/diagnostics/"

# After Slurm array (run_rare_diagnostics_chr_array.sh) finishes:
#   python merge_rare_diagnostics.py \
#     --chr_dirs_glob "$CONFIG_DIR/diagnostics/post_pergene/chr*" \
#     --out_dir "$CONFIG_DIR/diagnostics/post_pergene_merged"
