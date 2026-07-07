#!/bin/bash
# Expression tail analysis: common PRS residuals → rare recovery (tables only).
#
# Example (top-200, test split; default common+rare decoupled posteriors):
#   CONFIG_DIR=/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01_normG_collapse_variants \
#   POST_DIR=$CONFIG_DIR/post_pergene \
#   GENE_LIST=/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt \
#   bash run_expression_tails.sh
#
# Alternate beta sources (match post-train R² grid):
#   COMMON_BETA=train RARE_BETA=full OUT_DIR=$CONFIG_DIR/tails/train_full \
#   bash run_expression_tails.sh
#
# Genome-wide per chr (Slurm array):
#   sbatch --array=1-22 run_expression_tails.sh
#   MERGE=1 OUT_DIR=... bash run_expression_tails.sh  # merge chr*/ into parent
#
# Plotting: use expr_tails_plot.py from expr_tails_plots.ipynb (no plots from this job).

#SBATCH --job-name=expr-tails
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/plotting"
CONFIG_DIR="${CONFIG_DIR:-/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01_normG_collapse}"
POST_DIR="${POST_DIR:-${CONFIG_DIR}/post_pergene}"
OUT_DIR="${OUT_DIR:-${CONFIG_DIR}/tails}"
GENE_LIST="${GENE_LIST:-}"
CHROMOSOME="${CHROMOSOME:-${SLURM_ARRAY_TASK_ID:-}}"
RESID_TAIL_Q="${RESID_TAIL_Q:-0.05}"
EXPR_TAIL_Q="${EXPR_TAIL_Q:-0.01}"
COMMON_BETA="${COMMON_BETA:-common}"
RARE_BETA="${RARE_BETA:-rare}"
SPLITS="${SPLITS:-test}"
MERGE="${MERGE:-0}"

PY="${PYTHON:-/gpfs/commons/groups/knowles_lab/software/anaconda3/bin/python}"
if ! "$PY" -c "import matplotlib, pyro" 2>/dev/null; then
    source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate 2>/dev/null || true
    PY=/gpfs/commons/groups/knowles_lab/software/anaconda3/bin/python
fi

if [[ "$MERGE" == "1" ]]; then
    echo "Merging chr subdirs under $OUT_DIR"
    "$PY" "$SCRIPT_DIR/merge_expression_tails.py" --root "$OUT_DIR" --splits "$SPLITS"
    exit 0
fi

POST_DIR_ABS="$(cd "$(dirname "$POST_DIR")" && pwd)/$(basename "$POST_DIR")"
OUT_DIR_ABS="$OUT_DIR"
if [[ -n "$CHROMOSOME" && "$OUT_DIR_ABS" != */chr"${CHROMOSOME}" ]]; then
    OUT_DIR_ABS="${OUT_DIR_ABS}/chr${CHROMOSOME}"
fi
mkdir -p "$OUT_DIR_ABS"

args=(
    --config_dir "$CONFIG_DIR"
    --post_dir "$POST_DIR_ABS"
    --out_dir "$OUT_DIR_ABS"
    --resid_tail_quantile "$RESID_TAIL_Q"
    --expr_tail_quantile "$EXPR_TAIL_Q"
    --common_beta "$COMMON_BETA"
    --rare_beta "$RARE_BETA"
    --splits "$SPLITS"
)
[[ -n "$GENE_LIST" ]] && args+=(--gene_list "$GENE_LIST")
[[ -n "$CHROMOSOME" ]] && args+=(--chromosome "$CHROMOSOME")

echo "Config:  $CONFIG_DIR"
echo "Post:    $POST_DIR_ABS"
echo "Out:     $OUT_DIR_ABS"
[[ -n "$CHROMOSOME" ]] && echo "Chr:     $CHROMOSOME"

"$PY" "$SCRIPT_DIR/analyze_expression_tails.py" "${args[@]}"
