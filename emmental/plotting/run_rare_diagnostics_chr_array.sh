#!/bin/bash
# Slurm array: R² decomposition + T-gate diagnostics per chromosome (post-pergene).
#
#   sbatch --array=1-22 run_rare_diagnostics_chr_array.sh
#
# After all tasks finish:
#   python merge_rare_diagnostics.py \
#     --chr_dirs_glob .../diagnostics/post_pergene/chr* \
#     --out_dir .../diagnostics/post_pergene_merged

#SBATCH --job-name=rare_diag
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

PLOT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/plotting"
ROOT="${TRAIN_COMMON01_ROOT:-/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01}"
CONFIG="${CONFIG:-full}"
BASELINE="${BASELINE_ROOT:-/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01}"

CHR="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID or submit as array 1-22}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/$CONFIG}"
OUT="$CONFIG_DIR/diagnostics/post_pergene/chr${CHR}"
POST="$CONFIG_DIR/post_pergene/chr${CHR}"

source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
mkdir -p "$OUT"

echo "chr${CHR} diagnostics -> $OUT"
python "$PLOT_DIR/diagnose_posttrain_rare.py" \
    --config_dir "$CONFIG_DIR" \
    --post_source post_pergene \
    --post_dir "$POST" \
    --chromosome "$CHR" \
    --out_dir "$OUT" \
    --baseline_root "$BASELINE"

echo "Done chr${CHR}"
