#!/bin/bash
# |lin1| gate diagnostics for train_common01* full-model joint runs.
#
# Submit:
#   sbatch run_lin1_gate_train_common01.sh
#
# Remaining normG-only (after partial local run):
#   sbatch --export=ONLY=normG run_lin1_gate_train_common01.sh

#SBATCH --job-name=lin1-gate
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

PLOT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/plotting"
PLOT_PY="${PLOT_DIR}/plot_lin1_gate_train_common01.py"
MYOUT_ROOT="${MYOUT_ROOT:-/gpfs/commons/home/vmazeeva/firvTWAS_myout}"
OUT_DIR="${OUT_DIR:-${MYOUT_ROOT}/plots/lin1_gate_train_common01}"

set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -euo pipefail

PY=python

py_args=(
    "$PLOT_PY"
    --myout_root "$MYOUT_ROOT"
    --out_dir "$OUT_DIR"
    --skip-existing
)

if [[ -n "${ONLY:-}" ]]; then
    # ONLY may be a space-separated list of substrings, e.g. ONLY="normG_fixed normG_full"
    for needle in $ONLY; do
        py_args+=(--only "$needle")
    done
fi

echo "Running: $PY ${py_args[*]}"
"$PY" "${py_args[@]}"

echo "Done. Plots in $OUT_DIR"
