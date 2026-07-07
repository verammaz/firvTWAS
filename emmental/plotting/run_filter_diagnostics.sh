#!/bin/bash
# Slurm: genome-wide T-gate / λ filter diagnostics (after joint + pergene training).
#
#   CONFIG_DIR=/path/to/experiment sbatch run_filter_diagnostics.sh
#
# Or multiple experiments:
#   EXPERIMENT_ROOTS="/path/a /path/b" sbatch run_filter_diagnostics.sh
#
# Writes under {experiment_root}/filter/:
#   - filter_summary_*.csv, per_gene_filter_counts_genome.csv, bar plots
#   - chr*/lambda_panel/{chr}_{ENSG}_lambda_panel.csv.gz  (λ + common/rare + T-gate flags)

#SBATCH --job-name=filter_diag
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

PLOT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/plotting"
CONFIG_DIR="${CONFIG_DIR:-/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01_normG_collapse}"

source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

if [[ -n "${EXPERIMENT_ROOTS:-}" ]]; then
  # shellcheck disable=SC2206
  ROOTS=( ${EXPERIMENT_ROOTS} )
else
  ROOTS=( "${CONFIG_DIR}" )
fi

for exp_root in "${ROOTS[@]}"; do
  echo "=== filter diagnostics (genome): ${exp_root} ==="
  python "${PLOT_DIR}/plot_filter_diagnostics.py" \
    --scope genome \
    --fit_source "${FIT_SOURCE:-auto}" \
    --experiment_roots "${exp_root}"
done

echo "Done."
