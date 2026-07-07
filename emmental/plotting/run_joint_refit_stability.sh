#!/bin/bash
# Beta + R² stability across joint refits for train_common01* (excludes tau1norm).
#
#   sbatch run_joint_refit_stability.sh

#SBATCH --job-name=refit-stab
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -euo pipefail

PLOT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/plotting"
MYOUT="/gpfs/commons/home/vmazeeva/firvTWAS_myout"
SUMMARY="${MYOUT}/train_common01/stability_report/cross_config_summary.csv"

python "$PLOT_DIR/joint_refit_stability.py" \
    --discover_all \
    --myout_root "$MYOUT" \
    --cross_summary "$SUMMARY"

echo "Done. Per-experiment reports: */joint/stability_report/"
echo "Combined summary: $SUMMARY"
