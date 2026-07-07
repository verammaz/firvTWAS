#!/bin/bash
# λ distribution plots for train_common01* (excludes tau1norm).
#   sbatch run_lambda_distribution.sh

#SBATCH --job-name=lambda-dist
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -euo pipefail

PLOT_PY="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/plotting/plot_lambda_distribution.py"
MYOUT="/gpfs/commons/home/vmazeeva/firvTWAS_myout"

python "$PLOT_PY" --myout_root "$MYOUT" --skip-existing
echo "Done: $MYOUT/plots/lambda_train_common01/"
