#!/bin/bash
# Parameter recovery: simulate y, refit full + collapsed joint, compare to truth.
#
#   sbatch run_param_recovery.sh
#   sbatch run_param_recovery.sh --output_dir /path/to/out --simulation_seed 0
#   sbatch --export=ALL,CONFIG=config_simulation_200.yaml run_param_recovery.sh

#SBATCH --job-name=param-recovery
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

SCRIPT_ARGS=("$@")
set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -- "${SCRIPT_ARGS[@]}"

SRC="/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src"
cd "$SRC"

config_file="${CONFIG:-config_simulation_200.yaml}"
output_dir="${OUTPUT_DIR:-/gpfs/commons/home/vmazeeva/firvTWAS_myout/param_recovery/200genes/seed1}"
simulation_seed="${SIMULATION_SEED:-1}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            config_file="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --simulation_seed)
            simulation_seed="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$output_dir")"

echo "Config: $config_file"
echo "Output: $output_dir"
echo "Simulation seed: $simulation_seed"

python -u param_recovery.py \
    --config "$config_file" \
    --output_dir "$output_dir" \
    --simulation_seed "$simulation_seed" \
    --run_full true \
    --run_collapsed true

echo "Done. Results in $output_dir"
