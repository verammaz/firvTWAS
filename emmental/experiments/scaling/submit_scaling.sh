#!/bin/bash

#SBATCH --job-name=scaling
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=50G  
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Save script arguments
SCRIPT_ARGS=("$@")

# Activate conda environment (clear arguments first to avoid passing them to activate)
set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -- "${SCRIPT_ARGS[@]}"

cd /gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src

python scale_annotations_global.py

