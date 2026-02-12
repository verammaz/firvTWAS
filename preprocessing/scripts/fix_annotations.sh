#!/bin/bash
#SBATCH --output=slurm/logs/fix_anno/fix_anno_chr%a.out
#SBATCH --error=slurm/logs/fix_anno/fix_anno_chr%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --array=1-22

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

# Get chromosome number
CHRO_NB=$((SLURM_ARRAY_TASK_ID))

# Run script
scripts=/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts
python ${scripts}/fix_annotations.py ${CHRO_NB}
