#!/bin/bash

#SBATCH --job-name=annotate_${SET}
#SBATCH --output=logs/annotate/annotate_${SET}.chr%a.out
#SBATCH --error=logs/annotate/annotate_${SET}.chr%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-22

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

# Get chromosome number
CHRO_NB=$((SLURM_ARRAY_TASK_ID))

# Run script
scripts=/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts
python ${scripts}/map_annotate_variants.py -chrom ${CHRO_NB} --use_ref_alleles -set ${SET}
