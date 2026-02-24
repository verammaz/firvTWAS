#!/bin/bash

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

# Get chromosome number
CHRO_NB=$((SLURM_ARRAY_TASK_ID))

# Run script
scripts=/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts
python -u ${scripts}/map_annotate_variants.py -chrom ${CHRO_NB} --use_ref_alleles
