#!/bin/bash
# Wrapper script to submit SLURM jobs for mapping and annotating variants   
# Usage: ./submit_annotate_jobs.sh

set -e

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
GENOTYPE_SCRIPT="${SCRIPT_DIR}/genotype.sh"

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting annotate jobs"
echo "=========================================="

for SET in "Train" "Test"; do
    
    echo "Submitting job for set: ${SET}"
    sbatch --export=SET="${SET}" \
    --job-name="genotype__mat_${SET}" \
    --output="logs/genotype/genotype_mat_${SET}.chr%a.out" \
    --error="logs/genotype/genotype_mat_${SET}.chr%a.err" \
    --array=1-22 \
    --time=8:00:00 \
    --mem=100G \
    --cpus-per-task=8 \
    ${GENOTYPE_SCRIPT}
done