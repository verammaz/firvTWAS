#!/bin/bash
# Wrapper script to submit SLURM jobs for mapping and annotating variants   
# Usage: ./submit_annotate_jobs.sh

set -e

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
GENOTYPE_SCRIPT="${SCRIPT_DIR}/genotype.sh"

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting genotype jobs"
echo "=========================================="

JOB_ID=$(sbatch --job-name="genotype" \
    --output="logs/genotype/genotype_chr%a.out" \
    --error="logs/genotype/genotype_chr%a.err" \
    --array=1-22\
    --time=20:00:00 \
    --mem=100G \
    --cpus-per-task=8 \
    "${GENOTYPE_SCRIPT}" | grep -oP '\d+')
echo "  Job ID: ${JOB_ID}"