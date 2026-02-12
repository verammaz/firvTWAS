#!/bin/bash
# Wrapper script to submit SLURM jobs for merging all cohorts by chromosome
# Usage: ./submit_merge_by_chromosome_jobs.sh

set -e

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_cohorts_by_chromosome.sh"

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting chromosome merging jobs"
echo "=========================================="
echo ""

# Submit SLURM job
JOB_ID=$(sbatch --job-name="merge" \
                --output="logs/merge/merge_chr%a.out" \
                --error="logs/merge/merge_chr%a.err" \
                --time=6:00:00 \
                --mem=256G \
                --cpus-per-task=8 \
                --array=1-22 \
                "${MERGE_SCRIPT}" | grep -oP '\d+')
echo "  Job ID: ${JOB_ID}"



echo "=========================================="
echo "Job submission complete!"
echo "=========================================="

echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"

