#!/bin/bash
# Wrapper script to submit SLURM jobs for mapping and annotating variants   
# Usage: ./submit_annotate_jobs.sh

set -e

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
ANNOTATE_SCRIPT="${SCRIPT_DIR}/annotate.sh"


echo "=========================================="
echo "Submitting annotate jobs"
echo "=========================================="

JOB_ID=$(sbatch --job-name="annotate" \
    --output="logs/annotate/annotate_chr%a.out" \
    --error="logs/annotate/annotate_chr%a.err" \
    --array=21 \
    --time=10:00:00 \
    --mem=100G \
    --cpus-per-task=8 \
    "${ANNOTATE_SCRIPT}" | grep -oP '\d+')
echo "  Job ID: ${JOB_ID}"