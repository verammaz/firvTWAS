#!/bin/bash
# Wrapper script to submit SLURM jobs for mapping and annotating variants   
# Usage: ./submit_annotate_jobs.sh

set -e

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
ANNOTATE_SCRIPT="${SCRIPT_DIR}/annotate.sh"

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting annotate jobs"
echo "=========================================="

for SET in "Train" "Test"; do
    echo "Submitting job for set: ${SET}"
    sbatch --export=SET="${SET}" \
    --job-name="annotate_${SET}" \
    --output="logs/annotate/annotate_${SET}.chr%a.out" \
    --error="logs/annotate/annotate_${SET}.chr%a.err" \
    --array=1-22 \
    --time=10:00:00 \
    --mem=100G \
    --cpus-per-task=8 \
    ${ANNOTATE_SCRIPT}
done