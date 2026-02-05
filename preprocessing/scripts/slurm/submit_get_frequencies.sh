#!/bin/bash
# Wrapper script to submit SLURM jobs for getting allele frequencies
# Usage: ./submit_check_frequencies.sh [SET]
#   SET: Train or Test (default: Train)

set -e

SET="${1:-Train}"

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
GET_FREQ_SCRIPT="${SCRIPT_DIR}/get_allele_frequencies.sh"

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting get allele frequency jobs"
echo "=========================================="
echo "Set: ${SET}"
echo ""

# Submit SLURM job
JOB_ID=$(sbatch --export=SET="${SET}" \
                --job-name="get_freq_${SET}" \
                --output="logs/allele_freq/get_freq_${SET}_chr%a.out" \
                --error="logs/allele_freq/get_freq_${SET}_chr%a.err" \
                --time=2:00:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --array=1-22 \
                "${GET_FREQ_SCRIPT}" | grep -oP '\d+')

if [ -n "${JOB_ID}" ]; then
    echo "  Job ID: ${JOB_ID}"
    echo ""
    echo "=========================================="
    echo "Job submission complete!"
    echo "=========================================="
    echo ""
    echo "Monitor job with: squeue -j ${JOB_ID}"
    echo "Check logs in: logs/"
else
    echo "Error: Failed to submit job"
    exit 1
fi

