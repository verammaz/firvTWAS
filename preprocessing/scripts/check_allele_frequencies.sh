#!/bin/bash
# Submit SLURM job for allele frequency analysis
# Usage: ./submit_check_allele_frequencies.sh [SET]
#   SET: Train or Test (default: Train)

set -e

SET="${1:-Train}"

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
PYTHON_SCRIPT="${SCRIPT_DIR}/check_allele_frequencies.py"

# Create logs directory
mkdir -p logs/allele_freq

echo "=========================================="
echo "Submitting allele frequency analysis job"
echo "=========================================="
echo "Set: ${SET}"
echo ""

# Submit SLURM job
JOB_ID=$(sbatch \
                --job-name="allele_freq_${SET}" \
                --output="logs/allele_freq/allele_freq_${SET}.%j.out" \
                --error="logs/allele_freq/allele_freq_${SET}.%j.err" \
                --time=4:00:00 \
                --mem=200G \
                --cpus-per-task=16 \
                --wrap="python3 ${PYTHON_SCRIPT} ${SET}" | grep -oP '\d+')

if [ -n "${JOB_ID}" ]; then
    echo "  Job ID: ${JOB_ID}"
    echo ""
    echo "=========================================="
    echo "Job submission complete!"
    echo "=========================================="
    echo ""
    echo "Monitor job with: squeue -j ${JOB_ID}"
    echo "Check logs in: logs/allele_freq/"
    echo "Results will be in: Processed/${SET}/chroms/allele_freq_analysis/"
else
    echo "Error: Failed to submit job"
    exit 1
fi



