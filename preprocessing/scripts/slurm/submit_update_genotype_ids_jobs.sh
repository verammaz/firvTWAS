#!/bin/bash
# Wrapper script to submit SLURM jobs for updating genotype IDs for all cohorts
# Usage: ./submit_update_genotype_ids_jobs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
UPDATE_SCRIPT="${SCRIPT_DIR}/update_genotype_ids_cohort.sh"
REMAPPING_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Genotypes/remapping"

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting genotype ID update jobs"
echo "=========================================="
echo ""

# Find all update_ids files
MAPPING_FILES=("${REMAPPING_DIR}"/*_update_ids.txt)

if [ ${#MAPPING_FILES[@]} -eq 0 ] || [ ! -f "${MAPPING_FILES[0]}" ]; then
    echo "ERROR: No mapping files found in ${REMAPPING_DIR}"
    echo "Please run rename_genotype_mapping.py first"
    exit 1
fi

echo "Found ${#MAPPING_FILES[@]} mapping files:"
COHORTS=()
for mapping_file in "${MAPPING_FILES[@]}"; do
    cohort=$(basename "${mapping_file}" _update_ids.txt)
    COHORTS+=("${cohort}")
    echo "  - ${cohort}"
done
echo ""

# Submit jobs for each cohort
JOB_IDS=()
for cohort in "${COHORTS[@]}"; do
  
    echo "Submitting job for cohort: ${cohort}"
    
    # Submit SLURM job
    JOB_ID=$(sbatch --export=COHORT="${cohort}" \
                    --job-name="update_ids_${cohort}" \
                    --output="logs/id/update_genotype_ids_${cohort}_chr%a.out" \
                    --error="logs/id/update_genotype_ids_${cohort}_chr%a.err" \
                    --time=10:00:00 \
                    --mem=200G \
                    --cpus-per-task=4 \
                    --array=1-22 \
                    --cpus-per-task=4 \
                    "${UPDATE_SCRIPT}" | grep -oP '\d+')
    
    if [ -n "${JOB_ID}" ]; then
        JOB_IDS+=("${JOB_ID}")
        echo "  Job ID: ${JOB_ID}"
    else
        echo "  ERROR: Failed to submit job"
    fi
    echo ""
done

echo "=========================================="
echo "Job submission complete!"
echo "=========================================="
echo "Submitted ${#JOB_IDS[@]} jobs:"
for job_id in "${JOB_IDS[@]}"; do
    echo "  Job ID: ${job_id}"
done
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"

