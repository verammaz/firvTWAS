#!/bin/bash
# Wrapper script to submit SLURM jobs for subsetting genotype files for all cohorts

set -e

SCRIPT_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts"
SUBSET_SCRIPT="${SCRIPT_DIR}/subset_genotype_cohort.sh"
GENOTYPE_SUBSET_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Genotypes/subset"

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting genotype subsetting jobs"
echo "=========================================="
echo ""

# Find all keep files
KEEP_FILES=("${GENOTYPE_SUBSET_DIR}"/*_keep_participants.txt)

if [ ${#KEEP_FILES[@]} -eq 0 ] || [ ! -f "${KEEP_FILES[0]}" ]; then
    echo "ERROR: No keep files found in ${GENOTYPE_SUBSET_DIR}"
    echo "Please run extract_common_participants.py first"
    exit 1
fi

echo "Found ${#KEEP_FILES[@]} keep files:"
for keep_file in "${KEEP_FILES[@]}"; do
    cohort=$(basename "${keep_file}" _keep_participants.txt)
    echo "  - ${cohort}"
done
echo ""

# Extract cohort names and submit jobs
JOB_IDS=()
for keep_file in "${KEEP_FILES[@]}"; do
    cohort=$(basename "${keep_file}" _keep_participants.txt)
    
    echo "Submitting job for cohort: ${cohort}"
    
    # Submit SLURM job
    JOB_ID=$(sbatch --export=COHORT="${cohort}" \
                    --job-name="subset_geno_${cohort}" \
                    --output="logs/subset/subset_genotype_${cohort}.out" \
                    --error="logs/subset/subset_genotype_${cohort}.err" \
                    --time=4:00:00 \
                    --mem=16G \
                    --cpus-per-task=4 \
                    "${SUBSET_SCRIPT}" | grep -oP '\d+')
    
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

