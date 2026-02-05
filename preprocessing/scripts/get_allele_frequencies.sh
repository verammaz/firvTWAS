#!/bin/bash
#SBATCH --job-name=get_freq_${SET}
#SBATCH --output=logs/allele_freq/get_freq_${SET}_chr%a.out
#SBATCH --error=logs/allele_freq/get_freq_${SET}_chr%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-22

# Script to get A1/A2 allele frequencies between merged files and original cohort files
# Usage: sbatch --export=SET=<set> get_allele_frequencies.sh

set -euo pipefail

# Check if SET is set
if [ -z "${SET:-}" ]; then
    echo "ERROR: SET environment variable not set"
    echo "Usage: sbatch --export=SET=<set> get_allele_frequencies.sh"
    exit 1
fi

# Define cohorts for set
if [ "$SET" == "Train" ]; then
    COHORTS=("AnswerALS" "Mayo" "MSBB" "NYGC" "ROSMAP" "GTEX")
elif [ "$SET" == "Test" ]; then
    COHORTS=("ROSMAP_DLPFC")
else
    echo "ERROR: Invalid set: ${SET}"
    exit 1
fi

# Ensure PATH includes standard directories
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Load PLINK module
source /etc/profile.d/modules.sh 2>/dev/null || true
module load PLINK/1.9 2>/dev/null
command -v plink >/dev/null || { echo "ERROR: plink not found"; exit 1; }

CHR=${SLURM_ARRAY_TASK_ID}
CHROM_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Genotypes/final"
FINAL_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/${SET}/chroms"
OUTPUT_DIR="${FINAL_DIR}/freq_checks"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Getting Allele Frequencies"
echo "=========================================="
echo "Set: ${SET}"
echo "Chromosome: ${CHR}"
echo "Date: $(date)"
echo ""

# Step 1: Calculate frequencies for merged file
echo "Step 1: Calculating frequencies for merged file..."
MERGED_PREFIX="${FINAL_DIR}/merged_chr${CHR}"
MERGED_FREQ="${OUTPUT_DIR}/merged_chr${CHR}.frq"

if [ ! -f "${MERGED_PREFIX}.bed" ]; then
    echo "ERROR: Merged file not found: ${MERGED_PREFIX}.bed"
    exit 1
fi

plink --bfile "${MERGED_PREFIX}" \
      --freq \
      --allow-no-sex \
      --out "${OUTPUT_DIR}/merged_chr${CHR}" \
      >/dev/null 2>&1

if [ ! -f "${MERGED_FREQ}" ]; then
    echo "ERROR: Failed to generate frequency file for merged data"
    exit 1
fi

echo "  ✓ Calculated frequencies for merged file: $(wc -l < "${MERGED_FREQ}") variants"

# Step 2: Calculate frequencies for each original cohort
echo ""
echo "Step 2: Calculating frequencies for original cohort files..."
COHORT_FREQ_FILES=()
for cohort in "${COHORTS[@]}"; do
    COHORT_PREFIX="${CHROM_DIR}/${cohort}_final_chr${CHR}"
    COHORT_FREQ="${OUTPUT_DIR}/${cohort}_chr${CHR}.frq"
    
    if [ -f "${COHORT_PREFIX}.bed" ]; then
        plink --bfile "${COHORT_PREFIX}" \
              --freq \
              --allow-no-sex \
              --out "${OUTPUT_DIR}/${cohort}_chr${CHR}" \
              >/dev/null 2>&1
        
        if [ -f "${COHORT_FREQ}" ]; then
            COHORT_FREQ_FILES+=("${COHORT_FREQ}")
            echo "  ✓ ${cohort}: $(wc -l < "${COHORT_FREQ}") variants"
        else
            echo "  ⚠ ${cohort}: Failed to generate frequency file"
        fi
    else
        echo "  ⚠ ${cohort}: File not found (${COHORT_PREFIX}.bed)"
    fi
done

echo ""
echo "=========================================="
echo "Frequency Get Complete"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""

