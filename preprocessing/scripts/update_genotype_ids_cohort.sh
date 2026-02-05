#!/bin/bash
#SBATCH --job-name=update_ids_${COHORT}
#SBATCH --output=logs/id/update_genotype_ids_${COHORT}_chr%a.out
#SBATCH --error=logs/id/update_genotype_ids_${COHORT}_chr%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-22

set -euo pipefail

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# -------------------------
# Config & paths
# -------------------------
if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set"
    exit 1
fi

BASE_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain"
GENO_DIR="${BASE_DIR}/Genotypes"
SUBSET_DIR="${GENO_DIR}/subsetted"
REMAP_DIR="${GENO_DIR}/remapping"
FINAL_DIR="${GENO_DIR}/final"
mkdir -p "${FINAL_DIR}" logs

CHR=${SLURM_ARRAY_TASK_ID}
INPUT_BASE="${SUBSET_DIR}/${COHORT}_subset_chr${CHR}"
OUTPUT_BASE="${FINAL_DIR}/${COHORT}_final_chr${CHR}"
MAPPING_FILE="${REMAP_DIR}/${COHORT}_update_ids.txt"

# Verify input files exist (PLINK 1.9 format)
if [ ! -f "${INPUT_BASE}.bed" ] || [ ! -f "${INPUT_BASE}.bim" ] || [ ! -f "${INPUT_BASE}.fam" ]; then
    echo "ERROR: Input files not found: ${INPUT_BASE}.{bed,bim,fam}"
    exit 1
fi

# Load PLINK module (initialize module system first)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load PLINK/1.9 2>/dev/null
command -v plink >/dev/null || { echo "ERROR: plink not found"; exit 1; }

echo "Chromosome ${CHR} - Cohort ${COHORT}"
echo "Input: ${INPUT_BASE}"
echo "Output: ${OUTPUT_BASE}"
echo "Date: $(date)"

TEMP_DIR=$(mktemp -d -t genotype_update_${COHORT}_chr${CHR}_XXXXXX)
PER_SAMPLE_DIR="${TEMP_DIR}/per_sample"
mkdir -p "${PER_SAMPLE_DIR}"

# Initialize multiallelic variant count
MULTIALLELIC_COUNT=0

# -------------------------
# Step 0: filter mapping per chromosome
# -------------------------
echo "Filtering mapping file to samples present in chr${CHR} .fam..."
# Extract IID column (field 2 in .fam file)
# Use awk to handle both space and tab delimiters automatically
awk "{print \$2}" "${INPUT_BASE}.fam" | sort > "${TEMP_DIR}/fam.iids"

# Match mapping file based on IID (field 1 in mapping file: OLD_IID NEW_IID)
# awk handles both tab and space delimiters automatically
awk "NR==FNR { fam[\$1]=1; next } \$1 in fam" "${TEMP_DIR}/fam.iids" "${MAPPING_FILE}" > "${TEMP_DIR}/mapping.filtered.txt"

MAPPING_FILE="${TEMP_DIR}/mapping.filtered.txt"
N_SAMPLES=$(wc -l < "${MAPPING_FILE}")
echo "  Samples in chr fam: $(wc -l < "${TEMP_DIR}/fam.iids")"
echo "  Samples after filter: ${N_SAMPLES}"
if [ "${N_SAMPLES}" -eq 0 ]; then
    echo "ERROR: No samples left after filtering"
    exit 1
fi

# -------------------------
# Step 1: per-sample extraction (two-step)
# -------------------------
echo "Creating per-sample genotype files..."

i=0
while read -r OLD_IID NEW_IID; do
    i=$((i+1))
    OUT="${PER_SAMPLE_DIR}/${NEW_IID}"

    # ---------------------
    # Step 1a: extract sample (keep OLD IDs)
    # ---------------------
    KEEP_FILE="${TEMP_DIR}/keep_${NEW_IID}.txt"
    # Use 0 for FID since it is not used (PSAM files do not have meaningful FID)
    echo -e "0\t${OLD_IID}" > "${KEEP_FILE}"

    plink \
      --bfile "${INPUT_BASE}" \
      --keep "${KEEP_FILE}" \
      --make-bed \
      --out "${OUT}_step1" \
      >/dev/null

    # ---------------------
    # Step 1b: rename IDs (update IDs)
    # ---------------------
    UPDATE_FILE="${TEMP_DIR}/update_${NEW_IID}.txt"
    # Use 0 for FID since it is not used
    echo -e "0\t${OLD_IID}\t0\t${NEW_IID}" > "${UPDATE_FILE}"

    plink \
      --bfile "${OUT}_step1" \
      --update-ids "${UPDATE_FILE}" \
      --keep-allele-order \
      --make-bed \
      --out "${OUT}" \
      >/dev/null

    # Verify success
    if [ ! -f "${OUT}.bed" ]; then
        echo "ERROR: Failed to create sample ${NEW_IID}"
        exit 1
    fi

    # Optional cleanup (including PLINK log files to prevent accumulation)
    rm -f "${KEEP_FILE}" "${UPDATE_FILE}" "${OUT}_step1".{bed,bim,fam,log} "${OUT}".log

    if (( i % 50 == 0 )); then
        echo "  Created ${i}/${N_SAMPLES} samples..."
    fi
done < "${MAPPING_FILE}"

echo "  ✔ Per-sample files created"

# -------------------------
# Step 2: merge per-sample files
# -------------------------
echo "Merging per-sample files..."
ls -lh ${PER_SAMPLE_DIR}/*.bed | wc -l

ls "${PER_SAMPLE_DIR}"/*.bed | sed "s/\.bed\$//" > "${TEMP_DIR}/merge_list.txt"
FIRST=$(head -n 1 "${TEMP_DIR}/merge_list.txt")
tail -n +2 "${TEMP_DIR}/merge_list.txt" > "${TEMP_DIR}/merge_list_tail.txt"

# Attempt merge using PLINK 1.9 --merge-list
# If merge fails due to multiallelic variants, we will remove them and retry
MULTIALLELIC_REMOVED=0
MERGE_ATTEMPT=1
MAX_MERGE_ATTEMPTS=2

while [ ${MERGE_ATTEMPT} -le ${MAX_MERGE_ATTEMPTS} ]; do
    echo "Merge attempt ${MERGE_ATTEMPT}..."
    
    if plink \
        --bfile "${FIRST}" \
        --merge-list "${TEMP_DIR}/merge_list_tail.txt" \
        --keep-allele-order \
        --allow-no-sex \
        --make-bed \
        --out "${OUTPUT_BASE}" 2>&1 | tee "${TEMP_DIR}/merge_attempt.log"; then
                # Merge succeeded
                echo "  ✔ Merge successful"
                break
    else
        # Merge failed - check if it is due to multiallelic variants
        if grep -q "variants with 3+ alleles present" "${TEMP_DIR}/merge_attempt.log"; then
            MULTIALLELIC_COUNT=$(grep "Error:" "${TEMP_DIR}/merge_attempt.log" | sed -n "s/.*Error: \\([0-9]*\\) variants with 3\\+ alleles.*/\\1/p")
            if [ -z "${MULTIALLELIC_COUNT}" ]; then
                MULTIALLELIC_COUNT=0
            fi
            echo "  Merge failed due to ${MULTIALLELIC_COUNT} multiallelic variants (3+ alleles)"
            
            if [ ${MERGE_ATTEMPT} -eq 1 ]; then
                # First attempt failed - try removing multiallelic variants
                echo "  Removing multiallelic variants and retrying merge..."
                
                # Check for .missnp file created by PLINK
                MISSNP_FILE="${OUTPUT_BASE}-merge.missnp"
                if [ -f "${MISSNP_FILE}" ]; then
                    echo "  Found .missnp file with multiallelic variants: $(wc -l < "${MISSNP_FILE}") variants"
                    MULTIALLELIC_REMOVED=$(wc -l < "${MISSNP_FILE}")
                    
                    # Remove multiallelic variants from all per-sample files
                    echo "  Removing multiallelic variants from all per-sample files..."
                    for sample_file in "${PER_SAMPLE_DIR}"/*.bed; do
                        sample_base="${sample_file%.bed}"
                        if ! plink \
                          --bfile "${sample_base}" \
                          --exclude "${MISSNP_FILE}" \
                          --make-bed \
                          --out "${sample_base}_filtered" \
                          >/dev/null 2>&1; then
                            echo "  WARNING: Failed to exclude variants from ${sample_base}"
                            echo "  This may be due to variant ID mismatch. Continuing anyway..."
                        fi
                        
                        # Replace original with filtered version if it exists
                        if [ -f "${sample_base}_filtered.bed" ]; then
                            mv "${sample_base}_filtered.bed" "${sample_base}.bed"
                            mv "${sample_base}_filtered.bim" "${sample_base}.bim"
                            mv "${sample_base}_filtered.fam" "${sample_base}.fam"
                        fi
                    done
                    
                    # Recreate merge list after filtering
                    ls "${PER_SAMPLE_DIR}"/*.bed | sed "s/\.bed\$//" > "${TEMP_DIR}/merge_list.txt"
                    FIRST=$(head -n 1 "${TEMP_DIR}/merge_list.txt")
                    tail -n +2 "${TEMP_DIR}/merge_list.txt" > "${TEMP_DIR}/merge_list_tail.txt"
                    
                    echo "  Removed ${MULTIALLELIC_REMOVED} multiallelic variants. Retrying merge..."
                    MERGE_ATTEMPT=$((MERGE_ATTEMPT + 1))
                else
                    # No .missnp file - try to identify multiallelic variants by position
                    echo "  WARNING: No .missnp file found. Cannot automatically remove multiallelic variants."
                    echo "ERROR: Merge failed. Check ${TEMP_DIR}/merge_attempt.log for details."
                    exit 1
                fi
            else
                # Second attempt also failed - give up
                echo "ERROR: Merge failed after removing multiallelic variants. Check ${TEMP_DIR}/merge_attempt.log for details."
                exit 1
            fi
        else
            # Merge failed for a different reason
    echo "ERROR: Merge failed. Check ${TEMP_DIR}/merge_attempt.log for details."
            exit 1
        fi
    fi
done

if [ ${MERGE_ATTEMPT} -gt ${MAX_MERGE_ATTEMPTS} ]; then
    echo "ERROR: Merge failed after ${MAX_MERGE_ATTEMPTS} attempts."
    exit 1
fi

# Report multiallelic variant handling
if [ ${MULTIALLELIC_REMOVED} -gt 0 ]; then
    echo "  Multiallelic variants removed: ${MULTIALLELIC_REMOVED}"
else
    # Check for multiallelic variants that were split (not removed)
MULTIALLELIC_COUNT=0
if [ -f "${OUTPUT_BASE}-merge.missnp" ]; then
    MULTIALLELIC_COUNT=$(wc -l < "${OUTPUT_BASE}-merge.missnp")
    echo "  Multiallelic variants written to .missnp file: ${MULTIALLELIC_COUNT}"
else
    # Count positions that appear multiple times (indicates multiallelic split)
        MULTIALLELIC_POSITIONS=$(cut -f1,4 "${OUTPUT_BASE}.bim" 2>/dev/null | sort | uniq -d | wc -l)
    if [ "${MULTIALLELIC_POSITIONS}" -gt 0 ]; then
        MULTIALLELIC_COUNT="${MULTIALLELIC_POSITIONS}"
        echo "  Multiallelic variants detected (split into pseudo-biallelic): ${MULTIALLELIC_COUNT} positions"
    else
        echo "  No multiallelic variants detected"
        fi
    fi
fi

# -------------------------
# Step 3: verification
# -------------------------
OUT_FAM="${OUTPUT_BASE}.fam"
N_OUT=$(wc -l < "${OUT_FAM}")
echo "Output samples: ${N_OUT}"

# Check uniqueness (awk handles both tab and space delimiters)
DUPS=$(awk "{print \$1, \$2}" "${OUT_FAM}" | sort | uniq -d | wc -l)
if [ "${DUPS}" -ne 0 ]; then
    echo "ERROR: duplicate sample IDs in output!"
    exit 1
fi


# Clean up temporary merge files
rm -f "${TEMP_DIR}/merge_attempt.log" 2>/dev/null || true

echo "Chromosome ${CHR} completed successfully."
echo "Date: $(date)"

# -------------------------
# Step 4: summary
# -------------------------
echo ""
echo "=========================================="
echo "Chromosome ${CHR} summary for cohort ${COHORT}"
echo "=========================================="

# Original sample count
ORIG_N=$(wc -l < "${INPUT_BASE}.fam")
echo "  Original number of samples      : ${ORIG_N}"

# Total number of participants in original .fam (awk handles both tab and space delimiters)
TOTAL_PARTICIPANTS=$(awk "{print \$2}" "${INPUT_BASE}.fam" | sort | uniq | wc -l)
echo "  Total participants              : ${TOTAL_PARTICIPANTS}"

# Number of samples after filtering / duplication
NEW_N=$(wc -l < "${MAPPING_FILE}")
echo "  Number of samples after duplication: ${NEW_N}"

# Participants with multiple samples (awk handles both tab and space delimiters)
TEMP_FILE="${TEMP_DIR}/multi_parts_counts.txt"
awk "{print \$1}" "${MAPPING_FILE}" | sort | uniq -c > "${TEMP_FILE}"
MULTI_PARTS=$(awk "\$1>1 {count++} END {print count+0}" "${TEMP_FILE}")
echo "  Number of participants with multiple samples: ${MULTI_PARTS}"

# Output file check
OUT_FAM="${OUTPUT_BASE}.fam"
N_OUT=$(wc -l < "${OUT_FAM}")
echo "  Number of samples in final merged output    : ${N_OUT}"

# Check uniqueness (awk handles both tab and space delimiters)
DUPS=$(awk "{print \$1, \$2}" "${OUT_FAM}" | sort | uniq -d | wc -l)
if [ "${DUPS}" -ne 0 ]; then
    echo "  WARNING: duplicate sample IDs in output!"
else
    echo "  Sample IDs are unique in final output"
fi

# Count variants
ORIG_VARIANTS=$(wc -l < "${INPUT_BASE}.bim")
FINAL_VARIANTS=$(wc -l < "${OUTPUT_BASE}.bim")
echo "  Variants in input: ${ORIG_VARIANTS}"
echo "  Variants in final output: ${FINAL_VARIANTS}"

# Detect multiallelic variants that were split (PLINK 1.9 splits multiallelic into pseudo-biallelic rows)
# Count unique positions in input vs output
# If output has more variants than input, some multiallelic variants were likely split
ORIG_POSITIONS=$(cut -f1,4 "${INPUT_BASE}.bim" | sort -u | wc -l)
FINAL_POSITIONS=$(cut -f1,4 "${OUTPUT_BASE}.bim" | sort -u | wc -l)
VARIANTS_ADDED=$((FINAL_VARIANTS - ORIG_VARIANTS))
POSITIONS_ADDED=$((FINAL_POSITIONS - ORIG_POSITIONS))

if [ ${MULTIALLELIC_REMOVED} -gt 0 ]; then
    echo "  Multiallelic variants removed: ${MULTIALLELIC_REMOVED}"
    echo "  (Variants removed to allow merge to succeed)"
    echo "  (Variant count decreased from ${ORIG_VARIANTS} to ${FINAL_VARIANTS} due to multiallelic removal)"
elif [ "${VARIANTS_ADDED}" -gt 0 ]; then
    echo "  Note: ${VARIANTS_ADDED} additional variant rows created (likely from multiallelic variants split into pseudo-biallelic)"
    echo "  Unique positions in input: ${ORIG_POSITIONS}"
    echo "  Unique positions in output: ${FINAL_POSITIONS}"
    if [ "${POSITIONS_ADDED}" -gt 0 ]; then
        echo "  New positions added: ${POSITIONS_ADDED}"
    fi
else
    echo "  No multiallelic variants detected (variant count unchanged)"
fi
echo "=========================================="
echo ""


# Clean up
rm -rf "${TEMP_DIR}"


