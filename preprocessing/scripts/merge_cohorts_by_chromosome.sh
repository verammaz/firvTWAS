#!/bin/bash


# Merge all cohorts for a specific chromosome
# Usage: sbatch merge_cohorts_by_chromosome.sh

set -euo pipefail  # Exit on error, treat unset vars as error, pipefail for proper exit codes in pipelines
ncores=8


# Define cohorts 
COHORTS=("AnswerALS" "Mayo" "MSBB" "NYGC" "ROSMAP" "GTEX")

# Ensure PATH includes standard 
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Hardcoded paths
CHR=${SLURM_ARRAY_TASK_ID}
CHROM_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Genotypes/final"
FINAL_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/chroms"

# Create directories
mkdir -p "${FINAL_DIR}"
mkdir -p logs

echo "=========================================="
echo "Merging cohorts for chromosome ${CHR}"
echo "Cohorts: ${COHORTS[@]}"
echo "=========================================="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# Load PLINK 1.9 module
source /etc/profile.d/modules.sh 2>/dev/null || true
module load PLINK/1.9 2>/dev/null
command -v plink >/dev/null || { echo "ERROR: plink not found"; exit 1; }

# Function to remove .nosex files created by PLINK
# .nosex files are created when PLINK cannot determine sex from the data
# (missing or invalid sex codes in FAM file). They're harmless but can clutter
# the directory. We use --allow-no-sex to allow analysis to continue, and then
# clean up any .nosex files that are created.
cleanup_nosex() {
    local base_path="$1"
    if [ -f "${base_path}.nosex" ]; then
        rm -f "${base_path}.nosex"
        # Optionally suppress the message by commenting out the next line
        # echo "    Removed .nosex file: ${base_path}.nosex"
    fi
}

# Step 1: Find all cohort files for this chromosome (PLINK 1.9 format)
echo "Step 1: Finding cohort files for chromosome ${CHR}..."
COHORT_FILES=()
for cohort in "${COHORTS[@]}"; do
    bed_file="${CHROM_DIR}/${cohort}_final_chr${CHR}.bed"
    echo "  BED file: ${bed_file}"
    base_path="${CHROM_DIR}/${cohort}_final_chr${CHR}"
    if [ -f "${bed_file}" ]; then
        
        # Verify all three PLINK 1.9 format files exist
        if [ -f "${base_path}.bed" ] && \
           [ -f "${base_path}.bim" ] && \
           [ -f "${base_path}.fam" ]; then
            COHORT_FILES+=("${base_path}")
            echo "  Found: ${base_path}"
        else
            echo "  WARNING: Incomplete files for ${base_path}, skipping"
        fi
    fi
done

if [ ${#COHORT_FILES[@]} -eq 0 ]; then
    echo "ERROR: No cohort files found for chromosome ${CHR}"
    exit 1
fi

echo "  Total cohorts to merge: ${#COHORT_FILES[@]}"
echo ""

# Step 2: Create merge list file, sorted by number of variants (largest first)
# This ensures the cohort with most variants becomes the base file for --keep-allele-order
echo "Step 2: Creating merge list file (sorted by variant count)..."
MERGE_LIST="${FINAL_DIR}/merge_list_chr${CHR}.txt"

# Create temporary directory early (needed for sorting)
TEMP_DIR=$(mktemp -d -t merge_chr${CHR}_XXXXXX)
trap "rm -rf ${TEMP_DIR}" EXIT

# Count variants in each cohort file and sort by count (descending)
echo "  Counting variants per cohort..."
# Create temporary file to store counts for sorting
COUNT_FILE="${TEMP_DIR}/cohort_counts.txt"
> "${COUNT_FILE}"

for cohort_path in "${COHORT_FILES[@]}"; do
    if [ -f "${cohort_path}.bim" ]; then
        variant_count=$(wc -l < "${cohort_path}.bim" 2>/dev/null || echo "0")
        echo "${variant_count} ${cohort_path}" >> "${COUNT_FILE}"
        echo "    $(basename "${cohort_path}"): ${variant_count} variants"
    fi
done

# Sort by variant count (descending) and extract paths
# Create merge list with largest cohort first (as base file)
> "${MERGE_LIST}"  # Create empty file
sort -rn -k1 "${COUNT_FILE}" | cut -d' ' -f2- | while read cohort_path; do
    echo "${cohort_path}" >> "${MERGE_LIST}"
done

echo "  Merge list created: ${MERGE_LIST}"
echo "  Cohorts in merge list (sorted by variant count, largest first):"
cat "${MERGE_LIST}" | while read line; do
    cohort_name=$(basename "${line}")
    variant_count=$(wc -l < "${line}.bim" 2>/dev/null || echo "0")
    echo "    - ${cohort_name} (${variant_count} variants)"
done
echo ""

# Step 3: Merge genotype files
echo "Step 3: Merging genotype files for chromosome ${CHR}..."
OUTPUT_BASE="${FINAL_DIR}/merged_chr${CHR}"

# PLINK merge requires the first file as base, then --merge-list for the rest
# First file is now the one with most variants
FIRST_FILE=$(head -n 1 "${MERGE_LIST}")
BASE_COHORT=$(basename "${FIRST_FILE}")
BASE_VARIANTS=$(wc -l < "${FIRST_FILE}.bim" 2>/dev/null || echo "0")
echo "  Base file: ${BASE_COHORT} (${BASE_VARIANTS} variants)"
REMAINING_FILES="${TEMP_DIR}/merge_list_chr${CHR}_remaining.txt"
tail -n +2 "${MERGE_LIST}" > "${REMAINING_FILES}"

if [ ! -s "${REMAINING_FILES}" ]; then
    # Only one file, just copy it
    echo "  Only one cohort found, copying to merged directory..."
    cp "${FIRST_FILE}.bed" "${OUTPUT_BASE}.bed"
    cp "${FIRST_FILE}.bim" "${OUTPUT_BASE}.bim"
    cp "${FIRST_FILE}.fam" "${OUTPUT_BASE}.fam"
    MULTIALLELIC_REMOVED=0
    # Clean up any .nosex files (shouldn't exist, but just in case)
    cleanup_nosex "${OUTPUT_BASE}"
else
    # Multiple files, merge them using PLINK 1.9
    echo "  Merging ${#COHORT_FILES[@]} cohort files..."
    echo "  Base file: $(basename "${FIRST_FILE}")"
    echo "  Additional files: $((${#COHORT_FILES[@]} - 1))"
    
    # Attempt merge with retry logic for multiallelic variants
    # Attempts: 1) Original merge, 2) After removing multiallelic variants
    MULTIALLELIC_REMOVED=0
    MERGE_ATTEMPT=1
    MAX_MERGE_ATTEMPTS=2
    
    while [ ${MERGE_ATTEMPT} -le ${MAX_MERGE_ATTEMPTS} ]; do
        echo "  Merge attempt ${MERGE_ATTEMPT}/${MAX_MERGE_ATTEMPTS}..."
        
        if plink \
          --bfile "${FIRST_FILE}" \
          --merge-list "${REMAINING_FILES}" \
          --keep-allele-order \
          --allow-no-sex \
          --make-bed \
          --threads ${ncores} \
          --out "${OUTPUT_BASE}" 2>&1 | tee "${TEMP_DIR}/merge_attempt.log"; then
            # Merge succeeded
            echo "  ✔ Merge successful"
            # Remove .nosex file if created
            cleanup_nosex "${OUTPUT_BASE}"
            break
        else
            # Merge failed - check if it is due to multiallelic variants
            if grep -q "variants with 3+ alleles present" "${TEMP_DIR}/merge_attempt.log"; then
                # Extract count from error message
                # Pattern: "Error: 1249 variants with 3+ alleles present."
                MULTIALLELIC_COUNT=$(grep "Error:" "${TEMP_DIR}/merge_attempt.log" | \
                                     sed -n 's/.*Error: \([0-9][0-9]*\) variants with 3+ alleles.*/\1/p' | head -1)
                if [ -z "${MULTIALLELIC_COUNT}" ]; then
                    # Fallback: try to count from .missnp file if it exists
                    if [ -f "${OUTPUT_BASE}-merge.missnp" ]; then
                        MULTIALLELIC_COUNT=$(wc -l < "${OUTPUT_BASE}-merge.missnp")
                    else
                        MULTIALLELIC_COUNT=0
                    fi
                fi
                echo "  Merge failed due to ${MULTIALLELIC_COUNT} multiallelic variants (3+ alleles)"
                
                # Check for .missnp file created by PLINK
                MISSNP_FILE="${OUTPUT_BASE}-merge.missnp"
                
                if [ ${MERGE_ATTEMPT} -eq 1 ]; then
                    # First attempt failed - remove multiallelic variants and retry
                    if [ -f "${MISSNP_FILE}" ]; then
                        echo "  Removing multiallelic variants from all cohort files..."
                        MULTIALLELIC_REMOVED=$(wc -l < "${MISSNP_FILE}")
                        
                        # Remove multiallelic variants from all cohort files
                        for cohort_path in "${COHORT_FILES[@]}"; do
                            if ! plink \
                              --bfile "${cohort_path}" \
                              --exclude "${MISSNP_FILE}" \
                              --keep-allele-order \
                              --allow-no-sex \
                              --threads ${ncores} \
                              --make-bed \
                              --out "${cohort_path}_filtered" \
                              >/dev/null 2>&1; then
                                echo "  WARNING: Failed to exclude variants from ${cohort_path}"
                                echo "  This may be due to variant ID mismatch. Continuing anyway..."
                            fi
                            
                            # Replace original with filtered version if it exists
                            if [ -f "${cohort_path}_filtered.bed" ]; then
                                mv "${cohort_path}_filtered.bed" "${cohort_path}.bed"
                                mv "${cohort_path}_filtered.bim" "${cohort_path}.bim"
                                mv "${cohort_path}_filtered.fam" "${cohort_path}.fam"
                                cleanup_nosex "${cohort_path}_filtered"
                            fi
                        done
                        
                        # Recreate merge list after filtering
                        > "${MERGE_LIST}"
                        for cohort_path in "${COHORT_FILES[@]}"; do
                            echo "${cohort_path}" >> "${MERGE_LIST}"
                        done
                        FIRST_FILE=$(head -n 1 "${MERGE_LIST}")
                        tail -n +2 "${MERGE_LIST}" > "${REMAINING_FILES}"
                        
                        echo "  Removed ${MULTIALLELIC_REMOVED} multiallelic variants. Retrying merge..."
                        MERGE_ATTEMPT=$((MERGE_ATTEMPT + 1))
                    else
                        # No .missnp file - this shouldn't happen if we detected multiallelic error
                        echo "  WARNING: Multiallelic error detected but no .missnp file found."
                        echo "  Cannot automatically remove multiallelic variants."
                        echo "ERROR: Merge failed. Check ${TEMP_DIR}/merge_attempt.log and ${OUTPUT_BASE}.log for details."
                        exit 1
                    fi
                else
                    # Second attempt failed - give up
                    echo "ERROR: Merge failed after removing multiallelic variants."
                    echo "  Check ${TEMP_DIR}/merge_attempt.log and ${OUTPUT_BASE}.log for details."
                    exit 1
                fi
            else
                # Merge failed for a different reason (not multiallelic variants)
                echo "ERROR: Merge failed for a reason other than multiallelic variants."
                echo "  Check ${TEMP_DIR}/merge_attempt.log and ${OUTPUT_BASE}.log for details."
                # Show last few lines of error for debugging
                echo "  Last lines of PLINK log:"
                tail -10 "${OUTPUT_BASE}.log" 2>/dev/null | sed 's/^/    /' || true
                exit 1
            fi
        fi
    done
fi

echo "  Merged genotype saved to: ${OUTPUT_BASE}"
echo ""

# Final cleanup of .nosex files
cleanup_nosex "${OUTPUT_BASE}"

# Step 4: Verify output files
echo "Step 4: Verifying output files..."
if [ -f "${OUTPUT_BASE}.bed" ] && [ -f "${OUTPUT_BASE}.bim" ] && [ -f "${OUTPUT_BASE}.fam" ]; then
    ls -lh "${OUTPUT_BASE}".{bed,bim,fam}
    echo "  ✓ Merged files created successfully"
else
    echo "  ✗ ERROR: Some merged files are missing!"
    exit 1
fi
echo ""

# Step 5: Clean up temporary files
echo "Step 5: Cleaning up temporary files..."
if [ -f "${MERGE_LIST}" ]; then
    rm -f "${MERGE_LIST}"
    echo "  Removed merge list: ${MERGE_LIST}"
fi
# TEMP_DIR cleanup is handled by trap, but clean up explicitly here too
if [ -n "${TEMP_DIR}" ] && [ -d "${TEMP_DIR}" ]; then
    rm -rf "${TEMP_DIR}"
    echo "  Removed temporary directory"
fi
echo ""

# ==========================================
# Summary Section
# ==========================================
echo "=========================================="
echo "MERGE SUMMARY"
echo "=========================================="

# Count final variants and samples
N_VARIANTS=0
N_SAMPLES=0
if [ -f "${OUTPUT_BASE}.bim" ] && [ -f "${OUTPUT_BASE}.fam" ]; then
    N_VARIANTS=$(wc -l < "${OUTPUT_BASE}.bim" 2>/dev/null || echo "0")
    N_SAMPLES=$(wc -l < "${OUTPUT_BASE}.fam" 2>/dev/null || echo "0")
    echo "Total variants: ${N_VARIANTS}"
    echo "Total samples: ${N_SAMPLES}"
else
    echo "ERROR: Output files not found for summary"
    echo "  Looking for: ${OUTPUT_BASE}.bim and ${OUTPUT_BASE}.fam"
    [ -f "${OUTPUT_BASE}.bim" ] && echo "  .bim file exists" || echo "  .bim file missing"
    [ -f "${OUTPUT_BASE}.fam" ] && echo "  .fam file exists" || echo "  .fam file missing"
fi

# Report multiallelic variant handling
echo ""
echo "Multiallelic Variant Handling:"
if [ ${MULTIALLELIC_REMOVED:-0} -gt 0 ]; then
    echo "  Multiallelic variants removed: ${MULTIALLELIC_REMOVED}"
else
    # Check for multiallelic variants that were split (not removed)
    MULTIALLELIC_COUNT=0
    if [ -f "${OUTPUT_BASE}-merge.missnp" ]; then
        MULTIALLELIC_COUNT=$(wc -l < "${OUTPUT_BASE}-merge.missnp" 2>/dev/null || echo "0")
        echo "  Multiallelic variants written to .missnp file: ${MULTIALLELIC_COUNT}"
    else
        # Count positions that appear multiple times (indicates multiallelic split)
        if [ -f "${OUTPUT_BASE}.bim" ]; then
            MULTIALLELIC_POSITIONS=$(cut -f1,4 "${OUTPUT_BASE}.bim" 2>/dev/null | sort | uniq -d | wc -l 2>/dev/null || echo "0")
            if [ "${MULTIALLELIC_POSITIONS}" -gt 0 ]; then
                MULTIALLELIC_COUNT="${MULTIALLELIC_POSITIONS}"
                echo "  Multiallelic variants detected (split into pseudo-biallelic): ${MULTIALLELIC_COUNT} positions"
            else
                echo "  No multiallelic variants detected"
            fi
        else
            echo "  No multiallelic variants detected"
        fi
    fi
fi

# Report A1/A2 flip information
echo ""
echo "A1/A2 Allele Flip Information:"
A1A2_FLIPS=0
A1A2_WARNINGS=0

# Check for .flip file created by PLINK (if merge attempted flips)
if [ -f "${OUTPUT_BASE}-merge.flip" ]; then
    A1A2_FLIPS=$(wc -l < "${OUTPUT_BASE}-merge.flip" 2>/dev/null | tr -d '[:space:]' || echo "0")
    # Ensure it's a valid integer
    A1A2_FLIPS=$((A1A2_FLIPS)) 2>/dev/null || A1A2_FLIPS=0
    echo "  Variants in .flip file: ${A1A2_FLIPS}"
    echo "  Note: With --keep-allele-order, these were NOT flipped"
fi

# Check PLINK log for A1/A2 related warnings
if [ -f "${OUTPUT_BASE}.log" ]; then
    # Count warnings about A1/A2 swaps/flips/mismatches
    A1A2_WARNINGS=$(grep -i "a1/a2\|allele.*swap\|allele.*flip\|allele.*mismatch" "${OUTPUT_BASE}.log" 2>/dev/null | wc -l | tr -d '[:space:]' || echo "0")
    # Ensure it's a valid integer
    A1A2_WARNINGS=$((A1A2_WARNINGS)) 2>/dev/null || A1A2_WARNINGS=0
    if [ "${A1A2_WARNINGS}" -gt 0 ]; then
        echo "  A1/A2 related warnings in PLINK log: ${A1A2_WARNINGS}"
        # Try to extract specific flip count if reported
        FLIP_COUNT_MSG=$(grep -iE "flip|swap" "${OUTPUT_BASE}.log" 2>/dev/null | grep -oE "[0-9]+" | head -1 || echo "")
        if [ -n "${FLIP_COUNT_MSG}" ]; then
            echo "  Reported flip/swap count: ${FLIP_COUNT_MSG}"
        fi
    else
        echo "  No A1/A2 flip warnings detected"
        echo "  (Using --keep-allele-order, so alleles are preserved as-is)"
    fi
else
    echo "  PLINK log file not found: ${OUTPUT_BASE}.log"
fi

# Check merge attempt logs if available
if [ -n "${TEMP_DIR:-}" ] && [ -d "${TEMP_DIR}" ] && [ -f "${TEMP_DIR}/merge_attempt.log" ]; then
    TEMP_A1A2_WARNINGS=$(grep -i "a1/a2\|allele.*swap\|allele.*flip\|allele.*mismatch" "${TEMP_DIR}/merge_attempt.log" 2>/dev/null | wc -l | tr -d '[:space:]' || echo "0")
    # Ensure it's a valid integer
    TEMP_A1A2_WARNINGS=$((TEMP_A1A2_WARNINGS)) 2>/dev/null || TEMP_A1A2_WARNINGS=0
    if [ "${TEMP_A1A2_WARNINGS}" -gt 0 ] && [ "${TEMP_A1A2_WARNINGS}" -ne "${A1A2_WARNINGS}" ]; then
        echo "  Additional warnings in merge attempt log: ${TEMP_A1A2_WARNINGS}"
    fi
fi

# Ensure A1A2_FLIPS is set and is an integer (in case it wasn't set above)
A1A2_FLIPS=${A1A2_FLIPS:-0}
A1A2_FLIPS=$((A1A2_FLIPS)) 2>/dev/null || A1A2_FLIPS=0
A1A2_WARNINGS=${A1A2_WARNINGS:-0}

if [ "${A1A2_FLIPS}" -eq 0 ] && [ "${A1A2_WARNINGS}" -eq 0 ]; then
    echo "  ✓ No A1/A2 flips detected"
fi

echo "=========================================="
echo ""
echo "Merge complete for chromosome ${CHR}!"
echo "Date: $(date)"

