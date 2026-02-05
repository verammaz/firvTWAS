#!/bin/bash
#SBATCH --job-name=dedup_chr
#SBATCH --output=logs/dedup/dedup_chr%a.out
#SBATCH --error=logs/dedup/dedup_chr%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-22

set -euo pipefail

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate


# Load PLINK module (initialize module system first)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load plink
command -v plink >/dev/null || { echo "ERROR: plink not found"; exit 1; }


# -----------------------
# CONFIGURATION
# -----------------------
CHR=${SLURM_ARRAY_TASK_ID}

BASE_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/Train/chroms"

IN_PREFIX=${BASE_DIR}/merged_chr${CHR}
OUT_PREFIX=${BASE_DIR}/clean_chr${CHR}

echo "======================================"
echo "Chromosome ${CHR}"
echo "Input:  ${IN_PREFIX}"
echo "Output: ${OUT_PREFIX}"
echo "Started: $(date)"
echo "======================================"

# -----------------------
# STEP 1: REPORT VARIANT COUNTS (BEFORE ANY CHANGES)
# -----------------------
BIM=${IN_PREFIX}.bim

TOTAL_VARS=$(wc -l < "${BIM}")
UNIQUE_POS=$(awk '{print $1 ":" $4}' "${BIM}" | sort -u | wc -l)

DUP_COUNT=$((TOTAL_VARS - UNIQUE_POS))
UNIQUE_PCT=$((UNIQUE_POS * 100 / TOTAL_VARS))
DUP_PCT=$((DUP_COUNT * 100 / TOTAL_VARS))

echo "[REPORT] Total variants           : ${TOTAL_VARS}"
echo "[REPORT] Unique positions (CHR:BP): ${UNIQUE_POS} (${UNIQUE_PCT}%)"
echo "[REPORT] Duplicated positions     : ${DUP_COUNT} (${DUP_PCT}%)"

# -----------------------
# STEP 2: IDENTIFY TRUE DUPLICATES (ALLELE-AWARE)
# -----------------------
# True duplicate definition:
#   same CHR
#   same BP
#   same unordered allele pair {A1,A2}
# Multiallelic sites (A/G vs A/C) are KEPT

DUP_FILE=${OUT_PREFIX}.true_duplicates.txt

python << EOF
bim = "${BIM}"
outf = "${DUP_FILE}"

seen = {}
duplicates = []

with open(bim) as f:
    for line in f:
        chrom, vid, cm, bp, a1, a2 = line.strip().split()
        alleles = tuple(sorted([a1, a2]))
        key = (chrom, bp, alleles)

        if key in seen:
            duplicates.append(vid)
        else:
            seen[key] = vid

with open(outf, "w") as out:
    for vid in duplicates:
        out.write(vid + "\\n")

print(f"[REPORT] True duplicate variants removed: {len(duplicates)}")
EOF

# -----------------------
# STEP 3: REMOVE TRUE DUPLICATES WITH PLINK
# -----------------------
if [[ -s "${DUP_FILE}" ]]; then
    plink --bfile "${IN_PREFIX}" \
             --exclude "${DUP_FILE}" \
             --allow-no-sex \
             --make-bed \
             --out "${OUT_PREFIX}_no_dups"
else
    echo "[INFO] No true duplicates found; copying input files"
    plink --bfile "${IN_PREFIX}" \
             --allow-no-sex \
             --make-bed \
             --out "${OUT_PREFIX}_no_dups"
fi

echo "[REPORT] True duplicates removed: $(wc -l < "${DUP_FILE}")"
echo "[REPORT] Final variants: $(wc -l < "${OUT_PREFIX}_no_dups.bim")"

# -----------------------
# STEP 4: ID NORMALIZATION
# -----------------------

# ---- Allele-aware IDs (annotation-facing)
plink --bfile "${OUT_PREFIX}_no_dups" \
         --allow-no-sex \
         --make-bed \
         --out "${OUT_PREFIX}"

# -----------------------
# STEP 5: SANITY CHECKS
# -----------------------
FINAL_VARS=$(wc -l < "${OUT_PREFIX}.bim")

MULTIALLELIC_POS=$(awk '{print $1 ":" $4}' "${OUT_PREFIX}.bim" | sort | uniq -c | awk '$1>1' | wc -l)

echo "[REPORT] Final variants            : ${FINAL_VARS}"
echo "[REPORT] Positions with >1 variant : ${MULTIALLELIC_POS}"

# -----------------------
# STEP 6: CLEAN UP
# -----------------------
rm "${DUP_FILE}"
rm "${OUT_PREFIX}_no_dups.bed"
rm "${OUT_PREFIX}_no_dups.bim"
rm "${OUT_PREFIX}_no_dups.fam"
rm "${OUT_PREFIX}_no_dups.log"

echo "Finished chromosome ${CHR} at $(date)"
