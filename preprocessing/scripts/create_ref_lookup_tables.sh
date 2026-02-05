#!/bin/bash
#SBATCH --job-name=build_ref_chr
#SBATCH --output=logs/build_ref_chr%a.out
#SBATCH --error=logs/build_ref_chr%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --array=1-22

set -euo pipefail

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

module load samtools >/dev/null 2>&1 || { echo "ERROR: samtools not found"; exit 1; }

# -----------------------
# CONFIG
# -----------------------
CHR=${SLURM_ARRAY_TASK_ID}

REF_FASTA=/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/hg38.fa
BIM_DIR=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed
OUT_DIR=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/reference

mkdir -p "${OUT_DIR}"

BIM_TRAIN=${BIM_DIR}/Train/chroms/clean_chr${CHR}.bim
BIM_TEST=${BIM_DIR}/Test/chroms/merged_chr${CHR}.bim
OUT=${OUT_DIR}/chr${CHR}_ref.tsv.gz

echo "Chromosome ${CHR}"
echo "BIM Train: ${BIM_TRAIN}"
echo "BIM Test: ${BIM_TEST}"
echo "FASTA: ${REF_FASTA}"
echo "Output: ${OUT}"
echo "Started: $(date)"

# -----------------------
# CHECK FASTA INDEX
# -----------------------
if [[ ! -f "${REF_FASTA}.fai" ]]; then
    echo "Indexing FASTA..."
    samtools faidx "${REF_FASTA}"
fi

# -----------------------
# EXTRACT UNIQUE POSITIONS
# -----------------------
TMP_POS=$(mktemp)
TMP_REGIONS=$(mktemp)

cat "${BIM_TRAIN}" "${BIM_TEST}" \
| awk '{
    c=$1
    sub(/^chr/,"",c)
    print "chr" c "\t" $4
}' \
| sort -u > "${TMP_POS}"

echo "Unique positions (Train + Test): $(wc -l < "${TMP_POS}")"


# -----------------------
# QUERY FASTA 
# -----------------------
echo "Querying FASTA..."

# Convert to samtools region format: chr:pos-pos
awk '{print $1 ":" $2 "-" $2}' "${TMP_POS}" > "${TMP_REGIONS}"

# Query all regions in one FASTA pass
samtools faidx "${REF_FASTA}" -r "${TMP_REGIONS}" \
| awk '
    /^>/ {
        split(substr($0,2),a,":|-")
        chrom=a[1]
        pos=a[2]
        next
    }
    {
        base=toupper($0)
        if (base == "") base="N"
        print chrom "\t" pos "\t" base
    }
' \
| sort -k1,1 -k2,2n \
| gzip > "${OUT}"

echo "Reference table written to: ${OUT}"

# Cleanup
echo "Removing temporary files..."
rm -f "${TMP_POS}" "${TMP_REGIONS}"

# -----------------------
# REPORTING
# -----------------------

echo
echo "Reference table preview (first 5 rows):"
zcat "${OUT}" | sed -n '1,5p'

echo
echo "Total reference entries:"
echo "$(zcat "${OUT}" | wc -l)"

echo
echo "Number of positions with missing reference base (N):"
echo "$(zcat "${OUT}" | awk '$3=="N"' | wc -l)"

echo
echo "Finished chr${CHR} at $(date)"


