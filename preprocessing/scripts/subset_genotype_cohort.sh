#!/bin/bash
#SBATCH --job-name=subset_geno_${COHORT}
#SBATCH --output=logs/subset/subset_genotype_${COHORT}.out
#SBATCH --error=logs/subset/subset_genotype_${COHORT}.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Subset genotype files for a specific cohort using PLINK
# Usage: sbatch --export=COHORT=<cohort_name> subset_genotype_cohort.sh

set -e  # Exit on error

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Check if COHORT is set
if [ -z "$COHORT" ]; then
    echo "ERROR: COHORT environment variable not set"
    echo "Usage: sbatch --export=COHORT=<cohort_name> subset_genotype_cohort.sh"
    exit 1
fi

# Hardcoded paths
BASE_DIR="/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain"
GENOTYPE_DIR="${BASE_DIR}/Genotypes"
GENOTYPE_SUBSET_DIR="${GENOTYPE_DIR}/subset"
OUTPUT_DIR="${GENOTYPE_DIR}/subsetted"
FILE_SHEET="/gpfs/commons/home/vmazeeva/BigBrain_files_sheet.tsv"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=========================================="
echo "Subsetting genotype for cohort: ${COHORT}"
echo "=========================================="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# Load PLINK 1.9 module
source /etc/profile.d/modules.sh 2>/dev/null || true
module load plink 2>/dev/null

# Check if plink is available
if ! command -v plink &> /dev/null; then
    echo "ERROR: plink command not found"
    exit 1
fi

echo "PLINK version:"
plink --version || true
echo ""

# Find keep file
KEEP_FILE="${GENOTYPE_SUBSET_DIR}/${COHORT}_keep_participants.txt"

if [ ! -f "${KEEP_FILE}" ]; then
    echo "ERROR: Keep file not found: ${KEEP_FILE}"
    exit 1
fi

echo "Keep file: ${KEEP_FILE}"
# Count lines using Python (most reliable, already used in script)
N_PARTICIPANTS=$(python3 -c "with open('${KEEP_FILE}') as f: print(sum(1 for _ in f))" 2>/dev/null || echo "0")
echo "Number of participants to keep: ${N_PARTICIPANTS}"
echo ""

# Get genotype file path from file sheet
# Normalize cohort name for file sheet lookup
if [ "${COHORT}" == "NYGC" ]; then
    SHEET_COHORT="NYGC ALS"
elif [ "${COHORT}" == "Mayo" ]; then
    SHEET_COHORT="Mayo Clinic"
elif [ "${COHORT}" == "ROSMAP_DLPFC" ]; then
    SHEET_COHORT="ROSMAP"
else
    SHEET_COHORT="${COHORT}"
fi

# Extract genotype file path from file sheet using Python
GENOTYPE_PATH=$(python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv("${FILE_SHEET}", sep="\t")
    cohort_col = df.columns[0]  # First column should be Cohort
    geno_col = None
    
    # Find genotype column
    for col in df.columns:
        if "genotype" in col.lower() or "Genotype" in col:
            geno_col = col
            break
    
    if geno_col is None:
        sys.exit(1)
    
    # Find row matching cohort
    row = df[df[cohort_col] == "${SHEET_COHORT}"]
    if len(row) == 0:
        sys.exit(1)
    
    geno_path = row[geno_col].iloc[0]
    if pd.notna(geno_path) and str(geno_path).strip():
        print(str(geno_path).strip())
    else:
        sys.exit(1)
except:
    sys.exit(1)
EOF
)

if [ -z "${GENOTYPE_PATH}" ]; then
    echo "ERROR: Could not find genotype file path for cohort ${COHORT} in file sheet"
    exit 1
fi

# Convert relative path to absolute if needed
if [[ ! "${GENOTYPE_PATH}" = /* ]]; then
    GENOTYPE_PATH="${BASE_DIR}/${GENOTYPE_PATH}"
fi

echo "Genotype file path: ${GENOTYPE_PATH}"
echo ""

# Determine PLINK format and base path
# Handle special notation: .bim/.bed/.fam or .pgen/.psam/.pvar
if [[ "${GENOTYPE_PATH}" == *".bim/.bed/.fam"* ]]; then
    BASE_PATH="${GENOTYPE_PATH%.bim/.bed/.fam}"
    FORMAT="plink1"
    EXTENSIONS=".bed .bim .fam"
elif [[ "${GENOTYPE_PATH}" == *".pgen/.psam/.pvar"* ]]; then
    BASE_PATH="${GENOTYPE_PATH%.pgen/.psam/.pvar}"
    FORMAT="plink2"
    EXTENSIONS=".pgen .psam .pvar"
else
    # Try to detect format by checking which files exist
    if [ -f "${GENOTYPE_PATH}.bed" ] && [ -f "${GENOTYPE_PATH}.bim" ] && [ -f "${GENOTYPE_PATH}.fam" ]; then
        BASE_PATH="${GENOTYPE_PATH}"
        FORMAT="plink1"
        EXTENSIONS=".bed .bim .fam"
    elif [ -f "${GENOTYPE_PATH}.pgen" ] && [ -f "${GENOTYPE_PATH}.psam" ] && [ -f "${GENOTYPE_PATH}.pvar" ]; then
        BASE_PATH="${GENOTYPE_PATH}"
        FORMAT="plink2"
        EXTENSIONS=".pgen .psam .pvar"
    else
        echo "ERROR: Could not determine PLINK format for ${GENOTYPE_PATH}"
        echo "Looking for .bed/.bim/.fam or .pgen/.psam/.pvar files"
        exit 1
    fi
fi

echo "Base path: ${BASE_PATH}"
echo "Format: ${FORMAT}"
echo ""

# Verify input files exist
if [ "${FORMAT}" == "plink1" ]; then
    for ext in bed bim fam; do
        if [ ! -f "${BASE_PATH}.${ext}" ]; then
            echo "ERROR: Input file not found: ${BASE_PATH}.${ext}"
            exit 1
        fi
    done
else
    for ext in pgen psam pvar; do
        if [ ! -f "${BASE_PATH}.${ext}" ]; then
            echo "ERROR: Input file not found: ${BASE_PATH}.${ext}"
            exit 1
        fi
    done
fi

# Set output path
OUTPUT_BASE="${OUTPUT_DIR}/${COHORT}_subset"

echo "Output will be saved to: ${OUTPUT_BASE}"
echo ""

# Subset genotype file using PLINK
echo "Subsetting genotype file..."
echo ""

# Step 1: Subset samples first (to count variants in subsetted data)
echo "Step 1: Subsetting samples..."
TEMP_SUBSET_ALL="${OUTPUT_BASE}_temp_subset_all"
TEMP_SUBSET="${OUTPUT_BASE}_temp_subset"

# 1a) Subset samples without max-alleles filter (to get baseline variant count)
if [ "${FORMAT}" == "plink1" ]; then
    plink --bfile "${BASE_PATH}" \
          --keep "${KEEP_FILE}" \
          --keep-allele-order \
          --set-all-var-ids '@:#_$1_$2' \
          --new-id-max-allele-len 100 missing \
          --make-bed \
          --out "${TEMP_SUBSET_ALL}" \
          --threads ${SLURM_CPUS_PER_TASK} \
          --memory ${SLURM_MEM_PER_NODE} || {
        echo "ERROR: PLINK subsetting (baseline) failed"
        exit 1
    }
else
    # Convert PLINK 2 format to PLINK 1.9 format first, then subset
    TEMP_CONVERTED="${OUTPUT_BASE}_temp_converted"
    plink --pfile "${BASE_PATH}" \
          --make-bed \
          --out "${TEMP_CONVERTED}" \
          --threads ${SLURM_CPUS_PER_TASK} \
          --memory ${SLURM_MEM_PER_NODE} || {
        echo "ERROR: Failed to convert PLINK 2 input to PLINK 1.9 format"
        exit 1
    }
    plink --bfile "${TEMP_CONVERTED}" \
          --keep "${KEEP_FILE}" \
          --keep-allele-order \
          --set-all-var-ids '@:#_$1_$2' \
          --new-id-max-allele-len 100 missing \
          --make-bed \
          --out "${TEMP_SUBSET_ALL}" \
          --threads ${SLURM_CPUS_PER_TASK} \
          --memory ${SLURM_MEM_PER_NODE} || {
        echo "ERROR: PLINK subsetting (baseline) failed"
        exit 1
    }
    rm -f "${TEMP_CONVERTED}".{bed,bim,fam,log} 2>/dev/null || true
fi

# Count total variants after sample subsetting, before max-alleles 2
TOTAL_VARIANTS_BEFORE_MAX=$(wc -l < "${TEMP_SUBSET_ALL}.bim" 2>/dev/null || echo "0")
echo "  Total variants after sample subsetting (before --max-alleles 2): ${TOTAL_VARIANTS_BEFORE_MAX}"

# 1b) Apply --max-alleles 2 on the subsetted data
plink --bfile "${TEMP_SUBSET_ALL}" \
      --max-alleles 2 \
      --make-bed \
      --out "${TEMP_SUBSET}" \
      --threads ${SLURM_CPUS_PER_TASK} \
      --memory ${SLURM_MEM_PER_NODE} || {
    echo "ERROR: PLINK --max-alleles 2 filtering failed"
    exit 1
}

# Count total variants after max-alleles 2 (still before SNP-only filtering)
TOTAL_VARIANTS=$(wc -l < "${TEMP_SUBSET}.bim" 2>/dev/null || echo "0")
MULTIALLELIC_DROPPED=$((TOTAL_VARIANTS_BEFORE_MAX - TOTAL_VARIANTS))

echo ""
echo "=========================================="
echo "Max-alleles 2 Filtering Statistics"
echo "=========================================="
echo "  Total variants after --max-alleles 2: ${TOTAL_VARIANTS}"
echo "  Variants dropped by --max-alleles 2: ${MULTIALLELIC_DROPPED}"
echo ""

# Step 2: Filter to SNPs only
echo "Step 2: Filtering to SNPs only..."
TEMP_SNP_FILTERED="${OUTPUT_BASE}_snps_only"

plink --bfile "${TEMP_SUBSET}" \
      --snps-only just-acgt \
      --make-bed \
      --out "${TEMP_SNP_FILTERED}" \
      --threads ${SLURM_CPUS_PER_TASK} \
      --memory ${SLURM_MEM_PER_NODE} || {
    echo "ERROR: PLINK SNP filtering failed"
    exit 1
}


# Count SNPs after filtering
SNPS_KEPT=$(wc -l < "${TEMP_SNP_FILTERED}.bim" 2>/dev/null || echo "0")
INDELS_FILTERED=$((TOTAL_VARIANTS - SNPS_KEPT))

echo ""
echo "=========================================="
echo "SNP Filtering Statistics"
echo "=========================================="
echo "  Total variants (before SNP filtering): ${TOTAL_VARIANTS}"
echo "  SNPs kept: ${SNPS_KEPT}"
echo "  Indels/other variants filtered out: ${INDELS_FILTERED}"

# Calculate percentage
if [ "${TOTAL_VARIANTS}" -gt 0 ]; then
    PERCENTAGE=$(python3 -c "print(f'{${SNPS_KEPT}/${TOTAL_VARIANTS}*100:.2f}%')" 2>/dev/null || echo "N/A")
    echo "  Percentage SNPs: ${PERCENTAGE}"
else
    PERCENTAGE="N/A"
    echo "  Percentage SNPs: ${PERCENTAGE}"
fi
echo ""

# Step 3: Split SNP-filtered data by chromosome
# Note: Multiallelic variant filtering is handled in the merge step (update_genotype_ids_cohort.sh)
# because multiallelic issues only appear when merging per-sample files, not in the original cohort file
echo "Step 3: Splitting SNP-filtered data by chromosome..."

# Process each chromosome (1-22)
for CHR in {1..22}; do
    CHR_OUTPUT="${OUTPUT_BASE}_chr${CHR}"
    echo "  Processing chromosome ${CHR}..."
    
    plink --bfile "${TEMP_SNP_FILTERED}" \
          --chr ${CHR} \
          --make-bed \
          --out "${CHR_OUTPUT}" \
          --threads ${SLURM_CPUS_PER_TASK} \
          --memory ${SLURM_MEM_PER_NODE} || {
        echo "WARNING: PLINK command failed for chromosome ${CHR} (may not exist in dataset)"
        continue
    }
    
    # Check if output was created (chromosome may not exist)
    if [ -f "${CHR_OUTPUT}.bed" ]; then
        CHR_VARIANTS=$(wc -l < "${CHR_OUTPUT}.bim" 2>/dev/null || echo "0")
        echo "    ✓ Created: ${CHR_OUTPUT}.{bed,bim,fam} (${CHR_VARIANTS} variants)"
    fi
done
    


# Clean up temporary files (after chromosome splitting is complete)
echo ""
echo "Cleaning up temporary files..."
rm -f "${TEMP_SUBSET_ALL}".{bed,bim,fam,log} 2>/dev/null || true
rm -f "${TEMP_SUBSET}".{bed,bim,fam,log} 2>/dev/null || true
rm -f "${TEMP_SNP_FILTERED}".{bed,bim,fam,log} 2>/dev/null || true
echo "  ✓ Removed temporary files"

echo ""
echo "=========================================="
echo "Subsetting complete!"
echo "=========================================="
echo ""
echo "Chromosome-specific output files:"
for CHR in {1..22}; do
    CHR_OUTPUT="${OUTPUT_BASE}_chr${CHR}"
    if [ -f "${CHR_OUTPUT}.bed" ]; then
        echo "  Chromosome ${CHR}:"
        ls -lh "${CHR_OUTPUT}".{bed,bim,fam} 2>/dev/null | awk '{print "    " $0}' || true
    fi
done
echo ""
echo "Date: $(date)"

