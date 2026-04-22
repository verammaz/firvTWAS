#!/bin/bash

CHRO_NB=${SLURM_ARRAY_TASK_ID}

echo "Chromosome ${CHRO_NB}"
echo

annotations=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_raw/chr${CHRO_NB}
plink=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/chroms
output_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/genotypes/chr${CHRO_NB}

# Use a GPFS-backed scratch directory instead of node-local /tmp to avoid
# "File write failure" issues when /tmp fills up.
SCRATCH_BASE="/gpfs/commons/groups/knowles_lab/vmazeeva/tmp/genotype"
mkdir -p "${SCRATCH_BASE}"
scratch_dir=$(mktemp -d "${SCRATCH_BASE}/genotype_matrices_chr${CHRO_NB}_XXXXXX")

echo "Using scratch directory: ${scratch_dir}"

# Ensure PLINK and other tools use this scratch for temporary files
export TMPDIR="${scratch_dir}"

# Parallelization strategy:
# - Run multiple genes in parallel 
# - But still allow PLINK to use multiple threads per job in case it helps
# 
# Allocate threads per PLINK job, and run the rest in parallel
total_cpus=${SLURM_CPUS_PER_TASK:-8}
threads_per_plink=1 # --recode A doesn't use multithreading
max_parallel=$((total_cpus / threads_per_plink))

# Ensure we run at least 1 gene at a time
if [ "${max_parallel}" -lt 1 ]; then
    max_parallel=1
    threads_per_plink=${total_cpus}
fi

echo "Parallelization: ${max_parallel} genes in parallel, ${threads_per_plink} threads per PLINK job"

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

module load PLINK/1.9 2>/dev/null

echo "$(date '+%Y-%m-%d %H:%M:%S')  Starting plink extraction/recode..."
echo

mkdir -p "${output_dir}"

skipped_genes_log=${output_dir}/skipped_genes_chr${CHRO_NB}.txt
: > "${skipped_genes_log}"

success_genes_log="${scratch_dir}/successful_genes_chr${CHRO_NB}.txt"
error_genes_log="${scratch_dir}/error_genes_chr${CHRO_NB}.txt"
: > "${success_genes_log}"
: > "${error_genes_log}"

run_gene() {
    local gene_file="$1"
    local gene

    gene=$(basename "${gene_file}" "_annotations.tsv.gz")   # extract just the gene name

    echo "Processing ${gene}..."

    # Extract positions from annotation file
    # Format for --extract range: CHR START_POS END_POS LABEL
    # We use the same position for start and end to extract specific variants
    local snp_file="${scratch_dir}/${gene}_snps.txt"
    > "${snp_file}"  # Create empty file

    # Extract positions from annotation file
    # Format for --extract range: CHR START_POS END_POS LABEL
    # Annotation file has variant_id as first column (index) with format: chr:pos_ref_alt
    # Extract position from variant_id (column 1) instead of data columns
    zcat "${gene_file}" | tail -n +2 | awk -v chr="${CHRO_NB}" '{
        # variant_id is in first column: format is "chr:pos_ref_alt"
        variant_id = $1
        # Extract position: split by ":" then by "_" to get position
        split(variant_id, parts, ":")
        pos_part = parts[2]  # "pos_ref_alt"
        split(pos_part, pos_parts, "_")
        pos = pos_parts[1]  # position is first part before "_"
        # Convert to integer (handles any decimal values)
        pos_int = int(pos)
        if (pos_int > 0) {
            print chr "\t" pos_int "\t" pos_int "\t" "X"
        }
    }' >> "${snp_file}"

    local n_snps
    n_snps=$(wc -l < "${snp_file}")
    echo "  ${gene}: Extracted ${n_snps} variant positions"

    if [ "${n_snps}" -eq 0 ]; then
        echo "  WARNING: No variants found in annotation file for ${gene}, skipping"
        echo -e "${gene}\tno_variants_in_annotations" >> "${skipped_genes_log}"
        return 0
    fi

    echo "  ${gene}: Running plink extraction/recode"
    echo

    # Use PLINK 1.9 (not PLINK2) - ensure we're using the right version
    local bfile="${plink}/merged_chr${CHRO_NB}"

    plink --bfile "${bfile}" \
        --extract range "${snp_file}" \
        --const-fid \
        --maf 0.0001 \
        --out "${scratch_dir}/${gene}" \
        --threads "${threads_per_plink}" \
        --recode A include-alt 2>&1 | tee "${scratch_dir}/${gene}_plink.log"

    local plink_exit_code=${PIPESTATUS[0]}

    if [ "${plink_exit_code}" -ne 0 ]; then
        echo "  ERROR: PLINK extraction failed for ${gene}"
        echo "  Check ${scratch_dir}/${gene}_plink.log for details"

        if grep -Eq "No variants remaining after main filters|All variants removed due to minor allele threshold" \
            "${scratch_dir}/${gene}_plink.log" 2>/dev/null; then

            echo "  Reason: No variants passed filters (expected for some genes)"
            echo -e "${gene}\tMAF_or_filter_failure" >> "${skipped_genes_log}"
            return 0
        fi

        # Anything else is unexpected → record the error and continue;
        # we will fail the job after all genes are processed.
        echo "  ERROR: Unexpected PLINK failure for ${gene}"
        tail -n 20 "${scratch_dir}/${gene}_plink.log" || true
        echo "${gene}" >> "${error_genes_log}"
        return 1
    fi

    echo "${gene}" >> "${success_genes_log}"

    echo "  ✓ PLINK extraction successful for ${gene}"
    echo

    return 0
}

# Parallel loop over genes, up to ${max_parallel} at a time
pids=()

for gene_file in ${annotations}/*_annotations.tsv.gz; do
    # Skip if glob didn't match anything
    [ -e "${gene_file}" ] || continue

    run_gene "${gene_file}" &
    pids+=($!)

    # Throttle the number of concurrent PLINK runs
    while [ "${#pids[@]}" -ge "${max_parallel}" ]; do
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    done
done

# Wait for any remaining background jobs
for pid in "${pids[@]}"; do
    wait "${pid}"
done

# If any unexpected PLINK failures occurred, abort before downstream processing
if [ -s "${error_genes_log}" ]; then
    echo "ERROR: One or more genes failed unexpectedly. See:"
    echo "  ${error_genes_log}"
    echo "  and per-gene PLINK logs in ${scratch_dir}"
    exit 1
fi

n_genes=0
n_skipped_genes=0

if [ -f "${success_genes_log}" ]; then
    n_genes=$(wc -l < "${success_genes_log}")
fi

if [ -f "${skipped_genes_log}" ]; then
    n_skipped_genes=$(wc -l < "${skipped_genes_log}")
fi

echo "$(date '+%Y-%m-%d %H:%M:%S')  Finished plink extraction/recode..."
echo "Total genes with successful PLINK extraction: ${n_genes}"
echo "Skipped genes (no variants or no variants after MAF + QC filter): ${n_skipped_genes}"
echo

echo "$(date '+%Y-%m-%d %H:%M:%S')  Starting genotype processing..."
echo

scripts=/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts

# Set NUM_THREADS for Python script to match SLURM_CPUS_PER_TASK
export NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "Using ${NUM_THREADS} threads for Python genotype processing"

if python -u "${scripts}/process_genotype.py" "${CHRO_NB}" "${scratch_dir}" "${output_dir}"; then
    echo
    echo "$(date '+%Y-%m-%d %H:%M:%S')  Finished genotype processing..."
    echo "Cleaning up scratch directory: ${scratch_dir}"
    rm -rf "${scratch_dir}"
else
    echo "ERROR: genotype processing failed for chromosome ${CHRO_NB}."
    echo "Scratch directory kept at ${scratch_dir} for debugging."
    exit 1
fi
