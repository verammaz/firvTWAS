#!/bin/bash

#SBATCH --job-name=genotype_mat_${SET}
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p cpu,bigmem,dev
#SBATCH --mem=100G
#SBATCH -t 0-8:00 # Runtime in D-HH:MM
#SBATCH --output=logs/genotype/genotype_mat_${SET}.chr%a.out 
#SBATCH --error=logs/genotype/genotype_mat_${SET}.chr%a.err
#SBATCH --cpus-per-task=8                   # CPUs per task
#SBATCH --array=1-22 

CHRO_NB=${SLURM_ARRAY_TASK_ID}

echo "Chromosome ${CHRO_NB}"
echo 


annotations=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/${SET}/annotations/chr${CHRO_NB}
plink=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/${SET}/chroms
output_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/${SET}/genotypes/chr${CHRO_NB}

scratch_dir=$(mktemp -d -t genotype_matrices${SET}_chr${CHRO_NB}_XXXXXX)

ncores=8


# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

module load PLINK/1.9 2>/dev/null

echo "$(date '+%Y-%m-%d %H:%M:%S')  Starting plink extraction/recode..."
echo 

mkdir -p "${output_dir}"

skipped_genes_log=${output_dir}/skipped_genes_chr${CHRO_NB}.txt
touch ${skipped_genes_log}

n_genes=0
n_skipped_genes=0

for gene_file in ${annotations}/*_annotations.tsv.gz; do
    gene=$(basename "$gene_file" "_annotations.tsv.gz")   # extract just the gene name

    echo "Processing $gene..."
    
    # Extract positions from annotation file
    # Format for --extract range: CHR START_POS END_POS LABEL
    # We use the same position for start and end to extract specific variants
    snp_file="${scratch_dir}/${gene}_snps.txt"
    > "${snp_file}"  # Create empty file
    
    # Extract positions from annotation file
    # Format for --extract range: CHR START_POS END_POS LABEL
    # Based on annotation file structure, 'pos' is column 2
    zcat "$gene_file" | tail -n +2 | awk -v chr=${CHRO_NB} '{
        pos = $2  # pos should be column 2
        # Convert to integer (handles any decimal values)
        pos_int = int(pos)
        if (pos_int > 0) {
            print chr "\t" pos_int "\t" pos_int "\t" "X"
        }
    }' >> "${snp_file}"
    
    n_snps=$(wc -l < "${snp_file}")
    echo "  Extracted ${n_snps} variant positions"
    
    if [ ${n_snps} -eq 0 ]; then
        echo "  WARNING: No variants found in annotation file for ${gene}, skipping"
        continue
    fi
    
    echo "  Running plink extraction/recode"
    echo

    # Use PLINK 1.9 (not PLINK2) - ensure we're using the right version
    if [ "${SET}" = "Test" ]; then
        bfile=${plink}/merged_chr${CHRO_NB}
    else
        bfile=${plink}/clean_chr${CHRO_NB}
    fi

    plink --bfile ${bfile} \
        --extract range "${snp_file}" \
        --const-fid \
        --maf 0.0001 \
        --out ${scratch_dir}/${gene} \
        --threads ${ncores} \
        --recode A include-alt 2>&1 | tee ${scratch_dir}/${gene}_plink.log
    
    plink_exit_code=${PIPESTATUS[0]}

    
    if [ ${plink_exit_code} -ne 0 ]; then
        echo "  ERROR: PLINK extraction failed for ${gene}"
        echo "  Check ${scratch_dir}/${gene}_plink.log for details"
        
        # Check if it's the "no variants remaining" error
        if [ ${plink_exit_code} -ne 0 ]; then
            echo "  WARNING: PLINK failed for ${gene}, skipping gene"
            echo "  Log: ${scratch_dir}/${gene}_plink.log"

            if grep -Eq "No variants remaining after main filters|All variants removed due to minor allele threshold" \
                ${scratch_dir}/${gene}_plink.log 2>/dev/null; then

                echo "  Reason: No variants passed filters (expected for some genes)"
                echo -e "${gene}\tMAF_or_filter_failure" >> ${skipped_genes_log}
                ((n_skipped_genes++))
                continue
            fi

            # Anything else is unexpected → fail the job
            echo "  ERROR: Unexpected PLINK failure for ${gene}"
            tail -n 20 ${scratch_dir}/${gene}_plink.log
            exit 1
        fi

    fi
    ((n_genes++))
    
    echo "  ✓ PLINK extraction successful for ${gene}"
    echo 

done

echo "$(date '+%Y-%m-%d %H:%M:%S')  Finished plink extraction/recode..."
echo "Total genes: ${n_genes}"
echo "Skipped genes (no variants after MAF + QC filter): ${n_skipped_genes}"
echo 

echo "$(date '+%Y-%m-%d %H:%M:%S')  Starting genotype processing..."
echo 

scripts=/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/scripts
python ${scripts}/process_genotype.py ${CHRO_NB} ${SET} ${scratch_dir} ${output_dir} 


echo
echo "$(date '+%Y-%m-%d %H:%M:%S')  Finished genotype processing..."

