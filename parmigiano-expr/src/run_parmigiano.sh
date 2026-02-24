#!/bin/bash

#SBATCH --job-name=parmigiano_seed_genes
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Save script arguments
SCRIPT_ARGS=("$@")

# Activate conda environment (clear arguments first to avoid passing them to activate)
set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -- "${SCRIPT_ARGS[@]}"

cd /gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src

# Default values
scale_anno=False
output_dir=seed_genes_minmax
config_file=config_genewise.yaml
gene_list=genes_list_seed.txt
expression_path=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/tpm_genes_subset.tsv
annotation_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_minmax/
train_test=True
lr=0.1
epochs=500
random_genes=False
log_level=INFO

# Parse command line arguments with long-form options
while [[ $# -gt 0 ]]; do
    case $1 in
        --scale_anno|--scale-anno)
            scale_anno="$2"
            shift 2
            ;;
        --output_dir|--output-dir)
            output_dir="$2"
            shift 2
            ;;
        --config|--config_file|--config-file)
            config_file="$2"
            shift 2
            ;;
        --gene_list|--gene-list|--genes)
            gene_list="$2"
            shift 2
            ;;
        --expression_path|--expression-path|--expr)
            expression_path="$2"
            shift 2
            ;;
        --annotation_dir|--annotation-dir|--annotations)
            annotation_dir="$2"
            shift 2
            ;;
        --train_test|--train-test)
            train_test="$2"
            shift 2
            ;;
        --lr|--learning_rate|--learning-rate)
            lr="$2"
            shift 2
            ;;
        --epochs|--epoch)
            epochs="$2"
            shift 2
            ;;
        --phenotype|--pheno)
            phenotype="$2"
            shift 2
            ;;
        --random_genes|--random-genes)
            random_genes="$2"
            shift 2
            ;;
        --log_level|--log-level)
            log_level="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --scale_anno, --scale-anno BOOL          Scale annotation matrix (default: True)"
            echo "  --output_dir, --output-dir DIR          Output directory (default: seed_genes)"
            echo "  --config, --config_file FILE            Config YAML file (default: config_genewise.yaml)"
            echo "  --gene_list, --gene-list, --genes FILE   Gene list file (default: genes_list_seed.txt)"
            echo "  --expression_path, --expr FILE          Expression data file"
            echo "  --annotation_dir, --annotations DIR      Annotation directory"
            echo "  --train_test, --train-test BOOL         Enable train/test split (default: False)"
            echo "  --lr, --learning_rate FLOAT              Learning rate (default: 0.1)"
            echo "  --epochs, --epoch INT                    Number of epochs (default: 500)"
            echo "  --phenotype, --pheno FILE                Phenotype file"
            echo "  --random_genes, --random-genes BOOL      Use random genes (default: False)"
            echo "  --log_level, --log-level LEVEL           Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
            echo "  -h, --help                               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$random_genes" == "True" ]; then
    echo "Generating random genes list..."
    all_genes_file='/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src/scaling/gene_list_all.txt'
    # Generate random genes and join with commas (remove trailing newline and replace newlines with commas)
    gene_list=$(shuf -n 50 < "$all_genes_file" | tr '\n' ',' | sed 's/,$//')
    echo "Generated $(echo "$gene_list" | tr ',' '\n' | wc -l) random genes"
    
    output_dir=random_genes
fi


# scale annotation matrix
python -u run_parmigiano.py --config $config_file \
                         --gene_list $gene_list \
                         --expression_path $expression_path \
                         --annotation_dir $annotation_dir \
                         --output_dir $output_dir \
                         --scale_anno $scale_anno \
                         --train_test $train_test \
                         --lr $lr \
                         --epochs $epochs \
                         --log_level $log_level

# Move SLURM output files to the run output directory if it exists
if [ -n "$output_dir" ] && [ -f "${output_dir}/current_run_output_dir.txt" ]; then
    run_output_dir=$(cat "${output_dir}/current_run_output_dir.txt")
    if [ -n "$run_output_dir" ] && [ -d "$run_output_dir" ]; then
        # Get the current SLURM job ID
        if [ -n "$SLURM_JOB_ID" ]; then
            # Find and move .out and .err files (they might be in bash_outputs/seed_genes/ or current directory)
            # Check common locations for SLURM output files
            for slurm_file in "bash_outputs/seed_genes/slurm_${SLURM_JOB_ID}.out" \
                             "bash_outputs/seed_genes/slurm_${SLURM_JOB_ID}.err" \
                             "slurm_${SLURM_JOB_ID}.out" \
                             "slurm_${SLURM_JOB_ID}.err"\
                             "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.out" \
                             "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.err" \
                             "/gpfs/commons/home/vmazeeva/bash_outputs/parmigiano_seed_genes_${SLURM_JOB_ID}.out" \
                             "/gpfs/commons/home/vmazeeva/bash_outputs/parmigiano_seed_genes_${SLURM_JOB_ID}.err"; do
                if [ -f "$slurm_file" ]; then
                    mv "$slurm_file" "${run_output_dir}/$(basename "$slurm_file")"
                    echo "Moved $slurm_file to ${run_output_dir}/"
                fi
            done
        fi
        # Clean up the temporary file
        rm -f "${output_dir}/current_run_output_dir.txt"
    fi
fi
