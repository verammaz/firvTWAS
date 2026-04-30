#!/bin/bash

#SBATCH --job-name=parmigiano
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
joint_output_dir=output/joint
config_file=config_base.yaml
gene_list=200genes_list_seed_random_full.txt
expression_path=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/tpm_genes_subset.tsv
annotation_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_scaled/
brr_results_dir=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
train_test=True
clip_norm=10.0
lr=0.01
refits=10
epochs=500
log_level=INFO
tau1_normal_prior=False
annotations=()
threshold_prior_alpha=2.0
threshold_prior_beta=20.0
maf_beta=1
maf_threshold=""
lin2_clip=""
gate_mode="hard_abs"
gate_sharpness="20.0"


# Parse command line arguments with long-form options
while [[ $# -gt 0 ]]; do
    case $1 in
        --joint_output_dir|--joint-output-dir)
            joint_output_dir="$2"
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
        --annotation_dir|--annotation-dir)
            annotation_dir="$2"
            shift 2
            ;;
        --brr_results_dir|--brr-results-dir)
            brr_results_dir="$2"
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
        --refits|--refits-number)
            refits="$2"
            shift 2
            ;;
        --epochs|--epoch)
            epochs="$2"
            shift 2
            ;;
        --lin2_clip|--lin2-clip)
            lin2_clip="$2"
            shift 2
            ;;
        --gate_mode|--gate-mode)
            gate_mode="$2"
            shift 2
            ;;
        --gate_sharpness|--gate-sharpness)
            gate_sharpness="$2"
            shift 2
            ;;
        --log_level|--log-level)
            log_level="$2"
            shift 2
            ;;
        --clip_norm|--clip-norm)
            clip_norm="$2"
            shift 2
            ;;
        --refits|--refits-number)   
            refits="$2"
            shift 2
            ;;
        --tau1_normal_prior|--tau1-normal-prior)
            tau1_normal_prior="$2"
            shift 2
            ;;
        --annotations|--annotations-list)
            shift
            annotations=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                annotations+=("$1")
                shift
            done
            ;;
        --threshold_prior_alpha|--threshold-prior-alpha)
            threshold_prior_alpha="$2"
            shift 2
            ;;
        --threshold_prior_beta|--threshold-prior-beta)
            threshold_prior_beta="$2"
            shift 2
            ;;
        --maf_beta|--maf-beta)
            maf_beta="$2"
            shift 2
            ;;
        --maf_threshold|--maf-threshold)
            maf_threshold="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --joint_output_dir, --joint-output-dir DIR          Joint output directory (default: seed_genes)"
            echo "  --config, --config_file FILE            Config YAML file (default: config_genewise.yaml)"
            echo "  --gene_list, --gene-list, --genes FILE   Gene list file (default: genes_list_seed.txt)"
            echo "  --expression_path, --expr FILE          Expression data file"
            echo "  --annotation_dir, --annotations DIR      Annotation directory"
            echo "  --brr_results_dir, --brr-results-dir DIR  BRR results directory"
            echo "  --train_test, --train-test BOOL         Enable train/test split (default: False)"
            echo "  --lr, --learning_rate FLOAT              Learning rate (default: 0.1)"
            echo "  --epochs, --epoch INT                    Number of epochs (default: 500)"
            echo "  --lin2_clip, --lin2-clip FLOAT          Clip lin2 before exp (default: None)"
            echo "  --gate_mode, --gate-mode STR            Gate mode: hard_abs or smooth_abs (default: hard_abs)"
            echo "  --gate_sharpness, --gate-sharpness FLOAT  Sharpness for smooth_abs gate (default: 20.0)"
            echo "  --refits, --refits-number INT           Number of refits (default: 10)"
            echo "  --clip_norm, --clip-norm FLOAT          Clip gradient norm (default: 10.0)"
            echo "  --annotations, --annotations-list LIST    List of annotations to use (default: None)"
            echo "  --threshold_prior_alpha, --threshold-prior-alpha FLOAT   Alpha for threshold Beta prior (default: 2.0)"
            echo "  --threshold_prior_beta, --threshold-prior-beta FLOAT     Beta for threshold Beta prior (default: 20.0)"
            echo "  --maf_beta, --maf-beta FLOAT              Beta parameter for MAF weights (default: 1)"
            echo "  --maf_threshold, --maf-threshold FLOAT   MAF cutoff when maf_threshold is not None (default: None)"
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


annotations_args=()
if [ "${#annotations[@]}" -gt 0 ]; then
    echo "Using annotations: ${annotations[@]}"
    echo "Setting annotation_dir to raw annotations directory..."
    annotation_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_raw/
    annotations_args=(--annotations "${annotations[@]}")
    negative_annotations=True

fi

maf_args=()
if [ -n "$maf_threshold" ] && [ "$maf_threshold" != "None" ] && [ "$maf_threshold" != "null" ]; then
  maf_args=(--maf_threshold "$maf_threshold")
fi

lin2_clip_args=()
if [ -n "$lin2_clip" ] && [ "$lin2_clip" != "None" ] && [ "$lin2_clip" != "null" ]; then
  lin2_clip_args=(--lin2_clip "$lin2_clip")
fi


python -u parmigiano_joint.py --config $config_file \
                         --gene_list $gene_list \
                         --expression_path $expression_path \
                         --annotation_dir $annotation_dir \
                         --joint_output_dir $joint_output_dir \
                         --train_test $train_test \
                         --lr $lr \
                         --epochs $epochs \
                         --clip_norm $clip_norm \
                         --log_level $log_level \
                         --brr_results_dir $brr_results_dir \
                         --refits $refits \
                         --tau1_normal_prior $tau1_normal_prior \
                         "${annotations_args[@]}" \
                         --threshold_prior_alpha $threshold_prior_alpha \
                         --threshold_prior_beta $threshold_prior_beta \
                         --gate_mode $gate_mode \
                         --gate_sharpness $gate_sharpness \
                         --maf_beta $maf_beta \
                         "${maf_args[@]}" \
                         "${lin2_clip_args[@]}" \

# Move SLURM output files to the run output directory if it exists
if [ -n "$joint_output_dir" ]; then
    # Get the current SLURM job ID
    if [ -n "$SLURM_JOB_ID" ]; then
        # Find and move .out and .err files (they might be in bash_outputs/seed_genes/ or current directory)
        # Check common locations for SLURM output files
        for slurm_file in "slurm_${SLURM_JOB_ID}.out" \
                            "slurm_${SLURM_JOB_ID}.err" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.out" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.err"; do
            if [ -f "$slurm_file" ]; then
                mv "$slurm_file" "${joint_output_dir}/$(basename "$slurm_file")"
                echo "Moved $slurm_file to ${joint_output_dir}/"
            fi
        done
    fi
fi