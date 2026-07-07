#!/bin/bash

#SBATCH --job-name=emmental-joint
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=100G
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

cd /gpfs/commons/home/vmazeeva/firvTWAS/emmental/src

# Model variants (use a distinct --joint_output_dir per run; flags are saved in config.yaml):
#   baseline:     --no_wg false --no_rhog false 
#   no_wg:        --no_wg true  --no_rhog false 
#   no_rhog:      --no_wg false --no_rhog true 
#   no_wg+no_rhog: --no_wg true  --no_rhog true 

# Default values
joint_output_dir=output/joint
config_file=config_base.yaml
gene_list=genes/top200_BRR_genes.txt
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
threshold_init=""
threshold_init_quantile=""
maf_beta=1
maf_threshold=""
no_wg="false"
no_rhog="false"
normalize_G="false"
collapsed_model="false"
no_T="false"
init_wg_zero="false"
wg_rhog_delta_guide="false"

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
        --no_wg|--no-wg)
            no_wg="$2"
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
        --no_rhog|--no-rhog)
            no_rhog="$2"
            shift 2
            ;;
        --normalize_G|--normalize-G)
            normalize_G="$2"
            shift 2
            ;;
        --collapsed_model|--collapsed-model)
            collapsed_model="$2"
            shift 2
            ;;
        --no_T|--no-T)
            no_T="$2"
            shift 2
            ;;
        --init_wg_zero|--init-wg-zero)
            init_wg_zero="$2"
            shift 2
            ;;
        --wg_rhog_delta_guide|--wg-rhog-delta-guide)
            wg_rhog_delta_guide="$2"
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
        --threshold_init|--threshold-init)
            threshold_init="$2"
            shift 2
            ;;
        --threshold_init_quantile|--threshold-init-quantile)
            threshold_init_quantile="$2"
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
            echo "  --no_wg, --no-wg BOOL                     Fix w_g=1 (do not sample w_g)"
            echo "  --no_rhog, --no-rhog BOOL                 Fix rho_g=0; mu=w*lambda, sigma=|w*lambda|"
            echo "  --init_wg_zero, --init-wg-zero BOOL       Initialize w_g guide loc to 0 (default: false)"
            echo "  --wg_rhog_delta_guide BOOL                AutoDelta guide for w_g/rho_g (default: false)"
            echo "  --normalize_G, --normalize-G BOOL        Normalize G matrix (default: False)"
            echo "  --no_T, --no-T BOOL                      Disable annotation gate threshold (T=0; all |Z·tau1| pass)"
            echo "  --refits, --refits-number INT           Number of refits (default: 10)"
            echo "  --clip_norm, --clip-norm FLOAT          Clip gradient norm (default: 10.0)"
            echo "  --annotations, --annotations-list LIST    List of annotations to use (default: None)"
            echo "  --threshold_prior_alpha, --threshold-prior-alpha FLOAT   Alpha for threshold Beta prior (default: 2.0)"
            echo "  --threshold_prior_beta, --threshold-prior-beta FLOAT     Beta for threshold Beta prior (default: 20.0)"
            echo "  --threshold_init STR                     prior_mean | prior_mode | data_quantile (default: prior_mean)"
            echo "  --threshold_init_quantile FLOAT          Quantile for data_quantile init (default: 0.25)"
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


maf_args=()
if [ -n "$maf_threshold" ] && [ "$maf_threshold" != "None" ] && [ "$maf_threshold" != "null" ]; then
  maf_args=(--maf_threshold "$maf_threshold")
fi

threshold_init_args=()
if [ -n "$threshold_init" ]; then
  threshold_init_args=(--threshold_init "$threshold_init")
fi
threshold_quantile_args=()
if [ -n "$threshold_init_quantile" ]; then
  threshold_quantile_args=(--threshold_init_quantile "$threshold_init_quantile")
fi

mkdir -p "$joint_output_dir"

move_slurm_logs() {
    if [ -z "$joint_output_dir" ] || [ -z "${SLURM_JOB_ID:-}" ]; then
        return
    fi
    for slurm_file in "slurm_${SLURM_JOB_ID}.out" \
                        "slurm_${SLURM_JOB_ID}.err" \
                        "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.out" \
                        "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.err"; do
        if [ -f "$slurm_file" ]; then
            mv "$slurm_file" "${joint_output_dir}/$(basename "$slurm_file")"
            echo "Moved $slurm_file to ${joint_output_dir}/"
        fi
    done
}
trap move_slurm_logs EXIT

python -u emmental_joint.py --config $config_file \
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
                         --no_rhog $no_rhog \
                        "${annotations_args[@]}" \
                         --threshold_prior_alpha $threshold_prior_alpha \
                         --threshold_prior_beta $threshold_prior_beta \
                         "${threshold_init_args[@]}" \
                         "${threshold_quantile_args[@]}" \
                         --no_wg $no_wg \
                         --maf_beta $maf_beta \
                         "${maf_args[@]}" \
                         --normalize_G $normalize_G \
                         --collapsed_model $collapsed_model \
                         --no_T $no_T \
                         --init_wg_zero $init_wg_zero \
                         --wg_rhog_delta_guide $wg_rhog_delta_guide \
