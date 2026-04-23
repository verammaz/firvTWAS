#!/bin/bash

#SBATCH --job-name=parmigiano
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=80G  # 200 genes using max ~60GB RAM
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

## TODO: too many command line arguments, need to simplify or use config file 

# Default values
scale_anno=False
output_dir=seed_genes_minmax
config_file=config_base.yaml
gene_list=genes_list_seed.txt
expression_path=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/tpm_genes_subset.tsv
annotation_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_raw/
brr_results_dir=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
train_test=True
use_clip_norm=True
clip_norm=10.0
lr=0.1
epochs=500
random_genes=False
log_level=INFO
no_filter=False
burden=False
skat=False
refits=100
no_wg=False
no_rhog=False
use_brr=True
scale_center=True
tau12=False
tau1_normal_prior=False
tau2_normal_prior=False
chrombpnet_dist_only=False
chrombpnet_dist_only_cfg_num=1
annotations=()
negative_annotations=False
negative_gate_mode=hard_abs
negative_gate_sharpness=20.0
lin2_clip=
tau2_link=exp
wg_positive=False
threshold_prior_alpha=2.0
threshold_prior_beta=20.0
common_variants_only=False
tau1_intercept=False
maf_beta=1

# TODO: match scale_center with brr_results_dir if use_brr is True

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
        --use_clip_norm|--use-clip-norm)
            use_clip_norm="$2"
            shift 2
            ;;
        --clip_norm|--clip-norm)
            clip_norm="$2"
            shift 2
            ;;
        --no_filter|--no-filter)
            no_filter="$2"
            shift 2
            ;;
        --burden|--burden-test)
            burden="$2"
            shift 2
            ;;
        --skat|--skat-test)
            skat="$2"
            shift 2
            ;;
        --refits|--refits-number)
            refits="$2"
            shift 2
            ;;
        --no_wg|--no-wg)
            no_wg="$2"
            shift 2
            ;;
        --no_rhog|--no-rhog)
            no_rhog="$2"
            shift 2
            ;;
        --use_brr|--use-brr)
            use_brr="$2"
            shift 2
            ;;
        --scale_center|--scale-center)
            scale_center="$2"
            shift 2
            ;;
        --tau12|--tau12)
            tau12="$2"
            shift 2
            ;;
        --tau1_normal_prior|--tau1-normal-prior)
            tau1_normal_prior="$2"
            shift 2
            ;;
        --tau2_normal_prior|--tau2-normal-prior)
            tau2_normal_prior="$2"
            shift 2
            ;;
        --chrombpnet_dist_only|--chrombpnet-dist-only)
            chrombpnet_dist_only="$2"
            shift 2
            ;;
        --chrombpnet_dist_only_cfg_num|--chrombpnet-dist-only-cfg-num)
            chrombpnet_dist_only_cfg_num="$2"
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
        --negative_annotations|--negative-annotations)
            negative_annotations="$2"
            shift 2
            ;;
        --negative_gate_mode|--negative-gate-mode)
            negative_gate_mode="$2"
            shift 2
            ;;
        --negative_gate_sharpness|--negative-gate-sharpness)
            negative_gate_sharpness="$2"
            shift 2
            ;;
        --lin2_clip|--lin2-clip)
            lin2_clip="$2"
            shift 2
            ;;
        --tau2_link|--tau2-link)
            tau2_link="$2"
            shift 2
            ;;
        --wg_positive|--wg-positive)
            wg_positive="$2"
            shift 2
            ;;
        --threshold_prior_alpha|--threshold-prior-alpha)
            threshold_prior_alpha="$2"
            shift 2
            ;;
        --threshold_prior_beta|--threshold-prior-beta)
            threshold_prior_beta="$2"
            shift 2
            ;;
        --common_variants_only|--common-variants-only)
            common_variants_only="$2"
            shift 2
            ;;
        --tau1_intercept|--tau1-intercept)
            tau1_intercept="$2"
            shift 2
            ;;
        --maf_beta|--maf-beta)
            maf_beta="$2"
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
            echo "  --brr_results_dir, --brr-results-dir DIR  BRR results directory"
            echo "  --train_test, --train-test BOOL         Enable train/test split (default: False)"
            echo "  --lr, --learning_rate FLOAT              Learning rate (default: 0.1)"
            echo "  --epochs, --epoch INT                    Number of epochs (default: 500)"
            echo "  --phenotype, --pheno FILE                Phenotype file"
            echo "  --random_genes, --random-genes BOOL      Use random genes (default: False)"
            echo "  --use_clip_norm, --use-clip-norm BOOL    Use clip gradient norm (default: True)"
            echo "  --clip_norm, --clip-norm FLOAT          Clip gradient norm (default: 10.0)"
            echo "  --no_filter, --no-filter BOOL            Don't learn filter (default: False)"
            echo "  --burden, --burden-test BOOL            Burden test (default: False)"
            echo "  --skat, --skat-test BOOL                SKAT test (default: False)"
            echo "  --refits, --refits-number INT           Number of refits (default: 1)"
            echo "  --no_wg, --no-wg BOOL                    Remove w_g from model (default: False)"
            echo "  --no_rhog, --no-rhog BOOL                Remove rho_g from model(default: False)"
            echo "  --use_brr, --use-brr BOOL                Use BRR results(default: True)"
            echo "  --scale_center, --scale-center BOOL      Scale expression matrix by center and scale (default: True)"
            echo "  --tau12, --tau12 BOOL                    Use tau1 and tau2 for nonlinear annotation interaction (default: False)"
            echo "  --tau1_normal_prior, --tau1-normal-prior BOOL    Use normal prior for tau1 (default: False)"
            echo "  --tau2_normal_prior, --tau2-normal-prior BOOL    Use normal prior for tau2 (default: False)"
            echo "  --chrombpnet_dist_only, --chrombpnet-dist-only BOOL    Use chrombpnet and dist_to_TSS annotations only (default: False)"
            echo "  --chrombpnet_dist_only_cfg_num, --chrombpnet-dist-only-cfg-num INT    Configuration number for chrombpnet and dist_to_TSS annotations only (default: 1)"
            echo "  --annotations, --annotations-list LIST    List of annotations to use (default: None)"
            echo "  --negative_annotations, --negative-annotations BOOL    Use negative annotations (default: False)"
            echo "  --negative_gate_mode, --negative-gate-mode STR         Negative gate mode: hard_abs|smooth_abs|signed_relu_abs (default: hard_abs)"
            echo "  --negative_gate_sharpness, --negative-gate-sharpness FLOAT   Sharpness for smooth_abs gate (default: 20.0)"
            echo "  --lin2_clip, --lin2-clip FLOAT              Clip abs(lin2=Z@tau2) in nonlinear mode (default: unset/no clip)"
            echo "  --tau2_link, --tau2-link STR                Modulation link: exp|softplus (default: exp)"
            echo "  --wg_positive, --wg-positive BOOL            Use positive LogNormal prior for w_g (default: False)"
            echo "  --threshold_prior_alpha, --threshold-prior-alpha FLOAT   Alpha for threshold Beta prior (default: 2.0)"
            echo "  --threshold_prior_beta, --threshold-prior-beta FLOAT     Beta for threshold Beta prior (default: 20.0)"
            echo "  --common_variants_only, --common-variants-only BOOL    Use common variants only (default: False)"
            echo "  --tau1_intercept, --tau1-intercept BOOL    Use intercept for tau1 (default: False)"
            echo "  --maf_beta, --maf-beta FLOAT              Beta parameter for MAF weights (default: 1)"
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

if [ "$chrombpnet_dist_only" == "True" ]; then
    echo "Using chrombpnet and dist_to_TSS annotations only (config ${chrombpnet_dist_only_cfg_num})..."
    case "$chrombpnet_dist_only_cfg_num" in
        1|2)
            echo "Need raw annotations directory... (will override)"
            annotation_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_raw/
            ;;
        3)
            echo "Need raw annotations directory... (will override)"
            annotation_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_minmax/
            ;;
        *)
            echo "Invalid --chrombpnet_dist_only_cfg_num: $chrombpnet_dist_only_cfg_num (expected 1, 2, or 3)"
            exit 1
            ;;
    esac
    echo "Setting no_filter=True..."
    no_filter=True
fi

if [ "$tau2_normal_prior" == "True" ] && [ "$tau12" == "False" ]; then
    echo "Using normal prior for tau2..."
    echo "Require nonlinear mode (overriding tau12=False)..."
    tau12=True
fi

if [ "$tau1_normal_prior" == "True" ] && [ "$tau12" == "False" ]; then
    echo "Using normal prior for tau1..."
    echo "Require nonlinear mode (overriding tau12=False)..."
    tau12=True
fi

annotations_args=()
if [ "${#annotations[@]}" -gt 0 ]; then
    echo "Using annotations: ${annotations[@]}"
    echo "Setting annotation_dir to raw annotations directory..."
    annotation_dir=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_raw/
    annotations_args=(--annotations "${annotations[@]}")
    negative_annotations=True

fi

lin2_clip_args=()
if [ -n "${lin2_clip}" ]; then
    lin2_clip_args=(--lin2_clip "${lin2_clip}")
fi

# another check if experiments already run / is running --> exit if so
if [ -d "$output_dir" ]; then
    echo "Output directory $output_dir already exists. Exiting..."
    exit 1
fi



python -u parmigiano_joint.py --config $config_file \
                         --gene_list $gene_list \
                         --expression_path $expression_path \
                         --annotation_dir $annotation_dir \
                         --output_dir $output_dir \
                         --scale_anno $scale_anno \
                         --train_test $train_test \
                         --lr $lr \
                         --epochs $epochs \
                         --use_clip_norm $use_clip_norm \
                         --no_filter $no_filter \
                         --clip_norm $clip_norm \
                         --log_level $log_level \
                         --no_wg $no_wg \
                         --no_rhog $no_rhog \
                         --use_brr $use_brr \
                         --brr_results_dir $brr_results_dir \
                         --refits $refits \
                         --scale_center $scale_center \
                         --tau12 $tau12 \
                         --tau1_normal_prior $tau1_normal_prior \
                         --tau2_normal_prior $tau2_normal_prior \
                         --chrombpnet_dist_only $chrombpnet_dist_only \
                         --chrombpnet_dist_only_cfg_num $chrombpnet_dist_only_cfg_num \
                         --burden $burden \
                         "${annotations_args[@]}" \
                         --negative_annotations $negative_annotations \
                         --negative_gate_mode $negative_gate_mode \
                         --negative_gate_sharpness $negative_gate_sharpness \
                         "${lin2_clip_args[@]}" \
                         --tau2_link $tau2_link \
                         --wg_positive $wg_positive \
                         --threshold_prior_alpha $threshold_prior_alpha \
                         --threshold_prior_beta $threshold_prior_beta \
                         --common_variants_only $common_variants_only \
                         --tau1_intercept $tau1_intercept \
                         --maf_beta $maf_beta \

# Move SLURM output files to the run output directory if it exists
if [ -n "$output_dir" ]; then
    # Get the current SLURM job ID
    if [ -n "$SLURM_JOB_ID" ]; then
        # Find and move .out and .err files (they might be in bash_outputs/seed_genes/ or current directory)
        # Check common locations for SLURM output files
        for slurm_file in "bash_outputs/seed_genes/slurm_${SLURM_JOB_ID}.out" \
                            "bash_outputs/seed_genes/slurm_${SLURM_JOB_ID}.err" \
                            "slurm_${SLURM_JOB_ID}.out" \
                            "slurm_${SLURM_JOB_ID}.err" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.out" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}.err" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/parmigiano_seed_genes_${SLURM_JOB_ID}.out" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/parmigiano_seed_genes_${SLURM_JOB_ID}.err"; do
            if [ -f "$slurm_file" ]; then
                mv "$slurm_file" "${output_dir}/$(basename "$slurm_file")"
                echo "Moved $slurm_file to ${output_dir}/"
            fi
        done
    fi
fi