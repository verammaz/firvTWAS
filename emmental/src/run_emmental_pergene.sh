#!/bin/bash

#SBATCH --job-name=emmental-pergene
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=24G  
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --array=1-22

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Save script arguments
SCRIPT_ARGS=("$@")

# Activate conda environment (clear arguments first to avoid passing them to activate)
set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -- "${SCRIPT_ARGS[@]}"

cd /gpfs/commons/home/vmazeeva/firvTWAS/emmental/src


# Default values
pergene_output_dir=output/pergene
joint_output_dir=output/joint
config_file=config_base.yaml
clip_norm=10.0
lr=0.01
epochs=1000
log_level=INFO
refit=""
collapsed_model="false"
init_wg_zero="false"
wg_rhog_delta_guide="false"
early_stop=""
early_stop_min_epochs=""
early_stop_patience=""
early_stop_rel_tol=""

# if not submitted as an array job, run for chromosome 21
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
chromosome=21
else
chromosome=$SLURM_ARRAY_TASK_ID
fi

# Parse command line arguments with long-form options
while [[ $# -gt 0 ]]; do
    case $1 in
        --pergene_output_dir|--pergene-output-dir)
            pergene_output_dir="$2"
            shift 2
            ;;
        --joint_output_dir|--joint-output-dir)
            joint_output_dir="$2"
            shift 2
            ;;
        --config|--config_file|--config-file)
            config_file="$2"
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
        --log_level|--log-level)
            log_level="$2"
            shift 2
            ;;
        --clip_norm|--clip-norm)
            clip_norm="$2"
            shift 2
            ;;
        --refit)
            refit="true"
            shift
            ;;
        --collapsed_model|--collapsed-model)
            collapsed_model="$2"
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
        --early_stop|--early-stop)
            early_stop="$2"
            shift 2
            ;;
        --early_stop_min_epochs|--early-stop-min-epochs)
            early_stop_min_epochs="$2"
            shift 2
            ;;
        --early_stop_patience|--early-stop-patience)
            early_stop_patience="$2"
            shift 2
            ;;
        --early_stop_rel_tol|--early-stop-rel-tol)
            early_stop_rel_tol="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --pergene_output_dir, --pergene-output-dir DIR          Per-gene output directory (default: output/pergene)"
            echo "  --joint_output_dir, --joint-output-dir DIR          Joint output directory (default: output/joint)"
            echo "  --config, --config_file FILE            Config YAML file (default: config_genewise.yaml)"
            echo "  --train_test, --train-test BOOL         Enable train/test split (default: False)"
            echo "  --lr, --learning_rate FLOAT              Learning rate (default: 0.1)"
            echo "  --epochs, --epoch INT                    Number of epochs (default: 500)"
            echo "  --clip_norm, --clip-norm FLOAT          Clip gradient norm (default: 10.0)"
            echo "  --log_level, --log-level LEVEL           Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
            echo "  --refit                                  Match joint refits (run_N/chr*/; tau/T from joint run_N)"
            echo "  --collapsed_model BOOL                   Integrate out beta (collapsed likelihood)"
            echo "  --init_wg_zero BOOL                      Initialize w_g guide loc to 0 (default: false)"
            echo "  --early_stop BOOL                        Stop when ELBO plateaus (default on for collapsed)"
            echo "  --early_stop_min_epochs INT              Min epochs before early stop (default: 50)"
            echo "  --early_stop_patience INT                Patience epochs (default: 30)"
            echo "  --early_stop_rel_tol FLOAT               Relative improvement threshold (default: 1e-3)"
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


# check if joint output directory exists
if [ ! -d "$joint_output_dir" ]; then
    echo "Joint output directory $joint_output_dir does not exist. Exiting..."
    exit 1
fi

# Collapsed per-gene: only w_g/rho_g are fit; ELBO plateaus quickly (~100-150 epochs).
if [ "$collapsed_model" = "true" ] && [ "$epochs" -eq 1000 ]; then
    epochs=300
    echo "Collapsed model: defaulting to epochs=300 (override with --epochs)."
fi

pergene_py_args=(
    --config "$config_file"
    --pergene_output_dir "$pergene_output_dir"
    --joint_output_dir "$joint_output_dir"
    --chromosome "$chromosome"
    --lr "$lr"
    --epochs "$epochs"
    --clip_norm "$clip_norm"
    --log_level "$log_level"
)
if [ "$refit" = "true" ]; then
    pergene_py_args+=(--refit true)
fi
pergene_py_args+=(--collapsed_model "$collapsed_model")
if [ "$init_wg_zero" = "true" ]; then
    pergene_py_args+=(--init_wg_zero true)
fi
if [ "$wg_rhog_delta_guide" = "true" ]; then
    pergene_py_args+=(--wg_rhog_delta_guide true)
fi
if [ -n "$early_stop" ]; then
    pergene_py_args+=(--early_stop "$early_stop")
fi
if [ -n "$early_stop_min_epochs" ]; then
    pergene_py_args+=(--early_stop_min_epochs "$early_stop_min_epochs")
fi
if [ -n "$early_stop_patience" ]; then
    pergene_py_args+=(--early_stop_patience "$early_stop_patience")
fi
if [ -n "$early_stop_rel_tol" ]; then
    pergene_py_args+=(--early_stop_rel_tol "$early_stop_rel_tol")
fi

python -u emmental_pergene.py "${pergene_py_args[@]}"
                        

# Move SLURM output files to the run output directory if it exists
if [ -n "$pergene_output_dir" ]; then
    # Get the current SLURM job ID
    if [ -n "$SLURM_JOB_ID" ]; then
        # Find and move .out and .err files (they might be in bash_outputs/seed_genes/ or current directory)
        # Check common locations for SLURM output files
        for slurm_file in "slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
                            "slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
                            "/gpfs/commons/home/vmazeeva/bash_outputs/slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"; do
            if [ -f "$slurm_file" ]; then
                mv "$slurm_file" "${pergene_output_dir}/$(basename "$slurm_file")"
                echo "Moved $slurm_file to ${pergene_output_dir}/"
            fi
        done
    fi
fi