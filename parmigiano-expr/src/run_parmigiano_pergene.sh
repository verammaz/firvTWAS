#!/bin/bash

#SBATCH --job-name=parmigiano
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=100G  
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

cd /gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src


# Default values
pergene_output_dir=output/pergene
joint_output_dir=output/joint
config_file=config_base.yaml
clip_norm=10.0
lr=0.01
epochs=1000
log_level=INFO

chromosome=$SLURM_ARRAY_TASK_ID

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
        --refits|--refits-number)   
            refits="$2"
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


python -u parmigiano_pergene.py --config $config_file \
                         --pergene_output_dir $pergene_output_dir \
                         --joint_output_dir $joint_output_dir \
                         --chromosome $chromosome \
                         --lr $lr \
                         --epochs $epochs \
                         --clip_norm $clip_norm \
                         --log_level $log_level
                        

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