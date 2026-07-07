#!/bin/bash
# Post-train full-panel betas + R².
#
# Two modes (per-gene wins if both dirs are passed):
#   Post-joint:   --joint_output_dir .../full/joint  -> .../full/post_joint/
#   Post-pergene: --pergene_output_dir .../full/pergene  -> .../full/post_pergene/
#
# Genome-wide per-gene (parallel by chromosome):
#   sbatch --array=1-22 \
#       --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.out \
#       --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j_%a.err \
#       run_emmental_post_train.sh --pergene_output_dir .../full/pergene
#   -> .../full/post_pergene/chr1/ ... chr22/  (one array task per chr)
#
# Single chromosome (no array):
#   sbatch run_emmental_post_train.sh --pergene_output_dir .../pergene --chromosome 10
#
# Examples:
#   sbatch run_emmental_post_train.sh --joint_output_dir .../full/joint
#   sbatch run_emmental_post_train.sh --pergene_output_dir .../full/pergene \
#       --gene_list .../genes/top200_BRR_genes.txt
#
#SBATCH --job-name=emmental-post-train
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

SCRIPT_ARGS=("$@")
set --
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
set -- "${SCRIPT_ARGS[@]}"

cd /gpfs/commons/home/vmazeeva/firvTWAS/emmental/src

joint_output_dir=""
pergene_output_dir=""
out_dir=""
skip_r2=""
joint_run=""
gene_list=""
chromosome=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    chromosome="${SLURM_ARRAY_TASK_ID}"
fi

usage() {
    cat <<'EOF'
Usage: run_emmental_post_train.sh [OPTIONS]

  --joint_output_dir DIR    Post-joint mode (when --pergene_output_dir is omitted)
  --pergene_output_dir DIR  Post-per-gene mode (wins if both dirs are passed)
  --out_dir DIR             Override output directory
  --chromosome N            Process chrN only (auto-set from SLURM_ARRAY_TASK_ID)
  --gene_list FILE          Only process genes in this list
  --skip_r2                 Skip R² computation
  --joint_run N             Post-joint only: use joint run_N for common beta

  Genome-wide per-gene: sbatch --array=1-22 run_emmental_post_train.sh --pergene_output_dir DIR
  With --out_dir DIR and --array, output goes to DIR/chrN/ per task.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --joint_output_dir|--joint-output-dir)
            joint_output_dir="$2"
            shift 2
            ;;
        --pergene_output_dir|--pergene-output-dir)
            pergene_output_dir="$2"
            shift 2
            ;;
        --out_dir|--out-dir)
            out_dir="$2"
            shift 2
            ;;
        --chromosome|--chr)
            chromosome="$2"
            shift 2
            ;;
        --gene_list|--gene-list)
            gene_list="$2"
            shift 2
            ;;
        --skip_r2)
            skip_r2="--skip_r2"
            shift
            ;;
        --joint_run)
            joint_run="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -n "$pergene_output_dir" ]]; then
    mode="pergene"
elif [[ -n "$joint_output_dir" ]]; then
    mode="joint"
else
    echo "Provide --joint_output_dir (post-joint) or --pergene_output_dir (post-per-gene)." >&2
    usage >&2
    exit 1
fi

if [[ -n "$out_dir" ]]; then
    POST_DIR="$out_dir"
    if [[ -n "$chromosome" && "$POST_DIR" != */chr"${chromosome}" ]]; then
        POST_DIR="${POST_DIR}/chr${chromosome}"
    fi
elif [[ "$mode" == "pergene" ]]; then
    POST_DIR="$(dirname "${pergene_output_dir}")/post_pergene"
    [[ -n "$chromosome" ]] && POST_DIR="${POST_DIR}/chr${chromosome}"
else
    POST_DIR="$(dirname "${joint_output_dir}")/post_joint"
fi
mkdir -p "${POST_DIR}"

JOB_START_SEC=$(date +%s)

finalize_slurm_logs() {
    if [[ -n "${SLURM_LOGS_FINALIZED:-}" ]]; then
        return 0
    fi
    SLURM_LOGS_FINALIZED=1

    [[ -n "${SLURM_JOB_ID:-}" ]] || return 0

    local elapsed=$(( $(date +%s) - JOB_START_SEC ))
    local runtime_line memory_line
    runtime_line=$(awk -v e="$elapsed" 'BEGIN { printf "Total runtime: %.2f minutes (%.1f sec)", e / 60.0, e }')
    memory_line="Peak memory: unknown"

    if command -v sacct >/dev/null 2>&1; then
        local jid="${SLURM_JOB_ID}"
        [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && jid="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        local maxrss
        maxrss=$(sacct -j "$jid" -X -n -P --format=MaxRSS 2>/dev/null | awk -F'|' 'NF { print $1 }' | tail -1)
        if [[ -n "$maxrss" && "$maxrss" != "0" ]]; then
            if [[ "$maxrss" =~ ^([0-9.]+)K$ ]]; then
                memory_line=$(awk -v kb="${BASH_REMATCH[1]}" 'BEGIN { printf "Peak memory: %.2f GB", kb / 1024.0 / 1024.0 }')
            else
                memory_line="Peak memory: ${maxrss}"
            fi
        fi
    fi
    if [[ "$memory_line" == "Peak memory: unknown" && -r /proc/self/status ]]; then
        local vmhwm_kb
        vmhwm_kb=$(awk '/VmHWM:/ { print $2 }' /proc/self/status 2>/dev/null)
        if [[ -n "$vmhwm_kb" ]]; then
            memory_line=$(awk -v kb="$vmhwm_kb" 'BEGIN { printf "Peak memory: %.2f GB", kb / 1024.0 / 1024.0 }')
        fi
    fi

    local dest_base="slurm_${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
    local -a search_dirs=("${PWD}" "/gpfs/commons/home/vmazeeva/bash_outputs")
    local -a names=(
        "${dest_base}.out"
        "${dest_base}.err"
        "${dest_base}.log"
        "slurm_${SLURM_JOB_ID}.out"
        "slurm_${SLURM_JOB_ID}.err"
        "slurm_${SLURM_JOB_ID}.log"
    )
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        names+=(
            "slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
            "slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
            "slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
        )
    fi

    declare -A moved=()
    local name dir src dest
    for name in "${names[@]}"; do
        for dir in "${search_dirs[@]}"; do
            src="${dir}/${name}"
            [[ -f "$src" ]] || continue
            [[ -n "${moved[$src]:-}" ]] && continue
            moved["$src"]=1
            dest="${POST_DIR}/${dest_base}.${name##*.}"
            printf '%s\n%s\n' "$runtime_line" "$memory_line" >> "$src"
            mv "$src" "$dest"
        done
    done
}
trap finalize_slurm_logs EXIT

py_args=(--maf_threshold 0.01)
if [[ "$mode" == "pergene" ]]; then
    py_args+=(--pergene_output_dir "${pergene_output_dir}")
    [[ -n "$joint_output_dir" ]] && py_args+=(--joint_output_dir "${joint_output_dir}")
else
    py_args+=(--joint_output_dir "${joint_output_dir}")
fi
py_args+=(--out_dir "${POST_DIR}")
[[ -n "$chromosome" ]] && py_args+=(--chromosome "$chromosome")
[[ -n "$gene_list" ]] && py_args+=(--gene_list "$gene_list")
[[ -n "$skip_r2" ]] && py_args+=("$skip_r2")
[[ -n "$joint_run" ]] && py_args+=(--joint_run "$joint_run")

echo "Mode: ${mode}"
[[ -n "$chromosome" ]] && echo "Chromosome: chr${chromosome}"
echo "Output: ${POST_DIR}"

python -u emmental_post_train_betas.py "${py_args[@]}"
