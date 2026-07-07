#!/bin/bash
# 1) Re-run full_collapse pergene with correct joint tau/T (+ post-train + filter diagnostics).
# 2) Joint T-gate sweep on normG_collapse settings (top-200, collapsed + normalize_G).
#
# Usage:
#   bash submit_train_common01_Tgate_fixes.sh
#   SUBMIT_T_SWEEP=0 bash submit_train_common01_Tgate_fixes.sh   # pergene fix only
#   SUBMIT_FULL_COLLAPSE=0 bash submit_train_common01_Tgate_fixes.sh  # T sweep only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
PLOT_DIR="${SCRIPT_DIR}/../plotting"
MYOUT="/gpfs/commons/home/vmazeeva/firvTWAS_myout"

SUBMIT_FULL_COLLAPSE="${SUBMIT_FULL_COLLAPSE:-1}"
SUBMIT_T_SWEEP="${SUBMIT_T_SWEEP:-1}"

FULL_ROOT="${MYOUT}/train_common01_full_collapse"
FULL_JOINT="${FULL_ROOT}/joint"
FULL_PERGENE="${FULL_ROOT}/pergene"
FULL_CONFIG="${FULL_PERGENE}/config.yaml"

BRR_COMMON01="/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01/bayesian_ridge"
GENE_LIST="genes/top200_BRR_genes.txt"
CONFIG_BASE="${SRC_DIR}/config_base.yaml"

LR=0.01
EPOCHS=500
REFITS_SWEEP=3

submit_full_collapse_rerun() {
    echo "=== full_collapse: pergene chr array (correct joint tau/T) ==="
    local pergene_job
    pergene_job="$(
        sbatch --parsable \
            --job-name=fc_pergene_fix \
            --array=1-22 \
            "${SRC_DIR}/run_emmental_pergene.sh" \
            --config "${FULL_CONFIG}" \
            --pergene_output_dir "${FULL_PERGENE}" \
            --joint_output_dir "${FULL_JOINT}" \
            --collapsed_model true \
            --lr "${LR}" \
            --epochs 300
    )"
    echo "  pergene array job: ${pergene_job}"

    local post_job
    post_job="$(
        sbatch --parsable \
            --dependency=afterok:"${pergene_job}" \
            --job-name=fc_post_pergene \
            --array=1-22 \
            "${SRC_DIR}/run_emmental_post_train.sh" \
            --pergene_output_dir "${FULL_PERGENE}"
    )"
    echo "  post_pergene array job: ${post_job} (depends on ${pergene_job})"

    local filter_job
    filter_job="$(
        sbatch --parsable \
            --dependency=afterok:"${post_job}" \
            --job-name=fc_filter_diag \
            --output=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.out \
            --error=/gpfs/commons/home/vmazeeva/bash_outputs/slurm_%j.err \
            --time=24:00:00 --mem=32G --cpus-per-task=4 --partition=cpu \
            --wrap "CONFIG_DIR=${FULL_ROOT} bash ${PLOT_DIR}/run_filter_diagnostics.sh"
    )"
    echo "  filter diagnostics job: ${filter_job} (depends on ${post_job})"
}

submit_t_sweep_joint() {
    local label="$1"
    local out_root="$2"
    shift 2
    local extra_args=("$@")
    local joint_dir="${out_root}/joint"
    mkdir -p "${joint_dir}"

    echo ""
    echo "=== T sweep joint: ${label} -> ${joint_dir} ==="
    sbatch --parsable \
        --job-name="T_${label}" \
        --time=20:00:00 --mem=100G --cpus-per-task=8 --partition=cpu \
        "${SRC_DIR}/run_emmental_joint.sh" \
        --config "${CONFIG_BASE}" \
        --gene_list "${GENE_LIST}" \
        --joint_output_dir "${joint_dir}" \
        --brr_results_dir "${BRR_COMMON01}" \
        --train_test True \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS_SWEEP}" \
        --maf_threshold 0.01 \
        --normalize_G true \
        --collapsed_model true \
        "${extra_args[@]}"
}

run_t_sweep() {
    echo "=== normG_collapse T-gate joint sweep (${REFITS_SWEEP} refits, top-200) ==="

    submit_t_sweep_joint \
        "prior_2_8" \
        "${MYOUT}/train_common01_normG_collapse_Tprior_2_8" \
        --threshold_prior_alpha 2 \
        --threshold_prior_beta 8

    submit_t_sweep_joint \
        "prior_1_1" \
        "${MYOUT}/train_common01_normG_collapse_Tprior_1_1" \
        --threshold_prior_alpha 1 \
        --threshold_prior_beta 1

    submit_t_sweep_joint \
        "init_data_q25" \
        "${MYOUT}/train_common01_normG_collapse_Tinit_data_quantile" \
        --threshold_prior_alpha 2 \
        --threshold_prior_beta 20 \
        --threshold_init data_quantile \
        --threshold_init_quantile 0.25
}

if [[ "${SUBMIT_FULL_COLLAPSE}" == "1" ]]; then
    submit_full_collapse_rerun
fi

if [[ "${SUBMIT_T_SWEEP}" == "1" ]]; then
    run_t_sweep
fi

echo ""
echo "Done submitting."
