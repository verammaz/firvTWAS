#!/bin/bash
# Same setup as train_common01_normG_collapse (collapsed + normalize_G, common01 BRR, top-200 joint,
# genome-wide pergene chr array) but with --init_wg_zero true.
#
# Usage:
#   bash submit_train_common01_normG_collapse_initwg0.sh
#   SUBMIT_PERGENE=0 bash submit_train_common01_normG_collapse_initwg0.sh   # joint only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
MYOUT="/gpfs/commons/home/vmazeeva/firvTWAS_myout"

ROOT="${MYOUT}/train_common01_normG_collapse_initwg0"
JOINT="${ROOT}/joint"
PERGENE="${ROOT}/pergene"

BRR_COMMON01="/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01/bayesian_ridge"
GENE_LIST="genes/top200_BRR_genes.txt"
CONFIG_BASE="${SRC_DIR}/config_base.yaml"

LR=0.01
EPOCHS_JOINT=500
EPOCHS_PERGENE=300
REFITS=10
SUBMIT_PERGENE="${SUBMIT_PERGENE:-1}"

mkdir -p "${JOINT}" "${PERGENE}"

echo "=== Joint: normG_collapse + init_wg_zero -> ${JOINT} ==="
JOINT_JOB="$(
    sbatch --parsable \
        --job-name=nc_initwg0_joint \
        --time=20:00:00 --mem=100G --cpus-per-task=8 --partition=cpu \
        "${SRC_DIR}/run_emmental_joint.sh" \
        --config "${CONFIG_BASE}" \
        --gene_list "${GENE_LIST}" \
        --joint_output_dir "${JOINT}" \
        --brr_results_dir "${BRR_COMMON01}" \
        --train_test True \
        --lr "${LR}" \
        --epochs "${EPOCHS_JOINT}" \
        --refits "${REFITS}" \
        --maf_threshold 0.01 \
        --normalize_G true \
        --collapsed_model true \
        --init_wg_zero true
)"
echo "  joint job: ${JOINT_JOB}"

if [[ "${SUBMIT_PERGENE}" == "1" ]]; then
    echo "=== Pergene chr array (depends on joint) -> ${PERGENE} ==="
    PERGENE_JOB="$(
        sbatch --parsable \
            --dependency=afterok:"${JOINT_JOB}" \
            --job-name=nc_initwg0_pergene \
            --array=1-22 \
            --time=20:00:00 --mem=24G --cpus-per-task=8 --partition=cpu \
            "${SRC_DIR}/run_emmental_pergene.sh" \
            --config "${CONFIG_BASE}" \
            --pergene_output_dir "${PERGENE}" \
            --joint_output_dir "${JOINT}" \
            --collapsed_model true \
            --init_wg_zero true \
            --lr "${LR}" \
            --epochs "${EPOCHS_PERGENE}"
    )"
    echo "  pergene array job: ${PERGENE_JOB} (depends on ${JOINT_JOB})"
fi

echo "Done. Root: ${ROOT}"
