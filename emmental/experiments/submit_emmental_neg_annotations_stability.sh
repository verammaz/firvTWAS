#!/bin/bash
# Submit Parmigiano joint jobs:
# Full nonlinear negative-annotation model + stability ablations.
#
# Mirrors submit_parmigiano_neg_annotations_T_tau1_intercept.sh style and
# routes everything through run_parmigiano_joint.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_parmigiano_joint.sh"

LR=0.01
EPOCHS=500
REFITS=10
CONFIG_NUM=1
PREFIX=topgenes_neg_annotations_stability

BRR_RESULTS_DIR=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
GENE_LIST_200=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/top200_BRR_genes.txt

ANNOTATIONS=('chrombpnet' 'dist_to_TSS' 'ABC' 'CADD_raw' 'Eigen-raw' 'enformer' 'lof' 'missense' 'splice' 'MAP20' 'alphamissense')

submit_one() {
    local output_dir="$1"
    local lin2_clip="$2"
    local tau2_link="$3"
    local gate_mode="$4"
    local gate_sharpness="$5"
    local wg_positive="$6"

    echo -e "\nsbatch: ${output_dir} clip=${lin2_clip:-none} tau2_link=${tau2_link} gate=${gate_mode} sharp=${gate_sharpness} wg_positive=${wg_positive}"

    clip_args=()
    if [ -n "${lin2_clip}" ]; then
        clip_args=(--lin2_clip "${lin2_clip}")
    fi

    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR}" \
        --tau12 True \
        --tau2_normal_prior True \
        --tau1_normal_prior True \
        --tau1_intercept True \
        --common_variants_only False \
        --gene_list "${GENE_LIST_200}" \
        --maf_beta 1 \
        --chrombpnet_dist_only_cfg_num "${CONFIG_NUM}" \
        --annotations "${ANNOTATIONS[@]}" \
        --negative_annotations True \
        --negative_gate_mode "${gate_mode}" \
        --negative_gate_sharpness "${gate_sharpness}" \
        "${clip_args[@]}" \
        --tau2_link "${tau2_link}" \
        --wg_positive "${wg_positive}"
}

# 1) Current full-model behavior (control)
submit_one "${PREFIX}/control_full" "" "exp" "hard_abs" "20.0" "False"

# 2) Clip lin2 to limit exp explosion
submit_one "${PREFIX}/lin2_clip5" "5.0" "exp" "hard_abs" "20.0" "False"

# 3) Smooth gate (soft transition around threshold)
submit_one "${PREFIX}/smooth_gate" "" "exp" "smooth_abs" "20.0" "False"

# 4) Positive gene-level scaling
submit_one "${PREFIX}/wg_positive" "" "exp" "hard_abs" "20.0" "True"

# 5) Combined stabilization
submit_one "${PREFIX}/combined_clip5_smooth_wgpos" "5.0" "exp" "smooth_abs" "20.0" "True"

