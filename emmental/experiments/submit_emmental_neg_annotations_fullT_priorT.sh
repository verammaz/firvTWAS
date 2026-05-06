#!/bin/bash
# Submit Parmigiano joint jobs:
#
# Full nonlinear negative-annotation model (learn T) with threshold-prior ablations.
# Uses the shared wrapper run_parmigiano_joint.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_parmigiano_joint.sh"

# Core training settings
LR=0.01
EPOCHS=500
REFITS=10
CONFIG_NUM=1

# Output root
PREFIX=200genes_neg_annotations_fullT_priorT

# Data/Baseline settings
BRR_RESULTS_DIR_FULL=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
GENE_LIST_RANDOM_FULL=/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src/genes_list_seed_random_full.txt

# Annotation set for negative-annotation nonlinear model
ANNOTATIONS=('chrombpnet' 'dist_to_TSS' 'ABC' 'CADD_raw' 'Eigen-raw' 'enformer' 'lof' 'missense' 'splice' 'MAP20' 'alphamissense')

submit_one() {
    local output_dir="$1"
    local tau1_normal_prior="$2"
    local tau1_intercept="$3"
    local threshold_alpha="$4"
    local threshold_beta="$5"

    echo -e "\nsbatch: ${output_dir} tau1_normal_prior=${tau1_normal_prior} tau1_intercept=${tau1_intercept} threshold_prior=Beta(${threshold_alpha},${threshold_beta})"
    echo "  resolved config:"
    echo "    gene_list=${GENE_LIST_RANDOM_FULL}"
    echo "    tau12=True  negative_annotations=True  tau1_intercept=${tau1_intercept}"
    echo "    tau1_normal_prior=${tau1_normal_prior}  tau2_normal_prior=True"
    echo "    threshold_prior_alpha=${threshold_alpha}  threshold_prior_beta=${threshold_beta}"
    echo "    maf_beta=1  common_variants_only=False"
    echo "    lr=${LR}  epochs=${EPOCHS}  refits=${REFITS}  chrombpnet_dist_only_cfg_num=${CONFIG_NUM}"
    echo "    brr_results_dir=${BRR_RESULTS_DIR_FULL}"

    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR_FULL}" \
        --tau12 True \
        --tau2_normal_prior True \
        --tau1_normal_prior "${tau1_normal_prior}" \
        --tau1_intercept "${tau1_intercept}" \
        --common_variants_only False \
        --gene_list "${GENE_LIST_RANDOM_FULL}" \
        --maf_beta 1 \
        --chrombpnet_dist_only_cfg_num "${CONFIG_NUM}" \
        --annotations "${ANNOTATIONS[@]}" \
        --negative_annotations True \
        --threshold_prior_alpha "${threshold_alpha}" \
        --threshold_prior_beta "${threshold_beta}"
}

# ---------------------------------------------------------------------------
# Prior-T sweep (full model, learn threshold)
# ---------------------------------------------------------------------------

# Keep your preferred full model family for comparability.
# A) tau1 Normal + intercept
submit_one "${PREFIX}/tau1_norm_intercept/prior_2_20" True  True  2 20   # current default
submit_one "${PREFIX}/tau1_norm_intercept/prior_2_8"  True  True  2 8    # less concentrated, less low-biased
submit_one "${PREFIX}/tau1_norm_intercept/prior_1_1"  True  True  1 1    # uniform
submit_one "${PREFIX}/tau1_norm_intercept/prior_4_20" True  True  4 20   # slightly higher prior mean

# B) tau1 Dirichlet + intercept
submit_one "${PREFIX}/tau1_dir_intercept/prior_2_20"  False True  2 20
submit_one "${PREFIX}/tau1_dir_intercept/prior_2_8"   False True  2 8
submit_one "${PREFIX}/tau1_dir_intercept/prior_1_1"   False True  1 1
submit_one "${PREFIX}/tau1_dir_intercept/prior_4_20"  False True  4 20

