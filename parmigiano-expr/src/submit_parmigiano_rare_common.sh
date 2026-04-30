#!/bin/bash
# Rare vs common variant panels for Parmigiano (joint τ, T).
#
# Narrative (matched baselines):
#   1) panel_full     — All variants in BRR-full betas (rare + common). Baselines: baseline_full.
#   2) panel_brr_common — Variants exactly as in BRR run on common-only panel. Baselines: baseline_full_common.
#   3) panel_maf_common — Same BRR-full directory as (1), but genotypes are subset to MAF >= maf_threshold
#                         on training samples (annotation-guided model uses the same shrunk panel).
#
# Prerequisite: Bayesian ridge (or other baselines) must be run on the same gene list and comparable
# expression scaling for each panel directory.
#
# NOTE: common_variants_only was wired in Python to apply after BRR column sync; train and test use
#       the same variant list (MAF from training genotypes only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_parmigiano_joint.sh"

LR=0.01
EPOCHS=500
REFITS=10
CONFIG_NUM=1
MAF_BETA=25
MAF_THRESHOLD=0.01

PREFIX=300genes_rare_common

BRR_RESULTS_DIR_FULL=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
BRR_RESULTS_DIR_FULL_COMMON=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common/bayesian_ridge_v2

GENE_LIST="/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src/300genes_list_seed_random_full.txt"

ANNOTATIONS=('chrombpnet' 'dist_to_TSS' 'ABC' 'CADD_raw' 'Eigen-raw' 'enformer' 'lof' 'missense' 'splice' 'MAP20' 'alphamissense', 'gnomAD')

# Fixed model story: nonlinear, tau1 dirichlet + intercept, tau2 normal, negative annotations, learn T.
submit_one() {
    local output_dir="$1"
    local common_variants_only="$2"
    local brr_results_dir="$3"
    local label="$4"

    echo -e "\nsbatch [${label}]: output_dir=${output_dir} common_variants_only=${common_variants_only} brr=${brr_results_dir}"
    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --use_brr True \
        --brr_results_dir "${brr_results_dir}" \
        --tau12 True \
        --tau2_normal_prior True \
        --tau1_normal_prior False \
        --tau1_intercept True \
        --common_variants_only "${common_variants_only}" \
        --maf_threshold "${MAF_THRESHOLD}" \
        --gene_list "${GENE_LIST}" \
        --maf_beta "${MAF_BETA}" \
        --annotations "${ANNOTATIONS[@]}" \
        --negative_annotations True
}


submit_one "${PREFIX}/rare_common" False "${BRR_RESULTS_DIR_FULL}" "rare_common"

submit_one "${PREFIX}/common" True "${BRR_RESULTS_DIR_FULL_COMMON}" "common"
