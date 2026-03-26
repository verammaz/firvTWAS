#!/bin/bash
# Submit Parmigiano joint jobs: 
# 
# Single tau (linear)
# BRR + T + log1p + scale_center (defualt True) 
# BRR + T + log1p + scale_center
# BRR + noT + log1p + scale_center
# BRR + noT + log1p + no_wg (nowg)
# BRR + noT + log1p + scale_center + no w_g (nowg) ablations
# BRR + noT + log1p + scale_center + no rho_g (norhog) ablations
# BRR + noT + log1p + scale_center + no w_g (nowg) + no rho_g (norhog) (nowg_norhog) ablations
#
# --> repeat with scale_center=False and brr_results_dir=baseline_log1p
# --> repeat with tau12=True    



set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_parmigiano_joint.sh"

# prefix for output directory
PREFIX=200genes

# command line arguments: lr, epochs, refits
LR="$1"
EPOCHS="$2"
REFITS="$3"

# paths to BRR results
BRR_RESULTS_DIR_FULL=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
BRR_RESULTS_DIR_LOG1P=/gpfs/commons/home/adas/uTWAS/src/results/baseline_log1p/bayesian_ridge

# paths to gene lists
GENE_LIST_FULL=genes_list_seed_random_full.txt
GENE_LIST_LOG1P=genes_list_seed_random_log1p.txt


submit_one_full() {
    local output_dir="$1"
    local no_wg="$2"
    local no_rhog="$3"
    local no_filter="$4"
    local tau12="$5"

    echo "sbatch: ${output_dir} (no_wg=${no_wg} no_rhog=${no_rhog} no_filter=${no_filter} tau12=${tau12} scale_center=True brr_results_dir=${BRR_RESULTS_DIR_FULL})"
    sbatch "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --no_wg "${no_wg}" \
        --no_rhog "${no_rhog}" \
        --no_filter "${no_filter}" \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR_FULL}" \
        --scale_center True \
        --tau12 "${tau12}" \
        --gene_list "${GENE_LIST_FULL}"
}

submit_one_log1p() {
    local output_dir="$1"
    local no_wg="$2"
    local no_rhog="$3"
    local no_filter="$4"
    local tau12="$5"

    echo "sbatch: ${output_dir} (no_wg=${no_wg} no_rhog=${no_rhog} no_filter=${no_filter} tau12=${tau12} scale_center=False brr_results_dir=${BRR_RESULTS_DIR_LOG1P})"
    sbatch "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --no_wg "${no_wg}" \
        --no_rhog "${no_rhog}" \
        --no_filter "${no_filter}" \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR_LOG1P}" \
        --scale_center False \
        --tau12 "${tau12}" \
        --gene_list "${GENE_LIST_LOG1P}"
}


### Linear model ############################################################

## log1p + scale_center
submit_one_full "${PREFIX}_full/tau/full" False False False False # full model
submit_one_full "${PREFIX}_full/tau/nowg" True False False False
submit_one_full "${PREFIX}_full/tau/norhog" False True False False
submit_one_full "${PREFIX}_full/tau/nowg_norhog" True True False False
submit_one_full "${PREFIX}_full/tau/noT" False False True False

# log1p only
submit_one_log1p "${PREFIX}_log1p/tau/full" False False False False
submit_one_log1p "${PREFIX}_log1p/tau/nowg" True False False False
submit_one_log1p "${PREFIX}_log1p/tau/norhog" False True False False
submit_one_log1p "${PREFIX}_log1p/tau/nowg_norhog" True True False False
submit_one_log1p "${PREFIX}_log1p/tau/noT" False False True False


### Nonlinear model ############################################################

## log1p + scale_center
submit_one_full "${PREFIX}_full/tau12/full" False False False True # full model
submit_one_full "${PREFIX}_full/tau12/nowg" True False False True
submit_one_full "${PREFIX}_full/tau12/norhog" False True False True
submit_one_full "${PREFIX}_full/tau12/nowg_norhog" True True False True
submit_one_full "${PREFIX}_full/tau12/noT" False False True True

# log1p only
submit_one_log1p "${PREFIX}_log1p/tau12/full" False False False True
submit_one_log1p "${PREFIX}_log1p/tau12/nowg" True False False True
submit_one_log1p "${PREFIX}_log1p/tau12/norhog" False True False True
submit_one_log1p "${PREFIX}_log1p/tau12/nowg_norhog" True True False True
submit_one_log1p "${PREFIX}_log1p/tau12/noT" False False True True