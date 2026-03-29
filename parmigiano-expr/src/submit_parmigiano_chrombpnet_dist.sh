#!/bin/bash
# Submit Parmigiano joint jobs: 
# 
# Two taus (nonlinear) + chrombpnet and dist_to_TSS annotations only --> negative annotations 
# noT because Ztau could be < 0 and removing ReLU would make it even more negative
# tau12=True
# BRR + noT + log1p + scale_center
# BRR + noT + log1p + scale_center + no w_g (nowg) ablations
# BRR + noT + log1p + scale_center + no rho_g (norhog) ablations
# BRR + noT + log1p + scale_center + no w_g (nowg) + no rho_g (norhog) (nowg_norhog) ablations
#
# --> repeat with scale_center=False and brr_results_dir=baseline_log1p




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

    echo "sbatch: ${output_dir} (no_wg=${no_wg} no_rhog=${no_rhog} no_filter=True tau12=True scale_center=True brr_results_dir=${BRR_RESULTS_DIR_FULL})"
    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --no_wg "${no_wg}" \
        --no_rhog "${no_rhog}" \
        --no_filter True \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR_FULL}" \
        --scale_center True \
        --tau12 True \
        --gene_list "${GENE_LIST_FULL}" \
        --chrombpnet_dist_only True
}

submit_one_log1p() {
    local output_dir="$1"
    local no_wg="$2"
    local no_rhog="$3"

    echo "sbatch: ${output_dir} (no_wg=${no_wg} no_rhog=${no_rhog} no_filter=True tau12=True scale_center=False brr_results_dir=${BRR_RESULTS_DIR_LOG1P})"
    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --no_wg "${no_wg}" \
        --no_rhog "${no_rhog}" \
        --no_filter True \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR_LOG1P}" \
        --scale_center False \
        --tau12 True \
        --gene_list "${GENE_LIST_LOG1P}" \
        --chrombpnet_dist_only True
}




### Nonlinear model ############################################################

## log1p + scale_center
submit_one_full "${PREFIX}_full/tau12_chrombpnet_dist/full" False False # full model
submit_one_full "${PREFIX}_full/tau12_chrombpnet_dist/nowg" True False
submit_one_full "${PREFIX}_full/tau12_chrombpnet_dist/norhog" False True
submit_one_full "${PREFIX}_full/tau12_chrombpnet_dist/nowg_norhog" True True

# log1p only
submit_one_log1p "${PREFIX}_log1p/tau12_chrombpnet_dist/full" False False
submit_one_log1p "${PREFIX}_log1p/tau12_chrombpnet_dist/nowg" True False
submit_one_log1p "${PREFIX}_log1p/tau12_chrombpnet_dist/norhog" False True
submit_one_log1p "${PREFIX}_log1p/tau12_chrombpnet_dist/nowg_norhog" True True