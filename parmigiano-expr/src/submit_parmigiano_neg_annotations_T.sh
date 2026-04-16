#!/bin/bash
# Submit Parmigiano joint jobs: 
# 
# Two taus (nonlinear) + select annotations --> negative annotations 
# tau2 normal prior
# noT because Ztau could be < 0 so removing ReLU 
# tau12=True
# config 1 scaling of chrombpnet and dist_to_TSS 
# BRR + noT + log1p + scale_center
# BRR + noT + log1p + scale_center + no w_g (nowg) ablations
# BRR + noT + log1p + scale_center + no rho_g (norhog) ablations
# BRR + noT + log1p + scale_center + no w_g + no rho_g (nowg_norhog) ablations
#
# --> repeat with scale_center=False and brr_results_dir=baseline_log1p




set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_parmigiano_joint.sh"


# command line arguments: lr, epochs, refits
LR="$1"
EPOCHS="$2"
REFITS="$3"
CONFIG_NUM=1

# prefix for output directory
PREFIX=200genes_neg_annotations

# paths to BRR results
BRR_RESULTS_DIR_FULL=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge

# paths to gene lists
GENE_LIST_FULL=genes_list_seed_random_full.txt

# annotations to use
ANNOTATIONS=('chrombpnet' 'dist_to_TSS' 'ABC' 'CADD_raw' 'Eigen-raw' 'enformer' 'lof' 'missense' 'splice' 'MAP20' 'alphamissense')

submit_one_full() {
    local output_dir="$1"
    local no_wg="$2"
    local no_rhog="$3"
    local no_filter="$4"


    
    echo -e "\nsbatch: ${output_dir} (no_wg=${no_wg} no_rhog=${no_rhog} no_filter=${no_filter} tau12=True scale_center=True brr_results_dir=${BRR_RESULTS_DIR_FULL})"
    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
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
        --tau12 True \
        --tau2_normal_prior True \
        --gene_list "${GENE_LIST_FULL}" \
        --chrombpnet_dist_only_cfg_num "${CONFIG_NUM}" \
        --annotations "${ANNOTATIONS[@]}" \
        --negative_annotations True
}



## log1p + scale_center
submit_one_full "${PREFIX}/full/noT" False False True # full model (normal prior for tau2, dirichlet prior for tau1, no filter)
submit_one_full "${PREFIX}/nowg/noT" True False True # nowg model 
submit_one_full "${PREFIX}/norhog/noT" False True True # norhog model 
submit_one_full "${PREFIX}/nowg_norhog/noT" True True True # nowg_norhog model 

submit_one_full "${PREFIX}/full/T" False False False # full model (normal prior for tau2, dirichlet prior for tau1, filter)
submit_one_full "${PREFIX}/nowg/T" True False False # nowg model 
submit_one_full "${PREFIX}/norhog/T" False True False # norhog model 
submit_one_full "${PREFIX}/nowg_norhog/T" True True False # nowg_norhog model 


