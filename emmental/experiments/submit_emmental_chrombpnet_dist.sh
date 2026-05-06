#!/bin/bash
# Submit Parmigiano joint jobs: 
# 
# Two taus (nonlinear) + chrombpnet and dist_to_TSS annotations only --> negative annotations 
# noT because Ztau could be < 0 so removing ReLU 
# tau12=True
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
CONFIG_NUM="$4"
TAU="$5" # tau or tau12

# prefix for output directory
PREFIX=200genes_chrombpnet_dist_absmu/config${CONFIG_NUM}/lr${LR}_${TAU}


# paths to BRR results
BRR_RESULTS_DIR_FULL=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
BRR_RESULTS_DIR_LOG1P=/gpfs/commons/home/adas/uTWAS/src/results/baseline_log1p/bayesian_ridge

# paths to gene lists
GENE_LIST_FULL=genes_list_seed_random_full.txt
GENE_LIST_LOG1P=genes_list_seed_random_log1p.txt

TAU12=True
if [ "$TAU" == "tau12" ]; then
    TAU12=True
else
    TAU12=False
fi

submit_one_full() {
    local output_dir="$1"
    local no_wg="$2"
    local no_rhog="$3"
    local burden="$4"
    local tau1_normal_prior="$5"
    local tau2_normal_prior="$6"

    # check if output_dir exitsts -- already ran or currently running --> skip
    if [ -d "${output_dir}" ]; then
        echo -e "\nOutput directory ${output_dir} already exists -- skipping (likely already ran or currently running)"
        return
    fi
    
    echo -e "\nsbatch: ${output_dir} (no_wg=${no_wg} no_rhog=${no_rhog} no_filter=True tau12=${TAU12} scale_center=True brr_results_dir=${BRR_RESULTS_DIR_FULL})"
    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --no_wg "${no_wg}" \
        --no_rhog "${no_rhog}" \
        --burden "${burden}" \
        --no_filter True \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR_FULL}" \
        --scale_center True \
        --tau12 "${TAU12}" \
        --tau2_normal_prior "${tau2_normal_prior}" \
        --tau1_normal_prior "${tau1_normal_prior}" \
        --gene_list "${GENE_LIST_FULL}" \
        --chrombpnet_dist_only True \
        --chrombpnet_dist_only_cfg_num "${CONFIG_NUM}"
}

submit_one_log1p() {
    local output_dir="$1"
    local no_wg="$2"
    local no_rhog="$3"
    local burden="$4"
    local tau1_normal_prior="$5"
    local tau2_normal_prior="$6"


    # check if output_dir exitsts -- already ran or currently running --> skip
    if [ -d "${output_dir}" ]; then
        echo -e "\nOutput directory ${output_dir} already exists -- skipping (likely already ran or currently running)"
        return
    fi

    echo -e "\nsbatch: ${output_dir} (no_wg=${no_wg} no_rhog=${no_rhog} no_filter=True tau12=${TAU12} scale_center=False brr_results_dir=${BRR_RESULTS_DIR_LOG1P})"
    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --no_wg "${no_wg}" \
        --no_rhog "${no_rhog}" \
        --burden "${burden}" \
        --no_filter True \
        --use_brr True \
        --brr_results_dir "${BRR_RESULTS_DIR_LOG1P}" \
        --scale_center False \
        --tau12 "${TAU12}" \
        --tau1_normal_prior "${tau1_normal_prior}" \
        --gene_list "${GENE_LIST_LOG1P}" \
        --chrombpnet_dist_only True \
        --chrombpnet_dist_only_cfg_num "${CONFIG_NUM}" \
        --tau2_normal_prior "${tau2_normal_prior}"
}




## log1p + scale_center
submit_one_full "${PREFIX}_full/burden" False False True False False

if [ "$TAU12" == "True" ]; then # if nonlinear mode, try dirichlet and normal priors for tau2
    submit_one_full "${PREFIX}_full/full/tau12_dir_prior" False False False False False # full model (dirichlet prior for tau1 amd tau2)
    submit_one_full "${PREFIX}_full/full/tau2_norm_prior" False False False False True # full model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_full "${PREFIX}_full/full/tau1_norm_prior" False False False True False # full model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_full "${PREFIX}_full/full/tau12_norm_prior" False False False True True # full model (normal prior for tau1 and tau2)

    submit_one_full "${PREFIX}_full/nowg/tau12_dir_prior" True False False False False # nowg model (dirichlet prior for tau1 amd tau2)
    submit_one_full "${PREFIX}_full/nowg/tau2_norm_prior" True False False False True # nowg model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_full "${PREFIX}_full/nowg/tau1_norm_prior" True False False True False # nowg model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_full "${PREFIX}_full/nowg/tau12_norm_prior" True False False True True # nowg model (normal prior for tau1 and tau2)

    submit_one_full "${PREFIX}_full/norhog/tau12_dir_prior" False True False False False # norhog model (dirichlet prior for tau1 amd tau2)
    submit_one_full "${PREFIX}_full/norhog/tau2_norm_prior" False True False False True # norhog model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_full "${PREFIX}_full/norhog/tau1_norm_prior" False True False True False # norhog model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_full "${PREFIX}_full/norhog/tau12_norm_prior" False True False True True # norhog model (normal prior for tau1 and tau2)

    submit_one_full "${PREFIX}_full/nowg_norhog/tau12_dir_prior" True True False False False # nowg_norhog model (dirichlet prior for tau1 amd tau2)
    submit_one_full "${PREFIX}_full/nowg_norhog/tau2_norm_prior" True True False False True # nowg_norhog model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_full "${PREFIX}_full/nowg_norhog/tau1_norm_prior" True True False True False # nowg_norhog model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_full "${PREFIX}_full/nowg_norhog/tau12_norm_prior" True True False True True # nowg_norhog model (normal prior for tau1 and tau2)


else # if linear mode, keep dirichlet prior for tau
    submit_one_full "${PREFIX}_full/full" False False False False False # full model (default priors)
    submit_one_full "${PREFIX}_full/nowg" True False False False False # nowg model (default priors)
    submit_one_full "${PREFIX}_full/norhog" False True False False False # norhog model (default priors)
    submit_one_full "${PREFIX}_full/nowg_norhog" True True False False False # nowg_norhog model (default priors)
fi


# log1p only
submit_one_log1p "${PREFIX}_log1p/burden" False False True False False

if [ "$TAU12" == "True" ]; then
    submit_one_log1p "${PREFIX}_log1p/full/tau12_dir_prior" False False False False False # full model (dirichlet prior for tau1 amd tau2)
    submit_one_log1p "${PREFIX}_log1p/full/tau2_norm_prior" False False False False True # full model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_log1p "${PREFIX}_log1p/full/tau1_norm_prior" False False False True False # full model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_log1p "${PREFIX}_log1p/full/tau12_norm_prior" False False False True True # full model (normal prior for tau1 and tau2)

    submit_one_log1p "${PREFIX}_log1p/nowg/tau12_dir_prior" True False False False False # nowg model (dirichlet prior for tau1 amd tau2)
    submit_one_log1p "${PREFIX}_log1p/nowg/tau2_norm_prior" True False False False True # nowg model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_log1p "${PREFIX}_log1p/nowg/tau1_norm_prior" True False False True False # nowg model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_log1p "${PREFIX}_log1p/nowg/tau12_norm_prior" True False False True True # nowg model (normal prior for tau1 and tau2)

    submit_one_log1p "${PREFIX}_log1p/norhog/tau12_dir_prior" False True False False False # norhog model (dirichlet prior for tau1 amd tau2)
    submit_one_log1p "${PREFIX}_log1p/norhog/tau2_norm_prior" False True False False True # norhog model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_log1p "${PREFIX}_log1p/norhog/tau1_norm_prior" False True False True False # norhog model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_log1p "${PREFIX}_log1p/norhog/tau12_norm_prior" False True False True True # norhog model (normal prior for tau1 and tau2)

    submit_one_log1p "${PREFIX}_log1p/nowg_norhog/tau12_dir_prior" True True False False False # nowg_norhog model (dirichlet prior for tau1 amd tau2)
    submit_one_log1p "${PREFIX}_log1p/nowg_norhog/tau2_norm_prior" True True False False True # nowg_norhog model (normal prior for tau2, dirichlet prior for tau1)
    submit_one_log1p "${PREFIX}_log1p/nowg_norhog/tau1_norm_prior" True True False True False # nowg_norhog model (normal prior for tau1, dirichlet prior for tau2)
    submit_one_log1p "${PREFIX}_log1p/nowg_norhog/tau12_norm_prior" True True False True True # nowg_norhog model (normal prior for tau1 and tau2)

else # if linear mode, only try dirichlet prior for tau
    submit_one_log1p "${PREFIX}_log1p/full" False False False False False # full model (default priors)
    submit_one_log1p "${PREFIX}_log1p/nowg" True False False False False # nowg model (default priors)
    submit_one_log1p "${PREFIX}_log1p/norhog" False True False False False # norhog model (default priors)
    submit_one_log1p "${PREFIX}_log1p/nowg_norhog" True True False False False # nowg_norhog model (default priors)
fi