#!/bin/bash
# Submit Parmigiano joint jobs: 
# 
# Two taus (nonlinear) + select annotations --> negative annotations 
# learn T
# tau2 normal prior vs dirichlet prior
# tau1 intercept vs no intercept
# common variants only vs all variants
# full model 
# config 1 scaling of chrombpnet and dist_to_TSS 



set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_parmigiano_joint.sh"


# command line arguments: lr, epochs, refits -- hardcoded
LR=0.01
EPOCHS=500
REFITS=10
CONFIG_NUM=1

# prefix for output directory
PREFIX=topgenes_neg_annotations_fullT

# paths to BRR results
BRR_RESULTS_DIR_FULL=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge
BRR_RESULTS_DIR_FULL_COMMON=/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common/bayesian_ridge_v2

# paths to gene lists
GENE_LIST_200=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/top200_BRR_genes.txt
GENE_LIST_500=/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/top500_BRR_genes.txt
GENE_LIST_RANDOM_200=/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src/genes_list_seed_random_full.txt

# annotations to use
ANNOTATIONS=('chrombpnet' 'dist_to_TSS' 'ABC' 'CADD_raw' 'Eigen-raw' 'enformer' 'lof' 'missense' 'splice' 'MAP20' 'alphamissense')

submit_one() {
    local output_dir="$1"
    local gene_list="$2"
    local tau1_normal_prior="$3"
    local tau1_intercept="$4"
    local common_variants_only="$5"
    local brr_results_dir="$6"

    
    echo -e "\nsbatch: ${output_dir} gene_list=${gene_list} tau1_normal_prior=${tau1_normal_prior} tau1_intercept=${tau1_intercept} common_variants_only=${common_variants_only} brr_results_dir=${brr_results_dir}"
    sbatch --time=20:00:00 --mem=200G "${RUNNER}" \
        --output_dir "${output_dir}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --refits "${REFITS}" \
        --use_brr True \
        --brr_results_dir "${brr_results_dir}" \
        --tau12 True \
        --tau2_normal_prior True \
        --tau1_normal_prior "${tau1_normal_prior}" \
        --tau1_intercept "${tau1_intercept}" \
        --common_variants_only "${common_variants_only}" \
        --gene_list "${gene_list}" \
        --maf_beta 25 \
        --chrombpnet_dist_only_cfg_num "${CONFIG_NUM}" \
        --annotations "${ANNOTATIONS[@]}" \
        --negative_annotations True
}




# rare + common variants
# submit_one "${PREFIX}/rare_common/tau1_norm_intercept" "${GENE_LIST_200}" True True False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, tau1 intercept 
# submit_one "${PREFIX}/rare_common/tau1_norm_no_intercept" "${GENE_LIST_200}" True False False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, no tau1 intercept 
# submit_one "${PREFIX}/rare_common/tau1_dir_intercept" "${GENE_LIST_200}" False True False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, tau1 intercept 
# submit_one "${PREFIX}/rare_common/tau1_dir_no_intercept" "${GENE_LIST_200}" False False False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, no tau1 intercept 

# submit_one "${PREFIX}/rare_common_maf_beta25/tau1_norm_intercept" "${GENE_LIST_200}" True True False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, tau1 intercept 
# submit_one "${PREFIX}/rare_common_maf_beta25/tau1_norm_no_intercept" "${GENE_LIST_200}" True False False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, no tau1 intercept 
# submit_one "${PREFIX}/rare_common_maf_beta25/tau1_dir_intercept" "${GENE_LIST_200}" False True False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, tau1 intercept 
# submit_one "${PREFIX}/rare_common_maf_beta25/tau1_dir_no_intercept" "${GENE_LIST_200}" False False False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, no tau1 intercept 

submit_one "${PREFIX}/rare_common_random_maf_beta25/tau1_norm_intercept" "${GENE_LIST_RANDOM_200}" True True False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, tau1 intercept 
submit_one "${PREFIX}/rare_common_random_maf_beta25/tau1_norm_no_intercept" "${GENE_LIST_RANDOM_200}" True False False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, no tau1 intercept 
submit_one "${PREFIX}/rare_common_random_maf_beta25/tau1_dir_intercept" "${GENE_LIST_RANDOM_200}" False True False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, tau1 intercept 
submit_one "${PREFIX}/rare_common_random_maf_beta25/tau1_dir_no_intercept" "${GENE_LIST_RANDOM_200}" False False False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, no tau1 intercept 

# submit_one "${PREFIX}/rare_common_random/tau1_norm_intercept" "${GENE_LIST_RANDOM_200}" True True False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, tau1 intercept 
# submit_one "${PREFIX}/rare_common_random/tau1_norm_no_intercept" "${GENE_LIST_RANDOM_200}" True False False "${BRR_RESULTS_DIR_FULL}" # tau1 normal, no tau1 intercept 
# submit_one "${PREFIX}/rare_common_random/tau1_dir_intercept" "${GENE_LIST_RANDOM_200}" False True False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, tau1 intercept 
# submit_one "${PREFIX}/rare_common_random/tau1_dir_no_intercept" "${GENE_LIST_RANDOM_200}" False False False "${BRR_RESULTS_DIR_FULL}" # tau1 dirichlet, no tau1 intercept 

# common variants only
# submit_one "${PREFIX}/common/tau1_norm_intercept" "${GENE_LIST_500}" True True True "${BRR_RESULTS_DIR_FULL_COMMON}" # tau1 normal, tau1 intercept 
# submit_one "${PREFIX}/common/tau1_norm_no_intercept" "${GENE_LIST_500}" True False True "${BRR_RESULTS_DIR_FULL_COMMON}" # tau1 normal, no tau1 intercept 
# submit_one "${PREFIX}/common/tau1_dir_intercept" "${GENE_LIST_500}" False True True "${BRR_RESULTS_DIR_FULL_COMMON}" # tau1 dirichlet, tau1 intercept 
# submit_one "${PREFIX}/common/tau1_dir_no_intercept" "${GENE_LIST_500}" False False True "${BRR_RESULTS_DIR_FULL_COMMON}" # tau1 dirichlet, no tau1 intercept 


