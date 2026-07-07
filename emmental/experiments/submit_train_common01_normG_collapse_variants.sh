#!/bin/bash
# Collapsed + normalize_G variant sweep (5 refits per job; default script refits remain 10).
#
# Usage:
#   bash submit_train_common01_normG_collapse_variants.sh
#   SUBMIT_T=0 bash submit_train_common01_normG_collapse_variants.sh   # structure only
#   SUBMIT_STRUCTURE=0 bash submit_train_common01_normG_collapse_variants.sh  # T sweep only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
MYOUT="/gpfs/commons/home/vmazeeva/firvTWAS_myout"
ROOT="${MYOUT}/train_common01_normG_collapse_variants"

SUBMIT_STRUCTURE="${SUBMIT_STRUCTURE:-1}"
SUBMIT_T="${SUBMIT_T:-1}"
REFITS=5

COMMON=(
  --collapsed_model true
  --normalize_G true
  --maf_threshold 0.01
  --brr_results_dir /gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01/bayesian_ridge
  --refits "${REFITS}"
)

submit() {
  local name="$1"
  shift
  sbatch "${SRC_DIR}/run_emmental_joint.sh" \
    --joint_output_dir "${ROOT}/${name}/joint" \
    "${COMMON[@]}" \
    "$@"
}

if [[ "${SUBMIT_STRUCTURE}" == "1" ]]; then
  echo "=== Structural variants (${REFITS} refits each) ==="
  submit full       --no_wg false --no_rhog false --no_T false
  submit no_wg      --no_wg true  --no_rhog false --no_T false
  submit no_rhog    --no_wg false --no_rhog true  --no_T false
  submit no_wg_rhog --no_wg true  --no_rhog true  --no_T false
  submit noT        --no_wg false --no_rhog false --no_T true
  submit no_wg_noT      --no_wg true  --no_rhog false --no_T true
  submit no_rhog_noT    --no_wg false --no_rhog true  --no_T true
  submit no_wg_rhog_noT --no_wg true  --no_rhog true  --no_T true
fi

if [[ "${SUBMIT_T}" == "1" ]]; then
  echo "=== T prior/init variants on full structure (${REFITS} refits each) ==="
  T_COMMON=(--no_wg false --no_rhog false --no_T false)
  submit T_default "${T_COMMON[@]}"
  submit Tprior_2_8 "${T_COMMON[@]}" --threshold_prior_alpha 2 --threshold_prior_beta 8
  submit Tprior_1_1 "${T_COMMON[@]}" --threshold_prior_alpha 1 --threshold_prior_beta 1
  submit Tinit_prior_mode "${T_COMMON[@]}" --threshold_init prior_mode
  submit Tinit_data_q25 "${T_COMMON[@]}" \
    --threshold_init data_quantile --threshold_init_quantile 0.25
fi

echo "=== T init variants on no_wg structure (${REFITS} refits each) ==="
T_COMMON=(--no_wg true --no_rhog false --no_T false)
submit T_no_wg_prior_2_8 "${T_COMMON[@]}" --threshold_prior_alpha 2 --threshold_prior_beta 8
submit T_no_wg_prior_1_1 "${T_COMMON[@]}" --threshold_prior_alpha 1 --threshold_prior_beta 1
submit T_no_wg_init_data_q25 "${T_COMMON[@]}" \
  --threshold_init data_quantile --threshold_init_quantile 0.25

echo "=== T prior variants on no_rhog structure (${REFITS} refits each) ==="
T_COMMON=(--no_wg false --no_rhog true --no_T false)
submit T_no_rhog_prior_2_8 "${T_COMMON[@]}" --threshold_prior_alpha 2 --threshold_prior_beta 8
submit T_no_rhog_prior_1_1 "${T_COMMON[@]}" --threshold_prior_alpha 1 --threshold_prior_beta 1
submit T_no_rhog_init_data_q25 "${T_COMMON[@]}" \
  --threshold_init data_quantile --threshold_init_quantile 0.25

echo "=== T prior variants on no_wg_rhog structure (${REFITS} refits each) ==="
T_COMMON=(--no_wg true --no_rhog true --no_T false)
submit T_no_wg_rhog_prior_2_8 "${T_COMMON[@]}" --threshold_prior_alpha 2 --threshold_prior_beta 8
submit T_no_wg_rhog_prior_1_1 "${T_COMMON[@]}" --threshold_prior_alpha 1 --threshold_prior_beta 1
submit T_no_wg_rhog_init_data_q25 "${T_COMMON[@]}" \
  --threshold_init data_quantile --threshold_init_quantile 0.25

echo "Done. Outputs under ${ROOT}/"
