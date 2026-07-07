#!/bin/bash
# Submit post-pergene chr array for variant-sweep runs with complete pergene outputs.
#
# Skips when pergene is incomplete or post_pergene already has 22/22 chr outputs.
#
# Usage:
#   bash submit_train_common01_normG_collapse_variants_post_train.sh
#   DRY_RUN=1 bash submit_train_common01_normG_collapse_variants_post_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
MYOUT="/gpfs/commons/home/vmazeeva/firvTWAS_myout"
ROOT="${MYOUT}/train_common01_normG_collapse_variants"
N_CHR=22

count_chr_r2() {
  local base="$1"
  local n=0 i
  [[ -d "$base" ]] || { echo 0; return; }
  for i in $(seq 1 "$N_CHR"); do
    if [[ -f "${base}/chr${i}/test_r2_scores.csv" ]]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

is_joint_complete() {
  local joint_dir="$1"
  local cfg="${joint_dir}/config.yaml"
  [[ -f "$cfg" ]] || return 1
  local planned refits runs
  planned=$(grep 'n_refits_planned' "$cfg" | awk '{print $2}')
  planned=${planned:-5}
  refits=$(grep -A20 '^refits:' "$cfg" | grep '^-' | wc -l)
  runs=$(ls -d "${joint_dir}"/run_* 2>/dev/null | wc -l)
  [[ "$refits" -ge "$planned" ]] && [[ "$runs" -ge "$planned" ]]
}

is_pergene_complete() {
  local pergene_dir="$1"
  [[ "$(count_chr_r2 "$pergene_dir")" -eq "$N_CHR" ]]
}

is_post_train_complete() {
  local post_dir="$1"
  [[ "$(count_chr_r2 "$post_dir")" -eq "$N_CHR" ]]
}

submit_post_train() {
  local name="$1"
  local joint_dir="${ROOT}/${name}/joint"
  local pergene_dir="${ROOT}/${name}/pergene"
  local post_dir="${ROOT}/${name}/post_pergene"

  if ! is_joint_complete "$joint_dir"; then
    echo "SKIP $name (joint not complete)"
    return 0
  fi

  if ! is_pergene_complete "$pergene_dir"; then
    local pg_done
    pg_done=$(count_chr_r2 "$pergene_dir")
    echo "SKIP $name (pergene incomplete: ${pg_done}/${N_CHR} chr)"
    return 0
  fi

  if is_post_train_complete "$post_dir"; then
    echo "SKIP $name (post_pergene complete: ${N_CHR}/${N_CHR} chr)"
    return 0
  fi

  local post_done
  post_done=$(count_chr_r2 "$post_dir")
  echo "=== post_pergene: ${name} (${post_done}/${N_CHR} chr done) ==="
  echo "  pergene: ${pergene_dir}"
  echo "  post:    ${post_dir}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  (dry run)"
    return 0
  fi

  local job
  job="$(
    sbatch --parsable \
      --job-name="post_${name}" \
      --array=1-"${N_CHR}" \
      "${SRC_DIR}/run_emmental_post_train.sh" \
      --pergene_output_dir "${pergene_dir}"
  )"
  echo "  job: ${job}"
}

for name in "$ROOT"/*/joint; do
  submit_post_train "$(basename "$(dirname "$name")")"
done

echo "Done."
