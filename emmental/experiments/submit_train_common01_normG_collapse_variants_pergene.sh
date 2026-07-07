#!/bin/bash
# Submit per-gene chr array for completed variant-sweep joint runs.
#
# Skips variants whose joint is incomplete or pergene already has 22/22 chr outputs.
#
# Usage:
#   bash submit_train_common01_normG_collapse_variants_pergene.sh
#   DRY_RUN=1 bash submit_train_common01_normG_collapse_variants_pergene.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
MYOUT="/gpfs/commons/home/vmazeeva/firvTWAS_myout"
ROOT="${MYOUT}/train_common01_normG_collapse_variants"
CONFIG_BASE="${SRC_DIR}/config_base.yaml"
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

submit_pergene() {
  local name="$1"
  local joint_dir="${ROOT}/${name}/joint"
  local pergene_dir="${ROOT}/${name}/pergene"

  if ! is_joint_complete "$joint_dir"; then
    echo "SKIP $name (joint not complete)"
    return 0
  fi

  if is_pergene_complete "$pergene_dir"; then
    echo "SKIP $name (pergene complete: ${N_CHR}/${N_CHR} chr)"
    return 0
  fi

  local pg_done
  pg_done=$(count_chr_r2 "$pergene_dir")
  mkdir -p "$pergene_dir"
  echo "=== pergene: ${name} (${pg_done}/${N_CHR} chr done) ==="
  echo "  joint:   ${joint_dir}"
  echo "  pergene: ${pergene_dir}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  (dry run)"
    return 0
  fi

  local job
  job="$(
    sbatch --parsable \
      --job-name="pg_${name}" \
      --array=1-"${N_CHR}" \
      "${SRC_DIR}/run_emmental_pergene.sh" \
      --config "${CONFIG_BASE}" \
      --joint_output_dir "${joint_dir}" \
      --pergene_output_dir "${pergene_dir}"
  )"
  echo "  job: ${job}"
}

for name in "$ROOT"/*/joint; do
  submit_pergene "$(basename "$(dirname "$name")")"
done

echo "Done."
