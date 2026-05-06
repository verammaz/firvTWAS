#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for targeted model-improvement ablations.
#
# This script submits JOINT fits first, then (optionally) submits PER-GENE
# array jobs that depend on each JOINT job finishing successfully.
#
# Usage:
#   bash submit_model_improvement_experiments.sh
#   bash submit_model_improvement_experiments.sh --prefix my_tag --dry-run
#
# Defaults match your current setup in run_parmigiano_joint.sh / run_parmigiano_pergene.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOINT_RUNNER="${SCRIPT_DIR}/run_parmigiano_joint.sh"
PERGENE_RUNNER="${SCRIPT_DIR}/run_parmigiano_pergene.sh"

# ---- Defaults ----
PREFIX="200genes_model_improve"
GENE_LIST="200genes_list_seed_random_full.txt"
LR="0.01"
EPOCHS_JOINT="500"
EPOCHS_PERGENE="1000"
REFITS="10"
TRAIN_TEST="True"
LOG_LEVEL="INFO"
CLIP_NORM="10.0"
BRR_DIR="/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge"
CONFIG_FILE="config_base.yaml"
SUBMIT_PERGENE="true"
DRY_RUN="false"

# ---- CLI ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --gene-list) GENE_LIST="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --epochs-joint) EPOCHS_JOINT="$2"; shift 2 ;;
    --epochs-pergene) EPOCHS_PERGENE="$2"; shift 2 ;;
    --refits) REFITS="$2"; shift 2 ;;
    --brr-dir) BRR_DIR="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --submit-pergene) SUBMIT_PERGENE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift 1 ;;
    -h|--help)
      cat <<'EOF'
Usage: submit_model_improvement_experiments.sh [options]

Options:
  --prefix STR            Output prefix (default: 200genes_model_improve)
  --gene-list PATH        Gene list file for joint runs
  --lr FLOAT              Learning rate (default: 0.01)
  --epochs-joint INT      Joint epochs (default: 500)
  --epochs-pergene INT    Per-gene epochs (default: 1000)
  --refits INT            Number of joint refits (default: 10)
  --brr-dir PATH          BRR results dir
  --config PATH           Base config YAML (default: config_base.yaml)
  --submit-pergene BOOL   true/false (default: true)
  --dry-run               Print commands only; do not submit
  -h, --help              Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${JOINT_RUNNER}" ]]; then
  echo "Missing joint runner: ${JOINT_RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${PERGENE_RUNNER}" ]]; then
  echo "Missing per-gene runner: ${PERGENE_RUNNER}" >&2
  exit 1
fi

submit_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] $*"
    return 0
  fi
  "$@"
}

extract_jobid() {
  # sbatch output usually: "Submitted batch job 123456"
  awk '{print $NF}'
}

submit_joint() {
  local name="$1"
  shift
  local joint_out="${PREFIX}/${name}/joint"
  local cmd=(sbatch "${JOINT_RUNNER}"
    --config "${CONFIG_FILE}"
    --joint_output_dir "${joint_out}"
    --gene_list "${GENE_LIST}"
    --brr_results_dir "${BRR_DIR}"
    --train_test "${TRAIN_TEST}"
    --lr "${LR}"
    --epochs "${EPOCHS_JOINT}"
    --refits "${REFITS}"
    --clip_norm "${CLIP_NORM}"
    --log_level "${LOG_LEVEL}"
    "$@"
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN][JOINT][${name}] ${cmd[*]}" >&2
    echo "DRYRUN_JOB_${name}"
    return 0
  fi

  local out
  out="$("${cmd[@]}")"
  local jobid
  jobid="$(printf '%s\n' "${out}" | extract_jobid)"
  echo "[JOINT][${name}] ${out}" >&2
  printf '%s\n' "${jobid}"
}

submit_pergene_after_joint() {
  local name="$1"
  local joint_jobid="$2"
  local joint_out="${PREFIX}/${name}/joint"
  local pergene_out="${PREFIX}/${name}/pergene"
  local cmd=(sbatch --dependency="afterok:${joint_jobid}" "${PERGENE_RUNNER}"
    --config "${CONFIG_FILE}"
    --joint_output_dir "${joint_out}"
    --pergene_output_dir "${pergene_out}"
    --lr "${LR}"
    --epochs "${EPOCHS_PERGENE}"
    --clip_norm "${CLIP_NORM}"
    --log_level "${LOG_LEVEL}"
  )
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN][PERGENE][${name}] ${cmd[*]}"
    return 0
  fi
  local out
  out="$("${cmd[@]}")"
  echo "[PERGENE][${name}] ${out}"
}

echo "Launching model-improvement ablations with prefix: ${PREFIX}"

# 1) Baseline reproducibility
job_baseline="$(submit_joint baseline)"

# 2) Wider threshold prior around higher T (mean=0.20)
#    Useful if |lin1| is typically > 0.05 and gate is mostly always-on.
job_t_wide="$(submit_joint threshold_beta2_8 \
  --threshold_prior_alpha 2.0 \
  --threshold_prior_beta 8.0)"

# 3) Stronger filtering tendency (mean=0.30)
job_t_strict="$(submit_joint threshold_beta3_7 \
  --threshold_prior_alpha 3.0 \
  --threshold_prior_beta 7.0)"

# 4) Stabilize exp(lin2) tail with clipping (often helps generalization)
job_lin2_clip="$(submit_joint lin2clip_10 \
  --lin2_clip 10.0)"

# 5) Keep only common variants (reduce noisy rare-variant effects)
job_maf_01="$(submit_joint maf_threshold_001 \
  --maf_threshold 0.01)"

# 6) Smooth gate vs hard gate
job_smooth_gate="$(submit_joint smooth_gate_k20 \
  --gate_mode smooth_abs \
  --gate_sharpness 20.0)"

if [[ "${SUBMIT_PERGENE}" == "true" ]]; then
  submit_pergene_after_joint baseline "${job_baseline}"
  submit_pergene_after_joint threshold_beta2_8 "${job_t_wide}"
  submit_pergene_after_joint threshold_beta3_7 "${job_t_strict}"
  submit_pergene_after_joint lin2clip_10 "${job_lin2_clip}"
  submit_pergene_after_joint maf_threshold_001 "${job_maf_01}"
  submit_pergene_after_joint smooth_gate_k20 "${job_smooth_gate}"
fi

echo "Done."
