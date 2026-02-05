#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu,bigmem
#SBATCH --mem=100G
#SBATCH -t 0-12:00
#SBATCH -J gruyere_runs
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vmazeeva@nygenome.org
#SBATCH --array=0-21               # adjust number of runs

#SBATCH --output=/gpfs/commons/home/vmazeeva/gruyere-expr/tau_runs/joint_model_run_%a/outputs/joint_model/log.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/gruyere-expr/tau_runs/joint_model_run_%a/outputs/joint_model/log.err

source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

root_dir=/gpfs/commons/home/vmazeeva/gruyere-expr/tau_runs

run_dir=${root_dir}/joint_model_run_${SLURM_ARRAY_TASK_ID}
config=${run_dir}/config.yaml
scripts=/gpfs/commons/home/vmazeeva/gruyere-expr/src

echo "Running Gruyere for: ${run_dir}"
python ${scripts}/gruyere_joint.py ${config}
