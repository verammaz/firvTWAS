#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --mem=20G
#SBATCH --time=0-02:00
#SBATCH --array=0-21
#SBATCH -J plot_runs

#SBATCH --output=/gpfs/commons/home/vmazeeva/gruyere-expr/tau_runs/joint_model_run_%a/outputs/joint_model/plots/log.out
#SBATCH --error=/gpfs/commons/home/vmazeeva/gruyere-expr/tau_runs/joint_model_run_%a/outputs/joint_model/plots/log.err

source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
root_dir=/gpfs/commons/home/vmazeeva/gruyere-expr/tau_runs

run_dir=${root_dir}/joint_model_run_${SLURM_ARRAY_TASK_ID}
config=${run_dir}/config.yaml
scripts=/gpfs/commons/home/vmazeeva/gruyere-expr/src

python ${scripts}/plotting.py $config taus
python ${scripts}/plotting.py $config losses

