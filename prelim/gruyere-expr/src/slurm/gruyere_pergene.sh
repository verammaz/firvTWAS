#!/bin/bash

#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p cpu,bigmem,dev
#SBATCH --mem=300G
#SBATCH -t 0-24:00 # Runtime in D-HH:MM
#SBATCH -J gruyere_pergene # <-- name of job
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vmazeevanygenome.org        # Where to send mail]
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/gruyere_pergene_chr%a.out # Standard output and error log
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/gruyere_pergene_chr%a.err
#SBATCH --cpus-per-task=5                   # CPUs per task
#SBATCH --array=1-22


source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

scripts=/gpfs/commons/home/vmazeeva/gruyere-expr/src
config=/gpfs/commons/home/vmazeeva/gruyere-expr/tau_runs_/joint_model_run_0/config.yaml
python ${scripts}/gruyere_pergene.py ${config} ${SLURM_ARRAY_TASK_ID} 
