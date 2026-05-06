#!/bin/bash

#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p cpu,bigmem,dev
#SBATCH --mem=100G
#SBATCH -t 0-8:00 # Runtime in D-HH:MM
#SBATCH -J genotype_matrices # <-- name of job
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vmazeevanygenome.org        # Where to send mail]
#SBATCH --output=/gpfs/commons/home/vmazeeva/bash_outputs/%j_gruyere_joint.out # Standard output and error log
#SBATCH --error=/gpfs/commons/home/vmazeeva/bash_outputs/%j_gruyere_joint.err
#SBATCH --cpus-per-task=5                   # CPUs per task


source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

scripts=/gpfs/commons/home/vmazeeva/gruyere-expr/src
python ${scripts}/gruyere_joint.py /gpfs/commons/home/vmazeeva/gruyere-expr/example/inputs.yaml 
