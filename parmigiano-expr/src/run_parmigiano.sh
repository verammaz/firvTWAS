#!/bin/bash
#SBATCH --job-name=parmigiano_seed_genes
#SBATCH --output=seed_genes/slurm_%j.out
#SBATCH --error=seed_genes/slurm_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

cd /gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src

python run_parmigiano.py --gene_list gene_list_seed.txt --config config_genewise.yaml --output_dir seed_genes
