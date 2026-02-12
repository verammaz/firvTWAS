#!/bin/bash
#SBATCH --job-name=parmigiano_scaling
#SBATCH --output=scaling_experiments/slurm_%j.out
#SBATCH --error=scaling_experiments/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# This script submits multiple SLURM jobs for scaling experiments
# Usage: bash submit_scaling_experiments.sh

# Configuration
CONFIG_FILE="config_genewise.yaml"
GENE_LIST="gene_list_all.txt"  # Update with your full gene list file
OUTPUT_DIR="experiments"
BASE_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src"

# Array of gene counts to test
# Adjust these based on your computational constraints
GENE_COUNTS=(25 50 100 200 500 1000)

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Submit jobs for each gene count
for num_genes in "${GENE_COUNTS[@]}"; do
    echo "Submitting job for ${num_genes} genes..."
    
    # Adjust memory based on number of genes (rough estimate: ~0.1GB per gene)
    # You may need to tune this based on your data
    mem_gb=$((num_genes / 10 + 50))
    if [ $mem_gb -lt 50 ]; then
        mem_gb=50
    fi
    if [ $mem_gb -gt 500 ]; then
        mem_gb=500
    fi
    
    # Adjust time based on number of genes (rough estimate: ~1 min per gene)
    time_hours=$((num_genes / 60 + 2))
    if [ $time_hours -lt 2 ]; then
        time_hours=2
    fi
    if [ $time_hours -gt 48 ]; then
        time_hours=48
    fi
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=parmigiano_${num_genes}genes
#SBATCH --output=${OUTPUT_DIR}/slurm_${num_genes}genes_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_${num_genes}genes_%j.err
#SBATCH --time=${time_hours}:00:00
#SBATCH --mem=${mem_gb}G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Load necessary modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8  # if using GPU

# Activate conda environment if needed
# source activate parmigiano_env

cd ${BASE_DIR}

# Run the experiment
python run_scaling_experiment.py \
    --num_genes ${num_genes} \
    --config ${CONFIG_FILE} \
    --gene_list ${GENE_LIST} \
    --output_dir ${OUTPUT_DIR}

echo "Job completed for ${num_genes} genes"
EOF

    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Check results in: ${OUTPUT_DIR}/"

