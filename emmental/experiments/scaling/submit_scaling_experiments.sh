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
CONFIG_FILE="/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src/config_genewise.yaml"
GENE_LIST="/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src/scaling/gene_list_all.txt"  
OUTPUT_DIR="experiments"
BASE_DIR="/gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src/scaling"

# Array of gene counts to test
# Adjust these based on your computational constraints
GENE_COUNTS=(25) # 50 100 200 500 1000)

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Ensure PATH includes standard directories (important for SLURM environments)
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Activate conda environment
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate

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
    
    # Adjust time based on number of genes
    time_hours=$((num_genes / 60 + 10))
    if [ $time_hours -lt 10 ]; then
        time_hours=10
    fi
    if [ $time_hours -gt 48 ]; then
        time_hours=48
    fi
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=scaling_${num_genes}genes
#SBATCH --output=${OUTPUT_DIR}/slurm_${num_genes}genes.out
#SBATCH --error=${OUTPUT_DIR}/slurm_${num_genes}genes.err
#SBATCH --time=48:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Load necessary modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8  # if using GPU

# Activate conda environment if needed
# source activate parmigiano_env

cd ${BASE_DIR}

# Run the experiment
python -u run_scaling_experiment.py \
    --num_genes ${num_genes} \
    --config ${CONFIG_FILE} \
    --gene_list ${GENE_LIST} \
    --output_dir ${OUTPUT_DIR}/${num_genes}genes \
    --log_level DEBUG

echo "Job completed for ${num_genes} genes"
EOF

    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Check results in: ${OUTPUT_DIR}/"

