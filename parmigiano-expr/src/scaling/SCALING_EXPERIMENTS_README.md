# Scaling Experiments for Parmigiano Joint Fitting

This directory contains scripts to run scaling experiments that measure memory usage and computation time for joint fitting with different numbers of genes.

## Overview

The joint fitting step in Parmigiano learns global parameters (tau and threshold T) by fitting on a set of genes. Since fitting on all ~20K genes is computationally infeasible, these experiments help determine the optimal number of genes to use based on computational constraints.

## Files

1. **`run_scaling_experiment.py`**: Runs a single experiment with a specified number of genes, tracking memory and time
2. **`submit_scaling_experiments.sh`**: SLURM job submission script that submits multiple experiments in parallel
3. **`aggregate_and_plot_results.py`**: Aggregates results from all experiments and generates plots

## Setup

### 1. Prepare Gene List

Create a file containing all available genes (one per line, in the format used by your config):
```bash
# Example: gene_list_full.txt
chr1/ENSG00000123456
chr1/ENSG00000123457
...
```

### 2. Configure Experiment Parameters

Edit `submit_scaling_experiments.sh` to set:
- `CONFIG_FILE`: Path to your YAML config file
- `GENE_LIST`: Path to your full gene list file
- `GENE_COUNTS`: Array of gene counts to test (e.g., `(10 25 50 100 200 500 1000 2000)`)
- Memory and time estimates (adjust based on your cluster's limits)

## Usage

### Option 1: Submit All Experiments via SLURM (Recommended)

```bash
cd /gpfs/commons/home/vmazeeva/firvTWAS/parmigiano-expr/src
bash submit_scaling_experiments.sh
```

This will submit separate SLURM jobs for each gene count. Monitor jobs with:
```bash
squeue -u $USER
```


### Option 2: Run Single Experiment via SLURM

```bash
sbatch --job-name=parmigiano_100genes \
       --time=4:00:00 \
       --mem=100G \
       --cpus-per-task=4 \
       --wrap="python run_scaling_experiment.py --num_genes 100 --config config_genewise.yaml --gene_list gene_list_full.txt --output_dir scaling_experiments"
```

## Results

### Output Files

Each experiment creates:
- `results_{N}genes.json`: Summary statistics (memory, time, etc.)
- `times_{N}genes.txt`: Time per epoch
- `memory_{N}genes.txt`: Memory usage per epoch

### Aggregating Results

After all experiments complete, aggregate and plot results:

```bash
python aggregate_and_plot_results.py \
    --results_dir scaling_experiments \
    --output_dir scaling_experiments
```

This generates:
- `aggregated_results.csv`: All results in a single CSV file
- `scaling_plots.png`: Main plots (memory vs genes, time vs genes)
- `scaling_plots_detailed.png`: Additional efficiency plots

### Plots Generated

1. **Peak Memory vs Number of Genes** (linear and log-log scales)
2. **Total Training Time vs Number of Genes**
3. **Time per Epoch vs Number of Genes**
4. **Variants vs Genes** (to understand data scaling)
5. **Memory Efficiency** (memory per gene)
6. **Time Efficiency** (time per gene)

## Interpreting Results

### Memory Scaling
- Look for linear or sub-linear scaling
- Identify the maximum number of genes that fit in available memory
- Check if memory per gene decreases with more genes (efficiency gains)

### Time Scaling
- Estimate total time for different gene counts
- Consider time per epoch for planning iterations
- Identify bottlenecks (e.g., if time scales super-linearly)

### Recommendations

1. **Start small**: Begin with 10-50 genes to establish baseline
2. **Exponential search**: Use powers of 2 (10, 25, 50, 100, 200, 500, 1000, 2000)
3. **Resource limits**: Set SLURM memory/time based on:
   - Memory: ~0.1-0.2 GB per gene (adjust based on variant density)
   - Time: ~1-2 minutes per gene per epoch (adjust based on epochs)

4. **Optimal gene count**: Choose based on:
   - Maximum genes that fit in available memory
   - Acceptable computation time
   - Sufficient statistical power (more genes = better tau estimates)

## Troubleshooting

### Out of Memory Errors
- Reduce number of genes in experiment
- Increase SLURM memory allocation
- Check if data loading is memory-efficient

### Jobs Timing Out
- Increase SLURM time limit
- Reduce number of epochs for scaling experiments
- Check for computational bottlenecks

### Missing Results
- Check SLURM error logs: `scaling_experiments/slurm_*_*.err`
- Verify gene list file exists and is readable
- Ensure config file paths are correct

## Example Workflow

```bash
# 1. Prepare full gene list (if not already done)
# ... create gene_list_full.txt ...

# 2. Edit submit_scaling_experiments.sh with your parameters

# 3. Submit all experiments
bash submit_scaling_experiments.sh

# 4. Wait for jobs to complete (check with squeue)

# 5. Aggregate and plot results
python aggregate_and_plot_results.py

# 6. Review plots and select optimal gene count
```

## Notes

- Experiments run independently, so you can submit them all at once
- Each experiment uses the first N genes from your gene list
- Consider randomizing gene selection if gene order matters
- Memory tracking uses `psutil` - ensure it's installed: `pip install psutil`

