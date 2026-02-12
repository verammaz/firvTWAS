#!/usr/bin/env python3
"""
Script to run scaling experiments with different numbers of genes.
Tracks memory usage and time for joint fitting step.
"""

import torch
import pyro
from pyro import poutine
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm import tqdm
import time
import psutil
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

import utils
import load_data
from models import parmigiano_expression


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB


def fit_parmigiano_with_tracking(data, config, track_memory=True):
    """
    Fit parmigiano model with memory and time tracking
    INPUT:
        - data: DataTensors object containing all input data
        - config: configuration dictionary
        - track_memory: whether to track memory usage
    OUTPUT:
        - losses: list of loss values per epoch
        - times: list of training times per epoch
        - memory_usage: list of memory usage per epoch (GB)
        - peak_memory: peak memory usage (GB)
        - posterior_stats: dictionary of posterior statistics
    """
    
    model = parmigiano_expression()
    to_optimize = ['tau', 'threshold']
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(poutine.block(model, hide=to_optimize))) 
    guide.add(AutoDelta(poutine.block(model, expose=to_optimize)))    
    
    adam = pyro.optim.Adam({"lr": config['lr']})
    svi = SVI(model, guide=guide, optim=adam, loss=Trace_ELBO()) 
    pyro.clear_param_store()
    
    # Training loop with tracking
    losses = []
    times = []
    memory_usage = []
    
    print(f"Training for {config['epochs']} epochs...")
    initial_memory = get_memory_usage() if track_memory else 0
    
    for epoch in tqdm(range(config['epochs'])): 
        start = time.time()
        
        if track_memory:
            mem_before = get_memory_usage()
        
        loss = svi.step(data, config)
        
        elapsed = time.time() - start
        times.append(elapsed)
        losses.append(float(loss))
        
        if track_memory:
            mem_after = get_memory_usage()
            memory_usage.append(mem_after)
        
        if epoch % 10 == 0:
            mem_str = f", Memory: {mem_after:.2f} GB" if track_memory else ""
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Time = {elapsed:.2f}s{mem_str}")
    
    peak_memory = max(memory_usage) if memory_usage else get_memory_usage()
    
    # Generate posterior samples
    print("Generating posterior samples...")
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=config['n_posterior']) 
    samples = predictive(data, config)
    
    # Compute posterior statistics
    posterior_stats = {
        k: {
            'mean': v.mean(0).cpu().numpy(),
            'std': v.std(0).cpu().numpy()
        } 
        for k, v in samples.items()
    }
    
    print("Training complete!")
    return losses, times, memory_usage, peak_memory, posterior_stats


def run_experiment(num_genes, config, output_dir, gene_list_file, random_seed=None):
    """
    Run a single experiment with specified number of genes
    
    Args:
        num_genes: Number of genes to use for joint fitting
        config: Configuration dictionary
        output_dir: Directory to save results
        gene_list_file: Path to file with all available genes
        random_seed: Optional random seed for gene selection (None = use first N genes)
    """
    print("=" * 70)
    print(f"EXPERIMENT: Joint fitting with {num_genes} genes")
    print("=" * 70)
    
    # Load full gene list
    with open(gene_list_file, 'r') as f:
        all_genes = [line.strip() for line in f if line.strip()]
    
    # Select subset of genes
    if num_genes > len(all_genes):
        print(f"Warning: Requested {num_genes} genes but only {len(all_genes)} available. Using all genes.")
        selected_genes = all_genes
        num_genes = len(all_genes)
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
            selected_genes = np.random.choice(all_genes, size=num_genes, replace=False).tolist()
            print(f"Randomly selected {num_genes} genes (seed={random_seed})")
        else:
            selected_genes = all_genes[:num_genes]
            print(f"Using first {num_genes} genes from list")
    
    # Update config with selected genes
    config['genes'] = selected_genes
    config['gene_list'] = selected_genes
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Track memory before loading data
    mem_before_load = get_memory_usage()
    print(f"Memory before loading data: {mem_before_load:.2f} GB")
    
    # Load data
    print("Loading phenotype and covariate data...")
    X, Y = load_data.load_residualized_covariates(config, device)
    
    print("\nLoading genotype and annotation data...")
    G, Z = load_data.load_genes(config)
    
    mem_after_load = get_memory_usage()
    print(f"Memory after loading data: {mem_after_load:.2f} GB")
    print(f"Data loading memory: {mem_after_load - mem_before_load:.2f} GB")
    
    # Create data tensors
    print("\nCreating data tensors...")
    data = load_data.DataTensors.from_pandas(G, Z, X, Y, device, config)
    
    mem_after_tensors = get_memory_usage()
    print(f"Memory after creating tensors: {mem_after_tensors:.2f} GB")
    
    print(f"\nData summary:")
    print(f"  Samples: {data.G.shape[0]}")
    print(f"  Variants: {data.G.shape[1]}")
    print(f"  Genes: {data.num_genes}")
    print(f"  Annotations: {data.num_anno}")
    print(f"  Covariates: {data.num_cov}")
    print()
    
    # Fit model with tracking
    start_time = time.time()
    losses, times, memory_usage, peak_memory, posterior_stats = fit_parmigiano_with_tracking(
        data, config, track_memory=True
    )
    total_time = time.time() - start_time
    
    # Collect results
    results = {
        'num_genes': num_genes,
        'num_variants': data.G.shape[1],
        'num_samples': data.G.shape[0],
        'num_annotations': data.num_anno,
        'total_time': total_time,
        'avg_time_per_epoch': np.mean(times),
        'std_time_per_epoch': np.std(times),
        'peak_memory': peak_memory,
        'avg_memory': np.mean(memory_usage) if memory_usage else peak_memory,
        'std_memory': np.std(memory_usage) if memory_usage else 0,
        'memory_after_load': mem_after_load,
        'memory_after_tensors': mem_after_tensors,
        'final_loss': losses[-1],
        'epochs': config['epochs']
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"results_{num_genes}genes.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed times and memory
    times_file = os.path.join(output_dir, f"times_{num_genes}genes.txt")
    with open(times_file, 'w') as f:
        f.write('\n'.join(map(str, times)))
    
    memory_file = os.path.join(output_dir, f"memory_{num_genes}genes.txt")
    with open(memory_file, 'w') as f:
        f.write('\n'.join(map(str, memory_usage)))
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Number of genes: {num_genes}")
    print(f"Number of variants: {data.G.shape[1]}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per epoch: {np.mean(times):.2f} seconds")
    print(f"Peak memory: {peak_memory:.2f} GB")
    print(f"Average memory: {np.mean(memory_usage):.2f} GB")
    print(f"Results saved to: {results_file}")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run scaling experiments for Parmigiano")
    parser.add_argument('--num_genes', type=int, required=True,
                       help='Number of genes to use for joint fitting')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--gene_list', type=str, required=True,
                       help='Path to file with list of all available genes')
    parser.add_argument('--output_dir', type=str, default='scaling_experiments',
                       help='Directory to save experiment results')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for gene selection (None = use first N genes)')
    
    args = parser.parse_args()
    
    # Load configuration
    yaml_config = utils.load_yaml(args.config)
    # Create a minimal args object for fill_defaults (uses getattr with None default)
    class EmptyArgs:
        pass
    dummy_args = EmptyArgs()
    config = utils.fill_defaults(dummy_args, yaml_config)
    
    # Run experiment
    results = run_experiment(
        num_genes=args.num_genes,
        config=config,
        output_dir=args.output_dir,
        gene_list_file=args.gene_list,
        random_seed=args.random_seed
    )
    
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()

