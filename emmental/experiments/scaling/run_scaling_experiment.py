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

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import utils
import load_data
from models import parmigiano_expression

# Setup logging
logger = None


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
        - posterior_sampling_time: time taken for posterior sampling (seconds)
        - posterior_stats: dictionary of posterior statistics
    """
    global logger
    if logger is None:
        logger = utils.get_logger()
    
    model = parmigiano_expression()
    to_optimize = ['tau', 'threshold']
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(poutine.block(model, hide=to_optimize))) 
    guide.add(AutoDelta(poutine.block(model, expose=to_optimize)))    
    
    # Setup optimizer with gradient clipping to prevent NaNs (matching run_parmigiano.py)
    clip_norm = config.get('clip_norm', 10.0)
    try:
        from pyro.optim import ClippedAdam
        HAS_CLIPPED_ADAM = True
    except (ImportError, AttributeError):
        HAS_CLIPPED_ADAM = False
    
    if HAS_CLIPPED_ADAM:
        adam = ClippedAdam({"lr": config['lr'], "clip_norm": clip_norm})
        logger.info(f"Using ClippedAdam with clip_norm={clip_norm}")
    else:
        logger.warning("ClippedAdam not available, using regular Adam. Consider reducing learning rate if NaNs occur.")
        adam = pyro.optim.Adam({"lr": config['lr']})
    
    svi = SVI(model, guide=guide, optim=adam, loss=Trace_ELBO()) 
    pyro.clear_param_store()
    
    # Training loop with tracking
    losses = []
    times = []
    memory_usage = []
    
    logger.info(f"Training for {config['epochs']} epochs...")
    initial_memory = get_memory_usage() if track_memory else 0
    
    for epoch in tqdm(range(config['epochs'])): 
        start = time.time()
        
        if track_memory:
            mem_before = get_memory_usage()
        
        try:
            loss = svi.step(data, config)
            # Check for NaN loss
            if torch.isnan(torch.tensor(loss)) or not np.isfinite(loss):
                logger.warning(f"NaN or infinite loss detected at epoch {epoch}. Stopping training.")
                break
        except ValueError as e:
            if "nan" in str(e).lower() or "invalid" in str(e).lower():
                logger.error(f"Numerical instability detected at epoch {epoch}: {e}")
                logger.error("This may be due to gradient explosion in the Dirichlet parameter 'tau'.")
                raise
            else:
                raise
        
        elapsed = time.time() - start
        times.append(elapsed)
        losses.append(float(loss))
        
        if track_memory:
            mem_after = get_memory_usage()
            memory_usage.append(mem_after)
        
        if epoch % 10 == 0:
            mem_str = f", Memory: {mem_after:.2f} GB" if track_memory else ""
            logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Time = {elapsed:.2f}s{mem_str}")
    
    peak_memory = max(memory_usage) if memory_usage else get_memory_usage()
    
    logger.info("Training complete!")
    
    # Generate posterior samples with timing
    logger.info("Generating posterior samples...")
    if track_memory:
        mem_before_posterior = get_memory_usage()
    
    posterior_start = time.time()
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=config.get('n_posterior', 50))
    samples = predictive(data, config)
    posterior_sampling_time = time.time() - posterior_start
    
    if track_memory:
        mem_after_posterior = get_memory_usage()
        logger.info(f"Posterior sampling memory: {mem_after_posterior:.2f} GB (delta: {mem_after_posterior - mem_before_posterior:.2f} GB)")
    
    # Compute posterior statistics
    posterior_stats = {
        k: {
            'mean': v.mean(0).cpu().numpy(),
            'std': v.std(0).cpu().numpy()
        } 
        for k, v in samples.items()
    }
    
    logger.info(f"Posterior sampling complete! Time: {posterior_sampling_time:.2f}s ({posterior_sampling_time/60:.2f} minutes)")
    
    return losses, times, memory_usage, peak_memory, posterior_sampling_time, posterior_stats


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
    global logger
    # Ensure logger is initialized (should be set up in main(), but ensure it exists)
    if logger is None:
        # Get log level from config if available, otherwise default to INFO
        log_level = config.get('log_level', 'INFO')
        log_file = config.get('log_file', None)
        logger = utils.setup_logging(log_level, log_file)
    
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: Joint fitting with {num_genes} genes")
    logger.info("=" * 70)
    
    # Load full gene list
    with open(gene_list_file, 'r') as f:
        all_genes = [line.strip() for line in f if line.strip()]
    
    # Select subset of genes
    if num_genes > len(all_genes):
        logger.warning(f"Requested {num_genes} genes but only {len(all_genes)} available. Using all genes.")
        selected_genes = all_genes
        num_genes = len(all_genes)
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
            selected_genes = np.random.choice(all_genes, size=num_genes, replace=False).tolist()
            logger.info(f"Randomly selected {num_genes} genes (seed={random_seed})")
        else:
            selected_genes = all_genes[:num_genes]
            logger.info(f"Using first {num_genes} genes from list")
    
    # Update config with selected genes
    config['genes'] = selected_genes
    config['gene_list'] = selected_genes
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")
    
    # Track memory before loading data
    mem_before_load = get_memory_usage()
    logger.info(f"Memory before loading data: {mem_before_load:.2f} GB")
    
    # Load data
    logger.info("Loading phenotype and covariate data...")
    logger.info(f"About to call load_residualized_covariates with device={device}")
    try:
        X, Y = load_data.load_residualized_covariates(config, device)
        logger.info("Successfully loaded residualized covariates")
    except Exception as e:
        logger.error(f"Error in load_residualized_covariates: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    logger.info("\nLoading genotype and annotation data...")
    # Handle new return signature from load_genes (returns 4 values)
    load_result = load_data.load_genes(config)
    if len(load_result) == 4:
        G, Z, variant_ids_G, variant_ids_Z = load_result
    else:
        # Fallback for older version
        G, Z = load_result[:2]
        variant_ids_G = list(G.columns) if hasattr(G, 'columns') else None
        variant_ids_Z = list(Z.index) if hasattr(Z, 'index') else None
    
    mem_after_load = get_memory_usage()
    logger.info(f"Memory after loading data: {mem_after_load:.2f} GB")
    logger.info(f"Data loading memory: {mem_after_load - mem_before_load:.2f} GB")
    
    # Create data tensors
    logger.info("\nCreating data tensors...")
    data = load_data.DataTensors.from_pandas(G, Z, X, Y, device, config)
    
    mem_after_tensors = get_memory_usage()
    logger.info(f"Memory after creating tensors: {mem_after_tensors:.2f} GB")
    
    logger.info(f"\nData summary:")
    logger.info(f"  Samples: {data.G.shape[0]}")
    logger.info(f"  Variants: {data.G.shape[1]}")
    logger.info(f"  Genes: {data.num_genes}")
    logger.info(f"  Annotations: {data.num_anno}")
    logger.info(f"  Covariates: {data.num_cov}")
    logger.info("")
    
    # Fit model with tracking
    start_time = time.time()
    losses, times, memory_usage, peak_memory, posterior_sampling_time, posterior_stats = fit_parmigiano_with_tracking(
        data, config, track_memory=True
    )
    total_time = time.time() - start_time
    training_time = total_time - posterior_sampling_time
    
    # Collect results
    results = {
        'num_genes': num_genes,
        'num_variants': data.G.shape[1],
        'num_samples': data.G.shape[0],
        'num_annotations': data.num_anno,
        'total_time': total_time,
        'training_time': training_time,
        'posterior_sampling_time': posterior_sampling_time,
        'avg_time_per_epoch': np.mean(times),
        'std_time_per_epoch': np.std(times),
        'peak_memory': peak_memory,
        'avg_memory': np.mean(memory_usage) if memory_usage else peak_memory,
        'std_memory': np.std(memory_usage) if memory_usage else 0,
        'memory_after_load': mem_after_load,
        'memory_after_tensors': mem_after_tensors,
        'final_loss': losses[-1],
        'epochs': config['epochs'],
        'n_posterior': config.get('n_posterior', 50)
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
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Number of genes: {num_genes}")
    logger.info(f"Number of variants: {data.G.shape[1]}")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"  Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    logger.info(f"  Posterior sampling time: {posterior_sampling_time:.2f} seconds ({posterior_sampling_time/60:.2f} minutes)")
    logger.info(f"Average time per epoch: {np.mean(times):.2f} seconds")
    logger.info(f"Peak memory: {peak_memory:.2f} GB")
    logger.info(f"Average memory: {np.mean(memory_usage):.2f} GB")
    logger.info(f"Results saved to: {results_file}")
    logger.info("=" * 70)
    
    return results


def main():
    """Main function to run scaling experiments"""
    global logger
    
    # Parse arguments for scaling experiment
    parser = argparse.ArgumentParser(description="Run scaling experiments for Parmigiano")
    parser.add_argument('--num_genes', type=int, required=True,
                       help='Number of genes to use for joint fitting')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--output_dir', type=str, default='scaling_experiments',
                       help='Directory to save experiment results')
    parser.add_argument('--gene_list', type=str, required=True,
                       help='Path to file with list of all available genes')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for gene selection (None = use first N genes)')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Optional path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = args.log_level
    log_file = args.log_file
    logger = utils.setup_logging(log_level, log_file)
    
    # Load configuration
    yaml_config = utils.load_yaml(args.config)
    # Create a minimal args object for fill_defaults (uses getattr with None default)
    class EmptyArgs:
        pass
    dummy_args = EmptyArgs()
    config = utils.fill_defaults(dummy_args, yaml_config)
    
    logger.info("=" * 50)
    logger.info("SCALING EXPERIMENTS - Bayesian Hierarchical Gene Analysis")
    logger.info("=" * 50)
    logger.info("\nConfiguration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Run experiment
    results = run_experiment(
        num_genes=args.num_genes,
        config=config,
        output_dir=args.output_dir,
        gene_list_file=args.gene_list,
        random_seed=args.random_seed
    )
    
    logger.info("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()

