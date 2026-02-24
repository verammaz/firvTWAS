import torch
import pyro
import yaml
from pyro import poutine
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO, Predictive
# Try to import ClippedAdam, fallback to Adam if not available
try:
    from pyro.optim import ClippedAdam
    HAS_CLIPPED_ADAM = True
except (ImportError, AttributeError):
    HAS_CLIPPED_ADAM = False
from tqdm import tqdm
import time
from sklearn.metrics import r2_score
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os

import utils
import load_data
from save_outputs import save_results
from models import parmigiano_expression


def fit_parmigiano(data, config):
    """
    Fit parmigiano model using stochastic variational inference
    INPUT:
        - data: DataTensors object containing all input data
        - config: configuration dictionary
    OUTPUT:
        - losses: list of loss values per epoch
        - times: list of training times per epoch
        - posterior_stats: dictionary of posterior statistics for each parameter
        - beta_samples: dictionary of beta samples per gene
        - mu_samples: dictionary of mu samples per gene
    """
    logger = utils.get_logger()
    
    model = parmigiano_expression() # Initialize model and guide
    to_optimize = ['tau', 'threshold'] # Define paramater-specific guide optimizations
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(poutine.block(model, hide=to_optimize))) 
    guide.add(AutoDelta(poutine.block(model, expose=to_optimize)))    
    
    # Setup optimizer with gradient clipping to prevent NaNs
    # Use ClippedAdam to prevent gradient explosion that can cause NaNs in Dirichlet parameters
    clip_norm = config.get('clip_norm', 10.0)  # Default gradient clipping norm
    if HAS_CLIPPED_ADAM:
        adam = ClippedAdam({"lr": config['lr'], "clip_norm": clip_norm})
        logger.info(f"Using ClippedAdam with clip_norm={clip_norm}")
    else:
        # Fallback to regular Adam if ClippedAdam is not available
        logger.warning("ClippedAdam not available, using regular Adam. Consider reducing learning rate if NaNs occur.")
        adam = pyro.optim.Adam({"lr": config['lr']})
    svi = SVI(model, guide=guide, optim=adam, loss=Trace_ELBO()) 
    pyro.clear_param_store()
    
    # Training loop
    losses = []
    times = []
    tau_history = []  # Track tau values across epochs
    
    logger.info(f"Training for {config['epochs']} epochs...")
    for epoch in tqdm(range(config['epochs'])): 
        start = time.time()
        try:
            loss = svi.step(data, config)
            # Check for NaN loss
            if torch.isnan(torch.tensor(loss)) or not np.isfinite(loss):
                logger.warning(f"NaN or infinite loss detected at epoch {epoch}. Stopping training.")
                break
            times.append(time.time() - start)
            losses.append(float(loss))
            
            # Extract tau values from the guide (tau is optimized with AutoDelta)
            # Following the approach from gruyere_joint.py
            try:
                try:
                    current_tau = pyro.param("tau").detach().cpu().numpy().copy()
                except KeyError:
                    logger.debug("KeyError: tau not found in param store")
                    logger.debug(f"Param store keys: {pyro.get_param_store().keys()}")
                    # sometimes pyro stores parameter names with prefixes (e.g., guide.tau)
                    tau_params = {name: pyro.param(name).detach().cpu().numpy().copy()
                                for name in pyro.get_param_store().keys() if "tau" in name}
                    # Try AutoGuideList.1.tau (AutoDelta is the second guide, index 1)
                    if "AutoGuideList.1.tau" in tau_params:
                        current_tau = tau_params["AutoGuideList.1.tau"]
                    elif len(tau_params) > 0:
                        # Use the first tau parameter found
                        current_tau = list(tau_params.values())[0]
                    else:
                        current_tau = None
                
                if current_tau is not None:
                    # Ensure it's 1D array
                    if current_tau.ndim > 1:
                        current_tau = current_tau.flatten()
                    tau_history.append(current_tau.copy())
                else:
                    tau_history.append(None)
            except Exception as e:
                # If tau not accessible, skip tracking for this epoch
                if epoch == 0:
                    logger.debug(f"Could not access tau parameter: {e}")
                tau_history.append(None)
            
            if epoch % 10 == 0:  # Print every 10 epochs
                logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
        except ValueError as e:
            if "nan" in str(e).lower() or "invalid" in str(e).lower():
                logger.error(f"Numerical instability detected at epoch {epoch}: {e}")
                logger.error("This may be due to gradient explosion in the Dirichlet parameter 'tau'.")
                logger.error("Suggested fixes:")
                logger.error("  1. Reduce learning rate (try lr=0.01 or lower)")
                logger.error("  2. Add 'clip_norm' to config (e.g., clip_norm: 5.0)")
                logger.error("  3. Check for issues in input data (NaNs, extreme values)")
                raise
            else:
                raise
    
    # Generate posterior samples
    logger.info("Generating posterior samples...")
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
    
    # Extract beta values from posterior samples
    # Beta is now directly available from pyro.deterministic in the model
    logger.info("Extracting beta values from posterior...")
    beta_samples = {}
    try:
        for gene_name in data.gene_names:
            beta_key = f"beta_{gene_name}"
            if beta_key in samples:
                # Beta is directly available from the samples
                beta_gene_samples = samples[beta_key]  # (n_posterior, num_variants) or (n_posterior, 1, num_variants)
                # Convert to numpy if it's a tensor
                if isinstance(beta_gene_samples, torch.Tensor):
                    beta_gene_samples = beta_gene_samples.cpu().numpy()
                # Squeeze middle dimension if present: (n_posterior, 1, num_variants) -> (n_posterior, num_variants)
                if beta_gene_samples.ndim == 3:
                    beta_gene_samples = beta_gene_samples.squeeze(axis=1)
                beta_samples[gene_name] = beta_gene_samples
            else:
                logger.warning(f"beta_{gene_name} not found in samples. Available keys: {list(samples.keys())}")
    except Exception as e:
        logger.error(f"Error extracting beta values: {e}")
    
    # Extract mu values from posterior samples
    # Mu is now directly available from pyro.deterministic in the model
    logger.info("Extracting mu values from posterior...")
    mu_samples = {}
    try:
        for gene_name in data.gene_names:
            mu_key = f"mu_{gene_name}"
            if mu_key in samples:
                # Mu is directly available from the samples
                mu_gene_samples = samples[mu_key]  # (n_posterior, num_variants) or (n_posterior, 1, num_variants)
                # Convert to numpy if it's a tensor
                if isinstance(mu_gene_samples, torch.Tensor):
                    mu_gene_samples = mu_gene_samples.cpu().numpy()
                # Squeeze middle dimension if present: (n_posterior, 1, num_variants) -> (n_posterior, num_variants)
                if mu_gene_samples.ndim == 3:
                    mu_gene_samples = mu_gene_samples.squeeze(axis=1)
                mu_samples[gene_name] = mu_gene_samples
            else:
                logger.warning(f"mu_{gene_name} not found in samples. Available keys: {list(samples.keys())}")
    except Exception as e:
        logger.error(f"Error extracting mu values: {e}")
        
    logger.info("Training complete!")
    return losses, times, posterior_stats, beta_samples, mu_samples, tau_history


def calculate_r2(data, beta_samples, gene_names):
    """
    Calculate R2 on a dataset using G * beta_mean (without running through model)
    INPUT:
        - data: DataTensors object containing data (train or test)
        - beta_samples: dictionary of beta samples for each gene (gene_name -> (n_posterior, num_variants))
        - gene_names: list of gene names
    OUTPUT:
        - dictionary of R2 scores per gene
    """
    r2_scores = {}
    logger = utils.get_logger()
    
    for gene_name in gene_names:
        G_gene, _, _ = data.get_gene_data(gene_name)
        
        # Extract gene key for Y dictionary
        if "/" in gene_name:
            gene_key = gene_name.split("/")[1]
        else:
            gene_key = gene_name
        
        # Get Y for this gene
        if gene_key in data.Y:
            Y_gene = data.Y[gene_key]
        elif gene_name in data.Y:
            Y_gene = data.Y[gene_name]
        else:
            logger.warning(f"Could not find Y for gene {gene_name} (tried keys: {gene_key}, {gene_name})")
            continue
        
        # Get mean beta for this gene
        beta_mean = torch.tensor(beta_samples[gene_name].mean(axis=0), device=data.device)
        # Ensure beta_mean is 1D for matmul
        if beta_mean.ndim > 1:
            beta_mean = beta_mean.squeeze()
        
        # Predict: G * beta (directly, no model forward pass)
        # G_gene: (num_samples, num_variants), beta_mean: (num_variants,)
        predictions = G_gene.matmul(beta_mean)

        # Calculate R2
        y_true = Y_gene.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        r2 = r2_score(y_true, y_pred)
        r2_scores[gene_name] = r2
    
    return r2_scores


def main():
    """Main execution function"""
    # Load configuration
    args = utils.parse_args()
    yaml_config = None
    if args.config:
        yaml_config = utils.load_yaml(args.config)
    config = utils.fill_defaults(args, yaml_config)
    
    # Setup logging
    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', None)
    logger = utils.setup_logging(log_level, log_file)
    
    logger.info("=" * 50)
    logger.info("PARMIGIANO - Bayesian Hierarchical Gene Analysis")
    logger.info("=" * 50)
    logger.info("\nConfiguration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")
    
    if isinstance(config['gene_list'], str):  # Could be path to file or comma-separated list
        # Check if it's a file path (contains '/' or ends with common file extensions)
        if os.path.exists(config['gene_list']):
            # It's a file path
            with open(config['gene_list'], 'r') as f:
                config['genes'] = [line.strip() for line in f if line.strip()]
        else:
            # It's a comma-separated list
            config['genes'] = [gene.strip() for gene in config['gene_list'].split(',') if gene.strip()]
    else:  # List provided directly
        config['genes'] = config['gene_list']

    # Load data
    logger.info("Loading data...")
    logger.info("Loading phenotype and covariate data...")
    covariates_scaled, residualized_Y = load_data.load_residualized_covariates(config, device)
    logger.info("Loading genotype and annotation data...")
    G, Z, variant_ids_G, variant_ids_Z = load_data.load_genes(config)

    if config.get('train_test', False):
        logger.info("Train/Test Split")
        # Get train/test sample IDs before resetting index
        train_sample_ids = covariates_scaled[~((covariates_scaled['cohort_ROSMAP']==1.0) & (covariates_scaled['tissue_Dorsolateral Pre-frontal Cortex (DLPFC)']==1.0))].index
        test_sample_ids = covariates_scaled[((covariates_scaled['cohort_ROSMAP']==1.0) & (covariates_scaled['tissue_Dorsolateral Pre-frontal Cortex (DLPFC)']==1.0))].index
        
        # Index G using sample IDs (G has sample IDs as index, not integer positions)
        G_train_df = G.loc[train_sample_ids]
        G_test_df = G.loc[test_sample_ids]
        
        # Reset index for covariates after getting sample IDs
        covariates = covariates_scaled.reset_index()
        train_idx = covariates[~((covariates['cohort_ROSMAP']==1.0) & (covariates['tissue_Dorsolateral Pre-frontal Cortex (DLPFC)']==1.0))].index
        test_idx = covariates[((covariates['cohort_ROSMAP']==1.0) & (covariates['tissue_Dorsolateral Pre-frontal Cortex (DLPFC)']==1.0))].index
        
        X = covariates.loc[train_idx]
        X.set_index('sample_id', inplace=True)
        Y = {gene: expr[train_idx] for gene, expr in residualized_Y.items()}
        X_test = covariates.loc[test_idx]
        X_test.set_index('sample_id', inplace=True)
        Y_test = {gene: expr[test_idx] for gene, expr in residualized_Y.items()}
        
        G = G_train_df
        G_test = G_test_df

        logger.info("Creating test data tensors...")
        data_test = load_data.DataTensors.from_pandas(G_test, Z, X_test, Y_test, device, config)

    
    else:
        data_test = None
        X, Y = covariates_scaled, residualized_Y


    # Create data tensors
    logger.info("\nCreating train data tensors...")
    data_train = load_data.DataTensors.from_pandas(G, Z, X, Y, device, config)
    
    if data_test is not None:
        logger.info("\n" + "=" * 50)
        logger.info("Train/Test Split")
        logger.info("=" * 50)
        logger.info(f"  Training samples: {data_train.G.shape[0]}")
        logger.info(f"  Test samples: {data_test.G.shape[0]}")
        logger.info("")
    
    logger.info(f"\nData summary:")
    if data_test is not None:
        logger.info(f"  Samples: {data_train.G.shape[0]} Train + {data_test.G.shape[0]} Test = {data_train.G.shape[0] + data_test.G.shape[0]}")
    else:
        logger.info(f"  Samples: {data_train.G.shape[0]} Train")
    logger.info(f"  Variants: {data_train.G.shape[1]}") # same as test
    logger.info(f"  Genes: {data_train.num_genes}") # same as test
    logger.info(f"  Annotations: {data_train.num_anno}") # same as test
    logger.info(f"  Covariates: {data_train.num_cov}") # same as test
    logger.info("")
    
    # Warn if very few genes - can cause instability in Dirichlet parameter
    if data_train.num_genes < 5:
        logger.warning(f"Only {data_train.num_genes} gene(s) detected. This may cause numerical instability.")
        logger.warning("  The model works best with 10+ genes. Consider using more genes for stable training.")
        logger.warning("  If you must proceed, the learning rate will be automatically reduced.")
        # Automatically reduce learning rate for few genes
        original_lr = config['lr']
        config['lr'] = config['lr'] * 0.1  # Reduce by 10x
        logger.warning(f"  Learning rate reduced from {original_lr} to {config['lr']} for stability.")
        logger.info("")

    # save data summary to config
    config['data_summary'] = {
        'samples_train': data_train.G.shape[0],
        'samples_test': data_test.G.shape[0] if data_test is not None else None,
        'variants': data_train.G.shape[1],
        'genes': data_train.num_genes, 
        'annotations': data_train.num_anno,
        'covariates': data_train.num_cov
    }



    # Disbale gradient norm clipping
    HAS_CLIPPED_ADAM = False
    
    # Fit model on training data
    losses, times, posterior_stats, beta_samples, mu_samples, tau_history = fit_parmigiano(data_train, config)
  

    # Print summary statistics
    logger.info("\n" + "=" * 50)
    logger.info("Training Summary")
    logger.info("=" * 50)
    logger.info(f"Final loss: {losses[-1]:.4f}")
    logger.info(f"Average time per epoch: {sum(times)/len(times):.2f}s")
    logger.info(f"Total training time: {sum(times):.2f}s")

    # Calculate R2 on train set
    logger.info("\nCalculating R2 on train set...")
    train_r2 = calculate_r2(data_train, beta_samples, data_train.gene_names)
    if train_r2 is not None:
        logger.info("\n" + "=" * 50)
        logger.info("Average Train Set R2 Scores")
        logger.info("=" * 50)
        avg_r2 = np.mean(list(train_r2.values()))
        logger.info(f"\n  Average R2: {avg_r2:.4f}")
    
    # Calculate R2 on test set if available
    test_r2 = None
    if data_test is not None:
        logger.info("\nCalculating R2 on test set...")
        test_r2 = calculate_r2(data_test, beta_samples, data_train.gene_names)
        if test_r2 is not None:
            logger.info("\n" + "=" * 50)
            logger.info("Average Test Set R2 Scores")
            logger.info("=" * 50)
            avg_r2 = np.mean(list(test_r2.values()))
            logger.info(f"\n  Average R2: {avg_r2:.4f}")
    
    
    # create unique name for output subdir 
    base_output_dir = config.get('output_dir')
    if base_output_dir:
        os.makedirs(base_output_dir, exist_ok=True)
        run_ids = [f for f in os.listdir(base_output_dir) if f.startswith('run_')]
        run_ids = [int(f.split('_')[1]) for f in run_ids]
        run_id = 1
        if len(run_ids) > 0:
            run_id = max(run_ids) + 1
        output_dir = os.path.join(base_output_dir, f'run_{run_id}')
        
    else:
        output_dir = None
    
    config['output_dir'] = output_dir

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        # Write output directory path to a file for bash script to use
        if base_output_dir:
            os.makedirs(base_output_dir, exist_ok=True)
            output_path_file = os.path.join(base_output_dir, 'current_run_output_dir.txt')
            with open(output_path_file, 'w') as f:
                f.write(output_dir)
        logger.info(f"\nSaving results to {output_dir}...")
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        save_results(
            losses=losses,
            times=times,
            posterior_stats=posterior_stats,
            config=config, 
            annotations=list(Z.columns),
            beta_samples=beta_samples,
            mu_samples=mu_samples,
            train_r2=train_r2,
            test_r2=test_r2,
            tau_history=tau_history,
            variant_ids_G=variant_ids_G,
            variant_ids_Z=variant_ids_Z,
            gene_indices=data_train.gene_indices
        )
        logger.info("Results saved!")
    logger.info("\nDone!")


if __name__ == '__main__':
    main()