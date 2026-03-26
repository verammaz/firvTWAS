import torch
import pyro
from pyro import poutine
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO, Predictive
try:
    from pyro.optim import ClippedAdam
    HAS_CLIPPED_ADAM = True
except (ImportError, AttributeError):
    HAS_CLIPPED_ADAM = False
from tqdm import tqdm
import time
import os
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score

import utils
#import load_data
import load_data_avg
from save_outputs import save_results
from models import ParmigianoExpPerGene, simulate_expression

### this is per_gene.py
# ---------------------------------------------------------------------------
# Posterior extraction helpers
# ---------------------------------------------------------------------------

def _extract_samples(samples, data_obj, key_prefix):
    """
    Extract per-gene posterior samples for a given key_prefix ('beta' or 'mu').
    Returns dict: gene_name -> numpy array (n_posterior x n_variants).
    """
    logger = utils.get_logger()
    result = {}
    for gene_name in data_obj.gene_names:
        key = f"{key_prefix}_{gene_name}"
        if key in samples:
            arr = samples[key]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            if arr.ndim == 3:
                arr = arr.squeeze(axis=1)
            result[gene_name] = arr
        else:
            logger.warning(f"{key} not found in posterior samples.")
    return result



# ---------------------------------------------------------------------------
# R2 calculation
# ---------------------------------------------------------------------------

def calculate_r2(data, beta_samples, gene_names):
    """
    Calculate R2 using G * beta_mean directly (no model forward pass).
    Works for per-gene DataTensors where data.Y is a single tensor.
    """
    logger = utils.get_logger()
    r2_scores = {}
    for gene_name in gene_names:
        if gene_name not in beta_samples:
            logger.warning(f"No beta samples for {gene_name}, skipping R2.")
            continue
        G_gene, _, _ = data.get_gene_data(gene_name)
        beta_mean = torch.tensor(beta_samples[gene_name].mean(axis=0), device=data.device)
        if beta_mean.ndim > 1:
            beta_mean = beta_mean.squeeze()
        predictions = G_gene.matmul(beta_mean)
        y_true = data.Y.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        r2_scores[gene_name] = r2_score(y_true, y_pred)
    return r2_scores


# ---------------------------------------------------------------------------
# Main fit function
# ---------------------------------------------------------------------------

def fit_parmigiano(data, config):
    """
    Fit parmigiano_pergene model using SVI.

    INPUT:
        - data: dict with 'Train' and 'Test' DataTensors (both have tau, threshold, std set)
        - config: configuration dictionary
    OUTPUT:
        - losses, times, posterior_stats, beta_samples, mu_samples, simulated_parameters
    """
    logger = utils.get_logger()
    pyro.clear_param_store()

    if config.get('simulate', False):
        simulated_parameters = simulate_expression(data["Train"], config)
        logger.info("Finished simulating data")
    else:
        simulated_parameters = None

    model = ParmigianoExpPerGene(simulated_parameters=simulated_parameters)
    guide = AutoDiagonalNormal(model)

    clip_norm = config.get('clip_norm', 10.0)
    if HAS_CLIPPED_ADAM:
        adam = ClippedAdam({"lr": config['lr'], "clip_norm": clip_norm})
        logger.info(f"Using ClippedAdam with clip_norm={clip_norm}")
    else:
        logger.warning("ClippedAdam not available, using regular Adam.")
        adam = pyro.optim.Adam({"lr": config['lr']})

    svi = SVI(model, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()

    losses = []
    times = []

    logger.info(f"Training for {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        start = time.time()
        try:
            loss = svi.step(data["Train"], config)
        except ValueError as e:
            if "nan" in str(e).lower() or "invalid" in str(e).lower():
                logger.error(f"Numerical instability at epoch {epoch}: {e}")
                raise
            raise
        if not np.isfinite(loss):
            logger.warning(f"NaN/Inf loss at epoch {epoch}. Stopping.")
            break
        times.append(time.time() - start)
        losses.append(float(loss))
        if epoch % 50 == 0:
            logger.info(f"  Epoch {epoch}: Loss = {loss:.4f}")

    # --- Posterior samples ---
    logger.info("Generating posterior samples...")
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=config['n_posterior'])
    posterior_stats = {}

    for group in data:
        with torch.no_grad():
            samples = predictive(data[group], config)
        posterior_stats[group] = {
            k: {'mean': v.mean(0).cpu().numpy(), 'std': v.std(0).cpu().numpy()}
            for k, v in samples.items()
        }
        posterior_stats[group]["N_variants"] = len(data[group].Z)

    # Extract beta/mu from Train posterior (used for R2 and saving)
    with torch.no_grad():
        train_samples = predictive(data["Train"], config)

    beta_samples = _extract_samples(train_samples, data["Train"], "beta")
    mu_samples = _extract_samples(train_samples, data["Train"], "mu")

    # R2 on train and test
    train_r2 = calculate_r2(data["Train"], beta_samples, data["Train"].gene_names)
    test_r2 = calculate_r2(data["Test"], beta_samples, data["Test"].gene_names)
    posterior_stats["Train"]["R2"] = np.mean(list(train_r2.values())) if train_r2 else None
    posterior_stats["Test"]["R2"] = np.mean(list(test_r2.values())) if test_r2 else None

    logger.info(f"  Train R2: {posterior_stats['Train']['R2']:.4f}" if posterior_stats['Train']['R2'] is not None else "  Train R2: N/A")
    logger.info(f"  Test R2:  {posterior_stats['Test']['R2']:.4f}" if posterior_stats['Test']['R2'] is not None else "  Test R2: N/A")
    logger.info("Training complete!")

    return losses, times, posterior_stats, beta_samples, mu_samples, simulated_parameters




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = utils.parse_args()
    yaml_config = None
    if args.config:
        yaml_config = utils.load_yaml(args.config)
    config = utils.fill_defaults(args, yaml_config)

    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', None)
    logger = utils.setup_logging(log_level, log_file)

    logger.info("=" * 50)
    logger.info("PARMIGIANO - Per-Gene Bayesian Analysis")
    logger.info("=" * 50)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    all_genes = get_chr_genes(config)
    config['genes'] = all_genes.copy()

    logger.info(f"Loading phenotype and covariate data for {len(all_genes)} genes")
    X, Y, train_idx, test_idx = load_data_avg.load_residualized_covariates(config, device)

    # Load Z once to get annotation order for tau indexing
    config['genes'] = [all_genes[0]]
    _, Z_ref, _, _ = load_data_avg.load_genes(config)
    result = load_tau_T(config, device, Z_ref)
    try:
        with open(config["BR_alpha_path"], "rb") as f:
            alpha_dict = pickle.load(f)
    except:
        print("No bayesian ridge output provided -- using std = 1")
        alpha_dict = {}

    logger.info("Starting per-gene fitting loop...")
    OUTPUT = config['output_dir']

    for gene_idx, gene_name in enumerate(tqdm(all_genes)):
        try:
            config['genes'] = [gene_name]
            G, Z, variant_ids_G, variant_ids_Z = load_data_avg.load_genes(config)
            gene_output_dir = os.path.join(OUTPUT, config['chromosome'], gene_name.split("/")[1])
            if os.path.exists(os.path.join(gene_output_dir, f"{gene_name.split('/')[1]}_beta.csv.gz")):
                logger.info(f"Skipping {gene_name} — output already exists at {gene_output_dir}")
                continue

            config['output_dir'] = gene_output_dir
            data = {}
            for group_idx, (group, split_name) in enumerate(zip([train_idx, test_idx], ["Train", "Test"])):
                gene_key = gene_name.split("/")[1]
                group_data = load_data_avg.DataTensors.from_pandas(
                    G.iloc[group],
                    Z,
                    X.iloc[group],
                    Y[gene_key][group],
                    device,
                    config
                )

                if "tau" in result:
                    group_data.tau = result['tau']
                    group_data.threshold = result['threshold']
                else:
                    group_data.tau1 = result['tau1']
                    group_data.tau2 = result['tau2']  
                    group_data.threshold = 0
                group_data.std = torch.as_tensor(
                    1 / np.sqrt(alpha_dict.get(gene_key, 1)),
                    dtype=torch.float32, device=device
                )
                data[split_name] = group_data

            logger.info(f"\nGene {gene_idx+1}/{len(all_genes)}: {gene_name}")
            logger.info(f"  Samples (train/test): {data['Train'].G.shape[0]} / {data['Test'].G.shape[0]}")
            logger.info(f"  Variants: {data['Train'].G.shape[1]}")
            logger.info(f"  STD: {data['Train'].std:.4f}  |  Filter threshold: {data['Train'].threshold:.4f}")

            losses, times, posterior_stats, beta_samples, mu_samples, simulations = \
                fit_parmigiano(data, config)

            logger.info(f"  Final loss: {losses[-1]:.4f}")
            logger.info(f"  Avg time/epoch: {np.mean(times):.3f}s")

            annotations = list(Z.columns)
            os.makedirs(gene_output_dir, exist_ok=True)
            with open(os.path.join(gene_output_dir, 'config.yaml'), 'w') as f:
                yaml.dump(config, f)

            # gene_indices uses the gene key (no chr prefix) since DataTensors parses from column prefix
            gene_key_short = gene_name.split("/")[1]
            save_results(
                losses=losses,
                times=times,
                posterior_stats=posterior_stats.get("Train", {}),
                config=config,
                annotations=annotations,
                simulations=simulations,
                beta_samples=beta_samples,
                mu_samples=mu_samples,
                train_r2={gene_name: posterior_stats["Train"].get("R2")},
                test_r2={gene_name: posterior_stats["Test"].get("R2")},
                tau_history=None,
                variant_ids_G=variant_ids_G,
                variant_ids_Z=variant_ids_Z,
                gene_indices=data["Train"].gene_indices,
            )
            logger.info(f"  Results saved to {gene_output_dir}")

        except Exception as e:
            logger.warning(f"Issue with {gene_name}: {e}")
            continue


if __name__ == '__main__':
    main()