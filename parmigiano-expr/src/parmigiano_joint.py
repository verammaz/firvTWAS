import torch
import pyro
import yaml
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
from sklearn.metrics import r2_score
import numpy as np
import os
import pandas as pd

import utils
import load_data
from save_outputs import save_results
from models import (
    ParmigianoExpJoint,
    annotation_lambda,
)


# ---------------------------------------------------------------------------
# Posterior extraction helpers
# ---------------------------------------------------------------------------

def _extract_samples(samples, data_obj, key_prefix):
    """Extract per-gene posterior samples for beta or mu. Returns dict: gene_name -> numpy array."""
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


def _extract_tau_history_step(epoch, tau='tau'):
    """Try to read current tau from pyro param store. Returns numpy array or None."""
    logger = utils.get_logger()
    try:
        try:
            current_tau = pyro.param("AutoGuideList.1.tau").detach().cpu().numpy().copy()
        except KeyError:
            tau_params = {name: pyro.param(name).detach().cpu().numpy().copy()
                          for name in pyro.get_param_store().keys() if tau in name.lower()}
            if len(tau_params) == 0:
                return None
            current_tau = list(tau_params.values())[0]
        if current_tau.ndim > 1:
            current_tau = current_tau.flatten()
        return current_tau.copy()
    except Exception as e:
        if epoch == 0:
            logger.debug(f"Could not access tau parameter: {e}")
        return None


# ---------------------------------------------------------------------------
# R2 calculation
# ---------------------------------------------------------------------------

def calculate_r2(data, beta_samples, gene_names):
    """
    Calculate R2 using G * beta_mean directly (no model forward pass).
    Works for joint DataTensors where data.Y is a dict of tensors.
    """
    logger = utils.get_logger()
    r2_scores = {}
    for gene_name in gene_names:
        if gene_name not in beta_samples:
            logger.warning(f"No beta samples for {gene_name}, skipping R2.")
            continue
        G_gene, _, _ = data.get_gene_data(gene_name)
        gene_key = gene_name.split("/")[1] if "/" in gene_name else gene_name
        if gene_key in data.Y:
            Y_gene = data.Y[gene_key]
        elif gene_name in data.Y:
            Y_gene = data.Y[gene_name]
        else:
            logger.warning(f"Could not find Y for gene {gene_name}")
            continue
        beta_mean = torch.tensor(beta_samples[gene_name].mean(axis=0), device=data.device)
        if beta_mean.ndim > 1:
            beta_mean = beta_mean.squeeze()
        predictions = G_gene.matmul(beta_mean)
        r2_scores[gene_name] = r2_score(Y_gene.cpu().numpy(), predictions.cpu().numpy())
    return r2_scores

def summarize_refit_diagnostics(data_train, posterior_stats, losses, train_r2, test_r2=None):
    """Compute compact diagnostics for one refit."""
    tau2_stats = posterior_stats.get("tau2")
    tau2_mean = None if tau2_stats is None else tau2_stats.get("mean")

    if tau2_mean is not None:
        tau2 = torch.as_tensor(tau2_mean, dtype=torch.float32, device=data_train.device)
        if tau2.ndim > 1:
            tau2 = tau2.squeeze()
        lin2 = data_train.Z.matmul(tau2)
        mod = torch.exp(lin2)
        lin2_max = float(torch.max(torch.abs(lin2)).item())
        mod_median = float(torch.median(mod).item())
        mod_p99 = float(torch.quantile(mod, 0.99).item())
        mod_max = float(torch.max(mod).item())
    else:
        lin2_max = np.nan
        mod_median = np.nan
        mod_p99 = np.nan
        mod_max = np.nan

    train_vals = np.array(list(train_r2.values()), dtype=float) if train_r2 else np.array([])
    test_vals = np.array(list(test_r2.values()), dtype=float) if test_r2 else np.array([])

    return {
        "final_loss": float(losses[-1]) if losses else np.nan,
        "lin2_abs_max": lin2_max,
        "mod_median": mod_median,
        "mod_p99": mod_p99,
        "mod_max": mod_max,
        "train_avg_r2": float(np.mean(train_vals)) if train_vals.size else np.nan,
        "train_prop_r2_gt_001": float(np.mean(train_vals > 0.01)) if train_vals.size else np.nan,
        "train_prop_r2_gt_01": float(np.mean(train_vals > 0.1)) if train_vals.size else np.nan,
        "test_avg_r2": float(np.mean(test_vals)) if test_vals.size else np.nan,
        "test_prop_r2_gt_001": float(np.mean(test_vals > 0.01)) if test_vals.size else np.nan,
        "test_prop_r2_gt_01": float(np.mean(test_vals > 0.1)) if test_vals.size else np.nan,
    }


# ---------------------------------------------------------------------------
# Main fit function
# ---------------------------------------------------------------------------

def make_init_loc_fn(data, config, eps=1e-3):
    """
    Build an init_loc_fn(site) that warm-starts from BRR betas.

    - For w_g: start at 1.
    - For rho_g: start at 0.5.
    - For Z_{gene}: back out an initial Z_norm from BRR betas and approx lambda_.
    - Everything else: default to zeros.
    """
    device = data.device

    # tau1 and tau2 initialized to mean of priors
    num_anno = data.num_anno
    tau2_init = torch.zeros(num_anno, device=device) # normal(0, 1) prior
    if config.get('tau1_normal_prior', False):
        tau1_init = torch.zeros(num_anno + 1, device=device) # normal(0, 1) prior
    else:
        tau1_init = torch.ones(num_anno + 1, device=device) / (num_anno + 1) # uniform over annotations
    
    # threshold initialized to mean of Beta(2,20) prior
    threshold_alpha = config.get("threshold_prior_alpha", 2.0)
    threshold_beta = config.get("threshold_prior_beta", 20.0)
    threshold_init = torch.tensor(threshold_alpha / (threshold_alpha + threshold_beta), device=device)

    def _get_brr_beta_vector(gene_name):
        """
        Build BRR beta warm-start vector aligned to current model variant order.
        Missing BRR variants are initialized to 0.
        """
        start_idx, end_idx = data.gene_indices[gene_name]
        gene_variant_ids = data.variant_column_names[start_idx:end_idx]
        n_gene_variants = len(gene_variant_ids)

        if data.brr_betas is None or gene_name not in data.brr_betas:
            return torch.zeros(n_gene_variants, dtype=torch.float32, device=device)

        df = data.brr_betas[gene_name]
        if "beta" not in df.columns:
            return torch.zeros(n_gene_variants, dtype=torch.float32, device=device)

        # BRR index format: chr:pos_a1_a2 -> key on chr:pos
        brr_beta_by_chr_pos = {}
        for idx, row in df.iterrows():
            chr_pos = str(idx).split("_")[0]
            if chr_pos not in brr_beta_by_chr_pos:
                brr_beta_by_chr_pos[chr_pos] = float(row["beta"])

        beta_init = np.zeros(n_gene_variants, dtype=np.float32)
        for i, variant_id in enumerate(gene_variant_ids):
            # Model variant format: GENE_chr:pos
            parts = variant_id.split("_", 1)
            if len(parts) != 2:
                continue
            chr_pos = parts[1]
            if chr_pos in brr_beta_by_chr_pos:
                beta_init[i] = brr_beta_by_chr_pos[chr_pos]

        beta = torch.as_tensor(beta_init, dtype=torch.float32, device=device)
        return beta

    def init_loc_fn(site):
        name = site["name"]
        fn = site["fn"]

        # Only care about unobserved sample sites
        if site["type"] != "sample" or site.get("is_observed", False):
            return None

        # Global gene weights
        if name == "w_g":
            return torch.ones(data.num_genes, device=device)

        # Mixing parameter over mean vs variance component
        if name == "rho_g":
            return torch.full((data.num_genes,), 0.5, device=device)

        # Per-gene latent Normal vector (this is where we want BRR warm start)
        if name.startswith("Z_"):
            gene_name = name[2:]  # strip "Z_"

            beta_brr = _get_brr_beta_vector(gene_name)

            # Get per-gene annotations and weights
            G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)

            # Match generative λ 
            lambda_approx = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                threshold_init,
                tau1=tau1_init,
                tau2=tau2_init,
                lin2_clip=config.get("lin2_clip", None),
            )

            # Make sure shapes match; if not, just fall back
            if beta_brr.shape[0] != lambda_approx.shape[0]:
                return torch.zeros(fn.event_shape, device=device)

            # For warm start, approximate: beta ≈ lambda_ * Z_norm
            # => Z_norm_init ≈ beta_brr / lambda_approx
            z_init = beta_brr / (lambda_approx + eps)

            # Ensure we return the right shape
            # (fn.event_shape should be (num_variants_for_gene,))
            if z_init.shape != fn.event_shape:
                z_init = z_init.view(fn.event_shape)

            return z_init

        # Default: prior mean 0 for anything else
        return torch.zeros(fn.event_shape, device=device)

    return init_loc_fn

def setup_model(data_train, config):
    logger = utils.get_logger()
    to_optimize = []
    model = ParmigianoExpJoint()
    to_optimize.append('tau1')
    to_optimize.append('tau2')    
    to_optimize.append("threshold")

    guide = AutoGuideList(model)

    # ------------------------------------------------------------------
    # Decide whether there are any non-(tau/threshold) latents to put
    # under AutoDiagonalNormal. 
    # ------------------------------------------------------------------
    blocked_model = poutine.block(model, hide=to_optimize)
    trace = poutine.trace(blocked_model).get_trace(data_train, config)
    has_latents = any(
        (site["type"] == "sample") and (not site.get("is_observed", False))
        for site in trace.nodes.values()
    )

    # Decide whether to use BRR-based warm start (requires BRR betas)
    use_brr_init = (data_train.brr_betas is not None)

    if has_latents:
        if use_brr_init:
            init_loc_fn = make_init_loc_fn(data_train, config)
            guide.add(AutoDiagonalNormal(blocked_model, init_loc_fn=init_loc_fn))
        else:
            guide.add(AutoDiagonalNormal(blocked_model))
    else:
        logger.warning(
            "No latent variables found for AutoDiagonalNormal (after hiding tau/threshold); "
            "using AutoDelta-only guide for tau/threshold."
        )

    # Always put tau/threshold (if present) under AutoDelta.
    guide.add(AutoDelta(poutine.block(model, expose=to_optimize)))

    clip_norm = config.get('clip_norm', 10.0)
    if HAS_CLIPPED_ADAM:
        adam = ClippedAdam({"lr": config['lr'], "clip_norm": clip_norm})
        logger.info(f"Using ClippedAdam with clip_norm={clip_norm}")
    else:
        logger.warning("ClippedAdam not available, using regular Adam.")
        adam = pyro.optim.Adam({"lr": config['lr']})

    svi = SVI(model, guide=guide, optim=adam, loss=Trace_ELBO())

    return model, guide,svi

def fit_parmigiano(data_train, config):
    """
    Fit parmigiano_expression (joint) model using SVI.

    INPUT:
        - data_train: DataTensors for training
        - config: configuration dictionary
    OUTPUT:
        - losses, times, posterior_stats, beta_samples, mu_samples, tau_history
    """
    logger = utils.get_logger()
    model, guide, svi = setup_model(data_train, config)
    
    pyro.clear_param_store()

    losses = []
    times = []
    tau_history = []

    logger.info(f"Training for {config['epochs']} epochs...")
    for epoch in tqdm(range(config['epochs'])):
        start = time.time()
        try:
            loss = svi.step(data_train, config)
        except ValueError as e:
            if "nan" in str(e).lower() or "invalid" in str(e).lower():
                logger.error(f"Numerical instability at epoch {epoch}: {e}")
                logger.error("Try: reducing lr, adding clip_norm to config, or checking input data.")
                raise
            raise
        if not np.isfinite(loss):
            logger.warning(f"NaN/Inf loss at epoch {epoch}. Stopping.")
            break
        times.append(time.time() - start)
        losses.append(float(loss))
        tau_history.append((_extract_tau_history_step(epoch, 'tau1'), _extract_tau_history_step(epoch, 'tau2')))
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}: Loss = {loss:.4f}")

    # --- Posterior samples ---
    logger.info("Generating posterior samples...")
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=config['n_posterior'])
    with torch.no_grad():
        samples = predictive(data_train, config)

    posterior_stats = {
        k: {'mean': v.mean(0).cpu().numpy(), 'std': v.std(0).cpu().numpy()}
        for k, v in samples.items()
    }
    beta_samples = _extract_samples(samples, data_train, "beta")
    mu_samples = _extract_samples(samples, data_train, "mu")
    sigma_samples = _extract_samples(samples, data_train, "sigma")

    logger.info("Training complete!")
    return losses, times, posterior_stats, beta_samples, mu_samples, sigma_samples, tau_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = utils.parse_args()
    yaml_config = None
    if args.config:
        yaml_config = utils.load_yaml(args.config)
    config = utils.fill_defaults(args, yaml_config)

    config.pop('chromosome', None)
    config.pop('pergene_output_dir', None)

    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', None)
    logger = utils.setup_logging(log_level, log_file)

    logger.info("=" * 50)
    logger.info("PARMIGIANO - Joint Bayesian Gene Analysis")
    logger.info("=" * 50)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    OUTPUT_DIR = config.get('joint_output_dir', None)
    if not OUTPUT_DIR:
        raise ValueError("'joint_output_dir' must be set in config.")
    
    # Parse gene list
    if isinstance(config.get('gene_list'), str):
        if os.path.exists(config['gene_list']):
            with open(config['gene_list'], 'r') as f:
                config['genes'] = [line.strip() for line in f if line.strip()]
        else:
            config['genes'] = [g.strip() for g in config['gene_list'].split(',') if g.strip()]
    else:
        config['genes'] = config.get('gene_list', [])

    # Load data
    logger.info("Loading phenotype and covariate data...")
    X, Y, train_idx, test_idx = load_data.load_residualized_covariates(config, device)

    logger.info("Loading genotype and annotation data...")
    G, Z, variant_ids_G, variant_ids_Z = load_data.load_genes(config)

    # Load Bayesian Ridge Regression results
    if config.get('brr_results_dir', None) is not None:
        logger.info("Loading Bayesian Ridge Regression results...")
        brr_results = load_data.load_brr_results(config)
    else:
        brr_results = None

    # Load Bayesian Ridge Regression betas and alphas
    if brr_results is not None:
        brr_betas = brr_results.get('betas', None)
        brr_alphas = brr_results.get('alphas', None)
    else:
        brr_betas = None
        brr_alphas = None

    # Build train/test DataTensors
    if config.get('train_test', False):
        logger.info("Applying train/test split...")
        train_sample_ids = X.iloc[train_idx].index
        test_sample_ids = X.iloc[test_idx].index

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        Y_train = {gene: expr[train_idx] for gene, expr in Y.items()}
        Y_test = {gene: expr[test_idx] for gene, expr in Y.items()}

        G_train = G.loc[train_sample_ids]
        G_test = G.loc[test_sample_ids]

        # Train tensors first so MAF-based common-variant filtering (if enabled) defines
        # variant_column_names; test must reuse the same columns.
        logger.info("Creating train data tensors...")
        data_train = load_data.DataTensors.from_pandas(
            G_train,
            Z,
            X_train,
            Y_train,
            brr_betas,
            brr_alphas,
            device,
            config,
            train_G_for_maf_filter=G_train,
        )
        logger.info("Creating test data tensors...")
        data_test = load_data.DataTensors.from_pandas(
            G_test,
            Z,
            X_test,
            Y_test,
            brr_betas,
            brr_alphas,
            device,
            config,
            train_G_for_maf_filter=G_train,
            forced_variant_columns=data_train.variant_column_names,
        )
    else:
        data_test = None
        X_train, Y_train, G_train = X, Y, G
        logger.info("Creating train data tensors...")
        data_train = load_data.DataTensors.from_pandas(
            G_train,
            Z,
            X_train,
            Y_train,
            brr_betas,
            brr_alphas,
            device,
            config,
            train_G_for_maf_filter=G_train,
        )

    logger.info(f"\nData summary:")
    logger.info(f"  Train samples: {data_train.G.shape[0]}" +
                (f"  |  Test samples: {data_test.G.shape[0]}" if data_test else ""))
    logger.info(f"  Variants: {data_train.G.shape[1]}")
    logger.info(f"  Genes: {data_train.num_genes}")
    logger.info(f"  Annotations: {data_train.num_anno}")
    logger.info(f"  Covariates: {data_train.num_cov}\n")

    if data_train.num_genes < 5:
        logger.warning(f"Only {data_train.num_genes} gene(s) detected - may cause instability.")
        original_lr = config['lr']
        config['lr'] = config['lr'] * 0.1
        logger.warning(f"LR reduced: {original_lr} -> {config['lr']}")

    config['data_summary'] = {
        'samples_train': data_train.G.shape[0],
        'samples_test': data_test.G.shape[0] if data_test else None,
        'variants': data_train.G.shape[1],
        'genes': data_train.num_genes,
        'annotations': data_train.num_anno,
        'covariates': data_train.num_cov,
    }
    
    # all runs share config, save config to joint_output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs = []
    refit_summaries = []
    for run in range(config['refits']): # each run in child folder
        print("\nREFIT #",run+1)
        # Fit
        losses, times, posterior_stats, beta_samples, mu_samples, sigma_samples, tau_history = fit_parmigiano(data_train, config)
    
        logger.info(f"\nFinal loss: {losses[-1]:.4f}")
        logger.info(f"Avg time/epoch: {np.mean(times):.2f}s  |  Total: {np.sum(times):.2f}s")
    
        # R2
        logger.info("Calculating R2 on train set...")
        train_r2 = calculate_r2(data_train, beta_samples, data_train.gene_names)
        logger.info(f"  Avg Train R2: {np.mean(list(train_r2.values())):.4f}")
    
        test_r2 = None
        if data_test is not None:
            logger.info("Calculating R2 on test set...")
            test_r2 = calculate_r2(data_test, beta_samples, data_train.gene_names)
            logger.info(f"  Avg Test R2: {np.mean(list(test_r2.values())):.4f}")

        summary = summarize_refit_diagnostics(
            data_train=data_train,
            posterior_stats=posterior_stats,
            losses=losses,
            train_r2=train_r2,
            test_r2=test_r2,
        )
        summary["refit"] = run + 1
        refit_summaries.append(summary)
    
        # Output directory (auto-increment run_N)
        run_ids = [int(f.split('_')[1]) for f in os.listdir(OUTPUT_DIR)
                    if f.startswith('run_') and f.split('_')[1].isdigit()]
        run_id = max(run_ids) + 1 if run_ids else 1
        runs.append(run_id)
        output_dir = os.path.join(OUTPUT_DIR, f'run_{run_id}')
       
        save_results(
                output_dir=output_dir,
                losses=losses,
                times=times,
                posterior_stats=posterior_stats,
                annotations=list(Z.columns),
                data = data_train,
                beta_samples=beta_samples,
                mu_samples=mu_samples,
                train_r2=train_r2,
                test_r2=test_r2,
                tau_history=tau_history,
                variant_ids_G=variant_ids_G,
                variant_ids_Z=variant_ids_Z,
                gene_indices=data_train.gene_indices,
        )
        logger.info(f"Results saved to {output_dir}")

    if refit_summaries:
        logger.info("\nRefit summary diagnostics:")
        header = (
            "refit | final_loss | lin2_abs_max | mod_median | mod_p99 | mod_max | "
            "train_avg_r2 | train_prop>0.01 | train_prop>0.1 | "
            "test_avg_r2 | test_prop>0.01 | test_prop>0.1"
        )
        logger.info(header)
        for s in refit_summaries:
            logger.info(
                f"{s['refit']:>5d} | "
                f"{s['final_loss']:.4f} | {s['lin2_abs_max']:.4f} | "
                f"{s['mod_median']:.4f} | {s['mod_p99']:.4f} | {s['mod_max']:.4f} | "
                f"{s['train_avg_r2']:.4f} | {s['train_prop_r2_gt_001']:.4f} | {s['train_prop_r2_gt_01']:.4f} | "
                f"{s['test_avg_r2']:.4f} | {s['test_prop_r2_gt_001']:.4f} | {s['test_prop_r2_gt_01']:.4f}"
            )

    logger.info("\nDone with all refits!")

    # save config to output_dir
    config['refits'] = runs
    with open(os.path.join(OUTPUT_DIR, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Config saved to {os.path.join(OUTPUT_DIR, 'config.yaml')}")

    
if __name__ == '__main__':
    main()