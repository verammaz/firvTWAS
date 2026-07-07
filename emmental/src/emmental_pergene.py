import glob
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyro
import torch
import yaml
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from sklearn.metrics import r2_score
from tqdm import tqdm

try:
    from pyro.optim import ClippedAdam
    HAS_CLIPPED_ADAM = True
except (ImportError, AttributeError):
    HAS_CLIPPED_ADAM = False

import load_data
import utils
from models import EmmentalPerGene, annotation_lambda, simulate_expression
from save_outputs import save_results


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


def _y_vec_for_gene(data, gene_name):
    gk = gene_name.split("/")[-1] if "/" in gene_name else gene_name
    if isinstance(data.Y, dict):
        return data.Y[gk]
    return data.Y


def calculate_r2(data, beta_samples, gene_names):
    """
    Calculate R2 using G * beta_mean directly (no model forward pass).
    Works for per-gene DataTensors where ``data.Y`` is a tensor or a one-gene dict.
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
        y_true = _y_vec_for_gene(data, gene_name).cpu().numpy()
        y_pred = predictions.cpu().numpy()
        r2_scores[gene_name] = r2_score(y_true, y_pred)
    return r2_scores


# ---------------------------------------------------------------------------
# Config from joint directory
# ---------------------------------------------------------------------------


def merge_config_from_joint_dir(config, joint_dir):
    """
    Overlay model/data keys from ``config.yaml`` under a joint experiment root
    so per-gene matches the joint run. Per-gene-only keys (``output_dir``, ``gene_list``, ``lr``,
    etc.) are left to the user unless absent (then filled from joint).

    Also sets ``joint_output_dir`` so τ and T load from that tree.
    """
    joint_dir = os.path.abspath(joint_dir)
    cfg_path = os.path.join(joint_dir, "config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            joint_cfg = yaml.safe_load(f) or {}
        keys_to_copy = [
            "tau1_normal_prior",
            "maf_beta",
            "maf_threshold",
            "annotations",
            "annotation_dir",
            "genotype_dir",
            "expression_path",
            "covariates_path",
            "brr_results_dir",
            "joint_output_dir",
            "no_wg",
            "no_rhog",
            "collapsed_model",
            "no_T",
            "normalize_G",
            "threshold_prior_alpha",
            "threshold_prior_beta",
        ]
        skip_overlay = {
            "pergene_output_dir",
            "gene_list",
            "genes",
            "epochs",
            "lr",
            "n_posterior",
            "log_level",
            "log_file",
            "clip_norm",
            # Per-gene-only overrides; maf_threshold / brr_results_dir come from joint/config.yaml
            # (same values as the config file used for the joint run).
        }
        for k in keys_to_copy:
            if k in skip_overlay:
                continue
            if k in joint_cfg and joint_cfg[k] is not None:
                config[k] = joint_cfg[k]
        if joint_cfg.get("refits") is not None:
            config["_joint_refits"] = joint_cfg["refits"]
    return config


def _write_pergene_root_config(
    pergene_root: str,
    config: dict,
    *,
    joint_run_ids: list = None,
) -> None:
    """Single ``pergene/config.yaml`` (chromosome omitted; same as joint layout)."""
    os.makedirs(pergene_root, exist_ok=True)
    config_save = utils.config_for_yaml_save(
        {
            k: v
            for k, v in config.items()
            if k not in ("chromosome", "genes", "gene_list", "_joint_refits")
        }
    )
    if joint_run_ids is not None:
        config_save["refits"] = [int(x) for x in joint_run_ids]
    path = os.path.join(pergene_root, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config_save, f)


def summarize_pergene_refit(
    losses_all: list,
    train_r2_all: dict,
    test_r2_all: dict,
    refit_number: int,
) -> dict:
    """Compact diagnostics for one per-gene refit (all genes on this chromosome)."""
    train_vals = list(train_r2_all.values()) if train_r2_all else []
    test_vals = list(test_r2_all.values()) if test_r2_all else []
    return {
        "refit": refit_number,
        "n_genes_fit": len(train_r2_all),
        "final_loss_mean": float(np.mean(losses_all[-100:])) if losses_all else float("nan"),
        "train_avg_r2": float(np.mean(train_vals)) if train_vals else float("nan"),
        "train_prop_r2_gt_001": float(np.mean([r > 0.01 for r in train_vals])) if train_vals else float("nan"),
        "train_prop_r2_gt_01": float(np.mean([r > 0.1 for r in train_vals])) if train_vals else float("nan"),
        "test_avg_r2": float(np.mean(test_vals)) if test_vals else float("nan"),
        "test_prop_r2_gt_001": float(np.mean([r > 0.01 for r in test_vals])) if test_vals else float("nan"),
        "test_prop_r2_gt_01": float(np.mean([r > 0.1 for r in test_vals])) if test_vals else float("nan"),
    }


def resolve_gene_list(config):
    """Genes from ``gene_list`` file, inline list, or chromosome scan."""
    gl = config.get("gene_list")
    if isinstance(gl, str) and os.path.isfile(gl):
        with open(gl, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    if isinstance(gl, list) and gl:
        return gl
    return load_data.get_chr_genes(config)


def _log_run_timing(logger, started_wall: datetime, started_secs: float, *, status: str) -> None:
    """Log wall-clock start/end plus total elapsed runtime."""
    ended_wall = datetime.now().astimezone()
    elapsed = max(0.0, time.time() - started_secs)
    logger.info(f"Run {status} at: {ended_wall.strftime('%Y-%m-%d %H:%M:%S %z')}")
    logger.info(f"Run started at:  {started_wall.strftime('%Y-%m-%d %H:%M:%S %z')}")
    logger.info(f"Total runtime:   {elapsed / 60.0:.2f} minutes ({elapsed:.1f} sec)")


# ---------------------------------------------------------------------------
# Main fit function
# ---------------------------------------------------------------------------

def make_init_loc_fn(data, config=None, eps=1e-3):
    """
    Build an init_loc_fn(site) for per-gene warm start using BRR betas.

    - w_g -> 1 (or 0 if config init_wg_zero)
    - rho_g -> 0.5
    - Z_{gene} -> BRR-aligned initialization with zeros for missing variants
    - everything else -> 0
    """
    device = data.device
    cfg = config or {}
    wg_init = 0.0 if cfg.get("init_wg_zero", False) else 1.0

    def _get_brr_beta_vector(gene_name):
        start_idx, end_idx = data.gene_indices[gene_name]
        gene_variant_ids = data.variant_column_names[start_idx:end_idx]
        n_gene_variants = len(gene_variant_ids)

        if data.brr_betas is None or gene_name not in data.brr_betas:
            return torch.zeros(n_gene_variants, dtype=torch.float32, device=device)

        df = data.brr_betas[gene_name]
        if "beta" not in df.columns:
            return torch.zeros(n_gene_variants, dtype=torch.float32, device=device)

        brr_beta_by_chr_pos = {}
        for idx, row in df.iterrows():
            chr_pos = str(idx).split("_")[0]
            if chr_pos not in brr_beta_by_chr_pos:
                brr_beta_by_chr_pos[chr_pos] = float(row["beta"])

        beta_init = np.zeros(n_gene_variants, dtype=np.float32)
        for i, variant_id in enumerate(gene_variant_ids):
            parts = variant_id.split("_", 1)  # GENE_chr:pos
            if len(parts) != 2:
                continue
            chr_pos = parts[1]
            if chr_pos in brr_beta_by_chr_pos:
                beta_init[i] = brr_beta_by_chr_pos[chr_pos]
        return torch.as_tensor(beta_init, dtype=torch.float32, device=device)

    def init_loc_fn(site):
        name = site["name"]
        fn = site["fn"]

        if site["type"] != "sample" or site.get("is_observed", False):
            return None

        if name == "w_g":
            return torch.full((data.num_genes,), wg_init, device=device)
        if name == "rho_g":
            return torch.full((data.num_genes,), 0.5, device=device)

        if name.startswith("Z_"):
            gene_name = name[2:]
            beta_brr = _get_brr_beta_vector(gene_name)
            _, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
            lambda_approx = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                data.threshold,
                tau1=data.tau1,
                tau2=data.tau2,
            )
            if beta_brr.shape[0] != lambda_approx.shape[0]:
                return torch.zeros(fn.event_shape, device=device)
            z_init = beta_brr / (lambda_approx + eps)
            if z_init.shape != fn.event_shape:
                z_init = z_init.view(fn.event_shape)
            return z_init

        return torch.zeros(fn.event_shape, device=device)

    return init_loc_fn


def fit_emmental(data, config, simulated_parameters=None):
    """
    Fit per-gene Emmental with fixed global τ and T from joint outputs.

    INPUT:
        - data: Train DataTensors (BRR alphas set on data for obs. noise)
        - config: configuration dictionary
        - simulated_parameters: optional ground-truth dict from simulate_expression
    OUTPUT:
        - losses, times, posterior_stats, beta_samples, mu_samples, simulated_parameters
    """
    logger = utils.get_logger()
    if config.get("collapsed_model", False):
        from emmental_collapsed_fit import fit_pergene_collapsed
        from joint_guide_setup import pergene_wg_rhog_fully_ablated

        logger.info("Using collapsed per-gene model (beta integrated out).")
        if pergene_wg_rhog_fully_ablated(config):
            logger.info(
                "no_wg + no_rhog: deterministic collapsed beta (no SVI on w_g/rho_g)."
            )
        init_loc_fn = (
            make_init_loc_fn(data, config)
            if simulated_parameters is None and data.brr_betas is not None
            else None
        )
        return fit_pergene_collapsed(
            data, config, init_loc_fn=init_loc_fn, simulated_parameters=simulated_parameters
        )

    pyro.clear_param_store()

    model = EmmentalPerGene(config, data, simulated_parameters=simulated_parameters)
    use_brr_init = data.brr_betas is not None
    if use_brr_init:
        guide = AutoDiagonalNormal(model, init_loc_fn=make_init_loc_fn(data, config))
    else:
        guide = AutoDiagonalNormal(model)

    clip_norm = config.get("clip_norm", 10.0)
    if HAS_CLIPPED_ADAM:
        adam = ClippedAdam({"lr": config["lr"], "clip_norm": clip_norm})
        logger.info(f"Using ClippedAdam with clip_norm={clip_norm}")
    else:
        logger.warning("ClippedAdam not available; using Adam.")
        adam = Adam({"lr": config["lr"]})

    svi = SVI(model, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()

    losses = []
    times = []

    logger.info(f"Training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        start = time.time()
        try:
            loss = svi.step(data, config)
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

    logger.info("Generating posterior samples...")
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=config["n_posterior"])
    with torch.no_grad():
        samples = predictive(data, config)
    posterior_stats = {
        k: {'mean': v.mean(0).cpu().numpy(), 'std': v.std(0).cpu().numpy()}
        for k, v in samples.items()
    }
    beta_samples = _extract_samples(samples, data, "beta")
    mean_samples = _extract_samples(samples, data, "mean")

    logger.info("Training complete!")

    return losses, times, posterior_stats, beta_samples, mean_samples


def fit_chromosome_genes(
    chr_genes,
    config,
    X,
    Y,
    train_idx,
    test_idx,
    train_sample_ids,
    test_sample_ids,
    brr_betas,
    brr_alphas,
    tau1,
    tau2,
    th,
    device,
    logger,
):
    """
    Fit every gene on the chromosome once (single refit).

    Returns aggregated artifacts for ``save_results``.
    """
    losses_all, times_all = [], []
    history_rows = []
    posterior_stats_all = {}
    beta_samples_all = {}
    mean_samples_all = {}
    train_r2_all = {}
    test_r2_all = {}
    variant_ids_G_all = []
    variant_ids_Z_all = []
    gene_indices_all = {}

    for gene_idx, gene_name in enumerate(tqdm(chr_genes)):
        try:
            config["genes"] = [gene_name]
            G, Z, variant_ids_G, variant_ids_Z = load_data.load_genes(config)
            G, Z, variant_ids_G, variant_ids_Z, maf_weights = (
                load_data.prepare_genotypes_for_training(
                    G, Z, variant_ids_G, variant_ids_Z, train_sample_ids, config, device
                )
            )
            gene_key = gene_name.split("/")[-1] if "/" in gene_name else gene_name

            data = {}
            train_variant_columns = None
            for group_idx, sample_ids, split_name in zip(
                [train_idx, test_idx],
                [train_sample_ids, test_sample_ids],
                ["Train", "Test"],
            ):
                y_vec = Y[gene_key][group_idx]
                y_dict = {gene_key: y_vec}
                gene_brr_beta = None if brr_betas is None else brr_betas.get(gene_key)
                gene_brr_alpha = None if brr_alphas is None else brr_alphas.get(gene_key)
                brr_betas_arg = None if gene_brr_beta is None else {gene_key: gene_brr_beta}
                brr_alphas_arg = None if gene_brr_alpha is None else {gene_key: gene_brr_alpha}
                group_data = load_data.DataTensors.from_pandas(
                    G.loc[sample_ids],
                    Z,
                    X.loc[sample_ids],
                    y_dict,
                    brr_betas_arg,
                    brr_alphas_arg,
                    device,
                    config,
                    forced_variant_columns=train_variant_columns,
                    maf_weights_precomputed=maf_weights,
                    variants_pre_filtered=True,
                )
                if split_name == "Train":
                    train_variant_columns = group_data.variant_column_names
                data[split_name] = group_data
                data[split_name].tau1 = torch.as_tensor(tau1, dtype=torch.float32, device=device)
                data[split_name].tau2 = torch.as_tensor(tau2, dtype=torch.float32, device=device)
                data[split_name].threshold = torch.as_tensor(th, dtype=torch.float32, device=device)

            logger.info(f"\nGene {gene_idx + 1}/{len(chr_genes)}: {gene_name}")
            logger.info(f"  Samples (train/test): {data['Train'].G.shape[0]} / {data['Test'].G.shape[0]}")
            logger.info(f"  Variants: {data['Train'].G.shape[1]}")
            th_gene = float(data["Train"].threshold.item()) if hasattr(data["Train"].threshold, "item") else float(data["Train"].threshold)
            alpha_val = None if brr_alphas is None else brr_alphas.get(gene_key)
            if alpha_val is None:
                logger.info(f"  STD (1/sqrt(alpha)): N/A (missing BRR alpha)  |  Threshold T: {th_gene:.4f}")
            else:
                logger.info(f"  STD (1/sqrt(alpha)): {1.0 / np.sqrt(alpha_val):.4f}  |  Threshold T: {th_gene:.4f}")

            losses, times, posterior_stats, beta_samples, mean_samples = fit_emmental(data["Train"], config)

            logger.info(f"  Final loss: {losses[-1]:.4f}")
            logger.info(f"  Avg time/epoch: {np.mean(times):.3f}s")

            logger.info("Calculating R2 on train set...")
            train_r2 = calculate_r2(data["Train"], beta_samples, data["Train"].gene_names)
            logger.info(f" Train R2: {np.mean(list(train_r2.values())):.4f}")

            logger.info("Calculating R2 on test set...")
            test_r2 = calculate_r2(data["Test"], beta_samples, data["Test"].gene_names)
            logger.info(f" Test R2: {np.mean(list(test_r2.values())):.4f}")

            losses_all.extend(losses)
            times_all.extend(times)
            history_rows.extend(
                {
                    "gene": gene_name,
                    "gene_id": gene_key,
                    "epoch": epoch_idx,
                    "loss": float(loss_val),
                    "time_sec": float(time_val),
                }
                for epoch_idx, (loss_val, time_val) in enumerate(zip(losses, times))
            )
            for stat_name, stat_values in posterior_stats.items():
                posterior_stats_all[f"{gene_key}::{stat_name}"] = stat_values
            beta_samples_all.update(beta_samples)
            mean_samples_all.update(mean_samples)
            train_r2_all.update(train_r2)
            test_r2_all.update(test_r2)

            train_dt = data["Train"]
            for saved_gene in sorted(set(beta_samples.keys())):
                if saved_gene in gene_indices_all:
                    continue
                if saved_gene not in train_dt.gene_indices:
                    logger.warning(
                        f"Gene {saved_gene} missing from train_dt.gene_indices; "
                        "skipping variant ID registration for beta outputs."
                    )
                    continue
                start_idx, end_idx = train_dt.gene_indices[saved_gene]
                variant_cols = train_dt.variant_column_names[start_idx:end_idx]
                global_start = len(variant_ids_G_all)
                global_end = global_start + len(variant_cols)
                gene_indices_all[saved_gene] = (global_start, global_end)
                variant_ids_G_all.extend(variant_cols)
                variant_ids_Z_all.extend(variant_cols)

        except Exception as e:
            logger.warning(f"Issue with {gene_name}: {e}")
            continue

    return {
        "losses_all": losses_all,
        "times_all": times_all,
        "history_rows": history_rows,
        "posterior_stats_all": posterior_stats_all,
        "beta_samples_all": beta_samples_all,
        "mean_samples_all": mean_samples_all,
        "train_r2_all": train_r2_all,
        "test_r2_all": test_r2_all,
        "variant_ids_G_all": variant_ids_G_all,
        "variant_ids_Z_all": variant_ids_Z_all,
        "gene_indices_all": gene_indices_all,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = utils.parse_args()
    yaml_config = utils.load_yaml(args.config) if args.config else None
    config = utils.fill_defaults(args, yaml_config)
    use_refit = bool(config.get("refit", False))
    logger = utils.get_logger()

    if config.get("joint_output_dir"):
        if not os.path.exists(config["joint_output_dir"]):
            raise ValueError(f"Joint output directory {config['joint_output_dir']} does not exist.")
        merge_config_from_joint_dir(config, config["joint_output_dir"])
    else:
        raise ValueError("Provide --joint_output_dir to the emmental_joint output root.")

    if config.get("chromosome", None) is None:
        raise ValueError("Chromosome not specified in config")
    config["chromosome"] = str(config["chromosome"])

    PERGENE_ROOT = config.get("pergene_output_dir", None)
    if not PERGENE_ROOT:
        logger.info(
            "No pergene output directory provided, using sibling of joint output directory: %s",
            config["joint_output_dir"],
        )
        PERGENE_ROOT = os.path.join(os.path.dirname(config["joint_output_dir"]), "pergene")
        config["pergene_output_dir"] = PERGENE_ROOT
    PERGENE_ROOT = os.path.abspath(PERGENE_ROOT)
    os.makedirs(PERGENE_ROOT, exist_ok=True)

    chr_tag = f"chr{config['chromosome']}"
    joint_dir = os.path.abspath(config["joint_output_dir"])

    if use_refit:
        joint_cfg_for_runs = dict(config)
        if config.get("_joint_refits") is not None:
            joint_cfg_for_runs["refits"] = config["_joint_refits"]
        joint_run_ids = load_data.resolve_joint_refit_run_ids(joint_dir, joint_cfg_for_runs)
        logger.info("Refit mode: %d per-gene refit(s) aligned to joint run ids %s", len(joint_run_ids), joint_run_ids)
    else:
        joint_run_ids = None
        logger.info("Single-fit mode: using averaged joint tau/T")

    log_level = config.get("log_level", "INFO")
    log_file = config.get("log_file", None)
    logger = utils.setup_logging(log_level, log_file)
    run_started_wall = datetime.now().astimezone()
    run_started_secs = time.time()
    logger.info(f"Run started at: {run_started_wall.strftime('%Y-%m-%d %H:%M:%S %z')}")

    config.pop('gene_list', None)
    logger.info("=" * 50)
    logger.info(
        "EMMENTAL - Per-Gene (%s)",
        "refits matched to joint run_*" if use_refit else "single fit, averaged joint tau/T",
    )
    logger.info("Pergene root: %s  |  Chromosome: %s", PERGENE_ROOT, chr_tag)
    logger.info("=" * 50)
    for key, value in sorted(config.items()):
        logger.info(f"  {key}: {value}")
    if config.get("collapsed_model", False):
        logger.info("Inference: collapsed model (beta integrated out per gene)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get genes for chromosome
    chr_genes = load_data.get_chr_genes(config)
    config['genes'] = chr_genes # used by data loading functions

    # Load data
    logger.info("Loading phenotype and covariate data...")
    X, Y, train_idx, test_idx = load_data.load_residualized_covariates(config, device)

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

    train_sample_ids = X.index[train_idx]
    test_sample_ids = X.index[test_idx]

    refit_summaries = []
    all_history_rows = []
    any_saved = False

    refit_plan = (
        [(rid, os.path.join(PERGENE_ROOT, f"run_{rid}", chr_tag)) for rid in joint_run_ids]
        if use_refit
        else [(None, os.path.join(PERGENE_ROOT, chr_tag))]
    )

    for refit_idx, (joint_run_id, output_dir) in enumerate(refit_plan):
        if use_refit:
            logger.info(
                f"\n{'=' * 50}\nREFIT {refit_idx + 1}/{len(refit_plan)} "
                f"(joint run_{joint_run_id} -> {output_dir})\n{'=' * 50}"
            )
            mean_df, th, tau1, tau2 = load_data.load_tau_threshold_from_joint_run(
                joint_dir, joint_run_id
            )
        else:
            logger.info(f"\n{'=' * 50}\nSINGLE FIT -> {output_dir}\n{'=' * 50}")
            mean_df, th, tau1, tau2 = load_data.load_tau_threshold(config)

        os.makedirs(output_dir, exist_ok=True)
        mean_df.to_csv(os.path.join(output_dir, "tau_T.csv"), index=False)
        logger.info(f"Wrote tau/T to {os.path.join(output_dir, 'tau_T.csv')}")

        fit_out = fit_chromosome_genes(
            chr_genes,
            config,
            X,
            Y,
            train_idx,
            test_idx,
            train_sample_ids,
            test_sample_ids,
            brr_betas,
            brr_alphas,
            tau1,
            tau2,
            th,
            device,
            logger,
        )

        if not fit_out["beta_samples_all"] and not fit_out["mean_samples_all"]:
            logger.warning("No successful gene fits for this pass; skipping save.")
            continue

        save_results(
            output_dir=output_dir,
            losses=np.asarray(fit_out["losses_all"], dtype=float),
            times=np.asarray(fit_out["times_all"], dtype=float),
            posterior_stats=fit_out["posterior_stats_all"],
            annotations=config.get("annotations", []),
            beta_samples=fit_out["beta_samples_all"],
            mu_samples=None,
            mean_samples=fit_out["mean_samples_all"],
            train_r2=fit_out["train_r2_all"],
            test_r2=fit_out["test_r2_all"],
            variant_ids_G=fit_out["variant_ids_G_all"],
            variant_ids_Z=fit_out["variant_ids_Z_all"],
            gene_indices=fit_out["gene_indices_all"],
            mean_sample_ids=list(train_sample_ids),
        )
        logger.info(f"Saved outputs to {output_dir}")
        any_saved = True
        all_history_rows.extend(fit_out["history_rows"])

        if use_refit:
            summary = summarize_pergene_refit(
                fit_out["losses_all"],
                fit_out["train_r2_all"],
                fit_out["test_r2_all"],
                refit_idx + 1,
            )
            refit_summaries.append(summary)

    if not any_saved:
        logger.warning("No per-gene fits completed successfully; nothing to save.")
        _log_run_timing(logger, run_started_wall, run_started_secs, status="ended (no successful fits)")
        return

    if use_refit:
        history_path = os.path.join(PERGENE_ROOT, f"loss_time_by_gene_{chr_tag}.csv")
    else:
        history_path = os.path.join(PERGENE_ROOT, chr_tag, "loss_time_by_gene.csv")
    if all_history_rows:
        pd.DataFrame(all_history_rows).to_csv(history_path, index=False)
        logger.info(f"Saved per-gene training history to {history_path}")

    if refit_summaries:
        logger.info("\nRefit summary diagnostics:")
        header = (
            "refit | n_genes | final_loss_mean | train_avg_r2 | train_prop>0.01 | train_prop>0.1 | "
            "test_avg_r2 | test_prop>0.01 | test_prop>0.1"
        )
        logger.info(header)
        for s in refit_summaries:
            logger.info(
                f"{s['refit']:>5d} | {s['n_genes_fit']:>7d} | {s['final_loss_mean']:.4f} | "
                f"{s['train_avg_r2']:.4f} | {s['train_prop_r2_gt_001']:.4f} | {s['train_prop_r2_gt_01']:.4f} | "
                f"{s['test_avg_r2']:.4f} | {s['test_prop_r2_gt_001']:.4f} | {s['test_prop_r2_gt_01']:.4f}"
            )

    config.pop("_joint_refits", None)
    _write_pergene_root_config(
        PERGENE_ROOT,
        config,
        joint_run_ids=joint_run_ids if use_refit else None,
    )
    logger.info(f"Config saved to {os.path.join(PERGENE_ROOT, 'config.yaml')}")
    logger.info("\nDone!")
    _log_run_timing(logger, run_started_wall, run_started_secs, status="finished")


if __name__ == "__main__":
    main()
