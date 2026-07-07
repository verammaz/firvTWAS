"""
SVI training for collapsed Emmental models (joint and per-gene).

Uses models_collapsed (beta integrated out; pyro.factor data terms).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import pyro
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from tqdm import tqdm

try:
    from pyro.optim import ClippedAdam
    HAS_CLIPPED_ADAM = True
except (ImportError, AttributeError):
    HAS_CLIPPED_ADAM = False

import utils
from save_outputs import build_posterior_stats, summarize_predictive_tensor, wg_rhog_delta_sites
from joint_guide_setup import (
    pergene_wg_rhog_fully_ablated,
    setup_joint_guide,
    wg_rhog_guide_sites,
)
from pyro.infer.autoguide import AutoDelta
from collapsed_likelihood import collapsed_beta_and_logp, mu_sigma_from_lambda
from models import annotation_lambda, observation_alpha, observation_y
from models_collapsed import EmmentalJointCollapsed, EmmentalPerGeneCollapsed


def _extract_samples(samples, data_obj, key_prefix: str) -> Dict[str, np.ndarray]:
    """Pull per-gene arrays from Predictive output: gene_name -> (n_posterior x n_variants)."""
    logger = utils.get_logger()
    result = {}
    for gene_name in data_obj.gene_names:
        key = f"{key_prefix}_{gene_name}"  # e.g. beta_chr1/ENSG000...
        if key in samples:
            arr = samples[key]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            if arr.ndim == 3:
                arr = arr.squeeze(axis=1)  # Remove singleton event dim if present
            result[gene_name] = arr
        else:
            logger.warning("%s not found in posterior samples.", key)
    return result


def _make_optimizer(config):
    """Build Adam or ClippedAdam from config lr and clip_norm."""
    clip_norm = config.get("clip_norm", 10.0)
    logger = utils.get_logger()
    if HAS_CLIPPED_ADAM:
        logger.info("Using ClippedAdam with clip_norm=%s", clip_norm)
        return ClippedAdam({"lr": config["lr"], "clip_norm": clip_norm})
    logger.warning("ClippedAdam not available; using Adam.")
    return Adam({"lr": config["lr"]})


def setup_joint_collapsed(data_train, config, init_loc_fn=None, simulated_parameters=None):
    """Guide: AutoDiagonalNormal on w_g/rho_g; AutoDelta tau; AutoNormal threshold."""
    model = EmmentalJointCollapsed(simulated_parameters=simulated_parameters)
    # setup_joint_guide wires specialized guides for tau1/tau2/threshold + SVI object
    guide, svi = setup_joint_guide(
        model,
        data_train,
        config,
        init_loc_fn=init_loc_fn,  # Optional warm-start for guide loc parameters
    )
    return model, guide, svi


def setup_pergene_collapsed(data_train, config, init_loc_fn=None, simulated_parameters=None):
    """Guide on w_g and rho_g only (tau/T fixed in model buffers)."""
    logger = utils.get_logger()
    if pergene_wg_rhog_fully_ablated(config):
        raise ValueError(
            "setup_pergene_collapsed called with no_wg and no_rhog; "
            "use fit_pergene_collapsed_deterministic instead."
        )
    model = EmmentalPerGeneCollapsed(config, data_train, simulated_parameters=simulated_parameters)
    latent_sites = wg_rhog_guide_sites(config)
    use_delta = bool(config.get("wg_rhog_delta_guide", False))

    if not latent_sites:
        raise ValueError("Collapsed per-gene model has no latent w_g/rho_g sites.")

    guide_cls = AutoDelta if use_delta else AutoDiagonalNormal
    guide_name = "AutoDelta" if use_delta else "AutoDiagonalNormal"
    if init_loc_fn is not None:
        guide = guide_cls(model, init_loc_fn=init_loc_fn)
    else:
        guide = guide_cls(model)
    logger.info("Per-gene collapsed w_g/rho_g guide: %s on %s", guide_name, latent_sites)
    svi = SVI(model, guide=guide, optim=_make_optimizer(config), loss=Trace_ELBO())
    return model, guide, svi


def _run_svi(
    svi,           # Pyro SVI object (model + guide + optimizer + ELBO)
    data,          # DataTensors passed to model.forward(data, config)
    config,
    epochs: int,
    log_every: int = 10,
    use_tqdm: bool = True,
    after_epoch=None,  # Optional callback(epoch) after each step
):
    """Generic SVI training loop with optional early stopping."""
    logger = utils.get_logger()
    losses = []
    times = []
    early_stop = config.get("early_stop") is True
    min_epochs = int(config.get("early_stop_min_epochs", 50))
    patience = int(config.get("early_stop_patience", 30))
    rel_tol = float(config.get("early_stop_rel_tol", 1e-3))
    best_loss = float("inf")
    epochs_without_improvement = 0

    epoch_iter = tqdm(range(epochs)) if use_tqdm else range(epochs)
    for epoch in epoch_iter:
        start = time.time()
        try:
            loss = svi.step(data, config)  # One gradient step; returns estimated -ELBO
        except ValueError as e:
            if "nan" in str(e).lower() or "invalid" in str(e).lower():
                logger.error("Numerical instability at epoch %s: %s", epoch, e)
                raise
            raise
        if not np.isfinite(loss):
            logger.warning("NaN/Inf loss at epoch %s; stopping.", epoch)
            break
        times.append(time.time() - start)
        loss_f = float(loss)
        losses.append(loss_f)
        if early_stop:
            # Improvement = loss dropped by at least rel_tol fraction vs best so far
            if loss_f < best_loss * (1.0 - rel_tol):
                best_loss = loss_f
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                epoch + 1 >= min_epochs
                and epochs_without_improvement >= patience
            ):
                logger.info(
                    "Early stopping at epoch %s (best loss %.4f; "
                    "no >%.2g relative improvement for %s epochs).",
                    epoch,
                    best_loss,
                    rel_tol,
                    patience,
                )
                break
        if after_epoch is not None:
            after_epoch(epoch)
        if epoch % log_every == 0:
            logger.info("  Epoch %s: Loss = %.4f", epoch, loss_f)
    return losses, times


def _posterior_from_predictive(model, guide, data, config) -> Tuple[Dict, Dict, Dict]:
    """Sample from guide after training; summarize global sites + per-gene beta/mu/sigma."""
    logger = utils.get_logger()
    logger.info("Generating posterior samples (collapsed model)...")
    guide.requires_grad_(False)  # No gradients needed for posterior sampling
    predictive = Predictive(model, guide=guide, num_samples=config["n_posterior"])
    with torch.no_grad():
        samples = predictive(data, config)  # Dict[str, Tensor] with leading sample dim
    posterior_stats = build_posterior_stats(
        samples, delta_sites=wg_rhog_delta_sites(config)
    )
    beta_samples = _extract_samples(samples, data, "beta")
    mu_samples = _extract_samples(samples, data, "mu")
    sigma_samples = _extract_samples(samples, data, "sigma")
    return posterior_stats, beta_samples, mu_samples, sigma_samples


def fit_joint_collapsed(
    data_train,
    config,
    init_loc_fn=None,
    tau_history_fn=None,  # Optional: record tau1/tau2 each epoch for diagnostics
    simulated_parameters=None,
) -> Tuple[list, list, dict, dict, dict, dict, list]:
    """
    Fit collapsed joint model. Returns same tuple as emmental_joint.fit_emmental.
    tau_history_fn(epoch) -> optional (tau1, tau2) arrays for logging.
    """
    logger = utils.get_logger()
    pyro.clear_param_store()  # Fresh guide parameters for this run
    model, guide, svi = setup_joint_collapsed(
        data_train, config, init_loc_fn=init_loc_fn, simulated_parameters=simulated_parameters
    )
    logger.info("Training collapsed joint model for %s epochs...", config["epochs"])
    tau_history = []

    def _after_epoch(epoch):
        if tau_history_fn is not None:
            tau_history.append(tau_history_fn(epoch))

    losses, times = _run_svi(
        svi,
        data_train,
        config,
        config["epochs"],
        log_every=10,
        after_epoch=_after_epoch if tau_history_fn is not None else None,
    )
    posterior_stats, beta_samples, mu_samples, sigma_samples = _posterior_from_predictive(
        model, guide, data_train, config
    )
    logger.info("Collapsed joint training complete.")
    return losses, times, posterior_stats, beta_samples, mu_samples, sigma_samples, tau_history


def _fixed_wg_rhog_posterior_stats(config, num_genes: int) -> Dict[str, Dict[str, Any]]:
    """Point estimates for ablated w_g=1 and recorded rho_g=0 (no_rhog branch uses mu=w*lambda)."""
    n = int(num_genes)
    w_mean = np.ones(n, dtype=float)
    rho_mean = np.zeros(n, dtype=float)
    return {
        "w_g": summarize_predictive_tensor(w_mean, point_estimate=True),
        "rho_g": summarize_predictive_tensor(rho_mean, point_estimate=True),
    }


def fit_pergene_collapsed_deterministic(
    data_train,
    config,
    simulated_parameters=None,
) -> Tuple[list, list, dict, dict, dict]:
    """
    Collapsed per-gene with both w_g and rho_g ablated: analytic beta only (no SVI).

    Uses joint tau/T, fixed w_g=1, no_rhog mu/sigma (mu=w*lambda, sigma=|w*lambda|).
    """
    logger = utils.get_logger()
    logger.info(
        "Deterministic collapsed per-gene (no_wg + no_rhog): "
        "closed-form beta_hat; no w_g/rho_g inference."
    )

    if config.get("no_T", False):
        threshold = torch.zeros((), device=data_train.device)
    else:
        threshold = data_train.threshold

    tau1 = data_train.tau1
    tau2 = data_train.tau2
    no_rhog = True
    n_post = int(config.get("n_posterior", 1))
    n_post = max(n_post, 1)

    beta_samples: Dict[str, np.ndarray] = {}
    mean_samples: Dict[str, np.ndarray] = {}

    for gene_name in data_train.gene_names:
        G_gene, Z_gene, maf_weights_gene = data_train.get_gene_data(gene_name)
        lam = annotation_lambda(
            Z_gene,
            maf_weights_gene,
            threshold,
            tau1=tau1,
            tau2=tau2,
        )
        w_g = torch.ones((), device=data_train.device)
        rho_g = torch.zeros((), device=data_train.device)
        mu, sigma = mu_sigma_from_lambda(
            lam,
            w_g,
            rho_g,
            no_rhog=no_rhog,
            eps=0.0,
        )
        y_gene = observation_y(data_train, gene_name, simulated_parameters)
        alpha = observation_alpha(
            data_train, gene_name, config, simulated_parameters, default_std=0.5
        )
        beta_hat, _logp = collapsed_beta_and_logp(
            G_gene, y_gene, mu, sigma, alpha, lam
        )
        mean = G_gene.matmul(beta_hat).view(-1)

        if torch.isnan(mean).any():
            raise ValueError(f"NaNs in mean at gene {gene_name}")

        beta_np = beta_hat.detach().cpu().numpy().astype(np.float64, copy=False)
        mean_np = mean.detach().cpu().numpy().astype(np.float64, copy=False)
        beta_samples[gene_name] = np.tile(beta_np[np.newaxis, :], (n_post, 1))
        mean_samples[gene_name] = np.tile(mean_np[np.newaxis, :], (n_post, 1))

    posterior_stats = _fixed_wg_rhog_posterior_stats(config, data_train.num_genes)
    losses: list = []
    times: list = []
    logger.info("Deterministic collapsed per-gene complete.")
    return losses, times, posterior_stats, beta_samples, mean_samples


def fit_pergene_collapsed(
    data_train,
    config,
    init_loc_fn=None,
    simulated_parameters=None,
) -> Tuple[list, list, dict, dict, dict]:
    """
    Fit collapsed per-gene model on Train DataTensors.
    Returns (losses, times, posterior_stats, beta_samples, mean_samples).
    """
    logger = utils.get_logger()
    if pergene_wg_rhog_fully_ablated(config):
        return fit_pergene_collapsed_deterministic(
            data_train, config, simulated_parameters=simulated_parameters
        )

    pyro.clear_param_store()
    model, guide, svi = setup_pergene_collapsed(
        data_train, config, init_loc_fn=init_loc_fn, simulated_parameters=simulated_parameters
    )
    log_every = 50
    fit_config = dict(config)
    if fit_config.get("early_stop") is None:
        fit_config["early_stop"] = True  # Default on for per-gene refits
    logger.info(
        "Training collapsed per-gene model for up to %s epochs "
        "(early_stop=%s, patience=%s, rel_tol=%s)...",
        fit_config["epochs"],
        fit_config.get("early_stop"),
        fit_config.get("early_stop_patience", 30),
        fit_config.get("early_stop_rel_tol", 1e-3),
    )
    losses, times = _run_svi(
        svi,
        data_train,
        fit_config,
        fit_config["epochs"],
        log_every=log_every,
        use_tqdm=False,  # Per-gene runs are often batched externally; skip tqdm spam
    )
    logger.info("Generating posterior samples (collapsed per-gene)...")
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=config["n_posterior"])
    with torch.no_grad():
        samples = predictive(data_train, config)
    posterior_stats = build_posterior_stats(
        samples, delta_sites=wg_rhog_delta_sites(config)
    )
    beta_samples = _extract_samples(samples, data_train, "beta")
    mean_samples = _extract_samples(samples, data_train, "mean")
    logger.info("Collapsed per-gene training complete.")
    return losses, times, posterior_stats, beta_samples, mean_samples
