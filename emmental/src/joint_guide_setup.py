"""
Shared SVI guide setup for joint Emmental (full and collapsed).

Threshold uses AutoNormal (not AutoDelta) so T can move off the Beta(2,20) mode (0.05).
Tau1/tau2 remain under AutoDelta with explicit prior-centred init.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pyro
import torch
from pyro import poutine
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoGuideList, AutoNormal
from pyro.infer import SVI, Trace_ELBO

try:
    from pyro.optim import ClippedAdam
    HAS_CLIPPED_ADAM = True
except (ImportError, AttributeError):
    HAS_CLIPPED_ADAM = False

import utils
from models import annotation_lambda


def threshold_prior_mean(config: dict) -> float:
    alpha = float(config.get("threshold_prior_alpha", 2.0))
    beta = float(config.get("threshold_prior_beta", 20.0))
    return alpha / (alpha + beta)


def threshold_prior_mode(config: dict) -> float:
    alpha = float(config.get("threshold_prior_alpha", 2.0))
    beta = float(config.get("threshold_prior_beta", 20.0))
    if alpha <= 1.0:
        return threshold_prior_mean(config)
    return (alpha - 1.0) / (alpha + beta - 2.0)


def collect_abs_lin1(
    data,
    config: dict,
    tau1: Optional[torch.Tensor] = None,
    tau2: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Pool |Z·τ₁| (with intercept) over all variants in ``data.gene_names``.
    Uses prior-centred τ unless τ vectors are supplied.
    """
    device = data.device
    num_anno = data.num_anno
    if tau2 is None:
        tau2 = torch.zeros(num_anno, device=device)
    if tau1 is None:
        if config.get("tau1_normal_prior", False):
            tau1 = torch.zeros(num_anno + 1, device=device)
        else:
            tau1 = torch.ones(num_anno + 1, device=device) / (num_anno + 1)

    # Gate evaluation only — threshold cancels in |lin1| vs T comparison; use 0.
    th = torch.tensor(0.0, device=device)
    values = []
    for gene_name in data.gene_names:
        _, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
        Z_aug = torch.cat(
            [torch.ones(Z_gene.shape[0], 1, dtype=Z_gene.dtype, device=device), Z_gene],
            dim=1,
        )
        lin1 = Z_aug.matmul(tau1.to(device))
        values.append(lin1.abs().detach().cpu().numpy().ravel())
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate(values)


def resolve_threshold_init(data, config: dict) -> float:
    """Initial T in (0, 1) for the guide."""
    # When no_T is set, we fix T=0 deterministically; guide init follows.
    if config.get("no_T", False):
        return 0.0
    strategy = str(config.get("threshold_init", "prior_mean")).lower()
    if strategy == "prior_mode":
        return threshold_prior_mode(config)
    if strategy == "data_quantile":
        q = float(config.get("threshold_init_quantile", 0.25))
        abs_lin1 = collect_abs_lin1(data, config)
        if abs_lin1.size == 0:
            return threshold_prior_mean(config)
        val = float(np.quantile(abs_lin1, q))
        eps = 1e-4
        return float(np.clip(val, eps, 1.0 - eps))
    return threshold_prior_mean(config)


def make_tau_init_loc_fn(data, config: dict) -> Callable:
    """AutoDelta init for tau1 / tau2 at prior centres."""
    device = data.device
    num_anno = data.num_anno
    tau2_init = torch.zeros(num_anno, device=device)
    if config.get("tau1_normal_prior", False):
        tau1_init = torch.zeros(num_anno + 1, device=device)
    else:
        tau1_init = torch.ones(num_anno + 1, device=device) / (num_anno + 1)

    def init_loc_fn(site):
        if site["type"] != "sample" or site.get("is_observed", False):
            return None
        if site["name"] == "tau1":
            init = tau1_init
        elif site["name"] == "tau2":
            init = tau2_init
        else:
            return None
        if init.shape != site["fn"].event_shape:
            init = init.view(site["fn"].event_shape)
        return init

    return init_loc_fn


def make_threshold_init_loc_fn(data, config: dict) -> Callable:
    t0 = resolve_threshold_init(data, config)

    def init_loc_fn(site):
        if site["type"] != "sample" or site.get("is_observed", False):
            return None
        if site["name"] == "threshold":
            return torch.tensor(t0, dtype=torch.float32, device=data.device)
        return None

    return init_loc_fn


def gate_pass_rates(data, config: dict, threshold: float, tau1=None, tau2=None) -> dict:
    """Per-gene fraction of variants with |lin1| >= threshold."""
    device = data.device
    th = torch.tensor(float(threshold), device=device)
    if tau2 is None:
        tau2 = torch.zeros(data.num_anno, device=device)
    if tau1 is None:
        if config.get("tau1_normal_prior", False):
            tau1 = torch.zeros(data.num_anno + 1, device=device)
        else:
            tau1 = torch.ones(data.num_anno + 1, device=device) / (data.num_anno + 1)
    rates = {}
    for gene_name in data.gene_names:
        _, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
        lam = annotation_lambda(Z_gene, maf_weights_gene, th, tau1=tau1, tau2=tau2)
        active = (lam.abs() > 0).float().mean().item()
        rates[gene_name] = active
    return rates


def wg_rhog_guide_sites(config: dict) -> list:
    """Latent global sites for w_g and rho_g (empty when ablated via no_wg / no_rhog)."""
    sites = []
    if not config.get("no_wg", False):
        sites.append("w_g")
    if not config.get("no_rhog", False):
        sites.append("rho_g")
    return sites


def pergene_wg_rhog_fully_ablated(config: dict) -> bool:
    """True when collapsed per-gene has no w_g / rho_g latents (both ablated)."""
    return bool(config.get("no_wg", False)) and bool(config.get("no_rhog", False))


def add_wg_rhog_latent_guides(
    guide: AutoGuideList,
    model,
    data_train,
    config: dict,
    *,
    init_loc_fn=None,
) -> None:
    """
    Guide w_g / rho_g and any remaining latents (e.g. per-gene Z in full model).

    Default: AutoDiagonalNormal on w_g, rho_g, and Z sites.
    When ``wg_rhog_delta_guide`` is True: AutoDelta on w_g/rho_g; AutoDiagonalNormal on Z only.
    """
    logger = utils.get_logger()
    tau_threshold_hide = ["tau1", "tau2", "threshold"]
    wg_rhog_sites = wg_rhog_guide_sites(config)
    use_delta = bool(config.get("wg_rhog_delta_guide", False))

    hide_all_special = tau_threshold_hide + wg_rhog_sites
    blocked_rest = poutine.block(model, hide=hide_all_special)
    trace = poutine.trace(blocked_rest).get_trace(data_train, config)
    has_other_latents = any(
        site["type"] == "sample" and not site.get("is_observed", False)
        for site in trace.nodes.values()
    )

    if use_delta and wg_rhog_sites:
        guide.add(
            AutoDelta(
                poutine.block(model, expose=wg_rhog_sites),
                init_loc_fn=init_loc_fn,
            )
        )
        logger.info("w_g/rho_g guide: AutoDelta on %s", wg_rhog_sites)
        if has_other_latents:
            if init_loc_fn is not None:
                guide.add(AutoDiagonalNormal(blocked_rest, init_loc_fn=init_loc_fn))
            else:
                guide.add(AutoDiagonalNormal(blocked_rest))
            logger.info("Additional latent guide: AutoDiagonalNormal (e.g. Z sites)")
    elif wg_rhog_sites or has_other_latents:
        blocked_wg = poutine.block(model, hide=tau_threshold_hide)
        if init_loc_fn is not None:
            guide.add(AutoDiagonalNormal(blocked_wg, init_loc_fn=init_loc_fn))
        else:
            guide.add(AutoDiagonalNormal(blocked_wg))
        logger.info("Latent guide: AutoDiagonalNormal on w_g, rho_g, and other sites")
    else:
        logger.warning(
            "No latent variables for w_g/rho_g guide (after hiding tau/threshold)."
        )


def setup_joint_guide(
    model,
    data_train,
    config: dict,
    *,
    init_loc_fn=None,
    simulated_parameters=None,
) -> Tuple[AutoGuideList, SVI]:
    """
    Build AutoGuideList + SVI for joint models (full or collapsed).

    - w_g / rho_g: AutoDiagonalNormal (default) or AutoDelta (``wg_rhog_delta_guide``)
    - AutoDelta: tau1, tau2
    - AutoNormal: threshold (config ``threshold_normal_guide``, default True)
    """
    logger = utils.get_logger()
    tau_sites = ["tau1", "tau2"]
    threshold_site = ["threshold"]
    to_optimize = tau_sites + threshold_site

    guide = AutoGuideList(model)
    add_wg_rhog_latent_guides(
        guide, model, data_train, config, init_loc_fn=init_loc_fn
    )

    tau_init = make_tau_init_loc_fn(data_train, config)
    guide.add(AutoDelta(poutine.block(model, expose=tau_sites), init_loc_fn=tau_init))

    use_normal = config.get("threshold_normal_guide", True)
    t_init_val = resolve_threshold_init(data_train, config)
    t_init_fn = make_threshold_init_loc_fn(data_train, config)
    if use_normal:
        guide.add(AutoNormal(poutine.block(model, expose=threshold_site), init_loc_fn=t_init_fn))
        logger.info(
            "Threshold guide: AutoNormal, init=%.4f (strategy=%s, prior mode=%.4f, mean=%.4f)",
            t_init_val,
            config.get("threshold_init", "prior_mean"),
            threshold_prior_mode(config),
            threshold_prior_mean(config),
        )
    else:
        guide.add(
            AutoDelta(
                poutine.block(model, expose=threshold_site),
                init_loc_fn=t_init_fn,
            )
        )
        logger.info(
            "Threshold guide: AutoDelta, init=%.4f (strategy=%s)",
            t_init_val,
            config.get("threshold_init", "prior_mean"),
        )

    clip_norm = config.get("clip_norm", 10.0)
    if HAS_CLIPPED_ADAM:
        optim = ClippedAdam({"lr": config["lr"], "clip_norm": clip_norm})
    else:
        optim = pyro.optim.Adam({"lr": config["lr"]})

    svi = SVI(model, guide=guide, optim=optim, loss=Trace_ELBO())
    return guide, svi
