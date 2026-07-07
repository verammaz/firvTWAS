import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from tqdm import tqdm
from utils import get_logger
import load_data


def annotation_lambda(
    Z_gene,
    maf_weights_gene,
    threshold,
    tau1=None,
    tau2=None,
    logger=None
):
    """
    Per-variant λ in the joint / per-gene / simulate paths.

    Nonlinear:
      (Z·τ₁) * exp(Z·τ₂) * MAF  where abs(Z·τ₁) >= T, else 0

    """
    assert tau1 is not None and tau2 is not None
    # Ztau2
    lin2 = Z_gene.matmul(tau2)
    mod = torch.exp(lin2)
    # add intercept to Z_gene
    Z_gene = torch.cat([torch.ones(Z_gene.shape[0], 1, dtype=torch.float32, device=Z_gene.device), Z_gene], dim=1)
    assert Z_gene.shape[1] == tau1.shape[0], "Z_gene and tau1 must have the same number of columns"
    # Ztau1
    lin1 = Z_gene.matmul(tau1)
    gate = (torch.abs(lin1) >= threshold).to(lin1.dtype)
    return gate * lin1 * mod * maf_weights_gene


def observation_y(data, gene_name, simulated_parameters=None):
    """Expression vector for the likelihood (simulated y or real data.Y)."""
    if simulated_parameters is not None:
        gp = simulated_parameters.get(gene_name, {})
        if "y" in gp:
            return gp["y"]
        if "mean" in gp:
            return gp["mean"]
        raise KeyError(f"simulated_parameters missing y/mean for {gene_name}")
    gene_key = gene_name.split("/")[-1] if "/" in gene_name else gene_name
    if isinstance(data.Y, dict):
        return data.Y[gene_key]
    return data.Y


def observation_alpha(data, gene_name, config, simulated_parameters=None, default_std=0.5):
    """Observation noise precision alpha = 1 / std^2."""
    if simulated_parameters is not None:
        std = float(
            simulated_parameters.get(
                "obs_std",
                config.get("simulation_obs_std", default_std),
            )
        )
        return torch.as_tensor(1.0 / (std**2), dtype=torch.float32, device=data.device)
    gene_key = gene_name.split("/")[-1] if "/" in gene_name else gene_name
    if data.brr_alphas is not None:
        alpha = data.brr_alphas.get(gene_name)
        if alpha is None:
            alpha = data.brr_alphas.get(gene_key)
        if alpha is not None:
            return torch.as_tensor(float(alpha), dtype=torch.float32, device=data.device)
    return torch.as_tensor(1.0 / (default_std**2), dtype=torch.float32, device=data.device)


class EmmentalJoint(PyroModule):
    """
    Joint model across genes - used by run_joint.py.
    tau1 and tau2 for nonlinear annotation interaction
    """
    def __init__(self, simulated_parameters=None):
        super().__init__()

        self.simulated_parameters = simulated_parameters

    def forward(self, data, config):
        logger = get_logger()

        if not config.get('no_wg', False):
            w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([data.num_genes]).to_event(1)).to(data.device)
        else:
            w_g = torch.ones(data.num_genes, device=data.device)

        if config.get('no_rhog', False):
            rho_g = pyro.deterministic("rho_g", torch.zeros(data.num_genes, device=data.device))
        else:
            rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)).to(data.device)
        
        # Threshold on |Z·tau1| for gating variants. When no_T is set, fix T=0 deterministically.
        if config.get("no_T", False):
            threshold = pyro.deterministic("threshold", torch.zeros((), device=data.device))
        else:
            t_alpha = float(config.get("threshold_prior_alpha", 2.0))
            t_beta = float(config.get("threshold_prior_beta", 20.0))
            threshold = pyro.sample("threshold", dist.Beta(t_alpha, t_beta)).to(data.device)
    
        if config.get('tau1_normal_prior', False):
            # intercept on tau1
            tau1 = pyro.sample("tau1", dist.Normal(0, 1).expand([data.num_anno + 1]).to_event(1)).to(data.device) # normal prior
        else: # dirichlet prior for tau1
            prior_tau1 = torch.ones(data.num_anno + 1, device=data.device) / (data.num_anno + 1) # uniform over annotations (+ intercept)
            tau1 = pyro.sample('tau1', dist.Dirichlet(prior_tau1)).to(data.device)
        
        # prior for tau2 is normal
        tau2 = pyro.sample('tau2', dist.Normal(0, 1).expand([data.num_anno]).to_event(1)).to(data.device) # normal  prior

        for gene_idx, gene_name in enumerate(data.gene_names):
            G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
            num_indivs = G_gene.shape[0]
            lambda_ = annotation_lambda(
                    Z_gene,
                    maf_weights_gene,
                    threshold,
                    tau1=tau1,
                    tau2=tau2,
                    logger=logger,
            )
            
            if config.get('no_rhog', False):
                mu = w_g[gene_idx] * lambda_
                sigma_sqrt = torch.abs(w_g[gene_idx] * lambda_)
            else:
                mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_ 
                sigma_sqrt = (1 - rho_g[gene_idx]) * torch.abs(w_g[gene_idx] * lambda_)
            
            Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
            beta = mu + sigma_sqrt * Z_norm
                    
            pyro.deterministic(f"mu_{gene_name}", mu) 
            pyro.deterministic(f"sigma_{gene_name}", torch.pow(sigma_sqrt, 2))
            pyro.deterministic(f"beta_{gene_name}", beta)

            Gbeta = G_gene.matmul(beta)
            mean = Gbeta.view(num_indivs)

            pyro.deterministic(f"mean_{gene_name}", mean)

            if torch.isnan(mean).any():
                logger.warning(f"NaNs in mean at gene {gene_name}")
                raise ValueError(f"NaNs in mean at gene {gene_name}")

            with pyro.plate(f'data_{gene_name}', num_indivs):
                y_obs = observation_y(data, gene_name, self.simulated_parameters)
                alpha = observation_alpha(
                    data, gene_name, config, self.simulated_parameters, default_std=0.5
                )
                std = torch.as_tensor(1.0 / torch.sqrt(alpha), dtype=torch.float32, device=data.device)
                obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, std), obs=y_obs)
        return data


class EmmentalPerGene(PyroModule):
    """
    Per-gene model - used by run_pergene.py.
    tau and threshold are pre-loaded into data tensors (not inferred here).
    Uses data.std (from alpha_dict) as observation noise.
    """
    def __init__(self, config, data, simulated_parameters=None):
        super().__init__()
        self.simulated_parameters = simulated_parameters
        # exteranlly load tau and threshold from joint model fitting results
        self.__set_tau_threshold(data)
        self.data = data
        self.config = config

    def __set_tau_threshold(self, data):
        self.register_buffer("_fixed_tau1", data.tau1)
        self.register_buffer("_fixed_tau2", data.tau2)
        self.register_buffer("_fixed_threshold", data.threshold)


    def forward(self, data, config=None):
        """
        SVI passes (data, config). ``data`` must match the tensors bound at init
        (same layout as joint ``DataTensors`` for the genes in this fit).
        """
        import utils
        logger = utils.get_logger()
        if config is None:
            config = self.config
        self.data = data

        th_use = self._fixed_threshold
        tau1_use, tau2_use = self._fixed_tau1, self._fixed_tau2

        if not config.get("no_wg", False):
            w_g = pyro.sample("w_g", dist.Normal(0, 1).expand([self.data.num_genes]).to_event(1)).to(
                self.data.device
            )
        else:
            w_g = torch.ones(self.data.num_genes, device=self.data.device)

        if config.get("no_rhog", False):
            rho_g = pyro.deterministic("rho_g", torch.zeros(self.data.num_genes, device=self.data.device))
        else:
            rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([self.data.num_genes]).to_event(1)).to(
                self.data.device
            )

        for gene_idx, gene_name in enumerate(self.data.gene_names):
            G_gene, Z_gene, maf_weights_gene = self.data.get_gene_data(gene_name)
            num_indivs = G_gene.shape[0]
            lambda_ = annotation_lambda(
                    Z_gene,
                    maf_weights_gene,
                    th_use,
                    tau1=tau1_use,
                    tau2=tau2_use
            )

            if config.get("no_rhog", False):
                mu = w_g[gene_idx] * lambda_
                sigma_sqrt = torch.abs(w_g[gene_idx] * lambda_)
            else:
                mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_
                sigma_sqrt = (1 - rho_g[gene_idx]) * torch.abs(w_g[gene_idx] * lambda_)
        
            Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
            beta = mu + sigma_sqrt * Z_norm
        
            pyro.deterministic(f"beta_{gene_name}", beta)
            Gbeta = G_gene.matmul(beta)
            mean = Gbeta.view(num_indivs)
            pyro.deterministic(f"mean_{gene_name}", mean)

            if torch.isnan(mean).any():
                logger.warning(f"NaNs in mean at gene {gene_name}")
                raise ValueError(f"NaNs in mean at gene {gene_name}")

            with pyro.plate(f'data_{gene_name}', num_indivs):
                y_obs = observation_y(self.data, gene_name, self.simulated_parameters)
                alpha = observation_alpha(
                    self.data, gene_name, config, self.simulated_parameters, default_std=0.5
                )
                std = torch.as_tensor(1.0 / torch.sqrt(alpha), dtype=torch.float32, device=self.data.device)
                obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, std), obs=y_obs)
        return self.data



def simulate_expression(data, config, mode="linear", seed=None):
    """
    Simulate expression from the Emmental prior. Mirrors EmmentalJoint.forward.

    Returns a dict of ground-truth parameters plus per-gene ``y`` (noisy) and ``mean`` (noiseless).
    """
    if seed is not None:
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)

    simulated_parameters = {}
    no_wg = bool(config.get("no_wg", False))
    no_rhog = bool(config.get("no_rhog", False))

    if not no_wg:
        w_g = pyro.sample(
            "w_g", dist.Normal(0, 1).expand([data.num_genes]).to_event(1)
        ).to(data.device)
    else:
        w_g = torch.ones(data.num_genes, device=data.device)

    if no_rhog:
        rho_g = torch.zeros(data.num_genes, device=data.device)
    else:
        rho_g = pyro.sample(
            "rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)
        ).to(data.device)
    simulated_parameters["rho_g"] = rho_g

    t_alpha = float(config.get("threshold_prior_alpha", 2.0))
    t_beta = float(config.get("threshold_prior_beta", 20.0))
    threshold = pyro.sample("threshold", dist.Beta(t_alpha, t_beta)).to(data.device)

    if config.get("tau1_normal_prior", False):
        tau1 = pyro.sample(
            "tau1", dist.Normal(0, 1).expand([data.num_anno + 1]).to_event(1)
        ).to(data.device)
    else:
        prior_tau1 = torch.ones(data.num_anno + 1, device=data.device) / (data.num_anno + 1)
        tau1 = pyro.sample("tau1", dist.Dirichlet(prior_tau1)).to(data.device)

    tau2 = pyro.sample(
        "tau2", dist.Normal(0, 1).expand([data.num_anno]).to_event(1)
    ).to(data.device)

    obs_std = float(config.get("simulation_obs_std", 0.5))
    add_noise = bool(config.get("simulation_add_obs_noise", True))
    simulated_parameters["w_g"] = w_g
    simulated_parameters["threshold"] = threshold
    simulated_parameters["mode"] = mode
    simulated_parameters["tau1"] = tau1
    simulated_parameters["tau2"] = tau2
    simulated_parameters["obs_std"] = obs_std

    for gene_idx, gene_name in enumerate(data.gene_names):
        simulated_parameters[gene_name] = {}
        G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
        num_indivs = G_gene.shape[0]
        lambda_ = annotation_lambda(
            Z_gene,
            maf_weights_gene,
            threshold,
            tau1=tau1,
            tau2=tau2,
        )
        if no_rhog:
            mu = w_g[gene_idx] * lambda_
            sigma_sqrt = torch.abs(w_g[gene_idx] * lambda_)
        else:
            mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_
            sigma_sqrt = (1 - rho_g[gene_idx]) * torch.abs(w_g[gene_idx] * lambda_)
        Z_norm = pyro.sample(
            f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)
        ).to(data.device)
        beta = mu + sigma_sqrt * Z_norm
        Gbeta = G_gene.matmul(beta)
        mean = Gbeta.view(num_indivs)
        if add_noise:
            y = mean + obs_std * torch.randn(num_indivs, device=data.device, dtype=mean.dtype)
        else:
            y = mean.clone()
        simulated_parameters[gene_name]["beta"] = beta
        simulated_parameters[gene_name]["mu"] = mu
        simulated_parameters[gene_name]["mean"] = mean
        simulated_parameters[gene_name]["y"] = y

    return simulated_parameters
