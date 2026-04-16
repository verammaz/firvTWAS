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
    config,
    mode: str,
    threshold,
    tau=None,
    tau1=None,
    tau2=None,
):
    """
    Per-variant λ in the joint / per-gene / simulate paths.

    Nonlinear:
      - default (only positive annotations): relu(Z·τ₁ − T) * exp(Z·τ₂) * MAF
      - negative_annotations: (Z·τ₁) * exp(Z·τ₂) * MAF  where abs(Z·τ₁) > T, else 0

    Linear: relu(Z·τ − T) * MAF
    """
    if mode == "nonlinear":
        assert tau1 is not None and tau2 is not None
        lin1 = Z_gene.matmul(tau1)
        lin2 = Z_gene.matmul(tau2)
        mod = torch.exp(lin2)
        if config.get("negative_annotations", False):
            return torch.where(torch.abs(lin1) >= threshold, lin1 * mod * maf_weights_gene, 0)
        return F.relu(lin1 - threshold) * mod * maf_weights_gene

    assert tau is not None
    return F.relu(Z_gene.matmul(tau) - threshold) * maf_weights_gene


def joint_forward(data, config, simulated_parameters=None, mode="linear"):
    logger = get_logger()

    if not config.get('no_wg', False):
        w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([data.num_genes]).to_event(1)).to(data.device)
    else:
        w_g = torch.ones(data.num_genes, device=data.device)

    if (not config.get('burden', False)) and (not config.get('skat', False)):
        rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)).to(data.device)
    
    
    learn_threshold = not config.get("no_filter", False)
    if learn_threshold:
        threshold = pyro.sample("threshold", dist.Beta(2.0, 20.0)).to(data.device) # TODO: change prior?
    else:
        threshold = torch.as_tensor(0, dtype=torch.float32).to(data.device)
        pyro.deterministic("threshold", threshold)

    # prior for taus is uniform over annotations
    prior = torch.ones(data.num_anno, device=data.device) / data.num_anno

    if mode == "nonlinear": # try different priors for tau1 and tau2
        if config.get('tau1_normal_prior', False):
            # One coefficient per annotation (same shape as Dirichlet tau2)
            tau1 = pyro.sample(
                "tau1",
                dist.Normal(0, 1).expand([data.num_anno]).to_event(1),
            ).to(data.device)
        else:
            tau1 = pyro.sample('tau1', dist.Dirichlet(prior)).to(data.device)
        if config.get('tau2_normal_prior', False):
            tau2 = pyro.sample(
                "tau2",
                dist.Normal(0, 1).expand([data.num_anno]).to_event(1),
            ).to(data.device)
        else:
            tau2 = pyro.sample('tau2', dist.Dirichlet(prior)).to(data.device)
    else:
        tau = pyro.sample('tau', dist.Dirichlet(prior)).to(data.device)

    for gene_idx, gene_name in enumerate(data.gene_names):
        G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
        num_indivs = G_gene.shape[0]
        if mode == "nonlinear":
            lambda_ = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                config,
                "nonlinear",
                threshold,
                tau1=tau1,
                tau2=tau2,
            )
        else:
            lambda_ = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                config,
                "linear",
                threshold,
                tau=tau,
            )

        if config.get('no_rhog', False):
            mu = w_g[gene_idx] * lambda_ # if no_wg: wg=1, sigma=mu^2
            Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
            beta = mu + torch.abs(mu) * Z_norm  # beta = mu + sqrt(sigma) * Z_norm; torch.abs (not np) for grad/CUDA
            # try sigma=1 
            # beta = mu + Z_norm
        else:
            mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_ if (not config.get('burden', False)) and (not config.get('skat', False)) else lambda_ * w_g[gene_idx]
            if config.get('burden', False): #rho_g=1 --> sigma=0 (no variance, so beta=mu)
                beta = w_g[gene_idx]* lambda_ #beta=mu
            elif config.get('skat', False): #rho_g=0 --> mu=0
                Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
                beta = torch.abs(w_g[gene_idx] * lambda_) * Z_norm  # sqrt(sigma) * Z_norm
                # try sigma=1 
                # beta = Z_norm
            else: 
                Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
                beta = (
                    rho_g[gene_idx] * w_g[gene_idx] * lambda_
                    + (1 - rho_g[gene_idx]) * torch.abs(w_g[gene_idx] * lambda_) * Z_norm
                )
                # try sigma=1 
                #beta = rho_g[gene_idx] * w_g[gene_idx] * lambda_ + Z_norm

        pyro.deterministic(f"mu_{gene_name}", mu)            
        pyro.deterministic(f"beta_{gene_name}", beta)

        Gbeta = G_gene.matmul(beta)
        mean = Gbeta.view(num_indivs)

        pyro.deterministic(f"mean_{gene_name}", mean)

        if torch.isnan(mean).any():
            logger.warning(f"NaNs in mean at gene {gene_name}")
            return "FAIL"

        with pyro.plate(f'data_{gene_name}', num_indivs):
            gene_key = gene_name.split("/")[1] if "/" in gene_name else gene_name
            if simulated_parameters:
                obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, 0.5),
                                    obs=simulated_parameters[gene_name]['mean'])
            else:
                if data.brr_alphas is not None:
                    std = torch.as_tensor(1.0 / np.sqrt(data.brr_alphas[gene_name]), dtype=torch.float32, device=data.device)
                else:
                    std = 0.5
                obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, std),
                                    obs=data.Y[gene_key])
    return data


class ParmigianoExpJoint(PyroModule):
    """
    Joint model across genes - used by run_joint.py.
    Supports burden, skat, no_wg, no_threshold modes.
    tau (and threshold) are inferred, not pre-loaded.
    """
    def __init__(self, simulated_parameters=None):
        super().__init__()

        self.simulated_parameters = simulated_parameters

    def forward(self, data, config):
        return joint_forward(data, config, self.simulated_parameters)


class ParmigianoExpJointNonlinear(PyroModule):
    """
    Same setup as parmigiano_expression, but with nonlinear annotation interaction.
    tau1 and tau2 for nonlinear annotation interaction
    """
    def __init__(self, simulated_parameters=None):
        super().__init__()

        self.simulated_parameters = simulated_parameters

    def forward(self, data, config):
        return joint_forward(data, config, self.simulated_parameters, mode="nonlinear")


class ParmigianoExpPerGene(PyroModule):
    """
    Per-gene model - used by run_pergene.py.
    tau and threshold are pre-loaded into data tensors (not inferred here).
    Supports burden, skat, no_wg, no_filter modes.
    Uses data.std (from alpha_dict) as observation noise.
    """
    def __init__(self, config, data,simulated_parameters=None):
        super().__init__()
        self.simulated_parameters = simulated_parameters
        # exteranlly load tau and threshold from joint model fitting results
        data = self.__load_tau_threshold(config, data)
        self.data = data
        self.config = config

    def __load_tau_threshold(self, config, data):
        threshold, tau, tau1, tau2 = load_data.load_tau_threshold(config)
        data.threshold = torch.as_tensor(threshold, dtype=torch.float32, device=data.device)
        mode = 'nonlinear' if config.get('tau12', False) else 'linear'
        if mode == 'nonlinear':
            assert tau1 is not None and tau2 is not None, "tau1 and tau2 must be provided for nonlinear mode"
            data.tau1 = torch.as_tensor(tau1, dtype=torch.float32, device=data.device)
            data.tau2 = torch.as_tensor(tau2, dtype=torch.float32, device=data.device)
        else:
            assert tau is not None, "tau must be provided for linear mode"
            data.tau = torch.as_tensor(tau, dtype=torch.float32, device=data.device)
        return data


    def forward(self, mode="linear"):
        import utils
        logger = utils.get_logger()

        if not self.config.get('no_wg', False):
            w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([self.data.num_genes]).to_event(1)).to(self.data.device)
        else:
            w_g = torch.ones(self.data.num_genes, device=self.data.device)

        if (not self.config.get('burden', False)) and (not self.config.get('skat', False)):
            rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([self.data.num_genes]).to_event(1)).to(self.data.device)

        for gene_idx, gene_name in enumerate(self.data.gene_names):
            G_gene, Z_gene, maf_weights_gene = self.data.get_gene_data(gene_name)
            num_indivs = G_gene.shape[0]

            # lambda_ does NOT include w_g — applied explicitly per branch, matching joint model
            if mode == "nonlinear":
                lambda_ = annotation_lambda(
                    Z_gene,
                    maf_weights_gene,
                    self.config,
                    "nonlinear",
                    self.data.threshold,
                    tau1=self.data.tau1,
                    tau2=self.data.tau2,
                )
            else:
                lambda_ = annotation_lambda(
                    Z_gene,
                    maf_weights_gene,
                    self.config,
                    "linear",
                    self.data.threshold,
                    tau=self.data.tau,
                )

            if self.config.get('no_rhog', False):
                mu = w_g[gene_idx] * lambda_
                Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(self.data.device)
                beta = mu + mu * Z_norm
            else:
                mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_ if (not self.config.get('burden', False)) and (not self.config.get('skat', False)) else lambda_ * w_g[gene_idx]
                if self.config.get('burden', False):
                    beta = w_g[gene_idx] * lambda_
                elif self.config.get('skat', False):
                    Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(self.data.device)
                    beta = w_g[gene_idx] * lambda_ * Z_norm
                else:
                    Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(self.data.device)
                    beta = rho_g[gene_idx] * w_g[gene_idx] * lambda_ + (1 - rho_g[gene_idx]) * w_g[gene_idx] * lambda_ * Z_norm

        
            pyro.deterministic(f"beta_{gene_name}", beta)
            Gbeta = G_gene.matmul(beta)
            mean = Gbeta.view(num_indivs)
            pyro.deterministic(f"mean_{gene_name}", mean)

            if torch.isnan(mean).any():
                logger.warning(f"NaNs in mean at gene {gene_name}")
                return "FAIL"

            with pyro.plate(f'data_{gene_name}', num_indivs):
                gene_key = gene_name.split("/")[1] if "/" in gene_name else gene_name
                if self.simulated_parameters:
                    obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, 1),
                                      obs=self.simulated_parameters[gene_name]['mean'])
                else:
                    if self.data.brr_alphas is not None:
                        std = torch.as_tensor(1.0 / np.sqrt(self.data.brr_alphas[gene_name]), dtype=torch.float32, device=self.data.device)
                    else:
                        std = 0.5
                    obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, std),
                                      obs=self.data.Y[gene_key])
        return self.data


def simulate_expression(data, config, mode="linear"):
    """
    Simulate expression from prior. Mirrors parmigiano_expression.forward exactly.
    """
    simulated_parameters = {}

    if not config.get('no_wg', False):
        w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([data.num_genes]).to_event(1)).to(data.device)
    else:
        w_g = torch.ones(data.num_genes, device=data.device)

    if (not config.get('burden', False)) and (not config.get('skat', False)):
        rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)).to(data.device)
        simulated_parameters['rho_g'] = rho_g

    learn_threshold = (not config.get("no_filter", False)) and (
        not config.get("chrombpnet_dist_only", False)
    )
    if learn_threshold:
        threshold = pyro.sample("threshold", dist.Beta(2.0, 20.0)).to(data.device)
    else:
        threshold = torch.as_tensor(0, dtype=torch.float32).to(data.device)
        pyro.deterministic("threshold", threshold)

    prior = torch.ones(data.num_anno, device=data.device) / data.num_anno
    if mode == "nonlinear":
        tau1 = pyro.sample("tau1", dist.Dirichlet(prior)).to(data.device)
        if config.get("tau2_normal_prior", False):
            tau2 = pyro.sample(
                "tau2",
                dist.Normal(0, 1).expand([data.num_anno]).to_event(1),
            ).to(data.device)
        else:
            tau2 = pyro.sample("tau2", dist.Dirichlet(prior)).to(data.device)
        simulated_parameters["tau1"] = tau1
        simulated_parameters["tau2"] = tau2
    else:
        tau = pyro.sample("tau", dist.Dirichlet(prior)).to(data.device)
        simulated_parameters["tau"] = tau

    simulated_parameters['w_g'] = w_g
    simulated_parameters['threshold'] = threshold
    simulated_parameters['mode'] = mode

    for gene_idx, gene_name in enumerate(data.gene_names):
        simulated_parameters[gene_name] = {}
        G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
        num_indivs = G_gene.shape[0]
        if mode == "nonlinear":
            lambda_ = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                config,
                "nonlinear",
                threshold,
                tau1=tau1,
                tau2=tau2,
            )
        else:
            lambda_ = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                config,
                "linear",
                threshold,
                tau=tau,
            )
        mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_ if (not config.get('burden', False)) and (not config.get('skat', False)) else lambda_ * w_g[gene_idx]
        if config.get('burden', False):
            beta = w_g[gene_idx] * lambda_
        elif config.get('skat', False):
            Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
            beta = w_g[gene_idx] * lambda_ * Z_norm
        else:
            Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
            beta = rho_g[gene_idx] * w_g[gene_idx] * lambda_ + (1 - rho_g[gene_idx]) * w_g[gene_idx] * lambda_ * Z_norm
        Gbeta = G_gene.matmul(beta)
        mean = Gbeta.view(num_indivs)
        simulated_parameters[gene_name]['beta'] = beta
        simulated_parameters[gene_name]['mu'] = mu
        simulated_parameters[gene_name]['mean'] = mean
    return simulated_parameters
