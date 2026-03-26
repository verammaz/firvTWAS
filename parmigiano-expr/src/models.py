import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, AutoMultivariateNormal, AutoNormal
from pyro.infer import SVI, Trace_ELBO, RenyiELBO, Predictive
from pyro.optim import PyroLRScheduler
from tqdm import tqdm
from utils import get_logger



def joint_forward(data, config, simulated_parameters=None, mode="linear"):
    logger = get_logger()

    if not config.get('no_wg', False):
        w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([data.num_genes]).to_event(1)).to(data.device)
    else:
        w_g = torch.ones(data.num_genes, device=data.device)

    if (not config.get('burden', False)) and (not config.get('skat', False)):
        rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)).to(data.device)
    
    if not config.get('no_filter', False):
        threshold = pyro.sample("threshold", dist.Beta(2.0, 20.0)).to(data.device)
    else:
        threshold = torch.as_tensor(0, dtype = torch.float32).to(data.device)
        pyro.deterministic(f"threshold", threshold)

    # prior for taus is uniform over annotations
    prior = torch.ones(data.num_anno, device=data.device) / data.num_anno

    if mode == "nonlinear":
        tau1 = pyro.sample('tau1', dist.Dirichlet(prior)).to(data.device)
        tau2 = pyro.sample('tau2', dist.Dirichlet(prior)).to(data.device)
    
    else:
        tau = pyro.sample('tau', dist.Dirichlet(prior)).to(data.device)

    for gene_idx, gene_name in enumerate(data.gene_names):
        G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
        num_indivs = G_gene.shape[0]
        if mode == "nonlinear":
            lambda_ = F.relu((Z_gene.matmul(tau1)) - threshold) * torch.exp(Z_gene.matmul(tau2)) * maf_weights_gene
        else:
            lambda_ = F.relu((Z_gene.matmul(tau)) - threshold) * maf_weights_gene

        if config.get('no_rhog', False):
            mu = w_g[gene_idx] * lambda_ # if no_wg: wg=1, sigma=mu^2
            Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
            beta = mu + mu * Z_norm #beta = mu + \sqrt{sigma} * Z_norm
        else:
            mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_ if (not config.get('burden', False)) and (not config.get('skat', False)) else lambda_ * w_g[gene_idx]
            if config.get('burden', False): #rho_g=1 --> sigma=0 (no variance, so beta=mu)
                beta = w_g[gene_idx]* lambda_ # beta=mu
            elif config.get('skat', False): #rho_g=0 --> mu=0
                Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
                beta = w_g[gene_idx]* lambda_ * Z_norm #beta = \sqrt{sigma} * Z_norm
            else: 
                Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
                beta = rho_g[gene_idx] * w_g[gene_idx] * lambda_  + (1 - rho_g[gene_idx]) * w_g[gene_idx]* lambda_ * Z_norm


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
    def __init__(self, alphas=None, simulated_parameters=None):
        super().__init__()
        self.simulated_parameters = simulated_parameters

        # Bayesian Ridge Regression alphas
        self.alphas = alphas # dict: gene_name -> float alpha

    def forward(self, data, config):
        import utils
        logger = utils.get_logger()

        if not config.get('no_wg', False):
            w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([data.num_genes]).to_event(1)).to(data.device)
        else:
            w_g = torch.ones(data.num_genes, device=data.device)

        if (not config.get('burden', False)) and (not config.get('skat', False)):
            rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)).to(data.device)

        for gene_idx, gene_name in enumerate(data.gene_names):
            G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
            num_indivs = G_gene.shape[0]

            # lambda_ does NOT include w_g — applied explicitly per branch, matching joint model
            lambda_ = F.relu((Z_gene.matmul(data.tau)) - data.threshold) * maf_weights_gene

            mu = rho_g[gene_idx] * w_g[gene_idx] * lambda_ if (not config.get('burden', False)) and (not config.get('skat', False)) else lambda_ * w_g[gene_idx]
            pyro.deterministic(f"mu_{gene_name}", mu)

            if config.get('burden', False):
                beta = w_g[gene_idx] * lambda_
            elif config.get('skat', False):
                Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
                beta = w_g[gene_idx] * lambda_ * Z_norm
            else:
                Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device)
                beta = rho_g[gene_idx] * w_g[gene_idx] * lambda_ + (1 - rho_g[gene_idx]) * w_g[gene_idx] * lambda_ * Z_norm

            pyro.deterministic(f"beta_{gene_name}", beta)
            Gbeta = G_gene.matmul(beta)
            mean = Gbeta.view(num_indivs)
            pyro.deterministic(f"mean_{gene_name}", mean)

            if torch.isnan(mean).any():
                logger.warning(f"NaNs in mean at gene {gene_name}")
                return "FAIL"

            with pyro.plate(f'data_{gene_name}', num_indivs):
                if self.simulated_parameters:
                    obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, 1),
                                      obs=self.simulated_parameters[gene_name]['mean'])
                else:
                    if self.alphas is not None:
                        std = torch.as_tensor(1.0 / np.sqrt(self.alphas[gene_name]), dtype=torch.float32, device=data.device)
                    else:
                        std = data.std
                    obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, std),
                                      obs=data.Y)
        return data


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

    if not config.get('no_filter', False):
        threshold = pyro.sample("threshold", dist.Beta(2.0, 20.0)).to(data.device)
    else:
        threshold = torch.as_tensor(0, dtype=torch.float32).to(data.device)
        pyro.deterministic("threshold", threshold)

    prior = torch.ones(data.num_anno, device=data.device) / data.num_anno
    tau = pyro.sample('tau', dist.Dirichlet(prior)).to(data.device)
    if mode == "nonlinear":
        tau2 = pyro.sample('tau2', dist.Dirichlet(prior)).to(data.device)
        simulated_parameters['tau1'] = tau
        simulated_parameters['tau2'] = tau2
    else:
        simulated_parameters['tau'] = tau

    simulated_parameters['w_g'] = w_g
    simulated_parameters['threshold'] = threshold
    simulated_parameters['mode'] = mode

    for gene_idx, gene_name in enumerate(data.gene_names):
        simulated_parameters[gene_name] = {}
        G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
        num_indivs = G_gene.shape[0]
        if mode == "nonlinear":
            lambda_ = F.relu((Z_gene.matmul(tau)) - threshold) * torch.exp(Z_gene.matmul(tau2)) * maf_weights_gene
        else:
            lambda_ = F.relu((Z_gene.matmul(tau)) - threshold) * maf_weights_gene
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
