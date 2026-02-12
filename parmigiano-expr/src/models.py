import sys
import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, AutoMultivariateNormal, AutoNormal
from pyro.infer import SVI, Trace_ELBO, RenyiELBO, Predictive
from pyro.optim import PyroLRScheduler
from tqdm import tqdm

class parmigiano_expression(PyroModule):
    def __init__(self, simulated_parameters=None):
        super().__init__()
        
    def forward(self, data, config):
        w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([data.num_genes]).to_event(1)).to(data.device)
        rho_g = pyro.sample("rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)).to(data.device)
        threshold = pyro.sample("threshold", dist.Beta(2.0, 20.0)).to(data.device)
        prior = torch.ones(data.num_anno, device=data.device) / data.num_anno
        tau = pyro.sample('tau', dist.Dirichlet(prior)).to(data.device)        
        for gene_idx, gene_name in enumerate(data.gene_names): 
            G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name) # Index gene-specific data
            num_indivs = G_gene.shape[0]
            lambda_ = F.relu((Z_gene.matmul(tau)) - threshold) # Compute functional weights
            lambda_ = lambda_ * maf_weights_gene * w_g[gene_idx]
            Z_norm = pyro.sample(f"Z_{gene_name}", dist.Normal(0, 1).expand([len(Z_gene)]).to_event(1)).to(data.device) # Sample variance 
            beta = rho_g[gene_idx] * lambda_ + (1 - rho_g[gene_idx]) * lambda_ * Z_norm # variant effect beta
            Gbeta = G_gene.matmul(beta) 
            mean = (Gbeta.reshape(-1, 1)).view(num_indivs) # Predict phenotype: Gbeta
            if torch.isnan(mean).any():
                print(f"NaNs in mean at gene {gene_name}")
                return "FAIL"
            with pyro.plate(f'data_{gene_name}', num_indivs): # Likelihood
                obs = pyro.sample(f'obs_{gene_name}', dist.Normal(mean, 1), obs=data.Y[gene_name.split("/")[1]])        
        return data

        