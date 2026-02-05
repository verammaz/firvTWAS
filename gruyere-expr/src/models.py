import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import torch


class gruyere(PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, data, params):
        num_indivs = data.G['train'].shape[0]

        # Global weights for genes
        w_g = pyro.sample('w_g', dist.Normal(0, 1).expand([data.num_genes]).to_event(1))

        # Annotation weights
        prior = torch.ones(data.num_anno) / data.num_anno
        tau = pyro.sample('tau', dist.Dirichlet(prior))

        # Observation noise (shared across genes)
        sigma_y = pyro.sample('sigma_y', dist.HalfCauchy(1.0))

        for gene in range(data.num_genes):
            # Covariate effects
            alpha = pyro.sample(
                f'alpha_{data.genes[gene]}', dist.Normal(0, 1).expand([data.num_cov]).to_event(1)
            )

            # Genetic effects
            beta = (
                (data.Z[data.gene_indices == gene].T * data.maf_weights[data.gene_indices == gene])
                .T.matmul(tau)
            ) * w_g[gene]

            # Genetic score for each individual
            Gbeta = data.G['train'][:, data.gene_indices == gene].matmul(beta)

            # Linear predictor
            mean = torch.matmul(data.X['train'], alpha).reshape(-1, 1) + Gbeta.reshape(-1, 1)
            mean = mean.view(num_indivs)
           
            # correct observation: pick only the column for this gene
            obs_col = data.Y['train'][:, gene]   # shape: [num_indivs]

            # Continuous outcome likelihood
            with pyro.plate(f'data_{data.genes[gene]}', num_indivs):
                pyro.sample(
                    f'obs_{data.genes[gene]}',
                    dist.Normal(mean, sigma_y),
                    obs=data.Y['train'][:, gene],
                )

        return data
