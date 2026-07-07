"""
Collapsed Emmental Pyro models: integrate out per-gene beta (no Z_{gene} sites).

Observation enters via pyro.factor(log p(y | G, mu, sigma, alpha)).
Posterior beta means are exposed as deterministic sites for downstream R2 / export.
"""
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule

from collapsed_likelihood import collapsed_beta_and_logp, mu_sigma_from_lambda
from models import annotation_lambda, observation_alpha, observation_y
from utils import get_logger


class EmmentalJointCollapsed(PyroModule):
    """Joint model with collapsed Gaussian likelihood per gene (no Z sites)."""

    def __init__(self, simulated_parameters=None):
        super().__init__()
        # If set, y and alpha come from simulation truth instead of real data
        self.simulated_parameters = simulated_parameters

    def forward(self, data, config):
        """Pyro model: sample global params, then one collapsed likelihood factor per gene."""
        logger = get_logger()

        # --- Global latent variables (shared across all genes) ---

        if not config.get("no_wg", False):
            w_g = pyro.sample(
                "w_g", dist.Normal(0, 1).expand([data.num_genes]).to_event(1)
            ).to(data.device)
        else:
            w_g = torch.ones(data.num_genes, device=data.device)

        if config.get("no_rhog", False):
            rho_g = pyro.deterministic("rho_g", torch.zeros(data.num_genes, device=data.device))
        else:
            rho_g = pyro.sample(
                "rho_g", dist.Beta(0.5, 0.5).expand([data.num_genes]).to_event(1)
            ).to(data.device)

        # Prior on annotation gate threshold T (controls which variants are "active").
        # When no_T is set, fix T=0 deterministically so all |Z·τ₁| pass the gate.
        if config.get("no_T", False):
            threshold = pyro.deterministic("threshold", torch.zeros((), device=data.device))
        else:
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

        no_rhog = bool(config.get("no_rhog", False))

        # --- Per-gene collapsed likelihood (beta integrated out analytically) ---

        for gene_idx, gene_name in enumerate(data.gene_names):
            # Load genotype matrix G, annotation features Z, MAF weights for this gene
            G_gene, Z_gene, maf_weights_gene = data.get_gene_data(gene_name)
            # Per-variant lambda from annotations, tau1/tau2, and hard gate at threshold
            lam = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                threshold,
                tau1=tau1,
                tau2=tau2,
                logger=logger,
            )
            # Convert lambda + w_g, rho_g into Gaussian prior mean and std for each beta_j
            mu, sigma = mu_sigma_from_lambda(
                lam,
                w_g[gene_idx],
                rho_g[gene_idx],
                no_rhog=no_rhog,
                eps=0.0,
            )
            # Observed expression y_g and observation precision alpha_g
            y_gene = observation_y(data, gene_name, self.simulated_parameters)
            alpha = observation_alpha(
                data, gene_name, config, self.simulated_parameters, default_std=0.5
            )

            # Analytic: posterior mean beta_hat and log p(y | G, mu, sigma, alpha)
            beta_hat, logp = collapsed_beta_and_logp(
                G_gene, y_gene, mu, sigma, alpha, lam
            )
            # Expose quantities for Predictive / downstream export (not sampled latents)
            pyro.deterministic(f"mu_{gene_name}", mu)
            pyro.deterministic(f"sigma_{gene_name}", sigma**2)  # store variance for compatibility
            pyro.deterministic(f"beta_{gene_name}", beta_hat)
            mean = G_gene.matmul(beta_hat)  # Predicted expression G @ beta_hat
            pyro.deterministic(f"mean_{gene_name}", mean.view(-1))

            if torch.isnan(mean).any():
                raise ValueError(f"NaNs in mean at gene {gene_name}")
            # Pyro site names cannot contain '/'; gene paths look like chr1/ENSG...
            safe = gene_name.replace("/", "_")
            # Collapsed data likelihood: adds logp to ELBO (no explicit beta sample site)
            pyro.factor(f"data_{safe}", logp)

        return data


class EmmentalPerGeneCollapsed(PyroModule):
    """Per-gene collapsed model; tau and threshold fixed from joint."""

    def __init__(self, config, data, simulated_parameters=None):
        super().__init__()
        self.simulated_parameters = simulated_parameters
        self.config = config
        self.data = data
        # tau1, tau2, threshold learned in joint stage — frozen buffers here
        self.register_buffer("_fixed_tau1", data.tau1)
        self.register_buffer("_fixed_tau2", data.tau2)
        self.register_buffer("_fixed_threshold", data.threshold)

    def forward(self, data, config=None):
        """Second-stage model: only w_g and rho_g are inferred; tau/T fixed."""
        import utils

        logger = utils.get_logger()
        if config is None:
            config = self.config
        self.data = data  # Allow passing updated DataTensors (e.g. train vs test)

        # If no_T is set, override stored threshold and treat all variants as passing the gate.
        if config.get("no_T", False):
            th_use = torch.zeros((), device=self.data.device)
        else:
            th_use = self._fixed_threshold
        tau1_use, tau2_use = self._fixed_tau1, self._fixed_tau2

        if not config.get("no_wg", False):
            w_g = pyro.sample(
                "w_g", dist.Normal(0, 1).expand([self.data.num_genes]).to_event(1)
            ).to(self.data.device)
        else:
            w_g = torch.ones(self.data.num_genes, device=self.data.device)

        if config.get("no_rhog", False):
            rho_g = pyro.deterministic("rho_g", torch.zeros(self.data.num_genes, device=self.data.device))
        else:
            rho_g = pyro.sample(
                "rho_g", dist.Beta(0.5, 0.5).expand([self.data.num_genes]).to_event(1)
            ).to(self.data.device)

        no_rhog = bool(config.get("no_rhog", False))

        for gene_idx, gene_name in enumerate(self.data.gene_names):
            G_gene, Z_gene, maf_weights_gene = self.data.get_gene_data(gene_name)
            lam = annotation_lambda(
                Z_gene,
                maf_weights_gene,
                th_use,           # Fixed threshold from joint fit
                tau1=tau1_use,    # Fixed tau1 from joint fit
                tau2=tau2_use,    # Fixed tau2 from joint fit
            )
            mu, sigma = mu_sigma_from_lambda(
                lam,
                w_g[gene_idx],
                rho_g[gene_idx],
                no_rhog=no_rhog,
                eps=0.0,
            )
            y_gene = observation_y(self.data, gene_name, self.simulated_parameters)
            alpha = observation_alpha(
                self.data, gene_name, config, self.simulated_parameters, default_std=0.5
            )

            beta_hat, logp = collapsed_beta_and_logp(
                G_gene, y_gene, mu, sigma, alpha, lam
            )
            pyro.deterministic(f"beta_{gene_name}", beta_hat)
            mean = G_gene.matmul(beta_hat)
            pyro.deterministic(f"mean_{gene_name}", mean.view(-1))

            if torch.isnan(mean).any():
                raise ValueError(f"NaNs in mean at gene {gene_name}")
            safe = gene_name.replace("/", "_")
            pyro.factor(f"data_{safe}", logp)

        return self.data
