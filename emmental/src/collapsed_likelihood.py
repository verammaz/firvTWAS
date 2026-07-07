"""
Torch collapsed likelihood for Emmental (integrate out beta).

See brr/MATH_EMMENTAL.md and brr/emmental_collapsed.py (numpy reference).
"""
from __future__ import annotations

import math
from typing import Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch

FormChoice = Literal["auto", "p", "n"]


# Minimum prior std for variants entering the collapsed integral (|lambda| > 0).
# Prevents division by zero in inv_var = 1/sigma^2 when sigma is exactly 0.
_SIGMA_FLOOR = 1e-6
# Internal linear algebra (A, b, solves) uses float64 for numerical stability.
_WORK_DTYPE = torch.float64
# Variants with lambda == 0 (gated out) are fixed at beta = 0 and skip the integral.
_LAM_ACTIVE_EPS = 0.0


def mu_sigma_from_lambda(
    lam: torch.Tensor,       # Per-variant annotation effect sizes (length p)
    w_g: torch.Tensor,       # Scalar gene-level scale (sampled or fixed to 1)
    rho_g: torch.Tensor,     # Scalar mixing weight in [0,1] (common vs rare split)
    *,
    no_rhog: bool = False,   # If True, ignore rho_g and use simpler mu/sigma formulas
    eps: float = _SIGMA_FLOOR,  # Unused here; kept for compatibility with models.py
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match models.py mu / sigma_sqrt (sigma returned is std dev, not variance)."""
    if no_rhog:
        # Ablation: all effect goes into mean; sigma tracks |w_g * lambda|
        mu = w_g * lam
        sigma = torch.abs(w_g * lam)
    else:
        # Standard Emmental split: rho_g fraction to mean, (1-rho_g) to prior std
        mu = rho_g * w_g * lam
        sigma = (1.0 - rho_g) * torch.abs(w_g * lam)
    return mu, sigma  # Both shape (p,): one value per variant


def _use_n_form(n: int, p: int) -> bool:
    """Use n×n sample-space algebra when there are more variants than individuals."""
    return p > n


def _resolve_form(n: int, p: int, form: FormChoice) -> FormChoice:
    if form == "auto":
        return "n" if _use_n_form(n, p) else "p"
    return form


def _promote_inputs(
    G: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    G64 = G.to(_WORK_DTYPE)
    y64 = y.to(_WORK_DTYPE).ravel()
    mu64 = mu.to(_WORK_DTYPE).ravel()
    sigma64 = torch.clamp(sigma.to(_WORK_DTYPE).ravel(), min=eps)
    alpha64 = alpha.to(_WORK_DTYPE).reshape([])
    n, p = G64.shape
    return G64, y64, mu64, sigma64, alpha64, n, p


def _omega_n_form(
    G64: torch.Tensor,
    sigma64: torch.Tensor,
    alpha64: torch.Tensor,
) -> torch.Tensor:
    """Omega = alpha^{-1} I + G diag(sigma^2) G^T (n x n)."""
    n = G64.shape[0]
    var = sigma64 * sigma64
    # (n, p) * (p,) -> scale columns by prior variance
    Omega = (1.0 / alpha64) * torch.eye(n, device=G64.device, dtype=G64.dtype)
    Omega = Omega + (G64 * var.unsqueeze(0)) @ G64.T
    jitter = 1e-10 * (torch.trace(Omega) / max(n, 1)).clamp(min=1.0)
    return Omega + jitter * torch.eye(n, device=G64.device, dtype=G64.dtype)


def active_variant_mask(lam: torch.Tensor, lam_eps: float = _LAM_ACTIVE_EPS) -> torch.Tensor:
    """True where variant contributes to collapsed likelihood (gate passed, lambda != 0)."""
    # Hard gate from annotation_lambda: inactive variants have lambda == 0 exactly
    return lam.abs() > lam_eps


def log_marginal_y_no_genotypes(
    y: torch.Tensor,      # Expression vector for one gene (n_samples,)
    alpha: torch.Tensor,  # Observation precision (scalar): y ~ N(0, alpha^{-1} I)
) -> torch.Tensor:
    """log p(y) when all beta are fixed at 0 (no active variants): y ~ N(0, alpha^{-1} I)."""
    y64 = y.to(_WORK_DTYPE).ravel()           # Promote to float64, ensure 1D
    alpha64 = alpha.to(_WORK_DTYPE).reshape([])  # Scalar precision as 0-d tensor
    n = y64.shape[0]                          # Number of individuals
    sse = (y64 * y64).sum()                   # Sum of squared errors vs mean 0
    log_2pi = math.log(2.0 * math.pi)         # Normalizing constant for Gaussian
    # log N(y; 0, alpha^{-1}): -0.5 * (alpha * ||y||^2 + n log(2pi) - n log alpha)
    logp = -0.5 * (alpha64 * sse + n * log_2pi - n * torch.log(alpha64))
    return logp.to(dtype=y.dtype)  # Cast back to input dtype for Pyro factor


def collapsed_beta_and_logp(
    G: torch.Tensor,       # Genotype matrix (n_samples x p)
    y: torch.Tensor,       # Expression (n_samples,)
    mu: torch.Tensor,      # Prior mean per variant (p,)
    sigma: torch.Tensor,   # Prior std per variant (p,)
    alpha: torch.Tensor,   # Observation precision (scalar)
    lam: torch.Tensor,     # Annotation lambdas (p,) — used only for gating mask
    lam_eps: float = _LAM_ACTIVE_EPS,   # Threshold for "active" variant
    sigma_floor: float = _SIGMA_FLOOR,  # Floor on sigma for active variants
    form: FormChoice = "auto",          # "auto" picks n-form when p > n, else p-form
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collapsed posterior mean and log p(y) with gated variants fixed at beta = 0.

    Variants with |lambda| <= lam_eps are excluded from the Gaussian integral (no 1/sigma^2
    spike). Returns beta_hat of shape (p,) with zeros on inactive columns.
    """
    p = G.shape[1]  # Number of variants for this gene
    active = active_variant_mask(lam, lam_eps)  # Boolean mask (p,)
    beta_full = torch.zeros(p, device=G.device, dtype=G.dtype)  # Full beta; inactive stay 0

    if not bool(active.any()):
        # No variants passed the gate: beta = 0 everywhere; likelihood is just p(y|beta=0)
        return beta_full, log_marginal_y_no_genotypes(y, alpha)

    if bool(active.all()):
        # All variants active: use full G without column subsetting (fast path)
        sigma_a = torch.clamp(sigma, min=sigma_floor)  # Avoid infinite prior precision
        beta_full = posterior_beta_mean(
            G, y, mu, sigma_a, alpha, eps=sigma_floor, form=form
        )
        logp = log_marginal_y(G, y, mu, sigma_a, alpha, eps=sigma_floor, form=form)
        return beta_full, logp

    # Mixed active/inactive: integrate only over active columns; inactive beta fixed at 0
    G_a = G[:, active]                    # Submatrix (n x p_active)
    mu_a = mu[active]                     # Prior means for active variants only
    sigma_a = torch.clamp(sigma[active], min=sigma_floor)
    beta_a = posterior_beta_mean(
        G_a, y, mu_a, sigma_a, alpha, eps=sigma_floor, form=form
    )
    beta_full[active] = beta_a.to(beta_full.dtype)  # Scatter active betas into full vector
    logp = log_marginal_y(G_a, y, mu_a, sigma_a, alpha, eps=sigma_floor, form=form)
    return beta_full, logp


def _linear_system(
    G64: torch.Tensor,
    y64: torch.Tensor,
    mu64: torch.Tensor,
    sigma64: torch.Tensor,
    alpha64: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build p×p posterior precision A and linear term b in float64 with diagonal jitter."""
    inv_var = 1.0 / (sigma64 * sigma64)
    A = torch.diag(inv_var) + alpha64 * (G64.T @ G64)
    b = inv_var * mu64 + alpha64 * (G64.T @ y64)

    p = A.shape[0]
    jitter = 1e-10 * (torch.trace(A) / max(p, 1)).clamp(min=1.0)
    A = A + jitter * torch.eye(p, device=A.device, dtype=A.dtype)
    return A, b, inv_var


def posterior_beta_mean_p_form(
    G: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = _SIGMA_FLOOR,
) -> torch.Tensor:
    """E[beta | y] via p×p solve: beta_hat = A^{-1} b."""
    G64, y64, mu64, sigma64, alpha64, _, _ = _promote_inputs(G, y, mu, sigma, alpha, eps)
    A, b, _ = _linear_system(G64, y64, mu64, sigma64, alpha64)
    beta_hat = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    return beta_hat.to(dtype=G.dtype)


def posterior_beta_mean_n_form(
    G: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = _SIGMA_FLOOR,
) -> torch.Tensor:
    """E[beta | y] via n×n solve: mu + diag(sigma^2) G^T Omega^{-1} (y - G mu)."""
    G64, y64, mu64, sigma64, alpha64, _, _ = _promote_inputs(G, y, mu, sigma, alpha, eps)
    resid = y64 - G64 @ mu64
    Omega = _omega_n_form(G64, sigma64, alpha64)
    v = torch.linalg.solve(Omega, resid.unsqueeze(-1)).squeeze(-1)
    var = sigma64 * sigma64
    beta_hat = mu64 + var * (G64.T @ v)
    return beta_hat.to(dtype=G.dtype)


def posterior_beta_mean(
    G: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = _SIGMA_FLOOR,
    form: FormChoice = "auto",
) -> torch.Tensor:
    """
    E[beta | y, mu, sigma, alpha] under beta ~ N(mu, diag(sigma^2)), y ~ N(G beta, alpha^{-1} I).

    Uses n-form when p > n (or when form="n"), otherwise p-form.
    """
    _, _, _, _, _, n, p = _promote_inputs(G, y, mu, sigma, alpha, eps)
    if _resolve_form(n, p, form) == "n":
        return posterior_beta_mean_n_form(G, y, mu, sigma, alpha, eps=eps)
    return posterior_beta_mean_p_form(G, y, mu, sigma, alpha, eps=eps)


def log_marginal_y_p_form(
    G: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = _SIGMA_FLOOR,
) -> torch.Tensor:
    """Scalar log p(y | G, mu, sigma, alpha) with beta integrated out (p×p algebra)."""
    G64, y64, mu64, sigma64, alpha64, n_samples, _ = _promote_inputs(
        G, y, mu, sigma, alpha, eps
    )
    A, b, inv_var = _linear_system(G64, y64, mu64, sigma64, alpha64)
    beta_hat = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    resid = y64 - G64 @ beta_hat
    sse = (resid * resid).sum()
    delta = beta_hat - mu64
    quad = alpha64 * sse + (delta * inv_var * delta).sum()

    sign, logdet_A = torch.linalg.slogdet(A)
    if sign <= 0:
        p = A.shape[0]
        A = A + (1e-6 * torch.eye(p, device=A.device, dtype=A.dtype))
        sign, logdet_A = torch.linalg.slogdet(A)
        if sign <= 0:
            raise torch.linalg.LinAlgError(
                "Posterior precision A is not positive definite after jitter."
            )

    logdet_omega = (
        -n_samples * torch.log(alpha64)
        + logdet_A
        + torch.log(sigma64 * sigma64).sum()
    )
    log_2pi = math.log(2.0 * math.pi)
    logp = -0.5 * (quad + logdet_omega + n_samples * log_2pi)
    return logp.to(dtype=G.dtype)


def log_marginal_y_n_form(
    G: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = _SIGMA_FLOOR,
) -> torch.Tensor:
    """Scalar log p(y | G, mu, sigma, alpha) with beta integrated out (n×n algebra)."""
    G64, y64, mu64, sigma64, alpha64, n_samples, _ = _promote_inputs(
        G, y, mu, sigma, alpha, eps
    )
    resid = y64 - G64 @ mu64
    Omega = _omega_n_form(G64, sigma64, alpha64)
    v = torch.linalg.solve(Omega, resid.unsqueeze(-1)).squeeze(-1)
    quad = (resid * v).sum()

    sign, logdet_omega = torch.linalg.slogdet(Omega)
    if sign <= 0:
        n = Omega.shape[0]
        Omega = Omega + (1e-6 * torch.eye(n, device=Omega.device, dtype=Omega.dtype))
        sign, logdet_omega = torch.linalg.slogdet(Omega)
        if sign <= 0:
            raise torch.linalg.LinAlgError(
                "Marginal covariance Omega is not positive definite after jitter."
            )

    log_2pi = math.log(2.0 * math.pi)
    logp = -0.5 * (quad + logdet_omega + n_samples * log_2pi)
    return logp.to(dtype=G.dtype)


def log_marginal_y(
    G: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = _SIGMA_FLOOR,
    form: FormChoice = "auto",
) -> torch.Tensor:
    """Scalar log p(y | G, mu, sigma, alpha) with beta integrated out."""
    _, _, _, _, _, n, p = _promote_inputs(G, y, mu, sigma, alpha, eps)
    if _resolve_form(n, p, form) == "n":
        return log_marginal_y_n_form(G, y, mu, sigma, alpha, eps=eps)
    return log_marginal_y_p_form(G, y, mu, sigma, alpha, eps=eps)
