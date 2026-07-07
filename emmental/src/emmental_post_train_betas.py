#!/usr/bin/env python3
"""
Extend a **common-only** Emmental fit to the **full** variant panel (common + rare).

Writes per-variant β panels (μ, G_full / G_common / G_rare posteriors, saved training common β)
and gene-level ``summary.csv`` (metadata + β QC). All R² assemblies live in
``train_r2_scores.csv`` and ``test_r2_scores.csv`` (one file per split; no ``*_mixed`` files).

Plotting is done offline via ``param_plots.py`` / ``param_plots.ipynb``.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import r2_score

import load_data
import utils
from collapsed_likelihood import collapsed_beta_and_logp
from models import annotation_lambda

COMMON_BETA_SOURCES = ("mu", "full", "common", "train")
RARE_BETA_SOURCES = ("mu", "full", "rare")


def _parse_gene_from_beta_filename(filename: str, chr_dir: Optional[str] = None) -> Optional[str]:
    """
  Parse gene id from per-gene beta CSV name.

  Supports:
    - ``chr1_ENSG000..._beta.csv.gz`` (slash replaced in save_outputs)
    - ``ENSG000..._beta.csv.gz`` under ``pergene/chr1/`` (per-gene mode keys are ENSG-only)
    """
    base = os.path.basename(filename)
    if base.endswith("_beta.csv.gz"):
        stem = base[: -len("_beta.csv.gz")]
    elif base.endswith("_beta.csv"):
        stem = base[: -len("_beta.csv")]
    else:
        return None
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("chr"):
        return f"{parts[0]}/{parts[1]}"
    if stem.startswith("ENSG") and chr_dir:
        chr_stub = os.path.basename(chr_dir.rstrip("/"))
        if not chr_stub.startswith("chr"):
            chr_stub = f"chr{chr_stub}"
        return f"{chr_stub}/{stem}"
    return None


def _load_gene_list(path: str) -> set:
    genes = set()
    with open(path) as f:
        for line in f:
            g = line.strip()
            if g and not g.startswith("#"):
                genes.add(g)
    return genes


def _load_covariates_scaled(config: Dict[str, Any]) -> pd.DataFrame:
    cov = pd.read_csv(config["covariates_path"], sep="\t").set_index("sample_id")
    covariate_cols = [
        "biological_sex", "eas_prob", "afr_prob", "amr_prob", "sas_prob", "eur_prob",
        "tissue", "age", "pc1", "pc2", "pc3", "pc4", "pc5", "cohort",
        "rna_lib_prep_type", "rna_strandedness", "astrocyte", "endothelial_cell",
        "excitatory_neuron", "inhibitory_neuron", "microglia", "oligodendrocyte",
        "oligodendrocyte_progenitor_cell", "others", "pericyte",
    ]
    return utils.preprocess_covariates(cov, covariate_cols)


class _ExpressionResidualizer:
    """Load expression once; residualize one gene at a time (avoids 200× full TPM reads)."""

    def __init__(
        self,
        config: Dict[str, Any],
        cov_scaled: pd.DataFrame,
        device: torch.device,
    ) -> None:
        tpm = pd.read_csv(config["expression_path"], sep="\t").set_index("feature")
        self.tpm = tpm[cov_scaled.index]
        self.cov_scaled = cov_scaled
        self.device = device

    def residualize(self, gene_name: str) -> np.ndarray:
        chr_gene = utils.get_chr_gene(self.tpm, [gene_name])
        if chr_gene.empty:
            raise ValueError(f"Gene {gene_name} not in expression matrix")
        feat = chr_gene["feature"].iloc[0]
        expr = utils.scale_tpm_matrix(self.tpm.loc[[feat]]).loc[feat]
        resid_out = utils.residualize_expression_single_gene(
            expr, self.cov_scaled, device=self.device
        )
        if isinstance(resid_out, torch.Tensor):
            common_samples = expr.index.intersection(self.cov_scaled.index)
            resid = pd.Series(resid_out.detach().cpu().numpy(), index=common_samples)
        else:
            resid = resid_out
        return resid.reindex(self.cov_scaled.index).to_numpy(dtype=np.float64)


def _residualize_gene_y(
    config: Dict[str, Any],
    gene_name: str,
    cov_scaled: pd.DataFrame,
    device: torch.device,
    expr_cache: Optional[_ExpressionResidualizer] = None,
) -> np.ndarray:
    if expr_cache is not None:
        return expr_cache.residualize(gene_name)
    return _ExpressionResidualizer(config, cov_scaled, device).residualize(gene_name)


def _r2_common_only_col(common_src: str) -> str:
    return f"r2_common_only__c_{common_src}"


def _r2_common_plus_rare_col(common_src: str, rare_src: str) -> str:
    return f"r2_common_plus_rare__c_{common_src}__r_{rare_src}"


def _all_r2_score_columns() -> List[str]:
    cols = [_r2_common_only_col(cs) for cs in COMMON_BETA_SOURCES]
    cols.extend(
        _r2_common_plus_rare_col(cs, rs)
        for cs in COMMON_BETA_SOURCES
        for rs in RARE_BETA_SOURCES
    )
    return cols


def _save_r2_score_tables(
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    """Write train/test R² tables (all β assemblies; one file per split)."""
    pd.DataFrame(train_rows).to_csv(os.path.join(out_dir, "train_r2_scores.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(out_dir, "test_r2_scores.csv"), index=False)


def _r2_threshold_summary_from_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split, df in (("train", train_df), ("test", test_df)):
        for col in _all_r2_score_columns():
            if col not in df.columns:
                continue
            s = df[col].dropna()
            key = f"{split}_{col}"
            out[key] = {
                "prop_gt_0.01": float((s > 0.01).mean()) if len(s) else float("nan"),
                "prop_gt_0.1": float((s > 0.1).mean()) if len(s) else float("nan"),
                "mean_r2": float(s.mean()) if len(s) else float("nan"),
                "n_genes": int(len(s)),
            }
    return out


def _r2_predict(
    G: pd.DataFrame, y: np.ndarray, beta: np.ndarray, mask: np.ndarray
) -> float:
    if mask.sum() < 1 or np.var(y) < 1e-12:
        return float("nan")
    cols = G.columns[mask]
    b = np.asarray(beta, dtype=np.float64)[mask]
    if not np.isfinite(b).all():
        b = np.nan_to_num(b, nan=0.0)
    pred = G[cols].values @ b
    return float(r2_score(y, pred))


def _maf_threshold_from_config(config: Dict[str, Any], default: float) -> float:
    v = config.get("maf_threshold")
    if v is None:
        return default
    if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
        return default
    return float(v)


def _load_chr_config(chr_dir: str) -> Dict[str, Any]:
    path = os.path.join(chr_dir, "config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing per-gene config: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML in {path}")
    return cfg


def _stat_mean_from_posterior_entry(entry: Any) -> float:
    """Extract scalar mean from a ``save_results`` / per-gene npz entry (dict with 'mean' array)."""
    if isinstance(entry, dict) and "mean" in entry:
        return float(np.asarray(entry["mean"], dtype=np.float64).ravel()[0])
    if isinstance(entry, np.ndarray):
        return float(np.asarray(entry, dtype=np.float64).ravel()[0])
    return float(entry)


def _load_rho_w_from_posterior_npz(
    npz_path: str,
    gene_key: str,
    logger,
) -> Optional[Tuple[float, float]]:
    """
    Per-gene ``save_results`` stores keys like ``ENSG00001::rho_g`` and ``ENSG00001::w_g``,
    each a dict ``{'mean': ..., 'std': ...}``.
    """
    if not os.path.isfile(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    rho_key = f"{gene_key}::rho_g"
    w_key = f"{gene_key}::w_g"
    if rho_key not in data.files:
        return None
    rho = _stat_mean_from_posterior_entry(data[rho_key].item())
    w = _stat_mean_from_posterior_entry(data[w_key].item()) if w_key in data.files else None
    if w is None:
        logger.warning("Missing %s in %s", w_key, npz_path)
    return rho, w


def _mu_sigma_sqrt_np(
    rho: float,
    w: float,
    lam_np: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Match ``EmmentalJoint`` / ``EmmentalPerGene`` μ and σ definitions."""
    if cfg.get("no_rhog", False):
        mu = w * lam_np
        sigma_sqrt = np.abs(w * lam_np)
    else:
        mu = rho * w * lam_np
        sigma_sqrt = (1.0 - rho) * np.abs(w * lam_np)
    return mu, sigma_sqrt


def _gene_brr_alpha(
    gene_key: str,
    brr_alphas: Optional[Dict[str, float]],
    default_std: float = 0.5,
) -> float:
    """Observation precision α for collapsed posterior (matches ``observation_alpha`` default)."""
    if brr_alphas:
        a = brr_alphas.get(gene_key)
        if a is not None and np.isfinite(a) and float(a) > 0:
            return float(a)
    return 1.0 / (default_std ** 2)


def _compute_posterior_beta_hat(
    G_train: pd.DataFrame,
    y_train: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    lam_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Collapsed posterior E[β|y] on the training panel (same units as per-gene ``beta_mean``)."""
    G_t = torch.as_tensor(np.asarray(G_train.values, dtype=np.float32), device=device)
    y_t = torch.as_tensor(np.asarray(y_train, dtype=np.float32), device=device)
    mu_t = torch.as_tensor(mu.astype(np.float32), device=device)
    sigma_t = torch.as_tensor(sigma.astype(np.float32), device=device)
    alpha_t = torch.as_tensor(float(alpha), dtype=torch.float32, device=device)
    lam_t = torch.as_tensor(lam_np.astype(np.float32), device=device)
    beta_hat, _ = collapsed_beta_and_logp(G_t, y_t, mu_t, sigma_t, alpha_t, lam_t)
    return beta_hat.detach().cpu().numpy().astype(np.float64)


def _scatter_decoupled_posterior_beta(
    G_train: pd.DataFrame,
    y_train: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    lam_np: np.ndarray,
    variant_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Collapsed posterior on one stratum only (common or rare columns); NaN off-stratum.

    Matches per-gene training when that stratum was fit in isolation (G_common for saved β).
    """
    n = len(variant_mask)
    out = np.full(n, np.nan, dtype=np.float64)
    if not bool(np.any(variant_mask)):
        return out
    cols = G_train.columns[variant_mask]
    beta_sub = _compute_posterior_beta_hat(
        G_train.loc[:, cols],
        y_train,
        mu[variant_mask],
        sigma[variant_mask],
        alpha,
        lam_np[variant_mask],
        device,
    )
    out[variant_mask] = beta_sub
    return out


def _beta_for_source(
    source: str,
    *,
    mu_beta: np.ndarray,
    beta_full: np.ndarray,
    beta_common: np.ndarray,
    beta_rare: np.ndarray,
    beta_train: np.ndarray,
) -> np.ndarray:
    if source == "mu":
        return mu_beta
    if source == "full":
        return beta_full
    if source == "common":
        return beta_common
    if source == "train":
        return beta_train
    if source == "rare":
        return beta_rare
    raise ValueError(f"unknown beta source: {source}")


def _assemble_beta_vector(
    common_mask: np.ndarray,
    common_src: str,
    rare_src: str,
    *,
    mu_beta: np.ndarray,
    beta_full: np.ndarray,
    beta_common: np.ndarray,
    beta_rare: np.ndarray,
    beta_train: np.ndarray,
) -> np.ndarray:
    """Full-panel β from independent per-stratum source choices."""
    beta_kw = {
        "mu_beta": mu_beta,
        "beta_full": beta_full,
        "beta_common": beta_common,
        "beta_rare": beta_rare,
        "beta_train": beta_train,
    }
    out = np.zeros(len(common_mask), dtype=np.float64)
    out[common_mask] = _beta_for_source(common_src, **beta_kw)[common_mask]
    out[~common_mask] = _beta_for_source(rare_src, **beta_kw)[~common_mask]
    return out


def _compute_r2_grid(
    G_tr: pd.DataFrame,
    G_te: pd.DataFrame,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    common_mask: np.ndarray,
    *,
    mu_beta: np.ndarray,
    beta_full: np.ndarray,
    beta_common: np.ndarray,
    beta_rare: np.ndarray,
    beta_train: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """All common-only and common+rare R² assemblies for train and test."""
    all_mask = np.ones(len(common_mask), dtype=bool)
    beta_kw = {
        "mu_beta": mu_beta,
        "beta_full": beta_full,
        "beta_common": beta_common,
        "beta_rare": beta_rare,
        "beta_train": beta_train,
    }
    train_out: Dict[str, float] = {}
    test_out: Dict[str, float] = {}
    for common_src in COMMON_BETA_SOURCES:
        col = _r2_common_only_col(common_src)
        beta_vec = _beta_for_source(common_src, **beta_kw)
        train_out[col] = _r2_predict(G_tr, y_tr, beta_vec, common_mask)
        test_out[col] = _r2_predict(G_te, y_te, beta_vec, common_mask)
    for common_src in COMMON_BETA_SOURCES:
        for rare_src in RARE_BETA_SOURCES:
            col = _r2_common_plus_rare_col(common_src, rare_src)
            beta_vec = _assemble_beta_vector(
                common_mask, common_src, rare_src, **beta_kw
            )
            train_out[col] = _r2_predict(G_tr, y_tr, beta_vec, all_mask)
            test_out[col] = _r2_predict(G_te, y_te, beta_vec, all_mask)
    return train_out, test_out


def _build_beta_common_train_posterior(
    common_mask: np.ndarray,
    beta_np: np.ndarray,
) -> np.ndarray:
    """Saved training β on aligned common variants; NaN on rare and misaligned common."""
    out = np.full(len(common_mask), np.nan, dtype=np.float64)
    has_beta = common_mask & np.isfinite(beta_np)
    out[has_beta] = beta_np[has_beta]
    return out


def _beta_pair_metrics(
    saved: np.ndarray,
    posterior: np.ndarray,
    mask: np.ndarray,
    *,
    metric_suffix: str = "",
) -> Dict[str, Any]:
    """Compare saved vs recalculated β on a variant mask."""
    sfx = metric_suffix
    m = mask & np.isfinite(saved) & np.isfinite(posterior)
    n = int(m.sum())
    out: Dict[str, Any] = {
        f"common_beta_pearson{sfx}": float("nan"),
        f"common_beta_spearman{sfx}": float("nan"),
        f"common_beta_rmse{sfx}": float("nan"),
        f"common_beta_median_abs_diff{sfx}": float("nan"),
        f"n_common_beta_compared{sfx}": n,
    }
    if n < 2:
        return out
    x = saved[m].astype(np.float64)
    y = posterior[m].astype(np.float64)
    out[f"common_beta_pearson{sfx}"] = float(np.corrcoef(x, y)[0, 1])
    out[f"common_beta_spearman{sfx}"] = float(
        pd.Series(x).corr(pd.Series(y), method="spearman")
    )
    diff = x - y
    out[f"common_beta_rmse{sfx}"] = float(np.sqrt(np.mean(diff * diff)))
    out[f"common_beta_median_abs_diff{sfx}"] = float(np.median(np.abs(diff)))
    return out


def _gene_key(gene_name: str) -> str:
    return gene_name.split("/")[-1] if "/" in gene_name else gene_name


def _normalize_chr_tag(chromosome: str) -> str:
    """``10`` / ``chr10`` → ``chr10``."""
    s = str(chromosome).strip()
    if s.lower().startswith("chr"):
        return s if s.startswith("chr") else "chr" + s[3:]
    return f"chr{s}"


def _cleaned_g_column_id(gene_name: str, raw_variant_id: str) -> str:
    """Map raw genotype ID (chr:pos_a1_a2) to ``load_genes`` column label (ENSG_chr:pos)."""
    prefixed = f"{gene_name}_{raw_variant_id}"
    return (
        pd.Index([prefixed])
        .str.split("_")
        .str[0:2]
        .str.join("_")
        .str.split("/")
        .str[-1][0]
    )


def _map_beta_variant_id_to_g_column(gene_name: str, variant_id: str) -> str:
    """
    Map a beta CSV ``variant_id_G`` to ``load_genes`` column labels.

    Per-gene outputs store cleaned IDs (``ENSG000…_chr:pos``) matching ``G.columns``.
    Joint outputs store raw genotype IDs (``chr:pos_a1_a2``) and need ``_cleaned_g_column_id``.
    """
    vid = str(variant_id)
    gene_key = _gene_key(gene_name)
    if vid.startswith(f"{gene_key}_"):
        return vid
    return _cleaned_g_column_id(gene_name, vid)


def _series_reindex_to_columns(
    values: np.ndarray, index: pd.Index, G_columns: pd.Index
) -> Tuple[np.ndarray, int]:
    """Reindex beta values onto ``G.columns``; average duplicate index labels if needed."""
    s = pd.Series(values, index=index)
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    aligned = s.reindex(G_columns)
    n = int(np.isfinite(aligned.values).sum())
    return aligned.to_numpy(dtype=np.float64), n


def _chr_pos_from_g_column(col: str) -> str:
    return col.split("_", 1)[1]


def _chr_pos_from_raw_variant_id(raw: str) -> str:
    raw = str(raw)
    if raw.startswith("ENSG") and "_" in raw:
        return raw.split("_", 1)[1]
    return raw.split("_")[0]


def _load_gene_G_Z(
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Load genotypes for post-train.

    Returns ``(G_model, G_raw, Z, variant_ids_G, variant_ids_Z)`` where ``G_model`` is
    column-normalized when ``normalize_G`` is set and ``G_raw`` is always unnormalized
    (for MAF / MAF weights).
    """
    G_raw, Z, variant_ids_G, variant_ids_Z = load_data.load_genes(cfg)
    if cfg.get("normalize_G", False):
        G_model = utils.normalize_G(G_raw)
    else:
        G_model = G_raw
    return G_model, G_raw, Z, variant_ids_G, variant_ids_Z


def _joint_training_variant_columns(
    cfg: Dict[str, Any],
    gene_name: str,
    train_idx: np.ndarray,
    beta_n: Optional[int] = None,
) -> List[str]:
    """
    Columns kept by joint training's MAF filter (for aligning ``beta_samples`` CSV row order).

    When ``normalize_G`` is true, legacy joint runs filtered on **normalized** dosages; we try
    that first when ``beta_n`` matches. Newer training (``G_raw_for_maf`` in ``from_pandas``)
    uses raw dosages — we fall back to raw MAF when row counts do not match.
    """
    cfg_g = dict(cfg)
    cfg_g["genes"] = [gene_name]
    G_model, G_raw, _, _, _ = _load_gene_G_Z(cfg_g)
    maf_thr = _maf_threshold_from_config(cfg_g, 0.01)

    def _cols_for(G_ref: pd.DataFrame) -> List[str]:
        maf = load_data.variant_maf_series(G_ref.iloc[train_idx])
        return maf[maf >= maf_thr].index.tolist()

    candidates: List[Tuple[str, List[str]]] = []
    if cfg.get("normalize_G", False):
        candidates.append(("normalized", _cols_for(G_model)))
    candidates.append(("raw", _cols_for(G_raw)))

    if beta_n is not None:
        for _label, cols in candidates:
            if len(cols) == beta_n:
                return cols
    return candidates[0][1]


def _align_beta_to_G_columns(
    beta_df: pd.DataFrame,
    gene_name: str,
    G_columns: pd.Index,
    cfg: Dict[str, Any],
    train_idx: np.ndarray,
    logger,
) -> Tuple[np.ndarray, str]:
    """
    Align ``beta_mean`` to full-panel ``G.columns``.

    Joint ``beta_samples`` CSVs store raw ``variant_id_G`` strings, while ``load_genes`` uses
    cleaned column names (``ENSG_chr:pos``). After joint MAF filtering, saved ``variant_id_G``
    labels can also be misaligned with row order (see ``DataTensors.from_pandas``); row order
    still matches MAF-filtered training columns, so we fall back to positional alignment.
    """
    n = len(G_columns)
    if beta_df is None or beta_df.empty or "beta_mean" not in beta_df.columns:
        return np.full(n, np.nan, dtype=np.float64), "empty"

    values = beta_df["beta_mean"].astype(float).values
    best = np.full(n, np.nan, dtype=np.float64)
    best_method = "none"
    best_n = 0

    if "variant_id_G" in beta_df.columns:
        mapped = [_map_beta_variant_id_to_g_column(gene_name, v) for v in beta_df["variant_id_G"]]
        aligned_arr, n_clean = _series_reindex_to_columns(values, pd.Index(mapped), G_columns)
        if n_clean > best_n:
            best = aligned_arr
            best_method = "variant_id_mapped"
            best_n = n_clean

        by_pos = pd.Series(
            values,
            index=[_chr_pos_from_raw_variant_id(v) for v in beta_df["variant_id_G"]],
        )
        if by_pos.index.has_duplicates:
            by_pos = by_pos.groupby(level=0).mean()
        g_pos = [_chr_pos_from_g_column(c) for c in G_columns]
        aligned_pos = np.array([by_pos.get(p, np.nan) for p in g_pos], dtype=np.float64)
        n_pos = int(np.isfinite(aligned_pos).sum())
        if n_pos > best_n:
            best = aligned_pos
            best_method = "chr_pos"
            best_n = n_pos

    train_cols = _joint_training_variant_columns(
        cfg, gene_name, train_idx, beta_n=len(beta_df)
    )
    if train_cols and len(beta_df) == len(train_cols):
        aligned_arr, n_ord = _series_reindex_to_columns(values, pd.Index(train_cols), G_columns)
        if n_ord > best_n:
            best = aligned_arr
            best_method = "training_column_order"
            best_n = n_ord
            logger.debug(
                "%s: beta aligned by training column order (%d/%d variants).",
                gene_name,
                n_ord,
                len(train_cols),
            )

    return best, best_method


def _aggregate_joint_wg_rhog(
    joint_root: str,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Mean ``w_g`` / ``rho_g`` per ENSG across joint refit runs."""
    try:
        run_dirs = load_data.discover_joint_run_directories(joint_root)
    except FileNotFoundError:
        return None, None

    w_parts: List[pd.Series] = []
    r_parts: List[pd.Series] = []
    for rd in run_dirs:
        wp = os.path.join(rd, "w_g.csv")
        rp = os.path.join(rd, "rho_g.csv")
        if os.path.isfile(wp):
            d = pd.read_csv(wp).set_index("gene")["w_g_mean"].astype(float)
            w_parts.append(d)
        if os.path.isfile(rp):
            d = pd.read_csv(rp).set_index("gene")["rho_g_mean"].astype(float)
            r_parts.append(d)

    w_mean = pd.concat(w_parts, axis=1).mean(axis=1) if w_parts else None
    r_mean = pd.concat(r_parts, axis=1).mean(axis=1) if r_parts else None
    return w_mean, r_mean


def _load_rho_w_from_joint(
    joint_dir: str,
    gene_key: str,
    wg_mean: Optional[pd.Series],
    rh_mean: Optional[pd.Series],
    default_rho: float,
    default_w: float,
    no_wg: bool,
    logger,
    no_rhog: bool = False,
) -> Tuple[float, float]:
    """Posterior means from pre-aggregated joint ``w_g.csv`` / ``rho_g.csv`` means."""
    rho, w = float(default_rho), float(default_w)
    if rh_mean is not None and gene_key in rh_mean.index:
        rho = float(rh_mean.loc[gene_key])
    elif rh_mean is not None:
        logger.warning("No rho_g for %s in joint aggregates — using --rho_g=%s", gene_key, default_rho)
    if not no_wg:
        if wg_mean is not None and gene_key in wg_mean.index:
            w = float(wg_mean.loc[gene_key])
        elif wg_mean is not None:
            logger.warning("No w_g for %s in joint aggregates — using --w_g=%s", gene_key, default_w)
    if no_wg:
        w = 1.0
    if no_rhog:
        rho = 0.0
    return rho, w


def _load_joint_beta_df(
    joint_dir: str,
    gene_name: str,
    cfg: Dict[str, Any],
    train_idx: np.ndarray,
    avg_across_runs: bool = True,
    joint_run: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Load common-only ``beta_mean`` from joint ``beta_samples/{ENSG}_beta.csv.gz``.

    Averages across refits by **training column order** (not ``variant_id_G`` labels, which can
    be misaligned after joint MAF filtering).
    """
    gene_key = _gene_key(gene_name)
    try:
        run_dirs = load_data.discover_joint_run_directories(joint_dir)
    except FileNotFoundError:
        return None

    if joint_run is not None:
        run_dirs = [rd for rd in run_dirs if os.path.basename(rd) == f"run_{joint_run}"]
        if not run_dirs:
            return None

    series_list: List[pd.Series] = []
    last_df: Optional[pd.DataFrame] = None
    for rd in run_dirs:
        path = os.path.join(rd, "beta_samples", f"{gene_key}_beta.csv.gz")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, compression="infer")
        last_df = df
        train_cols = _joint_training_variant_columns(
            cfg, gene_name, train_idx, beta_n=len(df)
        )
        if train_cols and len(df) == len(train_cols):
            series_list.append(pd.Series(df["beta_mean"].values, index=train_cols))
        else:
            series_list.append(df.set_index("variant_id_G")["beta_mean"])

    if not series_list:
        return None
    if joint_run is not None or (not avg_across_runs) or len(series_list) == 1:
        return last_df

    mean_beta = pd.concat(series_list, axis=1).mean(axis=1)
    out = pd.DataFrame({"beta_mean": mean_beta.values})
    if isinstance(mean_beta.index, pd.Index) and mean_beta.index.name != "variant_id_G":
        out["training_column"] = mean_beta.index.astype(str)
    elif last_df is not None and "variant_id_G" in last_df.columns:
        out["variant_id_G"] = last_df["variant_id_G"].values[: len(out)]
    return out


def _load_pergene_root_config(pergene_root: str) -> Dict[str, Any]:
    """Load ``pergene/config.yaml`` (single file at experiment root)."""
    path = os.path.join(os.path.abspath(pergene_root), "config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing per-gene config: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML in {path}")
    return cfg


def _discover_pergene_chr_output_dirs(pergene_root: str) -> List[Tuple[str, List[str]]]:
    """
    Chromosome -> list of output dirs containing ``beta_samples``.

    Supports ``pergene/run_N/chrK/`` (refit) and legacy ``pergene/chrK/`` (single fit).
    """
    pergene_root = os.path.abspath(pergene_root)
    chr_map: Dict[str, List[str]] = {}

    for name in sorted(os.listdir(pergene_root)):
        if not name.startswith("run_"):
            continue
        run_root = os.path.join(pergene_root, name)
        if not os.path.isdir(run_root):
            continue
        for chr_name in sorted(os.listdir(run_root)):
            if not chr_name.startswith("chr"):
                continue
            chr_path = os.path.join(run_root, chr_name)
            if os.path.isdir(os.path.join(chr_path, "beta_samples")):
                chr_map.setdefault(chr_name, []).append(chr_path)

    for chr_name in sorted(os.listdir(pergene_root)):
        if not chr_name.startswith("chr"):
            continue
        chr_path = os.path.join(pergene_root, chr_name)
        if os.path.isdir(os.path.join(chr_path, "beta_samples")):
            chr_map.setdefault(chr_name, []).append(chr_path)

    return sorted((chr_tag, sorted(paths)) for chr_tag, paths in chr_map.items())


def _discover_pergene_run_dirs(pergene_root: str, chr_tag: str) -> List[str]:
    """All ``beta_samples`` roots for one chromosome (one or more refit runs)."""
    pergene_root = os.path.abspath(pergene_root)
    chr_tag = chr_tag if chr_tag.startswith("chr") else f"chr{chr_tag}"
    run_dirs: List[Tuple[int, str]] = []

    for name in sorted(os.listdir(pergene_root)):
        if not name.startswith("run_"):
            continue
        suffix = name.split("_", 1)[-1]
        if not suffix.isdigit():
            continue
        chr_path = os.path.join(pergene_root, name, chr_tag)
        if os.path.isdir(os.path.join(chr_path, "beta_samples")):
            run_dirs.append((int(suffix), chr_path))

    if run_dirs:
        return [p for _, p in sorted(run_dirs)]

    legacy = os.path.join(pergene_root, chr_tag)
    if os.path.isdir(os.path.join(legacy, "beta_samples")):
        return [legacy]
    return []


def _load_pergene_beta_df(
    pergene_root: str,
    chr_tag: str,
    gene_name: str,
    cfg: Dict[str, Any],
    train_idx: np.ndarray,
    avg_across_runs: bool = True,
) -> Optional[pd.DataFrame]:
    """Load per-gene common β, averaging across ``pergene/run_N/chrK`` refits when present."""
    gene_key = _gene_key(gene_name)
    run_dirs = _discover_pergene_run_dirs(pergene_root, chr_tag)
    if not run_dirs:
        return None

    series_list: List[pd.Series] = []
    last_df: Optional[pd.DataFrame] = None
    for rd in run_dirs:
        for fname in (f"{gene_key}_beta.csv.gz", f"{gene_name.replace('/', '_')}_beta.csv.gz"):
            path = os.path.join(rd, "beta_samples", fname)
            if os.path.isfile(path):
                break
        else:
            continue
        df = pd.read_csv(path, compression="infer")
        last_df = df
        train_cols = _joint_training_variant_columns(
            cfg, gene_name, train_idx, beta_n=len(df)
        )
        if train_cols and len(df) == len(train_cols):
            series_list.append(pd.Series(df["beta_mean"].values, index=train_cols))
        else:
            series_list.append(df.set_index("variant_id_G")["beta_mean"])

    if not series_list:
        return None
    if (not avg_across_runs) or len(series_list) == 1:
        return last_df

    mean_beta = pd.concat(series_list, axis=1).mean(axis=1)
    out = pd.DataFrame({"beta_mean": mean_beta.values})
    if isinstance(mean_beta.index, pd.Index):
        out["training_column"] = mean_beta.index.astype(str)
    elif last_df is not None and "variant_id_G" in last_df.columns:
        out["variant_id_G"] = last_df["variant_id_G"].values[: len(out)]
    return out


def _genes_from_joint_config(
    joint_cfg: Dict[str, Any],
    gene_allow: Optional[set],
) -> List[str]:
    genes = list(joint_cfg.get("genes") or [])
    if not genes:
        gene_list_path = joint_cfg.get("gene_list")
        if gene_list_path and os.path.isfile(gene_list_path):
            genes = sorted(_load_gene_list(gene_list_path))
    if gene_allow is not None:
        genes = [g for g in genes if g in gene_allow]
    return genes


def _load_rho_w(
    chr_dir: str,
    gene_key: str,
    default_rho: float,
    default_w: float,
    no_wg: bool,
    logger,
    no_rhog: bool = False,
) -> Tuple[float, float]:
    """Posterior means from per-chr ``posterior_stats.npz``, else ``w_g_rho_g.csv``, else CLI."""
    rho, w = float(default_rho), float(default_w)

    npz_path = os.path.join(chr_dir, "posterior_stats.npz")
    from_npz = _load_rho_w_from_posterior_npz(npz_path, gene_key, logger)
    if from_npz is not None:
        rho, w_npz = from_npz
        if w_npz is not None:
            w = w_npz
        logger.debug("Loaded rho_g=%.4f w_g=%.4f for %s from %s", rho, w, gene_key, npz_path)
    else:
        csv_path = os.path.join(chr_dir, "w_g_rho_g.csv")
        if os.path.isfile(csv_path):
            d = pd.read_csv(csv_path)
            row = None
            if "gene" in d.columns:
                m = d["gene"].astype(str) == gene_key
                if m.any():
                    row = d.loc[m].iloc[0]
            if row is None and len(d) == 1:
                row = d.iloc[0]
            if row is not None:
                if "rho_g_mean" in row.index:
                    rho = float(row["rho_g_mean"])
                if not no_wg and "w_g_mean" in row.index:
                    w = float(row["w_g_mean"])
            else:
                logger.warning("Could not pick a row in w_g_rho_g.csv for %s.", gene_key)
        else:
            logger.warning(
                "No posterior_stats.npz entry for %s and no w_g_rho_g.csv in %s — using --rho_g=%s, --w_g=%s",
                gene_key,
                chr_dir,
                default_rho,
                default_w,
            )

    if no_wg:
        w = 1.0
    if no_rhog:
        rho = 0.0
    return rho, w


def _tau_t_csv_paths_for_chr(run_dirs: List[str]) -> List[str]:
    return [
        os.path.join(rd, "tau_T.csv")
        for rd in run_dirs
        if os.path.isfile(os.path.join(rd, "tau_T.csv"))
    ]


def _load_tau_for_pergene_chr(run_dirs: List[str]) -> Tuple[pd.DataFrame, float, np.ndarray, np.ndarray]:
    paths = _tau_t_csv_paths_for_chr(run_dirs)
    if not paths:
        raise FileNotFoundError(
            f"No tau_T.csv under per-gene chr dir(s): {run_dirs[:3]}{'...' if len(run_dirs) > 3 else ''}"
        )
    if len(paths) == 1:
        return load_data.load_tau_threshold_from_csv(paths[0])
    mean_df = load_data.aggregate_tau_t_from_csv_files(paths)
    th = float(mean_df["Filter Threshold"].iloc[0])
    tau1, tau2 = load_data._tau_vectors_from_summary_df(mean_df)
    return mean_df, th, tau1, tau2


def _process_one_gene(
    gene_name: str,
    gene_cfg: Dict[str, Any],
    maf_thr: float,
    rho_default: float,
    w_default: float,
    device: torch.device,
    out_gene_dir: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    joint_dir: Optional[str] = None,
    beta_csv: Optional[str] = None,
    beta_df: Optional[pd.DataFrame] = None,
    chr_dir: Optional[str] = None,
    joint_wg_mean: Optional[pd.Series] = None,
    joint_rh_mean: Optional[pd.Series] = None,
    rho_w_from_joint: bool = False,
    cov_scaled: Optional[pd.DataFrame] = None,
    expr_cache: Optional[_ExpressionResidualizer] = None,
    base_config: Optional[Dict[str, Any]] = None,
    compute_r2: bool = True,
    tau_override: Optional[Tuple[float, np.ndarray, np.ndarray]] = None,
    brr_alphas: Optional[Dict[str, float]] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    logger = utils.get_logger()
    gene_key = _gene_key(gene_name)

    cfg = dict(gene_cfg)
    if joint_dir:
        cfg["joint_output_dir"] = joint_dir
    cfg["genes"] = [gene_name]
    cfg["maf_threshold"] = None
    no_wg = bool(cfg.get("no_wg", False))
    no_rhog = bool(cfg.get("no_rhog", False))

    if rho_w_from_joint:
        rho, w = _load_rho_w_from_joint(
            joint_dir,
            gene_key,
            joint_wg_mean,
            joint_rh_mean,
            rho_default,
            w_default,
            no_wg,
            logger,
            no_rhog=no_rhog,
        )
    else:
        if chr_dir is None:
            raise ValueError("chr_dir required when not using joint rho/w")
        rho, w = _load_rho_w(
            chr_dir, gene_key, rho_default, w_default, no_wg, logger,
            no_rhog=no_rhog,
        )

    G, G_raw, Z_ann, variant_ids_G, variant_ids_Z = _load_gene_G_Z(cfg)
    if G.shape[1] == 0:
        logger.warning("%s: empty genotype matrix; skip.", gene_name)
        return None

    n_var = G.shape[1]
    if not (len(variant_ids_G) == len(variant_ids_Z) == n_var):
        logger.warning(
            "%s: variant ID count mismatch (G cols=%d, ids_G=%d, ids_Z=%d); skip.",
            gene_name,
            n_var,
            len(variant_ids_G),
            len(variant_ids_Z),
        )
        return None
    if not G.columns.equals(Z_ann.index):
        logger.warning("%s: G.columns and Z_ann.index differ in order or labels; skip.", gene_name)
        return None

    G_train_raw = G_raw.iloc[train_idx]
    maf_s = load_data.variant_maf_series(G_train_raw)
    maf_beta = int(cfg.get("maf_beta", 1))
    G_raw_t = torch.as_tensor(
        np.asarray(G_train_raw.values, dtype=np.float32), device=device
    )
    maf_weights = utils.get_MAF_weights(G_raw_t, device, maf_beta)

    if tau_override is not None:
        th, tau1_np, tau2_np = tau_override
    elif joint_dir and cfg.get("joint_output_dir"):
        _, th, tau1_np, tau2_np = load_data.load_tau_threshold(cfg)
    else:
        raise ValueError(f"{gene_name}: no τ/T source (pass tau_override or joint_output_dir in config)")
    tau1 = torch.as_tensor(tau1_np, dtype=torch.float32, device=device)
    tau2 = torch.as_tensor(tau2_np, dtype=torch.float32, device=device)
    threshold = torch.as_tensor(float(th), dtype=torch.float32, device=device)

    Z_t = torch.as_tensor(np.asarray(Z_ann.values, dtype=np.float32), device=device)
    lam = annotation_lambda(Z_t, maf_weights, threshold, tau1=tau1, tau2=tau2)
    lam_np = lam.detach().cpu().numpy().ravel()

    if len(lam_np) != n_var:
        logger.warning("%s: lambda length mismatch; skip.", gene_name)
        return None

    maf_all = maf_s.reindex(G.columns).fillna(0.0)
    common_mask = (maf_all.values >= maf_thr).astype(bool)

    if beta_df is None:
        if beta_csv is None:
            logger.warning("%s: no beta CSV or dataframe; skip.", gene_name)
            return None
        beta_df = pd.read_csv(beta_csv, compression="infer")
    if "beta_mean" not in beta_df.columns:
        logger.warning("%s: beta table missing beta_mean; skip.", gene_name)
        return None
    beta_np, beta_align_method = _align_beta_to_G_columns(
        beta_df, gene_name, G.columns, cfg, train_idx, logger
    )

    eps = 1e-6
    mu, sigma_sqrt = _mu_sigma_sqrt_np(rho, w, lam_np, cfg)
    sigma = sigma_sqrt + eps

    # μ is defined per normalized genotype when normalize_G; scale rare / imputed β so
    # G_norm @ β ≈ G_raw @ μ (common fitted β from joint is already in normalized units).
    mu_beta = mu.astype(np.float64, copy=True)
    if cfg.get("normalize_G", False):
        std_raw = G_train_raw.std(axis=0, ddof=0).reindex(G.columns).fillna(1.0).values
        std_raw = np.maximum(std_raw, eps)
        mu_beta = mu_beta * std_raw

    if base_config is None:
        raise ValueError(f"{gene_name}: posterior β requires base_config for residualized y")
    if cov_scaled is None:
        cov_scaled = _load_covariates_scaled(base_config)
    y_all = _residualize_gene_y(
        base_config, gene_name, cov_scaled, device, expr_cache=expr_cache
    )
    y_train = y_all[train_idx]
    alpha = _gene_brr_alpha(gene_key, brr_alphas)
    G_tr = G.iloc[train_idx]
    # Posterior uses normalized G and unscaled μ, σ (matches collapsed training).
    beta_full = _compute_posterior_beta_hat(
        G_tr,
        y_train,
        mu,
        sigma,
        alpha,
        lam_np,
        device,
    )
    rare_mask = ~common_mask
    beta_common = _scatter_decoupled_posterior_beta(
        G_tr, y_train, mu, sigma, alpha, lam_np, common_mask, device
    )
    beta_rare = _scatter_decoupled_posterior_beta(
        G_tr, y_train, mu, sigma, alpha, lam_np, rare_mask, device
    )

    beta_common_train = _build_beta_common_train_posterior(common_mask, beta_np)

    has_train_beta = np.isfinite(beta_np)
    bad_common = common_mask & ~np.isfinite(beta_np)
    if np.any(bad_common):
        logger.warning(
            "%s: %d common SNPs without aligned training beta (method=%s).",
            gene_name,
            int(np.sum(bad_common)),
            beta_align_method,
        )

    n_beta_common = int((common_mask & has_train_beta).sum())
    n_common = int(common_mask.sum())
    if n_beta_common < n_common:
        logger.info(
            "%s: aligned training beta for %d/%d common variants (align=%s; %d on full panel).",
            gene_name,
            n_beta_common,
            n_common,
            beta_align_method,
            n_var,
        )

    out_cols: Dict[str, Any] = {
        "variant_id_G": G.columns,
        "variant_id_G_raw": variant_ids_G,
        "variant_id_Z_raw": variant_ids_Z,
        "maf_train": maf_all.values.astype(np.float64),
        "is_common_maf_ge_threshold": common_mask,
        "lambda_from_tau_Z_ann_maf": lam_np.astype(np.float64),
        "mu_rho_w_lambda": mu.astype(np.float64),
        "mu_beta": mu_beta.astype(np.float64),
        "sigma_from_model": sigma.astype(np.float64),
        "beta_hat": beta_full.astype(np.float64),
        "beta_common_train_posterior": beta_common_train.astype(np.float64),
        "beta_hat_common_decoupled": beta_common.astype(np.float64),
        "beta_hat_rare_decoupled": beta_rare.astype(np.float64),
        "rho_g": rho,
        "w_g": w,
    }
    out_df = pd.DataFrame(out_cols)
    os.makedirs(out_gene_dir, exist_ok=True)
    safe = gene_name.replace("/", "_")
    out_path = os.path.join(out_gene_dir, f"{safe}_full_panel_beta.csv.gz")
    out_df.to_csv(out_path, index=False, compression="gzip")

    summary_row: Dict[str, Any] = {
        "gene": gene_name,
        "n_variants": int(len(beta_np)),
        "n_common": int(common_mask.sum()),
        "n_rare": int((~common_mask).sum()),
        "rho_g": rho,
        "w_g": w,
        "maf_threshold": maf_thr,
        "out_csv": out_path,
        "filter_threshold_T": float(th),
        "beta_source": "joint" if rho_w_from_joint else "pergene",
        "beta_align_method": beta_align_method,
    }

    if np.any(common_mask):
        summary_row.update(_beta_pair_metrics(beta_np, beta_full, common_mask))
        summary_row.update(
            _beta_pair_metrics(
                beta_np,
                beta_common,
                common_mask,
                metric_suffix="_decoupled",
            )
        )

    train_r2_row: Dict[str, Any] = {"gene": gene_name}
    test_r2_row: Dict[str, Any] = {"gene": gene_name}
    if compute_r2:
        G_te = G.iloc[test_idx]
        y_tr = y_all[train_idx]
        y_te = y_all[test_idx]
        train_r2, test_r2 = _compute_r2_grid(
            G_tr,
            G_te,
            y_tr,
            y_te,
            common_mask,
            mu_beta=mu_beta,
            beta_full=beta_full,
            beta_common=beta_common,
            beta_rare=beta_rare,
            beta_train=beta_common_train,
        )
        train_r2_row.update(train_r2)
        test_r2_row.update(test_r2)

    return summary_row, train_r2_row, test_r2_row


def main() -> int:
    p = argparse.ArgumentParser(
        description="Full-panel post-train β panels, QC metrics, and R² assemblies."
    )
    p.add_argument(
        "--joint_output_dir",
        default=None,
        help="Joint experiment root → post-joint mode (τ/T and common β from joint/run_*).",
    )
    p.add_argument(
        "--pergene_output_dir",
        default=None,
        help="Per-gene output root → post-per-gene mode (wins if both dirs are passed).",
    )
    p.add_argument(
        "--joint_run",
        type=int,
        default=None,
        help="Post-joint mode: use a single joint run_N for common β (default: average refits).",
    )
    p.add_argument("--out_dir", default=None)
    p.add_argument(
        "--maf_threshold",
        type=float,
        default=None,
        help="Train MAF used only to label common (CSV β) vs rare (β=μ). YAML or default 0.01 if unset.",
    )
    p.add_argument(
        "--rho_g",
        type=float,
        default=0.5,
        help="Default rho_g if not found in posterior_stats.npz / w_g_rho_g.csv.",
    )
    p.add_argument(
        "--w_g",
        type=float,
        default=1.0,
        help="Default w_g if not found in posterior_stats.npz / w_g_rho_g.csv (ignored when no_wg in config).",
    )
    p.add_argument(
        "--skip_r2",
        action="store_true",
        help="Skip R² (default: compute train/test R² for common-only and common+rare variant sets).",
    )
    p.add_argument(
        "--gene_list",
        default=None,
        help="Optional file: only process genes in this list (e.g. 200-gene joint list).",
    )
    p.add_argument(
        "--chromosome",
        default=None,
        help="Process one chromosome only (e.g. 10 or chr10). Use with Slurm --array=1-22.",
    )
    args = p.parse_args()

    joint = os.path.abspath(args.joint_output_dir) if args.joint_output_dir else None
    pergene = os.path.abspath(args.pergene_output_dir) if args.pergene_output_dir else None

    if pergene:
        mode = "pergene"
    elif joint:
        mode = "joint"
    else:
        p.error("Provide --joint_output_dir (post-joint) or --pergene_output_dir (post-per-gene).")

    if mode == "joint" and not os.path.isdir(joint):
        p.error(f"Joint directory not found: {joint}")
    if mode == "pergene" and not os.path.isdir(pergene):
        p.error(f"Per-gene directory not found: {pergene}")

    chr_filter = _normalize_chr_tag(args.chromosome) if args.chromosome else None

    base_out = (
        os.path.join(os.path.dirname(pergene), "post_pergene")
        if mode == "pergene"
        else os.path.join(os.path.dirname(joint), "post_joint")
    )
    if args.out_dir:
        out = os.path.abspath(args.out_dir)
    elif chr_filter:
        out = os.path.abspath(os.path.join(base_out, chr_filter))
    else:
        out = os.path.abspath(base_out)
    utils.setup_logging("INFO", None)
    logger = utils.get_logger()

    os.makedirs(out, exist_ok=True)
    out_genes = os.path.join(out, "full_beta_panel")
    os.makedirs(out_genes, exist_ok=True)

    device = torch.device("cpu")

    if mode == "joint":
        joint_cfg_path = os.path.join(joint, "config.yaml")
        with open(joint_cfg_path) as f:
            base_config = yaml.safe_load(f) or {}
    else:
        try:
            base_config = _load_pergene_root_config(pergene)
        except FileNotFoundError:
            if joint and os.path.isfile(os.path.join(joint, "config.yaml")):
                logger.warning("No pergene/config.yaml; falling back to joint/config.yaml for paths.")
                with open(os.path.join(joint, "config.yaml")) as f:
                    base_config = yaml.safe_load(f) or {}
            else:
                raise

    train_idx, test_idx = load_data.get_train_test_indices(base_config["covariates_path"])
    compute_r2 = not args.skip_r2
    cov_scaled = _load_covariates_scaled(base_config)
    expr_cache = _ExpressionResidualizer(base_config, cov_scaled, device)
    brr_alphas: Optional[Dict[str, float]] = None
    if base_config.get("brr_results_dir"):
        try:
            brr_alphas = load_data.load_brr_results(base_config).get("alphas")
            logger.info(
                "Loaded %d BRR alphas from %s",
                len(brr_alphas or {}),
                base_config["brr_results_dir"],
            )
        except Exception as e:
            logger.warning("Could not load BRR alphas (%s); using default_std=0.5", e)
    gene_allow: Optional[set] = _load_gene_list(args.gene_list) if args.gene_list else None
    if gene_allow:
        logger.info("Restricting to %d genes from --gene_list", len(gene_allow))

    manifest: Dict[str, Any] = {
        "mode": mode,
        "joint_output_dir": joint,
        "pergene_output_dir": pergene if mode == "pergene" else None,
        "chromosome": chr_filter,
        "joint_run": args.joint_run,
        "out_dir": out,
        "maf_threshold_cli": args.maf_threshold,
        "default_rho_g": args.rho_g,
        "default_w_g": args.w_g,
        "compute_r2": compute_r2,
        "gene_list": args.gene_list,
        "r2_score_columns": _all_r2_score_columns(),
        "beta_panel_columns": [
            "mu_rho_w_lambda",
            "mu_beta",
            "beta_hat",
            "beta_common_train_posterior",
            "beta_hat_common_decoupled",
            "beta_hat_rare_decoupled",
        ],
    }

    summary_rows: List[Dict[str, Any]] = []
    train_r2_rows: List[Dict[str, Any]] = []
    test_r2_rows: List[Dict[str, Any]] = []

    try:
        if mode == "joint":
            tau_df = load_data.aggregate_tau_t_from_joint_runs(joint)
        else:
            chr_outputs = _discover_pergene_chr_output_dirs(pergene)
            all_tau_paths = []
            for _, run_dirs in chr_outputs:
                all_tau_paths.extend(_tau_t_csv_paths_for_chr(run_dirs))
            if not all_tau_paths:
                raise FileNotFoundError(f"No tau_T.csv under {pergene}")
            tau_df = load_data.aggregate_tau_t_from_csv_files(sorted(set(all_tau_paths)))
        manifest["tau_rows"] = int(len(tau_df))
    except Exception as e:
        manifest["tau_load_error"] = repr(e)
        logger.error("Could not load τ/T: %s", e)
        return 1

    if mode == "joint":
        maf_thr = (
            float(args.maf_threshold)
            if args.maf_threshold is not None
            else _maf_threshold_from_config(base_config, 0.01)
        )
        wg_mean, rh_mean = _aggregate_joint_wg_rhog(joint)
        if wg_mean is None and not base_config.get("no_wg", False):
            logger.warning("Could not load joint w_g.csv aggregates; will use --w_g default.")
        if rh_mean is None and not base_config.get("no_rhog", False):
            logger.warning("Could not load joint rho_g.csv aggregates; will use --rho_g default.")

        genes = _genes_from_joint_config(base_config, gene_allow)
        if chr_filter:
            genes = [g for g in genes if g.startswith(f"{chr_filter}/")]
        if not genes:
            logger.error(
                "No genes in joint config%s",
                f" on {chr_filter}" if chr_filter else " (genes: or gene_list:)",
            )
            manifest["error"] = "no_genes_in_joint_config"
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            return 1

        logger.info("Post-joint mode: processing %d genes", len(genes))
        for gene_name in genes:
            gene_cfg = dict(base_config)
            if "/" in gene_name:
                chr_stub = gene_name.split("/")[0]
                gene_cfg["chromosome"] = str(chr_stub).replace("chr", "")
            beta_df = _load_joint_beta_df(
                joint,
                gene_name,
                base_config,
                train_idx,
                avg_across_runs=(args.joint_run is None),
                joint_run=args.joint_run,
            )
            if beta_df is None:
                logger.warning("No joint beta_samples for %s; skip.", gene_name)
                continue
            try:
                result = _process_one_gene(
                    gene_name,
                    gene_cfg,
                    maf_thr,
                    args.rho_g,
                    args.w_g,
                    device,
                    out_genes,
                    train_idx,
                    test_idx,
                    joint_dir=joint,
                    beta_df=beta_df,
                    joint_wg_mean=wg_mean,
                    joint_rh_mean=rh_mean,
                    rho_w_from_joint=True,
                    cov_scaled=cov_scaled,
                    expr_cache=expr_cache,
                    base_config=base_config,
                    compute_r2=compute_r2,
                    brr_alphas=brr_alphas,
                )
                if result:
                    summary_row, train_r2_row, test_r2_row = result
                    summary_rows.append(summary_row)
                    if compute_r2:
                        train_r2_rows.append(train_r2_row)
                        test_r2_rows.append(test_r2_row)
            except Exception as e:
                logger.warning("Failed %s: %s", gene_name, e)
            finally:
                gc.collect()
    else:
        try:
            pergene_cfg = _load_pergene_root_config(pergene)
        except FileNotFoundError:
            pergene_cfg = dict(base_config)

        chr_outputs = _discover_pergene_chr_output_dirs(pergene)
        if chr_filter:
            chr_outputs = [(tag, dirs) for tag, dirs in chr_outputs if tag == chr_filter]
        if not chr_outputs:
            logger.error(
                "No pergene/chr* with beta_samples under %s%s",
                pergene,
                f" for {chr_filter}" if chr_filter else "",
            )
            manifest["error"] = "no_chr_dirs"
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            return 1

        for chr_tag, run_dirs in chr_outputs:
            chr_cfg = dict(pergene_cfg)
            chr_cfg["pergene_output_dir"] = pergene
            if joint:
                chr_cfg["joint_output_dir"] = joint
            chr_cfg["chromosome"] = str(chr_tag).replace("chr", "")
            maf_thr = (
                float(args.maf_threshold)
                if args.maf_threshold is not None
                else _maf_threshold_from_config(chr_cfg, 0.01)
            )

            try:
                _, th_chr, tau1_chr, tau2_chr = _load_tau_for_pergene_chr(run_dirs)
            except FileNotFoundError as e:
                logger.error("%s: %s", chr_tag, e)
                continue
            tau_override = (th_chr, tau1_chr, tau2_chr)

            genes_in_chr: set = set()
            for rd in run_dirs:
                for beta_csv in glob.glob(os.path.join(rd, "beta_samples", "*_beta.csv.gz")):
                    gene_name = _parse_gene_from_beta_filename(beta_csv, rd)
                    if gene_name is not None:
                        genes_in_chr.add(gene_name)

            chr_dir = run_dirs[0]
            for gene_name in sorted(genes_in_chr):
                if gene_allow is not None and gene_name not in gene_allow:
                    continue
                beta_df = _load_pergene_beta_df(pergene, chr_tag, gene_name, chr_cfg, train_idx)
                if beta_df is None:
                    logger.warning("No per-gene beta for %s under %s; skip.", gene_name, chr_tag)
                    continue
                try:
                    result = _process_one_gene(
                        gene_name,
                        chr_cfg,
                        maf_thr,
                        args.rho_g,
                        args.w_g,
                        device,
                        out_genes,
                        train_idx,
                        test_idx,
                        joint_dir=joint,
                        beta_df=beta_df,
                        chr_dir=chr_dir,
                        cov_scaled=cov_scaled,
                        expr_cache=expr_cache,
                        base_config=base_config,
                        compute_r2=compute_r2,
                        tau_override=tau_override,
                        brr_alphas=brr_alphas,
                    )
                    if result:
                        summary_row, train_r2_row, test_r2_row = result
                        summary_row["chromosome_dir"] = chr_tag
                        summary_rows.append(summary_row)
                        if compute_r2:
                            train_r2_row["chromosome_dir"] = chr_tag
                            test_r2_row["chromosome_dir"] = chr_tag
                            train_r2_rows.append(train_r2_row)
                            test_r2_rows.append(test_r2_row)
                except Exception as e:
                    logger.warning("Failed %s: %s", gene_name, e)
                finally:
                    gc.collect()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(out, "summary.csv"), index=False)
        manifest["n_genes_ok"] = len(summary_rows)
        manifest["status"] = "ok"
        if compute_r2 and train_r2_rows and test_r2_rows:
            train_df = pd.DataFrame(train_r2_rows)
            test_df = pd.DataFrame(test_r2_rows)
            _save_r2_score_tables(train_r2_rows, test_r2_rows, out)
            r2_summary = _r2_threshold_summary_from_tables(train_df, test_df)
            with open(os.path.join(out, "r2_threshold_summary.json"), "w") as f:
                json.dump(r2_summary, f, indent=2)
        if "common_beta_pearson" in summary_df.columns:
            cmp = summary_df["common_beta_pearson"].dropna()
            manifest["common_beta_saved_vs_posterior"] = {
                "n_genes_with_metrics": int(len(cmp)),
                "median_pearson": float(cmp.median()) if len(cmp) else float("nan"),
                "mean_pearson": float(cmp.mean()) if len(cmp) else float("nan"),
                "median_rmse": float(summary_df["common_beta_rmse"].dropna().median())
                if "common_beta_rmse" in summary_df.columns
                and summary_df["common_beta_rmse"].notna().any()
                else float("nan"),
            }
        if "common_beta_pearson_decoupled" in summary_df.columns:
            cmp_dec = summary_df["common_beta_pearson_decoupled"].dropna()
            manifest["common_beta_saved_vs_posterior_decoupled"] = {
                "n_genes_with_metrics": int(len(cmp_dec)),
                "median_pearson": float(cmp_dec.median()) if len(cmp_dec) else float("nan"),
                "mean_pearson": float(cmp_dec.mean()) if len(cmp_dec) else float("nan"),
                "median_rmse": float(
                    summary_df["common_beta_rmse_decoupled"].dropna().median()
                )
                if "common_beta_rmse_decoupled" in summary_df.columns
                and summary_df["common_beta_rmse_decoupled"].notna().any()
                else float("nan"),
            }
    else:
        manifest["n_genes_ok"] = 0
        manifest["status"] = "no_genes_processed"

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("emmental_postjoint_rare_beta: wrote %d gene table(s) under %s", len(summary_rows), out_genes)
    return 0 if summary_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
