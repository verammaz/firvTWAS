#!/usr/bin/env python3
"""
Post-train rare-variant diagnostics (A: diagnose before changing the model).

1. R² decomposition per gene: r2_common, r2_rare, r2_all (train + test).
2. Variant-level μ / λ: |μ| vs MAF, fraction λ≈0 by stratum and dominant annotation.
3. T-gate vs no-gate: μ_full = ρ·w·λ_gated vs μ_no_gate = ρ·w·(lin1·exp(lin2)·MAF).
4. Methods comparison table (baselines common-only vs Emmental common / full panel).

Reads existing ``full_beta_panel/*_full_panel_beta.csv.gz`` from post_pergene or post_joint.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Emmental src (load_data, post-train helpers)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import load_data  # noqa: E402
import utils  # noqa: E402
from emmental_post_train_betas import (  # noqa: E402
    _ExpressionResidualizer,
    _load_covariates_scaled,
    _load_gene_G_Z,
    _load_pergene_root_config,
    _load_tau_for_pergene_chr,
    _discover_pergene_run_dirs,
    _load_gene_list,
    _maf_threshold_from_config,
    _normalize_chr_tag,
    _r2_predict,
)

BASELINE_ROOT_DEFAULT = (
    "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01"
)
TRAIN_COMMON01_DEFAULT = "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01"
GENE_LIST_DEFAULT = (
    "/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt"
)

LAMBDA_ZERO_EPS = 1e-12
MAF_BINS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
MAF_BIN_LABELS = [
    "[0,1e-3)",
    "[1e-3,5e-3)",
    "[5e-3,1e-2)",
    "[1e-2,5e-2)",
    "[5e-2,1e-1)",
    "[1e-1,5e-1)",
    "[5e-1,1]",
]


def _ensg_from_gene(gene: str) -> str:
    g = str(gene).strip()
    return g.split("/")[-1] if "/" in g else g


def _parse_gene_from_beta_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.match(r"^(chr\d+)_(ENSG\d+)_full_panel_beta\.csv\.gz$", base)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def discover_beta_panel_files(post_dir: str, chromosome: Optional[str] = None) -> List[str]:
    """All ``full_beta_panel/*_full_panel_beta.csv.gz`` under post dir (optionally one chr)."""
    post_dir = os.path.abspath(post_dir)
    if chromosome:
        chr_tag = _normalize_chr_tag(chromosome)
        pattern = os.path.join(post_dir, chr_tag, "full_beta_panel", "*_full_panel_beta.csv.gz")
        paths = sorted(glob.glob(pattern))
        if paths:
            return paths
    pattern = os.path.join(post_dir, "full_beta_panel", "*_full_panel_beta.csv.gz")
    paths = sorted(glob.glob(pattern))
    if paths:
        return paths
    pattern = os.path.join(post_dir, "chr*", "full_beta_panel", "*_full_panel_beta.csv.gz")
    return sorted(glob.glob(pattern))


def _filter_paths_by_gene_list(paths: List[str], gene_allow: Optional[set]) -> List[str]:
    if not gene_allow:
        return paths
    out = []
    for p in paths:
        g = _parse_gene_from_beta_path(p)
        if g and g in gene_allow:
            out.append(p)
    return out


def _lambda_no_gate(
    Z_ann: pd.DataFrame,
    maf_weights: torch.Tensor,
    threshold: float,
    tau1: np.ndarray,
    tau2: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Same as annotation_lambda but without the |Z·τ₁| ≥ T gate."""
    Z_t = torch.as_tensor(np.asarray(Z_ann.values, dtype=np.float32), device=device)
    tau1_t = torch.as_tensor(tau1, dtype=torch.float32, device=device)
    tau2_t = torch.as_tensor(tau2, dtype=torch.float32, device=device)
    lin2 = Z_t.matmul(tau2_t)
    mod = torch.exp(lin2)
    Z_aug = torch.cat(
        [torch.ones(Z_t.shape[0], 1, dtype=torch.float32, device=device), Z_t], dim=1
    )
    lin1 = Z_aug.matmul(tau1_t)
    return (lin1 * mod * maf_weights).detach().cpu().numpy().ravel()


def _dominant_annotation_per_variant(
    Z_ann: pd.DataFrame,
    tau1: np.ndarray,
    annotation_names: List[str],
) -> np.ndarray:
    """Annotation (excluding intercept) with largest |Z_j · τ₁_j| per variant."""
    # tau1[0] is intercept; columns of Z_ann align with tau1[1:]
    z = Z_ann.values.astype(np.float64)
    t1 = tau1[1:].astype(np.float64)
    contrib = np.abs(z * t1.reshape(1, -1))
    idx = np.argmax(contrib, axis=1)
    names = np.array(annotation_names, dtype=object)
    return names[idx]


def collect_variant_level_from_csv(
    beta_paths: Iterable[str],
    maf_threshold: float = 0.01,
) -> pd.DataFrame:
    """Stack variant rows from full_beta_panel CSVs (no G reload)."""
    chunks: List[pd.DataFrame] = []
    for path in beta_paths:
        gene = _parse_gene_from_beta_path(path)
        if gene is None:
            continue
        df = pd.read_csv(path, compression="infer")
        df["gene"] = gene
        df["abs_mu"] = np.abs(df["mu_rho_w_lambda"].astype(np.float64))
        df["abs_lambda"] = np.abs(df["lambda_from_tau_Z_ann_maf"].astype(np.float64))
        df["lambda_zero"] = df["abs_lambda"] <= LAMBDA_ZERO_EPS
        df["is_rare"] = ~df["is_common_maf_ge_threshold"].astype(bool)
        df["maf_bin"] = pd.cut(
            df["maf_train"].astype(np.float64),
            bins=MAF_BINS,
            labels=MAF_BIN_LABELS,
            right=False,
            include_lowest=True,
        )
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def summarize_mu_maf_lambda(var_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Binned |μ| and λ≈0 fraction by MAF bin and common/rare stratum."""
    if var_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows_maf = []
    for stratum, sub in [
        ("all", var_df),
        ("common", var_df[~var_df["is_rare"]]),
        ("rare", var_df[var_df["is_rare"]]),
    ]:
        for maf_bin, grp in sub.groupby("maf_bin", observed=False):
            if len(grp) == 0:
                continue
            rows_maf.append(
                {
                    "stratum": stratum,
                    "maf_bin": str(maf_bin),
                    "n_variants": len(grp),
                    "median_abs_mu": float(grp["abs_mu"].median()),
                    "mean_abs_mu": float(grp["abs_mu"].mean()),
                    "frac_lambda_zero": float(grp["lambda_zero"].mean()),
                    "median_maf": float(grp["maf_train"].median()),
                }
            )
    maf_summary = pd.DataFrame(rows_maf)

    rows_stratum = []
    for stratum, sub in [
        ("all", var_df),
        ("common", var_df[~var_df["is_rare"]]),
        ("rare", var_df[var_df["is_rare"]]),
    ]:
        rows_stratum.append(
            {
                "stratum": stratum,
                "n_variants": len(sub),
                "median_abs_mu": float(sub["abs_mu"].median()) if len(sub) else np.nan,
                "mean_abs_mu": float(sub["abs_mu"].mean()) if len(sub) else np.nan,
                "frac_lambda_zero": float(sub["lambda_zero"].mean()) if len(sub) else np.nan,
                "frac_rare_among_gene_variants": float(sub["is_rare"].mean()) if len(sub) else np.nan,
            }
        )
    stratum_summary = pd.DataFrame(rows_stratum)
    return maf_summary, stratum_summary


def compute_r2_decomposition_for_gene(
    gene_name: str,
    beta_df: pd.DataFrame,
    base_config: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cov_scaled: pd.DataFrame,
    expr_cache: _ExpressionResidualizer,
    device: torch.device,
) -> Dict[str, Any]:
    """r2_common / r2_rare / r2_all on train and test."""
    gene_cfg = dict(base_config)
    gene_cfg["genes"] = [gene_name]
    gene_cfg["maf_threshold"] = None  # full panel

    G, _, _, _, _ = _load_gene_G_Z(gene_cfg)
    if G.empty or len(beta_df) != len(G.columns):
        return {"gene": gene_name, "error": "G/beta shape mismatch"}

    common_mask = beta_df["is_common_maf_ge_threshold"].astype(bool).values
    rare_mask = ~common_mask
    beta_full = beta_df["beta_full_common_csv_rare_mu"].astype(np.float64).values
    beta_common = np.where(common_mask, beta_full, 0.0)
    beta_rare = np.where(rare_mask, beta_df["mu_rho_w_lambda"].astype(np.float64).values, 0.0)

    y_all = expr_cache.residualize(gene_name)
    G_tr = G.iloc[train_idx]
    G_te = G.iloc[test_idx]
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]

    return {
        "gene": gene_name,
        "n_variants": len(beta_full),
        "n_common": int(common_mask.sum()),
        "n_rare": int(rare_mask.sum()),
        "r2_train_common": _r2_predict(G_tr, y_tr, beta_common, common_mask),
        "r2_test_common": _r2_predict(G_te, y_te, beta_common, common_mask),
        "r2_train_rare": _r2_predict(G_tr, y_tr, beta_rare, rare_mask),
        "r2_test_rare": _r2_predict(G_te, y_te, beta_rare, rare_mask),
        "r2_train_all": _r2_predict(G_tr, y_tr, beta_full, np.ones(len(beta_full), dtype=bool)),
        "r2_test_all": _r2_predict(G_te, y_te, beta_full, np.ones(len(beta_full), dtype=bool)),
    }


def compute_gate_comparison_for_gene(
    gene_name: str,
    beta_df: pd.DataFrame,
    tau1: np.ndarray,
    tau2: np.ndarray,
    threshold: float,
    annotation_names: List[str],
    base_config: Dict[str, Any],
    train_idx: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Per-variant μ gated vs no-gate; dominant annotation for λ≈0."""
    gene_cfg = dict(base_config)
    gene_cfg["genes"] = [gene_name]
    gene_cfg["maf_threshold"] = None

    G, Z_ann, _, _ = load_data.load_genes(gene_cfg)
    if G.empty or len(beta_df) != len(G.columns):
        return pd.DataFrame()

    G_train = G.iloc[train_idx]
    G_t = torch.as_tensor(np.asarray(G_train.values, dtype=np.float32), device=device)
    maf_beta = float(base_config.get("maf_beta", 1.0))
    maf_weights = utils.get_MAF_weights(G_t, device, maf_beta)

    lam_nogate = _lambda_no_gate(Z_ann, maf_weights, threshold, tau1, tau2, device)
    rho = float(beta_df["rho_g"].iloc[0])
    w = float(beta_df["w_g"].iloc[0])
    mu_full = beta_df["mu_rho_w_lambda"].astype(np.float64).values
    mu_nogate = rho * w * lam_nogate

    lam_full = beta_df["lambda_from_tau_Z_ann_maf"].astype(np.float64).values
    common_mask = beta_df["is_common_maf_ge_threshold"].astype(bool).values
    rare_mask = ~common_mask
    dom_ann = _dominant_annotation_per_variant(Z_ann, tau1, annotation_names)

    out = pd.DataFrame(
        {
            "gene": gene_name,
            "variant_id_G": beta_df["variant_id_G"].values,
            "maf_train": beta_df["maf_train"].astype(np.float64).values,
            "is_rare": rare_mask,
            "lambda_gated": lam_full,
            "lambda_no_gate": lam_nogate,
            "mu_gated": mu_full,
            "mu_no_gate": mu_nogate,
            "lambda_zero_gated": np.abs(lam_full) <= LAMBDA_ZERO_EPS,
            "lambda_zero_no_gate": np.abs(lam_nogate) <= LAMBDA_ZERO_EPS,
            "gate_killed": (np.abs(lam_full) <= LAMBDA_ZERO_EPS)
            & (np.abs(lam_nogate) > LAMBDA_ZERO_EPS),
            "dominant_annotation": dom_ann,
        }
    )
    return out


def summarize_gate_comparison(gate_df: pd.DataFrame) -> pd.DataFrame:
    if gate_df.empty:
        return pd.DataFrame()
    rows = []
    for stratum, sub in [
        ("all", gate_df),
        ("common", gate_df[~gate_df["is_rare"]]),
        ("rare", gate_df[gate_df["is_rare"]]),
    ]:
        rows.append(
            {
                "stratum": stratum,
                "n_variants": len(sub),
                "frac_lambda_zero_gated": float(sub["lambda_zero_gated"].mean()),
                "frac_lambda_zero_no_gate": float(sub["lambda_zero_no_gate"].mean()),
                "frac_gate_killed": float(sub["gate_killed"].mean()),
                "median_abs_mu_gated": float(np.abs(sub["mu_gated"]).median()),
                "median_abs_mu_no_gate": float(np.abs(sub["mu_no_gate"]).median()),
                "corr_mu_gated_no_gate": float(
                    np.corrcoef(sub["mu_gated"], sub["mu_no_gate"])[0, 1]
                )
                if len(sub) > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_lambda_zero_by_annotation(gate_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction λ≈0 (gated) by dominant annotation and stratum."""
    if gate_df.empty:
        return pd.DataFrame()
    rows = []
    for stratum, sub in [
        ("all", gate_df),
        ("common", gate_df[~gate_df["is_rare"]]),
        ("rare", gate_df[gate_df["is_rare"]]),
    ]:
        for ann, grp in sub.groupby("dominant_annotation", observed=True):
            rows.append(
                {
                    "stratum": stratum,
                    "dominant_annotation": ann,
                    "n_variants": len(grp),
                    "frac_lambda_zero_gated": float(grp["lambda_zero_gated"].mean()),
                    "frac_gate_killed": float(grp["gate_killed"].mean()),
                    "median_abs_mu_gated": float(np.abs(grp["mu_gated"]).median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["stratum", "n_variants"], ascending=[True, False])


def summarize_r2_decomposition(r2_df: pd.DataFrame) -> pd.DataFrame:
    if r2_df.empty:
        return pd.DataFrame()
    metrics = [
        "r2_train_common",
        "r2_test_common",
        "r2_train_rare",
        "r2_test_rare",
        "r2_train_all",
        "r2_test_all",
    ]
    rows = []
    for col in metrics:
        s = r2_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "metric": col,
                "n_genes": len(s),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "frac_positive": float((s > 0).mean()),
                "frac_gt_0p01": float((s > 0.01).mean()),
            }
        )
    # Per-gene deltas
    ok = r2_df.dropna(subset=["r2_train_common", "r2_train_rare", "r2_train_all"])
    if len(ok):
        rows.append(
            {
                "metric": "delta_train_all_minus_common",
                "n_genes": len(ok),
                "mean": float((ok["r2_train_all"] - ok["r2_train_common"]).mean()),
                "median": float((ok["r2_train_all"] - ok["r2_train_common"]).median()),
                "frac_positive": float((ok["r2_train_all"] > ok["r2_train_common"]).mean()),
                "frac_gt_0p01": np.nan,
            }
        )
        rows.append(
            {
                "metric": "delta_train_rare_contribution_implied",
                "n_genes": len(ok),
                "mean": float((ok["r2_train_all"] - ok["r2_train_common"]).mean()),
                "median": float((ok["r2_train_rare"]).median()),
                "frac_positive": float((ok["r2_train_rare"] > 0).mean()),
                "frac_gt_0p01": float((ok["r2_train_rare"] > 0.01).mean()),
            }
        )
    return pd.DataFrame(rows)


def load_baseline_r2(baseline_root: str, method: str = "bayesian_ridge") -> Tuple[pd.Series, pd.Series]:
    train_parts, test_parts = [], []
    method_dir = os.path.join(baseline_root, method)
    for fn in sorted(os.listdir(method_dir)):
        if not re.match(r"^chr\d+\.tsv$", fn):
            continue
        df = pd.read_csv(os.path.join(method_dir, fn), sep="\t", index_col=0)
        train_parts.append(df["R2_train"].rename("r2"))
        test_parts.append(df["R2_test"].rename("r2"))
    tr = pd.concat(train_parts).groupby(level=0).first()
    te = pd.concat(test_parts).groupby(level=0).first()
    return tr, te


def build_methods_comparison_table(
    r2_df: pd.DataFrame,
    var_df: pd.DataFrame,
    baseline_root: str,
    gene_ensgs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fair common-only comparison + full-panel variant counts."""
    btr, bte = load_baseline_r2(baseline_root, "bayesian_ridge")

    if gene_ensgs is None:
        gene_ensgs = sorted({_ensg_from_gene(g) for g in r2_df["gene"].astype(str)})

    r2_idx = r2_df.copy()
    r2_idx["ensg"] = r2_idx["gene"].map(_ensg_from_gene)
    r2_sub = r2_idx[r2_idx["ensg"].isin(gene_ensgs)]

    var_gene = (
        var_df.groupby("gene", observed=True)
        .agg(
            n_variants=("variant_id_G", "count"),
            n_common=("is_rare", lambda s: int((~s.astype(bool)).sum())),
            n_rare=("is_rare", lambda s: int(s.astype(bool).sum())),
        )
        .reset_index()
    )
    var_gene["ensg"] = var_gene["gene"].map(_ensg_from_gene)
    merged = r2_sub.merge(var_gene, on="ensg", how="left", suffixes=("", "_var"))

    def _row(
        method: str,
        split: str,
        r2_vals: pd.Series,
        *,
        variant_scope: str,
    ) -> Dict[str, Any]:
        r2_vals = r2_vals.reindex(gene_ensgs)
        if variant_scope == "common_only":
            mean_n_var = float(merged["n_common"].mean())
            mean_n_common = mean_n_var
            mean_n_rare = 0.0
            notes = "Common SNPs only (MAF≥0.01 at train)"
        elif variant_scope == "full_panel":
            mean_n_var = float(merged["n_variants"].mean())
            mean_n_common = float(merged["n_common"].mean())
            mean_n_rare = float(merged["n_rare"].mean())
            notes = "Full panel: common β from fit + rare μ=ρ·w·λ"
        else:  # rare_only
            mean_n_var = float(merged["n_rare"].mean())
            mean_n_common = 0.0
            mean_n_rare = mean_n_var
            notes = "Rare-only predictor (common β=0); diagnostic for μ contribution"

        if method == "BRR_baseline":
            notes = "sklearn BayesianRidge; " + notes

        return {
            "method": method,
            "split": split,
            "n_genes": len(gene_ensgs),
            "n_genes_with_r2": int(r2_vals.notna().sum()),
            "mean_r2": float(r2_vals.mean()),
            "median_r2": float(r2_vals.median()),
            "prop_r2_gt_0": float((r2_vals.fillna(-np.inf) > 0).mean()),
            "prop_r2_gt_0p01": float((r2_vals.fillna(-np.inf) > 0.01).mean()),
            "prop_r2_gt_0p1": float((r2_vals.fillna(-np.inf) > 0.1).mean()),
            "mean_n_variants_scored": mean_n_var,
            "mean_n_common_variants": mean_n_common,
            "mean_n_rare_variants": mean_n_rare,
            "notes": notes,
        }

    emm = merged.set_index("ensg")
    rows = [
        _row("BRR_baseline", "train", btr, variant_scope="common_only"),
        _row("BRR_baseline", "test", bte, variant_scope="common_only"),
        _row("Emmental_common", "train", emm["r2_train_common"], variant_scope="common_only"),
        _row("Emmental_common", "test", emm["r2_test_common"], variant_scope="common_only"),
        _row("Emmental_full_panel", "train", emm["r2_train_all"], variant_scope="full_panel"),
        _row("Emmental_full_panel", "test", emm["r2_test_all"], variant_scope="full_panel"),
        _row("Emmental_rare_only", "train", emm["r2_train_rare"], variant_scope="rare_only"),
        _row("Emmental_rare_only", "test", emm["r2_test_rare"], variant_scope="rare_only"),
    ]
    return pd.DataFrame(rows)


def _plot_abs_mu_vs_maf(var_df: pd.DataFrame, out_path: str) -> None:
    if var_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, stratum, sub in [
        (axes[0], "common", var_df[~var_df["is_rare"]]),
        (axes[1], "rare", var_df[var_df["is_rare"]]),
    ]:
        x = sub["maf_train"].astype(float).clip(lower=1e-6)
        y = sub["abs_mu"].astype(float).clip(lower=1e-12)
        ax.scatter(x, y, s=3, alpha=0.15, rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Train MAF")
        ax.set_ylabel("|μ|")
        ax.set_title(f"{stratum} variants (n={len(sub):,})")
    fig.suptitle("|μ| vs MAF (post-train full panel)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_r2_decomposition(r2_df: pd.DataFrame, out_path: str) -> None:
    if r2_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, split in zip(axes, ["train", "test"]):
        cols = [f"r2_{split}_common", f"r2_{split}_rare", f"r2_{split}_all"]
        data = [r2_df[c].replace([np.inf, -np.inf], np.nan).dropna().values for c in cols]
        bp = ax.boxplot(data, labels=["common", "rare", "all"], showfliers=False)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("R²")
        ax.set_title(split)
    fig.suptitle("Per-gene R² decomposition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_gate_effect(gate_df: pd.DataFrame, out_path: str) -> None:
    if gate_df.empty:
        return
    sub = gate_df[gate_df["is_rare"]]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.abs(sub["mu_no_gate"].astype(float))
    y = np.abs(sub["mu_gated"].astype(float))
    ax.scatter(x, y, s=4, alpha=0.2, rasterized=True)
    lim = max(x.max(), y.max(), 1e-6)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|μ| no gate")
    ax.set_ylabel("|μ| gated (full λ)")
    ax.set_title(f"Rare variants (n={len(sub):,})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _tau_for_gene_chr(
    pergene_root: Optional[str],
    gene_name: str,
    joint_tau_df: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    if pergene_root:
        chr_tag = gene_name.split("/")[0]
        run_dirs = _discover_pergene_run_dirs(pergene_root, chr_tag)
        mean_df, th, tau1, tau2 = _load_tau_for_pergene_chr(run_dirs)
    elif joint_tau_df is not None:
        th = float(joint_tau_df["Filter Threshold"].iloc[0])
        tau1, tau2 = load_data._tau_vectors_from_summary_df(joint_tau_df)
        mean_df = joint_tau_df
    else:
        raise ValueError("Need pergene_root or joint_tau_df for gate comparison")
    ann = mean_df["Annotation"].astype(str).tolist()
    ann_no_int = [a for a in ann if a.lower() != "intercept"]
    return tau1, tau2, th, ann_no_int


def main() -> int:
    p = argparse.ArgumentParser(description="Post-train rare-variant diagnostics.")
    p.add_argument(
        "--config_dir",
        default=os.path.join(TRAIN_COMMON01_DEFAULT, "full"),
        help="Experiment config root (contains pergene/, post_pergene/, etc.).",
    )
    p.add_argument(
        "--post_source",
        choices=("post_pergene", "post_joint"),
        default="post_pergene",
    )
    p.add_argument("--post_dir", default=None, help="Override post output dir.")
    p.add_argument("--gene_list", default=None)
    p.add_argument("--chromosome", default=None)
    p.add_argument(
        "--baseline_root",
        default=BASELINE_ROOT_DEFAULT,
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Default: {config_dir}/diagnostics/{post_source}/",
    )
    p.add_argument(
        "--skip_r2",
        action="store_true",
        help="Skip R² decomposition (variant-level + gate only).",
    )
    p.add_argument(
        "--skip_gate",
        action="store_true",
        help="Skip T-gate vs no-gate recomputation (needs Z reload per gene).",
    )
    p.add_argument(
        "--max_genes",
        type=int,
        default=None,
        help="Limit genes for R² / gate passes (variant CSV scan uses all).",
    )
    args = p.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    post_dir = os.path.abspath(
        args.post_dir
        or os.path.join(config_dir, args.post_source)
    )
    out_dir = os.path.abspath(
        args.out_dir or os.path.join(config_dir, "diagnostics", args.post_source)
    )
    os.makedirs(out_dir, exist_ok=True)

    utils.setup_logging("INFO", None)
    logger = utils.get_logger()

    gene_allow = _load_gene_list(args.gene_list) if args.gene_list else None
    beta_paths = discover_beta_panel_files(post_dir, args.chromosome)
    beta_paths = _filter_paths_by_gene_list(beta_paths, gene_allow)
    if not beta_paths:
        logger.error("No full_beta_panel CSVs under %s", post_dir)
        return 1
    logger.info("Found %d full_beta_panel files", len(beta_paths))

    pergene_root = os.path.join(config_dir, "pergene")
    try:
        base_config = _load_pergene_root_config(pergene_root)
    except FileNotFoundError:
        with open(os.path.join(config_dir, "joint", "config.yaml")) as f:
            base_config = yaml.safe_load(f) or {}
    maf_thr = _maf_threshold_from_config(base_config, 0.01)

    # --- A1: variant-level μ / λ from stored CSVs (fast, all genes) ---
    logger.info("Collecting variant-level μ / λ diagnostics...")
    var_df = collect_variant_level_from_csv(beta_paths, maf_thr)
    var_df.to_csv(os.path.join(out_dir, "variant_level.csv.gz"), index=False, compression="gzip")

    maf_summary, stratum_summary = summarize_mu_maf_lambda(var_df)
    maf_summary.to_csv(os.path.join(out_dir, "abs_mu_vs_maf_binned.csv"), index=False)
    stratum_summary.to_csv(os.path.join(out_dir, "mu_lambda_by_stratum.csv"), index=False)
    _plot_abs_mu_vs_maf(var_df, os.path.join(out_dir, "abs_mu_vs_maf_scatter.png"))

    # --- A2/A3: R² decomposition + gate (per gene, slower) ---
    genes_for_heavy = [_parse_gene_from_beta_path(p) for p in beta_paths]
    genes_for_heavy = [g for g in genes_for_heavy if g]
    if args.max_genes is not None:
        genes_for_heavy = genes_for_heavy[: args.max_genes]

    joint_tau_df = None
    if args.post_source == "post_joint":
        joint_root = os.path.join(config_dir, "joint")
        joint_tau_df = load_data.aggregate_tau_t_from_joint_runs(joint_root)

    r2_rows: List[Dict[str, Any]] = []
    gate_chunks: List[pd.DataFrame] = []

    if not args.skip_r2 or not args.skip_gate:
        train_idx, test_idx = load_data.get_train_test_indices(base_config["covariates_path"])
        device = torch.device("cpu")
        cov_scaled = None
        expr_cache = None
        if not args.skip_r2:
            cov_scaled = _load_covariates_scaled(base_config)
            expr_cache = _ExpressionResidualizer(base_config, cov_scaled, device)

        for i, path in enumerate(beta_paths):
            gene = _parse_gene_from_beta_path(path)
            if gene is None:
                continue
            if args.max_genes is not None and gene not in genes_for_heavy:
                continue
            beta_df = pd.read_csv(path, compression="infer")
            if not args.skip_r2 and expr_cache is not None:
                try:
                    r2_rows.append(
                        compute_r2_decomposition_for_gene(
                            gene,
                            beta_df,
                            base_config,
                            train_idx,
                            test_idx,
                            cov_scaled,
                            expr_cache,
                            device,
                        )
                    )
                except Exception as e:
                    r2_rows.append({"gene": gene, "error": repr(e)})
            if not args.skip_gate:
                try:
                    tau1, tau2, th, ann_names = _tau_for_gene_chr(
                        pergene_root if args.post_source == "post_pergene" else None,
                        gene,
                        joint_tau_df,
                    )
                    gate_chunks.append(
                        compute_gate_comparison_for_gene(
                            gene,
                            beta_df,
                            tau1,
                            tau2,
                            th,
                            ann_names,
                            base_config,
                            train_idx,
                            device,
                        )
                    )
                except Exception as e:
                    logger.warning("%s gate comparison failed: %s", gene, e)
            if (i + 1) % 50 == 0:
                logger.info("Processed %d / %d genes (R²/gate)", i + 1, len(beta_paths))

    if r2_rows:
        r2_df = pd.DataFrame(r2_rows)
        r2_df.to_csv(os.path.join(out_dir, "r2_decomposition_per_gene.csv"), index=False)
        r2_ok = r2_df[~r2_df["gene"].isna()]
        if "error" in r2_ok.columns:
            r2_ok = r2_ok[r2_ok["error"].isna()]
        r2_summary = summarize_r2_decomposition(r2_ok)
        r2_summary.to_csv(os.path.join(out_dir, "r2_decomposition_summary.csv"), index=False)
        if len(r2_ok):
            _plot_r2_decomposition(r2_ok, os.path.join(out_dir, "r2_decomposition_boxplot.png"))
    else:
        r2_df = pd.DataFrame()
        r2_ok = pd.DataFrame()

    if gate_chunks:
        gate_df = pd.concat(gate_chunks, ignore_index=True)
        gate_df.to_csv(os.path.join(out_dir, "gate_comparison_variants.csv.gz"), index=False, compression="gzip")
        gate_summary = summarize_gate_comparison(gate_df)
        gate_summary.to_csv(os.path.join(out_dir, "gate_comparison_summary.csv"), index=False)
        ann_summary = summarize_lambda_zero_by_annotation(gate_df)
        ann_summary.to_csv(os.path.join(out_dir, "lambda_zero_by_dominant_annotation.csv"), index=False)
        _plot_gate_effect(gate_df, os.path.join(out_dir, "gate_mu_rare_scatter.png"))
    else:
        gate_df = pd.DataFrame()

    # --- Methods comparison table ---
    gene_list_path = args.gene_list
    if gene_list_path and os.path.isfile(gene_list_path):
        with open(gene_list_path) as f:
            gene_ensgs = [
                _ensg_from_gene(line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    elif not r2_df.empty:
        gene_ensgs = sorted(r2_df["gene"].map(_ensg_from_gene).unique())
    else:
        gene_ensgs = sorted(var_df["gene"].map(_ensg_from_gene).unique())

    if not r2_df.empty:
        methods_df = build_methods_comparison_table(
            r2_df, var_df, args.baseline_root, gene_ensgs=gene_ensgs
        )
        methods_df.to_csv(os.path.join(out_dir, "methods_comparison.csv"), index=False)

    manifest = {
        "config_dir": config_dir,
        "post_dir": post_dir,
        "post_source": args.post_source,
        "n_beta_panel_files": len(beta_paths),
        "n_variants_pooled": int(len(var_df)),
        "maf_threshold": maf_thr,
        "n_genes_r2": int(len(r2_df)),
        "n_variants_gate": int(len(gate_df)),
        "gene_list": args.gene_list,
        "chromosome": args.chromosome,
        "max_genes": args.max_genes,
        "outputs": sorted(os.listdir(out_dir)),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Diagnostics written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
