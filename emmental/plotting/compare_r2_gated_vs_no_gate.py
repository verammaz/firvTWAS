#!/usr/bin/env python3
"""
Compare post-train R² with gated vs no-gate rare μ (ρ·w·λ).

Gated: rare β from saved ``mu_rho_w_lambda`` (T gate applied at post-train).
No-gate: rare β = ρ·w·(lin1·exp(lin2)·MAF) without |Z·τ₁| ≥ T gate.

Writes per-gene CSV + proportion summary/plots under {experiment}/plots/.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import load_data  # noqa: E402
import utils  # noqa: E402
from emmental_post_train_betas import (  # noqa: E402
    _ExpressionResidualizer,
    _load_covariates_scaled,
    _load_gene_G_Z,
    _load_gene_list,
    _load_tau_for_pergene_chr,
    _normalize_chr_tag,
    _r2_predict,
)

from diagnose_posttrain_rare import _lambda_no_gate  # noqa: E402


def _parse_gene_from_beta_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.match(r"^(chr\d+)_(ENSG\d+)_full_panel_beta\.csv\.gz$", base)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def discover_post_pergene_beta_paths(post_root: str) -> List[str]:
    pattern = os.path.join(
        os.path.abspath(post_root), "chr*", "full_beta_panel", "*_full_panel_beta.csv.gz"
    )
    return sorted(glob.glob(pattern))


def _r2_for_gene(
    gene_name: str,
    beta_df: pd.DataFrame,
    base_config: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    expr_cache: _ExpressionResidualizer,
    tau1: np.ndarray,
    tau2: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    gene_cfg = dict(base_config)
    gene_cfg["genes"] = [gene_name]
    gene_cfg["maf_threshold"] = None

    G, G_raw, Z_ann, _, _ = _load_gene_G_Z(gene_cfg)
    if G.empty or len(beta_df) != len(G.columns):
        return None

    common_mask = beta_df["is_common_maf_ge_threshold"].astype(bool).values
    rare_mask = ~common_mask
    beta_common_only = np.where(
        common_mask,
        beta_df["beta_full_common_csv_rare_mu"].astype(np.float64).values,
        0.0,
    )
    beta_all_gated = beta_df["beta_full_common_csv_rare_mu"].astype(np.float64).values

    G_train_raw = G_raw.iloc[train_idx]
    G_t = torch.as_tensor(np.asarray(G_train_raw.values, dtype=np.float32), device=device)
    maf_beta = float(base_config.get("maf_beta", 1.0))
    maf_weights = utils.get_MAF_weights(G_t, device, maf_beta)
    lam_nogate = _lambda_no_gate(Z_ann, maf_weights, threshold, tau1, tau2, device)
    rho = float(beta_df["rho_g"].iloc[0])
    w = float(beta_df["w_g"].iloc[0])
    mu_nogate = rho * w * lam_nogate
    if base_config.get("normalize_G", False):
        eps = 1e-6
        std_raw = G_train_raw.std(axis=0, ddof=0).fillna(1.0).values
        std_raw = np.maximum(std_raw, eps)
        mu_nogate = mu_nogate * std_raw
    beta_all_nogate = np.where(common_mask, beta_all_gated, mu_nogate)

    y_all = expr_cache.residualize(gene_name)
    G_tr = G.iloc[train_idx]
    G_te = G.iloc[test_idx]
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]
    full_mask = np.ones(len(beta_all_gated), dtype=bool)

    return {
        "gene": gene_name,
        "n_variants": len(beta_all_gated),
        "n_rare": int(rare_mask.sum()),
        "n_gate_killed": int(
            (
                rare_mask
                & (np.abs(beta_df["mu_rho_w_lambda"].astype(np.float64)) <= 1e-12)
                & (np.abs(mu_nogate) > 1e-12)
            ).sum()
        ),
        "r2_train_common": _r2_predict(G_tr, y_tr, beta_common_only, common_mask),
        "r2_test_common": _r2_predict(G_te, y_te, beta_common_only, common_mask),
        "r2_train_all_gated": _r2_predict(G_tr, y_tr, beta_all_gated, full_mask),
        "r2_test_all_gated": _r2_predict(G_te, y_te, beta_all_gated, full_mask),
        "r2_train_all_nogate": _r2_predict(G_tr, y_tr, beta_all_nogate, full_mask),
        "r2_test_all_nogate": _r2_predict(G_te, y_te, beta_all_nogate, full_mask),
    }


def _prop_summary(r2_df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for model, col in [
            ("common_only", f"r2_{split}_common"),
            ("all_gated", f"r2_{split}_all_gated"),
            ("all_no_gate", f"r2_{split}_all_nogate"),
        ]:
            vals = r2_df[col].astype(float)
            for thr in thresholds:
                rows.append(
                    {
                        "split": split,
                        "model": model,
                        "threshold": thr,
                        "prop": float((vals > thr).mean()),
                        "n_genes": int(vals.notna().sum()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_props(props_df: pd.DataFrame, thr: float, out_path: str, title: str) -> None:
    sub = props_df[props_df["threshold"] == thr].copy()
    models = ["common_only", "all_gated", "all_no_gate"]
    labels = {
        "common_only": "Common only",
        "all_gated": "All (gated rare)",
        "all_no_gate": "All (no-gate rare)",
    }
    splits = ["train", "test"]
    x = np.arange(len(models))
    width = 0.36
    colors = {"train": "#4C72B0", "test": "#DD8452"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, split in enumerate(splits):
        vals = []
        for m in models:
            row = sub[(sub["model"] == m) & (sub["split"] == split)]
            vals.append(float(row["prop"].iloc[0]) if len(row) else float("nan"))
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=split.capitalize(), color=colors[split])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models], rotation=15, ha="right")
    ax.set_ylabel(f"Fraction of genes with R² > {thr}")
    ax.set_ylim(0, min(1.05, max(0.15, ax.get_ylim()[1] + 0.08)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="R² gated vs no-gate rare μ")
    p.add_argument(
        "--experiment_root",
        required=True,
        help="Experiment dir with joint/config.yaml and post_pergene/",
    )
    p.add_argument(
        "--post_dir",
        default=None,
        help="Post-train dir (default: {experiment_root}/post_pergene)",
    )
    p.add_argument("--out_dir", default=None, help="Default: {experiment_root}/plots")
    p.add_argument("--max_genes", type=int, default=None)
    p.add_argument(
        "--gene_list",
        default=None,
        help="Optional chr/ENSG list file (one gene per line) to restrict comparison.",
    )
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.1])
    args = p.parse_args()

    exp_root = os.path.abspath(args.experiment_root)
    post_dir = os.path.abspath(args.post_dir or os.path.join(exp_root, "post_pergene"))
    out_dir = os.path.abspath(args.out_dir or os.path.join(exp_root, "plots"))
    os.makedirs(out_dir, exist_ok=True)

    cfg_path = os.path.join(exp_root, "joint", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cov_scaled = _load_covariates_scaled(cfg)
    train_idx, test_idx = load_data.get_train_test_indices(cfg["covariates_path"])
    expr_cache = _ExpressionResidualizer(cfg, cov_scaled, device)

    pergene_root = os.path.join(exp_root, "pergene")
    tau_cache: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    paths = discover_post_pergene_beta_paths(post_dir)
    if args.gene_list:
        gene_allow = _load_gene_list(args.gene_list)
        filtered = []
        for path in paths:
            gene = _parse_gene_from_beta_path(path)
            if gene and gene in gene_allow:
                filtered.append(path)
        paths = filtered
    if args.max_genes:
        paths = paths[: args.max_genes]
    if not paths:
        print(f"No beta panels under {post_dir}")
        return 1

    rows: List[Dict[str, Any]] = []
    for path in tqdm(paths, desc="R² gated vs no-gate"):
        gene = _parse_gene_from_beta_path(path)
        if gene is None:
            continue
        beta_df = pd.read_csv(path)
        chr_tag = gene.split("/")[0]
        if chr_tag not in tau_cache:
            chr_dir = os.path.join(pergene_root, chr_tag)
            _, th, tau1, tau2 = _load_tau_for_pergene_chr([chr_dir])
            tau_cache[chr_tag] = (tau1, tau2, th)
        tau1, tau2, th = tau_cache[chr_tag]
        try:
            row = _r2_for_gene(
                gene,
                beta_df,
                cfg,
                train_idx,
                test_idx,
                expr_cache,
                tau1,
                tau2,
                th,
                device,
            )
        except Exception as e:
            print(f"SKIP {gene}: {e}")
            continue
        if row:
            rows.append(row)

    if not rows:
        print("No genes processed.")
        return 1

    r2_df = pd.DataFrame(rows)
    tag = os.path.basename(exp_root.rstrip("/"))
    r2_path = os.path.join(out_dir, f"{tag}_r2_gated_vs_nogate_pergene.csv")
    r2_df.to_csv(r2_path, index=False)

    props = _prop_summary(r2_df, args.thresholds)
    props_path = os.path.join(out_dir, f"{tag}_r2_gated_vs_nogate_prop_summary.csv")
    props.to_csv(props_path, index=False)

    for thr in args.thresholds:
        slug = str(thr).replace(".", "p")
        _plot_props(
            props,
            thr,
            os.path.join(out_dir, f"{tag}_prop_r2_gt_{slug}_gated_vs_nogate.png"),
            f"{tag} pergene (n={len(r2_df):,}): R² > {thr} (gated vs no-gate rare)",
        )

    ok = r2_df.dropna(
        subset=["r2_train_all_gated", "r2_train_all_nogate", "r2_train_common"]
    )
    print(f"Genes: {len(r2_df):,} | gate-killed rare variants (median/gene): "
          f"{r2_df['n_gate_killed'].median():.0f}")
    if len(ok):
        d_tr = ok["r2_train_all_nogate"] - ok["r2_train_all_gated"]
        d_te = ok["r2_test_all_nogate"] - ok["r2_test_all_gated"]
        print(
            f"ΔR² (no-gate − gated) train: mean={d_tr.mean():.6f} median={d_tr.median():.6f} "
            f"| test: mean={d_te.mean():.6f} median={d_te.median():.6f}"
        )
        for thr in args.thresholds:
            for split in ("train", "test"):
                for model in ("all_gated", "all_no_gate"):
                    row = props[(props.threshold == thr) & (props.split == split) & (props.model == model)]
                    if len(row):
                        print(f"  {split} R²>{thr} {model}: {float(row['prop'].iloc[0]):.4f}")
    print(f"Wrote {r2_path}")
    print(f"Wrote {props_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
