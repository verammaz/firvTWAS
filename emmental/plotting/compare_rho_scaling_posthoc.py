#!/usr/bin/env python3
"""
Post-hoc R² with scaled rare μ (no retraining).

Compares common-only vs full-panel predictors under different rare μ scalings:
  - current: β from post-train (ρ·w·λ on rare, gated)
  - rho_1:   rare μ = w·λ
  - rho_05:  rare μ = 0.5·w·λ

Uses the same genotype / normalize_G handling as emmental_post_train_betas.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import load_data  # noqa: E402
from emmental_post_train_betas import (  # noqa: E402
    _ExpressionResidualizer,
    _load_covariates_scaled,
    _load_gene_G_Z,
    _load_gene_list,
    _r2_predict,
)

RHO_SCENARIOS = ("current", "rho_1", "rho_05")


def _parse_gene_from_beta_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.match(r"^(chr\d+)_(ENSG\d+)_full_panel_beta\.csv\.gz$", base)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def discover_beta_paths(post_root: str) -> List[str]:
    pattern = os.path.join(
        os.path.abspath(post_root), "chr*", "full_beta_panel", "*_full_panel_beta.csv.gz"
    )
    return sorted(glob.glob(pattern))


def _mu_to_beta_units(
    mu_raw: np.ndarray,
    G_raw_train: pd.DataFrame,
    G_columns: pd.Index,
    normalize_G: bool,
) -> np.ndarray:
    mu_beta = np.asarray(mu_raw, dtype=np.float64).copy()
    if normalize_G:
        eps = 1e-6
        std_raw = G_raw_train.std(axis=0, ddof=0).reindex(G_columns).fillna(1.0).values
        mu_beta = mu_beta * np.maximum(std_raw, eps)
    return mu_beta


def _build_beta_full(
    beta_current: np.ndarray,
    rare_mask: np.ndarray,
    mu_raw: np.ndarray,
    G_raw_train: pd.DataFrame,
    G_columns: pd.Index,
    normalize_G: bool,
) -> np.ndarray:
    out = np.asarray(beta_current, dtype=np.float64).copy()
    mu_beta = _mu_to_beta_units(mu_raw, G_raw_train, G_columns, normalize_G)
    out[rare_mask] = mu_beta[rare_mask]
    return out


def _score_gene(
    gene_name: str,
    beta_df: pd.DataFrame,
    base_config: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    expr_cache: _ExpressionResidualizer,
    device,
) -> Optional[Dict[str, Any]]:
    gene_cfg = dict(base_config)
    gene_cfg["genes"] = [gene_name]
    gene_cfg["maf_threshold"] = None

    G, G_raw, _, _, _ = _load_gene_G_Z(gene_cfg)
    if G.empty or len(beta_df) != len(G.columns):
        return None

    common_mask = beta_df["is_common_maf_ge_threshold"].astype(bool).values
    rare_mask = ~common_mask
    normalize_G = bool(base_config.get("normalize_G", False))

    beta_current = beta_df["beta_full_common_csv_rare_mu"].astype(np.float64).values
    lam = beta_df["lambda_from_tau_Z_ann_maf"].astype(np.float64).values
    w = float(beta_df["w_g"].iloc[0])
    rho = float(beta_df["rho_g"].iloc[0])

    G_raw_tr = G_raw.iloc[train_idx]
    G_tr = G.iloc[train_idx]
    G_te = G.iloc[test_idx]
    y_all = expr_cache.residualize(gene_name)
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]

    beta_common_only = np.where(common_mask, beta_current, 0.0)
    full_mask = np.ones(len(beta_current), dtype=bool)

    mu_by_scenario = {
        "current": beta_df["mu_rho_w_lambda"].astype(np.float64).values,
        "rho_1": w * lam,
        "rho_05": 0.5 * w * lam,
    }

    row: Dict[str, Any] = {
        "gene": gene_name,
        "n_variants": len(beta_current),
        "n_rare": int(rare_mask.sum()),
        "rho_g": rho,
        "w_g": w,
        "r2_train_common_only": _r2_predict(G_tr, y_tr, beta_common_only, common_mask),
        "r2_test_common_only": _r2_predict(G_te, y_te, beta_common_only, common_mask),
    }
    for scen in RHO_SCENARIOS:
        beta_full = _build_beta_full(
            beta_current,
            rare_mask,
            mu_by_scenario[scen],
            G_raw_tr,
            G.columns,
            normalize_G,
        )
        row[f"r2_train_{scen}"] = _r2_predict(G_tr, y_tr, beta_full, full_mask)
        row[f"r2_test_{scen}"] = _r2_predict(G_te, y_te, beta_full, full_mask)
        row[f"delta_test_{scen}_minus_common"] = (
            row[f"r2_test_{scen}"] - row["r2_test_common_only"]
        )
    return row


def _summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        common_col = f"r2_{split}_common_only"
        for scen in RHO_SCENARIOS:
            col = f"r2_{split}_{scen}"
            vals = df[col].astype(float)
            dcol = f"delta_{split}_{scen}_minus_common"
            if split == "test":
                dvals = df[f"delta_test_{scen}_minus_common"].astype(float)
            else:
                dvals = df[col].astype(float) - df[common_col].astype(float)
            rows.append(
                {
                    "subset": label,
                    "split": split,
                    "scenario": scen,
                    "n_genes": int(vals.notna().sum()),
                    "mean_r2": float(vals.mean()),
                    "median_r2": float(vals.median()),
                    "frac_gt_0.01": float((vals > 0.01).mean()),
                    "frac_gt_0.1": float((vals > 0.1).mean()),
                    "mean_delta_vs_common": float(dvals.mean()),
                    "median_delta_vs_common": float(dvals.median()),
                    "frac_improved": float((dvals > 0).mean()),
                    "frac_improved_gt_0.001": float((dvals > 0.001).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Post-hoc rare μ ρ scaling R² comparison")
    p.add_argument("--experiment_root", required=True)
    p.add_argument("--post_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gene_list", default=None)
    p.add_argument("--random_n", type=int, default=None, help="Random gene subsample size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset_label", default=None)
    args = p.parse_args()

    exp_root = os.path.abspath(args.experiment_root)
    post_dir = os.path.abspath(args.post_dir or os.path.join(exp_root, "post_pergene"))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg_path = os.path.join(exp_root, "joint", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    paths = discover_beta_paths(post_dir)
    if args.gene_list:
        allow = _load_gene_list(args.gene_list)
        filtered = []
        for path in paths:
            gene = _parse_gene_from_beta_path(path)
            if gene and gene in allow:
                filtered.append(path)
        paths = filtered
        label = args.subset_label or "gene_list"
    elif args.random_n is not None:
        rng = random.Random(args.seed)
        paths = sorted(rng.sample(paths, min(args.random_n, len(paths))))
        label = args.subset_label or f"random_{len(paths)}"
    else:
        label = args.subset_label or "all"

    if not paths:
        print("No beta panels to process.")
        return 1

    import torch

    device = torch.device("cpu")
    train_idx, test_idx = load_data.get_train_test_indices(cfg["covariates_path"])
    cov_scaled = _load_covariates_scaled(cfg)
    expr_cache = _ExpressionResidualizer(cfg, cov_scaled, device)

    rows: List[Dict[str, Any]] = []
    for path in tqdm(paths, desc=f"ρ scaling ({label})"):
        gene = _parse_gene_from_beta_path(path)
        if gene is None:
            continue
        beta_df = pd.read_csv(path)
        try:
            row = _score_gene(
                gene, beta_df, cfg, train_idx, test_idx, expr_cache, device
            )
        except Exception as e:
            print(f"SKIP {gene}: {e}")
            continue
        if row:
            rows.append(row)

    if not rows:
        print("No genes scored.")
        return 1

    r2_df = pd.DataFrame(rows)
    tag = os.path.basename(exp_root.rstrip("/"))
    per_gene_path = os.path.join(out_dir, f"{tag}_rho_scaling_{label}_per_gene.csv")
    r2_df.to_csv(per_gene_path, index=False)

    summary = _summarize(r2_df, label)
    summary_path = os.path.join(out_dir, f"{tag}_rho_scaling_{label}_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Subset: {label} | genes: {len(r2_df)}")
    sub = summary[summary["split"] == "test"]
    for scen in RHO_SCENARIOS:
        row = sub[sub["scenario"] == scen].iloc[0]
        print(
            f"  test {scen}: median R²={row['median_r2']:.4f} "
            f"median Δ vs common={row['median_delta_vs_common']:.6f} "
            f"frac improved={row['frac_improved']:.3f}"
        )
    print(f"Wrote {per_gene_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
