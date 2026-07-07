#!/usr/bin/env python3
"""
Compare per-gene R² using joint vs pergene ``beta_mean`` (common variants only).

Also summarizes |β| and |μ| by common/rare stratum from post_pergene full_beta_panel CSVs.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

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
from emmental_post_train_betas import (  # noqa: E402
    _ExpressionResidualizer,
    _align_beta_to_G_columns,
    _load_covariates_scaled,
    _load_gene_G_Z,
    _load_joint_beta_df,
    _load_pergene_beta_df,
    _load_pergene_root_config,
    _maf_threshold_from_config,
    _r2_predict,
)

GENE_LIST_DEFAULT = (
    "/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt"
)


def _load_gene_list(path: str) -> List[str]:
    genes = []
    with open(path) as f:
        for line in f:
            g = line.strip()
            if g and not g.startswith("#"):
                genes.append(g)
    return genes


def _common_mask_from_maf(G_raw: pd.DataFrame, train_idx: np.ndarray, maf_thr: float) -> np.ndarray:
    maf = load_data.variant_maf_series(G_raw.iloc[train_idx])
    return (maf >= maf_thr).reindex(G_raw.columns).fillna(False).values


def _r2_for_beta_source(
    gene_name: str,
    cfg: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    expr_cache: _ExpressionResidualizer,
    beta_df: Optional[pd.DataFrame],
    maf_thr: float,
    logger,
) -> Dict[str, Any]:
    gene_cfg = dict(cfg)
    gene_cfg["genes"] = [gene_name]
    gene_cfg["maf_threshold"] = None
    G_model, G_raw, _, _, _ = _load_gene_G_Z(gene_cfg)
    if G_model.empty:
        return {"gene": gene_name, "error": "empty G"}

    beta, method = _align_beta_to_G_columns(
        beta_df, gene_name, G_model.columns, cfg, train_idx, logger
    )
    if not np.isfinite(beta).any():
        return {"gene": gene_name, "error": "no aligned beta", "align": method}

    common_mask = _common_mask_from_maf(G_raw, train_idx, maf_thr)
    y = expr_cache.residualize(gene_name)
    G_tr, G_te = G_model.iloc[train_idx], G_model.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    return {
        "gene": gene_name,
        "n_variants": len(beta),
        "n_common": int(common_mask.sum()),
        "align_method": method,
        "r2_train": _r2_predict(G_tr, y_tr, beta, common_mask),
        "r2_test": _r2_predict(G_te, y_te, beta, common_mask),
    }


def _aggregate_effect_sizes(post_pergene_root: str, max_files: Optional[int] = None) -> pd.DataFrame:
    pattern = os.path.join(
        os.path.abspath(post_pergene_root),
        "chr*",
        "full_beta_panel",
        "*_full_panel_beta.csv.gz",
    )
    paths = sorted(glob.glob(pattern))
    if max_files:
        paths = paths[:max_files]

    rows = []
    for path in tqdm(paths, desc="effect sizes", leave=False):
        df = pd.read_csv(
            path,
            usecols=[
                "is_common_maf_ge_threshold",
                "beta_mean_from_pergene_csv",
                "mu_rho_w_lambda",
                "maf_train",
            ],
        )
        common = df["is_common_maf_ge_threshold"].astype(bool)
        rare = ~common
        b_common = df.loc[common, "beta_mean_from_pergene_csv"].astype(float).abs()
        b_rare_mu = df.loc[rare, "mu_rho_w_lambda"].astype(float).abs()
        rows.append(
            {
                "gene_file": os.path.basename(path),
                "n_common": int(common.sum()),
                "n_rare": int(rare.sum()),
                "median_abs_beta_common": float(b_common.median()) if len(b_common) else np.nan,
                "median_abs_mu_rare": float(b_rare_mu.median()) if len(b_rare_mu) else np.nan,
                "mean_abs_beta_common": float(b_common.mean()) if len(b_common) else np.nan,
                "mean_abs_mu_rare": float(b_rare_mu.mean()) if len(b_rare_mu) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out["ratio_median_mu_over_beta"] = out["median_abs_mu_rare"] / out["median_abs_beta_common"]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Joint vs pergene R² + effect-size summary")
    p.add_argument("--experiment_root", required=True)
    p.add_argument("--gene_list", default=GENE_LIST_DEFAULT)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--post_pergene_root", default=None)
    p.add_argument("--max_effect_genes", type=int, default=None)
    p.add_argument("--max_genes", type=int, default=None, help="Limit gene list for quick runs")
    args = p.parse_args()

    exp_root = os.path.abspath(args.experiment_root)
    joint_dir = os.path.join(exp_root, "joint")
    pergene_root = os.path.join(exp_root, "pergene")
    post_pergene = os.path.abspath(args.post_pergene_root or os.path.join(exp_root, "post_pergene"))
    out_dir = os.path.abspath(args.out_dir or os.path.join(exp_root, "diagnostics"))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(joint_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f) or {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cov_scaled = _load_covariates_scaled(cfg)
    train_idx, test_idx = load_data.get_train_test_indices(cfg["covariates_path"])
    expr_cache = _ExpressionResidualizer(cfg, cov_scaled, device)
    maf_thr = _maf_threshold_from_config(cfg, 0.01)
    logger = load_data.get_logger()

    genes = _load_gene_list(args.gene_list)
    if args.max_genes:
        genes = genes[: args.max_genes]
    rows = []
    for gene in tqdm(genes, desc="R² joint vs pergene"):
        chr_tag = gene.split("/")[0]
        joint_beta = _load_joint_beta_df(joint_dir, gene, cfg, train_idx)
        pergene_beta = _load_pergene_beta_df(pergene_root, chr_tag, gene, cfg, train_idx)

        gene_cfg = dict(cfg)
        gene_cfg["genes"] = [gene]
        gene_cfg["maf_threshold"] = None
        G_model, G_raw, _, _, _ = _load_gene_G_Z(gene_cfg)
        if G_model.empty:
            for source in ("joint", "pergene"):
                rows.append({"gene": gene, "beta_source": source, "error": "empty G"})
            continue
        common_mask = _common_mask_from_maf(G_raw, train_idx, maf_thr)
        y = expr_cache.residualize(gene_name=gene)
        G_tr, G_te = G_model.iloc[train_idx], G_model.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        for source, beta_df in [("joint", joint_beta), ("pergene", pergene_beta)]:
            if beta_df is None:
                rows.append({"gene": gene, "beta_source": source, "error": "missing beta"})
                continue
            beta, method = _align_beta_to_G_columns(
                beta_df, gene, G_model.columns, cfg, train_idx, logger
            )
            if not np.isfinite(beta).any():
                rows.append(
                    {"gene": gene, "beta_source": source, "error": "no aligned beta", "align": method}
                )
                continue
            rows.append(
                {
                    "gene": gene,
                    "beta_source": source,
                    "n_variants": len(beta),
                    "n_common": int(common_mask.sum()),
                    "align_method": method,
                    "r2_train": _r2_predict(G_tr, y_tr, beta, common_mask),
                    "r2_test": _r2_predict(G_te, y_te, beta, common_mask),
                }
            )

    r2_df = pd.DataFrame(rows)
    r2_path = os.path.join(out_dir, "joint_vs_pergene_r2_per_gene.csv")
    r2_df.to_csv(r2_path, index=False)

    ok = r2_df[r2_df["r2_test"].notna()].copy()
    if len(ok):
        wide = ok.pivot_table(index="gene", columns="beta_source", values=["r2_train", "r2_test"])
        summary_rows = []
        for split in ("train", "test"):
            j = wide[(f"r2_{split}", "joint")]
            pg = wide[(f"r2_{split}", "pergene")]
            aligned = pd.concat([j, pg], axis=1, dropna=True)
            aligned.columns = ["joint", "pergene"]
            diff = aligned["joint"] - aligned["pergene"]
            summary_rows.append(
                {
                    "split": split,
                    "n_genes": len(aligned),
                    "mean_r2_joint": float(aligned["joint"].mean()),
                    "mean_r2_pergene": float(aligned["pergene"].mean()),
                    "mean_joint_minus_pergene": float(diff.mean()),
                    "median_joint_minus_pergene": float(diff.median()),
                    "frac_joint_better": float((diff > 0).mean()),
                    "corr_r2": float(aligned["joint"].corr(aligned["pergene"])),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_dir, "joint_vs_pergene_r2_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(summary_df.to_string(index=False))
        print(f"Wrote {summary_path}")

    eff = _aggregate_effect_sizes(post_pergene, max_files=args.max_effect_genes)
    eff_path = os.path.join(out_dir, "effect_size_common_vs_rare_summary.csv")
    eff.to_csv(eff_path, index=False)
    if len(eff):
        finite = eff.dropna(subset=["median_abs_beta_common", "median_abs_mu_rare"])
        finite = finite[finite["n_rare"] > 0]
        print("\nEffect sizes (|pergene β| common vs |μ| rare):")
        print(f"  genes: {len(finite)}")
        print(
            f"  median |β_common|: {finite['median_abs_beta_common'].median():.6f}  "
            f"median |μ_rare|: {finite['median_abs_mu_rare'].median():.6f}"
        )
        print(
            f"  frac genes with median |μ_rare| > median |β_common|: "
            f"{(finite['ratio_median_mu_over_beta'] > 1).mean():.3f}"
        )
    print(f"Wrote {r2_path}")
    print(f"Wrote {eff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
