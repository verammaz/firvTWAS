#!/usr/bin/env python3
"""
Individual-level expression tail analysis after common-variant PRS.

Computation-only batch script; plotting lives in ``r2.py`` (notebook API).

For each gene:
  1. y = covariate-residualized expression
  2. pred_common = G @ beta_common  (common variants only; rare beta zeroed)
  3. pred_full = G @ assembled(common_beta, rare_beta)
  4. Flag |r_common| residual tails (``resid_tail_quantile``) and expression
     extremes (``expr_tail_quantile``) within each gene and split
  5. AUROC / AUPRC for top/bottom expression (common-only vs common+rare PRS)

Writes ``gene_tail_summary.csv``, ``individual_gene_residuals.csv.gz``, and
``manifest.json``. Gene-level R² deltas and |r|-shrinkage enrichment are derived
from post_pergene / the individual table via ``r2.py`` (not recomputed here).

Beta assembly matches ``emmental_post_train_betas._compute_r2_grid`` (default
``common`` + ``rare`` decoupled posteriors from ``full_beta_panel`` CSVs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import load_data  # noqa: E402
import utils  # noqa: E402
from emmental_post_train_betas import (  # noqa: E402
    COMMON_BETA_SOURCES,
    RARE_BETA_SOURCES,
    _ExpressionResidualizer,
    _assemble_beta_vector,
    _beta_for_source,
    _load_covariates_scaled,
    _load_gene_G_Z,
    _load_gene_list,
    _load_pergene_root_config,
)
from diagnose_posttrain_rare import (  # noqa: E402
    _filter_paths_by_gene_list,
    _parse_gene_from_beta_path,
    discover_beta_panel_files,
)

DEFAULT_COMMON_BETA = "common"
DEFAULT_RARE_BETA = "rare"

PANEL_SOURCE_COLS: Dict[str, str] = {
    "mu": "mu_beta",
    "full": "beta_hat",
    "common": "beta_hat_common_decoupled",
    "train": "beta_common_train_posterior",
    "rare": "beta_hat_rare_decoupled",
}


def _validate_beta_source(source: str, *, rare: bool) -> str:
    allowed = RARE_BETA_SOURCES if rare else COMMON_BETA_SOURCES
    if source not in allowed:
        raise ValueError(
            f"Invalid {'rare' if rare else 'common'} beta source {source!r}; "
            f"expected one of {allowed}"
        )
    return source


def _beta_kw_from_panel(beta_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Load per-source beta vectors from a post-train full_beta_panel CSV."""
    missing = [col for col in PANEL_SOURCE_COLS.values() if col not in beta_df.columns]
    if missing:
        raise KeyError(f"panel missing columns: {missing}")
    arrays: Dict[str, np.ndarray] = {}
    for src, col in PANEL_SOURCE_COLS.items():
        arrays[src] = np.nan_to_num(
            beta_df[col].astype(np.float64).values, nan=0.0
        )
    return {
        "mu_beta": arrays["mu"],
        "beta_full": arrays["full"],
        "beta_common": arrays["common"],
        "beta_rare": arrays["rare"],
        "beta_train": arrays["train"],
    }


def _beta_common_only_vector(
    common_mask: np.ndarray,
    common_src: str,
    beta_kw: Dict[str, np.ndarray],
) -> np.ndarray:
    out = np.zeros(len(common_mask), dtype=np.float64)
    out[common_mask] = _beta_for_source(common_src, **beta_kw)[common_mask]
    return out


def _beta_rare_only_vector(
    rare_mask: np.ndarray,
    rare_src: str,
    beta_kw: Dict[str, np.ndarray],
) -> np.ndarray:
    out = np.zeros(len(rare_mask), dtype=np.float64)
    out[rare_mask] = _beta_for_source(rare_src, **beta_kw)[rare_mask]
    return out


def default_tails_out_dir(config_dir: str, post_dir: str | None = None) -> str:
    """Default output: ``{experiment}/tails`` (sibling of ``post_pergene``)."""
    post_dir = os.path.abspath(post_dir or os.path.join(config_dir, "post_pergene"))
    return os.path.join(os.path.dirname(post_dir), "tails")


def _tail_flags(r: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low, high, and |r| tails at quantile q (each ~q fraction for signed; ~2q for abs union)."""
    r = np.asarray(r, dtype=np.float64)
    lo = np.nanquantile(r, q)
    hi = np.nanquantile(r, 1.0 - q)
    abs_hi = np.nanquantile(np.abs(r), 1.0 - q)
    return r <= lo, r >= hi, np.abs(r) >= abs_hi


def _expr_tail_labels(
    y: np.ndarray, expr_tail_quantile: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Top/bottom expression labels within a split (per gene)."""
    y = np.asarray(y, dtype=np.float64)
    bottom_thr = float(np.nanquantile(y, expr_tail_quantile))
    top_thr = float(np.nanquantile(y, 1.0 - expr_tail_quantile))
    return y <= bottom_thr, y >= top_thr, bottom_thr, top_thr


def _classification_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[float, float]:
    """Return (AUROC, AUPRC); nan when undefined (no positives/negatives or constant score)."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos < 1 or n_neg < 1 or np.std(scores) < 1e-12:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def compute_tail_analysis_for_gene(
    gene_name: str,
    beta_df: pd.DataFrame,
    base_config: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sample_ids: pd.Index,
    expr_cache: _ExpressionResidualizer,
    *,
    common_beta: str = DEFAULT_COMMON_BETA,
    rare_beta: str = DEFAULT_RARE_BETA,
    resid_tail_quantile: float = 0.05,
    expr_tail_quantile: float = 0.01,
    splits: Tuple[str, ...] = ("test",),
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Return (gene summary row, long individual table)."""
    common_beta = _validate_beta_source(common_beta, rare=False)
    rare_beta = _validate_beta_source(rare_beta, rare=True)

    gene_cfg = dict(base_config)
    gene_cfg["genes"] = [gene_name]
    gene_cfg["maf_threshold"] = None

    G, _, _, _, _ = _load_gene_G_Z(gene_cfg)
    if G.empty or len(beta_df) != len(G.columns):
        return {"gene": gene_name, "error": "G/beta shape mismatch"}, pd.DataFrame()

    try:
        beta_kw = _beta_kw_from_panel(beta_df)
    except KeyError as exc:
        return {"gene": gene_name, "error": str(exc)}, pd.DataFrame()

    common_mask = beta_df["is_common_maf_ge_threshold"].astype(bool).values
    rare_mask = ~common_mask

    beta_common_only = _beta_common_only_vector(common_mask, common_beta, beta_kw)
    beta_rare_only = _beta_rare_only_vector(rare_mask, rare_beta, beta_kw)
    beta_full = _assemble_beta_vector(
        common_mask, common_beta, rare_beta, **beta_kw
    )

    y_all = expr_cache.residualize(gene_name)
    pred_common = G.values @ beta_common_only
    pred_rare = G.values @ beta_rare_only
    pred_full = G.values @ beta_full

    r_common = y_all - pred_common
    r_full = y_all - pred_full
    delta_abs_r = np.abs(r_common) - np.abs(r_full)

    summary: Dict[str, Any] = {
        "gene": gene_name,
        "n_variants": int(len(beta_full)),
        "n_common": int(common_mask.sum()),
        "n_rare": int(rare_mask.sum()),
        "resid_tail_quantile": resid_tail_quantile,
        "expr_tail_quantile": expr_tail_quantile,
        "common_beta": common_beta,
        "rare_beta": rare_beta,
    }

    ind_rows: List[Dict[str, Any]] = []
    split_idx_map = {"train": train_idx, "test": test_idx}

    for split in splits:
        idx = split_idx_map[split]
        if len(idx) < 5:
            continue

        y_split = y_all[idx]
        r_split = r_common[idx]
        tail_lo, tail_hi, tail_abs = _tail_flags(r_split, resid_tail_quantile)
        expr_bottom, expr_top, bottom_thr, top_thr = _expr_tail_labels(
            y_split, expr_tail_quantile
        )
        n_tail_abs = int(tail_abs.sum())
        n_expr_top = int(expr_top.sum())
        n_expr_bottom = int(expr_bottom.sum())

        summary[f"n_{split}"] = int(len(idx))
        summary[f"n_{split}_tail_abs"] = n_tail_abs
        summary[f"n_{split}_expr_top"] = n_expr_top
        summary[f"n_{split}_expr_bottom"] = n_expr_bottom
        summary[f"expr_top_thr_{split}"] = top_thr
        summary[f"expr_bottom_thr_{split}"] = bottom_thr

        pc = pred_common[idx]
        pf = pred_full[idx]
        auroc_c_top, auprc_c_top = _classification_metrics(expr_top, pc)
        auroc_f_top, auprc_f_top = _classification_metrics(expr_top, pf)
        auroc_c_bot, auprc_c_bot = _classification_metrics(expr_bottom, -pc)
        auroc_f_bot, auprc_f_bot = _classification_metrics(expr_bottom, -pf)

        summary[f"auroc_{split}_common_top"] = auroc_c_top
        summary[f"auroc_{split}_full_top"] = auroc_f_top
        summary[f"delta_auroc_{split}_top"] = (
            auroc_f_top - auroc_c_top
            if np.isfinite(auroc_f_top) and np.isfinite(auroc_c_top)
            else float("nan")
        )
        summary[f"auprc_{split}_common_top"] = auprc_c_top
        summary[f"auprc_{split}_full_top"] = auprc_f_top
        summary[f"delta_auprc_{split}_top"] = (
            auprc_f_top - auprc_c_top
            if np.isfinite(auprc_f_top) and np.isfinite(auprc_c_top)
            else float("nan")
        )

        summary[f"auroc_{split}_common_bottom"] = auroc_c_bot
        summary[f"auroc_{split}_full_bottom"] = auroc_f_bot
        summary[f"delta_auroc_{split}_bottom"] = (
            auroc_f_bot - auroc_c_bot
            if np.isfinite(auroc_f_bot) and np.isfinite(auroc_c_bot)
            else float("nan")
        )
        summary[f"auprc_{split}_common_bottom"] = auprc_c_bot
        summary[f"auprc_{split}_full_bottom"] = auprc_f_bot
        summary[f"delta_auprc_{split}_bottom"] = (
            auprc_f_bot - auprc_c_bot
            if np.isfinite(auprc_f_bot) and np.isfinite(auprc_c_bot)
            else float("nan")
        )

        r_std = float(np.std(r_split))
        z_common = r_split / r_std if r_std > 1e-12 else r_split * np.nan

        for local_i, global_i in enumerate(idx):
            ind_rows.append(
                {
                    "sample_id": sample_ids[global_i],
                    "gene": gene_name,
                    "split": split,
                    "y": float(y_all[global_i]),
                    "pred_common": float(pred_common[global_i]),
                    "pred_rare": float(pred_rare[global_i]),
                    "pred_full": float(pred_full[global_i]),
                    "r_common": float(r_common[global_i]),
                    "r_full": float(r_full[global_i]),
                    "abs_r_common": float(np.abs(r_common[global_i])),
                    "abs_r_full": float(np.abs(r_full[global_i])),
                    "delta_abs_r": float(delta_abs_r[global_i]),
                    "z_r_common": float(z_common[local_i]),
                    "is_tail_low": bool(tail_lo[local_i]),
                    "is_tail_high": bool(tail_hi[local_i]),
                    "is_tail_abs": bool(tail_abs[local_i]),
                    "is_expr_top": bool(expr_top[local_i]),
                    "is_expr_bottom": bool(expr_bottom[local_i]),
                }
            )

    return summary, pd.DataFrame(ind_rows)


def _summarize_gene_column(
    gene_df: pd.DataFrame, col: str, out: Dict[str, Any], *, wilcoxon_greater: bool = False
) -> None:
    if col not in gene_df.columns:
        return
    s = gene_df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return
    out[f"{col}_median"] = float(s.median())
    out[f"{col}_mean"] = float(s.mean())
    out[f"{col}_frac_gt_0"] = float((s > 0).mean())
    if wilcoxon_greater and len(s) >= 10:
        stat, pval = stats.wilcoxon(s, alternative="greater")
        out[f"{col}_wilcoxon_stat"] = float(stat)
        out[f"{col}_wilcoxon_pval"] = float(pval)


def _genome_summary(gene_df: pd.DataFrame, split: str = "test") -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split, "n_genes": int(len(gene_df))}
    for col in (
        f"delta_auroc_{split}_top",
        f"delta_auroc_{split}_bottom",
        f"delta_auprc_{split}_top",
        f"delta_auprc_{split}_bottom",
    ):
        _summarize_gene_column(
            gene_df,
            col,
            out,
            wilcoxon_greater=col.startswith("delta_auroc")
            or col.startswith("delta_auprc"),
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Expression tail analysis after common PRS.")
    p.add_argument(
        "--config_dir",
        required=True,
        help="Experiment root (contains pergene/, post_pergene/, etc.).",
    )
    p.add_argument(
        "--post_dir",
        default=None,
        help="Post-train root (default: {config_dir}/post_pergene).",
    )
    p.add_argument("--out_dir", default=None, help="Default: {experiment}/tails/")
    p.add_argument("--gene_list", default=None)
    p.add_argument("--chromosome", default=None)
    p.add_argument(
        "--common_beta",
        default=DEFAULT_COMMON_BETA,
        choices=COMMON_BETA_SOURCES,
        help="Common-stratum beta source (matches post-train R² grid).",
    )
    p.add_argument(
        "--rare_beta",
        default=DEFAULT_RARE_BETA,
        choices=RARE_BETA_SOURCES,
        help="Rare-stratum beta source for assembled full PRS.",
    )
    p.add_argument(
        "--resid_tail_quantile",
        type=float,
        default=0.05,
        help="Quantile for low/high/|r_common| residual tails within each gene and split.",
    )
    p.add_argument(
        "--expr_tail_quantile",
        type=float,
        default=0.01,
        help="Quantile for top/bottom expression labels (AUROC/AUPRC).",
    )
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to analyze (train,test). Default: test only.",
    )
    p.add_argument("--max_genes", type=int, default=None)
    args = p.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    post_dir = os.path.abspath(args.post_dir or os.path.join(config_dir, "post_pergene"))
    out_dir = os.path.abspath(
        args.out_dir or default_tails_out_dir(config_dir, post_dir)
    )
    os.makedirs(out_dir, exist_ok=True)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    utils.setup_logging("INFO", None)
    logger = utils.get_logger()

    pergene_root = os.path.join(config_dir, "pergene")
    try:
        base_config = _load_pergene_root_config(pergene_root)
    except FileNotFoundError:
        with open(os.path.join(config_dir, "joint", "config.yaml")) as f:
            base_config = yaml.safe_load(f) or {}

    gene_allow = _load_gene_list(args.gene_list) if args.gene_list else None
    beta_paths = discover_beta_panel_files(post_dir, args.chromosome)
    beta_paths = _filter_paths_by_gene_list(beta_paths, gene_allow)
    if args.max_genes is not None:
        beta_paths = beta_paths[: args.max_genes]
    if not beta_paths:
        logger.error("No full_beta_panel CSVs under %s", post_dir)
        return 1

    train_idx, test_idx = load_data.get_train_test_indices(base_config["covariates_path"])
    cov_scaled = _load_covariates_scaled(base_config)
    expr_cache = _ExpressionResidualizer(base_config, cov_scaled, torch.device("cpu"))
    cov = pd.read_csv(base_config["covariates_path"], sep="\t").set_index("sample_id")
    sample_ids = cov.index

    logger.info(
        "Analyzing %d genes from %s (common_beta=%s, rare_beta=%s)",
        len(beta_paths),
        post_dir,
        args.common_beta,
        args.rare_beta,
    )

    gene_rows: List[Dict[str, Any]] = []
    ind_chunks: List[pd.DataFrame] = []

    for i, path in enumerate(beta_paths):
        gene = _parse_gene_from_beta_path(path)
        if gene is None:
            continue
        beta_df = pd.read_csv(path, compression="infer")
        try:
            summary, ind_df = compute_tail_analysis_for_gene(
                gene,
                beta_df,
                base_config,
                train_idx,
                test_idx,
                sample_ids,
                expr_cache,
                common_beta=args.common_beta,
                rare_beta=args.rare_beta,
                resid_tail_quantile=args.resid_tail_quantile,
                expr_tail_quantile=args.expr_tail_quantile,
                splits=splits,
            )
        except Exception as e:
            summary = {"gene": gene, "error": repr(e)}
            ind_df = pd.DataFrame()
        gene_rows.append(summary)
        if not ind_df.empty:
            ind_chunks.append(ind_df)
        if (i + 1) % 100 == 0:
            logger.info("  processed %d / %d genes", i + 1, len(beta_paths))

    gene_df = pd.DataFrame(gene_rows)
    gene_path = os.path.join(out_dir, "gene_tail_summary.csv")
    gene_df.to_csv(gene_path, index=False)

    ind_df = pd.concat(ind_chunks, ignore_index=True) if ind_chunks else pd.DataFrame()
    ind_path = os.path.join(out_dir, "individual_gene_residuals.csv.gz")
    if not ind_df.empty:
        ind_df.to_csv(ind_path, index=False, compression="gzip")

    manifest: Dict[str, Any] = {
        "config_dir": config_dir,
        "post_dir": post_dir,
        "out_dir": out_dir,
        "n_genes": int(len(gene_df)),
        "n_individual_rows": int(len(ind_df)),
        "resid_tail_quantile": args.resid_tail_quantile,
        "expr_tail_quantile": args.expr_tail_quantile,
        "common_beta": args.common_beta,
        "rare_beta": args.rare_beta,
        "panel_source_cols": PANEL_SOURCE_COLS,
        "splits": splits,
        "gene_list": args.gene_list,
        "chromosome": args.chromosome,
    }
    ok_genes = gene_df[~gene_df["error"].notna()] if "error" in gene_df.columns else gene_df
    for split in splits:
        manifest[f"genome_{split}"] = _genome_summary(ok_genes, split=split)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote %s", gene_path)
    if not ind_df.empty:
        logger.info("Wrote %s (%d rows)", ind_path, len(ind_df))
    logger.info("Tables and manifest under %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
