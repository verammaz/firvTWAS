"""
Plotting and aggregation for ``tails/`` outputs from ``analyze_expression_tails.py``.

Non-R² only: residual shrinkage, AUROC/AUPRC, ROC/PR curves.
Gene-level R² lives in ``r2.py`` + ``post_pergene/``.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def default_tails_dir(experiment_or_post_dir: str) -> str:
    """``{experiment}/tails`` — sibling of ``post_pergene``."""
    path = os.path.abspath(experiment_or_post_dir)
    if os.path.basename(path) == "post_pergene":
        path = os.path.dirname(path)
    return os.path.join(path, "tails")


def load_tails_manifest(tails_dir: str) -> dict[str, Any]:
    path = os.path.join(os.path.abspath(tails_dir), "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def load_gene_summary(tails_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(tails_dir), "gene_tail_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_individual_residuals(tails_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(tails_dir), "individual_gene_residuals.csv.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, compression="infer")


def prep_tails_plotting(
    tails_dir: str,
    split: str = "test",
) -> dict[str, Any]:
    """
    Load ``tails/`` tables and attach per-gene shrinkage metrics for plotting.

    Returns ``gene_df``, ``ind_df``, ``manifest``, and ``genome_summary``.
    """
    tails_dir = os.path.abspath(tails_dir)
    gene_df = load_gene_summary(tails_dir)
    manifest = load_tails_manifest(tails_dir)
    ind_df = pd.DataFrame()
    try:
        ind_df = load_individual_residuals(tails_dir)
        shrink = summarize_residual_shrinkage(ind_df, split=split)
        gene_df = gene_df.merge(shrink, on="gene", how="left")
    except FileNotFoundError:
        pass
    genome_summary = summarize_auroc_genome(gene_df, split=split)
    return {
        "tails_dir": tails_dir,
        "split": split,
        "gene_df": gene_df,
        "ind_df": ind_df,
        "manifest": manifest,
        "genome_summary": genome_summary,
    }


def summarize_residual_shrinkage(
    ind_df: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Per-gene |r|-shrinkage and tail diagnostics from the individual table."""
    sub = ind_df[ind_df["split"] == split]
    rows: list[dict] = []
    for gene, g in sub.groupby("gene", sort=False):
        d = g["delta_abs_r"].astype(float)
        tail = g["is_tail_abs"].astype(bool)
        d_tail = d[tail]
        d_nontail = d[~tail]
        row: dict = {
            "gene": gene,
            f"mean_delta_abs_r_{split}_all": float(d.mean()),
            f"mean_delta_abs_r_{split}_tail": float(d_tail.mean()) if len(d_tail) else float("nan"),
            f"mean_delta_abs_r_{split}_nontail": float(d_nontail.mean()) if len(d_nontail) else float("nan"),
        }
        row[f"tail_enrichment_delta_abs_r_{split}"] = (
            row[f"mean_delta_abs_r_{split}_tail"] - row[f"mean_delta_abs_r_{split}_nontail"]
            if np.isfinite(row[f"mean_delta_abs_r_{split}_tail"])
            and np.isfinite(row[f"mean_delta_abs_r_{split}_nontail"])
            else float("nan")
        )
        if tail.sum() >= 3:
            gt = g.loc[tail]
            pr = gt["pred_rare"].astype(float)
            rc = gt["r_common"].astype(float)
            if np.std(pr) > 1e-12 and np.std(rc) > 1e-12:
                row[f"corr_neg_r_common_pred_rare_{split}_tail"] = float(
                    np.corrcoef(-rc, pr)[0, 1]
                )
            else:
                row[f"corr_neg_r_common_pred_rare_{split}_tail"] = float("nan")
            row[f"sign_opposes_frac_{split}_tail"] = float(
                (np.sign(rc) != np.sign(pr)).mean()
            )
        else:
            row[f"corr_neg_r_common_pred_rare_{split}_tail"] = float("nan")
            row[f"sign_opposes_frac_{split}_tail"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_column(
    gene_df: pd.DataFrame,
    col: str,
    out: dict[str, Any],
    *,
    wilcoxon_greater: bool = False,
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


def summarize_auroc_genome(
    gene_df: pd.DataFrame,
    split: str = "test",
) -> dict[str, Any]:
    """Genome-wide AUROC/AUPRC and shrinkage summaries (recomputes manifest-style stats)."""
    ok = gene_df[~gene_df["error"].notna()] if "error" in gene_df.columns else gene_df
    out: dict[str, Any] = {"split": split, "n_genes": int(len(ok))}
    for col in (
        f"tail_enrichment_delta_abs_r_{split}",
        f"delta_auroc_{split}_top",
        f"delta_auroc_{split}_bottom",
        f"delta_auprc_{split}_top",
        f"delta_auprc_{split}_bottom",
    ):
        _summarize_column(
            ok,
            col,
            out,
            wilcoxon_greater=col.startswith("delta_auroc")
            or col.startswith("delta_auprc")
            or col == f"tail_enrichment_delta_abs_r_{split}",
        )
    return out


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos < 1 or n_neg < 1 or np.std(scores) < 1e-12:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _scores_for_expression_tail(
    ind_df: pd.DataFrame,
    *,
    gene: str,
    split: str,
    expr_extreme: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = ind_df[(ind_df["gene"] == gene) & (ind_df["split"] == split)]
    if sub.empty:
        raise ValueError(f"No rows for gene={gene!r} split={split!r}")
    if expr_extreme == "top":
        labels = sub["is_expr_top"].astype(int).values
        scores_common = sub["pred_common"].astype(float).values
        scores_full = sub["pred_full"].astype(float).values
    elif expr_extreme == "bottom":
        labels = sub["is_expr_bottom"].astype(int).values
        scores_common = -sub["pred_common"].astype(float).values
        scores_full = -sub["pred_full"].astype(float).values
    else:
        raise ValueError(f"expr_extreme must be 'top' or 'bottom', got {expr_extreme!r}")
    return labels, scores_common, scores_full


def plot_shrinkage_enrichment_hist(
    gene_df: pd.DataFrame,
    split: str = "test",
    *,
    out_path: str | None = None,
    show: bool = True,
) -> None:
    col = f"tail_enrichment_delta_abs_r_{split}"
    if col not in gene_df.columns:
        raise KeyError(f"Missing {col}; load individual residuals and merge shrinkage first")
    s = gene_df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=50, color="#4c78a8", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axvline(float(s.median()), color="#f58518", ls="-", lw=1.2, label=f"median={s.median():.4f}")
    ax.set_xlabel(f"mean(|r_common| - |r_full|)_tail - mean(...)_nontail  ({split})")
    ax.set_ylabel("Genes")
    ax.set_title("Tail enrichment: residual shrinkage from rare variants")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_delta_metric_hist(
    gene_df: pd.DataFrame,
    col: str,
    *,
    title: str,
    xlabel: str,
    out_path: str | None = None,
    show: bool = True,
) -> None:
    if col not in gene_df.columns:
        raise KeyError(f"Missing column {col!r}")
    s = gene_df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=50, color="#54a24b", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axvline(float(s.median()), color="#f58518", ls="-", lw=1.2, label=f"median={s.median():.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Genes")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pooled_shrinkage_boxplot(
    ind_df: pd.DataFrame,
    split: str = "test",
    *,
    out_path: str | None = None,
    show: bool = True,
) -> None:
    sub = ind_df[ind_df["split"] == split]
    if sub.empty:
        return
    tail = sub[sub["is_tail_abs"]]["delta_abs_r"].astype(float)
    nontail = sub[~sub["is_tail_abs"]]["delta_abs_r"].astype(float)
    if len(tail) < 5 or len(nontail) < 5:
        return
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.boxplot(
        [nontail.values, tail.values],
        labels=["non-tail", "|r_common| tail"],
        showfliers=False,
    )
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel(r"$|r_{\mathrm{common}}| - |r_{\mathrm{full}}|$")
    ax.set_title(f"Pooled individuals ({split})")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gene_roc_pr(
    ind_df: pd.DataFrame,
    *,
    gene: str,
    split: str = "test",
    expr_extreme: str = "top",
    expr_quantile: float | None = None,
    title: str | None = None,
    out_path: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """ROC + precision-recall: common-only vs common+rare PRS for one expression tail."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    labels, scores_c, scores_f = _scores_for_expression_tail(
        ind_df, gene=gene, split=split, expr_extreme=expr_extreme
    )
    auroc_c, auprc_c = _classification_metrics(labels, scores_c)
    auroc_f, auprc_f = _classification_metrics(labels, scores_f)

    extreme_label = "top" if expr_extreme == "top" else "bottom"
    q_note = f" (extreme {100 * expr_quantile:.1f}%)" if expr_quantile is not None else ""
    if title is None:
        title = f"{gene}: {extreme_label} expression{q_note}, {split}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    if np.isfinite(auroc_c):
        fpr, tpr, _ = roc_curve(labels, scores_c)
        ax.plot(fpr, tpr, color="#4c78a8", lw=2, label=f"common-only (AUROC={auroc_c:.3f})")
    if np.isfinite(auroc_f):
        fpr, tpr, _ = roc_curve(labels, scores_f)
        ax.plot(fpr, tpr, color="#54a24b", lw=2, label=f"common+rare (AUROC={auroc_f:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    baseline = float(labels.sum()) / len(labels) if len(labels) else float("nan")
    if np.isfinite(auprc_c):
        prec, rec, _ = precision_recall_curve(labels, scores_c)
        ax.plot(rec, prec, color="#4c78a8", lw=2, label=f"common-only (AUPRC={auprc_c:.3f})")
    if np.isfinite(auprc_f):
        prec, rec, _ = precision_recall_curve(labels, scores_f)
        ax.plot(rec, prec, color="#54a24b", lw=2, label=f"common+rare (AUPRC={auprc_f:.3f})")
    if np.isfinite(baseline):
        ax.axhline(baseline, color="gray", ls="--", lw=0.8, label=f"baseline={baseline:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "gene": gene,
        "split": split,
        "expr_extreme": expr_extreme,
        "auroc_common": auroc_c,
        "auroc_full": auroc_f,
        "auprc_common": auprc_c,
        "auprc_full": auprc_f,
        "delta_auroc": auroc_f - auroc_c if np.isfinite(auroc_f) and np.isfinite(auroc_c) else float("nan"),
        "delta_auprc": auprc_f - auprc_c if np.isfinite(auprc_f) and np.isfinite(auprc_c) else float("nan"),
    }


def pick_gene_for_roc_pr(
    gene_df: pd.DataFrame,
    *,
    split: str = "test",
    expr_extreme: str = "top",
    prefer_positive_delta: bool = True,
) -> str | None:
    col = f"delta_auroc_{split}_{expr_extreme}"
    if col not in gene_df.columns:
        return None
    sub = gene_df[["gene", col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return None
    if prefer_positive_delta:
        pos = sub[sub[col] > 0]
        if len(pos):
            sub = pos
    return str(sub.sort_values(col, ascending=False).iloc[0]["gene"])


def plot_summary_figures(
    prep: dict[str, Any],
    *,
    expr_quantile: float | None = None,
    show: bool = True,
) -> None:
    """Standard genome-wide tail plots from ``prep_tails_plotting`` output."""
    split = prep["split"]
    gene_df = prep["gene_df"]
    ind_df = prep["ind_df"]
    q_pct = 100 * expr_quantile if expr_quantile is not None else None
    q_suffix = f" {q_pct:.1f}%" if q_pct is not None else ""

    if f"tail_enrichment_delta_abs_r_{split}" in gene_df.columns:
        plot_shrinkage_enrichment_hist(gene_df, split, show=show)

    for extreme in ("top", "bottom"):
        col = f"delta_auroc_{split}_{extreme}"
        if col in gene_df.columns:
            plot_delta_metric_hist(
                gene_df,
                col,
                title=f"ΔAUROC (full − common) for {extreme} expression ({split})",
                xlabel=f"ΔAUROC {extreme}{q_suffix} expressors",
                show=show,
            )
        col = f"delta_auprc_{split}_{extreme}"
        if col in gene_df.columns:
            plot_delta_metric_hist(
                gene_df,
                col,
                title=f"ΔAUPRC (full − common) for {extreme} expression ({split})",
                xlabel=f"ΔAUPRC {extreme}{q_suffix} expressors",
                show=show,
            )

    if not ind_df.empty:
        plot_pooled_shrinkage_boxplot(ind_df, split, show=show)


def run_gene_roc_pr(
    *,
    config_dir: str,
    tails_dir: str | None = None,
    gene: str | None = None,
    ind_df: pd.DataFrame | None = None,
    post_dir: str | None = None,
    common_beta: str = "common",
    rare_beta: str = "rare",
    resid_tail_quantile: float = 0.05,
    expr_quantile: float = 0.01,
    split: str = "test",
    expr_extreme: str = "top",
    out_path: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """
    Plot ROC/PR for one gene using saved ``tails/`` individuals when available,
    otherwise recompute via ``analyze_expression_tails.compute_tail_analysis_for_gene``
    (requires pyro + emmental deps in the active kernel).
    """
    config_dir = os.path.abspath(config_dir)
    post_dir = os.path.abspath(post_dir or os.path.join(config_dir, "post_pergene"))
    tails_dir = os.path.abspath(tails_dir or default_tails_dir(config_dir))

    if gene is None:
        summary_path = os.path.join(tails_dir, "gene_tail_summary.csv")
        if os.path.isfile(summary_path):
            gene = pick_gene_for_roc_pr(
                pd.read_csv(summary_path), split=split, expr_extreme=expr_extreme
            )
    if gene is None:
        raise ValueError("gene is required when no suitable gene found in summary")

    if ind_df is None:
        ind_path = os.path.join(tails_dir, "individual_gene_residuals.csv.gz")
        if os.path.isfile(ind_path):
            ind_df = load_individual_residuals(tails_dir)

    if ind_df is not None and not ind_df.empty:
        sub = ind_df[(ind_df["gene"] == gene) & (ind_df["split"] == split)]
        if not sub.empty:
            return plot_gene_roc_pr(
                ind_df,
                gene=gene,
                split=split,
                expr_extreme=expr_extreme,
                expr_quantile=expr_quantile,
                out_path=out_path,
                show=show,
            )

    from analyze_expression_tails import compute_tail_analysis_for_gene

    import load_data
    import torch
    import yaml
    from emmental_post_train_betas import (
        _ExpressionResidualizer,
        _load_covariates_scaled,
        _load_pergene_root_config,
    )
    from diagnose_posttrain_rare import _parse_gene_from_beta_path, discover_beta_panel_files

    pergene_root = os.path.join(config_dir, "pergene")
    try:
        base_config = _load_pergene_root_config(pergene_root)
    except FileNotFoundError:
        with open(os.path.join(config_dir, "joint", "config.yaml")) as f:
            base_config = yaml.safe_load(f) or {}

    beta_path = None
    for path in discover_beta_panel_files(post_dir, None):
        if _parse_gene_from_beta_path(path) == gene:
            beta_path = path
            break
    if beta_path is None:
        raise FileNotFoundError(f"No full_beta_panel for {gene!r} under {post_dir}")

    train_idx, test_idx = load_data.get_train_test_indices(base_config["covariates_path"])
    cov_scaled = _load_covariates_scaled(base_config)
    expr_cache = _ExpressionResidualizer(base_config, cov_scaled, torch.device("cpu"))
    cov = pd.read_csv(base_config["covariates_path"], sep="\t").set_index("sample_id")

    _, ind_df = compute_tail_analysis_for_gene(
        gene,
        pd.read_csv(beta_path, compression="infer"),
        base_config,
        train_idx,
        test_idx,
        cov.index,
        expr_cache,
        common_beta=common_beta,
        rare_beta=rare_beta,
        resid_tail_quantile=resid_tail_quantile,
        expr_tail_quantile=expr_quantile,
        splits=(split,),
    )
    return plot_gene_roc_pr(
        ind_df,
        gene=gene,
        split=split,
        expr_extreme=expr_extreme,
        expr_quantile=expr_quantile,
        out_path=out_path,
        show=show,
    )
