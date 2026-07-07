#!/usr/bin/env python3
"""
Grouped bar plots: fraction of genes with R² > threshold (common-only).

Compares uTWAS baselines (baseline_full_common01) vs Emmental per-gene fits.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_ROOT_DEFAULT = (
    "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01"
)
EMMENTAL_PERGENE_DEFAULT = (
    "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common_01_only/pergene"
)
GENE_LIST_DEFAULT = (
    "/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt"
)
BASELINE_FULL_DEFAULT = "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full"

METHOD_LABELS = {
    "bayesian_ridge": "Bayesian ridge",
    "elasticnet": "Elastic net",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "emmental": "Emmental",
}


def _chr_dirs(pergene_root: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(pergene_root)):
        if re.match(r"^chr\d+$", name) and os.path.isdir(os.path.join(pergene_root, name)):
            out.append(name)
    return out


def load_gene_set(gene_list_path: str | None, joint_config: str | None) -> set[str] | None:
    """ENSG ids to restrict comparison (e.g. 200-gene joint list)."""
    if gene_list_path and os.path.isfile(gene_list_path):
        genes = set()
        with open(gene_list_path) as f:
            for line in f:
                g = line.strip()
                if not g or g.startswith("#"):
                    continue
                genes.add(g.split("/")[-1] if "/" in g else g)
        return genes
    if joint_config and os.path.isfile(joint_config):
        import yaml

        with open(joint_config) as f:
            cfg = yaml.safe_load(f) or {}
        out = set()
        for g in cfg.get("genes") or []:
            out.add(g.split("/")[-1] if "/" in str(g) else str(g))
        return out if out else None
    return None


def load_emmental_r2(pergene_root: str) -> Tuple[pd.Series, pd.Series]:
    """gene (ENSG) -> r2 for train and test."""
    train_parts, test_parts = [], []
    for chr_name in _chr_dirs(pergene_root):
        chr_dir = os.path.join(pergene_root, chr_name)
        tr = os.path.join(chr_dir, "train_r2_scores.csv")
        te = os.path.join(chr_dir, "test_r2_scores.csv")
        if not (os.path.isfile(tr) and os.path.isfile(te)):
            continue
        train_parts.append(pd.read_csv(tr))
        test_parts.append(pd.read_csv(te))
    if not train_parts:
        raise FileNotFoundError(f"No Emmental R² CSVs under {pergene_root}")
    tr = pd.concat(train_parts, ignore_index=True).drop_duplicates(subset=["gene"], keep="first")
    te = pd.concat(test_parts, ignore_index=True).drop_duplicates(subset=["gene"], keep="first")
    return tr.set_index("gene")["r2"], te.set_index("gene")["r2"]


def load_baseline_r2(baseline_root: str, method: str) -> Tuple[pd.Series, pd.Series]:
    train_parts, test_parts = [], []
    method_dir = os.path.join(baseline_root, method)
    if not os.path.isdir(method_dir):
        raise FileNotFoundError(method_dir)
    for fn in sorted(os.listdir(method_dir)):
        m = re.match(r"^chr(\d+)\.tsv$", fn)
        if not m:
            continue
        df = pd.read_csv(os.path.join(method_dir, fn), sep="\t", index_col=0)
        if "R2_train" not in df.columns or "R2_test" not in df.columns:
            raise ValueError(f"{fn}: expected R2_train, R2_test columns")
        train_parts.append(df["R2_train"].rename("r2"))
        test_parts.append(df["R2_test"].rename("r2"))
    if not train_parts:
        raise FileNotFoundError(f"No chr*.tsv under {method_dir}")
    tr = pd.concat(train_parts).groupby(level=0).first()
    te = pd.concat(test_parts).groupby(level=0).first()
    return tr, te


def prop_above(
    series: pd.Series,
    thr: float,
    genes: pd.Index | None = None,
    *,
    fixed_denominator: bool = False,
) -> Tuple[float, int, int]:
    """
    Returns (proportion, n_denominator, n_present_with_finite_r2).
    If fixed_denominator=True, NaN R² counts as not passing (denominator = len(genes)).
    """
    if genes is not None:
        s = series.reindex(genes)
    else:
        s = series
    n_present = int(s.notna().sum())
    if fixed_denominator and genes is not None:
        if len(genes) == 0:
            return float("nan"), 0, 0
        passed = int((s.fillna(-np.inf) > thr).sum())
        return passed / len(genes), len(genes), n_present
    s = s.dropna()
    if len(s) == 0:
        return float("nan"), 0, n_present
    return float((s > thr).mean()), int(len(s)), n_present


def collect_metrics(
    baseline_root: str,
    pergene_root: str,
    methods: List[str],
    use_intersection: bool = True,
    gene_allow: set[str] | None = None,
    baseline_methods: List[str] | None = None,
    intersection_mode: str = "auto",
) -> pd.DataFrame:
    emm_tr, emm_te = load_emmental_r2(pergene_root)
    all_train = {"emmental": emm_tr}
    all_test = {"emmental": emm_te}

    bl_methods = baseline_methods if baseline_methods is not None else methods
    for method in bl_methods:
        if method == "emmental":
            continue
        all_train[method], all_test[method] = load_baseline_r2(baseline_root, method)

    if gene_allow is not None:
        ga = pd.Index(sorted(gene_allow))
        for m in list(all_train.keys()):
            all_train[m] = all_train[m].reindex(ga)
            all_test[m] = all_test[m].reindex(ga)

    # auto: fixed 200-gene list when --joint_config/--gene_list; else strict across methods
    if intersection_mode == "auto":
        mode = "gene_list_fixed" if gene_allow is not None else "strict"
    else:
        mode = intersection_mode

    if mode == "gene_list_fixed" and gene_allow is not None:
        genes = pd.Index(sorted(gene_allow))
        fixed_denom = True
    elif use_intersection or mode == "strict":
        genes = None
        for m in all_train:
            idx = all_train[m].dropna().index.intersection(all_test[m].dropna().index)
            genes = idx if genes is None else genes.intersection(idx)
        genes = genes.sort_values() if genes is not None else None
        fixed_denom = False
    else:
        genes = None
        fixed_denom = False

    rows = []
    for thr in (0.01, 0.1):
        for method in ["emmental"] + [m for m in methods if m != "emmental"]:
            if method not in all_train:
                continue
            p_tr, n_tr, pres_tr = prop_above(
                all_train[method], thr, genes, fixed_denominator=fixed_denom
            )
            p_te, n_te, pres_te = prop_above(
                all_test[method], thr, genes, fixed_denominator=fixed_denom
            )
            rows.append(
                {
                    "method": method,
                    "threshold": thr,
                    "split": "train",
                    "prop": p_tr,
                    "n_genes": n_tr,
                    "n_present": pres_tr,
                }
            )
            rows.append(
                {
                    "method": method,
                    "threshold": thr,
                    "split": "test",
                    "prop": p_te,
                    "n_genes": n_te,
                    "n_present": pres_te,
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_bars(summary: pd.DataFrame, thr: float, out_path: str, title: str) -> None:
    sub = summary[summary["threshold"] == thr].copy()
    method_order = [m for m in METHOD_LABELS if m in sub["method"].unique()]
    splits = ["train", "test"]
    x = np.arange(len(method_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(method_order)), 5))
    colors = {"train": "#4C72B0", "test": "#DD8452"}

    for i, split in enumerate(splits):
        vals = [
            float(sub[(sub["method"] == m) & (sub["split"] == split)]["prop"].iloc[0])
            for m in method_order
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=split.capitalize(), color=colors[split])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in method_order], rotation=20, ha="right")
    ax.set_ylabel(f"Fraction of genes with R² > {thr}")
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.08))
    ax.legend(title="Split")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline vs Emmental R² proportion bar plots")
    p.add_argument("--baseline_root", default=BASELINE_ROOT_DEFAULT)
    p.add_argument("--pergene_root", "--pergene_output_dir", dest="pergene_root", default=EMMENTAL_PERGENE_DEFAULT)
    p.add_argument("--out_dir", default=None)
    p.add_argument(
        "--gene_list",
        default=GENE_LIST_DEFAULT,
        help="Restrict to genes in this file (default: top200_BRR_genes.txt)",
    )
    p.add_argument("--joint_config", default=None, help="Use genes: from joint config.yaml")
    p.add_argument("--baseline_methods", nargs="+", default=None)
    p.add_argument("--plot_title", default=None)
    p.add_argument(
        "--all_genes_per_method",
        action="store_true",
        help="Per-method gene sets (no shared intersection)",
    )
    p.add_argument(
        "--intersection_mode",
        choices=["auto", "strict", "gene_list_fixed"],
        default="auto",
        help="auto: fixed gene-list denominator with --joint_config; else strict across methods",
    )
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.pergene_root.rstrip("/")), "plots_baseline_vs_emmental"
    )
    os.makedirs(out_dir, exist_ok=True)

    if args.baseline_methods:
        bl_methods = args.baseline_methods
    else:
        default_bl = ["ridge", "lasso", "elasticnet", "bayesian_ridge"]
        bl_methods = [
            m for m in default_bl if os.path.isdir(os.path.join(args.baseline_root, m))
        ]
        if not bl_methods:
            bl_methods = sorted(
                d
                for d in os.listdir(args.baseline_root)
                if os.path.isdir(os.path.join(args.baseline_root, d))
            )

    gene_allow = load_gene_set(args.gene_list, args.joint_config)

    summary = collect_metrics(
        args.baseline_root,
        args.pergene_root,
        bl_methods + ["emmental"],
        use_intersection=not args.all_genes_per_method,
        gene_allow=gene_allow,
        baseline_methods=bl_methods,
        intersection_mode="gene_list_fixed" if args.all_genes_per_method else args.intersection_mode,
    )
    summary_path = os.path.join(out_dir, "prop_r2_threshold_summary.csv")
    summary.to_csv(summary_path, index=False)

    n_genes = int(summary["n_genes"].dropna().iloc[0]) if len(summary) else 0
    if args.all_genes_per_method:
        gene_note = "per-method gene counts"
    elif gene_allow is not None and args.intersection_mode in ("auto", "gene_list_fixed"):
        gene_note = f"n={n_genes} genes (fixed list; missing R² counts as fail)"
    else:
        gene_note = f"strict intersection n={n_genes}"
    title_prefix = args.plot_title or "TWAS"

    plot_grouped_bars(
        summary,
        0.01,
        os.path.join(out_dir, "prop_r2_gt_0p01_train_test.png"),
        f"{title_prefix}: proportion of genes with R² > 0.01 ({gene_note})",
    )
    plot_grouped_bars(
        summary,
        0.1,
        os.path.join(out_dir, "prop_r2_gt_0p1_train_test.png"),
        f"{title_prefix}: proportion of genes with R² > 0.1 ({gene_note})",
    )

    print(f"Wrote {summary_path}")
    print(summary.pivot_table(index="method", columns=["threshold", "split"], values="prop"))
    print(f"Plots in {out_dir}")


if __name__ == "__main__":
    main()
