#!/usr/bin/env python3
"""
Grouped bar plots: fraction of genes with R² > threshold, comparing Emmental configs.

Uses per-gene training outputs:
  {train_common01_root}/{config}/pergene/chr*/train_r2_scores.csv
  {train_common01_root}/{config}/pergene/chr*/test_r2_scores.csv

Example (genome-wide, all genes with scores in every config):
  python plot_emmental_configs_pergene_r2.py \\
    --train_common01_root /gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01 \\
    --out_dir /gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01/plots \\
    --tag genome

Example (top200_BRR genes only):
  python plot_emmental_configs_pergene_r2.py \\
    --gene_list /gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt \\
    --tag top200
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_COMMON01_DEFAULT = "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01"
GENE_LIST_DEFAULT = (
    "/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt"
)

CONFIGS_DEFAULT = ("full", "no_rhog", "no_wg", "no_wg_rhog")
THRESHOLDS_DEFAULT = (0.01, 0.1)

CONFIG_LABELS = {
    "full": "full",
    "no_rhog": "no ρ_g",
    "no_wg": "no w_g",
    "no_wg_rhog": "no w_g, ρ_g",
}


def _thr_slug(thr: float) -> str:
    if thr == 0:
        return "0"
    return f"{thr:g}".replace(".", "p")


def _thr_label(thr: float) -> str:
    return "0" if thr == 0 else f"{thr:g}"


def _ensg(gene: str) -> str:
    g = str(gene).strip()
    return g.split("/")[-1] if "/" in g else g


def load_gene_list(path: str) -> List[str]:
    ensgs: List[str] = []
    seen: Set[str] = set()
    with open(path) as f:
        for line in f:
            g = line.strip()
            if not g or g.startswith("#"):
                continue
            e = _ensg(g)
            if e in seen:
                continue
            seen.add(e)
            ensgs.append(e)
    return ensgs


def load_config_r2(
    train_common01_root: str,
    config: str,
) -> Tuple[pd.Series, pd.Series]:
    """Load and concat chr*/{train,test}_r2_scores.csv → ENSG-indexed series."""
    base = os.path.join(train_common01_root, config, "pergene")
    train_parts: List[pd.Series] = []
    test_parts: List[pd.Series] = []
    for split, parts in (
        ("train", train_parts),
        ("test", test_parts),
    ):
        pattern = os.path.join(base, "chr*", f"{split}_r2_scores.csv")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No {split}_r2_scores.csv under {base}/chr*/")
        for path in paths:
            df = pd.read_csv(path)
            if "gene" not in df.columns or "r2" not in df.columns:
                raise ValueError(f"Bad schema in {path}")
            s = df.set_index(df["gene"].map(_ensg))["r2"].astype(float)
            parts.append(s)
    train = pd.concat(train_parts).groupby(level=0).first()
    test = pd.concat(test_parts).groupby(level=0).first()
    return train, test


def intersection_ensgs(
    r2_by_config: Dict[str, Tuple[pd.Series, pd.Series]],
) -> List[str]:
    genes: Optional[Set[str]] = None
    for tr, te in r2_by_config.values():
        ok = set(tr.dropna().index) & set(te.dropna().index)
        genes = ok if genes is None else genes & ok
    return sorted(genes or [])


def collect_proportions(
    r2_by_config: Dict[str, Tuple[pd.Series, pd.Series]],
    gene_ensgs: List[str],
    thresholds: Tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    n = len(gene_ensgs)
    for thr in thresholds:
        for config in r2_by_config:
            tr, te = r2_by_config[config]
            tr_sub = tr.reindex(gene_ensgs)
            te_sub = te.reindex(gene_ensgs)
            for split, ser in [("train", tr_sub), ("test", te_sub)]:
                passed = int((ser.fillna(-np.inf) > thr).sum())
                rows.append(
                    {
                        "config": config,
                        "threshold": thr,
                        "split": split,
                        "prop": passed / n if n else float("nan"),
                        "n_genes": n,
                        "n_present": int(ser.notna().sum()),
                        "n_passed": passed,
                    }
                )
    return pd.DataFrame(rows)


def plot_grouped_bars(
    summary: pd.DataFrame,
    thr: float,
    out_path: str,
    title: str,
) -> None:
    sub = summary[summary["threshold"] == thr].copy()
    configs = [c for c in CONFIGS_DEFAULT if c in sub["config"].unique()]
    if not configs:
        configs = sorted(sub["config"].unique())

    x = np.arange(len(configs))
    width = 0.36
    colors = {"train": "#4C72B0", "test": "#DD8452"}

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(configs)), 5.5))

    for i, split in enumerate(["train", "test"]):
        vals = []
        for cfg in configs:
            row = sub[(sub["config"] == cfg) & (sub["split"] == split)]
            vals.append(float(row["prop"].iloc[0]) if len(row) else float("nan"))
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=split.capitalize(),
            color=colors[split],
        )
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
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
    ax.set_ylabel(f"Fraction of genes with R² > {_thr_label(thr)}")
    ax.set_ylim(0, min(1.05, max(0.15, (sub["prop"].max() if len(sub) else 0) + 0.08)))
    ax.set_title(title)
    ax.legend(title="Split", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(right=0.82)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Emmental config R² comparison (pergene chr scores).")
    p.add_argument("--train_common01_root", default=TRAIN_COMMON01_DEFAULT)
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS_DEFAULT),
        help="Config subdirs under train_common01 (default: full no_rhog no_wg no_wg_rhog).",
    )
    p.add_argument(
        "--gene_list",
        default=None,
        help="Optional gene list (chr/ENSG or ENSG). Default: intersection of all configs.",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(THRESHOLDS_DEFAULT),
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: {root}/plots).",
    )
    p.add_argument(
        "--tag",
        default="genome",
        help="Filename tag, e.g. genome → emmental_configs_genome_r2_0p01.png",
    )
    args = p.parse_args()

    root = os.path.abspath(args.train_common01_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(root, "plots"))
    os.makedirs(out_dir, exist_ok=True)

    r2_by_config: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    for cfg in args.configs:
        r2_by_config[cfg] = load_config_r2(root, cfg)

    if args.gene_list:
        gene_ensgs = load_gene_list(os.path.abspath(args.gene_list))
        title_suffix = f"top200 list (n={len(gene_ensgs)})"
    else:
        gene_ensgs = intersection_ensgs(r2_by_config)
        title_suffix = f"all genes, intersection (n={len(gene_ensgs)})"

    summary = collect_proportions(r2_by_config, gene_ensgs, tuple(args.thresholds))
    summary_path = os.path.join(out_dir, f"emmental_configs_{args.tag}_prop_r2_summary.csv")
    summary.to_csv(summary_path, index=False)

    pd.DataFrame({"ensg": gene_ensgs}).to_csv(
        os.path.join(out_dir, f"emmental_configs_{args.tag}_gene_set.csv"),
        index=False,
    )

    for thr in args.thresholds:
        slug = _thr_slug(thr)
        out_png = os.path.join(out_dir, f"emmental_configs_{args.tag}_r2_{slug}.png")
        plot_grouped_bars(
            summary,
            thr,
            out_png,
            title=f"Emmental configs (per-gene): {title_suffix}, R² > {_thr_label(thr)}",
        )
        print("Wrote", out_png)

    print("Wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
