#!/usr/bin/env python3
"""
Grouped bar plots: proportion of genes with R² > threshold.

Compares uTWAS baselines (common-only) vs Emmental post-train
(post-joint or post-pergene; common-only and common+rare variant sets).

Writes per experiment config (one PNG per threshold):
  {plots_dir}/prop_r2_gt_{thr}_train_test.png
  {plots_dir}/prop_r2_threshold_summary.csv
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

BASELINE_ROOT_DEFAULT = (
    "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01"
)
TRAIN_COMMON01_DEFAULT = "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01_normG_fixed"
GENE_LIST_DEFAULT = (
    "/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt"
)

METHOD_LABELS = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "Elastic net",
    "bayesian_ridge": "Bayesian ridge",
    "emmental_common": "Emmental\n(common)",
    "emmental_all": "Emmental\n(common+rare)",
}

METHOD_ORDER = [
    "ridge",
    "lasso",
    "elasticnet",
    "bayesian_ridge",
    "emmental_common",
    "emmental_all",
]

DEFAULT_THRESHOLDS = (0.0, 0.01, 0.1, 0.3, 0.5)


def _thr_slug(thr: float) -> str:
    """Filename-safe threshold label, e.g. 0.01 -> 0p01, 0 -> 0."""
    if thr == 0:
        return "0"
    return f"{thr:g}".replace(".", "p")


def _thr_label(thr: float) -> str:
    return "0" if thr == 0 else f"{thr:g}"


def _ensg_from_gene_id(gene: str) -> str:
    g = str(gene).strip()
    return g.split("/")[-1] if "/" in g else g


def load_gene_list(path: str) -> Tuple[List[str], List[str]]:
    """
    Load top200_BRR_genes.txt in file order.

    Returns (chr/ENSG path ids, ENSG ids) — exactly one row per gene, 200 total.
    """
    paths: List[str] = []
    ensgs: List[str] = []
    seen_ensg: Set[str] = set()
    with open(path) as f:
        for line in f:
            g = line.strip()
            if not g or g.startswith("#"):
                continue
            ensg = _ensg_from_gene_id(g)
            if ensg in seen_ensg:
                raise ValueError(f"Duplicate ENSG in gene list: {ensg}")
            seen_ensg.add(ensg)
            paths.append(g)
            ensgs.append(ensg)
    if len(ensgs) == 0:
        raise ValueError(f"No genes in {path}")
    return paths, ensgs


def _subset_to_gene_list(series: pd.Series, gene_ensgs: List[str]) -> pd.Series:
    """Keep only genes from the list, in list order (NaN if score missing)."""
    return series.reindex(gene_ensgs)


def validate_gene_coverage(
    gene_ensgs: List[str],
    series_by_method: Dict[str, Tuple[pd.Series, pd.Series]],
    *,
    strict: bool,
) -> pd.DataFrame:
    """
    Report per-gene R² availability per method. If strict, require all 200 in every method.
    """
    rows = []
    for ensg in gene_ensgs:
        row: Dict[str, object] = {"ensg": ensg}
        for method, (tr, te) in series_by_method.items():
            row[f"{method}_train_r2"] = tr.get(ensg, np.nan)
            row[f"{method}_test_r2"] = te.get(ensg, np.nan)
        rows.append(row)
    cov = pd.DataFrame(rows)

    missing: Dict[str, List[str]] = {}
    for method, (tr, te) in series_by_method.items():
        miss = [
            ensg
            for ensg in gene_ensgs
            if not (np.isfinite(tr.get(ensg, np.nan)) and np.isfinite(te.get(ensg, np.nan)))
        ]
        if miss:
            missing[method] = miss

    n = len(gene_ensgs)
    print(f"Gene set: {n} genes")
    for method, (tr, te) in series_by_method.items():
        n_ok = sum(
            1
            for ensg in gene_ensgs
            if np.isfinite(tr.get(ensg, np.nan)) and np.isfinite(te.get(ensg, np.nan))
        )
        print(f"  {method}: {n_ok}/{n} with train+test R²")
    if missing:
        for method, miss in missing.items():
            print(f"  WARNING {method} missing {len(miss)} genes: {miss[:5]}{'...' if len(miss) > 5 else ''}")
        if strict:
            raise ValueError(
                "Not all methods have scores for every gene in the gene set. "
                f"Missing counts: {{{', '.join(f'{m}: {len(v)}' for m, v in missing.items())}}}"
            )
    return cov


def _read_emmental_r2_tables(emmental_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test R² tables from a flat post-train dir or ``chr*/`` subdirs."""
    tr_path = os.path.join(emmental_dir, "train_r2_scores.csv")
    te_path = os.path.join(emmental_dir, "test_r2_scores.csv")
    if os.path.isfile(tr_path) and os.path.isfile(te_path):
        return pd.read_csv(tr_path), pd.read_csv(te_path)

    chr_train = sorted(glob.glob(os.path.join(emmental_dir, "chr*", "train_r2_scores.csv")))
    chr_test = sorted(glob.glob(os.path.join(emmental_dir, "chr*", "test_r2_scores.csv")))
    if chr_train and chr_test:
        tr = pd.concat([pd.read_csv(p) for p in chr_train], ignore_index=True)
        te = pd.concat([pd.read_csv(p) for p in chr_test], ignore_index=True)
        return tr, te

    alt = os.path.join(emmental_dir, "r2_comparison", "per_gene_r2_by_variant_set.csv")
    if os.path.isfile(alt):
        both = pd.read_csv(alt)
        tr = both[["gene", "r2_train_common", "r2_train_all"]].rename(
            columns={"r2_train_common": "r2_common_only", "r2_train_all": "r2_common_plus_rare"}
        )
        te = both[["gene", "r2_test_common", "r2_test_all"]].rename(
            columns={"r2_test_common": "r2_common_only", "r2_test_all": "r2_common_plus_rare"}
        )
        return tr, te

    raise FileNotFoundError(
        f"Missing train/test_r2_scores.csv (flat or chr*/), or r2_comparison CSV under {emmental_dir}"
    )


def load_emmental_r2(emmental_dir: str) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """Post-joint or post-pergene: common and common+rare (train, test) Series by ENSG."""
    tr, te = _read_emmental_r2_tables(emmental_dir)
    tr_col_c, tr_col_a = "r2_common_only", "r2_common_plus_rare"
    te_col_c, te_col_a = tr_col_c, tr_col_a

    for df in (tr, te):
        if "gene" not in df.columns:
            raise ValueError(f"{emmental_dir}: missing 'gene' column")
        df["ensg"] = df["gene"].map(_ensg_from_gene_id)

    tr = tr.drop_duplicates(subset=["ensg"], keep="first").set_index("ensg")
    te = te.drop_duplicates(subset=["ensg"], keep="first").set_index("ensg")

    out: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    if tr_col_c in tr.columns:
        out["common"] = (tr[tr_col_c], te[te_col_c])
    if tr_col_a in tr.columns:
        out["all"] = (tr[tr_col_a], te[te_col_a])
    if not out:
        raise ValueError(f"{emmental_dir}: expected common / common+rare R² columns")
    return out


def load_emmental_postjoint_r2(post_joint_dir: str) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """Alias for backward compatibility."""
    return load_emmental_r2(post_joint_dir)


def intersection_gene_ensgs(
    series_by_method: Dict[str, Tuple[pd.Series, pd.Series]],
) -> List[str]:
    """ENSGs with finite train and test R² in every method."""
    genes: Optional[Set[str]] = None
    for tr, te in series_by_method.values():
        ok = set(tr.dropna().index) & set(te.dropna().index)
        ok = {g for g in ok if np.isfinite(tr[g]) and np.isfinite(te[g])}
        genes = ok if genes is None else genes & ok
    return sorted(genes or [])


def load_baseline_r2(baseline_root: str, method: str) -> Tuple[pd.Series, pd.Series]:
    """ENSG-indexed train/test R² (common variants only)."""
    train_parts, test_parts = [], []
    method_dir = os.path.join(baseline_root, method)
    if not os.path.isdir(method_dir):
        raise FileNotFoundError(method_dir)
    for fn in sorted(os.listdir(method_dir)):
        if not re.match(r"^chr\d+\.tsv$", fn):
            continue
        df = pd.read_csv(os.path.join(method_dir, fn), sep="\t", index_col=0)
        train_parts.append(df["R2_train"].rename("r2"))
        test_parts.append(df["R2_test"].rename("r2"))
    if not train_parts:
        raise FileNotFoundError(f"No chr*.tsv under {method_dir}")
    tr = pd.concat(train_parts).groupby(level=0).first()
    te = pd.concat(test_parts).groupby(level=0).first()
    return tr, te


def prop_above_fixed(
    series: pd.Series,
    thr: float,
    gene_ensgs: List[str],
) -> Tuple[float, int, int]:
    """Proportion over fixed gene list (same order/count for all methods)."""
    s = _subset_to_gene_list(series, gene_ensgs)
    n = len(gene_ensgs)
    n_present = int(s.notna().sum())
    passed = int((s.fillna(-np.inf) > thr).sum())
    return passed / n if n else float("nan"), n, n_present


def collect_metrics(
    baseline_root: str,
    emmental_dir: str,
    baseline_methods: List[str],
    gene_ensgs: List[str],
    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
    *,
    strict_gene_list: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    emm = load_emmental_r2(emmental_dir)
    series_train: Dict[str, pd.Series] = {
        "emmental_common": _subset_to_gene_list(emm["common"][0], gene_ensgs),
        "emmental_all": _subset_to_gene_list(emm["all"][0], gene_ensgs),
    }
    series_test: Dict[str, pd.Series] = {
        "emmental_common": _subset_to_gene_list(emm["common"][1], gene_ensgs),
        "emmental_all": _subset_to_gene_list(emm["all"][1], gene_ensgs),
    }

    for method in baseline_methods:
        tr, te = load_baseline_r2(baseline_root, method)
        series_train[method] = _subset_to_gene_list(tr, gene_ensgs)
        series_test[method] = _subset_to_gene_list(te, gene_ensgs)

    series_by_method = {m: (series_train[m], series_test[m]) for m in series_train}
    coverage = validate_gene_coverage(
        gene_ensgs, series_by_method, strict=strict_gene_list
    )

    rows = []
    for thr in thresholds:
        for method in METHOD_ORDER:
            if method not in series_train:
                continue
            for split, ser in [("train", series_train[method]), ("test", series_test[method])]:
                p, n, pres = prop_above_fixed(ser, thr, gene_ensgs)
                rows.append(
                    {
                        "method": method,
                        "threshold": thr,
                        "split": split,
                        "prop": p,
                        "n_genes": n,
                        "n_present": pres,
                    }
                )
    return pd.DataFrame(rows), coverage


def plot_grouped_bars(summary: pd.DataFrame, thr: float, out_path: str, title: str) -> None:
    sub = summary[summary["threshold"] == thr].copy()
    methods = [m for m in METHOD_ORDER if m in sub["method"].unique()]
    splits = ["train", "test"]
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(methods)), 5.5))
    colors = {"train": "#4C72B0", "test": "#DD8452"}

    for i, split in enumerate(splits):
        vals = []
        for m in methods:
            row = sub[(sub["method"] == m) & (sub["split"] == split)]
            vals.append(float(row["prop"].iloc[0]) if len(row) else float("nan"))
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
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=0, ha="center")
    ax.set_ylabel(f"Fraction of genes with R² > {_thr_label(thr)}")
    ax.set_ylim(0, min(1.05, max(0.15, ax.get_ylim()[1] + 0.08)))
    ax.legend(
        title="Split",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(right=0.82)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_experiment(
    emmental_dir: str,
    plots_dir: str,
    baseline_root: str,
    gene_ensgs: List[str],
    title_prefix: str,
    baseline_methods: Optional[List[str]] = None,
    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
    *,
    strict_gene_list: bool = True,
) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    n_genes = len(gene_ensgs)

    if baseline_methods is None:
        baseline_methods = [
            m
            for m in ("ridge", "lasso", "elasticnet", "bayesian_ridge")
            if os.path.isdir(os.path.join(baseline_root, m))
        ]

    summary, coverage = collect_metrics(
        baseline_root,
        emmental_dir,
        baseline_methods,
        gene_ensgs,
        thresholds=thresholds,
        strict_gene_list=strict_gene_list,
    )

    summary.to_csv(os.path.join(plots_dir, "prop_r2_threshold_summary.csv"), index=False)
    pd.DataFrame({"ensg": gene_ensgs}).to_csv(
        os.path.join(plots_dir, "gene_set_used.csv"), index=False
    )
    coverage.to_csv(os.path.join(plots_dir, "gene_r2_coverage_by_method.csv"), index=False)

    for thr in thresholds:
        slug = _thr_slug(thr)
        label = _thr_label(thr)
        plot_grouped_bars(
            summary,
            thr,
            os.path.join(plots_dir, f"prop_r2_gt_{slug}_train_test.png"),
            f"{title_prefix}: n={n_genes} genes (R² > {label})",
        )
    return plots_dir


def plot_one_config(
    config_dir: str,
    baseline_root: str,
    gene_list_path: str,
    emmental_source: str = "post_joint",
    gene_set_mode: str = "fixed_list",
    plots_subdir: Optional[str] = None,
    baseline_methods: Optional[List[str]] = None,
    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
    *,
    strict_gene_list: bool = True,
) -> Optional[str]:
    emmental_dir = resolve_emmental_dir(config_dir, emmental_source)
    if not emmental_dir:
        return None

    cfg_name = os.path.basename(config_dir.rstrip("/"))
    if gene_set_mode == "intersection":
        if baseline_methods is None:
            baseline_methods = [
                m
                for m in ("ridge", "lasso", "elasticnet", "bayesian_ridge")
                if os.path.isdir(os.path.join(baseline_root, m))
            ]
        gene_ensgs = build_gene_set_for_experiment(
            emmental_dir,
            baseline_root,
            baseline_methods,
            "intersection",
            None,
        )
        title = f"{cfg_name} ({emmental_source}, intersection)"
        sub = plots_subdir or f"{emmental_source}_intersection"
    else:
        _, gene_ensgs = load_gene_list(gene_list_path)
        title = f"{cfg_name} ({emmental_source}, top200)"
        sub = plots_subdir or f"{emmental_source}_top200"

    if not gene_ensgs:
        print(f"Skip {config_dir}: empty gene set")
        return None

    return plot_experiment(
        emmental_dir,
        os.path.join(config_dir, "plots", sub),
        baseline_root,
        gene_ensgs,
        title,
        baseline_methods=baseline_methods,
        thresholds=thresholds,
        strict_gene_list=strict_gene_list,
    )


def resolve_emmental_dir(config_dir: str, emmental_source: str) -> Optional[str]:
    """Sibling ``post_joint/`` or ``post_pergene/`` (legacy nested paths also checked)."""
    if emmental_source == "post_joint":
        candidates = (
            os.path.join(config_dir, "post_joint"),
            os.path.join(config_dir, "joint", "post_joint"),
        )
    elif emmental_source == "post_pergene":
        candidates = (
            os.path.join(config_dir, "post_pergene"),
            os.path.join(config_dir, "pergene", "post_pergene"),
        )
    else:
        raise ValueError(f"Unknown emmental_source: {emmental_source}")

    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        if os.path.isfile(os.path.join(cand, "train_r2_scores.csv")):
            return cand
        if glob.glob(os.path.join(cand, "chr*", "train_r2_scores.csv")):
            return cand
        alt = os.path.join(cand, "r2_comparison", "per_gene_r2_by_variant_set.csv")
        if os.path.isfile(alt):
            return cand
    return None


def resolve_postjoint_dir(experiment_root: str, postjoint_subdir: str) -> Optional[str]:
    for cand in (
        os.path.join(experiment_root, postjoint_subdir),
        os.path.join(experiment_root, "joint", postjoint_subdir),
        os.path.join(experiment_root, "post_joint"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def build_gene_set_for_experiment(
    emmental_dir: str,
    baseline_root: str,
    baseline_methods: List[str],
    gene_set_mode: str,
    gene_list_path: Optional[str],
) -> List[str]:
    emm = load_emmental_r2(emmental_dir)
    series_by_method: Dict[str, Tuple[pd.Series, pd.Series]] = {
        "emmental_common": emm["common"],
        "emmental_all": emm["all"],
    }
    for method in baseline_methods:
        series_by_method[method] = load_baseline_r2(baseline_root, method)

    if gene_set_mode == "intersection":
        genes = intersection_gene_ensgs(series_by_method)
        print(f"Intersection across {len(series_by_method)} methods: {len(genes)} genes")
        return genes
    if not gene_list_path:
        raise ValueError("gene_set=fixed_list requires --gene-list")
    _, gene_ensgs = load_gene_list(gene_list_path)
    return gene_ensgs


def discover_config_dirs(root: str, emmental_source: str) -> List[str]:
    config_dirs = []
    for name in sorted(os.listdir(root)):
        cfg = os.path.join(root, name)
        if not os.path.isdir(cfg):
            continue
        if resolve_emmental_dir(cfg, emmental_source):
            config_dirs.append(cfg)
    return config_dirs


def main() -> int:
    p = argparse.ArgumentParser(description="Emmental post-train vs baseline R² proportion plots")
    p.add_argument("--train_common01_root", default=TRAIN_COMMON01_DEFAULT)
    p.add_argument(
        "--experiment-root",
        default=None,
        help="Single experiment dir (e.g. train_common_01_only); use with --postjoint-subdir",
    )
    p.add_argument(
        "--postjoint-subdir",
        default="post_joint",
        help="Post-train output subdir under --experiment-root (legacy names supported)",
    )
    p.add_argument(
        "--emmental-source",
        choices=["post_joint", "post_pergene"],
        default="post_joint",
        help="Which Emmental post-train outputs to use",
    )
    p.add_argument(
        "--plots-dir",
        default=None,
        help="Plot output (default: {experiment-root}/plots/... or {config}/plots/...)",
    )
    p.add_argument(
        "--plots-subdir",
        default=None,
        help="Subfolder under plots/ (default: {emmental-source}_top200 or _intersection)",
    )
    p.add_argument("--baseline_root", default=BASELINE_ROOT_DEFAULT)
    p.add_argument("--gene_list", default=GENE_LIST_DEFAULT)
    p.add_argument(
        "--gene-set",
        choices=["fixed_list", "intersection"],
        default="fixed_list",
        help="fixed_list: use --gene-list; intersection: genes with R² in all methods",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Experiment folders under train_common01 (default: all with post-train outputs)",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.1],
        help="R² thresholds (default: 0.01 0.1)",
    )
    p.add_argument(
        "--allow-missing-genes",
        action="store_true",
        help="Do not error if a method lacks R² for some list genes (still uses full list as denominator)",
    )
    args = p.parse_args()
    thresholds = tuple(args.thresholds)
    strict_gene_list = not args.allow_missing_genes

    baseline_methods = [
        m
        for m in ("ridge", "lasso", "elasticnet", "bayesian_ridge")
        if os.path.isdir(os.path.join(args.baseline_root, m))
    ]

    if args.experiment_root:
        exp_root = os.path.abspath(args.experiment_root)
        emmental_dir = resolve_postjoint_dir(exp_root, args.postjoint_subdir)
        if not emmental_dir:
            emmental_dir = resolve_emmental_dir(exp_root, args.emmental_source)
        if not emmental_dir:
            print(f"No {args.emmental_source} dir under {exp_root}")
            return 1
        if args.plots_dir:
            plots_dir = args.plots_dir
        elif args.plots_subdir:
            plots_dir = os.path.join(exp_root, "plots", args.plots_subdir)
        else:
            suffix = "intersection" if args.gene_set == "intersection" else "top200"
            plots_dir = os.path.join(exp_root, "plots", f"{args.emmental_source}_{suffix}")
        gene_ensgs = build_gene_set_for_experiment(
            emmental_dir,
            args.baseline_root,
            baseline_methods,
            args.gene_set,
            args.gene_list if args.gene_set == "fixed_list" else None,
        )
        if not gene_ensgs:
            print("Empty gene set")
            return 1
        title = os.path.basename(exp_root.rstrip("/"))
        if args.gene_set == "intersection":
            title += f" ({args.emmental_source}, intersection)"
        else:
            title += f" ({args.emmental_source}, top200)"
        out = plot_experiment(
            emmental_dir,
            plots_dir,
            args.baseline_root,
            gene_ensgs,
            title,
            baseline_methods=baseline_methods,
            thresholds=thresholds,
            strict_gene_list=strict_gene_list,
        )
        print(f"Wrote plots under {out}")
        return 0

    root = os.path.abspath(args.train_common01_root)
    if args.configs:
        config_dirs = [os.path.join(root, c) for c in args.configs]
    else:
        config_dirs = discover_config_dirs(root, args.emmental_source)

    if not config_dirs:
        print(f"No configs with {args.emmental_source} under {root}")
        return 1

    for cfg_dir in config_dirs:
        out = plot_one_config(
            cfg_dir,
            args.baseline_root,
            args.gene_list,
            emmental_source=args.emmental_source,
            gene_set_mode=args.gene_set,
            plots_subdir=args.plots_subdir,
            baseline_methods=baseline_methods,
            thresholds=thresholds,
            strict_gene_list=strict_gene_list,
        )
        if out:
            print(f"Wrote plots under {out}")
        else:
            print(f"Skip {cfg_dir}: no {args.emmental_source}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
