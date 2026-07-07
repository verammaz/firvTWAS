#!/usr/bin/env python3
"""
Plot |lin1| distribution and gate pass rate vs T for train_common01* full-model joint runs.

Uses learned tau1 (averaged over refits) and learned T from each experiment's tau_T.csv files.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# emmental/src on path
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import load_data
from joint_guide_setup import (
    collect_abs_lin1,
    resolve_threshold_init,
    threshold_prior_mean,
    threshold_prior_mode,
)


def _discover_full_experiments(root: str) -> List[Tuple[str, str]]:
    """Return (label, joint_dir) for non-collapsed train_common01* full-model runs."""
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return out

    for name in sorted(os.listdir(root)):
        if not name.startswith("train_common01"):
            continue
        if "collapse" in name:
            continue
        exp_root = os.path.join(root, name)

        # train_common01/{full,no_wg,...}/joint
        if name == "train_common01":
            for sub in sorted(os.listdir(exp_root)):
                joint = os.path.join(exp_root, sub, "joint")
                if os.path.isdir(joint) and _has_tau_t(joint):
                    out.append((f"{name}/{sub}", joint))
            continue

        joint = os.path.join(exp_root, "joint")
        if os.path.isdir(joint) and _has_tau_t(joint):
            out.append((name, joint))

    return out


def _has_tau_t(joint_dir: str) -> bool:
    for entry in os.listdir(joint_dir):
        if entry.startswith("run_") and os.path.isfile(os.path.join(joint_dir, entry, "tau_T.csv")):
            return True
    return os.path.isfile(os.path.join(joint_dir, "tau_T.csv"))


def _load_config(joint_dir: str, fallback_joint_dir: Optional[str] = None) -> dict:
    for path in (
        os.path.join(joint_dir, "config.yaml"),
        os.path.join(os.path.dirname(joint_dir), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(joint_dir)), "pergene", "config.yaml"),
    ):
        if os.path.isfile(path):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            cfg["joint_output_dir"] = joint_dir
            cfg.setdefault("collapsed_model", False)
            return cfg

    if fallback_joint_dir:
        cfg = _load_config(fallback_joint_dir)
        cfg["joint_output_dir"] = joint_dir
        return cfg

    raise FileNotFoundError(f"No config.yaml found for joint dir {joint_dir}")


def _config_data_key(cfg: dict) -> tuple:
    genes = cfg.get("genes") or cfg.get("gene_list")
    return (
        str(genes),
        cfg.get("maf_threshold"),
        tuple(cfg.get("annotations") or []),
        bool(cfg.get("normalize_G", False)),
        bool(cfg.get("train_test", True)),
    )


def _parse_gene_list(cfg: dict) -> None:
    if isinstance(cfg.get("gene_list"), str):
        path = cfg["gene_list"]
        if not os.path.isabs(path):
            path = os.path.join(_SRC, path)
        if os.path.isfile(path):
            with open(path) as f:
                cfg["genes"] = [line.strip() for line in f if line.strip()]
        else:
            cfg["genes"] = [g.strip() for g in cfg["gene_list"].split(",") if g.strip()]


def load_train_sample_ids(config) -> np.ndarray:
    """Train indices only (no expression load / residualization)."""
    covariates = pd.read_csv(config["covariates_path"], sep="\t").set_index("sample_id")
    train_mask = ~(
        (covariates["cohort"] == "ROSMAP")
        & (covariates["tissue"] == "Dorsolateral Pre-frontal Cortex (DLPFC)")
    )
    return np.where(train_mask)[0], covariates.index[train_mask]


def load_annotation_tensors_for_joint(cfg: dict, device: torch.device) -> load_data.AnnotationTensors:
    """
    Z-only tensors on the joint-training variant panel (MAF filter applied; no expression).
    """
    _parse_gene_list(cfg)
    _train_idx, train_sample_ids = load_train_sample_ids(cfg)
    G, Z, variant_ids_G, variant_ids_Z = load_data.load_genes(cfg)
    G, Z, variant_ids_G, variant_ids_Z, _maf_weights = load_data.prepare_genotypes_for_training(
        G, Z, variant_ids_G, variant_ids_Z, train_sample_ids, cfg, device
    )

    z_cols = list(G.columns)
    Z = Z.loc[z_cols]
    gene_indices = {}
    gene_names = []
    current_gene = None
    start_idx = 0
    for idx, col in enumerate(z_cols):
        gene = col.split("_")[0]
        if gene != current_gene:
            if current_gene is not None:
                gene_indices[current_gene] = (start_idx, idx)
                gene_names.append(current_gene)
            current_gene = gene
            start_idx = idx
    if current_gene is not None:
        gene_indices[current_gene] = (start_idx, len(z_cols))
        gene_names.append(current_gene)

    Z_t = torch.as_tensor(Z.values, dtype=torch.float32, device=device)
    return load_data.AnnotationTensors(
        Z=Z_t,
        gene_indices=gene_indices,
        gene_names=gene_names,
        device=device,
    )


def load_train_data(cfg: dict, device: torch.device):
    """Build annotation tensors matching joint training variant panel (fast path)."""
    return load_annotation_tensors_for_joint(cfg, device)


def plot_lin1_gate_diagnostic(
    abs_lin1: np.ndarray,
    config: dict,
    out_path: str,
    *,
    title: str,
    T_learned: float,
    n_variants: int,
    filter_frac: float,
    data=None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    lines = [
        (threshold_prior_mode(config), "Beta prior mode", "red"),
        (threshold_prior_mean(config), "Beta prior mean", "orange"),
        (T_learned, "Learned T (mean refits)", "#DD8452"),
    ]
    if data is not None:
        lines.insert(
            2,
            (resolve_threshold_init(data, config), "Guide init", "green"),
        )

    ax = axes[0]
    ax.hist(abs_lin1, bins=80, color="#4C72B0", alpha=0.75, density=True)
    for x, label, color in lines:
        ax.axvline(x, color=color, ls="--", lw=1.5, label=f"{label}: {x:.3f}")
    ax.set_xlabel("|lin1| = |Z·τ₁| (learned τ₁)")
    ax.set_ylabel("Density")
    ax.set_title("Gate input scale")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    hi = float(min(1.0, np.percentile(abs_lin1, 99.5)))
    lo = float(max(1e-4, abs_lin1.min() * 0.5))
    T_vals = np.linspace(lo, hi, 80)
    pass_rates = [(abs_lin1 >= t).mean() for t in T_vals]
    ax.plot(T_vals, pass_rates, color="#4C72B0", lw=2)
    for x, label, color in lines:
        ax.axvline(x, color=color, ls="--", lw=1, alpha=0.8)
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("Fraction variants passing gate")
    ax.set_title("Gate pass rate vs T")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{title}\n{n_variants:,} variants | filter @ learned T: {100 * filter_frac:.1f}%",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_experiment(
    label: str,
    joint_dir: str,
    data_cache: Dict[tuple, object],
    out_root: str,
    device: torch.device,
    fallback_joint_dir: Optional[str],
) -> dict:
    cfg = _load_config(joint_dir, fallback_joint_dir=fallback_joint_dir)
    key = _config_data_key(cfg)
    if key not in data_cache:
        data_cache[key] = load_train_data(cfg, device)
    data = data_cache[key]

    mean_df, T_learned, tau1, _tau2 = load_data.load_tau_threshold(cfg)
    tau1_t = torch.as_tensor(tau1, dtype=torch.float32, device=data.device)
    abs_lin1 = collect_abs_lin1(data, cfg, tau1=tau1_t)
    filter_frac = float((abs_lin1 < T_learned).mean())
    pass_frac = 1.0 - filter_frac

    safe = label.replace("/", "_")
    out_path = os.path.join(out_root, f"lin1_gate_diagnostic_{safe}.png")
    plot_kw = dict(
        title=label,
        T_learned=T_learned,
        n_variants=int(abs_lin1.size),
        filter_frac=filter_frac,
        data=data,
    )
    plot_lin1_gate_diagnostic(abs_lin1, cfg, out_path, **plot_kw)

    exp_plot = os.path.join(joint_dir, "plots", "lin1_gate_diagnostic.png")
    os.makedirs(os.path.dirname(exp_plot), exist_ok=True)
    plot_lin1_gate_diagnostic(abs_lin1, cfg, exp_plot, **plot_kw)

    return {
        "experiment": label,
        "joint_dir": joint_dir,
        "n_variants": int(abs_lin1.size),
        "T_learned": T_learned,
        "pass_frac": pass_frac,
        "filter_frac": filter_frac,
        "abs_lin1_median": float(np.median(abs_lin1)),
        "abs_lin1_p25": float(np.percentile(abs_lin1, 25)),
        "abs_lin1_p75": float(np.percentile(abs_lin1, 75)),
        "plot_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--myout_root",
        default="/gpfs/commons/home/vmazeeva/firvTWAS_myout",
        help="Root containing train_common01* directories",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for plots (default: myout_root/plots/lin1_gate_train_common01)",
    )
    parser.add_argument(
        "--fallback_joint",
        default="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01/full/joint",
        help="Config fallback when an experiment has no config.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run experiments whose label contains this substring (repeatable)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose output PNG already exists",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.myout_root, "plots", "lin1_gate_train_common01")
    os.makedirs(out_dir, exist_ok=True)

    experiments = _discover_full_experiments(args.myout_root)
    if args.only:
        needles = args.only
        experiments = [(lab, jd) for lab, jd in experiments if any(n in lab for n in needles)]
    if not experiments:
        raise SystemExit(f"No train_common01* full-model joint runs under {args.myout_root}")

    device = torch.device(args.device)
    data_cache: Dict[tuple, object] = {}
    rows = []

    print(f"Found {len(experiments)} experiment(s)")
    for label, joint_dir in experiments:
        safe = label.replace("/", "_")
        out_path = os.path.join(out_dir, f"lin1_gate_diagnostic_{safe}.png")
        if args.skip_existing and os.path.isfile(out_path):
            print(f"  {label} ... skip (exists)", flush=True)
            continue
        print(f"  {label} ...", flush=True)
        try:
            row = process_experiment(
                label,
                joint_dir,
                data_cache,
                out_dir,
                device,
                fallback_joint_dir=args.fallback_joint,
            )
            rows.append(row)
            print(
                f"    T={row['T_learned']:.4f} | filter={100*row['filter_frac']:.1f}% | "
                f"saved {row['plot_path']}"
            )
        except Exception as e:
            print(f"    FAILED: {e}")

    if rows:
        import pandas as pd

        summary_path = os.path.join(out_dir, "lin1_gate_summary.csv")
        new_df = pd.DataFrame(rows)
        if args.skip_existing and os.path.isfile(summary_path):
            old = pd.read_csv(summary_path)
            summary = (
                pd.concat([old, new_df], ignore_index=True)
                .drop_duplicates(subset=["experiment"], keep="last")
                .sort_values("experiment")
            )
        else:
            summary = new_df
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
