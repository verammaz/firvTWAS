#!/usr/bin/env python3
"""
Plot per-variant λ = gate(|Z·τ₁| ≥ T) · (Z·τ₁) · exp(Z·τ₂) · MAF
for train_common01* joint runs (learned τ/T averaged over refits).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import load_data
from models import annotation_lambda


@dataclass
class VariantLambdaData:
    Z: torch.Tensor
    maf_weights: torch.Tensor
    gene_indices: dict
    gene_names: list
    device: torch.device
    num_anno: int = 0

    def __post_init__(self):
        self.num_anno = self.Z.shape[1]

    def get_gene_data(self, gene_name: str):
        start, end = self.gene_indices[gene_name]
        return (
            torch.zeros(0, device=self.device),
            self.Z[start:end],
            self.maf_weights[start:end],
        )


def list_runs(joint_dir: str) -> bool:
    for name in os.listdir(joint_dir):
        if re.match(r"run_\d+$", name) and os.path.isfile(
            os.path.join(joint_dir, name, "tau_T.csv")
        ):
            return True
    return os.path.isfile(os.path.join(joint_dir, "tau_T.csv"))


def discover_experiments(root: str, exclude: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    exclude = exclude or ["tau1norm"]
    out: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("train_common01"):
            continue
        if any(x in name for x in exclude):
            continue
        exp_root = os.path.join(root, name)
        if name == "train_common01":
            for sub in sorted(os.listdir(exp_root)):
                joint = os.path.join(exp_root, sub, "joint")
                if os.path.isdir(joint) and list_runs(joint):
                    out.append((f"train_common01/{sub}", joint))
        else:
            joint = os.path.join(exp_root, "joint")
            if os.path.isdir(joint) and list_runs(joint):
                out.append((name, joint))
    return out


def _parse_gene_list(cfg: dict) -> None:
    if isinstance(cfg.get("gene_list"), str):
        path = cfg["gene_list"]
        if not os.path.isabs(path):
            path = os.path.join(_SRC, path)
        if os.path.isfile(path):
            with open(path) as f:
                cfg["genes"] = [line.strip() for line in f if line.strip()]


def _load_config(joint_dir: str, fallback: Optional[str]) -> dict:
    for path in (
        os.path.join(joint_dir, "config.yaml"),
        os.path.join(os.path.dirname(joint_dir), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(joint_dir)), "pergene", "config.yaml"),
    ):
        if os.path.isfile(path):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            cfg["joint_output_dir"] = joint_dir
            return cfg
    if fallback:
        cfg = _load_config(fallback, None)
        cfg["joint_output_dir"] = joint_dir
        return cfg
    raise FileNotFoundError(f"No config for {joint_dir}")


def _config_data_key(cfg: dict) -> tuple:
    return (
        str(cfg.get("genes") or cfg.get("gene_list")),
        cfg.get("maf_threshold"),
        tuple(cfg.get("annotations") or []),
        bool(cfg.get("normalize_G", False)),
    )


def load_variant_data(cfg: dict, device: torch.device) -> VariantLambdaData:
    _parse_gene_list(cfg)
    cov = pd.read_csv(cfg["covariates_path"], sep="\t").set_index("sample_id")
    train_mask = ~(
        (cov["cohort"] == "ROSMAP")
        & (cov["tissue"] == "Dorsolateral Pre-frontal Cortex (DLPFC)")
    )
    train_ids = cov.index[train_mask]
    G, Z, vG, vZ = load_data.load_genes(cfg)
    G, Z, vG, vZ, maf_s = load_data.prepare_genotypes_for_training(
        G, Z, vG, vZ, train_ids, cfg, device
    )
    cols = list(G.columns)
    Z = Z.loc[cols]
    gene_indices, gene_names = {}, []
    cur, start = None, 0
    for i, col in enumerate(cols):
        g = col.split("_")[0]
        if g != cur:
            if cur is not None:
                gene_indices[cur] = (start, i)
                gene_names.append(cur)
            cur, start = g, i
    if cur is not None:
        gene_indices[cur] = (start, len(cols))
        gene_names.append(cur)
    Z_t = torch.as_tensor(Z.values, dtype=torch.float32, device=device)
    maf_t = torch.as_tensor(maf_s.loc[cols].values, dtype=torch.float32, device=device)
    return VariantLambdaData(Z=Z_t, maf_weights=maf_t, gene_indices=gene_indices, gene_names=gene_names, device=device)


def collect_lambdas(
    data: VariantLambdaData,
    tau1: torch.Tensor,
    tau2: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    th = torch.tensor(float(threshold), dtype=torch.float32, device=data.device)
    chunks = []
    for gene in data.gene_names:
        _, Z_g, maf_g = data.get_gene_data(gene)
        lam = annotation_lambda(Z_g, maf_g, th, tau1=tau1, tau2=tau2)
        chunks.append(lam.detach().cpu().numpy().ravel())
    return np.concatenate(chunks)


def plot_lambda_distribution(
    lam: np.ndarray,
    out_path: str,
    *,
    title: str,
    T_learned: float,
) -> dict:
    active = lam[lam != 0]
    frac_zero = float((lam == 0).mean())
    stats = {
        "n_variants": int(lam.size),
        "frac_lambda_zero": frac_zero,
        "frac_lambda_active": 1.0 - frac_zero,
        "abs_lambda_median_active": float(np.median(np.abs(active))) if active.size else np.nan,
        "abs_lambda_p95_active": float(np.percentile(np.abs(active), 95)) if active.size else np.nan,
        "signed_lambda_median_active": float(np.median(active)) if active.size else np.nan,
        "T_learned": float(T_learned),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    abs_all = np.abs(lam)
    pos = abs_all[abs_all > 0]
    if pos.size:
        hi = float(np.percentile(pos, 99.5))
        ax.hist(pos[pos <= hi], bins=80, color="#4C72B0", alpha=0.75, density=True)
    ax.set_xlabel("|λ| (active variants)")
    ax.set_ylabel("Density")
    ax.set_title(f"Active |λ| (n={active.size:,})")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    if active.size:
        hi = float(np.percentile(np.abs(active), 99.5))
        in_range = active[(active >= -hi) & (active <= hi)]
        ax.hist(in_range, bins=80, color="#DD8452", alpha=0.75, density=True)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("λ (signed, active)")
    ax.set_ylabel("Density")
    ax.set_title("Signed λ (gated-in)")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    if active.size:
        log_abs = np.log10(np.abs(active) + 1e-12)
        ax.hist(log_abs, bins=80, color="#55A868", alpha=0.75, density=True)
    ax.set_xlabel("log10(|λ|)")
    ax.set_ylabel("Density")
    ax.set_title("log10|λ| (active)")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{title}\nT={T_learned:.4f} | {100 * frac_zero:.1f}% gated out (λ=0)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats


def process_experiment(
    label: str,
    joint_dir: str,
    data_cache: Dict[tuple, VariantLambdaData],
    out_root: str,
    device: torch.device,
    fallback: Optional[str],
) -> dict:
    cfg = _load_config(joint_dir, fallback)
    key = _config_data_key(cfg)
    if key not in data_cache:
        data_cache[key] = load_variant_data(cfg, device)
    data = data_cache[key]
    _df, T, tau1, tau2 = load_data.load_tau_threshold(cfg)
    tau1_t = torch.as_tensor(tau1, dtype=torch.float32, device=device)
    tau2_t = torch.as_tensor(tau2, dtype=torch.float32, device=device)
    lam = collect_lambdas(data, tau1_t, tau2_t, T)

    safe = label.replace("/", "_")
    out_path = os.path.join(out_root, f"lambda_distribution_{safe}.png")
    stats = plot_lambda_distribution(lam, out_path, title=label, T_learned=T)
    stats["experiment"] = label
    stats["plot_path"] = out_path

    exp_plot = os.path.join(joint_dir, "plots", "lambda_distribution.png")
    os.makedirs(os.path.dirname(exp_plot), exist_ok=True)
    plot_lambda_distribution(lam, exp_plot, title=label, T_learned=T)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--myout_root", default="/gpfs/commons/home/vmazeeva/firvTWAS_myout")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Default: myout_root/plots/lambda_train_common01",
    )
    p.add_argument(
        "--fallback_joint",
        default="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01/full/joint",
    )
    p.add_argument("--only", action="append", default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.myout_root, "plots", "lambda_train_common01")
    os.makedirs(out_dir, exist_ok=True)
    exps = discover_experiments(args.myout_root)
    if args.only:
        exps = [(l, j) for l, j in exps if any(n in l for n in args.only)]

    device = torch.device(args.device)
    cache: Dict[tuple, VariantLambdaData] = {}
    rows = []
    print(f"Found {len(exps)} experiment(s)")
    for label, joint_dir in exps:
        safe = label.replace("/", "_")
        out_path = os.path.join(out_dir, f"lambda_distribution_{safe}.png")
        if args.skip_existing and os.path.isfile(out_path):
            print(f"  {label} ... skip")
            continue
        print(f"  {label} ...", flush=True)
        try:
            row = process_experiment(label, joint_dir, cache, out_dir, device, args.fallback_joint)
            rows.append(row)
            print(
                f"    active={100*row['frac_lambda_active']:.1f}% | "
                f"med|λ|={row['abs_lambda_median_active']:.4g} -> {out_path}"
            )
        except Exception as e:
            print(f"    FAILED: {e}")

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(out_dir, "lambda_distribution_summary.csv"), index=False)
        print(f"Summary: {os.path.join(out_dir, 'lambda_distribution_summary.csv')}")


if __name__ == "__main__":
    main()
