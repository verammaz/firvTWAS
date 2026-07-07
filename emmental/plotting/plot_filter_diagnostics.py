#!/usr/bin/env python3
"""
T-gate filter diagnostics after joint + per-gene training.

For each fitted gene (from joint or pergene ``beta_samples/``):
  - Full annotation panel (all variants on the gene): common + rare (MAF < threshold).
  - Training/common variants are marked via ``used_in_training`` (variant IDs from
    joint/pergene ``beta_samples``).
  - λ and T-gate computed on the **full panel** (same G/Z/MAF-weight setup as post-train).
  - τ/T from ``pergene/chr*/tau_T.csv`` (genome) or averaged joint τ/T (top200).
  - Writes ``filter/chr*/lambda_panel/{chr}_{ENSG}_lambda_panel.csv.gz``.

Assumes joint + pergene training completed; config from ``pergene/config.yaml`` when present.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "src"))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import load_data  # noqa: E402
import utils  # noqa: E402


def _load_gene_G_Z(cfg: dict):
    """Full annotation panel G/Z (same as post-train; avoids importing pyro via models)."""
    G_raw, Z, variant_ids_G, variant_ids_Z = load_data.load_genes(cfg)
    if cfg.get("normalize_G", False):
        G_model = utils.normalize_G(G_raw)
    else:
        G_model = G_raw
    return G_model, G_raw, Z, variant_ids_G, variant_ids_Z

TRAIN_COMMON01_DEFAULT = "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01"
GENE_LIST_DEFAULT = (
    "/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt"
)
CONFIGS_DEFAULT = ["full", "no_wg", "no_rhog", "no_wg_rhog"]
LAMBDA_ZERO_EPS = 1e-12


def annotation_lambda_np(
    Z: np.ndarray,
    maf_weights: np.ndarray,
    threshold: float,
    tau1: np.ndarray,
    tau2: np.ndarray,
) -> np.ndarray:
    """Match models.annotation_lambda (nonlinear τ, T gate, MAF weights)."""
    lin2 = Z @ tau2
    mod = np.exp(lin2)
    Z1 = np.concatenate([np.ones((Z.shape[0], 1), dtype=np.float32), Z], axis=1)
    lin1 = Z1 @ tau1
    gate = (np.abs(lin1) >= threshold).astype(np.float32)
    return gate * lin1 * mod * maf_weights


def load_gene_paths(path: str) -> List[str]:
    genes: List[str] = []
    seen = set()
    with open(path) as f:
        for line in f:
            g = line.strip()
            if not g or g.startswith("#"):
                continue
            ensg = g.split("/")[-1]
            if ensg in seen:
                raise ValueError(f"Duplicate ENSG in gene list: {ensg}")
            seen.add(ensg)
            genes.append(g)
    if not genes:
        raise ValueError(f"No genes in {path}")
    return genes


def train_test_idx(covariates_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Train/test row indices (same ROSMAP DLPFC holdout as load_data)."""
    cov = pd.read_csv(covariates_path, sep="\t").set_index("sample_id")
    train_mask = ~(
        (cov["cohort"] == "ROSMAP")
        & (cov["tissue"] == "Dorsolateral Pre-frontal Cortex (DLPFC)")
    )
    return np.where(train_mask)[0], np.where(~train_mask)[0]


def _maf_threshold(cfg: dict, default: float = 0.01) -> float:
    v = cfg.get("maf_threshold")
    if v is None:
        return default
    if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
        return default
    return float(v)


def load_experiment_config(config_root: str) -> dict:
    """Prefer ``pergene/config.yaml`` (matches per-gene training); fall back to joint."""
    config_root = os.path.abspath(config_root)
    pergene_cfg = os.path.join(config_root, "pergene", "config.yaml")
    joint_cfg = os.path.join(config_root, "joint", "config.yaml")
    path = pergene_cfg if os.path.isfile(pergene_cfg) else joint_cfg
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No pergene or joint config under {config_root}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["joint_output_dir"] = os.path.join(config_root, "joint")
    cfg["pergene_output_dir"] = os.path.join(config_root, "pergene")
    return cfg


def load_tau_threshold_pergene_chr(
    config_root: str, chr_tag: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    path = os.path.join(config_root, "pergene", chr_tag, "tau_T.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    mean_df = pd.read_csv(path)
    th = float(mean_df["Filter Threshold"].iloc[0])
    tau1, tau2 = load_data._tau_vectors_from_summary_df(mean_df)
    return tau1, tau2, th


def discover_pergene_gene_beta_paths(
    config_root: str,
) -> List[Tuple[str, str]]:
    """(chr/ENSG, beta_samples path) for every per-gene fit."""
    config_root = os.path.abspath(config_root)
    out: List[Tuple[str, str]] = []
    for chr_dir in sorted(glob.glob(os.path.join(config_root, "pergene", "chr*"))):
        chr_tag = os.path.basename(chr_dir)
        for path in sorted(glob.glob(os.path.join(chr_dir, "beta_samples", "ENSG*_beta.csv.gz"))):
            ensg = os.path.basename(path).replace("_beta.csv.gz", "")
            out.append((f"{chr_tag}/{ensg}", path))
    return out


def _resolve_config_genes(cfg: dict) -> List[str]:
    """chr/ENSG paths from config ``genes`` or ``gene_list`` file."""
    genes = cfg.get("genes")
    if isinstance(genes, list) and genes:
        return [str(g) for g in genes]
    gene_list = cfg.get("gene_list")
    if isinstance(gene_list, str):
        if os.path.isfile(gene_list):
            return load_gene_paths(gene_list)
        alt = os.path.join(_SRC_DIR, "genes", os.path.basename(gene_list))
        if os.path.isfile(alt):
            return load_gene_paths(alt)
        if "/" not in gene_list and os.path.isfile(os.path.join(_SRC_DIR, gene_list)):
            return load_gene_paths(os.path.join(_SRC_DIR, gene_list))
    return []


def discover_joint_gene_beta_paths(
    config_root: str,
    cfg: dict,
) -> List[Tuple[str, str]]:
    """(chr/ENSG, joint beta_samples path) using first available joint run."""
    joint_dir = os.path.abspath(cfg.get("joint_output_dir") or os.path.join(config_root, "joint"))
    ensg_to_gene = {g.split("/")[-1]: g for g in _resolve_config_genes(cfg)}
    if not ensg_to_gene:
        raise ValueError("joint: no genes in config (genes or gene_list)")

    beta_paths: List[str] = []
    for run_dir in sorted(glob.glob(os.path.join(joint_dir, "run_*"))):
        paths = sorted(glob.glob(os.path.join(run_dir, "beta_samples", "ENSG*_beta.csv.gz")))
        if paths:
            beta_paths = paths
            break
    if not beta_paths:
        raise FileNotFoundError(f"No joint beta_samples under {joint_dir}/run_*")

    out: List[Tuple[str, str]] = []
    for path in beta_paths:
        ensg = os.path.basename(path).replace("_beta.csv.gz", "")
        gene = ensg_to_gene.get(ensg)
        if gene is None:
            print(f"  WARN joint beta {ensg}: not in config gene list; skip")
            continue
        out.append((gene, path))
    return out


def discover_fit_gene_beta_paths(
    config_root: str,
    cfg: dict,
    *,
    source: str = "auto",
) -> List[Tuple[str, str]]:
    """
    Gene list + beta_samples paths from joint or pergene training outputs.

    source: ``pergene`` | ``joint`` | ``auto`` (pergene if any fits, else joint).
    """
    source = source.lower()
    pergene_paths = discover_pergene_gene_beta_paths(config_root)
    if source == "pergene":
        if not pergene_paths:
            raise FileNotFoundError(f"No pergene beta_samples under {config_root}")
        return pergene_paths
    if source == "joint":
        return discover_joint_gene_beta_paths(config_root, cfg)
    # auto
    if pergene_paths:
        return pergene_paths
    return discover_joint_gene_beta_paths(config_root, cfg)


def _training_variant_ids_from_beta_path(beta_path: str) -> Set[str]:
    df = pd.read_csv(beta_path, compression="infer", usecols=["variant_id_G"])
    return set(df["variant_id_G"].astype(str).tolist())


def _lambda_panel_out_path(filter_root: str, gene_name: str) -> str:
    chr_tag, ensg = gene_name.split("/", 1)
    safe = gene_name.replace("/", "_")
    return os.path.join(filter_root, chr_tag, "lambda_panel", f"{safe}_lambda_panel.csv.gz")


def counts_from_lambda_panel(panel: pd.DataFrame) -> Dict[str, int]:
    common = panel["is_common_maf_ge_threshold"].astype(bool).to_numpy()
    rare = ~common
    strict_fail = ~panel["passes_T_gate"].astype(bool).to_numpy()
    lambda_zero = panel["lambda_is_zero"].astype(bool).to_numpy()

    out: Dict[str, int] = {}
    for label, mask in [("all", np.ones(len(panel), bool)), ("common", common), ("rare", rare)]:
        n = int(mask.sum())
        out[f"n_{label}"] = n
        out[f"n_strict_filtered_{label}"] = int((strict_fail & mask).sum())
        out[f"n_lambda_zero_{label}"] = int((lambda_zero & mask).sum())
    return out


def compute_gene_lambda_panel(
    gene_name: str,
    cfg: dict,
    train_idx: np.ndarray,
    tau1: np.ndarray,
    tau2: np.ndarray,
    threshold: float,
    maf_thr: float,
    *,
    training_variant_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    λ and T-gate on the **full annotation panel** (common + rare).

    ``used_in_training`` marks variants present in joint/pergene ``beta_samples``
    (MAF-filtered training set). ``is_common_maf_ge_threshold`` uses train MAF.
    """
    gcfg = dict(cfg)
    gcfg["genes"] = [gene_name]
    gcfg["maf_threshold"] = None

    _G_model, G_raw, Z_ann, variant_ids_G, variant_ids_Z = _load_gene_G_Z(gcfg)
    if G_raw.shape[1] == 0:
        raise ValueError(f"{gene_name}: empty genotype matrix")
    if not G_raw.columns.equals(Z_ann.index):
        raise ValueError(f"{gene_name}: G/Z column mismatch")

    G_train_raw = G_raw.iloc[train_idx]
    maf = load_data.variant_maf_series(G_train_raw)
    maf_beta = int(cfg.get("maf_beta", 1))
    G_raw_t = torch.as_tensor(
        np.asarray(G_train_raw.values, dtype=np.float32), device="cpu"
    )
    maf_w = utils.get_MAF_weights(G_raw_t, torch.device("cpu"), maf_beta)

    Z_np = np.asarray(Z_ann.values, dtype=np.float32)
    maf_w_np = maf_w.detach().cpu().numpy().ravel()
    tau1_np = np.asarray(tau1, dtype=np.float32)
    tau2_np = np.asarray(tau2, dtype=np.float32)
    T = float(threshold)

    Z1 = np.concatenate([np.ones((Z_np.shape[0], 1), dtype=np.float32), Z_np], axis=1)
    lin1 = Z1 @ tau1_np
    abs_lin1 = np.abs(lin1)
    passes_T_gate = abs_lin1 >= T

    lam_np = annotation_lambda_np(Z_np, maf_w_np.astype(np.float32), T, tau1_np, tau2_np)
    lambda_is_zero = np.abs(lam_np) < LAMBDA_ZERO_EPS

    maf_train = maf.reindex(G_raw.columns).fillna(0.0).values.astype(np.float64)
    is_common = maf_train >= maf_thr
    cols = list(G_raw.columns)
    if training_variant_ids is None:
        used_in_training = is_common.copy()
    else:
        used_in_training = np.array([c in training_variant_ids for c in cols], dtype=bool)

    return pd.DataFrame(
        {
            "variant_id_G": cols,
            "variant_id_Z": variant_ids_Z,
            "maf_train": maf_train,
            "is_common_maf_ge_threshold": is_common,
            "used_in_training": used_in_training,
            "abs_lin1": abs_lin1.astype(np.float64),
            "filter_threshold_T": T,
            "passes_T_gate": passes_T_gate,
            "lambda_from_tau_Z_ann_maf": lam_np.astype(np.float64),
            "lambda_is_zero": lambda_is_zero,
        }
    )


def _summarize_per_gene_to_config_tables(
    per_gene: pd.DataFrame, config_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    count_rows = []
    pct_rows = []
    th = float(per_gene["threshold_T"].iloc[0]) if len(per_gene) else float("nan")
    for metric, prefix in [
        ("strict_T_gate", "n_strict_filtered"),
        ("lambda_zero", "n_lambda_zero"),
    ]:
        for stratum in ("all", "common", "rare"):
            n_col = f"n_{stratum}"
            f_col = f"{prefix}_{stratum}"
            n_total = int(per_gene[n_col].sum())
            n_filt = int(per_gene[f_col].sum())
            pct = 100.0 * n_filt / n_total if n_total else float("nan")
            count_rows.append(
                {
                    "config": config_name,
                    "metric": metric,
                    "stratum": stratum,
                    "n_variants": n_total,
                    "n_filtered": n_filt,
                    "threshold_T": th,
                }
            )
            pct_rows.append(
                {
                    "config": config_name,
                    "metric": metric,
                    "stratum": stratum,
                    "pct_filtered": pct,
                    "n_variants": n_total,
                    "n_filtered": n_filt,
                }
            )
    return pd.DataFrame(count_rows), pd.DataFrame(pct_rows)


def _load_tau_for_genome_gene(
    config_root: str,
    cfg: dict,
    chr_tag: str,
    fit_source: str,
    tau_cache: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """τ/T from per-chr pergene files, or joint average when using joint fits only."""
    if chr_tag in tau_cache:
        return tau_cache[chr_tag]

    pergene_path = os.path.join(config_root, "pergene", chr_tag, "tau_T.csv")
    if fit_source != "joint" and os.path.isfile(pergene_path):
        tau_cache[chr_tag] = load_tau_threshold_pergene_chr(config_root, chr_tag)
        return tau_cache[chr_tag]

    if "_joint_tau" not in tau_cache:
        _, th, tau1, tau2 = load_data.load_tau_threshold(cfg)
        tau_cache["_joint_tau"] = (tau1, tau2, th)
    return tau_cache["_joint_tau"]


def aggregate_config_genome(
    config_name: str,
    config_root: str,
    *,
    filter_root: Optional[str] = None,
    save_lambda_panels: bool = True,
    fit_source: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genome-wide fitted genes; τ/T from per-chr pergene or joint (see ``fit_source``)."""
    config_root = os.path.abspath(config_root)
    filter_root = os.path.abspath(filter_root or os.path.join(config_root, "filter"))
    cfg = load_experiment_config(config_root)
    maf_thr = _maf_threshold(cfg)

    gene_beta_paths = discover_fit_gene_beta_paths(config_root, cfg, source=fit_source)
    if not gene_beta_paths:
        raise FileNotFoundError(f"No fitted genes under {config_root}")

    train_idx, _ = train_test_idx(cfg["covariates_path"])

    tau_cache: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    rows: List[Dict[str, int]] = []
    n_panels_written = 0
    tau_source = (
        "joint/run_*/tau_T.csv (averaged)"
        if fit_source == "joint"
        else "pergene/chr*/tau_T.csv (joint fallback if missing)"
    )

    logging.disable(logging.CRITICAL)
    for gene, beta_path in tqdm(gene_beta_paths, desc=f"{config_name} genome", leave=False):
        chr_tag = gene.split("/")[0]
        try:
            tau1, tau2, th = _load_tau_for_genome_gene(
                config_root, cfg, chr_tag, fit_source, tau_cache
            )

            training_ids = _training_variant_ids_from_beta_path(beta_path)

            panel = compute_gene_lambda_panel(
                gene,
                cfg,
                train_idx,
                tau1,
                tau2,
                th,
                maf_thr,
                training_variant_ids=training_ids,
            )
            counts = counts_from_lambda_panel(panel)
            counts["gene"] = gene
            rows.append(counts)

            if save_lambda_panels:
                out_path = _lambda_panel_out_path(filter_root, gene)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                panel.to_csv(out_path, index=False, compression="gzip")
                n_panels_written += 1
        except Exception as e:
            print(f"  SKIP {gene}: {e}")

    per_gene = pd.DataFrame(rows)
    per_gene.insert(0, "config", config_name)
    per_gene["threshold_T"] = (
        float(next(iter(tau_cache.values()))[2]) if tau_cache else float("nan")
    )
    per_gene["maf_threshold"] = maf_thr
    counts_df, pct_df = _summarize_per_gene_to_config_tables(per_gene, config_name)

    if save_lambda_panels:
        manifest = {
            "config_root": config_root,
            "filter_root": filter_root,
            "n_genes": int(len(per_gene)),
            "n_lambda_panels_written": n_panels_written,
            "maf_threshold": maf_thr,
            "threshold_T": per_gene["threshold_T"].iloc[0] if len(per_gene) else None,
            "tau_source": tau_source,
            "fit_source": fit_source,
            "gene_list_from": "joint or pergene beta_samples",
            "panel_scope": "full_annotation_common_and_rare",
        }
        with open(os.path.join(filter_root, "lambda_panel_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    return per_gene, counts_df, pct_df


def aggregate_config_top200(
    config_name: str,
    cfg: dict,
    genes: List[str],
    train_idx: np.ndarray,
    *,
    config_root: str,
    filter_root: Optional[str] = None,
    save_lambda_panels: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Top-N genes: joint-averaged τ/T; training variants from joint beta_samples."""
    cfg = dict(cfg)
    joint_dir = cfg.get("joint_output_dir") or os.path.join(cfg.get("_config_root", ""), "joint")
    cfg["joint_output_dir"] = joint_dir
    _, th, tau1, tau2 = load_data.load_tau_threshold(cfg)
    maf_thr = _maf_threshold(cfg)

    joint_paths = {g: p for g, p in discover_joint_gene_beta_paths(config_root, cfg)}

    rows: List[Dict[str, int]] = []
    logging.disable(logging.CRITICAL)
    for gene in tqdm([g for g in genes if g in joint_paths], desc=config_name, leave=False):
        try:
            training_ids = _training_variant_ids_from_beta_path(joint_paths[gene])
            panel = compute_gene_lambda_panel(
                gene,
                cfg,
                train_idx,
                tau1,
                tau2,
                th,
                maf_thr,
                training_variant_ids=training_ids,
            )
            counts = counts_from_lambda_panel(panel)
            counts["gene"] = gene
            rows.append(counts)
            if save_lambda_panels and filter_root:
                out_path = _lambda_panel_out_path(filter_root, gene)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                panel.to_csv(out_path, index=False, compression="gzip")
        except Exception as e:
            print(f"  SKIP {gene}: {e}")

    per_gene = pd.DataFrame(rows)
    per_gene.insert(0, "config", config_name)
    per_gene["threshold_T"] = th
    per_gene["maf_threshold"] = maf_thr
    counts_df, pct_df = _summarize_per_gene_to_config_tables(per_gene, config_name)
    return per_gene, counts_df, pct_df


def _legend_outside(ax, **kwargs) -> None:
    opts = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    opts.update(kwargs)
    ax.legend(**opts)


def plot_pct_bars(
    pct_df: pd.DataFrame,
    metric: str,
    out_path: str,
    title: str,
) -> None:
    sub = pct_df[pct_df["metric"] == metric].copy()
    configs = list(sub["config"].unique())
    strata = ["all", "common", "rare"]
    labels = {"all": "all", "common": "common", "rare": "rare"}
    colors = {"all": "#4C72B0", "common": "#55A868", "rare": "#C44E52"}

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * max(len(configs), len(strata))), 5.5))

    if len(configs) == 1:
        cfg = configs[0]
        x = np.arange(len(strata))
        vals = []
        for st in strata:
            row = sub[(sub["config"] == cfg) & (sub["stratum"] == st)]
            vals.append(float(row["pct_filtered"].iloc[0]) if len(row) else float("nan"))
        bars = ax.bar(
            x,
            vals,
            0.6,
            color=[colors[st] for st in strata],
        )
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.5,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[st] for st in strata])
    else:
        x = np.arange(len(configs))
        width = 0.25
        for i, st in enumerate(strata):
            vals = []
            for cfg in configs:
                row = sub[(sub["config"] == cfg) & (sub["stratum"] == st)]
                vals.append(float(row["pct_filtered"].iloc[0]) if len(row) else float("nan"))
            bars = ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                label=labels[st],
                color=colors[st],
            )
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.5,
                        f"{v:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        _legend_outside(ax, title="stratum")

    ylab = (
        "% variants with |Z·τ₁| < T"
        if metric == "strict_T_gate"
        else "% variants with λ = 0"
    )
    ax.set_ylabel(ylab)
    ax.set_ylim(0, min(100, ax.get_ylim()[1] + 8))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if len(configs) > 1:
        fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_filter_outputs(
    per_gene_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    pct_df: pd.DataFrame,
    out_dir: str,
    suffix: str,
    scope: str,
    gene_list_path: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    per_gene_df.to_csv(
        os.path.join(out_dir, f"per_gene_filter_counts{suffix}.csv"), index=False
    )
    counts_df.to_csv(
        os.path.join(out_dir, f"filter_summary_counts{suffix}.csv"), index=False
    )
    pct_df.to_csv(os.path.join(out_dir, f"filter_summary_pct{suffix}.csv"), index=False)

    if scope == "top200":
        n_genes = len(load_gene_paths(gene_list_path))
        title_prefix = f"Top {n_genes} BRR genes"
    else:
        n_genes = int(per_gene_df.groupby("config").size().max())
        title_prefix = f"All fitted genes, full panel (n≈{n_genes:,})"

    plot_pct_bars(
        pct_df,
        "strict_T_gate",
        os.path.join(out_dir, f"pct_strict_T_gate_filtered{suffix}.png"),
        f"{title_prefix}: % variants failing T gate (|Z·τ₁| < T)",
    )
    plot_pct_bars(
        pct_df,
        "lambda_zero",
        os.path.join(out_dir, f"pct_lambda_zero{suffix}.png"),
        f"{title_prefix}: % variants with λ = 0",
    )


def _print_config_summary(cname: str, counts: pd.DataFrame, pct: pd.DataFrame) -> None:
    th = counts.iloc[0]["threshold_T"] if len(counts) else float("nan")
    parts = []
    for st in ("all", "common", "rare"):
        strict = pct[(pct.metric == "strict_T_gate") & (pct.stratum == st)]["pct_filtered"]
        lam = pct[(pct.metric == "lambda_zero") & (pct.stratum == st)]["pct_filtered"]
        if len(strict) and len(lam):
            parts.append(
                f"{st}: strict={float(strict.iloc[0]):.1f}% λ=0={float(lam.iloc[0]):.1f}%"
            )
    print(f"{cname}: T={th:.4g} | " + " | ".join(parts))


def _run_standalone_experiments(args: argparse.Namespace, suffix: str) -> int:
    if args.scope != "genome":
        print("Standalone --experiment_roots only supports --scope genome.")
        return 1

    rc = 0
    for exp_root in args.experiment_roots:
        config_root = os.path.abspath(exp_root)
        cname = os.path.basename(config_root.rstrip("/"))
        out_dir = args.out_dir or os.path.join(config_root, "filter")
        print(f"\n=== {cname} ({config_root}) ===")
        try:
            per_gene, counts, pct = aggregate_config_genome(
                cname,
                config_root,
                filter_root=out_dir,
                save_lambda_panels=not args.no_lambda_panels,
                fit_source=args.fit_source,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Skip {cname}: {e}")
            rc = 1
            continue
        _print_config_summary(cname, counts, pct)
        _write_filter_outputs(
            per_gene, counts, pct, out_dir, suffix, args.scope, args.gene_list
        )
        print(f"Wrote diagnostics + lambda panels under {out_dir}")
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="T-gate filter diagnostics (joint + pergene)")
    p.add_argument("--train_common01_root", default=TRAIN_COMMON01_DEFAULT)
    p.add_argument(
        "--scope",
        choices=("top200", "genome"),
        default="top200",
        help="top200: joint τ/T on gene list; genome: all pergene-fitted genes",
    )
    p.add_argument("--gene_list", default=GENE_LIST_DEFAULT)
    p.add_argument("--configs", nargs="+", default=CONFIGS_DEFAULT)
    p.add_argument("--out_dir", default=None, help="Summary output dir (default: {root}/filter)")
    p.add_argument(
        "--experiment_roots",
        nargs="+",
        default=None,
        help="Standalone experiment dirs (pergene + joint). Writes filter/ under each.",
    )
    p.add_argument(
        "--fit_source",
        choices=("auto", "pergene", "joint"),
        default="auto",
        help="Gene/variant list from pergene or joint beta_samples (auto=pergene if present)",
    )
    p.add_argument(
        "--no_lambda_panels",
        action="store_true",
        help="Skip writing per-gene lambda_panel CSVs (summary only)",
    )
    p.add_argument(
        "--save_lambda_panels_top200",
        action="store_true",
        help="Also write lambda panels for top200 scope",
    )
    args = p.parse_args()

    suffix = "" if args.scope == "top200" else "_genome"

    if args.experiment_roots:
        return _run_standalone_experiments(args, suffix)

    root = os.path.abspath(args.train_common01_root)
    out_dir = args.out_dir or os.path.join(root, "filter")
    os.makedirs(out_dir, exist_ok=True)

    all_per_gene = []
    all_counts = []
    all_pct = []

    if args.scope == "top200":
        genes = load_gene_paths(args.gene_list)
        cov_path = None
        for cname in args.configs:
            cp = os.path.join(root, cname, "joint", "config.yaml")
            if os.path.isfile(cp):
                with open(cp) as f:
                    cov_path = yaml.safe_load(f).get("covariates_path")
                break
        if not cov_path:
            raise FileNotFoundError("No joint/config.yaml found for train index")
        train_idx, _ = train_test_idx(cov_path)
        print(f"Gene list: {len(genes)} genes; train samples for MAF: {len(train_idx)}")

        for cname in args.configs:
            cfg_path = os.path.join(root, cname, "joint", "config.yaml")
            if not os.path.isfile(cfg_path):
                print(f"Skip {cname}: no {cfg_path}")
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            cfg["_config_root"] = os.path.join(root, cname)
            cfg["joint_output_dir"] = os.path.join(root, cname, "joint")

            per_gene, counts, pct = aggregate_config_top200(
                cname,
                cfg,
                genes,
                train_idx,
                config_root=os.path.join(root, cname),
                filter_root=out_dir,
                save_lambda_panels=args.save_lambda_panels_top200,
            )
            all_per_gene.append(per_gene)
            all_counts.append(counts)
            all_pct.append(pct)
            _print_config_summary(cname, counts, pct)
    else:
        print("Genome scope: pergene fitted genes + per-chr tau/T")
        for cname in args.configs:
            config_root = os.path.join(root, cname)
            if not os.path.isdir(os.path.join(config_root, "pergene")):
                print(f"Skip {cname}: no pergene/")
                continue
            try:
                per_gene, counts, pct = aggregate_config_genome(
                    cname,
                    config_root,
                    filter_root=os.path.join(config_root, "filter"),
                    save_lambda_panels=not args.no_lambda_panels,
                    fit_source=args.fit_source,
                )
            except FileNotFoundError as e:
                print(f"Skip {cname}: {e}")
                continue
            all_per_gene.append(per_gene)
            all_counts.append(counts)
            all_pct.append(pct)
            _print_config_summary(cname, counts, pct)

    if not all_pct:
        print("No configurations processed.")
        return 1

    per_gene_df = pd.concat(all_per_gene, ignore_index=True)
    counts_df = pd.concat(all_counts, ignore_index=True)
    pct_df = pd.concat(all_pct, ignore_index=True)

    _write_filter_outputs(
        per_gene_df, counts_df, pct_df, out_dir, suffix, args.scope, args.gene_list
    )
    print(f"Wrote diagnostics under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
