#!/usr/bin/env python3
"""
Parameter recovery for Emmental: simulate ground truth, refit, compare posteriors.

Typical use (small gene list in config, no BRR warm start):

    python param_recovery.py --config recovery_config.yaml --output_dir results/recovery_run0

Runs joint refit twice when requested: full model (Z sites) vs collapsed (beta integrated out).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyro
import torch
from scipy.stats import pearsonr, spearmanr

import load_data
import utils
from emmental_joint import calculate_r2, fit_emmental
from models import simulate_expression
from param_recovery_plots import plot_recovery_dashboard, plot_single_loss_curve, save_posterior_estimates


def _str_to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_recovery_args():
    parser = argparse.ArgumentParser(description="Emmental parameter recovery (simulate + refit)")
    parser.add_argument("--config", type=str, required=True, help="YAML config (same as joint training)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for recovery outputs")
    parser.add_argument("--simulation_seed", type=int, default=0, help="RNG seed for simulate_expression")
    parser.add_argument(
        "--run_full",
        type=_str_to_bool,
        default=True,
        help="Run non-collapsed joint refit (default: true)",
    )
    parser.add_argument(
        "--run_collapsed",
        type=_str_to_bool,
        default=True,
        help="Run collapsed joint refit (default: true)",
    )
    parser.add_argument("--simulation_obs_std", type=float, default=None, help="Override observation noise std")
    parser.add_argument(
        "--no_plots",
        type=_str_to_bool,
        default=False,
        help="Skip generating recovery figures (default: false)",
    )
    parser.add_argument(
        "--plots_only",
        type=_str_to_bool,
        default=False,
        help="Regenerate plots from an existing output_dir (requires prior run)",
    )
    return parser.parse_args()


def _parse_gene_list(config: dict) -> None:
    if isinstance(config.get("gene_list"), str):
        if os.path.exists(config["gene_list"]):
            with open(config["gene_list"], "r") as f:
                config["genes"] = [line.strip() for line in f if line.strip()]
        else:
            config["genes"] = [g.strip() for g in config["gene_list"].split(",") if g.strip()]
    else:
        config["genes"] = config.get("gene_list", []) or config.get("genes", [])


def load_joint_data(config: dict, device: torch.device):
    """Load G, Z, covariates; build train DataTensors (no BRR during recovery)."""
    X, Y, train_idx, test_idx = load_data.load_residualized_covariates(config, device)
    G, Z, variant_ids_G, variant_ids_Z = load_data.load_genes(config)

    if config.get("train_test", False):
        train_sample_ids = X.iloc[train_idx].index
        G, Z, variant_ids_G, variant_ids_Z, maf_weights = load_data.prepare_genotypes_for_training(
            G, Z, variant_ids_G, variant_ids_Z, train_sample_ids, config, device
        )
        G_train = G.loc[train_sample_ids]
        Y_train = {gene: expr[train_idx] for gene, expr in Y.items()}
        data_train = load_data.DataTensors.from_pandas(
            G_train,
            Z,
            X.iloc[train_idx],
            Y_train,
            brr_betas=None,
            brr_alphas=None,
            device=device,
            config=config,
            maf_weights_precomputed=maf_weights,
            variants_pre_filtered=True,
        )
    else:
        G, Z, variant_ids_G, variant_ids_Z, maf_weights = load_data.prepare_genotypes_for_training(
            G, Z, variant_ids_G, variant_ids_Z, G.index, config, device
        )
        data_train = load_data.DataTensors.from_pandas(
            G,
            Z,
            X,
            Y,
            brr_betas=None,
            brr_alphas=None,
            device=device,
            config=config,
            maf_weights_precomputed=maf_weights,
            variants_pre_filtered=True,
        )
    return data_train


def inject_simulated_y(data, simulated_parameters: dict):
    """Point data.Y at simulated expression (for R2 and logging)."""
    Y_new = {}
    for gene_name in data.gene_names:
        gene_key = gene_name.split("/")[-1] if "/" in gene_name else gene_name
        Y_new[gene_key] = simulated_parameters[gene_name]["y"]
    data.Y = Y_new
    return data


def _recovery_fit_config(config: dict, collapsed: bool) -> dict:
    fit_config = dict(config)
    fit_config["collapsed_model"] = collapsed
    fit_config["refits"] = 1
    fit_config["brr_results_dir"] = None
    return fit_config


def _tensor_recovery_stats(truth: np.ndarray, est_mean: np.ndarray, est_std: Optional[np.ndarray] = None):
    t = np.asarray(truth).ravel()
    e = np.asarray(est_mean).ravel()
    out = {}
    if len(t) > 1 and np.std(t) > 0 and np.std(e) > 0:
        out["pearson_r"], _ = pearsonr(t, e)
        out["spearman_r"], _ = spearmanr(t, e)
    else:
        out["pearson_r"] = np.nan
        out["spearman_r"] = np.nan
    out["rmse"] = float(np.sqrt(np.mean((t - e) ** 2)))
    if est_std is not None:
        s = np.asarray(est_std).ravel()
        out["coverage_95"] = float(np.mean(np.abs(t - e) <= 1.96 * (s + 1e-12)))
    return out


def evaluate_recovery(
    truth: dict,
    posterior_stats: dict,
    beta_samples: dict,
    data,
    label: str,
) -> pd.DataFrame:
    rows = []

    if "threshold" in posterior_stats:
        t = float(truth["threshold"].detach().cpu())
        e = float(np.asarray(posterior_stats["threshold"]["mean"]).reshape(()))
        rows.append({"model": label, "parameter": "threshold", "metric": "abs_error", "value": abs(t - e)})

    for param in ("tau1", "tau2", "w_g", "rho_g"):
        if param not in posterior_stats or param not in truth:
            continue
        t = truth[param].detach().cpu().numpy()
        ps = posterior_stats[param]
        stats = _tensor_recovery_stats(t, ps["mean"], ps.get("std"))
        for metric, value in stats.items():
            rows.append({"model": label, "parameter": param, "metric": metric, "value": value})

    beta_truth, beta_est = [], []
    for gene_name in data.gene_names:
        if gene_name not in beta_samples or gene_name not in truth:
            continue
        beta_truth.append(truth[gene_name]["beta"].detach().cpu().numpy().ravel())
        beta_est.append(beta_samples[gene_name].mean(axis=0).ravel())
    if beta_truth:
        stats = _tensor_recovery_stats(np.concatenate(beta_truth), np.concatenate(beta_est))
        for metric, value in stats.items():
            if metric != "coverage_95":
                rows.append({"model": label, "parameter": "beta_pooled", "metric": metric, "value": value})

    return pd.DataFrame(rows)


def _json_scalar(val):
    if isinstance(val, str):
        return val
    if isinstance(val, (float, int, np.floating, np.integer)):
        return float(val)
    if hasattr(val, "detach"):
        return float(val.detach().cpu())
    return val


def save_truth_artifacts(truth: dict, data, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    scalars = {}
    for key in ("threshold", "obs_std", "mode"):
        if key in truth:
            scalars[key] = _json_scalar(truth[key])
    for key in ("tau1", "tau2", "w_g", "rho_g"):
        if key in truth:
            scalars[key] = truth[key].detach().cpu().numpy().tolist()
    with open(os.path.join(output_dir, "simulation_truth.json"), "w") as f:
        json.dump(scalars, f, indent=2)

    beta_rows = []
    for gene_name in data.gene_names:
        gp = truth[gene_name]
        for j, b in enumerate(gp["beta"].detach().cpu().numpy().ravel()):
            beta_rows.append({"gene": gene_name, "variant_idx": j, "beta": float(b)})
    pd.DataFrame(beta_rows).to_csv(os.path.join(output_dir, "simulation_beta_truth.csv"), index=False)


def _load_truth_from_disk(output_dir: str) -> dict:
    path = os.path.join(output_dir, "simulation_truth.json")
    with open(path, "r") as f:
        raw = json.load(f)
    truth = {}
    for key in ("threshold", "obs_std"):
        if key in raw:
            truth[key] = torch.tensor(raw[key], dtype=torch.float32)
    for key in ("tau1", "tau2", "w_g", "rho_g"):
        if key in raw:
            truth[key] = torch.tensor(raw[key], dtype=torch.float32)
    beta_df = pd.read_csv(os.path.join(output_dir, "simulation_beta_truth.csv"))
    for gene_name, sub in beta_df.groupby("gene"):
        truth[gene_name] = {"beta": torch.tensor(sub["beta"].values, dtype=torch.float32)}
    return truth


def _load_fit_results_from_disk(output_dir: str, model_labels=("full", "collapsed")) -> dict:
    fit_results = {}
    for label in model_labels:
        run_dir = os.path.join(output_dir, label)
        est_path = os.path.join(run_dir, "posterior_estimates.json")
        loss_path = os.path.join(run_dir, "losses.csv")
        if not os.path.isfile(est_path):
            continue
        with open(est_path, "r") as f:
            est = json.load(f)
        posterior_stats = {
            k: {"mean": np.asarray(v["mean"]), "std": np.asarray(v.get("std", []))}
            for k, v in est.items()
            if k != "beta_by_gene"
        }
        beta_samples = {}
        for gene_name, bv in est.get("beta_by_gene", {}).items():
            beta_samples[gene_name] = np.asarray([bv["mean"]])
        losses = []
        if os.path.isfile(loss_path):
            losses = pd.read_csv(loss_path)["loss"].tolist()
        train_r2 = {}
        r2_path = os.path.join(run_dir, "train_r2.csv")
        if os.path.isfile(r2_path):
            r2_df = pd.read_csv(r2_path)
            train_r2 = dict(zip(r2_df["gene"], r2_df["r2"]))
        fit_results[label] = {
            "posterior_stats": posterior_stats,
            "beta_samples": beta_samples,
            "losses": losses,
            "train_r2": train_r2,
        }
    return fit_results


class _GeneNamesStub:
    """Minimal object for re-plotting when full DataTensors are unavailable."""

    def __init__(self, gene_names: list):
        self.gene_names = gene_names


def run_one_recovery(
    data,
    config: dict,
    simulated_parameters: dict,
    collapsed: bool,
    label: str,
    output_dir: str,
) -> Tuple[pd.DataFrame, dict, dict, list, dict]:
    logger = utils.get_logger()
    fit_config = _recovery_fit_config(config, collapsed=collapsed)
    logger.info("Recovery refit: %s (collapsed=%s)", label, collapsed)
    pyro.clear_param_store()
    result = fit_emmental(data, fit_config, simulated_parameters=simulated_parameters)
    losses = result[0]
    posterior_stats = result[2]
    beta_samples = result[3]

    metrics = evaluate_recovery(simulated_parameters, posterior_stats, beta_samples, data, label)
    r2 = calculate_r2(data, beta_samples, data.gene_names)
    mean_r2 = float(np.mean(list(r2.values()))) if r2 else float("nan")
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [{"model": label, "parameter": "train_r2", "metric": "mean", "value": mean_r2}]
            ),
        ],
        ignore_index=True,
    )

    run_dir = os.path.join(output_dir, label.replace(" ", "_"))
    os.makedirs(run_dir, exist_ok=True)
    metrics.to_csv(os.path.join(run_dir, "recovery_metrics.csv"), index=False)
    pd.DataFrame({"epoch": range(len(losses)), "loss": losses}).to_csv(
        os.path.join(run_dir, "losses.csv"), index=False
    )
    if r2:
        pd.DataFrame([{"gene": g, "r2": v} for g, v in r2.items()]).to_csv(
            os.path.join(run_dir, "train_r2.csv"), index=False
        )
    save_posterior_estimates(posterior_stats, beta_samples, data, run_dir)
    plot_single_loss_curve(
        losses,
        label,
        os.path.join(run_dir, "loss_curve.png"),
    )
    return metrics, posterior_stats, beta_samples, losses, r2


def main():
    cli = parse_recovery_args()
    yaml_config = utils.load_yaml(cli.config)
    config = utils.fill_defaults(cli, yaml_config, keep_simulation_config=True)
    _parse_gene_list(config)

    if cli.simulation_obs_std is not None:
        config["simulation_obs_std"] = cli.simulation_obs_std

    os.makedirs(cli.output_dir, exist_ok=True)
    log_file = os.path.join(cli.output_dir, "recovery.log")
    logger = utils.setup_logging(config.get("log_level", "INFO"), log_file)

    if cli.plots_only:
        logger.info("Plots-only mode: loading saved recovery outputs from %s", cli.output_dir)
        truth = _load_truth_from_disk(cli.output_dir)
        fit_results = _load_fit_results_from_disk(cli.output_dir)
        summary_path = os.path.join(cli.output_dir, "recovery_summary.csv")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing {summary_path}; run recovery first.")
        summary = pd.read_csv(summary_path)
        beta_df = pd.read_csv(os.path.join(cli.output_dir, "simulation_beta_truth.csv"))
        data_stub = _GeneNamesStub(beta_df["gene"].unique().tolist())
        plots_dir = os.path.join(cli.output_dir, "plots")
        saved = plot_recovery_dashboard(
            truth, fit_results, data_stub, config, summary, plots_dir
        )
        logger.info("Wrote %s figures to %s", len(saved), plots_dir)
        for p in saved:
            logger.info("  %s", p)
        return

    logger.info("=" * 50)
    logger.info("EMMENTAL PARAMETER RECOVERY")
    logger.info("=" * 50)
    for key, value in config.items():
        logger.info("  %s: %s", key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_train = load_joint_data(config, device)
    logger.info(
        "Loaded %s genes, %s individuals, %s variants",
        data_train.num_genes,
        data_train.G.shape[0],
        data_train.G.shape[1],
    )

    pyro.clear_param_store()
    truth = simulate_expression(data_train, config, seed=cli.simulation_seed)
    inject_simulated_y(data_train, truth)
    save_truth_artifacts(truth, data_train, cli.output_dir)

    fit_results = {}
    all_metrics = []
    if cli.run_full:
        m, ps, bs, losses, r2 = run_one_recovery(
            data_train, config, truth, collapsed=False, label="full", output_dir=cli.output_dir
        )
        all_metrics.append(m)
        fit_results["full"] = {
            "posterior_stats": ps,
            "beta_samples": bs,
            "losses": losses,
            "train_r2": r2,
        }
    if cli.run_collapsed:
        m, ps, bs, losses, r2 = run_one_recovery(
            data_train, config, truth, collapsed=True, label="collapsed", output_dir=cli.output_dir
        )
        all_metrics.append(m)
        fit_results["collapsed"] = {
            "posterior_stats": ps,
            "beta_samples": bs,
            "losses": losses,
            "train_r2": r2,
        }

    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        summary.to_csv(os.path.join(cli.output_dir, "recovery_summary.csv"), index=False)
        logger.info("Recovery summary:\n%s", summary.to_string(index=False))

        if not cli.no_plots and fit_results:
            plots_dir = os.path.join(cli.output_dir, "plots")
            saved = plot_recovery_dashboard(
                truth,
                fit_results,
                data_train,
                config,
                summary,
                plots_dir,
            )
            logger.info("Wrote %s figures to %s", len(saved), plots_dir)
            for p in saved:
                logger.info("  %s", p)

    logger.info("Done. Outputs in %s", cli.output_dir)


if __name__ == "__main__":
    main()
