"""
Figures for Emmental parameter recovery (truth vs posterior).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pearson_safe(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _tau1_labels(config: dict, n: int) -> List[str]:
    ann = list(config.get("annotations") or [])
    labels = ["intercept"] + ann
    if len(labels) < n:
        labels = labels + [f"tau1_{i}" for i in range(len(labels), n)]
    return labels[:n]


def _tau2_labels(config: dict, n: int) -> List[str]:
    ann = list(config.get("annotations") or [])
    if len(ann) < n:
        ann = ann + [f"tau2_{i}" for i in range(len(ann), n)]
    return ann[:n]


def save_posterior_estimates(
    posterior_stats: dict,
    beta_samples: dict,
    data,
    run_dir: str,
) -> None:
    """Serialize posterior means/stds for offline re-plotting."""
    out = {}
    for key, stats in posterior_stats.items():
        out[key] = {
            "mean": np.asarray(stats["mean"]).tolist(),
            "std": np.asarray(stats.get("std", [])).tolist(),
        }
    beta = {}
    for gene_name in data.gene_names:
        if gene_name not in beta_samples:
            continue
        arr = beta_samples[gene_name]
        beta[gene_name] = {
            "mean": np.asarray(arr).mean(axis=0).tolist(),
            "std": np.asarray(arr).std(axis=0).tolist(),
        }
    out["beta_by_gene"] = beta
    path = os.path.join(run_dir, "posterior_estimates.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _scatter_truth_vs_est(
    ax,
    truth,
    est_mean,
    est_std=None,
    *,
    labels: Optional[Sequence[str]] = None,
    title: str,
    color: str = "#4C72B0",
):
    t = _to_numpy(truth).ravel()
    e = _to_numpy(est_mean).ravel()
    lo = hi = None
    if est_std is not None:
        s = _to_numpy(est_std).ravel()
        lo = e - 1.96 * s
        hi = e + 1.96 * s
        ax.errorbar(
            t,
            e,
            yerr=1.96 * s,
            fmt="o",
            color=color,
            ecolor=color,
            alpha=0.85,
            capsize=3,
            markersize=6,
            linewidth=1,
            label="posterior mean ± 1.96σ",
        )
    else:
        ax.scatter(t, e, color=color, s=55, alpha=0.85, zorder=3)

    lims = [
        np.nanmin([t.min(), e.min()]),
        np.nanmax([t.max(), e.max()]),
    ]
    pad = 0.05 * (lims[1] - lims[0] + 1e-9)
    lo_lim, hi_lim = lims[0] - pad, lims[1] + pad
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "k--", lw=1, alpha=0.5, zorder=1)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Simulated (truth)")
    ax.set_ylabel("Posterior mean")
    r = _pearson_safe(t, e)
    ax.set_title(f"{title}\nPearson r = {r:.3f}" if np.isfinite(r) else title)
    ax.grid(True, alpha=0.25)

    if labels is not None and len(labels) == len(t):
        for i, lab in enumerate(labels):
            if len(labels) <= 20:
                ax.annotate(
                    lab,
                    (t[i], e[i]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    alpha=0.8,
                )


def plot_vector_param_recovery(
    truth: dict,
    fit_results: Dict[str, dict],
    param: str,
    labels: List[str],
    out_path: str,
    *,
    colors: Optional[Dict[str, str]] = None,
):
    """One panel per model: truth vs posterior for a vector parameter."""
    models = list(fit_results.keys())
    n = len(models)
    if n == 0:
        return
    colors = colors or {"full": "#4C72B0", "collapsed": "#DD8452"}

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        ps = fit_results[model]["posterior_stats"]
        if param not in ps or param not in truth:
            ax.set_visible(False)
            continue
        _scatter_truth_vs_est(
            ax,
            truth[param],
            ps[param]["mean"],
            ps[param].get("std"),
            labels=labels,
            title=f"{param} ({model})",
            color=colors.get(model, "#4C72B0"),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_vector_param(
    truth: dict,
    fit_results: Dict[str, dict],
    param: str,
    labels: List[str],
    out_path: str,
):
    """Both models on one axes (different colors)."""
    if param not in truth or not fit_results:
        return
    colors = {"full": "#4C72B0", "collapsed": "#DD8452"}
    fig, ax = plt.subplots(figsize=(6, 6))
    t = _to_numpy(truth[param]).ravel()

    for model, bundle in fit_results.items():
        ps = bundle["posterior_stats"]
        if param not in ps:
            continue
        e = _to_numpy(ps[param]["mean"]).ravel()
        ax.scatter(t, e, s=55, alpha=0.8, color=colors.get(model, "#333333"), label=model)

    lims = [t.min(), t.max()]
    for bundle in fit_results.values():
        if param in bundle["posterior_stats"]:
            e = _to_numpy(bundle["posterior_stats"][param]["mean"]).ravel()
            lims[0] = min(lims[0], e.min())
            lims[1] = max(lims[1], e.max())
    pad = 0.05 * (lims[1] - lims[0] + 1e-9)
    ax.plot(
        [lims[0] - pad, lims[1] + pad],
        [lims[0] - pad, lims[1] + pad],
        "k--",
        lw=1,
        alpha=0.5,
    )
    ax.set_xlabel("Simulated (truth)")
    ax.set_ylabel("Posterior mean")
    ax.set_title(param)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_pooled_scatter(truth: dict, fit_results: Dict[str, dict], data, out_path: str):
    beta_truth = []
    for gene_name in data.gene_names:
        if gene_name in truth:
            beta_truth.append(_to_numpy(truth[gene_name]["beta"]).ravel())
    if not beta_truth:
        return
    t_all = np.concatenate(beta_truth)

    n = len(fit_results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
    colors = {"full": "#4C72B0", "collapsed": "#DD8452"}
    for ax, (model, bundle) in zip(axes.ravel(), fit_results.items()):
        beta_samples = bundle["beta_samples"]
        parts = []
        for gene_name in data.gene_names:
            if gene_name in beta_samples:
                parts.append(np.asarray(beta_samples[gene_name]).mean(axis=0).ravel())
        if not parts:
            ax.set_visible(False)
            continue
        e_all = np.concatenate(parts)
        n_pts = min(len(t_all), len(e_all))
        _scatter_truth_vs_est(
            ax,
            t_all[:n_pts],
            e_all[:n_pts],
            title=f"β pooled ({model})",
            color=colors.get(model, "#4C72B0"),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wg_rho_scatter(truth: dict, fit_results: Dict[str, dict], data, out_path: str):
    gene_labels = [g.split("/")[-1] if "/" in g else g for g in data.gene_names]
    n_models = len(fit_results)
    fig, axes = plt.subplots(2, n_models, figsize=(5.5 * n_models, 9), squeeze=False)
    colors = {"full": "#4C72B0", "collapsed": "#DD8452"}

    for col, (model, bundle) in enumerate(fit_results.items()):
        ps = bundle["posterior_stats"]
        for row, param in enumerate(("w_g", "rho_g")):
            ax = axes[row, col]
            if param not in ps or param not in truth:
                ax.set_visible(False)
                continue
            _scatter_truth_vs_est(
                ax,
                truth[param],
                ps[param]["mean"],
                ps[param].get("std"),
                labels=gene_labels,
                title=f"{param} ({model})",
                color=colors.get(model, "#4C72B0"),
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_recovery(truth: dict, fit_results: Dict[str, dict], out_path: str):
    if "threshold" not in truth:
        return
    t_val = float(_to_numpy(truth["threshold"]).reshape(()))
    models = list(fit_results.keys())
    means, stds = [], []
    for m in models:
        ps = fit_results[m]["posterior_stats"].get("threshold")
        if ps is None:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(np.asarray(ps["mean"]).reshape(())))
            stds.append(float(np.asarray(ps.get("std", [0])).reshape(())))

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(4, 2 * len(models)), 4))
    ax.axhline(t_val, color="black", ls="--", lw=1.5, label=f"truth T = {t_val:.3f}")
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=6, color="#4C72B0", markersize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("threshold (T)")
    ax.set_title("Threshold recovery")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summary: pd.DataFrame, out_path: str):
    """Bar chart of Pearson r across parameters, full vs collapsed."""
    sub = summary[
        (summary["metric"] == "pearson_r")
        & summary["parameter"].isin(["tau1", "tau2", "w_g", "rho_g", "beta_pooled"])
    ].copy()
    if sub.empty:
        return

    params = list(dict.fromkeys(sub["parameter"].tolist()))
    models = list(dict.fromkeys(sub["model"].tolist()))
    x = np.arange(len(params))
    width = 0.8 / max(len(models), 1)
    colors = {"full": "#4C72B0", "collapsed": "#DD8452"}

    fig, ax = plt.subplots(figsize=(max(8, len(params) * 1.4), 5))
    for i, model in enumerate(models):
        vals = []
        for p in params:
            row = sub[(sub["model"] == model) & (sub["parameter"] == p)]
            vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=model, color=colors.get(model, f"C{i}"))

    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=25, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Pearson r (truth vs posterior mean)")
    ax.set_title("Parameter recovery correlation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(fit_results: Dict[str, dict], out_path: str):
    """ELBO loss vs epoch; linear and log panels with optional rolling mean."""
    models = [m for m, b in fit_results.items() if b.get("losses")]
    if not models:
        return

    colors = {"full": "#4C72B0", "collapsed": "#DD8452"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for model in models:
        losses = np.asarray(fit_results[model]["losses"], dtype=float)
        epochs = np.arange(len(losses))
        color = colors.get(model, None)
        axes[0].plot(epochs, losses, label=model, color=color, lw=1, alpha=0.45)
        axes[1].plot(epochs, losses, label=model, color=color, lw=1, alpha=0.45)
        if len(losses) >= 10:
            window = min(25, max(5, len(losses) // 20))
            roll = pd.Series(losses).rolling(window, min_periods=1).mean().values
            axes[0].plot(epochs, roll, color=color, lw=2, label=f"{model} (roll mean)")
            axes[1].plot(epochs, roll, color=color, lw=2)

    axes[0].set_ylabel("ELBO loss")
    axes[0].set_title("SVI loss during recovery refit")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ELBO loss (log scale)")
    axes[1].grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_single_loss_curve(losses: Sequence[float], label: str, out_path: str) -> None:
    """Save loss curve for one model run (written as soon as that fit finishes)."""
    if not losses:
        return
    y = np.asarray(losses, dtype=float)
    x = np.arange(len(y))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(x, y, color="#4C72B0", lw=1, alpha=0.5)
    axes[1].plot(x, y, color="#4C72B0", lw=1, alpha=0.5)
    if len(y) >= 10:
        window = min(25, max(5, len(y) // 20))
        roll = pd.Series(y).rolling(window, min_periods=1).mean().values
        axes[0].plot(x, roll, color="#DD8452", lw=2, label=f"rolling mean (w={window})")
        axes[1].plot(x, roll, color="#DD8452", lw=2)
    axes[0].set_ylabel("ELBO loss")
    axes[0].set_title(f"SVI loss — {label}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ELBO loss (log scale)")
    axes[1].grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_train_r2(fit_results: Dict[str, dict], data, out_path: str):
    gene_names = getattr(data, "gene_names", None)
    if gene_names is None:
        return

    gene_labels = [g.split("/")[-1] if "/" in g else g for g in gene_names]
    models = list(fit_results.keys())
    if not models:
        return

    width = 0.8 / len(models)
    x = np.arange(len(gene_labels))
    fig, ax = plt.subplots(figsize=(max(6, len(gene_labels) * 1.2), 4.5))
    colors = {"full": "#4C72B0", "collapsed": "#DD8452"}

    for i, model in enumerate(models):
        bundle = fit_results[model]
        if "train_r2" in bundle:
            r2 = bundle["train_r2"]
        elif hasattr(data, "get_gene_data"):
            from emmental_joint import calculate_r2

            r2 = calculate_r2(data, bundle["beta_samples"], gene_names)
        else:
            r2 = {}
        vals = [r2.get(g, np.nan) for g in gene_names]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=model, color=colors.get(model, f"C{i}"))

    ax.set_xticks(x)
    ax.set_xticklabels(gene_labels, rotation=30, ha="right")
    ax.set_ylabel("Train R²")
    ax.set_title("Expression prediction R² (posterior mean β)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lin1_gate_diagnostic(
    data,
    config: dict,
    out_path: str,
    *,
    truth: dict = None,
    fit_results: dict = None,
) -> None:
    """
    Histogram of |Z·τ₁| with reference thresholds (prior mode/mean, init, learned, truth).
    """
    from joint_guide_setup import (
        collect_abs_lin1,
        resolve_threshold_init,
        threshold_prior_mean,
        threshold_prior_mode,
    )

    if not hasattr(data, "get_gene_data"):
        return

    abs_lin1 = collect_abs_lin1(data, config)
    if abs_lin1.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(abs_lin1, bins=80, color="#4C72B0", alpha=0.75, density=True)
    lines = [
        (threshold_prior_mode(config), "Beta prior mode (0.05)", "red"),
        (threshold_prior_mean(config), "Beta prior mean", "orange"),
        (resolve_threshold_init(data, config), "Guide init", "green"),
    ]
    if truth and "threshold" in truth:
        t = float(truth["threshold"]) if not hasattr(truth["threshold"], "item") else float(truth["threshold"])
        lines.append((t, "Simulated truth", "black"))
    if fit_results:
        for model, bundle in fit_results.items():
            ps = bundle.get("posterior_stats", {}).get("threshold")
            if ps is not None:
                tv = float(np.asarray(ps["mean"]).reshape(()))
                lines.append((tv, f"Learned ({model})", "#DD8452"))
    for x, label, color in lines:
        ax.axvline(x, color=color, ls="--", lw=1.5, label=f"{label}: {x:.3f}")
    ax.set_xlabel("|lin1| = |Z·τ₁| (uniform τ₁ prior)")
    ax.set_ylabel("Density")
    ax.set_title("Gate input scale")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    T_vals = np.linspace(max(1e-4, abs_lin1.min() * 0.5), min(1.0, np.percentile(abs_lin1, 99.5)), 80)
    pass_rates = [(abs_lin1 >= t).mean() for t in T_vals]
    ax.plot(T_vals, pass_rates, color="#4C72B0", lw=2)
    for x, label, color in lines:
        ax.axvline(x, color=color, ls="--", lw=1, alpha=0.8)
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("Fraction variants passing gate")
    ax.set_title("Gate pass rate vs T")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_dashboard(
    truth: dict,
    fit_results: Dict[str, dict],
    data,
    config: dict,
    summary: pd.DataFrame,
    plots_dir: str,
) -> List[str]:
    """Generate all recovery figures; returns list of saved paths."""
    os.makedirs(plots_dir, exist_ok=True)
    saved = []

    def _save(name, fn, *args, **kwargs):
        path = os.path.join(plots_dir, name)
        fn(*args, **kwargs, out_path=path)
        saved.append(path)

    n_tau1 = len(_to_numpy(truth["tau1"]).ravel()) if "tau1" in truth else 0
    n_tau2 = len(_to_numpy(truth["tau2"]).ravel()) if "tau2" in truth else 0
    tau1_labels = _tau1_labels(config, n_tau1)
    tau2_labels = _tau2_labels(config, n_tau2)

    _save("recovery_lin1_gate_diagnostic.png", plot_lin1_gate_diagnostic, data, config, truth=truth, fit_results=fit_results)
    _save("recovery_pearson_r_bars.png", plot_metric_comparison, summary)
    _save("recovery_loss_curves.png", plot_loss_curves, fit_results)
    _save("recovery_threshold.png", plot_threshold_recovery, truth, fit_results)
    _save("recovery_wg_rho.png", plot_wg_rho_scatter, truth, fit_results, data)
    _save("recovery_beta_pooled.png", plot_beta_pooled_scatter, truth, fit_results, data)
    _save("recovery_train_r2.png", plot_train_r2, fit_results, data)
    _save("recovery_tau1_by_model.png", plot_vector_param_recovery, truth, fit_results, "tau1", tau1_labels)
    _save("recovery_tau2_by_model.png", plot_vector_param_recovery, truth, fit_results, "tau2", tau2_labels)
    _save("recovery_tau1_overlay.png", plot_overlay_vector_param, truth, fit_results, "tau1", tau1_labels)
    _save("recovery_tau2_overlay.png", plot_overlay_vector_param, truth, fit_results, "tau2", tau2_labels)

    return saved
