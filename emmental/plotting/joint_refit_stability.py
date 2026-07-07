#!/usr/bin/env python3
"""
Assess stability across joint refits (run_1 .. run_N).

Reads per-run artifacts under {config}/joint/run_*:
  tau_T.csv, w_g.csv, rho_g.csv, train/test_r2_scores.csv

Writes PNG stability_report/ under each config's joint directory (plots only).
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DEFAULT = "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01"
CONFIGS_DEFAULT = ["full", "no_wg", "no_rhog", "no_wg_rhog"]


def _boxplot(ax, data, labels=None, **kwargs):
    """ax.boxplot with tick/label kwarg compatible across matplotlib versions."""
    if labels is not None:
        try:
            return ax.boxplot(data, tick_labels=labels, **kwargs)
        except TypeError:
            return ax.boxplot(data, labels=labels, **kwargs)
    return ax.boxplot(data, **kwargs)


def _legend_outside(ax, **kwargs) -> None:
    defaults = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    defaults.update(kwargs)
    ax.legend(**defaults)


def list_runs(joint_dir: str) -> List[Tuple[int, str]]:
    runs = []
    for name in os.listdir(joint_dir):
        m = re.match(r"run_(\d+)$", name)
        if m and os.path.isdir(os.path.join(joint_dir, name)):
            runs.append((int(m.group(1)), os.path.join(joint_dir, name)))
    return sorted(runs, key=lambda x: x[0])


def load_tau_T(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "tau_T.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"Annotation": "annotation"})
    if "Filter Threshold" in df.columns:
        df = df.rename(columns={"Filter Threshold": "T"})
    return df


def load_gene_r2(run_dir: str, split: str) -> pd.Series:
    path = os.path.join(run_dir, f"{split}_r2_scores.csv")
    df = pd.read_csv(path)
    return df.set_index("gene")["r2"].astype(float)


def load_wg_rhog(run_dir: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    w, r = None, None
    wp = os.path.join(run_dir, "w_g.csv")
    rp = os.path.join(run_dir, "rho_g.csv")
    if os.path.isfile(wp):
        d = pd.read_csv(wp)
        col = "w_g_mean" if "w_g_mean" in d.columns else d.columns[1]
        w = d.set_index("gene")[col].astype(float)
    if os.path.isfile(rp):
        d = pd.read_csv(rp)
        col = "rho_g_mean" if "rho_g_mean" in d.columns else d.columns[1]
        r = d.set_index("gene")[col].astype(float)
    return w, r


def analyze_config(config_name: str, joint_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    runs = list_runs(joint_dir)
    if not runs:
        raise FileNotFoundError(f"No run_* under {joint_dir}")

    run_ids = [r[0] for r in runs]

    # --- tau / T across refits ---
    tau_rows = []
    T_vals = []
    for rid, rdir in runs:
        tt = load_tau_T(rdir)
        T_vals.append(float(tt["T"].iloc[0]))
        for _, row in tt.iterrows():
            ann = row["annotation"]
            tau_rows.append(
                {
                    "refit": rid,
                    "annotation": ann,
                    "tau1": float(row["Tau1"]),
                    "tau2": float(row["Tau2"]) if pd.notna(row.get("Tau2")) else np.nan,
                    "T": float(row["T"]),
                }
            )
    tau_long = pd.DataFrame(tau_rows)

    T_series = pd.Series(T_vals, index=run_ids, name="T")
    tau1_wide = tau_long.pivot(index="annotation", columns="refit", values="tau1")
    tau2_wide = tau_long.pivot(index="annotation", columns="refit", values="tau2")

    # --- w_g / rho_g ---
    wg_mat, rh_mat = {}, {}
    for rid, rdir in runs:
        w, r = load_wg_rhog(rdir)
        if w is not None:
            wg_mat[rid] = w
        if r is not None:
            rh_mat[rid] = r
    wg_sd, rh_sd = None, None
    if wg_mat:
        wg_sd = pd.DataFrame(wg_mat).std(axis=1)
    if rh_mat:
        rh_sd = pd.DataFrame(rh_mat).std(axis=1)

    # --- plots ---
    _plot_T_and_tau(T_series, tau1_wide, tau2_wide, out_dir, config_name)
    _plot_wg_rhog_sd(wg_sd, rh_sd, out_dir, config_name)
    r2_summary = analyze_r2_stability(runs, out_dir, config_name)
    beta_df = analyze_beta_stability(runs, out_dir, config_name)

    global_row = {
        "config": config_name,
        "n_refits": len(runs),
        "T_mean": float(T_series.mean()),
        "T_std": float(T_series.std()),
    }
    global_row.update(r2_summary)
    if beta_df is not None and len(beta_df):
        global_row["beta_mean_pairwise_r_median"] = float(
            beta_df["beta_mean_pairwise_r"].median()
        )
        global_row["beta_median_sd_median"] = float(
            beta_df["beta_median_sd_across_refits"].median()
        )
    pd.DataFrame([global_row]).to_csv(
        os.path.join(out_dir, "global_stability_summary.csv"), index=False
    )
    return global_row


def _plot_T_boxplot(T_series: pd.Series, out_path: str, config_name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    _boxplot(ax, [T_series.values], labels=["T"])
    ax.set_ylabel("T")
    fig.suptitle(f"{config_name}: T across refits", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tau_boxplots(
    tau_wide: pd.DataFrame,
    out_path: str,
    config_name: str,
    tau_label: str,
    color: str = "#4C72B0",
) -> None:
    """One box per annotation: distribution of tau across refits."""
    # drop intercept for tau2 plots if all NaN; keep for tau1
    rows = []
    for ann in tau_wide.index:
        vals = tau_wide.loc[ann].astype(float).dropna().values
        if len(vals) == 0:
            continue
        rows.append((ann, vals))

    if not rows:
        return

    # sort by median absolute value (easy to scan important annotations)
    rows.sort(key=lambda x: np.median(np.abs(x[1])), reverse=True)

    labels = [r[0] for r in rows]
    data = [r[1] for r in rows]
    n = len(labels)
    fig_w = max(10, 0.55 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bp = _boxplot(ax, data, labels=labels, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(tau_label)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{config_name}: {tau_label} per annotation (across refits)", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_T_and_tau(T_series, tau1_wide, tau2_wide, out_dir, config_name):
    _plot_T_boxplot(T_series, os.path.join(out_dir, "T_across_refits_boxplot.png"), config_name)
    _plot_tau_boxplots(
        tau1_wide,
        os.path.join(out_dir, "tau1_by_annotation_boxplot.png"),
        config_name,
        r"$\tau_1$",
        color="#4C72B0",
    )
    if tau2_wide.notna().any().any():
        # tau2 undefined for intercept row
        t2 = tau2_wide.drop(index=["intercept"], errors="ignore")
        _plot_tau_boxplots(
            t2,
            os.path.join(out_dir, "tau2_by_annotation_boxplot.png"),
            config_name,
            r"$\tau_2$",
            color="#DD8452",
        )


def _plot_wg_rhog_sd(
    wg_sd: Optional[pd.Series],
    rh_sd: Optional[pd.Series],
    out_dir: str,
    config_name: str,
) -> None:
    """Histogram of per-gene SD(w_g) and SD(rho_g) across refits."""
    panels = []
    if wg_sd is not None and len(wg_sd.dropna()):
        panels.append((r"$w_g$", wg_sd.dropna().astype(float), "#4C72B0"))
    if rh_sd is not None and len(rh_sd.dropna()):
        panels.append((r"$\rho_g$", rh_sd.dropna().astype(float), "#DD8452"))
    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n + 1.5, 4.5), squeeze=False)
    axes = axes.ravel()

    for ax, (label, vals, color) in zip(axes, panels):
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            ax.set_visible(False)
            continue
        # degenerate (fixed param): nearly all SD ~ 0
        if np.nanmax(finite) < 1e-10:
            ax.hist(finite, bins=5, color=color, edgecolor="white", alpha=0.85)
            panel_title = f"{label}: SD ≈ 0 (fixed)"
        else:
            bins = min(50, max(15, len(finite) // 20))
            ax.hist(finite, bins=bins, color=color, edgecolor="white", alpha=0.85)
            panel_title = f"{label}: SD across refits"
        ax.axvline(np.median(finite), color="black", linestyle="--", linewidth=1, label=f"median={np.median(finite):.3g}")
        ax.axvline(np.mean(finite), color="gray", linestyle=":", linewidth=1, label=f"mean={np.mean(finite):.3g}")
        ax.set_xlabel(f"SD({label})")
        ax.set_ylabel("Number of genes")
        _legend_outside(ax, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(panel_title, pad=8)
        stats_txt = (
            f"n={len(finite)}\n"
            f"mean={np.mean(finite):.4g}\n"
            f"median={np.median(finite):.4g}\n"
            f"p90={np.quantile(finite, 0.9):.4g}\n"
            f"max={np.max(finite):.4g}"
        )
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.suptitle(f"{config_name}: per-gene parameter stability across refits", y=0.98)
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.45, right=0.88)
    fig.savefig(os.path.join(out_dir, "w_g_rho_g_sd_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_beta_mean(run_dir: str, gene_file: str) -> Optional[pd.Series]:
    path = os.path.join(run_dir, "beta_samples", gene_file)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if "variant_id_G" not in df.columns:
        return None
    return df.set_index("variant_id_G")["beta_mean"].astype(float)


def _mean_pairwise_corr(mat: pd.DataFrame) -> float:
    cols = list(mat.columns)
    if len(cols) < 2:
        return float("nan")
    cors = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            cors.append(float(mat[c1].corr(mat[c2])))
    return float(np.nanmean(cors)) if cors else float("nan")


def analyze_beta_stability(
    runs: List[Tuple[int, str]],
    out_dir: str,
    config_name: str,
    min_variants: int = 5,
) -> Optional[pd.DataFrame]:
    """Per-gene beta posterior-mean stability across joint refits."""
    beta_dir = os.path.join(runs[0][1], "beta_samples")
    if not os.path.isdir(beta_dir):
        return None

    gene_files = sorted(f for f in os.listdir(beta_dir) if f.endswith("_beta.csv.gz"))
    if not gene_files:
        return None

    rows = []
    for gf in gene_files:
        gene_key = gf.replace("_beta.csv.gz", "")
        mats: Dict[int, pd.Series] = {}
        for rid, rdir in runs:
            s = load_beta_mean(rdir, gf)
            if s is not None:
                mats[rid] = s
        if len(mats) < 2:
            continue
        mat = pd.DataFrame(mats).dropna(how="any")
        if mat.shape[0] < min_variants:
            continue
        sd_var = mat.std(axis=1)
        mean_vec = mat.mean(axis=1)
        rows.append(
            {
                "gene": gene_key,
                "n_variants": int(mat.shape[0]),
                "beta_mean_pairwise_r": _mean_pairwise_corr(mat),
                "beta_median_sd_across_refits": float(sd_var.median()),
                "beta_mean_sd_across_refits": float(sd_var.mean()),
                "beta_median_abs_mean": float(mean_vec.abs().median()),
                "beta_rmse_vs_refit_mean": float(
                    np.sqrt(((mat.sub(mean_vec, axis=0)) ** 2).mean().mean())
                ),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("beta_mean_pairwise_r")
    df.to_csv(os.path.join(out_dir, "beta_gene_stability.csv"), index=False)
    df.nsmallest(25, "beta_mean_pairwise_r").to_csv(
        os.path.join(out_dir, "beta_top25_unstable_genes_by_pairwise_r.csv"), index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    color = "#55A868"
    rvals = df["beta_mean_pairwise_r"].dropna()
    axes[0].hist(rvals, bins=40, color=color, edgecolor="white")
    axes[0].set_xlabel("Mean pairwise r(β) across refits")
    axes[0].set_ylabel("Genes")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(
        df["beta_median_abs_mean"],
        df["beta_mean_pairwise_r"],
        s=10,
        alpha=0.5,
        color=color,
    )
    axes[1].set_xlabel("Median |β| (refit-averaged)")
    axes[1].set_ylabel("Mean pairwise r(β)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].hist(
        df["beta_median_sd_across_refits"].dropna(),
        bins=40,
        color=color,
        edgecolor="white",
    )
    axes[2].set_xlabel("Median per-variant SD(β) across refits")
    axes[2].set_ylabel("Genes")
    axes[2].grid(axis="y", alpha=0.3)

    for ax, title in zip(
        axes,
        ["β refit agreement (r)", "|β| vs agreement", "Per-variant β variability"],
    ):
        ax.set_title(title, pad=10)

    fig.suptitle(f"{config_name} — β stability across refits", y=0.98)
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.35)
    fig.savefig(os.path.join(out_dir, "beta_stability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "metric": "beta_mean_pairwise_r_median",
                "value": float(rvals.median()),
            },
            {
                "metric": "beta_mean_pairwise_r_p10",
                "value": float(rvals.quantile(0.1)),
            },
            {
                "metric": "beta_median_sd_median",
                "value": float(df["beta_median_sd_across_refits"].median()),
            },
            {
                "metric": "n_genes_with_beta",
                "value": int(len(df)),
            },
        ]
    )
    summary.to_csv(os.path.join(out_dir, "beta_stability_summary.csv"), index=False)
    return df


def analyze_r2_stability(
    runs: List[Tuple[int, str]],
    out_dir: str,
    config_name: str,
) -> Dict[str, float]:
    """Per-gene R² variability across refits; writes CSVs and PNGs."""
    summary: Dict[str, float] = {}
    for split, color in [("train", "#4C72B0"), ("test", "#DD8452")]:
        mat = pd.DataFrame({rid: load_gene_r2(rdir, split) for rid, rdir in runs})
        mat.to_csv(os.path.join(out_dir, f"{split}_r2_by_refit.csv"))
        mean = mat.mean(axis=1)
        sd = mat.std(axis=1)
        gene_df = pd.DataFrame(
            {
                "gene": mean.index,
                "mean_r2": mean.values,
                "sd_r2": sd.values,
                "cv_r2": (sd / mean.replace(0, np.nan)).values,
            }
        ).sort_values("sd_r2", ascending=False)
        gene_df.to_csv(os.path.join(out_dir, f"{split}_r2_gene_stability.csv"), index=False)
        gene_df.nlargest(25, "sd_r2").to_csv(
            os.path.join(out_dir, f"{split}_top25_unstable_genes_by_sd.csv"), index=False
        )

        corr = mat.corr()
        corr.to_csv(os.path.join(out_dir, f"{split}_r2_refit_correlation.csv"))

        summary[f"{split}_mean_gene_sd"] = float(sd.mean())
        summary[f"{split}_median_gene_sd"] = float(sd.median())
        summary[f"{split}_mean_run_corr"] = float(
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().mean()
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        axes[0].hist(sd.dropna(), bins=40, color=color, edgecolor="white")
        axes[0].set_xlabel("SD(R²) per gene")
        axes[0].set_ylabel("Count")

        axes[1].scatter(mean, sd, s=8, alpha=0.5, color=color)
        axes[1].set_xlabel("Mean R² across refits")
        axes[1].set_ylabel("SD R² across refits")

        im = axes[2].imshow(corr.values, vmin=0, vmax=1, cmap="viridis")
        axes[2].set_xticks(range(len(corr.columns)))
        axes[2].set_yticks(range(len(corr.index)))
        axes[2].set_xticklabels(corr.columns, rotation=45, ha="right")
        axes[2].set_yticklabels(corr.index)
        cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Correlation", rotation=90, va="center")

        panel_titles = ["Per-gene R² variability", "Mean vs SD", "Refit×refit correlation"]
        for ax, pt in zip(axes, panel_titles):
            ax.set_title(pt, pad=10)
        axes[0].grid(axis="y", alpha=0.3)
        axes[1].grid(axis="y", alpha=0.3)

        fig.suptitle(f"{config_name} — {split} R² stability across refits", y=0.98)
        fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.38)
        fig.savefig(os.path.join(out_dir, f"{split}_r2_stability.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "r2_stability_summary.csv"), index=False)
    return summary


def _plot_r2_stability(joint_dir, runs, out_dir, config_name):
    analyze_r2_stability(runs, out_dir, config_name)


def discover_train_common01_joints(
    myout_root: str,
    *,
    exclude_substrings: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Return (label, joint_dir) for train_common01* experiments with run_* refits."""
    exclude = exclude_substrings or ["tau1norm"]
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(myout_root):
        return out

    for name in sorted(os.listdir(myout_root)):
        if not name.startswith("train_common01"):
            continue
        if any(x in name for x in exclude):
            continue
        exp_root = os.path.join(myout_root, name)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT_DEFAULT)
    p.add_argument("--myout_root", default="/gpfs/commons/home/vmazeeva/firvTWAS_myout")
    p.add_argument("--configs", nargs="+", default=CONFIGS_DEFAULT)
    p.add_argument(
        "--discover_all",
        action="store_true",
        help="Run all train_common01* joint dirs under --myout_root (excludes tau1norm)",
    )
    p.add_argument(
        "--joint_dirs",
        nargs="+",
        default=None,
        help=(
            "Standalone joint dirs (e.g. train_common01_full_collapse/joint). "
            "Label = parent experiment basename unless --labels provided."
        ),
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display names for --joint_dirs (same length as joint_dirs).",
    )
    p.add_argument(
        "--cross_summary",
        default=None,
        help="Optional path to write combined global_stability_summary rows",
    )
    args = p.parse_args()

    cross_rows = []

    if args.discover_all:
        pairs = discover_train_common01_joints(args.myout_root)
        for label, joint_dir in pairs:
            if len(list_runs(joint_dir)) < 2:
                print(f"Skipping {label}: <2 refits")
                continue
            out_dir = os.path.join(joint_dir, "stability_report")
            print(f"Analyzing {label} ({joint_dir}) ...")
            try:
                row = analyze_config(label, joint_dir, out_dir)
                cross_rows.append(row)
                print(f"  -> {out_dir}")
            except Exception as e:
                print(f"  FAILED {label}: {e}")
        if cross_rows and args.cross_summary:
            pd.DataFrame(cross_rows).to_csv(args.cross_summary, index=False)
            print(f"Cross-config summary: {args.cross_summary}")
        return

    if args.joint_dirs:
        labels = args.labels or [
            os.path.basename(os.path.dirname(os.path.abspath(d.rstrip("/"))))
            for d in args.joint_dirs
        ]
        if len(labels) != len(args.joint_dirs):
            raise SystemExit("--labels must match --joint_dirs in length")
        for label, joint_dir in zip(labels, args.joint_dirs):
            joint_dir = os.path.abspath(joint_dir)
            out_dir = os.path.join(joint_dir, "stability_report")
            print(f"Analyzing {label} ({joint_dir}) ...")
            try:
                row = analyze_config(label, joint_dir, out_dir)
                cross_rows.append(row)
                print(f"  -> {out_dir}")
            except Exception as e:
                print(f"  FAILED {label}: {e}")
        if cross_rows and args.cross_summary:
            pd.DataFrame(cross_rows).to_csv(args.cross_summary, index=False)
        return

    for cfg_name in args.configs:
        joint_dir = os.path.join(args.root, cfg_name, "joint")
        out_dir = os.path.join(joint_dir, "stability_report")
        print(f"Analyzing {cfg_name} ...")
        try:
            row = analyze_config(cfg_name, joint_dir, out_dir)
            cross_rows.append(row)
            print(f"  -> {out_dir}")
        except Exception as e:
            print(f"  FAILED {cfg_name}: {e}")

    if cross_rows and args.cross_summary:
        pd.DataFrame(cross_rows).to_csv(args.cross_summary, index=False)


if __name__ == "__main__":
    main()
