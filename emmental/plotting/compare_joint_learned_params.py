#!/usr/bin/env python3
"""
Compare learned joint parameters between two Emmental experiments (e.g. collapsed vs non-collapsed).

Reads per-refit artifacts under {root}/joint/run_*:
  tau_T.csv, w_g.csv, rho_g.csv

Writes summary CSVs and scatter plots under {out_dir}.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def list_runs(joint_dir: str) -> List[Tuple[int, str]]:
    runs = []
    for name in os.listdir(joint_dir):
        m = re.match(r"run_(\d+)$", name)
        if m and os.path.isdir(os.path.join(joint_dir, name)):
            runs.append((int(m.group(1)), os.path.join(joint_dir, name)))
    return sorted(runs, key=lambda x: x[0])


def load_tau_T(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(run_dir, "tau_T.csv"))
    df = df.rename(columns={"Annotation": "annotation", "Filter Threshold": "T"})
    return df.set_index("annotation")


def load_gene_param(run_dir: str, param: str) -> pd.Series:
    path = os.path.join(run_dir, f"{param}.csv")
    df = pd.read_csv(path)
    col = f"{param}_mean" if f"{param}_mean" in df.columns else df.columns[1]
    return df.set_index("gene")[col].astype(float)


def _corr(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    aligned = pd.concat([x, y], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return {"n": len(aligned), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}
    a = aligned.iloc[:, 0].astype(float).values
    b = aligned.iloc[:, 1].astype(float).values
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {
        "n": len(aligned),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def mean_across_runs(
    joint_dir: str,
) -> Tuple[pd.Series, pd.Series, float, pd.Series, pd.Series]:
    runs = list_runs(joint_dir)
    if not runs:
        raise FileNotFoundError(f"No run_* under {joint_dir}")

    tau1_frames, tau2_frames, T_vals = [], [], []
    wg_frames, rh_frames = [], []

    for _, rdir in runs:
        tt = load_tau_T(rdir)
        tau1_frames.append(tt["Tau1"].astype(float).rename(os.path.basename(rdir)))
        tau2_frames.append(tt["Tau2"].astype(float).rename(os.path.basename(rdir)))
        T_vals.append(float(tt["T"].iloc[0]))
        wg_frames.append(load_gene_param(rdir, "w_g").rename(os.path.basename(rdir)))
        rh_frames.append(load_gene_param(rdir, "rho_g").rename(os.path.basename(rdir)))

    tau1_mean = pd.concat(tau1_frames, axis=1).mean(axis=1)
    tau2_mean = pd.concat(tau2_frames, axis=1).mean(axis=1)
    wg_mean = pd.concat(wg_frames, axis=1).mean(axis=1)
    rh_mean = pd.concat(rh_frames, axis=1).mean(axis=1)
    T_mean = float(np.mean(T_vals))
    return tau1_mean, tau2_mean, T_mean, wg_mean, rh_mean


def per_run_correlations(
    joint_a: str,
    joint_b: str,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    runs_a = {rid: p for rid, p in list_runs(joint_a)}
    runs_b = {rid: p for rid, p in list_runs(joint_b)}
    shared_ids = sorted(set(runs_a) & set(runs_b))
    rows = []
    for rid in shared_ids:
        tt_a = load_tau_T(runs_a[rid])
        tt_b = load_tau_T(runs_b[rid])
        tau2_mask = tt_a.index != "intercept"
        for param, series_a, series_b in [
            ("tau1", tt_a["Tau1"], tt_b["Tau1"]),
            ("tau2", tt_a.loc[tau2_mask, "Tau2"], tt_b.loc[tau2_mask, "Tau2"]),
            ("w_g", load_gene_param(runs_a[rid], "w_g"), load_gene_param(runs_b[rid], "w_g")),
            ("rho_g", load_gene_param(runs_a[rid], "rho_g"), load_gene_param(runs_b[rid], "rho_g")),
        ]:
            c = _corr(series_a.astype(float), series_b.astype(float))
            rows.append(
                {
                    "refit": rid,
                    "param": param,
                    "label_a": label_a,
                    "label_b": label_b,
                    **c,
                }
            )
        rows.append(
            {
                "refit": rid,
                "param": "T",
                "label_a": label_a,
                "label_b": label_b,
                "n": 1,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "value_a": float(tt_a["T"].iloc[0]),
                "value_b": float(tt_b["T"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _scatter(
    x: pd.Series,
    y: pd.Series,
    xlab: str,
    ylab: str,
    title: str,
    out_path: str,
    annotate: Optional[List[str]] = None,
) -> None:
    aligned = pd.concat([x, y], axis=1, join="inner").dropna()
    aligned.columns = ["x", "y"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(aligned["x"], aligned["y"], alpha=0.7, s=36, edgecolors="none")
    if annotate:
        for idx in annotate:
            if idx in aligned.index:
                ax.annotate(
                    idx,
                    (aligned.loc[idx, "x"], aligned.loc[idx, "y"]),
                    fontsize=7,
                    alpha=0.8,
                )
    lim_lo = min(aligned["x"].min(), aligned["y"].min())
    lim_hi = max(aligned["x"].max(), aligned["y"].max())
    pad = 0.05 * (lim_hi - lim_lo) if lim_hi > lim_lo else 0.1
    ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad], "k--", lw=1, alpha=0.4)
    c = _corr(aligned["x"], aligned["y"])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(
        f"{title}\n"
        f"r={c['pearson_r']:.3f} (p={c['pearson_p']:.2e}), "
        f"ρ={c['spearman_r']:.3f}, n={c['n']}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare joint learned parameters")
    p.add_argument("--root_a", required=True, help="First experiment root")
    p.add_argument("--root_b", required=True, help="Second experiment root")
    p.add_argument("--label_a", default=None)
    p.add_argument("--label_b", default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    root_a = os.path.abspath(args.root_a)
    root_b = os.path.abspath(args.root_b)
    label_a = args.label_a or os.path.basename(root_a.rstrip("/"))
    label_b = args.label_b or os.path.basename(root_b.rstrip("/"))
    joint_a = os.path.join(root_a, "joint")
    joint_b = os.path.join(root_b, "joint")
    out_dir = args.out_dir or os.path.join(root_a, "joint", f"param_compare_{label_b}")
    os.makedirs(out_dir, exist_ok=True)

    tau1_a, tau2_a, T_a, wg_a, rh_a = mean_across_runs(joint_a)
    tau1_b, tau2_b, T_b, wg_b, rh_b = mean_across_runs(joint_b)

    tau2_mask = tau1_a.index.intersection(tau1_b.index)
    tau2_mask = [a for a in tau2_mask if a != "intercept"]

    tau1_ex = [a for a in tau1_a.index.intersection(tau1_b.index) if a != "intercept"]

    summary_rows = []
    for param, sa, sb in [
        ("tau1", tau1_a, tau1_b),
        ("tau1_ex_intercept", tau1_a.loc[tau1_ex], tau1_b.loc[tau1_ex]),
        ("tau2", tau2_a.loc[tau2_mask], tau2_b.loc[tau2_mask]),
        ("w_g", wg_a, wg_b),
        ("rho_g", rh_a, rh_b),
    ]:
        c = _corr(sa, sb)
        summary_rows.append({"comparison": "mean_across_refits", "param": param, **c})

    summary_rows.append(
        {
            "comparison": "mean_across_refits",
            "param": "T",
            "n": 1,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "value_a": T_a,
            "value_b": T_b,
        }
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "param_correlation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    per_run_df = per_run_correlations(joint_a, joint_b, label_a, label_b)
    per_run_path = os.path.join(out_dir, "param_correlation_per_refit.csv")
    per_run_df.to_csv(per_run_path, index=False)

    means_path = os.path.join(out_dir, "param_means_comparison.csv")
    pd.DataFrame(
        {
            f"tau1_{label_a}": tau1_a,
            f"tau1_{label_b}": tau1_b,
            f"tau2_{label_a}": tau2_a,
            f"tau2_{label_b}": tau2_b,
        }
    ).to_csv(means_path)
    pd.DataFrame(
        {
            f"w_g_{label_a}": wg_a,
            f"w_g_{label_b}": wg_b,
            f"rho_g_{label_a}": rh_a,
            f"rho_g_{label_b}": rh_b,
        }
    ).to_csv(os.path.join(out_dir, "gene_param_means_comparison.csv"))

    top_tau1 = tau1_a.drop("intercept", errors="ignore").abs().sort_values(ascending=False).head(5).index.tolist()

    _scatter(
        tau1_a.loc[tau1_ex], tau1_b.loc[tau1_ex],
        f"τ₁ mean ({label_a})", f"τ₁ mean ({label_b})",
        f"Joint τ₁ (ex-intercept): {label_a} vs {label_b}",
        os.path.join(out_dir, "scatter_tau1_ex_intercept_mean.png"),
        annotate=top_tau1,
    )
    _scatter(
        tau2_a.loc[tau2_mask], tau2_b.loc[tau2_mask],
        f"τ₂ mean ({label_a})", f"τ₂ mean ({label_b})",
        f"Joint τ₂ (non-intercept): {label_a} vs {label_b}",
        os.path.join(out_dir, "scatter_tau2_mean.png"),
        annotate=top_tau1,
    )
    _scatter(
        wg_a, wg_b,
        f"w_g mean ({label_a})", f"w_g mean ({label_b})",
        f"Per-gene w_g: {label_a} vs {label_b}",
        os.path.join(out_dir, "scatter_w_g_mean.png"),
    )
    _scatter(
        rh_a, rh_b,
        f"ρ_g mean ({label_a})", f"ρ_g mean ({label_b})",
        f"Per-gene ρ_g: {label_a} vs {label_b}",
        os.path.join(out_dir, "scatter_rho_g_mean.png"),
    )

    runs_a = len(list_runs(joint_a))
    runs_b = len(list_runs(joint_b))
    print(f"Compared {label_a} ({runs_a} refits) vs {label_b} ({runs_b} refits)")
    print(f"T: {label_a}={T_a:.6f}  {label_b}={T_b:.6f}")
    for _, row in summary_df.iterrows():
        if row["param"] == "T":
            continue
        print(
            f"{row['param']:6s}  r={row['pearson_r']:7.4f}  "
            f"ρ={row['spearman_r']:7.4f}  n={int(row['n'])}"
        )
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
