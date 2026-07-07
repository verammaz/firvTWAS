#!/usr/bin/env python3
"""
Side-by-side boxplots of τ₁ / τ₂ per annotation across joint refits,
comparing two joint training runs (e.g. collapsed vs non-collapsed full).
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joint_refit_stability import _boxplot, load_tau_T, list_runs


def _tau_long_from_joint(joint_dir: str, model_label: str) -> pd.DataFrame:
    rows = []
    for rid, rdir in list_runs(joint_dir):
        tt = load_tau_T(rdir)
        for _, row in tt.iterrows():
            rows.append(
                {
                    "model": model_label,
                    "refit": rid,
                    "annotation": row["annotation"],
                    "tau1": float(row["Tau1"]),
                    "tau2": float(row["Tau2"]) if pd.notna(row.get("Tau2")) else np.nan,
                    "T": float(row["T"]),
                }
            )
    return pd.DataFrame(rows)


def _annotation_order(df: pd.DataFrame, tau_col: str) -> List[str]:
    med = (
        df.groupby("annotation")[tau_col]
        .apply(lambda s: np.median(np.abs(s.astype(float))))
        .sort_values(ascending=False)
    )
    return med.index.tolist()


def _plot_paired_tau_boxplots(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
    tau_col: str,
    tau_title: str,
    out_path: str,
    drop_intercept: bool = False,
) -> None:
    combined = pd.concat([left, right], ignore_index=True)
    if drop_intercept:
        combined = combined[combined["annotation"] != "intercept"]

    annotations = _annotation_order(combined, tau_col)
    if not annotations:
        return

    n = len(annotations)
    x = np.arange(n)
    width = 0.34
    colors = {left_label: "#4C72B0", right_label: "#DD8452"}

    fig, ax = plt.subplots(figsize=(max(12, 0.65 * n), 5.5))
    for i, (label, sub, offset) in enumerate(
        [(left_label, left, -width / 2), (right_label, right, width / 2)]
    ):
        if drop_intercept:
            sub = sub[sub["annotation"] != "intercept"]
        data = []
        for ann in annotations:
            vals = sub.loc[sub["annotation"] == ann, tau_col].astype(float).dropna().values
            data.append(vals if len(vals) else [np.nan])
        bp = _boxplot(
            ax,
            data,
            positions=x + offset,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.7)
        ax.plot([], [], color=colors[label], linewidth=8, alpha=0.7, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(annotations, rotation=45, ha="right")
    ax.set_ylabel(tau_title)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.suptitle(
        f"{tau_title} per annotation across joint refits\n{left_label} vs {right_label}",
        y=0.98,
    )
    fig.subplots_adjust(top=0.86, bottom=0.22, right=0.82)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_T_across_refits(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
    out_path: str,
) -> None:
    """Boxplot of learned filter threshold T per joint refit."""
    t_left = left.groupby("refit", as_index=False)["T"].first()["T"].astype(float).values
    t_right = right.groupby("refit", as_index=False)["T"].first()["T"].astype(float).values
    colors = {left_label: "#4C72B0", right_label: "#DD8452"}

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bp = ax.boxplot(
        [t_left, t_right],
        labels=[left_label, right_label],
        patch_artist=True,
        showfliers=True,
    )
    for patch, label in zip(bp["boxes"], [left_label, right_label]):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.7)
    ax.set_ylabel(r"Filter threshold $T$")
    ax.set_title(r"Learned $T$ across joint refits")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare τ boxplots across joint refits")
    p.add_argument(
        "--joint_dir_a",
        default="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01_full_collapse/joint",
    )
    p.add_argument(
        "--joint_dir_b",
        default="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01/full/joint",
    )
    p.add_argument("--label_a", default="collapsed")
    p.add_argument("--label_b", default="non-collapsed full")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    joint_a = os.path.abspath(args.joint_dir_a)
    joint_b = os.path.abspath(args.joint_dir_b)
    out_dir = args.out_dir or os.path.join(
        joint_a, "param_compare_noncollapsed_full"
    )
    os.makedirs(out_dir, exist_ok=True)

    tau_a = _tau_long_from_joint(joint_a, args.label_a)
    tau_b = _tau_long_from_joint(joint_b, args.label_b)
    combined = pd.concat([tau_a, tau_b], ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "tau_per_refit_long.csv"), index=False)

    _plot_paired_tau_boxplots(
        tau_a,
        tau_b,
        args.label_a,
        args.label_b,
        "tau1",
        r"$\tau_1$",
        os.path.join(out_dir, "tau1_by_annotation_refit_compare.png"),
        drop_intercept=False,
    )
    _plot_paired_tau_boxplots(
        tau_a,
        tau_b,
        args.label_a,
        args.label_b,
        "tau2",
        r"$\tau_2$",
        os.path.join(out_dir, "tau2_by_annotation_refit_compare.png"),
        drop_intercept=True,
    )
    _plot_paired_tau_boxplots(
        tau_a,
        tau_b,
        args.label_a,
        args.label_b,
        "tau1",
        r"$\tau_1$ (ex-intercept)",
        os.path.join(out_dir, "tau1_ex_intercept_by_annotation_refit_compare.png"),
        drop_intercept=True,
    )

    _plot_T_across_refits(
        tau_a,
        tau_b,
        args.label_a,
        args.label_b,
        os.path.join(out_dir, "T_across_refits_compare.png"),
    )

    print(f"Refits: {args.label_a}={tau_a['refit'].nunique()}  {args.label_b}={tau_b['refit'].nunique()}")
    print(f"Wrote plots under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
