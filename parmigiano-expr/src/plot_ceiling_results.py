import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IN_DIR = "diagnostics_ceiling"
OUT_DIR = os.path.join(IN_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def short_cfg_name(cfg):
    return cfg.split("__")[-1]


def plot_train_test_gap(gap_df):
    df = gap_df.copy()
    df["label"] = df["config"].map(short_cfg_name)
    df = df.sort_values("mean_test_r2", ascending=False)

    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, df["mean_train_r2"], width=width, label="Train mean R2")
    ax.bar(x + width / 2, df["mean_test_r2"], width=width, label="Test mean R2")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=20, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Parmigiano Train vs Test Mean R2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "train_vs_test_mean_r2.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["label"], df["mean_train_test_gap"])
    ax.set_ylabel("Train-Test Gap (mean R2)")
    ax.set_title("Generalization Gap by Configuration")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "generalization_gap.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["label"], df["frac_test_gt_0p01"], marker="o", label="Test R2 > 0.01")
    ax.plot(df["label"], df["frac_test_gt_0p1"], marker="o", label="Test R2 > 0.1")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of Genes")
    ax.set_title("Test R2 Threshold Hit Rate")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "test_r2_threshold_proportions.png"), dpi=150)
    plt.close(fig)


def plot_vs_baselines(matched_df):
    df = matched_df.copy()
    df["label"] = df["config"].map(short_cfg_name)

    fig, ax = plt.subplots(figsize=(9, 5))
    for bl in sorted(df["baseline"].unique()):
        sub = df[df["baseline"] == bl].sort_values("label")
        ax.plot(sub["label"], sub["mean_delta"], marker="o", label=f"Delta vs {bl}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Mean Delta R2 (Parm - Baseline)")
    ax.set_title("Mean Test R2 Delta vs Baselines")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mean_delta_vs_baselines.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for bl in sorted(df["baseline"].unique()):
        sub = df[df["baseline"] == bl].sort_values("label")
        ax.plot(sub["label"], sub["beats_baseline_frac"], marker="o", label=f"Beat frac vs {bl}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of Genes Parmigiano Beats Baseline")
    ax.set_title("Win Rate vs Baselines")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "beat_fraction_vs_baselines.png"), dpi=150)
    plt.close(fig)


def main():
    gap_path = os.path.join(IN_DIR, "train_test_gap_summary.csv")
    matched_path = os.path.join(IN_DIR, "matched_summary.csv")
    gap_df = pd.read_csv(gap_path)
    matched_df = pd.read_csv(matched_path)

    plot_train_test_gap(gap_df)
    plot_vs_baselines(matched_df)
    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()

