"""
Ceiling analysis for Parmigiano vs baseline models.

Purpose
-------
Given a set of Parmigiano experiment roots (each containing run_*/train_r2_scores.csv
and test_r2_scores.csv), quantify:
  1) Matched-gene performance vs baselines (intersection only)
  2) Train->test generalization gap
  3) Top improved / degraded genes vs each baseline

Outputs
-------
Writes CSVs to OUT_DIR:
  - matched_summary.csv
  - train_test_gap_summary.csv
  - per_gene_vs_<baseline>__<config>.csv
  - top_improved_vs_<baseline>__<config>.csv
  - top_degraded_vs_<baseline>__<config>.csv
"""

import glob
import os
import yaml
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# User-configurable paths
# ---------------------------------------------------------------------------
CONFIG_ROOTS = [
    # Example:
    "topgenes_neg_annotations_stability_random_genes/control_full",
    #"topgenes_neg_annotations_stability_random_genes/combined_clip5_smooth_wgpos"
    "topgenes_neg_annotations_stability_random_genes/lin2_clip5",
    "topgenes_neg_annotations_stability_random_genes/smooth_gate",
    "topgenes_neg_annotations_stability_random_genes/wg_positive"

]

BASELINE_DIR = "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full"
BASELINES = ["ridge", "bayesian_ridge", "lasso", "elasticnet"]
OUT_DIR = "diagnostics_ceiling"


def _safe_name(path):
    return path.strip("/").replace("/", "__")


def load_baselines(chroms_needed):
    out = {}
    for bl in BASELINES:
        dfs = []
        for chrom in chroms_needed:
            path = os.path.join(BASELINE_DIR, bl, f"chr{chrom}.tsv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path, sep="\t", index_col=0))
        if dfs:
            out[bl] = pd.concat(dfs)
    return out


def summarize_parmigiano_config(cfg_root):
    run_dirs = sorted(glob.glob(os.path.join(cfg_root, "run_*")))
    if not run_dirs:
        raise ValueError(f"No run_* folders found under: {cfg_root}")

    train_list, test_list = [], []
    for i, rd in enumerate(run_dirs):
        tr_path = os.path.join(rd, "train_r2_scores.csv")
        te_path = os.path.join(rd, "test_r2_scores.csv")
        if not (os.path.exists(tr_path) and os.path.exists(te_path)):
            continue
        tr = pd.read_csv(tr_path).set_index("gene")["r2"].rename(f"run{i}")
        te = pd.read_csv(te_path).set_index("gene")["r2"].rename(f"run{i}")
        train_list.append(tr)
        test_list.append(te)

    if len(train_list) == 0 or len(test_list) == 0:
        raise ValueError(f"No valid run score files found under: {cfg_root}")

    tr_df = pd.concat(train_list, axis=1)
    te_df = pd.concat(test_list, axis=1)

    tr_mean = tr_df.mean(axis=1)
    te_mean = te_df.mean(axis=1)
    tr_std = tr_df.std(axis=1)
    te_std = te_df.std(axis=1)

    return {
        "cfg_root": cfg_root,
        "n_runs": te_df.shape[1],
        "train_df": tr_df,
        "test_df": te_df,
        "train_mean": tr_mean,
        "test_mean": te_mean,
        "train_std": tr_std,
        "test_std": te_std,
    }


def train_test_gap_summary(cfg_name, summ):
    per_gene = pd.DataFrame(
        {
            "gene": summ["test_mean"].index,
            "r2_train_parmigiano": summ["train_mean"].values,
            "r2_test_parmigiano": summ["test_mean"].values,
            "train_test_gap": (summ["train_mean"] - summ["test_mean"]).values,
            "train_std_across_runs": summ["train_std"].values,
            "test_std_across_runs": summ["test_std"].values,
        }
    ).sort_values("train_test_gap", ascending=False)
    per_gene["config"] = cfg_name
    return per_gene


def matched_vs_baseline(cfg_name, summ, baselines):
    rows = []
    per_gene_tables = {}

    parm = summ["test_mean"].rename("r2_parmigiano")

    for bl_name, bl_df in baselines.items():
        if "R2_test" not in bl_df.columns:
            continue
        bl = bl_df["R2_test"].rename(f"r2_{bl_name}")
        merged = pd.concat([parm, bl], axis=1, join="inner").dropna()
        if merged.empty:
            continue

        merged["delta"] = merged["r2_parmigiano"] - merged[f"r2_{bl_name}"]
        merged = merged.sort_values("delta", ascending=False)
        merged["gene"] = merged.index
        merged["config"] = cfg_name
        per_gene_tables[bl_name] = merged.reset_index(drop=True)

        rows.append(
            {
                "config": cfg_name,
                "baseline": bl_name,
                "matched_genes": int(len(merged)),
                "parm_mean_r2_matched": float(merged["r2_parmigiano"].mean()),
                "baseline_mean_r2_matched": float(merged[f"r2_{bl_name}"].mean()),
                "mean_delta": float(merged["delta"].mean()),
                "median_delta": float(merged["delta"].median()),
                "beats_baseline_frac": float((merged["delta"] > 0).mean()),
                "parm_frac_r2_gt_0p01": float((merged["r2_parmigiano"] > 0.01).mean()),
                "parm_frac_r2_gt_0p1": float((merged["r2_parmigiano"] > 0.1).mean()),
                f"{bl_name}_frac_r2_gt_0p01": float((merged[f"r2_{bl_name}"] > 0.01).mean()),
                f"{bl_name}_frac_r2_gt_0p1": float((merged[f"r2_{bl_name}"] > 0.1).mean()),
            }
        )

    return pd.DataFrame(rows), per_gene_tables


def infer_chroms_from_config(cfg_root):
    run_dirs = sorted(glob.glob(os.path.join(cfg_root, "run_*")))
    if not run_dirs:
        return []
    cfg_path = os.path.join(run_dirs[0], "config.yaml")
    if not os.path.exists(cfg_path):
        return []
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    genes = cfg.get("genes", [])
    chroms = sorted({g.split("/")[0].replace("chr", "") for g in genes}, key=lambda x: int(x))
    return chroms


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if len(CONFIG_ROOTS) == 0:
        raise ValueError("Set CONFIG_ROOTS at top of file before running.")

    all_matched_rows = []
    all_gap_rows = []

    for cfg_root in CONFIG_ROOTS:
        cfg_name = _safe_name(cfg_root)
        print(f"\n=== {cfg_root} ===")

        summ = summarize_parmigiano_config(cfg_root)
        print(f"Runs: {summ['n_runs']}, genes (test union): {len(summ['test_mean'])}")
        print(
            f"Parmigiano mean test R2 (gene-mean): {summ['test_mean'].mean():.4f}, "
            f"median: {summ['test_mean'].median():.4f}"
        )

        gap_df = train_test_gap_summary(cfg_name, summ)
        gap_df.to_csv(os.path.join(OUT_DIR, f"train_test_gap__{cfg_name}.csv"), index=False)
        all_gap_rows.append(
            {
                "config": cfg_name,
                "genes": int(len(gap_df)),
                "mean_train_r2": float(gap_df["r2_train_parmigiano"].mean()),
                "mean_test_r2": float(gap_df["r2_test_parmigiano"].mean()),
                "mean_train_test_gap": float(gap_df["train_test_gap"].mean()),
                "median_train_test_gap": float(gap_df["train_test_gap"].median()),
                "frac_test_gt_0p01": float((gap_df["r2_test_parmigiano"] > 0.01).mean()),
                "frac_test_gt_0p1": float((gap_df["r2_test_parmigiano"] > 0.1).mean()),
            }
        )

        chroms = infer_chroms_from_config(cfg_root)
        baselines = load_baselines(chroms)
        if len(baselines) == 0:
            print(f"[warn] No baseline files found for {cfg_root}; skipping matched comparison.")
            continue

        matched_df, per_gene = matched_vs_baseline(cfg_name, summ, baselines)
        if not matched_df.empty:
            all_matched_rows.append(matched_df)
            for bl_name, table in per_gene.items():
                table.to_csv(
                    os.path.join(OUT_DIR, f"per_gene_vs_{bl_name}__{cfg_name}.csv"),
                    index=False,
                )
                table.head(25).to_csv(
                    os.path.join(OUT_DIR, f"top_improved_vs_{bl_name}__{cfg_name}.csv"),
                    index=False,
                )
                table.tail(25).sort_values("delta").to_csv(
                    os.path.join(OUT_DIR, f"top_degraded_vs_{bl_name}__{cfg_name}.csv"),
                    index=False,
                )

    if all_matched_rows:
        matched_summary = pd.concat(all_matched_rows, ignore_index=True)
        matched_summary.to_csv(os.path.join(OUT_DIR, "matched_summary.csv"), index=False)
        print("\nSaved matched_summary.csv")

    gap_summary = pd.DataFrame(all_gap_rows)
    gap_summary.to_csv(os.path.join(OUT_DIR, "train_test_gap_summary.csv"), index=False)
    print("Saved train_test_gap_summary.csv")
    print(f"\nAll outputs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()

