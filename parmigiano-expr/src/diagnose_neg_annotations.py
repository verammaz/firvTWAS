"""
Diagnostics for negative-annotation nonlinear Parmigiano runs.

Run:
    python3 diagnose_neg_annotations.py
    RUN_LAMBDA=1 python3 diagnose_neg_annotations.py
    SKIP_LIN2=1 python3 diagnose_neg_annotations.py
"""

import os
import sys
import glob
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "topgenes_neg_annotations_fullT/rare_common_random"
OUT_DIR = "diagnostics_neg_annotations"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    ("tau1_dir_intercept", "Dirichlet (intercept)"),
    ("tau1_dir_no_intercept", "Dirichlet (no intercept)"),
    ("tau1_norm_intercept", "Normal (intercept)"),
    ("tau1_norm_no_intercept", "Normal (no intercept)"),
]

BASELINE_DIR = "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full"
BASELINES = ["ridge", "bayesian_ridge", "lasso", "elasticnet"]


def load_losses(run_dir):
    path = os.path.join(run_dir, "losses.txt")
    return np.loadtxt(path) if os.path.exists(path) else None


def load_tauT(run_dir):
    path = os.path.join(run_dir, "tau_T.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_gene_scalars(run_dir):
    wg = pd.read_csv(os.path.join(run_dir, "w_g.csv"))
    rho = pd.read_csv(os.path.join(run_dir, "rho_g.csv"))
    tr = pd.read_csv(os.path.join(run_dir, "train_r2_scores.csv"))
    te = pd.read_csv(os.path.join(run_dir, "test_r2_scores.csv"))
    return wg, rho, tr, te


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


def summarize_config(cfg):
    cfg_root = os.path.join(ROOT, cfg)
    run_dirs = sorted(glob.glob(os.path.join(cfg_root, "run_*")))
    if not run_dirs:
        return None

    last_losses, converged_flags, all_losses = [], [], []
    tauT_list, T_finals = [], []
    T1_runs, T2_runs, anno = [], [], None
    wg_list, rho_list, tr_list, te_list = [], [], [], []

    for rd in run_dirs:
        L = load_losses(rd)
        if L is not None:
            all_losses.append(L)
            last_losses.append(L[-1])
            tail = L[-max(1, len(L) // 10):]
            rel = (L[-1] - tail.mean()) / max(1e-9, abs(tail.mean()))
            converged_flags.append(abs(rel) < 0.01)

        tT = load_tauT(rd)
        if tT is not None:
            tauT_list.append(tT)
            T_finals.append(float(tT["Filter Threshold"].iloc[0]))
            if anno is None:
                anno = tT["Annotation"].tolist()
            T1_runs.append(tT["Tau1"].values)
            T2_runs.append(tT["Tau2"].values)

        try:
            wg, rho, tr, te = load_gene_scalars(rd)
            wg_list.append(wg)
            rho_list.append(rho)
            tr_list.append(tr)
            te_list.append(te)
        except Exception as e:
            print(f"[warn] {rd}: {e}")

    tau_all = pd.concat([t.assign(run=i) for i, t in enumerate(tauT_list)], ignore_index=True)
    tau1_stats = tau_all.groupby("Annotation")["Tau1"].agg(["mean", "std", "min", "max"])
    tau2_stats = tau_all.groupby("Annotation")["Tau2"].agg(["mean", "std", "min", "max"])

    wg_df = pd.concat([w.set_index("gene")["w_g_mean"].rename(f"run{i}") for i, w in enumerate(wg_list)], axis=1)
    rho_df = pd.concat([r.set_index("gene")["rho_g_mean"].rename(f"run{i}") for i, r in enumerate(rho_list)], axis=1)
    tr_df = pd.concat([t.set_index("gene")["r2"].rename(f"run{i}") for i, t in enumerate(tr_list)], axis=1)
    te_df = pd.concat([t.set_index("gene")["r2"].rename(f"run{i}") for i, t in enumerate(te_list)], axis=1)

    return dict(
        cfg=cfg,
        run_dirs=run_dirs,
        last_losses=np.array(last_losses),
        converged_flags=np.array(converged_flags),
        all_losses=all_losses,
        tauT_list=tauT_list,
        tau1_stats=tau1_stats,
        tau2_stats=tau2_stats,
        T_finals=np.array(T_finals),
        anno=anno,
        Tau1_runs=np.vstack(T1_runs) if T1_runs else None,
        Tau2_runs=np.vstack(T2_runs) if T2_runs else None,
        wg_df=wg_df,
        rho_df=rho_df,
        tr_df=tr_df,
        te_df=te_df,
    )


def plot_losses(all_summaries):
    fig, axes = plt.subplots(1, len(all_summaries), figsize=(5 * len(all_summaries), 4))
    if len(all_summaries) == 1:
        axes = [axes]
    for ax, s in zip(axes, all_summaries):
        for L in s["all_losses"]:
            ax.plot(L, alpha=0.5, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(s["cfg"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("-ELBO (log)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "losses_by_config.png"), dpi=120)
    plt.close(fig)


def plot_tau_posteriors_strip(summaries_by_cfg):
    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(5 * len(CONFIGS), 9), sharey="row")
    for col, (cfg, title) in enumerate(CONFIGS):
        s = summaries_by_cfg.get(cfg)
        if s is None or s["Tau1_runs"] is None:
            continue
        anno = s["anno"]
        T1 = s["Tau1_runs"]
        T2 = s["Tau2_runs"]
        K = T1.shape[1]
        xs = np.arange(K)

        ax = axes[0, col]
        for r in range(T1.shape[0]):
            ax.scatter(xs + np.random.uniform(-0.15, 0.15, size=K), T1[r], s=14, alpha=0.5, color="C0")
        ax.errorbar(xs, T1.mean(0), yerr=T1.std(0), fmt="_", color="k", capsize=4, lw=2, markersize=15)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(anno, rotation=90, fontsize=8)
        ax.set_title(f"Tau1 - {title}")

        ax = axes[1, col]
        valid = ~np.isnan(T2).all(axis=0)
        t2 = T2[:, valid]
        anno_t2 = [a for a, v in zip(anno, valid) if v]
        xs2 = np.arange(len(anno_t2))
        for r in range(t2.shape[0]):
            ax.scatter(xs2 + np.random.uniform(-0.15, 0.15, size=len(anno_t2)), t2[r], s=14, alpha=0.5, color="C1")
        ax.errorbar(xs2, np.nanmean(t2, 0), yerr=np.nanstd(t2, 0), fmt="_", color="k", capsize=4, lw=2, markersize=15)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(xs2)
        ax.set_xticklabels(anno_t2, rotation=90, fontsize=8)
        ax.set_title(f"Tau2 - {title}")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tau_posterior_strip.png"), dpi=130)
    plt.close(fig)


def threshold_summary(all_summaries):
    rows = []
    for s in all_summaries:
        T = s["T_finals"]
        rows.append(
            dict(
                cfg=s["cfg"],
                T_mean=float(T.mean()),
                T_std=float(T.std()),
                T_min=float(T.min()),
                T_max=float(T.max()),
                prior_mode=1.0 / 21.0,
                stuck_at_prior=bool(abs(T.mean() - 0.05) < 0.01 and T.std() < 0.005),
            )
        )
    return pd.DataFrame(rows)


def save_gate_summary(all_summaries):
    """
    Aggregate gate/filter behavior from lambda_stats_<cfg>.csv files
    written by RUN_LAMBDA=1 diagnostics.
    """
    rows = []
    for s in all_summaries:
        p = os.path.join(OUT_DIR, f"lambda_stats_{s['cfg']}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if df.empty or "gated_frac" not in df.columns:
            continue
        rows.append(
            dict(
                cfg=s["cfg"],
                genes=int(len(df)),
                gated_frac_mean=float(df["gated_frac"].mean()),
                gated_frac_median=float(df["gated_frac"].median()),
                gated_frac_p10=float(df["gated_frac"].quantile(0.10)),
                gated_frac_p90=float(df["gated_frac"].quantile(0.90)),
            )
        )
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "gate_filter_summary.csv"), index=False)


def r2_summary_vs_baselines(all_summaries, baselines):
    rows = []
    for s in all_summaries:
        te = s["te_df"].mean(axis=1)
        te_std = s["te_df"].std(axis=1)
        merged = te.to_frame(name="r2_parmigiano").assign(r2_std=te_std)
        for bl, df in baselines.items():
            merged[f"r2_{bl}"] = df["R2_test"].reindex(te.index)

        merged.to_csv(os.path.join(OUT_DIR, f"per_gene_r2_{s['cfg']}.csv"))

        agg = dict(
            cfg=s["cfg"],
            genes=len(merged),
            parm_mean_r2=float(merged["r2_parmigiano"].mean()),
            parm_median_r2=float(merged["r2_parmigiano"].median()),
            parm_std_across_runs_median=float(merged["r2_std"].median()),
            parm_frac_r2_gt_0p01=float((merged["r2_parmigiano"] > 0.01).mean()),
            parm_frac_r2_gt_0p1=float((merged["r2_parmigiano"] > 0.1).mean()),
        )
        for bl in BASELINES:
            c = f"r2_{bl}"
            if c in merged:
                agg[f"{bl}_mean_r2"] = float(merged[c].mean())
                agg[f"beats_{bl}_frac"] = float((merged["r2_parmigiano"] > merged[c]).mean())
                agg[f"{bl}_frac_r2_gt_0p01"] = float((merged[c] > 0.01).mean())
                agg[f"{bl}_frac_r2_gt_0p1"] = float((merged[c] > 0.1).mean())
        rows.append(agg)
    return pd.DataFrame(rows)


def compute_lin2_exp_lin2():
    import load_data as ld

    any_cfg_path = os.path.join(ROOT, CONFIGS[0][0], "run_1", "config.yaml")
    with open(any_cfg_path) as f:
        cfg = yaml.safe_load(f)
    logger = logging.getLogger()

    Z_df, gene_indices, gene_names, _ = ld.load_annotations_only(cfg)
    Z_df = ld.process_annotations(Z_df, cfg, logger).fillna(0)
    anno = Z_df.columns.tolist()
    Z = Z_df.values.astype(np.float32)

    results = {}
    for cfg_name, title in CONFIGS:
        run_dirs = sorted(glob.glob(os.path.join(ROOT, cfg_name, "run_*")))
        lin2_runs, mod_runs, tau2_runs = [], [], []
        for rd in run_dirs:
            path = os.path.join(rd, "tau_T.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path).dropna(subset=["Tau2"]).set_index("Annotation").reindex(anno)
            if df["Tau2"].isna().any():
                df = df.fillna({"Tau2": 0.0})
            tau2 = df["Tau2"].values.astype(np.float32)
            lin2 = Z @ tau2
            mod = np.exp(np.clip(lin2, -50, 50))
            lin2_runs.append(lin2)
            mod_runs.append(mod)
            tau2_runs.append(tau2)
        if lin2_runs:
            results[cfg_name] = dict(
                title=title,
                lin2=np.vstack(lin2_runs),
                mod=np.vstack(mod_runs),
                tau2=np.vstack(tau2_runs),
                gene_indices=gene_indices,
                anno=anno,
            )
    return results


def plot_lin2_explosion(res):
    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(5 * len(CONFIGS), 8))
    summary_rows = []

    for col, (cfg_name, _) in enumerate(CONFIGS):
        if cfg_name not in res:
            continue
        r = res[cfg_name]
        lin2 = r["lin2"].ravel()
        mod = r["mod"].ravel()

        ax = axes[0, col]
        ax.hist(lin2, bins=120, density=True, alpha=0.7, color="C0")
        ax.set_title(f"{r['title']} lin2=Z*tau2")
        ax.set_xlabel("lin2")

        ax = axes[1, col]
        mpos = mod[mod > 0]
        ax.hist(np.log10(mpos), bins=120, density=True, alpha=0.7, color="C3")
        ax.set_title("log10(exp(lin2))")
        ax.set_xlabel("log10(mod)")

        summary_rows.append(
            dict(
                cfg=cfg_name,
                lin2_min=float(lin2.min()),
                lin2_max=float(lin2.max()),
                lin2_p005=float(np.percentile(lin2, 0.5)),
                lin2_p995=float(np.percentile(lin2, 99.5)),
                mod_min=float(mod.min()),
                mod_max=float(mod.max()),
                mod_p005=float(np.percentile(mod, 0.5)),
                mod_p995=float(np.percentile(mod, 99.5)),
            )
        )

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lin2_exp_explosion.png"), dpi=130)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "lin2_explosion_summary.csv"), index=False)


def compute_lambda_stats(all_summaries, device="cpu"):
    import load_data

    results = []
    for s in all_summaries:
        rd = s["run_dirs"][0]
        with open(os.path.join(rd, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        try:
            torch_device = torch.device(device)
            X, Y, train_idx, _ = load_data.load_residualized_covariates(cfg, torch_device)
            G, Z, _, _ = load_data.load_genes(cfg)

            if not cfg.get("use_brr", True):
                cfg["brr_results_dir"] = None
            if cfg.get("brr_results_dir", None) is not None:
                brr_results = load_data.load_brr_results(cfg)
                brr_betas = brr_results.get("betas", None)
                brr_alphas = brr_results.get("alphas", None)
            else:
                brr_betas = None
                brr_alphas = None

            if cfg.get("train_test", False):
                train_sample_ids = X.iloc[train_idx].index
                X_use = X.iloc[train_idx]
                Y_use = {gene: expr[train_idx] for gene, expr in Y.items()}
                G_use = G.loc[train_sample_ids]
            else:
                X_use, Y_use, G_use = X, Y, G

            data = load_data.DataTensors.from_pandas(
                G_use, Z, X_use, Y_use, brr_betas, brr_alphas, torch_device, cfg
            )
        except Exception as e:
            print(f"[warn] Could not build DataTensors for {s['cfg']}: {e}. Skipping.")
            continue

        tauT = s["tauT_list"][0]
        tau1 = torch.tensor(tauT["Tau1"].values, dtype=torch.float32)
        tau2 = torch.tensor(tauT["Tau2"].dropna().values, dtype=torch.float32)
        T = float(tauT["Filter Threshold"].iloc[0])
        add_intercept = cfg.get("tau1_intercept", False)
        neg_anno = cfg.get("negative_annotations", False)

        gene_stats = []
        for gene in data.gene_names:
            _, Z_g, maf_g = data.get_gene_data(gene)
            Z1 = torch.cat([torch.ones(Z_g.shape[0], 1), Z_g], dim=1) if add_intercept else Z_g
            lin1 = Z1.matmul(tau1).detach().cpu().numpy()
            lin2 = Z_g.matmul(tau2).detach().cpu().numpy()
            mod = np.exp(np.clip(lin2, -30, 30))
            m = maf_g.detach().cpu().numpy()
            if neg_anno:
                keep = np.abs(lin1) >= T
                lam = np.where(keep, lin1 * mod * m, 0.0)
            else:
                lam = np.maximum(0, lin1 - T) * mod * m
                keep = lam > 0

            gene_stats.append(
                dict(
                    gene=gene,
                    n_variants=len(lin1),
                    gated_frac=float(keep.mean()),
                    median_abs_lin1=float(np.median(np.abs(lin1))),
                    p95_abs_lin1=float(np.percentile(np.abs(lin1), 95)),
                    median_mod=float(np.median(mod)),
                    p95_mod=float(np.percentile(mod, 95)),
                    max_mod=float(mod.max()),
                )
            )
        gdf = pd.DataFrame(gene_stats).assign(cfg=s["cfg"])
        gdf.to_csv(os.path.join(OUT_DIR, f"lambda_stats_{s['cfg']}.csv"), index=False)
        results.append(gdf)
    return pd.concat(results) if results else None


def main():
    all_summaries = []
    summaries_by_cfg = {}
    for cfg, _title in CONFIGS:
        print(f"\n=== Summarizing {cfg} ===")
        s = summarize_config(cfg)
        if s is None:
            print("  skipped (no runs found)")
            continue
        all_summaries.append(s)
        summaries_by_cfg[cfg] = s
        print(f"  # runs: {len(s['run_dirs'])}")
        print(f"  final ELBO: mean={s['last_losses'].mean():.3e}  std={s['last_losses'].std():.3e}")
        print(f"  threshold final across runs: mean={s['T_finals'].mean():.4f} std={s['T_finals'].std():.4f}")

    if not all_summaries:
        print("No configurations found. Check ROOT path.")
        return

    plot_losses(all_summaries)
    plot_tau_posteriors_strip(summaries_by_cfg)

    thresh_df = threshold_summary(all_summaries)
    thresh_df.to_csv(os.path.join(OUT_DIR, "threshold_summary.csv"), index=False)

    with open(os.path.join(all_summaries[0]["run_dirs"][0], "config.yaml")) as f:
        cfg0 = yaml.safe_load(f)
    chroms = sorted({g.split("/")[0].replace("chr", "") for g in cfg0["genes"]}, key=lambda x: int(x))
    baselines = load_baselines(chroms)

    r2_df = r2_summary_vs_baselines(all_summaries, baselines)
    r2_df.to_csv(os.path.join(OUT_DIR, "r2_vs_baselines.csv"), index=False)

    if os.environ.get("SKIP_LIN2", "0") != "1":
        res = compute_lin2_exp_lin2()
        plot_lin2_explosion(res)

    if os.environ.get("RUN_LAMBDA", "0") == "1":
        lam = compute_lambda_stats(all_summaries)
        if lam is not None:
            lam.to_csv(os.path.join(OUT_DIR, "lambda_stats_all.csv"), index=False)
            save_gate_summary(all_summaries)

    print(f"\nAll diagnostic outputs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()

