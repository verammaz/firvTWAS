import sys
import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from get_joint_analysis_genes import read_pergene_regression_outputs


def plot_taus(params):
    taus_file = os.path.join(params['output'], 'joint_model', 'taus.csv')
    taus_df = pd.read_csv(taus_file)
    taus_df['epoch'] = range(len(taus_df))

    # Define groups
    groups = {
        "Splicing": [c for c in taus_df.columns if c.startswith("SpliceAI_")] + ["splice"],
        "Conservation": ["phyloP17way_primate","phyloP30way_mammalian",
                         "phastCons17way_primate_rankscore","phastCons30way_mammalian","GERP_RS","MAP20"],
        "Functional": ["bStatistic","integrated_fitCons_score","H1-hESC_fitCons_score",
                       "funseq2_noncoding_score","CADD_raw","CADD_phred","DANN_score",
                       "fathmm-MKL_non-coding_score","fathmm-MKL_coding_score","fathmm-XF_score",
                       "Eigen-raw","Eigen-PC-raw"],
        "Regulatory": ["EnhancerFinder_brain_enhancer","FANTOM5_CAGE_peak_robust"],
        "Roadmap": [c for c in taus_df.columns if c.startswith("Roadmap_")],
        "Frequency": [c for c in taus_df.columns if c.startswith("gnomAD_genomes_")],
        "Distance_other": ["dist_to_TSS","intercept"],
    }

    plots_dir = os.path.join(params['output'], 'joint_model', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    def plot_group(cols, group_name):
        if not cols: return
        plt.figure(figsize=(10, 6))
        for c in cols:
            plt.plot(taus_df.index, taus_df[c], label=c, alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Tau value")
        plt.title(f"Tau trajectories — {group_name}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"tau_{group_name}.png"), dpi=200)
        plt.close()

    # Generate plots
    for group_name, cols in groups.items():
        plot_group(cols, group_name)

    print(f"Saved grouped tau plots in: {plots_dir}")



def plot_losses(params):
    losses_file = os.path.join(params['output'], 'joint_model', 'losses.txt') 
    losses = pd.read_csv(losses_file, header=None)[0]
    plt.figure(figsize=(8, 5))
    plt.plot(losses.index, losses.values, color='black', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Across Epochs")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plots_dir = os.path.join(params['output'], 'joint_model', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
    plt.close()
    print("Saved loss plot to:", os.path.join(plots_dir, 'loss_curve.png'))


def load_fit_time(run):
    path = os.path.join(run, "outputs", "joint_model", "fit_time.txt")
    with open(path, "r") as f:
        n_genes_str, fit_time_str = f.readline().split()

    # convert to numbers
    n_genes = int(n_genes_str)
    fit_time = float(fit_time_str)

    return n_genes, fit_time


def plot_fit_times(root="tau_runs"):
    runs = sorted(
        [os.path.join(root, d) for d in os.listdir(root) if d.startswith("joint_model_run")]
    )

    fit_times = []
    for r in runs:
        n_genes, fit_time = load_fit_time(r)
        fit_times.append((n_genes, fit_time))

    time_df = pd.DataFrame(fit_times, columns=["num_genes", "fit_time_seconds"])
    time_df = time_df.sort_values("num_genes")

    # ---- Plot ----
    plt.figure(figsize=(8,5))
    time_df.boxplot(column="fit_time_seconds", by="num_genes")
    plt.xlabel("Number of genes")
    plt.ylabel("Fit time (seconds)")
    plt.title("Fit time distribution by number of genes")
    plt.suptitle("")  # remove pandas boxplot subtitle
    plt.tight_layout()

    # Save figure in output folder
    plt.savefig(os.path.join(root, "fit_times.png"))
    plt.close()

    print(f"Saved fit time bar chart.")


def plot_pergene_regression_pvals(params):
    # Extract p-values
    outdir = os.path.join(params['output'], 'gene_association')
    results = read_pergene_regression_outputs(outdir)
    pvals = results["pval"].dropna().values

    # --- 1) Histogram of p-values (raw) ---
    plt.figure(figsize=(7, 5))
    plt.hist(pvals, bins=50, edgecolor='black')
    plt.xlabel("p-value")
    plt.ylabel("Count")
    plt.title("Distribution of Gene p-values (All Chromosomes)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pvalue_histogram.png"))
    plt.close()

    # --- 2) Histogram of -log10(p) ---
    plt.figure(figsize=(7, 5))
    plt.hist(-np.log10(pvals), bins=50, edgecolor='black')
    plt.xlabel("-log10(p-value)")
    plt.ylabel("Count")
    plt.title("Distribution of -log10 Gene p-values (All Chromosomes)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pvalue_neglog10_histogram.png"))
    plt.close()

    # --- 3) ECDF plot ---
    sorted_p = np.sort(pvals)
    ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)

    plt.figure(figsize=(7, 5))
    plt.plot(sorted_p, ecdf, linewidth=2)
    plt.xlabel("p-value")
    plt.ylabel("ECDF")
    plt.title("ECDF of Gene p-values (All Chromosomes)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pvalue_ecdf.png"))
    plt.close()

    print("Saved:")
    print(" - pvalue_histogram.png")
    print(" - pvalue_neglog10_histogram.png")
    print(" - pvalue_ecdf.png")


def load_final_tau(run):
    taus = os.path.join(run, "outputs", "joint_model", "taus.csv")
    if not os.path.exists(taus):
        return None
    df = pd.read_csv(taus)
    return df.iloc[-1]


def plot_aggregate_taus(root="tau_runs"):
    runs = sorted(
        [os.path.join(root, d) for d in os.listdir(root) if d.startswith("joint_model_run")]
    )

    final_taus = []
    for r in runs:
        last = load_final_tau(r)
        if last is not None:
            last.name = r.split("_")[-1]  # run index
            final_taus.append(last)

    tau_df = pd.DataFrame(final_taus)

 
    # ----- Bar plot for run 0 ----- (all 100 top genes)
    tau_df.iloc[0].plot(kind="bar", figsize=(14,5))
    plt.title("Final tau for full top gene set (run 0)")
    plt.tight_layout()
    plt.savefig(os.path.join(root, "tau_run0.png"))
    plt.close()

    # ----- Bar plot for run 1 ----- (top 50 genes)
    tau_df.iloc[1].plot(kind="bar", figsize=(14,5))
    plt.title("Final tau for top gene set (run 1)")
    plt.tight_layout()
    plt.savefig(os.path.join(root, "tau_run1.png"))
    plt.close()

    # Create a new DataFrame with the two runs
    combined = pd.DataFrame({
        "top_100 (run0)": tau_df.iloc[0],
        "top_50 (run1)": tau_df.iloc[1]
    })

    plt.figure(figsize=(14, 6))

    combined.plot(kind="bar", figsize=(14, 6))

    plt.title("Final tau comparison: 100-gene run vs 50-gene run")
    plt.xlabel("Annotation")
    plt.ylabel("Tau value")
    plt.legend(["Full top 100 genes (run 0)", "Top 50 genes (run 1)"])
    plt.tight_layout()

    plt.savefig(os.path.join(root, "tau_runs01.png"))
    plt.close()


    # ----- Boxplot of random subsets -----
    plt.figure(figsize=(14,10))
    tau_df.boxplot()
    plt.title("Tau variability across subsets")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(root, "tau_runs_boxplot.png"))
    plt.close()

    print("Saved aggregate plots.")



def main():
    if len(sys.argv) == 1:
        plot_aggregate_taus()
        plot_fit_times()
        return 

    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)  
    
    plot_type = sys.argv[2]

    if plot_type == 'taus':
        plot_taus(params)

    
    elif plot_type == 'losses':
        plot_losses(params)
    
    elif plot_type == 'pvals':
        plot_pergene_regression_pvals(params)

if __name__ == "__main__":
    main()
