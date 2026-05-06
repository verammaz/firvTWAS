from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys, yaml
import torch

import load_data
import data_class
from datetime import datetime
import models
import performance


def write_outputs(params, wgs, betas, preds, top50, chrom):
    os.makedirs(os.path.join(params['output'], 'pergene_regression'), exist_ok=True)
    outdir = os.path.join(params['output'], 'pergene_regression')
    wgs.to_csv(os.path.join(outdir, f"chr{chrom}_wgs.csv"))
    betas.to_csv(os.path.join(outdir, f"chr{chrom}_betas.csv"))
    preds.to_csv(os.path.join(outdir, f"chr{chrom}_preds_expr.csv"))
    top50.to_csv(os.path.join(outdir, f"chr{chrom}_top50_genes.csv"))
    return


def run_gruyere(params, chrom):
    # --- Load data ---
    X = load_data.load_covariates(params)
    Y = load_data.load_phenotypes(params)  # expression phenotypes (n x genes)
    Gs, Zs = load_data.load_genotypes_annotations(params, chro=chrom, all_genes=True)
    tau = load_data.load_tau(params)  # pre-fit annotation weights (vector)
    
    wgs = {} # gene weights
    betas = {} # gene-specific genetic predictor weights --> variant effect sizes
    preds = {}

    total_time = 0

    for gene in tqdm(Zs.index.get_level_values("gene").unique()):
        start_time = datetime.now()
        print(f"Fitting model for gene {gene}...", end="", flush=True)
        
        # --- Select data for this gene ---
        gene_mask = Zs.index.get_level_values("gene") == gene
        G = Gs.loc[:, Gs.columns.str.startswith(f"{gene}:")]  # select variants for this gene
        Z = Zs.loc[gene_mask]
        Y_gene = Y[gene]
        data = data_class.PerGene.from_pandas(G, Z, X, Y_gene, params)
        
        # --- Compute gene-specific genetic predictor ---
        beta = (data.Z.T * data.maf_weights).T.matmul(tau)
        Gbeta = data.G['train'] @ beta
        X_input = torch.cat((data.X['train'], Gbeta.reshape(-1, 1)), 1)
        y_train = data.Y['train']

        # --- Convert to numpy arrays ---
        X_np = X_input.detach().cpu().numpy()
        y_np = y_train.detach().cpu().numpy()
       
        # ---- Guards ----
        if not np.isfinite(X_np).all():
            print(f"[SKIP] {gene}: non-finite X")
            continue

        if np.var(X_np[:, -1]) < 1e-8:
            print(f"[SKIP] {gene}: constant genetic predictor")
            continue

        rank = np.linalg.matrix_rank(X_np)
        if rank < X_np.shape[1]:
            print(f"[SKIP] {gene}: rank deficient")
            continue

        n, p = X_np.shape
        if n <= p:
            print(f"[SKIP] {gene}: n <= p")
            continue

   
        # --- Fit linear regression ---
        model = LinearRegression().fit(X_input, y_train)
        y_pred = model.predict(X_input)
        r2 = r2_score(y_train, y_pred)

        # --- Approximate p-value for gene-level effect ---
        # test last coefficient (genetic effect) with simple t-test
        n, p = X_input.shape
        residuals = y_train - y_pred
        s2 = np.sum(residuals.detach().cpu().numpy() ** 2) / (n - p)
        XtX_inv = np.linalg.pinv(X_np.T @ X_np)
        cov = s2 * XtX_inv
        se = np.sqrt(np.diag(cov))
        t_stat = model.coef_[-1] / se[-1]
        pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p))


        beta = (data.Z.T * data.maf_weights).T.matmul(tau) * model.coef_[-1]
        bs = zip(data.variants, beta)
        for variant, beta in bs:
            betas[variant] = {"coef": beta.item()}

        wgs[gene] = {"pval": pval, "s2": s2, "t_stat": t_stat, "coef": model.coef_[-1], "r2": r2}
        preds[gene] = y_pred

        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"Done ({elapsed})")

        total_time += elapsed.total_seconds()

    wgs = pd.DataFrame(wgs).T
    betas = pd.DataFrame(betas).T
    preds = pd.DataFrame(preds)


    print("\n")
    print(f"Total time elapsed: {total_time}")
    
    from statsmodels.stats.multitest import multipletests
    # Drop NaNs for correction
    mask = wgs['pval'].notna()

    wgs.loc[mask, 'fdr_qval'] = multipletests(
        wgs.loc[mask, 'pval'],
        method='fdr_bh'
    )[1]

    # Significant genes (may be empty!)
    significant = wgs[wgs['fdr_qval'] <= 0.05]

    print("Min pval:", wgs['pval'].min())
    print("Min FDR:", wgs['fdr_qval'].min())
    print("Number FDR ≤ 0.05:", (wgs['fdr_qval'] <= 0.05).sum())


    # Always report top-ranked genes
    top50 = wgs.sort_values('pval').head(50)

    write_outputs(params, wgs, betas, preds, top50, chrom)


if __name__ == "__main__":
    params_file = sys.argv[1]
    chrom = sys.argv[2]
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    run_gruyere(params, chrom)
