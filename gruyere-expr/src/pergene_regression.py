from sklearn.linear_model import LinearRegression
from scipy.stats import chi2
from scipy.stats import t
from scipy import sparse
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os, sys, yaml
import torch
from datetime import datetime

import load_data
import data_class
import utils



def run_regression(G, X, Y, chrom_to_gene_to_indices, thr=0.05, num_genes=50):
    """
    Run per-gene linear regressions and compute gene-level p-values via LRT.
    
    Parameters
    ----------
    G : scipy.sparse or np.ndarray
        Genotype matrix (n_samples x n_variants)
    X : np.ndarray
        Covariate matrix (n_samples x n_covariates)
    Y : pandas.DataFrame
        Expression matrix (n_samples x n_genes), columns = gene names
    chrom_to_gene_to_indices : dict
        Nested dict: {chrom : {gene : [variant indices]}}
    thr : float
        pvalue significance threshold (default = 0.05)
    num_genes : int
        Number of top significant genes to return (default = 50)

    Returns
    -------
    results : pd.DataFrame
        All gene-level results with p-values
    top_genes : pd.DataFrame
        Top `num_genes` genes among significant ones
    """

    print("Running per-gene regression...")

    pvals = {}
    chroms_list = []
    genes_list = []

    n = X.shape[0]  # number of individuals
    linreg = LinearRegression()
    scaler = StandardScaler()
    
    # scale expression values between 0-1
    Y_scaled = scaler.fit_transform(Y)

    Y_scaled = pd.DataFrame(Y_scaled, index=Y.index, columns=Y.columns)

    for chrom, gene_indices in chrom_to_gene_to_indices.items():
        start_time = datetime.now()
        print(f"\tChromosome {chrom} ... ", end="", flush=True)
        for g, idxs in gene_indices.items():
            Y_g = Y_scaled[g].values  # gene expression vector

            # Extract subset of variant columns for this gene
            G_g = G.iloc[:, idxs].values # G is pandas dataframe
            
            # Skip genes with no variants 
            if G_g.size == 0:
                pvals[g] = np.nan
                chroms_list.append(chrom)
                genes_list.append(g)
                print(f"skipping gene {g} (chrom {chrom}) -- no variants")
                continue

             # Skip genes with constant expression
            if np.all(Y_g == Y_g[0]):
                pvals[g] = np.nan
                chroms_list.append(chrom)
                genes_list.append(g)
                print(f"skipping gene {g} (chrom {chrom}) -- constant expression {Y_g[0]}")
                continue

            # --- Null model: Y ~ X ---
            linreg.fit(X, Y_g)
            rss_null = np.sum((Y_g - linreg.predict(X)) ** 2)

            # --- Alternative model: Y ~ X + sum(G_g) --- 
            # For gene g, does adding nearby genotypes G_g (i.e. gene variants) improve prediction of that gene's expression?
       
            G_g_sum = np.sum(G_g, axis=1, keepdims=True)
            n = Y_g.shape[0]

            # Add intercept
            X_full = np.hstack([np.ones((n,1)), X, G_g_sum])  # shape: n x (1+c+1)
            p = X_full.shape[1]

            # OLS coefficients
            beta_hat = np.linalg.lstsq(X_full, Y_g, rcond=None)[0]  # shape (p,)
            
            # Predicted values and residuals
            y_pred = X_full @ beta_hat
            residuals = Y_g - y_pred
            
            # Residual variance
            sigma2 = np.sum(residuals**2) / (n - p)
            
            # Covariance matrix of coefficients
            XTX_inv = np.linalg.pinv(X_full.T @ X_full)
            se_beta = np.sqrt(np.diag(sigma2 * XTX_inv))
            
            # t-statistic for genetic coefficient (last column)
            t_stat = beta_hat[-1] / se_beta[-1]
            pval = 2 * t.sf(np.abs(t_stat), df=n - p)

            pvals[g] = pval
            chroms_list.append(chrom)
            genes_list.append(g)
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"Done ({elapsed})")

    # Convert results to DataFrame
    results = pd.DataFrame({
        'chrom': chroms_list,
        'gene': genes_list,
        'pval': [pvals[g] for g in genes_list]
    })

    # Drop missing and non-finite p-values
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna(subset=['pval'])


    results_sorted = results.sort_values('pval', ascending=True)
    sig_results = results_sorted[results_sorted['pval'] < thr]

    top_genes = sig_results.head(num_genes)

    print(f"    Found {len(sig_results)} significant genes out of {len(results)} total. Taking top {min(num_genes, len(top_genes))}.\n")
    return results, top_genes[['chrom', 'gene']]



def plot(results, outdir, chro):
    plt.figure(figsize=(10,5))
    plt.scatter(range(len(results)), -np.log10(results['pval']), s=8)
    plt.axhline(-np.log10(0.05 / len(results)), color='red', linestyle='--')  # Bonferroni
    plt.ylabel('-log10(p-value)')
    plt.xlabel('Genes')
    plt.xticks([])
    plt.title(f'Chromosome {chro} Gene-level association p-values')
    plt.savefig(os.path.join(outdir, f'chr{chro}_gene_association.png'))


if __name__ == "__main__":
    params_file = sys.argv[1]
    chro = sys.argv[2]
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)

    num_genes = params.get('num_genes', 50) 
    start_time = datetime.now()
    print("Loading data...", end="", flush=True)

    Y = load_data.load_phenotypes(params)
    X = load_data.load_covariates(params)

    G, chro_to_gene_to_indices = load_data.load_genotypes(params, chro=chro)
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Done ({elapsed})\n")

    results, top_genes = run_regression(G, X, Y, chro_to_gene_to_indices, num_genes=num_genes)

    outdir = os.path.join(params['output'], 'gene_association')
    os.makedirs(outdir, exist_ok=True)
    plot(results, outdir, chro)
    results.to_csv(os.path.join(outdir, f'chr{chro}_gene_association_results.csv'), index=False)
    # top_genes.to_csv(os.path.join(params['output'], f'chr{chro}_top_genes.txt'), sep='\t', index=False, header=False)
    