import os
import yaml
import argparse
import pyro
import pyro.distributions as dist
from sklearn.metrics import r2_score, mean_squared_error
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression

def scale(df):
    return (df-df.min())/ (df.max() - df.min())

def impute_missing(G, Z):
    '''
    Impute missing values in genotype and annotation matrices
    INPUT:
        - G: genotype matrix (N individuals x P variants)
        - Z: annotation matrix (P variants x Q annotations)
    OUTPUT:
        - G: genotype matrix with missing values imputed (variant means)
        - Z: annotation matrix with missing values imputed (zeros)
    '''
    # Impute genotypes with variant mean (column-wise)
    if G.isnull().any().any():
        n_missing_g = G.isnull().sum().sum()
        G = G.fillna(G.mean(axis=0))
        print(f"Imputed {n_missing_g} missing genotype values using variant means")
    
    if Z.isnull().any().any(): # Fill annotations with zeros
        n_missing_z = Z.isnull().sum().sum()
        Z = Z.fillna(0)
        print(f"Imputed {n_missing_z} missing annotation values with zeros")
    return G, Z
    
def get_MAF_weights(G, device, b_dist = 25):
    '''
    Set variant weights based on MAF and from Beta(1,b_dist) distribution
    This should be run before imputation     
    '''
    maf = torch.round(G).mean(0) / 2
    maf[maf>0.5] = 1 - maf[maf>0.5] # this shouldn't change anything, just checking that AF is the correct direction
    maf = torch.clamp(maf, min=maf[maf > 0].min(), max=0.5).to(device)
    beta = dist.Beta(torch.tensor(1.0, device=device), torch.tensor(float(b_dist), device=device))
    return torch.exp(beta.log_prob(maf))



def preprocess_covariates(covariates, covariate_cols):    
    df = covariates[covariate_cols].copy()
    if 'age' in df.columns:
        global_mean = df['age'].mean()
        cohort_means = df.groupby('cohort')['age'].mean()
        df['age'] = df.apply(
            lambda row: (
                cohort_means[row['cohort']]
                if pd.notna(cohort_means[row['cohort']])
                else global_mean
            ) if pd.isna(row['age']) else row['age'],
            axis=1
        )
    categorical_vars = ['biological_sex', 'tissue', 'cohort', 'rna_lib_prep_type', 'rna_strandedness']
    categorical_vars = [c for c in categorical_vars if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_vars,drop_first=True) # one hot encode
    continuous_vars = df.select_dtypes(include='number').columns
    continuous_vars = [c for c in continuous_vars if not c.startswith(tuple(categorical_vars))]
    scaler = StandardScaler()
    df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    df = df.astype(float)
    return df


def residualize_expression_single_gene(expr, covariates):
    common_samples = expr.index.intersection(covariates.index)
    expr_aligned = expr.loc[common_samples]
    cov_aligned = covariates.loc[common_samples]
    lr = LinearRegression(fit_intercept=True) # Fit linear regression
    lr.fit(cov_aligned, expr_aligned) 
    pred = lr.predict(cov_aligned) # Predict and compute residuals
    resid = expr_aligned - pred
    coef = pd.Series(
        lr.coef_,
        index=cov_aligned.columns,
        name='beta'
    )
    stats = {
        'r2': r2_score(expr_aligned, pred),
        'mse': mean_squared_error(expr_aligned, pred),
        'n_samples': len(common_samples),
        'expr_mean': expr_aligned.mean(),
        'expr_std': expr_aligned.std(),
        'resid_std': resid.std()
    }
    return pd.Series(resid, index=common_samples), coef, stats
    
def scale_tpm_matrix(tpm, median_filter = 0):
    tpm_scaled = tpm[tpm.median(1) > median_filter].copy() # remove lowly expressed genes
    tpm_scaled = np.log1p(tpm_scaled)
    row_means = tpm_scaled.mean(axis=1)
    row_stds = tpm_scaled.std(axis=1)
    tpm_scaled = tpm_scaled.sub(row_means, axis=0).div(row_stds, axis=0)
    return tpm_scaled


def residualize_expression_single_gene(expr, covariates, device = 'cpu', stats = False):
    common_samples = expr.index.intersection(covariates.index)
    expr_aligned = expr.loc[common_samples]
    cov_aligned = covariates.loc[common_samples]
    lr = LinearRegression(fit_intercept=True) # Fit linear regression
    lr.fit(cov_aligned, expr_aligned) 
    pred = lr.predict(cov_aligned) # Predict and compute residuals
    resid = expr_aligned - pred
    if stats:
        coef = pd.Series(
            lr.coef_,
            index=cov_aligned.columns,
            name='beta'
        )
        stats = {
            'r2': r2_score(expr_aligned, pred),
            'mse': mean_squared_error(expr_aligned, pred),
            'n_samples': len(common_samples),
            'expr_mean': expr_aligned.mean(),
            'expr_std': expr_aligned.std(),
            'resid_std': resid.std()
        }
        return resid, stats
    return torch.as_tensor(resid.values, dtype=torch.float32, device=device)


def get_chr_gene(tpm, genes):
    ens = pd.read_csv("/gpfs/commons/home/adas/58K_data/gencode/genes.bed", sep = "\t", header = None)[[0,3]]
    ens.columns = ['CHR','ens']
    temp = pd.DataFrame(tpm.index)
    temp['ens'] = temp['feature'].str.split(".").str[0]
    chr_gene = temp.merge(ens)    
    chr_gene['label'] = chr_gene['CHR'] + "/" + chr_gene['ens']
    chr_gene = chr_gene[chr_gene['label'].isin(genes)]  
    return chr_gene[['feature','CHR']]

def get_MAF(G):
    maf = torch.round(G).mean(0) / 2
    return maf

def load_yaml(yaml_path):
    """
    Loads the YAML configuration file from the provided path.
    """
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def fill_defaults(args, yaml_config=None):
    """
    Fill in missing configuration values with defaults or override with YAML config.
    """
    defaults = {
        'model': 'parmigiano',
        'epochs': 300,
        'n_posterior': 50,
        'lr': 0.1,
        'output_dir': 'parmigiano_outputs',
        'beta': 25,
        'diagnosis_col': 'Diagnosis',
        'sample_col': 'SampleID',
    }

    # Override defaults with YAML config if provided
    if yaml_config:
        defaults.update(yaml_config)

    # Override with CLI args if not None
    for key in defaults:
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            defaults[key] = arg_val

    # Required values that are not in the base defaults
    required_keys = ['covariates_path', 'expression_path']
    for key in required_keys:
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            defaults[key] = arg_val
        elif key not in defaults or defaults[key] is None:
            raise ValueError(f"You must provide --{key} argument or specify it in the YAML.")

    return defaults


def parse_args():
    """
    Parses the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Configuration for Parmigiano Package")
    parser.add_argument('--config', type=str, help="Path to the YAML configuration file")
    parser.add_argument('--covariates_path', type=str, help="Path to covariates file")
    parser.add_argument('--expression_path', type=str, help="Path to expression file")
    parser.add_argument('--diagnosis_col', type=str, help="Column name for trait. Default: Diagnosis")
    parser.add_argument('--sample_col', type=str, help="Column name for sample IDs. Default: SampleID")
    parser.add_argument('--genotype_path', type=str, help="Path to genotype matrix")
    parser.add_argument('--annotation_path', type=str, help="Path to annotation matrix")
    parser.add_argument('--genotype_dir', type=str, help="Path to genotypes directory")
    parser.add_argument('--annotation_dir', type=str, help="Path to annotations directory")
    parser.add_argument('--gene_list', type=str, help="List of genes or path to file with gene list")
    parser.add_argument('--n_posterior', type=int, help="Number of posterior samples")
    parser.add_argument('--output_dir', type=str, help="Path to output directory")
    parser.add_argument('--epochs', type=int, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--beta', type=int, help="Beta parameter for MAF weights")
    return parser.parse_args()