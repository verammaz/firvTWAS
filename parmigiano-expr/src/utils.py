import os
import yaml
import argparse
import time
import logging
import sys
from sklearn.metrics import r2_score, mean_squared_error
import torch
from torch.distributions import Beta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression

# Global logger - will be initialized by setup_logging
logger = None


def setup_logging(level='INFO', log_file=None):
    """
    Setup logging configuration for the parmigiano package.
    INPUT:
        - level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        - log_file: Optional path to log file. If None, logs only to console.
    OUTPUT:
        - logger: Configured logger object
    """
    global logger
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logger = logging.getLogger('parmigiano')
    logger.setLevel(numeric_level)
    logger.handlers = []
    formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger():
    """Get the global logger, creating it if it doesn't exist."""
    global logger
    if logger is None:
        logger = setup_logging()
    return logger


def scale(df):
    return (df - df.min()) / (df.max() - df.min())


def impute_missing(G, Z):
    """
    Impute missing values in genotype and annotation matrices.
    INPUT:
        - G: genotype matrix (N individuals x P variants)
        - Z: annotation matrix (P variants x Q annotations)
    OUTPUT:
        - G: genotype matrix with missing values imputed (variant means)
        - Z: annotation matrix with missing values imputed (zeros)
    """
    logger = get_logger()
    if G.isnull().any().any():
        n_missing_g = G.isnull().sum().sum()
        G = G.fillna(G.mean(axis=0))
        logger.info(f"Imputed {n_missing_g} missing genotype values using variant means")
    if Z.isnull().any().any():
        n_missing_z = Z.isnull().sum().sum()
        Z = Z.fillna(0)
        logger.info(f"Imputed {n_missing_z} missing annotation values with zeros")
    return G, Z


def get_MAF_weights(G, device, b_dist=25):
    maf = torch.round(G).mean(0) / 2
    maf[maf > 0.5] = 1 - maf[maf > 0.5]
    
    # Guard against all-zero MAF (monomorphic variants in this split)
    nonzero_maf = maf[maf > 0]
    if len(nonzero_maf) == 0:
        # All variants monomorphic - return zero weights
        return torch.zeros_like(maf)
    
    maf = torch.clamp(maf, min=nonzero_maf.min(), max=0.5).to(device)
    beta = Beta(
        torch.tensor(1.0, device=device),
        torch.tensor(float(b_dist), device=device),
    )
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
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
    continuous_vars = df.select_dtypes(include='number').columns
    continuous_vars = [c for c in continuous_vars if not c.startswith(tuple(categorical_vars))]
    scaler = StandardScaler()
    df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    df = df.astype(float)
    return df


def scale_tpm_matrix(tpm, median_filter=0, scale_center=True):
    tpm_scaled = tpm[tpm.median(1) > median_filter].copy()
    tpm_scaled = np.log1p(tpm_scaled)
    # No centering or scaling?
    if scale_center:
        row_means = tpm_scaled.mean(axis=1)
        row_stds = tpm_scaled.std(axis=1)
        tpm_scaled = tpm_scaled.sub(row_means, axis=0).div(row_stds, axis=0)
    return tpm_scaled


def residualize_expression_single_gene(expr, covariates, device='cpu', stats=False):
    logger = get_logger()
    common_samples = expr.index.intersection(covariates.index)
    logger.debug(f"Found {len(common_samples)} common samples")
    expr_aligned = expr.loc[common_samples]
    cov_aligned = covariates.loc[common_samples]
    logger.debug("Fitting LinearRegression...")
    lr = LinearRegression(fit_intercept=True, n_jobs=1)
    lr.fit(cov_aligned, expr_aligned)
    logger.debug("LinearRegression.fit() completed")
    pred = lr.predict(cov_aligned)
    resid = expr_aligned - pred
    if stats:
        coef = pd.Series(lr.coef_, index=cov_aligned.columns, name='beta')
        stats_out = {
            'r2': r2_score(expr_aligned, pred),
            'mse': mean_squared_error(expr_aligned, pred),
            'n_samples': len(common_samples),
            'expr_mean': expr_aligned.mean(),
            'expr_std': expr_aligned.std(),
            'resid_std': resid.std()
        }
        return resid, stats_out
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device
    result = torch.as_tensor(resid.values, dtype=torch.float32, device=device_str)
    logger.debug(f"Residuals tensor shape={result.shape}, device={result.device}")
    return result


def get_chr_gene(tpm, genes):
    ens = pd.read_csv("/gpfs/commons/home/adas/58K_data/gencode/genes.bed", sep="\t", header=None)[[0, 3]]
    ens.columns = ['CHR', 'ens']
    temp = pd.DataFrame(tpm.index)
    temp['ens'] = temp['feature'].str.split(".").str[0]
    chr_gene = temp.merge(ens)
    chr_gene['label'] = chr_gene['CHR'] + "/" + chr_gene['ens']
    chr_gene = chr_gene[chr_gene['label'].isin(genes)]
    return chr_gene[['feature', 'CHR']]




def get_MAF(G):
    maf = torch.round(G).mean(0) / 2
    return maf


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def fill_defaults(args, yaml_config=None):
    """
    Fill in missing configuration values with defaults or override with YAML config.
    Supports both joint and per-gene modes.
    """
    defaults = {
        'model': 'parmigiano',
        'epochs': 300,
        'n_posterior': 50,
        'lr': 0.1,
        'output_dir': 'parmigiano_outputs',
        'beta_maf': 25,
        'brr_results_dir': None,
        'diagnosis_col': 'Diagnosis',
        'sample_col': 'SampleID',
        'simulate': False,
        'burden': False,
        'no_wg': False,
        'no_rhog': False,
        'skat': False,
        'chromosome': 'chr21', # not used in joint mode
        'no_filter': False,
        'scale_anno': False, # is this used?
        'train_test': False,
        'log_level': 'INFO',
        'log_file': None,
        'refits': 1,
        'use_brr': True,
        'scale_center': True,
        'tau12': False,
        'tau1_normal_prior': False,
        'tau2_normal_prior': False,
        'chrombpnet_dist_only': False,
        'chrombpnet_dist_only_cfg_num': 1,
    }

    # TODO: match scale_center with brr_results_dir if use_brr is True
    # TODO: add config check -- some flags have to bet set together

    if yaml_config:
        defaults.update(yaml_config)
    for key in defaults:
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            defaults[key] = arg_val
    return defaults


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Parmigiano Package")
    parser.add_argument('--config', type=str, help="Path to the YAML configuration file")
    # Data paths - joint mode uses covariates_path/expression_path; per-gene uses phenotype_path
    parser.add_argument('--phenotype_path', type=str, help="Path to phenotype file (per-gene mode)")
    parser.add_argument('--covariates_path', type=str, help="Path to covariates file (joint mode)")
    parser.add_argument('--expression_path', type=str, help="Path to expression file (joint mode)")
    parser.add_argument('--diagnosis_col', type=str, help="Column name for trait. Default: Diagnosis")
    parser.add_argument('--sample_col', type=str, help="Column name for sample IDs. Default: SampleID")
    parser.add_argument('--genotype_path', type=str, help="Path to genotype matrix")
    parser.add_argument('--annotation_path', type=str, help="Path to annotation matrix")
    parser.add_argument('--genotype_dir', type=str, help="Path to genotypes directory")
    parser.add_argument('--annotation_dir', type=str, help="Path to annotations directory")
    parser.add_argument('--chrombpnet_dist_only', type=str_to_bool, help="Only use chrombpnet and dist_to_tss annotations")
    parser.add_argument('--chrombpnet_dist_only_cfg_num', type=int, help="Configuration number for chrombpnet and dist_to_tss annotations only")
    parser.add_argument('--gene_list', type=str, help="List of genes or path to file with gene list")
    parser.add_argument('--n_posterior', type=int, help="Number of posterior samples")
    parser.add_argument('--output_dir', type=str, help="Path to output directory")
    parser.add_argument('--epochs', type=int, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--beta_maf', type=int, help="Beta parameter for MAF weights")
    parser.add_argument('--use_brr', type=str_to_bool, help="Use Bayesian Ridge Regression results")
    parser.add_argument('--brr_results_dir', type=str, help="Path to Bayesian Ridge Regression results directory")
    parser.add_argument('--scale_center', type=str_to_bool, help="Scale expression matrix by center and scale")
    parser.add_argument('--tau12', type=str_to_bool, help="Use tau1 and tau2 for nonlinear annotation interaction")
    parser.add_argument('--tau1_normal_prior', type=str_to_bool, help="Use normal prior for tau1")
    parser.add_argument('--tau2_normal_prior', type=str_to_bool, help="Use normal prior for tau2")
    # Per-gene specific flags
    parser.add_argument('--simulate', type=str_to_bool, help="Simulate phenotypes")
    parser.add_argument('--burden', type=str_to_bool, help="Remove rho (burden mode - just mean)")
    parser.add_argument('--no_wg', type=str_to_bool, help="Remove w_g gene weight")
    parser.add_argument('--no_rhog', type=str_to_bool, help="Remove rho_g proportion burden versus dispersion")
    parser.add_argument('--skat', type=str_to_bool, help="SKAT mode (just variance)")
    parser.add_argument('--chromosome', type=str, help="Chromosome of focus (per-gene analysis only)")
    parser.add_argument('--tauT_path', type=str, help="Path for tauT weights (per-gene analysis only)")
    parser.add_argument('--no_filter', type=str_to_bool, help="Don't filter variants by parmigiano threshold")
    # Joint mode flags
    parser.add_argument('--scale_anno', type=str_to_bool, help="Scale annotation matrix")
    parser.add_argument('--train_test', type=str_to_bool, help="Split data into train and test sets")
    # Logging
    parser.add_argument('--log_level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Logging level (default: INFO)")
    parser.add_argument('--log_file', type=str, help="Optional path to log file")
    parser.add_argument('--refits', type=int, help="Number of refits")
    parser.add_argument('--use_clip_norm', type=str_to_bool, help="Use clip norm of weights (default: True)")
    parser.add_argument('--clip_norm', type=float, help="Clip norm of weights (default: 10.0)")
    return parser.parse_args()