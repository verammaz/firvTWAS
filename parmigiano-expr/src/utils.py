import os
import yaml
import argparse
import time
import logging
import sys
import pyro
import pyro.distributions as dist
from sklearn.metrics import r2_score, mean_squared_error
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression

# Global logger - will be initialized by setup_logging
logger = None

def setup_logging(level='INFO', log_file=None):
    """
    Setup logging configuration for the parmigiano-expr package
    
    INPUT:
        - level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        - log_file: Optional path to log file. If None, logs only to console.
    
    OUTPUT:
        - logger: Configured logger object
    """
    global logger
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    logger = logging.getLogger('parmigiano')
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )
    formatter = logging.Formatter('%(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger():
    """Get the global logger, creating it if it doesn't exist"""
    global logger
    if logger is None:
        logger = setup_logging()
    return logger


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
    logger = get_logger()
    # Impute genotypes with variant mean (column-wise)
    if G.isnull().any().any():
        n_missing_g = G.isnull().sum().sum()
        G = G.fillna(G.mean(axis=0))
        logger.info(f"Imputed {n_missing_g} missing genotype values using variant means")
    
    if Z.isnull().any().any(): # Fill annotations with zeros
        n_missing_z = Z.isnull().sum().sum()
        Z = Z.fillna(0)
        logger.info(f"Imputed {n_missing_z} missing annotation values with zeros")
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


def scale_tpm_matrix(tpm, median_filter = 0):
    tpm_scaled = tpm[tpm.median(1) > median_filter].copy() # remove lowly expressed genes
    tpm_scaled = np.log1p(tpm_scaled)
    # row_means = tpm_scaled.mean(axis=1)
    # row_stds = tpm_scaled.std(axis=1)
    # tpm_scaled = tpm_scaled.sub(row_means, axis=0).div(row_stds, axis=0)
    return tpm_scaled


def residualize_expression_single_gene(expr, covariates, device = 'cpu', stats = False):
    logger = get_logger()
    
    common_samples = expr.index.intersection(covariates.index)
    logger.debug(f"Found {len(common_samples)} common samples")
    
    expr_aligned = expr.loc[common_samples]
    cov_aligned = covariates.loc[common_samples]
    logger.debug(f"Aligned shapes: expr={expr_aligned.shape}, cov={cov_aligned.shape}")

    # Use simple LinearRegression like the original version - pass DataFrames directly
    # This matches what worked in run_parmigiano.py
    logger.debug("Fitting LinearRegression...")
    lr = LinearRegression(fit_intercept=True, n_jobs=1)
    lr.fit(cov_aligned, expr_aligned) 
    logger.debug("LinearRegression.fit() completed")
    
    logger.debug("Predicting...")
    pred = lr.predict(cov_aligned) # Predict and compute residuals
    logger.debug("LinearRegression.predict() completed")
    
    logger.debug("Computing residuals...")
    resid = expr_aligned - pred
    logger.debug("Residuals computed")
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
    
    # Convert device to string if it's a torch.device object
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device
    
    logger.debug(f"Converting residuals to tensor on device={device_str}...")
    result = torch.as_tensor(resid.values, dtype=torch.float32, device=device_str)
    logger.debug(f"Tensor conversion completed, result shape={result.shape}, device={result.device}")
   
    return result


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
        'scale_anno': True,
        'train_test': False,
        'log_level': 'INFO',  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        'log_file': None  # Optional log file path
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


def str_to_bool(v):
    """
    Convert string to boolean. Handles common string representations.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


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
    parser.add_argument('--scale_anno', type=str_to_bool, help="Scale annotation matrix")
    parser.add_argument('--train_test', type=str_to_bool, help="Split data into train and test sets")
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       help="Logging level (default: INFO)")
    parser.add_argument('--log_file', type=str, help="Optional path to log file")

    return parser.parse_args()