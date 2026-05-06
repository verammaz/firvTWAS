import pandas as pd
import numpy as np
import os, sys
import torch
from tqdm import tqdm
import pyro.distributions as dist
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler


def minmax_scale_columns(X, skip_cols=None):
    if skip_cols is None:
        skip_cols = []

    # Columns to scale
    cols = [i for i in range(X.shape[1]) if i not in skip_cols]

    # Extract only columns to scale
    X_to_scale = X[:, cols]

    # Compute min-max on the selected columns
    col_min = X_to_scale.min(0, keepdim=True).values
    col_max = X_to_scale.max(0, keepdim=True).values
    denom = col_max - col_min
    denom[denom == 0] = 1

    # Scale them
    X_scaled_part = (X_to_scale - col_min) / denom

    # Reassemble full matrix
    X_scaled = X.clone()
    X_scaled[:, cols] = X_scaled_part

    return X_scaled


def scale(df):
    # function that min-max scales a dataframe 
    return (df-df.min())/ (df.max() - df.min())

def get_weights(G):
    '''
    Set variant weights based on MAF and from Beta(1,25) distribution
    This should be run before imputation     
    '''
    d = dist.Beta(1,25)
    maf = torch.tensor(G.mean(0)/2, dtype = torch.float) 
    maf[maf>0.5] = 1 - maf[maf>0.5] # this shouldn't change anything, just checking that AF is the correct direction
    maf[maf>0.05] = 0.05 # in case of any leaks
    weights = torch.exp(d.log_prob(maf)) 
    return weights



def perform_anno_nmf(anno_df, params):
    """
    Group related annotations using NMF.
    
    Parameters
    ----------
    anno_df : pd.DataFrame
        Rows = variants, columns = annotations
    params : dict
        anno_nmf : bool
        anno_groups : dict[str, list[str]]
        anno_nmf_k : int (default=2)
        anno_nmf_keep_original : bool (default=False)
    """
    anno_nmf = params.get('anno_nmf', False)
    if not anno_nmf:
        return anno_df

    GROUPINGS = {
        'conservation': [
            'MAP20', 'phyloP17way_primate', 'phyloP30way_mammalian',
            'phastCons30way_mammalian', 'phastCons17way_primate_rankscore',
            'integrated_fitCons_score', 'H1-hESC_fitCons_score',
            'bStatistic', 'GERP_RS'
        ],
        'roadmap': [
            'Roadmap_E074_GenoSkyline_Plus_score',
            'Roadmap_E068_GenoSkyline_Plus_score',
            'Roadmap_E069_GenoSkyline_Plus_score',
            'Roadmap_E072_GenoSkyline_Plus_score',
            'Roadmap_E067_GenoSkyline_Plus_score',
            'Roadmap_E073_GenoSkyline_Plus_score',
            'Roadmap_E070_GenoSkyline_Plus_score',
            'Roadmap_E030_GenoSkyline_Plus_score',
            'Roadmap_E050_GenoSkyline_Plus_score',
            'Roadmap_E051_GenoSkyline_Plus_score',
            'Roadmap_E124_GenoSkyline_Plus_score'
        ],
        'pathogenicity': [
            'funseq2_noncoding_score', 'fathmm-MKL_non-coding_score',
            'fathmm-MKL_coding_score', 'fathmm-XF_score',
            'CADD_raw', 'CADD_phred', 'DANN_score',
            'Eigen-raw', 'Eigen-PC-raw'
        ],
        'maf': [
            'gnomAD_genomes_POPMAX_AF',
            'gnomAD_genomes_AFR_AF',
            'gnomAD_genomes_AMR_AF',
            'gnomAD_genomes_NFE_AF'
        ],
    }

    groupings = params.get("anno_groups", GROUPINGS)
    k_default = params.get("anno_nmf_k", 2)
    random_state = 0

    new_features = []
    cols_to_drop = []

    for group_name, cols in groupings.items():
        cols = [c for c in cols if c in anno_df.columns]
        if len(cols) < 2:
            continue

        X = anno_df[cols].values

        # --- scale within group ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- shift to non-negative ---
        X_scaled -= X_scaled.min()
        if np.allclose(X_scaled, 0):
            continue

        k = min(k_default, X_scaled.shape[1])

        nmf = NMF(
            n_components=k,
            init="nndsvda",
            max_iter=1000,
            random_state=random_state
        )

        W = nmf.fit_transform(X_scaled)  # variants × components
        H = nmf.components_              # components × annotations

        # --- store new grouped features ---
        for i in range(k):
            colname = f"{group_name}_nmf{i+1}"
            new_features.append(pd.Series(W[:, i], index=anno_df.index, name=colname))

        cols_to_drop.extend(cols)

        
    # --- assemble final dataframe ---
    anno_out = anno_df.drop(columns=cols_to_drop, errors="ignore")
    if new_features:
        anno_out = pd.concat([anno_out] + new_features, axis=1)

    return anno_out

