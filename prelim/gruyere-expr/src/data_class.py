import pandas as pd
import numpy as np
import os, sys
import torch
import utils
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
    
@dataclass
class GenomeWide: 
    G: dict
    Z: torch.Tensor
    X: dict
    gene_indices: torch.Tensor
    genes: pd.Index
    Y: dict
    annotations: pd.Index 
    variants: pd.Index
    covariates: pd.Index
    maf_weights: torch.Tensor
    device: torch.device
    num_genes: int = 0
    num_anno: int = 0
    num_cov: int = 0
    
    
    def __post_init__(self):
        self.num_genes = max(self.gene_indices) + 1
        self.num_anno = self.Z.shape[1]
        self.num_cov = self.X['train'].shape[1]
        
    @staticmethod
    def from_pandas(Gs, Zs, X, Y, params, device = "cpu"): 
        scaler = StandardScaler()
        gene_indices, genes = pd.factorize(Zs.index.get_level_values("gene"))
        covariates = X.columns
        Y = Y[genes] # columns of expression matrix [num_indivs x genes] match order of genes list
        
        # scale expression values between 0-1
        Y_scaled = scaler.fit_transform(Y)

        if params['test_prop'] == 0: # no need for test
            X = {'train': torch.tensor(np.array(X),dtype = torch.float, device = device), 'test': None}
            G = {'train': torch.tensor(np.array(Gs),dtype = torch.float, device = device), 'test': None}
            Y = {'train': torch.tensor(np.array(Y_scaled),dtype = torch.float, device = device), 'test': None}
        else:
            X_train, X_test = train_test_split(X, test_size = params['test_prop'], random_state=RANDOM_STATE)
            G_train, G_test = train_test_split(Gs, test_size = params['test_prop'], random_state=RANDOM_STATE)
            Y_train, Y_test = train_test_split(Y_scaled, test_size = params['test_prop'], random_state=RANDOM_STATE)
            X_train = torch.tensor(np.array(X_train),dtype = torch.float, device = device)
            X_test = torch.tensor(np.array(X_test),dtype = torch.float, device = device)
            G_train = torch.tensor(np.array(G_train),dtype = torch.float, device = device)
            G_test = torch.tensor(np.array(G_test),dtype = torch.float, device = device)
            Y_train = torch.tensor(np.array(Y_train),dtype = torch.float, device = device)
            Y_test = torch.tensor(np.array(Y_test),dtype = torch.float, device = device)
            G = {'train': G_train, 'test': G_test}
            X = {'train': X_train, 'test': X_test}
            Y = {'train': Y_train, 'test': Y_test}

        # scale annotation values between 0-1
        Zs_scaled = utils.minmax_scale_columns(torch.tensor(np.array(Zs), dtype=torch.float, device=device))

        return GenomeWide(
            gene_indices = torch.tensor(gene_indices, dtype = torch.long, device = device), 
            genes = genes,
            Z = Zs_scaled,
            maf_weights = utils.get_weights(Gs), 
            G = G,
            X = X,
            Y = Y,
            annotations = Zs.columns,
            variants = Gs.columns,
            covariates = covariates,
            device = device 
        )
    
    
@dataclass
class PerGene: 
    G: dict
    Z: torch.Tensor
    X: dict
    Y: dict
    annotations: pd.Index 
    variants: pd.Index
    covariates: pd.Index
    maf_weights: torch.Tensor
    device: torch.device
    num_genes: int = 0
    num_anno: int = 0
    num_cov: int = 0

    
    
    def __post_init__(self):
        self.num_anno = self.Z.shape[1]
        self.num_cov = self.X['train'].shape[1]
        self.num_genes = 1
        
    @staticmethod
    def from_pandas(Gs, Zs, X, Y, params, device = "cpu"): 
        covariates = X.columns

        if params['test_prop'] == 0: # If no train test split
            X = {'train': torch.tensor(np.array(X),dtype = torch.float, device = device), 'test': None}
            G = {'train': torch.tensor(np.array(Gs),dtype = torch.float, device = device), 'test': None}
            Y = {'train': torch.tensor(np.array(Y),dtype = torch.float, device = device), 'test': None}
        else:
            X_train, X_test = train_test_split(X, test_size = params['test_prop'], random_state=RANDOM_STATE)
            G_train, G_test = train_test_split(Gs, test_size = params['test_prop'], random_state=RANDOM_STATE)
            Y_train, Y_test = train_test_split(Y, test_size = params['test_prop'], random_state=RANDOM_STATE)
            X_train = torch.tensor(np.array(X_train),dtype = torch.float, device = device)
            X_test = torch.tensor(np.array(X_test),dtype = torch.float, device = device)
            G_train = torch.tensor(np.array(G_train),dtype = torch.float, device = device)
            G_test = torch.tensor(np.array(G_test),dtype = torch.float, device = device)
            Y_train = torch.tensor(np.array(Y_train),dtype = torch.float, device = device)
            Y_test = torch.tensor(np.array(Y_test),dtype = torch.float, device = device)
            G = {'train': G_train, 'test': G_test}
            X = {'train': X_train, 'test': X_test}
            Y = {'train': Y_train, 'test': Y_test}
        
        # scale annotation values between 0-1
        Zs_scaled = utils.minmax_scale_columns(torch.tensor(np.array(Zs), dtype=torch.float, device=device))
        
        return PerGene(
            Z = Zs_scaled,
            maf_weights = utils.get_weights(Gs), 
            G = G,
            X = X,
            Y = Y,
            annotations = Zs.columns,
            variants = Gs.columns,
            covariates = covariates,
            device = device 
        )
    
   