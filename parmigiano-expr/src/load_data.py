
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
import torch
import utils


def load_residualized_covariates(config, device):
    covariates = pd.read_csv(config['covariates_path'], sep = "\t").set_index("sample_id")
    tpm = pd.read_csv(config['expression_path'], sep = "\t").set_index("feature")
    print("# Genes overall:", len(tpm))
    tpm = tpm[covariates.index] # order same as covariates.
    chr_gene = utils.get_chr_gene(tpm, config['genes'])
    tpm = tpm.loc[chr_gene['feature']]
    print("# Genes for joint fit: ", len(tpm))
    
    covariate_cols = ['biological_sex','eas_prob',
     'afr_prob',
     'amr_prob',
     'sas_prob',
     'eur_prob', 'tissue',
      'age', 'pc1','pc2','pc3','pc4','pc5', 'cohort', 
     'rna_lib_prep_type','rna_strandedness','astrocyte',
     'endothelial_cell',
     'excitatory_neuron',
     'inhibitory_neuron',
     'microglia',
     'oligodendrocyte',
     'oligodendrocyte_progenitor_cell',
     'others',
     'pericyte']
    covariates_scaled = utils.preprocess_covariates(covariates, covariate_cols)
    tpm_scaled = utils.scale_tpm_matrix(tpm, median_filter = 0)
    residualized_Y = {}
    for GENE in tpm_scaled.index:
        residualized_Y[GENE.split(".")[0]] = utils.residualize_expression_single_gene(tpm_scaled.loc[GENE], covariates_scaled, device) # residualize out covariates
    return covariates_scaled, residualized_Y



def load_genes(config):
    '''
    Load genotype and annotation matrices
    INPUT:
        - config: configuration dictionary
            Option 1 (pre-generated matrices):
                - genotype_path: path to N x total_variants matrix
                - annotation_path: path to total_variants x Q matrix
            Option 2 (individual gene files):
                - genotype_dir: directory containing gene-specific genotype files
                - annotation_dir: directory containing gene-specific annotation files
                - gene_list: list of gene names OR path to file with gene names (one per line)
            - sample_col: name of sample ID column (default: 'SampleID')
    OUTPUT:
        - G: genotype matrix (N individuals x total variants)
        - Z: annotation matrix (total variants x Q annotations)
    '''
    # Get sample column name
    sample_col = config.get('sample_col', 'SampleID')
    
    # Option 1: Load pre-generated matrices
    if 'genotype_path' in config and 'annotation_path' in config:
        G = pd.read_csv(config['genotype_path'], sep='\t', index_col=0)
        Z = pd.read_csv(config['annotation_path'], sep='\t', index_col=0)
        
    # Option 2: Load and concatenate individual gene files
    elif 'genotype_dir' in config and 'annotation_dir' in config:
        
        G_list = []
        Z_list = []
        
        for gene in config['genes']:
            # Load genotype file
            g_path = os.path.join(config['genotype_dir'], f"{gene}_genotypes.tsv.gz")
            g = pd.read_csv(g_path, sep='\t', index_col=0)
            
            # Load annotation file
            z_path = os.path.join(config['annotation_dir'], f"{gene}_annotations.tsv.gz")
            z = pd.read_csv(z_path, sep='\t', index_col=0)
            
            # Prefix variant names with gene
            g.columns = [f"{gene}_{col}" for col in g.columns]
            z.index = [f"{gene}_{idx}" for idx in z.index]
            
            G_list.append(g)
            Z_list.append(z)
        
        # Concatenate all genes
        G = pd.concat(G_list, axis=1)
        Z = pd.concat(Z_list, axis=0)
    
    else:
        raise ValueError("Config must specify either (genotype_path, annotation_path) "
                        "or (genotype_dir, annotation_dir, gene_list)")
    G.columns = G.columns.str.split("_").str[0]
    Z.index = Z.index.str.split("_").str[0]
    print(G.shape, Z.shape)
    print(Z.index[0:10])
    print(G.columns[0:10])
    # Ensure alignment between genotype and annotation
    assert set(G.columns) == set(Z.index), "Genotype columns must match annotation rows"
    Z = Z.loc[G.columns]  # Reorder annotations to match genotype column order
    
    # Impute missing values
    G, Z = utils.impute_missing(G, Z)
    Z = utils.scale(Z)
    
    print(f"Loaded {G.shape[0]} individuals, {G.shape[1]} variants across genes")
    print(f"Annotation matrix: {Z.shape[1]} annotations per variant")
    
    return G, Z


@dataclass
class DataTensors: 
    '''
    Container for all input data tensors required by Parmigiano model
    '''
    G: torch.Tensor           # Genotype matrix (N x P)
    Z: torch.Tensor           # Annotation matrix (P x Q)
    X: torch.Tensor           # Covariate matrix (N x K)
    Y: torch.Tensor           # Phenotype vector (N, G)
    maf_weights: torch.Tensor # MAF-based variant weights (P,)
    gene_indices: dict        # Maps gene_name -> (start_idx, end_idx) for columns in G
    gene_names: list          # Ordered list of gene names
    device: torch.device
    num_anno: int = 0         # Number of annotations (Q)
    num_cov: int = 0          # Number of covariates (K)
    num_genes: int = 0        # Number of genes
    
    def __post_init__(self):
        '''Set dimensions after initialization'''
        self.num_anno = self.Z.shape[1]
        self.num_cov = self.X.shape[1]
        self.num_genes = len(self.gene_names)
    
    def get_gene_data(self, gene_name):
        '''
        Extract genotype and annotation data for a specific gene
        INPUT:
            - gene_name: name of gene
        OUTPUT:
            - G_gene: genotype tensor for this gene (N x P_gene)
            - Z_gene: annotation tensor for this gene (P_gene x Q)
            - maf_weights_gene: MAF weights for this gene (P_gene,)
        '''
        start_idx, end_idx = self.gene_indices[gene_name]
        G_gene = self.G[:, start_idx:end_idx]
        Z_gene = self.Z[start_idx:end_idx, :]
        maf_weights_gene = self.maf_weights[start_idx:end_idx]
        return G_gene, Z_gene, maf_weights_gene
        
    @staticmethod
    def from_pandas(G, Z, X, Y, device, config): 
        '''
        Create DataTensors from pandas DataFrames
        INPUT:
            - G: genotype DataFrame (N x P) with columns like "GENE1_var1", "GENE1_var2", "GENE2_var1"
            - Z: annotation DataFrame (P x Q) with index matching G columns
            - X: covariate DataFrame (N x K)
            - Y: phenotype Series (N,)
            - device: torch device (cpu or cuda)
            - config: configuration dictionary with 'beta' parameter for MAF weights
        OUTPUT:
            - DataTensors object with all data as torch tensors
        '''
        # Ensure sample alignment between X/Y and G
        assert set(X.index) == set(G.index), "Sample IDs must match between phenotype and genotype files"
        
        # Reorder G to match X/Y sample order
        G = G.loc[X.index]
        
        # Extract gene names and indices from column names
        gene_indices = {}
        gene_names = []
        current_gene = None
        start_idx = 0
        
        for idx, col in enumerate(G.columns):
            # Assume columns are named "GENE_variantID"
            gene = col.split('_')[0]
            
            if gene != current_gene:
                if current_gene is not None:
                    gene_indices[current_gene] = (start_idx, idx)
                    gene_names.append(current_gene)
                current_gene = gene
                start_idx = idx
        
        # Don't forget the last gene
        if current_gene is not None:
            gene_indices[current_gene] = (start_idx, len(G.columns))
            gene_names.append(current_gene)
        G = torch.as_tensor(G.values, dtype=torch.float32, device=device)
        return DataTensors(
            G = G,
            Z = torch.as_tensor(Z.values, dtype=torch.float32, device=device),
            X = torch.as_tensor(X.values, dtype=torch.float32, device=device),
            Y = Y,
            maf_weights = torch.as_tensor(
                utils.get_MAF_weights(G, device, config['beta']), 
                dtype=torch.float32, 
                device=device
            ),
            gene_indices = gene_indices,
            gene_names = gene_names,
            device = device 
        )
