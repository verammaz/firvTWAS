
import numpy as np
import pandas as pd
import os
import time
from dataclasses import dataclass
import torch
import utils

# Get logger
def get_logger():
    return utils.get_logger()


def load_residualized_covariates(config, device, dir=None):
    """
    Load residualized covariates and expression data
    If train_dir and test_dir are provided, load from those directories
    Otherwise, load from original paths in config
    """
    if dir is not None:
        # Loading from train/test split directories
        covariates_path = os.path.join(dir, 'covariates.tsv')
        expression_path = os.path.join(dir, 'tpm.tsv')
    else:
        covariates_path = config['covariates_path']
        expression_path = config['expression_path']
    
    logger = get_logger()
    covariates = pd.read_csv(covariates_path, sep = "\t").set_index("sample_id")
    tpm = pd.read_csv(expression_path, sep = "\t").set_index("feature")
    logger.debug(f"# Genes overall: {len(tpm)}")
    tpm = tpm[covariates.index] # order same as covariates.
    chr_gene = utils.get_chr_gene(tpm, config['genes'])
    tpm = tpm.loc[chr_gene['feature']]
    logger.info(f"# Genes for joint fit: {len(tpm)}")
    
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
    logger.debug(f"Covariates shape: {covariates_scaled.shape}")
    tpm_scaled = utils.scale_tpm_matrix(tpm, median_filter = 0)
    logger.info(f"# Genes after scaling/filtering: {len(tpm_scaled)}")
    logger.debug(f"TPM scaled shape: {tpm_scaled.shape}")
    logger.info(f"Residualizing expression for {len(tpm_scaled)} genes...")
    residualized_Y = {}
    for idx, GENE in enumerate(tpm_scaled.index, 1):
        gene_start = time.time()
        try:
            if idx == 1:
                logger.debug(f"  Starting residualization (device: {device})...")
                logger.debug(f"  Gene {idx}: {GENE}, expr shape: {tpm_scaled.loc[GENE].shape}, covariates shape: {covariates_scaled.shape}")
            expr_series = tpm_scaled.loc[GENE]
            logger.debug(f"  Gene {idx}/{len(tpm_scaled)}: {GENE} - Starting residualization...")
            logger.debug(f"  Gene {idx}: Calling residualize_expression_single_gene...")
            result = utils.residualize_expression_single_gene(expr_series, covariates_scaled, device)
            logger.debug(f"  Gene {idx}: residualize_expression_single_gene completed")
            residualized_Y[GENE.split(".")[0]] = result
            gene_time = time.time() - gene_start
            logger.debug(f"  Gene {idx}/{len(tpm_scaled)}: {GENE} - Completed in {gene_time:.2f}s")
            if idx == 1 or idx % 5 == 0 or idx == len(tpm_scaled):
                logger.debug(f"  Progress: {idx}/{len(tpm_scaled)} genes residualized (last gene: {gene_time:.2f}s)")
        except Exception as e:
            import traceback
            logger.error(f"ERROR residualizing gene {idx} ({GENE}): {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    logger.debug(f"Completed residualization for {len(residualized_Y)} genes")
    return covariates_scaled, residualized_Y



def load_genes(config, genotype_dir=None, annotation_dir=None):
    '''
    Load genotype and annotation matrices
    INPUT:
        - config: configuration dictionary
        - genotype_dir: optional override for genotype directory
        - annotation_dir: optional override for annotation directory
    OUTPUT:
        - G: genotype matrix (N individuals x total variants)
        - Z: annotation matrix (total variants x Q annotations)
    '''
    logger = get_logger() 
    
    # Get sample column name
    sample_col = config.get('sample_col', 'SampleID')
    
    # Use provided directories or fall back to config
    if genotype_dir is None:
        genotype_dir = config.get('genotype_dir')
    if annotation_dir is None:
        annotation_dir = config.get('annotation_dir')
    
    # Option 1: Load pre-generated matrices
    if 'genotype_path' in config and 'annotation_path' in config:
        G = pd.read_csv(config['genotype_path'], sep='\t', index_col=0)
        Z = pd.read_csv(config['annotation_path'], sep='\t', index_col=0)
        
    # Option 2: Load and concatenate individual gene files
    elif genotype_dir is not None and annotation_dir is not None:
        var_ids_ref_alt = []
        var_ids_counted_other = []
        G_list = []
        Z_list = []
        
        for gene in config['genes']:
           
            try:
                # Load genotype file
                g_path = os.path.join(genotype_dir, f"{gene}_genotypes.tsv.gz")
                logger.debug(f"Loading genotype file {g_path}")
                g = pd.read_csv(g_path, sep='\t', index_col=0) 
                g.index.name = 'IID'
                assert g.index.name == 'IID', "IID column not found in genotype file"
                logger.debug(f"Genotype file head:\n{g.head()}")
                
                # Load annotation file
                z_path = os.path.join(annotation_dir, f"{gene}_annotations.tsv.gz")
                logger.debug(f"Loading annotation file {z_path}")
                z = pd.read_csv(z_path, sep='\t', index_col=0) 
                if "chr" == z.index.name: #TODO: some files lost index column --> fix this
                    z = z.reset_index()
                    z.index.name = 'variant_id'
                    z.index = z['chr'] + ":" + z['pos'] 
                # assume variant_id is index --> chr:pos(_A1_A2)
                z.index.name = 'variant_id'
                logger.debug(f"Annotation file head:\n{z.head()}")

                var_ids_ref_alt.extend(z.index.tolist()) # chr:pos_ref_alt
                var_ids_counted_other.extend(g.columns.tolist()) # chr:pos_a1_a2

                # Prefix variant names with gene
                g.columns = [f"{gene}_{col}" for col in g.columns]
                z.index = [f"{gene}_{idx}" for idx in z.index]
                
                G_list.append(g)
                Z_list.append(z)

            except Exception as e:
                logger.warning(f"Error loading gene {gene}: {e}")
                logger.warning(f"Skipping gene {gene}")
                continue
                
        # Concatenate all genes
        G = pd.concat(G_list, axis=1)
        Z = pd.concat(Z_list, axis=0)
    
    else:
        raise ValueError("Config must specify either (genotype_path, annotation_path) "
                        "or (genotype_dir, annotation_dir, gene_list)")
    # Debug: Check variant IDs before processing
    logger.debug(f"\nBefore processing variant IDs:")
    logger.debug(f"  G column example: {G.columns[0] if len(G.columns) > 0 else 'N/A'}")
    logger.debug(f"  Z index example: {Z.index[0] if len(Z.index) > 0 else 'N/A'}")
    if Z.index.duplicated().sum() > 0:
        logger.warning(f"Z has duplicate indices: {Z.index.duplicated().sum()}")
    if G.columns.duplicated().sum() > 0:
        logger.warning(f"G has duplicate columns: {G.columns.duplicated().sum()}")
    
    # chr/GENE_chr:pos(_A1_A2) --> GENE_chr:pos
    G.columns = (G.columns.str.split("_").str[0:2].str.join("_")).str.split("/").str[1]
    Z.index = (Z.index.str.split("_").str[0:2].str.join("_")).str.split("/").str[1]

    logger.debug(f"\nAfter processing variant IDs:")
    logger.info(f"Genotype matrix: {G.shape}, Annotation matrix: {Z.shape}")
    logger.debug(f"  G column example: {G.columns[0] if len(G.columns) > 0 else 'N/A'}")
    logger.debug(f"  Z index example: {Z.index[0] if len(Z.index) > 0 else 'N/A'}")
    logger.debug(f"  Z has duplicate indices: {Z.index.duplicated().sum()}")
    logger.debug(f"  Unique Z indices: {Z.index.nunique()}/{len(Z.index)}")
    logger.debug(f"  Unique G columns: {G.columns.nunique()}/{len(G.columns)}")
    logger.debug(f"  First 10 Z indices: {list(Z.index[0:10])}")
    logger.debug(f"  First 10 G columns: {list(G.columns[0:10])}")

    if Z.index.duplicated().sum() > 0:
        logger.warning(f"Z has duplicate indices: {Z.index.duplicated().sum()}")
    if G.columns.duplicated().sum() > 0:
        logger.warning(f"G has duplicate columns: {G.columns.duplicated().sum()}")

    # Ensure alignment between genotype and annotation
    if set(G.columns) != set(Z.index):
        # print mismatched elements
        g_cols_set = set(G.columns)
        z_idx_set = set(Z.index)
        logger.error(f"Variants in G but not in Z: {g_cols_set - z_idx_set}")
        logger.error(f"Variants in Z but not in G: {z_idx_set - g_cols_set}")
        raise ValueError("Genotype columns must match annotation rows")
    
    Z = Z.loc[G.columns]  # Reorder annotations to match genotype column order
    
    Z = Z.drop(['promoter_3000', 'promoter_2000'], axis=1, errors='ignore')
    
    log_cols = Z.filter(like="log_counts").columns
    Z["chromBPnet"] = Z[log_cols].max(axis=1)
    Z = Z.drop(columns=log_cols)

    enformer = Z.filter(like="TF_delta_min").columns
    Z["TF_delta_min"] = Z[enformer].max(axis=1)
    Z = Z.drop(columns=enformer)

    enformer = Z.filter(like="TF_delta_max").columns
    Z["TF_delta_max"] = Z[enformer].max(axis=1)
    Z = Z.drop(columns=enformer)
    
    # Impute missing values
    G, Z = utils.impute_missing(G, Z)
    if config['scale_anno']:
        Z = utils.scale(Z)
    else:
        Z = Z
    
    logger.info(f"Loaded {G.shape[0]} individuals, {G.shape[1]} variants across genes")
    logger.info(f"Loaded {Z.shape[1]} annotations per variant")
    
    return G, Z, var_ids_counted_other, var_ids_ref_alt


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
        assert set(G.columns) == set(Z.index), "Variant IDs must match between genotype and annotation files"
        
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
