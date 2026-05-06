import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import utils


# Note: assumes data is split by chromosome --> per gene files

def load_phenotypes(params):
    # Note: specific to how i currently have tpm matrix saved --> TODO: standardize
    """
    Load phenotypes.
    INPUT:
        - params: input yaml file loaded as params dict.
    OUTPUT:
        - Y: phenotype (gene expression) matrix [individuals x genes] as float numpy array
    """
    samples_list = load_sample_ids(params)

    expr = pd.read_csv(params['Y'], sep="\t", index_col=0)
    expr['feature'] = expr['feature'].str.replace(r'\.\d+', '', regex=True)
    expr = expr.T.reset_index(drop=False)
    expr.columns = expr.iloc[0, :]
    expr = expr[1:]
    expr = expr.rename(columns={'feature': 'sample_id'}).copy()
    expr = expr.set_index('sample_id', drop=True)
    expr = expr.reindex(samples_list)
    expr = expr.reset_index()
    expr = expr.drop(columns=['sample_id'])
    Y = expr.apply(pd.to_numeric, errors='coerce')
    Y = Y.fillna(0) # check 

    return Y


def load_sample_ids(params):
    """
    Load sample ids to ensure ordering consistency across
        Y (phenotypes), X (covariates), and G (genotypes) matrices.
    INPUT:
        - params: input yaml file laoded as params dict.
    OUTPUT:
        - list of sample ids (rows of per-gene genotype matrices are ordered this way)"""

    samples = pd.read_csv(params['samples'], sep="\t", header=None)
    return samples[1].to_list()




def load_covariates(params):
    '''
    Load covariates.
    INPUT:
        - params: input yaml file loaded as params dict.
    OUTPUT:
        -X: covariates dataframe [individuals x covariates]
    '''
    samples_list = load_sample_ids(params)
    cov = pd.read_csv(params['X'], sep="\t", header=None)
    cov = cov.T
    cov.columns = cov.iloc[0]             
    cov = cov[1:]              
    cov = cov.set_index('ID', drop=True)
    cov = cov.reindex(samples_list)
    cov['intercept'] = 1.0
    cov = cov.reset_index()
    cov = cov.drop(columns=['ID'])
    X = cov.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    return X


def get_genes(params):
    '''
    Load genes to be used.
    INPUT:
        - params: input yaml file loaded as params dict.
    OUTPUT:
        - genes: dict {chrom: [gene1, gene2, ...] (ensgene ids)} to use.
    '''
    if 'genes' not in params.keys():
        return None
    chrom_genes = defaultdict(list)
    with open(params['genes'], 'r') as f:
        for line in f.readlines():
            chrom, gene = line.split() # lines: 1 ENS
            chrom_genes[chrom].append(gene)
    if len(chrom_genes) == 0 :
        return None
    return chrom_genes


def collect_gene_files(params):
    chrom_dict = {}

    for chrom_dir in os.listdir(params['G']):
        chrom_path = os.path.join(params['G'], chrom_dir)
        if os.path.isdir(chrom_path):
            genes = []
            for filename in os.listdir(chrom_path):
                if filename.endswith("_genotypes.pt"):
                    gene = filename.replace("_genotypes.pt", "")
                    if os.path.exists(os.path.join(params['Z'], chrom_dir, f"{gene}_annotations.tsv.gz")):
                        genes.append(gene)
            chrom_dict[chrom_dir.replace("chr", "")] = genes

    return chrom_dict


def load_genotypes(params, chro=None):
    chrom_to_genes = collect_gene_files(params)
    chrom_to_gene_to_variant_indices = defaultdict(dict)
    
    Gs = []

    for chrom, chrom_genes in chrom_to_genes.items():
        if chro != None and chro != chrom:
            continue
        chrom_to_gene_to_variant_indices[chrom] = defaultdict(list)
        for gene in chrom_genes:
            genotype_file = os.path.join(params['G'], f'chr{chrom}', f'{gene}_genotypes.pt')
            geno_ts = torch.load(genotype_file, map_location='cpu')
            if geno_ts.is_sparse:
                geno_ts = geno_ts.to_dense()

            variant_file = os.path.join(params['G'], f'chr{chrom}', f'{gene}_variant_ids.txt')
            with open(variant_file, 'r') as f:
                variant_ids = [line.strip() for line in f if line.strip()]
            gene_variant_ids = [f"{gene}:{var}" for var in variant_ids]
            geno_df = pd.DataFrame(geno_ts.cpu().numpy(), columns=gene_variant_ids)
            
            # record column index range for this gene
            start_idx = sum(df.shape[1] for df in Gs)
            end_idx = start_idx + len(gene_variant_ids)
            chrom_to_gene_to_variant_indices[chrom][gene].extend(range(start_idx, end_idx))

            Gs.append(geno_df)
    
    G = pd.concat(Gs, axis=1)

    return G, chrom_to_gene_to_variant_indices


def load_genotypes_annotations(params, chro=None, all_genes=False):
    '''
    Load genes to be used.
    INPUT:
        - params: input yaml file loaded as params dict
        - chro: (optional) load data for single chromosome
    OUTPUT:
        - Gs: genotype dataframe [individuals x variants]
        - Zs: annotation dataframe [variants x annotations]
    '''

    # get genes to load 
    chrom_to_genes = get_genes(params) if not all_genes else collect_gene_files(params)

    if chrom_to_genes is None:
        chrom_to_genes = collect_gene_files(params)

    if chro is not None:
        chrom_to_genes = {chro: chrom_to_genes[chro]}

    Gs = []
    Zs = []

    # flag to perform NMF grouping of related annotations
    anno_nmf = params.get('anno_nmf', False)

    for chrom, chrom_genes in chrom_to_genes.items():
        if (chro is None) or (chrom == chro):
            for gene in chrom_genes:
                annotation_file = os.path.join(params['Z'], f'chr{chrom}', f'{gene}_annotations.tsv.gz')
                genotype_file = os.path.join(params['G'], f'chr{chrom}', f'{gene}_genotypes.pt')

                if not os.path.exists(annotation_file):
                    print(f"Annotation file for gene {gene} not found. Skipping.")
                    continue

                if not os.path.exists(genotype_file):
                    print(f"Genotype file for gene {gene} not found. Skipping.")
                    continue

                # --- Load annotation file ---
                anno_df = pd.read_csv(annotation_file, sep='\t', compression='gzip')
                anno_df = anno_df.drop(['chr', 'pos'], axis = 1) 
                # --- Group related annotations ---
                anno_df = utils.perform_anno_nmf(anno_df, params)
                anno_df['gene'] = gene


                # --- Load genotype tensor ---
                geno_ts = torch.load(genotype_file, map_location='cpu')
                if geno_ts.is_sparse:
                    geno_ts = geno_ts.to_dense()

                # --- Load variant IDs ---
                variant_file = os.path.join(params['G'], f'chr{chrom}', f'{gene}_variant_ids.txt')
                with open(variant_file, 'r') as f:
                    variant_ids = [line.strip() for line in f if line.strip()]

                # --- Sanity checks ---
                assert geno_ts.shape[1] == len(variant_ids), f"{gene}: genotype cols != variant count"
                assert anno_df.shape[0] == len(variant_ids), f"{gene}: annotation rows != variant count"

                # --- Create genotype DataFrame ---
                gene_variant_ids = [f"{gene}:{var}" for var in variant_ids]
                geno_df = pd.DataFrame(geno_ts.cpu().numpy(), columns=gene_variant_ids)

                # --- Append gene dataframes ---
                Gs.append(geno_df) 
                Zs.append(anno_df) 

 
    # --- Concatenate all gene dataframes ----
    G = pd.concat(Gs, axis=1)   # horizontally
    Z = pd.concat(Zs, axis=0)   # vertically

    # --- Add intercept term ---
    if ("Intercept" not in Z.columns) or ("intercept" not in Z.columns):
        Z['intercept'] = 1

    # --- Set index ---
    Z.set_index('gene', append=True, inplace=True)


    # --- Sanity check ---
    assert Z.shape[0] == G.shape[1], f"Zs rows ({Z.shape[0]}) != Gs cols ({G.shape[1]})"

    return G, Z


def load_tau(params):
    try:
        tau = torch.tensor(pd.read_csv(os.path.join(os.path.join(params['output'],'joint_model','final_tau.csv')), index_col = 0)['mean'], dtype = torch.float32)
        return tau
    except:
        print("Issue loading tau. Please make sure you have trained joint model first.")
        return None






