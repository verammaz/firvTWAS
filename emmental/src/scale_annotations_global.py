import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

ANNOTATIONS_DIR_RAW = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_raw"
ANNOTATIONS_DIR_SCALED = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_scaled"
PLOT_DIR = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/plots"

annotations = ['chr', 'pos', 'MAP20', 'phyloP17way_primate', 'phyloP30way_mammalian', 'phastCons17way_primate_rankscore', 'phastCons30way_mammalian', 
                'GERP_RS', 'bStatistic', 'integrated_fitCons_score', 'H1-hESC_fitCons_score', 'gnomAD_genomes_POPMAX_AF', 'gnomAD_genomes_AFR_AF', 
                'gnomAD_genomes_AMR_AF', 'gnomAD_genomes_NFE_AF', 'funseq2_noncoding_score', 'CADD_raw', 'CADD_phred', 'DANN_score', 
                'fathmm-MKL_non-coding_score', 'fathmm-MKL_coding_score', 'fathmm-XF_score', 'Eigen-raw', 'Eigen-PC-raw', 'EnhancerFinder_brain_enhancer', 
                'FANTOM5_CAGE_peak_robust', 'Roadmap_E030_GenoSkyline_Plus_score', 'Roadmap_E050_GenoSkyline_Plus_score', 'Roadmap_E051_GenoSkyline_Plus_score', 
                'Roadmap_E067_GenoSkyline_Plus_score', 'Roadmap_E068_GenoSkyline_Plus_score', 'Roadmap_E069_GenoSkyline_Plus_score', 'Roadmap_E070_GenoSkyline_Plus_score', 
                'Roadmap_E072_GenoSkyline_Plus_score', 'Roadmap_E073_GenoSkyline_Plus_score', 'Roadmap_E074_GenoSkyline_Plus_score', 'Roadmap_E124_GenoSkyline_Plus_score', 
                'lof', 'missense', 'microglia_TF_delta_max', 'microglia_TF_delta_min', 'astrocyte_TF_delta_max', 'astrocyte_TF_delta_min', 'neuronal_TF_delta_max', 
                'neuronal_TF_delta_min', 'oligodendrocyte_TF_delta_max', 'oligodendrocyte_TF_delta_min', 'log_counts_diff_chrombpnet_microglia', 
                'log_counts_diff_chrombpnet_astrocyte', 'log_counts_diff_chrombpnet_neuron', 'log_counts_diff_chrombpnet_oligodendrocyte', 
                'alphamissense', 'splice', 'ABC_microglia', 'ABC_neuron', 'ABC_oligodendrocyte', 'ABC_astrocyte', 'dist_to_TSS']
                

def _bytes_to_gb(n_bytes):
    return n_bytes / (1024 ** 3)


def minmax_scale_column(column):
    """
    Minmax scale a column.
    """
    return (column - column.min()) / (column.max() - column.min())

def zscore_scale_column(column):
    """
    Z-score scale a column.
    """
    return (column - column.mean()) / column.std()

def clip_column(column, min_val, max_val):
    """
    Clip a column to a given range.
    """
    return np.clip(column, min_val, max_val)


def load_full_annotation_matrix(annotations_dir):
    """
    Load all per-gene annotation matrices and concatenate them into a single matrix.
    """
    all_annotations = []
    for chrom in range(1, 23):
        print(f"Loading annotations for chromosome {chrom} ({len(os.listdir(os.path.join(annotations_dir, f'chr{chrom}')))} files)...")
        chrom_annotations = []
        for file in tqdm(os.listdir(os.path.join(annotations_dir, f'chr{chrom}')), desc=f"Loading annotations for chromosome {chrom}"):
            if file.endswith('.tsv.gz'):
                gene = file.split('_')[0]
                annotations = pd.read_csv(os.path.join(annotations_dir, f'chr{chrom}', file), sep='\t', compression='gzip')
                assert 'variant_id' in annotations.columns, f"variant_id column not found in {file}"
                # add gene column
                annotations['gene'] = gene
                chrom_annotations.append(annotations)

        if chrom_annotations:
            chrom_df = pd.concat(chrom_annotations, ignore_index=True)
            chrom_mem_gb = _bytes_to_gb(chrom_df.memory_usage(deep=True).sum())
            print(f"\tchr{chrom} loaded matrix shape: {chrom_df.shape}, memory: {chrom_mem_gb:.2f} GB")
            all_annotations.append(chrom_df)

    full_df = pd.concat(all_annotations, ignore_index=True)
    total_mem_gb = _bytes_to_gb(full_df.memory_usage(deep=True).sum())
    print(f"Full loaded matrix memory usage: {total_mem_gb:.2f} GB")
    return full_df


def scale_annotations_global(annotations_matrix):
    """
    Scale annotations globally.
    """
    minmax = dict()
    zscore = dict()

    assert 'variant_id' in annotations_matrix.columns, f"variant_id column not found in annotations_matrix"
    for column in annotations_matrix.columns:
        if column in ['gene','variant_id', 'chr', 'pos']:
            continue
        elif 'ABC_' in column:
            continue
        elif 'gnomAD_genomes_POPMAX_AF' == column:
            continue
        else:
            print(f"Scaling {column}...")
            values = annotations_matrix[column].values
            # clip dist_to_TSS at 0
            if column == 'dist_to_TSS':
                print(f"\tclipping at 0... (before: min: {np.min(values)} max: {np.max(values)})")
                values = clip_column(values, 0, None)
            # if negative values --> zcale
            if np.any(values < 0):
                zscore[column] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                print(f"\tzscoring... mean: {np.mean(values)} std: {np.std(values)}")
                if column.startswith('chrombpnet'): # already scaled in map_annotate_variants.py
                    continue
                annotations_matrix[column] = zscore_scale_column(values)
            # if positive values --> minmax scale
            else:
                minmax[column] = {
                    'min': np.min(values),
                    'max': np.max(values)
                }
                print(f"\tminmax scaling... min: {np.min(values)} max: {np.max(values)}")
                annotations_matrix[column] = minmax_scale_column(values)
    annotations_matrix = annotations_matrix.set_index('variant_id')
    return annotations_matrix, minmax, zscore

def save_per_gene_annotations(annotations_matrix):
    """
    Save per-gene annotations.
    """
    if 'variant_id' not in annotations_matrix.columns:
        if annotations_matrix.index.name == 'variant_id':
            annotations_matrix = annotations_matrix.reset_index()
        else:
            raise KeyError("variant_id not found as a column or index name.")

    for chrom in range(1, 23):
        print(f"Saving per-gene annotations for chromosome {chrom}...")
        os.makedirs(os.path.join(ANNOTATIONS_DIR_SCALED, f'chr{chrom}'), exist_ok=True)
        chrom_annotations = annotations_matrix[annotations_matrix['chr']==chrom]
        for gene in tqdm(chrom_annotations['gene'].unique(), desc=f"Saving per-gene annotations for chromosome {chrom}"):

            gene_annotations = chrom_annotations[chrom_annotations['gene']==gene].copy()
            gene_annotations = gene_annotations.drop(columns=['gene', 'chr', 'pos'])
            # save tsv
            gene_annotations.set_index('variant_id', inplace=True)
            gene_annotations.to_csv(os.path.join(ANNOTATIONS_DIR_SCALED, f'chr{chrom}', f'{gene}_annotations.tsv.gz'), sep='\t', compression='gzip', index=True)
            # save tensor
            # gene_annotations_tensor = torch.tensor(gene_annotations.values)
            # torch.save(gene_annotations_tensor, os.path.join(ANNOTATIONS_DIR_SCALED, f'chr{chrom}', f'{gene}_annotations.pt'))

def plot_distribution_of_annotations(annotations_matrix):
    """
    Plot distribution of each annotation column.
    """
    # create one plot with subplots for each annotation column
    fig, axes = plt.subplots(len(annotations_matrix.columns), 1, figsize=(10, 10))
    for i, column in enumerate(annotations_matrix.columns):
        axes[i].hist(annotations_matrix[column], bins=100)
        axes[i].set_title(column)
    plt.savefig(os.path.join(PLOT_DIR, 'annotations_distribution.png'))
    plt.close()


if __name__ == "__main__":
    annotations_matrix = load_full_annotation_matrix(ANNOTATIONS_DIR_RAW)
    print(f"Full annotations matrix shape: {annotations_matrix.shape}")
    # plot distirbution of each annotation column -- pre scaling
    # plot_distribution_of_annotations(annotations_matrix)
    annotations_matrix, minmax, zscore = scale_annotations_global(annotations_matrix)
    # plot distirbution of each annotation column -- post scaling
    # plot_distribution_of_annotations(annotations_matrix)
    save_per_gene_annotations(annotations_matrix)
    # save minmax and zscore dictionaries
    with open(os.path.join(ANNOTATIONS_DIR_SCALED, 'minmax.json'), 'w') as f:
        json.dump(minmax, f, indent=4)
    with open(os.path.join(ANNOTATIONS_DIR_SCALED, 'zscore.json'), 'w') as f:
        json.dump(zscore, f, indent=4)
