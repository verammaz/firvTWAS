import os
import pandas as pd
import numpy as np
import torch

ANNOTATIONS_DIR_RAW = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_raw"
ANNOTATIONS_DIR_SCALED = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_scaled"
CHROMBPNET_ZSCORED_DIR = "/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/58K_preview_compact/rare_variants/chrombpnet/variant_peak_pairs_scored/zscore_normalized"


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
        for file in os.listdir(os.path.join(annotations_dir, f'chr{chrom}')):
            if file.endswith('.tsv.gz'):
                gene = file.split('_')[0]
                annotations = pd.read_csv(os.path.join(annotations_dir, f'chr{chrom}', file), sep='\t', compression='gzip')
                assert 'variant_id' in annotations.columns, f"variant_id column not found in {file}"
                # add chrombpnet zscored value
                for cell in ['microglia', 'astrocyte', 'neuron', 'oligodendrocyte']:
                    print(f"\t adding chrombpnet zscored {cell} values...")
                    chrombp = pd.read_csv(f"{CHROMBPNET_ZSCORED_DIR}/rare_variant_chrombpnet_{cell}_ATAC_chr{chrom}_zscore.tsv.gz", sep='\t', compression='gzip',
                        usecols = ['CHR', 'BP', 'zscore']    )
                    chrombp = chrombp.rename(columns={'CHR': 'chr', 'BP': 'pos'})
                    chrombp = chrombp[chrombp['chr']==chrom]
                    chrombp = chrombp.drop(columns=['chr'])
                    annotations = annotations.merge(chrombp, on=['pos'], how='left')
                    annotations[f'chrombpnet_{cell}'] = np.where(annotations['zscore'].isna(), 0, annotations['zscore'])
                    annotations = annotations.drop(columns=['zscore'])
                    annotations = annotations.drop(columns=[f'chrombpnet_log_counts_diff_{cell}'])
                    # rename 
                    annotations = annotations.rename(columns={f'chrombpnet_{cell}': f'chrombpnet_log_counts_diff_{cell}'})
                    # add gene column
                    annotations['gene'] = gene
                all_annotations.append(annotations)
    return pd.concat(all_annotations)


def scale_annotations_global(annotations_matrix):
    """
    Scale annotations globally.
    """
    assert 'variant_id' in annotations_matrix.columns, f"variant_id column not found in annotations_matrix"
    for column in annotations_matrix.columns:
        if column in ['variant_id', 'chr', 'pos','dist_to_TSS']:
            continue
        elif column.startswith('chrombpnet_log_counts_diff_'): # already scaled in per-gene annotations load and merge step
            continue
        else:
            values = annotations_matrix[column].values
            # if negative values --> zcale
            if np.any(values < 0):
                print(f"zscoring {column}...")
                print(f"\tmean: {np.mean(annotations_matrix[column])}")
                print(f"\tstd: {np.std(annotations_matrix[column])}")
                annotations_matrix[column] = zscore_scale_column(annotations_matrix[column])
            # if positive values --> minmax scale
            else:
                print(f"minmax scaling {column}...")
                print(f"\tmin: {np.min(annotations_matrix[column])}")
                print(f"\tmax: {np.max(annotations_matrix[column])}")
                annotations_matrix[column] = minmax_scale_column(annotations_matrix[column])
    annotations_matrix = annotations_matrix.set_index('variant_id')
    return annotations_matrix

def save_per_gene_annotations(annotations_matrix):
    """
    Save per-gene annotations.
    """
    for chrom in range(1, 23):
        chrom_annotations = annotations_matrix[annotations_matrix['chr']==chrom]
        for gene in chrom_annotations['gene'].unique():
            gene_annotations = chrom_annotations[chrom_annotations['gene']==gene]
            gene_annotations = gene_annotations.drop(columns=['gene', 'chr', 'pos'])
            # save tsv
            gene_annotations.to_csv(os.path.join(ANNOTATIONS_DIR_SCALED, f'chr{chrom}', f'{gene}_annotations.tsv.gz'), sep='\t', compression='gzip', index=True)
            # save tensor
            # gene_annotations_tensor = torch.tensor(gene_annotations.values)
            # torch.save(gene_annotations_tensor, os.path.join(ANNOTATIONS_DIR_SCALED, f'chr{chrom}', f'{gene}_annotations.pt'))
        

if __name__ == "__main__":
    annotations_matrix = load_full_annotation_matrix(ANNOTATIONS_DIR_RAW)
    annotations_matrix = scale_annotations_global(annotations_matrix)
    save_per_gene_annotations(annotations_matrix)
