"""
apply_zscore_rare_variants.py

Example: apply the saved scaler to rare variant ChromBPNet VEP scores.

For each (cell_type, assay, score):
  1. Load scored variant-peak pairs across all chromosomes
  2. Per variant: pick the row with max |score|, preserving sign
  3. Normalize using the saved scaler from zscore_scalers.joblib
     (StandardScaler with_mean=False: divides by std, sign preserved)

Output: one parquet per cell type with columns:
  CHR, SNP, BP, A1, A2, assay, score_name, raw_score, zscore
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
from tqdm import tqdm

CHR_NUM = sys.argv[1]

ANNOTATION_DIR = '/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations/'


# ─── Constants ───────────────────────────────────────────────────────────────
meta_cols = ['CHR', 'SNP', 'BP', 'A1', 'A2']
cell_type_list = ['microglia', 'astrocyte', 'neuron', 'oligodendrocyte']
assay_list = ['ATAC', 'H3K27ac', 'H3K4me3']
chrs = list(range(1, 23))

score_cols = [
    'log_counts_diff_chrombpnet',
    'log_probs_diff_abs_sum_chrombpnet',
    'probs_jsd_diff_chrombpnet',
]

short_names = {
    'log_counts_diff_chrombpnet': 'log_counts_diff',
    'log_probs_diff_abs_sum_chrombpnet': 'log_probs_diff_abs',
    'probs_jsd_diff_chrombpnet': 'probs_jsd_diff',
}

base_dir = '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations_hg38/merged_annotations_ADSP_v2'
chrombpnet_dir = f'{base_dir}/chrombpnet/variant_peak_pairs_scored'
scaler_dir = f'{chrombpnet_dir}/zscore_scaler'

scored_dir = '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/58K_preview_compact/rare_variants/chrombpnet/variant_peak_pairs_scored'
out_dir = os.path.join(scored_dir, 'zscore_normalized')

# ─── Load scalers ────────────────────────────────────────────────────────────
scalers_path = os.path.join(scaler_dir, 'zscore_scalers.joblib')
print(f'Loading scalers from {scalers_path}...')
scalers = joblib.load(scalers_path)

# Process each gene annotation matrix for this chromosome
# 1. load annotation matrix
# 2. scale chrombpnet
# 3. save annotation matrix

# ─── Process each cell type ──────────────────────────────────────────────────
os.makedirs(out_dir, exist_ok=True)

for ct in tqdm(cell_type_list):
    print(f'\nProcessing {ct}...')

    # Load all chromosomes
    ct_dfs = []
    for chrom in chrs:
        f = f'{scored_dir}/rare_variant_chrombpnet_{ct}_chr{chrom}_variant_peak_pairs_scored.csv'
        try:
            df = pd.read_csv(f, sep='\t',
                             usecols=['Chr', 'SNP', 'BP', 'ALT', 'REF', 'assay'] + score_cols)
        except FileNotFoundError:
            print(f'  Missing: chr{chrom}')
            continue
        df.rename(columns={'Chr': 'CHR', 'ALT': 'A1', 'REF': 'A2'}, inplace=True)
        ct_dfs.append(df)

    ct_df = pd.concat(ct_dfs, ignore_index=True)
    del ct_dfs
    print(f'  Loaded {ct_df.shape[0]:,} variant-peak pairs')

    results = []

    for assay in tqdm(assay_list):
        sub = ct_df[ct_df['assay'] == assay].copy()
        print(f'  Assay: {assay} ({sub.shape[0]:,} rows)')

        for score_col in score_cols:
            short = short_names[score_col]
            key = f'{ct}__{assay}__{short}'
            scaler = scalers[key]

            # Max-abs with sign preservation per variant (drop all-NaN groups)
            sub['_abs'] = sub[score_col].abs()
            idx = sub.groupby(meta_cols)['_abs'].idxmax().dropna().astype(int)
            sub_max = sub.loc[idx, meta_cols + [score_col]].copy()

            # Z-score normalize
            raw = sub_max[score_col].values.reshape(-1, 1)
            zscored = scaler.transform(raw).ravel()

            out = sub_max[meta_cols].copy()
            out['assay'] = assay
            out['score_name'] = short
            out['raw_score'] = sub_max[score_col].values
            out['zscore'] = zscored
            results.append(out)

            print(f'    {key}: {len(out):,} variants  '
                  f'raw=[{out["raw_score"].min():.4f}, {out["raw_score"].max():.4f}]  '
                  f'z=[{out["zscore"].min():.2f}, {out["zscore"].max():.2f}]')

    result_df = pd.concat(results, ignore_index=True)
    out_path = f'{out_dir}/rare_variant_chrombpnet_{ct}_zscore.parquet'
    result_df.to_parquet(out_path, index=False)
    print(f'  Written: {out_path}  ({result_df.shape[0]:,} rows)')

    del ct_df, results, result_df

print('\nDone!')