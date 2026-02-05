#!/usr/bin/env python3
"""
Check A1/A2 swapped alleles and duplicate positions
Compare merged frequencies with original cohort frequencies
Processes chromosome by chromosome to reduce memory usage
Usage: python check_allele_frequencies.py [SET]
  SET: Train or Test (default: Train)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Define dtype for faster loading (avoids type inference overhead)
freq_dtypes = {
    'CHR': 'int32',
    'SNP': 'string',
    'A1': 'string',
    'A2': 'string',
    'MAF': 'float32',
    'NCHROBS': 'int32'
}

def load_freq_file(freq_file):
    """Load and process a frequency file"""
    if not freq_file.exists():
        return None
    try:
        df = pd.read_csv(freq_file, sep=r'\s+', dtype=freq_dtypes, engine='c',
                        usecols=['CHR', 'SNP', 'A1', 'A2', 'MAF', 'NCHROBS'])
        # Vectorized BP extraction
        df['BP'] = df['SNP'].str.split(':').str[1].str.split('_').str[0]
        df = df[df['BP'].notna()].copy()
        df['key'] = df['CHR'].astype(str) + ':' + df['BP'].astype(str)
        return df
    except Exception as e:
        print(f"Error loading {freq_file}: {e}", file=sys.stderr)
        return None

def process_chromosome(chrom, freq_dir, cohorts):
    """Process a single chromosome and return statistics"""
    merged_file = freq_dir / f"merged_chr{chrom}.frq"
    if not merged_file.exists():
        return None
    
    # Load merged frequencies for this chromosome
    merged_df = load_freq_file(merged_file)
    if merged_df is None or len(merged_df) == 0:
        return None
    
    merged_df['chromosome'] = chrom
    # Create sorted allele pair string for duplicate detection (vectorized, much faster)
    # Use string representation: sort alleles and join with comma
    alleles_sorted = pd.DataFrame({
        'min_allele': merged_df[['A1', 'A2']].min(axis=1),
        'max_allele': merged_df[['A1', 'A2']].max(axis=1)
    })
    merged_df['alleles_set'] = (alleles_sorted['min_allele'] + ',' + alleles_sorted['max_allele']).astype(str)
    
    # Load cohort frequencies for this chromosome
    cohort_dfs = {}
    for cohort in cohorts:
        cohort_file = freq_dir / f"{cohort}_chr{chrom}.frq"
        if cohort_file.exists():
            df = load_freq_file(cohort_file)
            if df is not None and len(df) > 0:
                cohort_dfs[cohort] = df
    
    if not cohort_dfs:
        return None
    
    # Process swapped alleles per cohort
    swapped_stats = []
    swapped_maf_data = []
    
    for cohort, cohort_df in cohort_dfs.items():
        # Merge with merged frequencies
        comparison = cohort_df.merge(
            merged_df[['key', 'A1', 'A2', 'MAF', 'chromosome']],
            on='key',
            how='inner',
            suffixes=('_cohort', '_merged')
        )
        
        if len(comparison) == 0:
            continue
        
        # Detect swapped alleles
        comparison['alleles_swapped'] = (
            (comparison['A1_cohort'] == comparison['A2_merged']) & 
            (comparison['A2_cohort'] == comparison['A1_merged'])
        )
        
        n_swapped = comparison['alleles_swapped'].sum()
        n_total = len(comparison)
        pct_swapped = (n_swapped / n_total * 100) if n_total > 0 else 0
        
        swapped_stats.append({
            'chromosome': chrom,
            'cohort': cohort,
            'total_variants_in_both': n_total,
            'swapped_alleles': n_swapped,
            'pct_swapped': pct_swapped
        })
        
        # Collect MAF data for swapped alleles
        swapped_variants = comparison[comparison['alleles_swapped']].copy()
        if len(swapped_variants) > 0:
            swapped_variants['MAF_cohort_original'] = swapped_variants['MAF_cohort']
            swapped_variants['MAF_cohort_adjusted'] = 1.0 - swapped_variants['MAF_cohort']
            maf_diff_adjusted = (swapped_variants['MAF_merged'] - swapped_variants['MAF_cohort_adjusted']).abs()
            
            swapped_maf_data.append({
                'chromosome': chrom,
                'cohort': cohort,
                'n_swapped': len(swapped_variants),
                'mean_maf_merged': swapped_variants['MAF_merged'].mean(),
                'mean_maf_cohort_original': swapped_variants['MAF_cohort_original'].mean(),
                'mean_maf_cohort_adjusted': swapped_variants['MAF_cohort_adjusted'].mean(),
                'mean_maf_diff_adjusted': maf_diff_adjusted.mean(),
                'median_maf_diff_adjusted': maf_diff_adjusted.median()
            })
    
    # Count duplicate positions for this chromosome
    position_allele_counts = merged_df.groupby('key')['alleles_set'].nunique()
    duplicate_positions = position_allele_counts[position_allele_counts > 1]
    
    dup_pos_details = []
    for pos_key in duplicate_positions.index:
        pos_variants = merged_df[merged_df['key'] == pos_key]
        unique_allele_sets = pos_variants['alleles_set'].nunique()
        
        if unique_allele_sets == 1:
            dup_type = "same_allele_set"
        else:
            dup_type = "different_allele_set"
        
        dup_pos_details.append({
            'chromosome': chrom,
            'position': pos_key,
            'n_variants': len(pos_variants),
            'n_unique_allele_sets': unique_allele_sets,
            'type': dup_type
        })
    
    return {
        'swapped_stats': swapped_stats,
        'swapped_maf': swapped_maf_data,
        'dup_positions': dup_pos_details,
        'n_total_positions': len(position_allele_counts),
        'n_duplicate_positions': len(duplicate_positions),
        'n_variants': len(merged_df)
    }

def main():
    # Get SET from command line or use default
    SET = sys.argv[1] if len(sys.argv) > 1 else "Train"
    
    # Define paths
    GENOTYPE_TRAIN_DIR = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/Train/chroms"
    GENOTYPE_TEST_DIR = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/Test/chroms"
    
    if SET == "Train":
        BASE_DIR = Path(GENOTYPE_TRAIN_DIR)
        COHORTS = ["AnswerALS", "Mayo", "MSBB", "NYGC", "ROSMAP", "GTEX"]
    elif SET == "Test":
        BASE_DIR = Path(GENOTYPE_TEST_DIR)
        COHORTS = ["ROSMAP_DLPFC"]
    else:
        print(f"ERROR: Invalid set: {SET}. Must be 'Train' or 'Test'", file=sys.stderr)
        sys.exit(1)
    
    FREQ_CHECK_DIR = BASE_DIR / "freq_checks"
    OUTPUT_DIR = BASE_DIR / "allele_freq_analysis"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    train_cohorts = [c for c in COHORTS if c != 'ROSMAP_DLPFC']
    
    print("=" * 60)
    print(f"Allele Frequency Analysis for {SET}")
    print("=" * 60)
    print(f"Date: {pd.Timestamp.now()}")
    print("Processing chromosome by chromosome to reduce memory usage...")
    print()
    
    # Process each chromosome separately
    all_swapped_stats = []
    all_swapped_maf = []
    all_dup_positions = []
    total_positions = 0
    total_dup_positions = 0
    total_variants = 0
    
    for chrom in range(1, 23):
        print(f"Processing chromosome {chrom}...", end=' ', flush=True)
        start_time = time.time()
        result = process_chromosome(chrom, FREQ_CHECK_DIR, train_cohorts)
        elapsed = time.time() - start_time
        
        if result is None:
            print("skipped (no data)")
            continue
        
        all_swapped_stats.extend(result['swapped_stats'])
        all_swapped_maf.extend(result['swapped_maf'])
        all_dup_positions.extend(result['dup_positions'])
        total_positions += result['n_total_positions']
        total_dup_positions += result['n_duplicate_positions']
        total_variants += result['n_variants']
        
        print(f"done ({result['n_variants']:,} variants, {result['n_duplicate_positions']:,} duplicate positions, {elapsed:.1f}s)", flush=True)
    
    print("\n" + "=" * 60)
    print("Analysis 1: Count variants with A1/A2 swapped")
    print("=" * 60)
    
    # Aggregate swapped stats by cohort
    if all_swapped_stats:
        swapped_df = pd.DataFrame(all_swapped_stats)
        swapped_summary = swapped_df.groupby('cohort').agg({
            'total_variants_in_both': 'sum',
            'swapped_alleles': 'sum'
        }).reset_index()
        swapped_summary['pct_swapped'] = (swapped_summary['swapped_alleles'] / 
                                          swapped_summary['total_variants_in_both'] * 100)
        
        for _, row in swapped_summary.iterrows():
            print(f"{row['cohort']}: {row['swapped_alleles']:,} / {row['total_variants_in_both']:,} "
                  f"variants have swapped alleles ({row['pct_swapped']:.2f}%)")
        
        swapped_summary_file = OUTPUT_DIR / f"swapped_alleles_summary_{SET}.txt"
        swapped_summary.to_csv(swapped_summary_file, sep='\t', index=False)
        print(f"\nSummary saved to: {swapped_summary_file}")
    
    print("\n" + "=" * 60)
    print("Analysis 2: Count duplicate positions")
    print("=" * 60)
    
    pct_duplicate = (total_dup_positions / total_positions * 100) if total_positions > 0 else 0
    print(f"Total unique positions: {total_positions:,}")
    print(f"Positions with duplicate variants (different allele sets): {total_dup_positions:,} ({pct_duplicate:.2f}%)")
    
    if all_dup_positions:
        dup_pos_df = pd.DataFrame(all_dup_positions)
        same_set_count = (dup_pos_df['type'] == 'same_allele_set').sum()
        diff_set_count = (dup_pos_df['type'] == 'different_allele_set').sum()
        
        print(f"\nBreakdown of duplicate positions:")
        print(f"  Positions with same allele set (different A1/A2 order): {same_set_count:,}")
        if total_dup_positions > 0:
            print(f"    Percentage: {same_set_count/total_dup_positions*100:.2f}%")
        print(f"  Positions with different allele sets (multiallelic): {diff_set_count:,}")
        if total_dup_positions > 0:
            print(f"    Percentage: {diff_set_count/total_dup_positions*100:.2f}%")
        
        dup_pos_file = OUTPUT_DIR / f"duplicate_positions_{SET}.txt"
        dup_pos_df.to_csv(dup_pos_file, sep='\t', index=False)
        print(f"\nDuplicate positions saved to: {dup_pos_file}")
        
        # Summary by chromosome
        dup_by_chr = dup_pos_df.groupby('chromosome').size()
        print(f"\nDuplicate variants by chromosome:")
        print(dup_by_chr.to_string())
    
    print("\n" + "=" * 60)
    print("Analysis 3: MAF for swapped alleles (cohort vs merged)")
    print("=" * 60)
    
    if all_swapped_maf:
        swapped_maf_df = pd.DataFrame(all_swapped_maf)
        
        # Aggregate by cohort (weighted by n_swapped)
        for cohort in swapped_maf_df['cohort'].unique():
            cohort_data = swapped_maf_df[swapped_maf_df['cohort'] == cohort]
            total_swapped = cohort_data['n_swapped'].sum()
            
            if total_swapped > 0:
                mean_maf_merged = (cohort_data['n_swapped'] * cohort_data['mean_maf_merged']).sum() / total_swapped
                mean_maf_orig = (cohort_data['n_swapped'] * cohort_data['mean_maf_cohort_original']).sum() / total_swapped
                mean_maf_adj = (cohort_data['n_swapped'] * cohort_data['mean_maf_cohort_adjusted']).sum() / total_swapped
                mean_diff = (cohort_data['n_swapped'] * cohort_data['mean_maf_diff_adjusted']).sum() / total_swapped
                median_diff = cohort_data['median_maf_diff_adjusted'].median()
                
                print(f"\n{cohort} - Swapped Alleles MAF Comparison:")
                print(f"  Number of swapped variants: {total_swapped:,}")
                print(f"  Mean MAF in merged: {mean_maf_merged:.6f}")
                print(f"  Mean MAF in cohort (original): {mean_maf_orig:.6f}")
                print(f"  Mean MAF in cohort (adjusted for swap): {mean_maf_adj:.6f}")
                print(f"  Mean |MAF_diff| (adjusted): {mean_diff:.6f}")
                print(f"  Median |MAF_diff| (adjusted): {median_diff:.6f}")
        
        swapped_maf_file = OUTPUT_DIR / f"swapped_maf_summary_{SET}.txt"
        swapped_maf_df.to_csv(swapped_maf_file, sep='\t', index=False)
        print(f"\nDetailed MAF data saved to: {swapped_maf_file}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"Total variants processed: {total_variants:,}")
    print(f"Results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
