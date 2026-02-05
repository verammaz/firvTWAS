#!/usr/bin/env python3
"""
Extract common participants across genotype, covariates, and expression files.

This script:
1. Extracts participant IDs from genotype files (IID = participant_id)
2. Finds intersection of participants in genotype, covariates, and expression files
3. Creates keep files for each cohort: {cohort}_keep_participants.txt
4. Resaves subsetted covariates files
"""

import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json

# ============================================================================
# HARDCODED PATHS
# ============================================================================
FILE_SHEET = "/gpfs/commons/home/vmazeeva/BigBrain_files_sheet.tsv"
BASE_DIR = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain"
GENOTYPE_DIR = os.path.join(BASE_DIR, "Genotypes")
EXPRESSION_DIR = os.path.join(BASE_DIR, "Phenotypes_TPM")
COVARIATE_FILE_TRAIN = os.path.join(BASE_DIR, "Processed", "Train", "covariates.tsv")
COVARIATE_FILE_TEST = os.path.join(BASE_DIR, "Processed", "Test", "covariates.tsv")
OUTPUT_DIR_TRAIN = os.path.join(BASE_DIR, "Processed", "Train")
OUTPUT_DIR_TEST = os.path.join(BASE_DIR, "Processed", "Test")
GENOTYPE_SUBSET_DIR = os.path.join(GENOTYPE_DIR, "subset")

# Create output directories
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)
os.makedirs(GENOTYPE_SUBSET_DIR, exist_ok=True)

# Create figures directory
FIGURES_DIR = "/gpfs/commons/home/vmazeeva/gruyere-expr/preprocess/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Statistics tracking
stats = defaultdict(lambda: {
    'cohort': None,
    'set': None,  # 'train' or 'test'
    'n_genotype': 0,
    'n_covariates': 0,
    'n_expression': 0,
    'n_final': 0,
    'n_samples_final': 0,
    'has_expression': False
})

print("=" * 80)
print("EXTRACT COMMON PARTICIPANTS")
print("=" * 80)
print(f"Train output: {OUTPUT_DIR_TRAIN}")
print(f"Test output: {OUTPUT_DIR_TEST}")
print(f"Keep files dir: {GENOTYPE_SUBSET_DIR}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_cohort_name(cohort):
    """Normalize cohort names."""
    if cohort == "NYGC ALS":
        return "NYGC"
    elif cohort == "Mayo Clinic":
        return "Mayo"
    return cohort

def get_genotype_participant_ids(genotype_path):
    """
    Extract participant IDs (IIDs) from genotype files.
    IID in PLINK = participant_id in covariates.
    """
    # Handle the special notation used in file sheet
    if '.bim/.bed/.fam' in genotype_path:
        base_path = genotype_path.replace('.bim/.bed/.fam', '')
    elif '.pgen/.psam/.pvar' in genotype_path:
        base_path = genotype_path.replace('.pgen/.psam/.pvar', '')
    else:
        base_path = genotype_path
    
    # Check for PLINK1 format (.fam file)
    fam_file = f"{base_path}.fam"
    if os.path.exists(fam_file):
        fam_df = pd.read_csv(fam_file, sep='\s+', header=None)
        # .fam format: FID IID ... (IID is column 1)
        return set(fam_df.iloc[:, 1].astype(str))
    
    # Check for PLINK2 format (.psam file)
    psam_file = f"{base_path}.psam"
    if os.path.exists(psam_file):
        # .psam files have header starting with #, skip it
        psam_df = pd.read_csv(psam_file, sep='\s+', skiprows=1, header=None)
        # First column (index 0) is IID
        return set(psam_df.iloc[:, 0].astype(str))
    
    raise FileNotFoundError(f"Could not find .fam or .psam file for: {genotype_path}")

def get_expression_participant_ids(expression_file, sample_to_participant_map):
    """
    Get participant IDs from expression file.
    Expression files have sample_ids as columns, map to participant_ids.
    """
    # Read just the header to get sample IDs
    expr_df = pd.read_csv(expression_file, sep="\t", index_col=0, nrows=0)
    expr_sample_ids = set(expr_df.columns.astype(str))
    
    # Map sample IDs to participant IDs
    participant_ids = set()
    unmapped_samples = []
    for sample_id in expr_sample_ids:
        # Check if sample_id maps to a participant_id
        # The sample_to_participant_map should have this mapping
        if sample_id in sample_to_participant_map:
            participant_ids.add(sample_to_participant_map[sample_id])
        # Also check if sample_id is already a participant_id
        elif sample_id in set(sample_to_participant_map.values()):
            participant_ids.add(sample_id)
        else:
            unmapped_samples.append(sample_id)
    
    # if unmapped_samples:
    #     print(f"    ⚠ WARNING: {len(unmapped_samples)} expression sample IDs could not be mapped to participants")
    #     print(f"    Example unmapped sample IDs: {unmapped_samples[:5]}")
    
    return participant_ids, expr_sample_ids

def create_filtering_visualizations(stats_dict, output_dir):
    """
    Create visualizations showing filtering statistics.
    """
    # Convert stats dict to DataFrame for easier plotting
    stats_list = []
    for key, stat in stats_dict.items():
        if stat['cohort'] is not None:
            stats_list.append(stat)
    
    if not stats_list:
        print("  ⚠ No statistics to plot")
        return
    
    stats_df = pd.DataFrame(stats_list)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_width = 14
    fig_height = 10
    
    # ========================================================================
    # Figure 1: Per-cohort filtering comparison (bar chart)
    # ========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # Separate train and test
    for idx, (set_name, ax) in enumerate(zip(['train', 'test'], axes)):
        set_stats = stats_df[stats_df['set'] == set_name].copy()
        if len(set_stats) == 0:
            ax.text(0.5, 0.5, f'No {set_name} cohorts processed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{set_name.upper()} Set: Participant Counts Before and After Filtering', 
                        fontsize=14, fontweight='bold')
            continue
        
        cohorts = set_stats['cohort'].values
        x = np.arange(len(cohorts))
        width = 0.2
        
        # Get counts
        geno_counts = set_stats['n_genotype'].values
        cov_counts = set_stats['n_covariates'].values
        expr_counts = set_stats['n_expression'].values
        final_counts = set_stats['n_final'].values
        
        # Plot bars
        ax.bar(x - 1.5*width, geno_counts, width, label='Genotype', alpha=0.8, color='#3498db')
        ax.bar(x - 0.5*width, cov_counts, width, label='Covariates', alpha=0.8, color='#2ecc71')
        ax.bar(x + 0.5*width, expr_counts, width, label='Expression', alpha=0.8, color='#e74c3c')
        ax.bar(x + 1.5*width, final_counts, width, label='Final (Intersection)', alpha=0.8, color='#f39c12', edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cohort', fontsize=12)
        ax.set_ylabel('Number of Participants', fontsize=12)
        ax.set_title(f'{set_name.upper()} Set: Participant Counts Before and After Filtering', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cohorts, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (g, c, e, f) in enumerate(zip(geno_counts, cov_counts, expr_counts, final_counts)):
            if g > 0:
                ax.text(i - 1.5*width, g, f'{int(g)}', ha='center', va='bottom', fontsize=8)
            if c > 0:
                ax.text(i - 0.5*width, c, f'{int(c)}', ha='center', va='bottom', fontsize=8)
            if e > 0:
                ax.text(i + 0.5*width, e, f'{int(e)}', ha='center', va='bottom', fontsize=8)
            if f > 0:
                ax.text(i + 1.5*width, f, f'{int(f)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'filtering_comparison_by_cohort.pdf'), transparent=True, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: filtering_comparison_by_cohort.pdf")
    
    # ========================================================================
    # Figure 2: Filtering efficiency (what was kept vs filtered out)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
    
    for idx, set_name in enumerate(['train', 'test']):
        ax = axes[idx]
        set_stats = stats_df[stats_df['set'] == set_name].copy()
        if len(set_stats) == 0:
            ax.text(0.5, 0.5, f'No {set_name} cohorts processed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{set_name.upper()} Set: Filtering Efficiency', 
                        fontsize=14, fontweight='bold')
            continue
        
        cohorts = set_stats['cohort'].values
        x = np.arange(len(cohorts))
        
        # Calculate what was kept (final) vs filtered out
        # Use covariates as the "starting point" since that's what we're filtering
        starting_counts = []
        kept_counts = []
        for _, row in set_stats.iterrows():
            # Starting point is the number of participants in the covariates file
            starting = row['n_covariates']
            starting_counts.append(starting)
            kept_counts.append(row['n_final'])
        
        starting_counts = np.array(starting_counts)
        kept_counts = np.array(kept_counts)
        filtered_counts = starting_counts - kept_counts
        
        # Stacked bar chart
        ax.bar(x, kept_counts, label='Kept', alpha=0.8, color='#2ecc71')
        ax.bar(x, filtered_counts, bottom=kept_counts, label='Filtered Out', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Cohort', fontsize=12)
        ax.set_ylabel('Number of Participants', fontsize=12)
        ax.set_title(f'{set_name.upper()} Set: Filtering Efficiency\n(Based on Covariates)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cohorts, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add percentage labels
        for i, (start, kept) in enumerate(zip(starting_counts, kept_counts)):
            if start > 0:
                pct = (kept / start) * 100
                ax.text(i, kept + filtered_counts[i]/2, f'{pct:.1f}%', 
                       ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'filtering_efficiency.pdf'), transparent=True, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: filtering_efficiency.pdf")
    
    # ========================================================================
    # Figure 3: Overall summary (train vs test)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    summary_data = []
    for set_name in ['train', 'test']:
        set_stats = stats_df[stats_df['set'] == set_name]
        if len(set_stats) > 0:
            summary_data.append({
                'Set': set_name.upper(),
                'Total Genotype': set_stats['n_genotype'].sum(),
                'Total Covariates': set_stats['n_covariates'].sum(),
                'Total Expression': set_stats['n_expression'].sum(),
                'Total Final': set_stats['n_final'].sum(),
                'Total Samples': set_stats['n_samples_final'].sum()
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        x = np.arange(len(summary_df))
        width = 0.15
        
        ax.bar(x - 2*width, summary_df['Total Genotype'], width, label='Genotype', alpha=0.8, color='#3498db')
        ax.bar(x - width, summary_df['Total Covariates'], width, label='Covariates', alpha=0.8, color='#2ecc71')
        ax.bar(x, summary_df['Total Expression'], width, label='Expression', alpha=0.8, color='#e74c3c')
        ax.bar(x + width, summary_df['Total Final'], width, label='Final Participants', alpha=0.8, color='#f39c12', edgecolor='black', linewidth=2)
        ax.bar(x + 2*width, summary_df['Total Samples'], width, label='Final Samples', alpha=0.8, color='#9b59b6', edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Set', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Overall Summary: Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['Set'])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, row in summary_df.iterrows():
            for j, (col, offset) in enumerate(zip(['Total Genotype', 'Total Covariates', 'Total Expression', 'Total Final', 'Total Samples'],
                                                  [-2*width, -width, 0, width, 2*width])):
                val = row[col]
                if val > 0:
                    ax.text(i + offset, val, f'{int(val):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overall_summary.pdf'), transparent=True, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: overall_summary.pdf")
    
    # ========================================================================
    # Figure 4: Overall summary split by cohort
    # ========================================================================
    cohort_summary = []
    for set_name in ['train', 'test']:
        set_stats = stats_df[stats_df['set'] == set_name].copy()
        if len(set_stats) == 0:
            continue
        
        for cohort_name, grp in set_stats.groupby('cohort'):
            if set_name == "test":
                cohort_name = cohort_name + "_DLPFC"
                assert cohort_name == "ROSMAP_DLPFC", "Cohort name is not ROSMAP_DLPFC"
            cohort_summary.append({
                'Cohort': cohort_name,
                'Total Genotype': grp['n_genotype'].sum(),
                'Total Covariates': grp['n_covariates'].sum(),
                'Total Expression': grp['n_expression'].sum(),
                'Total Final': grp['n_final'].sum(),
                'Total Samples': grp['n_samples_final'].sum()
            })
    
    if cohort_summary:
        cohort_summary_df = pd.DataFrame(cohort_summary)
        x = np.arange(len(cohort_summary_df))
        width = 0.18
        # Scale width of figure with number of cohorts to keep labels readable and allow wider bars
        fig_width_cohort = max(14, len(cohort_summary_df) * 1.8)
        fig, ax = plt.subplots(figsize=(fig_width_cohort, 7))
        
        ax.bar(x - 2*width, cohort_summary_df['Total Genotype'], width, label='Genotype', alpha=0.8, color='#3498db')
        ax.bar(x - width, cohort_summary_df['Total Covariates'], width, label='Covariates', alpha=0.8, color='#2ecc71')
        ax.bar(x, cohort_summary_df['Total Expression'], width, label='Expression', alpha=0.8, color='#e74c3c')
        ax.bar(x + width, cohort_summary_df['Total Final'], width, label='Final Participants', alpha=0.8, color='#f39c12', edgecolor='black', linewidth=2)
        ax.bar(x + 2*width, cohort_summary_df['Total Samples'], width, label='Final Samples', alpha=0.8, color='#9b59b6', edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cohort', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Overall Summary: By Cohort', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_summary_df['Cohort'], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, row in cohort_summary_df.iterrows():
            for col, offset in zip(['Total Genotype', 'Total Covariates', 'Total Expression', 'Total Final', 'Total Samples'],
                                   [-2*width, -width, 0, width, 2*width]):
                val = row[col]
                if val > 0:
                    ax.text(i + offset, val, f'{int(val):,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'overall_summary_by_cohort.pdf'), transparent=True, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: overall_summary_by_cohort.pdf")
  
  

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load file sheet
file_sheet = pd.read_csv(FILE_SHEET, sep="\t")

# Build cohort mappings
COHORT_TO_GENOTYPE_FILE = {}
COHORT_TO_EXPRESSION_FILE = {}

for cohort in file_sheet["Cohort"].unique():
    if pd.notna(cohort):
        norm_cohort = normalize_cohort_name(cohort)
        
        # Get genotype files
        geno_files = file_sheet[file_sheet["Cohort"] == cohort]["Genotype file"].dropna().unique().tolist()
        geno_files = [os.path.join(BASE_DIR, f) for f in geno_files if f and f.strip()]
        if geno_files:
            COHORT_TO_GENOTYPE_FILE[norm_cohort] = geno_files
        
        # Get expression files
        expr_files = file_sheet[file_sheet["Cohort"] == cohort]["gene expression file not normalized (All ancestries)"].dropna().unique().tolist()
        processed_expr_files = []
        for f in expr_files:
            if f and f.strip():
                if os.path.isabs(f):
                    processed_expr_files.append(f)
                else:
                    processed_expr_files.append(os.path.join(EXPRESSION_DIR, f.split("/")[-1]))
        if processed_expr_files:
            COHORT_TO_EXPRESSION_FILE[norm_cohort] = processed_expr_files

# Load covariates
covariates_train = pd.read_csv(COVARIATE_FILE_TRAIN, sep="\t")
covariates_test = pd.read_csv(COVARIATE_FILE_TEST, sep="\t")

print(f"Train covariates: {len(covariates_train):,} samples, {covariates_train['participant_id'].nunique():,} participants")
print(f"Test covariates: {len(covariates_test):,} samples, {covariates_test['participant_id'].nunique():,} participants")

# Create sample_id to participant_id mappings
sample_to_participant_train = dict(zip(
    covariates_train['sample_id'].astype(str),
    covariates_train['participant_id'].astype(str)
))
sample_to_participant_test = dict(zip(
    covariates_test['sample_id'].astype(str),
    covariates_test['participant_id'].astype(str)
))

# ============================================================================
# PROCESS TRAIN SET
# ============================================================================

print("\n" + "=" * 80)
print("PROCESSING TRAIN SET")
print("=" * 80)

train_cohorts = sorted(covariates_train['cohort'].unique())
print(f"Train cohorts: {train_cohorts}")

train_final_covariates = []

train_missing_genotype_participants_cov = dict()
train_missing_genotype_participants_expr = dict()

for cohort in train_cohorts:
    print(f"\n--- Processing {cohort} ---")
    
    if cohort not in COHORT_TO_GENOTYPE_FILE:
        print(f"  ⚠ No genotype file found for {cohort}")
        continue
    
    genotype_path = COHORT_TO_GENOTYPE_FILE[cohort][0]
    print(f"  Genotype file: {genotype_path}")
    
    # Step 1: Get participant IDs from genotype
    try:
        geno_participant_ids = get_genotype_participant_ids(genotype_path)
        print(f"  Genotype participants: {len(geno_participant_ids):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    
    # Step 2: Get participant IDs from covariates
    cohort_cov = covariates_train[covariates_train['cohort'] == cohort].copy()
    cov_participant_ids = set(cohort_cov['participant_id'].astype(str))
    print(f"  Covariates participants: {len(cov_participant_ids):,}")
    
    # Save cov participant ids with missing genotypes
    missing_genotype_participants_cov = cov_participant_ids - geno_participant_ids
    if missing_genotype_participants_cov:        # Convert set to sorted list for JSON serialization
        train_missing_genotype_participants_cov[cohort] = sorted(list(missing_genotype_participants_cov))
        print(f"  Missing genotype participants in covariates for {cohort}: {len(missing_genotype_participants_cov):,}")
    
    
    # Step 3: Get participant IDs from expression files
    expr_participant_ids = None
    has_expression = False
    if cohort in COHORT_TO_EXPRESSION_FILE:
        all_expr_participants = set()
        for expr_file in COHORT_TO_EXPRESSION_FILE[cohort]:
            try:
                expr_pids, expr_sids = get_expression_participant_ids(expr_file, sample_to_participant_train)
                all_expr_participants.update(expr_pids)
                print(f"  Expression file {os.path.basename(expr_file)}: {len(expr_pids):,} participants")
            except Exception as e:
                print(f"  ⚠ Error reading expression file {os.path.basename(expr_file)}: {e}")
        
        if all_expr_participants:
            expr_participant_ids = all_expr_participants
            has_expression = True
            print(f"  Expression participants (total): {len(expr_participant_ids):,}")
        else:
            print(f"  ⚠ WARNING: Expression files exist but no participants could be extracted!")
            print(f"    This might indicate a mapping issue. Will use genotype ∩ covariates only.")
    else:
        print(f"  ⚠ No expression files found for {cohort}")
    
    # Save epxr participant ids with missing genotypes
    missing_genotype_participants_expr = expr_participant_ids - geno_participant_ids
    if missing_genotype_participants_expr:        # Convert set to sorted list for JSON serialization
        train_missing_genotype_participants_expr[cohort] = sorted(list(missing_genotype_participants_expr))
        print(f"  Missing genotype participants in expression for {cohort}: {len(missing_genotype_participants_expr):,}")
    
    # Check if missing set is the same for both covariates and expression
    if missing_genotype_participants_cov != missing_genotype_participants_expr:
        print(f"  ⚠ WARNING: Missing genotype participants in covariates and expression are not the same for {cohort}")
        print(f"    Covariates: {len(missing_genotype_participants_cov):,}")
        print(f"    Expression: {len(missing_genotype_participants_expr):,}")


    # Step 4: Find intersection
    if expr_participant_ids is not None:
        # Intersection of all three
        final_participant_ids = geno_participant_ids & cov_participant_ids & expr_participant_ids
        print(f"  Intersection (genotype ∩ covariates ∩ expression): {len(final_participant_ids):,}")
    else:
        # Intersection of genotype and covariates only
        final_participant_ids = geno_participant_ids & cov_participant_ids
        print(f"  Intersection (genotype ∩ covariates): {len(final_participant_ids):,}")
    
    if len(final_participant_ids) == 0:
        print(f"  ⚠ WARNING: No overlapping participants! Skipping.")
        continue
    
    # Step 5: Subset covariates to final participants
    final_covariates = cohort_cov[cohort_cov['participant_id'].isin(final_participant_ids)].copy()
    
    # Verification: Check if all final participants have expression data
    if expr_participant_ids is not None:
        final_participants_set = set(final_covariates['participant_id'].astype(str))
        missing_expr = final_participants_set - expr_participant_ids
        if missing_expr:
            print(f"  ⚠ WARNING: {len(missing_expr)} participants in final covariates don't have expression data!")
            print(f"    This suggests the intersection may not have worked correctly.")
            print(f"    Example participants: {list(missing_expr)[:5]}")
        else:
            print(f"  ✓ Verified: All {len(final_participants_set)} participants have expression data")
    
    train_final_covariates.append(final_covariates)
    print(f"  Final covariates: {len(final_covariates):,} samples, {final_covariates['participant_id'].nunique():,} participants")
    
    # Track statistics
    stats_key = f"train_{cohort}"
    stats[stats_key] = {
        'cohort': cohort,
        'set': 'train',
        'n_genotype': len(geno_participant_ids),
        'n_covariates': len(cov_participant_ids),
        'n_expression': len(expr_participant_ids) if expr_participant_ids is not None else 0,
        'n_final': len(final_participant_ids),
        'n_samples_final': len(final_covariates),
        'has_expression': has_expression
    }
    
    # Step 6: Create keep file for this cohort
    keep_file = os.path.join(GENOTYPE_SUBSET_DIR, f"{cohort}_keep_participants.txt")
    with open(keep_file, 'w') as f:
        # PLINK keep format: FID IID (one per line)
        for participant_id in sorted(final_participant_ids):
            f.write(f"0\t{participant_id}\n")
    print(f"  ✓ Saved keep file: {keep_file} ({len(final_participant_ids):,} participants)")

# Save missing genotype participants in covariates
if train_missing_genotype_participants_cov:
    output_file = os.path.join(OUTPUT_DIR_TRAIN, "missing_genotype_participants_cov.json")
    with open(output_file, 'w') as f:
        json.dump(train_missing_genotype_participants_cov, f)
    print(f"  ✓ Saved missing genotype participants in covariates: {output_file} ({len(train_missing_genotype_participants_cov):,} participants)")

# Save missing genotype participants in expression
if train_missing_genotype_participants_expr:
    output_file = os.path.join(OUTPUT_DIR_TRAIN, "missing_genotype_participants_expr.json")
    with open(output_file, 'w') as f:
        json.dump(train_missing_genotype_participants_expr, f)
    print(f"  ✓ Saved missing genotype participants in expression: {output_file} ({len(train_missing_genotype_participants_expr):,} participants)")    

# Save train covariates
if train_final_covariates:
    train_covariates_combined = pd.concat(train_final_covariates, ignore_index=True)
    output_cov_file = os.path.join(OUTPUT_DIR_TRAIN, "covariates.tsv")
    train_covariates_combined.to_csv(output_cov_file, sep="\t", index=False)
    print(f"\n✓ Saved train covariates: {len(train_covariates_combined):,} samples, {train_covariates_combined['participant_id'].nunique():,} participants")
    print(f"  File: {output_cov_file}")

# ============================================================================
# PROCESS TEST SET (ROSMAP DLPFC only)
# ============================================================================

print("\n" + "=" * 80)
print("PROCESSING TEST SET (ROSMAP DLPFC)")
print("=" * 80)

cohort = "ROSMAP"  # Test set only has ROSMAP DLPFC
print(f"\n--- Processing {cohort} (DLPFC) ---")

if cohort not in COHORT_TO_GENOTYPE_FILE:
    print(f"  ⚠ No genotype file found for {cohort}")
else:
    genotype_path = COHORT_TO_GENOTYPE_FILE[cohort][0]
    print(f"  Genotype file: {genotype_path}")
    
    # Step 1: Get participant IDs from genotype
    try:
        geno_participant_ids = get_genotype_participant_ids(genotype_path)
        print(f"  Genotype participants: {len(geno_participant_ids):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        geno_participant_ids = None
    
    if geno_participant_ids is not None:
        # Step 2: Get participant IDs from covariates (test set only)
        cohort_cov = covariates_test[covariates_test['cohort'] == cohort].copy()
        cov_participant_ids = set(cohort_cov['participant_id'].astype(str))
        print(f"  Covariates participants: {len(cov_participant_ids):,}")

        # Step 3: Get participant IDs from expression files
        # Add ROSMAP DLPFC expression file if not already in mapping
        rosmap_dlpfc_expr = os.path.join(EXPRESSION_DIR, "ROSMAP_genes_tpm.tsv")
        if cohort not in COHORT_TO_EXPRESSION_FILE and os.path.exists(rosmap_dlpfc_expr):
            COHORT_TO_EXPRESSION_FILE[cohort] = [rosmap_dlpfc_expr]
        
        expr_participant_ids = None
        has_expression = False
        if cohort in COHORT_TO_EXPRESSION_FILE:
            all_expr_participants = set()
            for expr_file in COHORT_TO_EXPRESSION_FILE[cohort]:
                try:
                    expr_pids, expr_sids = get_expression_participant_ids(expr_file, sample_to_participant_test)
                    all_expr_participants.update(expr_pids)
                    print(f"  Expression file {os.path.basename(expr_file)}: {len(expr_pids):,} participants")
                except Exception as e:
                    print(f"  ⚠ Error reading expression file {os.path.basename(expr_file)}: {e}")
            
            if all_expr_participants:
                expr_participant_ids = all_expr_participants
                has_expression = True
                print(f"  Expression participants (total): {len(expr_participant_ids):,}")
        else:
            print(f"  ⚠ No expression files found for {cohort}")
        
        # Save participant ids with missing genotypes
        test_missing_genotype_participants_cov = []
        test_missing_genotype_participants_expr = []

        missing_genotype_participants_cov = cov_participant_ids - geno_participant_ids
        missing_genotype_participants_expr = expr_participant_ids - geno_participant_ids
        if missing_genotype_participants_cov:
            # Convert set to sorted list for JSON serialization
            test_missing_genotype_participants_cov = sorted(list(missing_genotype_participants_cov))
            print(f"  Missing genotype participants in covariates for {cohort}: {len(test_missing_genotype_participants_cov):,}")
        if missing_genotype_participants_expr:
            # Convert set to sorted list for JSON serialization
            test_missing_genotype_participants_expr = sorted(list(missing_genotype_participants_expr))
            print(f"  Missing genotype participants in expression for {cohort}: {len(test_missing_genotype_participants_expr):,}")
        
        # Check if missing set is the same for both covariates and expression
        if missing_genotype_participants_cov != missing_genotype_participants_expr:
            print(f"  ⚠ WARNING: Missing genotype participants in covariates and expression are not the same for {cohort}")
            print(f"    Covariates: {len(missing_genotype_participants_cov):,}")
            print(f"    Expression: {len(missing_genotype_participants_expr):,}")
        
        
        # Step 4: Find intersection
        if expr_participant_ids is not None:
            # Intersection of all three
            final_participant_ids = geno_participant_ids & cov_participant_ids & expr_participant_ids
            print(f"  Intersection (genotype ∩ covariates ∩ expression): {len(final_participant_ids):,}")
        else:
            # Intersection of genotype and covariates only
            final_participant_ids = geno_participant_ids & cov_participant_ids
            print(f"  Intersection (genotype ∩ covariates): {len(final_participant_ids):,}")
        
        if len(final_participant_ids) == 0:
            print(f"  ⚠ WARNING: No overlapping participants! Skipping.")
        else:
            # Step 5: Subset covariates to final participants
            final_covariates = cohort_cov[cohort_cov['participant_id'].isin(final_participant_ids)].copy()
            
            # Verification: Check if all final participants have expression data
            if expr_participant_ids is not None:
                final_participants_set = set(final_covariates['participant_id'].astype(str))
                missing_expr = final_participants_set - expr_participant_ids
                if missing_expr:
                    print(f"  ⚠ WARNING: {len(missing_expr)} participants in final covariates don't have expression data!")
                    print(f"    This suggests the intersection may not have worked correctly.")
                    print(f"    Example participants: {list(missing_expr)[:5]}")
                else:
                    print(f"  ✓ Verified: All {len(final_participants_set)} participants have expression data")
            
            print(f"  Final covariates: {len(final_covariates):,} samples, {final_covariates['participant_id'].nunique():,} participants")
            
            # Track statistics
            stats_key = f"test_{cohort}"
            stats[stats_key] = {
                'cohort': cohort,
                'set': 'test',
                'n_genotype': len(geno_participant_ids),
                'n_covariates': len(cov_participant_ids),
                'n_expression': len(expr_participant_ids) if expr_participant_ids is not None else 0,
                'n_final': len(final_participant_ids),
                'n_samples_final': len(final_covariates),
                'has_expression': has_expression
            }
            
            # Step 6: Create keep file (use ROSMAP_DLPFC to distinguish from train set)
            keep_file = os.path.join(GENOTYPE_SUBSET_DIR, "ROSMAP_DLPFC_keep_participants.txt")
            with open(keep_file, 'w') as f:
                # PLINK keep format: FID IID (one per line)
                for participant_id in sorted(final_participant_ids):
                    f.write(f"0\t{participant_id}\n")
            print(f"  ✓ Saved keep file: {keep_file} ({len(final_participant_ids):,} participants)")
            
            # Save test covariates
            output_cov_file = os.path.join(OUTPUT_DIR_TEST, "covariates.tsv")
            final_covariates.to_csv(output_cov_file, sep="\t", index=False)
            print(f"\n✓ Saved test covariates: {len(final_covariates):,} samples, {final_covariates['participant_id'].nunique():,} participants")
            print(f"  File: {output_cov_file}")
        
        # Save missing genotype participants
        if test_missing_genotype_participants_cov:
            output_file = os.path.join(OUTPUT_DIR_TEST, "missing_genotype_participants_cov.json")
            with open(output_file, 'w') as f:
                json.dump(test_missing_genotype_participants_cov, f)
            print(f"  ✓ Saved missing genotype participants in covariates: {output_file} ({len(test_missing_genotype_participants_cov):,} participants)")
        if test_missing_genotype_participants_expr:
            output_file = os.path.join(OUTPUT_DIR_TEST, "missing_genotype_participants_expr.json")
            with open(output_file, 'w') as f:
                json.dump(test_missing_genotype_participants_expr, f)
            print(f"  ✓ Saved missing genotype participants in expression: {output_file} ({len(test_missing_genotype_participants_expr):,} participants)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nTrain set:")
if train_final_covariates:
    train_cov_combined = pd.concat(train_final_covariates, ignore_index=True)
    print(f"  Total samples: {len(train_cov_combined):,}")
    print(f"  Total participants: {train_cov_combined['participant_id'].nunique():,}")
    print(f"  Cohorts: {sorted(train_cov_combined['cohort'].unique())}")

print(f"\nTest set:")
test_cov_file = os.path.join(OUTPUT_DIR_TEST, "covariates.tsv")
if os.path.exists(test_cov_file):
    test_cov_combined = pd.read_csv(test_cov_file, sep="\t")
    print(f"  Total samples: {len(test_cov_combined):,}")
    print(f"  Total participants: {test_cov_combined['participant_id'].nunique():,}")
    print(f"  Cohorts: {sorted(test_cov_combined['cohort'].unique())}")
else:
    print(f"  No test covariates file found")

print(f"\nKeep files saved to: {GENOTYPE_SUBSET_DIR}")
keep_files = [f for f in os.listdir(GENOTYPE_SUBSET_DIR) if f.endswith('_keep_participants.txt')]
print(f"  {len(keep_files)} keep files created")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print(f"Figures will be saved to: {FIGURES_DIR}")

try:
    create_filtering_visualizations(stats, FIGURES_DIR)
    print("\n✓ All visualizations generated successfully!")
except Exception as e:
    print(f"\n⚠ Error generating visualizations: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)

