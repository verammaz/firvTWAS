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
COVARIATE_FILE = os.path.join(BASE_DIR, "All_combined_covariates.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed")
GENOTYPE_SUBSET_DIR = os.path.join(GENOTYPE_DIR, "subset")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENOTYPE_SUBSET_DIR, exist_ok=True)

# Create figures directory
FIGURES_DIR = "/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/figures"
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
print(f"Processed output: {OUTPUT_DIR}")
print(f"Keep files dir: {GENOTYPE_SUBSET_DIR}")


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
    
    if unmapped_samples:
        print(f"    ⚠ WARNING: {len(unmapped_samples)} expression sample IDs could not be mapped to participants")
        print(f"    Example unmapped sample IDs: {unmapped_samples[:5]}")
    
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
  
    cohorts = stats_df['cohort'].values
    x = np.arange(len(cohorts))
    width = 0.2
    
    # Get counts
    geno_counts = stats_df['n_genotype'].values
    cov_counts = stats_df['n_covariates'].values
    expr_counts = stats_df['n_expression'].values
    final_counts = stats_df['n_final'].values
    
    # Plot bars
    ax.bar(x - 1.5*width, geno_counts, width, label='Genotype Participants', alpha=0.8, color='#3498db')
    ax.bar(x - 0.5*width, cov_counts, width, label='Covariates Participants', alpha=0.8, color='#2ecc71')
    ax.bar(x + 0.5*width, expr_counts, width, label='Expression Participants', alpha=0.8, color='#e74c3c')
    ax.bar(x + 1.5*width, final_counts, width, label='Final (Intersection)', alpha=0.8, color='#f39c12', edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Cohort', fontsize=12)
    ax.set_ylabel('Number of Participants', fontsize=12)
    ax.set_title(f'Participant Counts by Cohort', 
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
    fig.savefig(os.path.join(output_dir, 'participant_counts_by_cohort.pdf'), transparent=True, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: participant_counts_by_cohort.pdf")
    
    # ========================================================================
    # Figure 2: Filtering efficiency (what was kept vs filtered out)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(fig_width, 6))
 
    cohorts = stats_df['cohort'].values
    x = np.arange(len(cohorts))
    
    # Calculate what was kept (final) vs filtered out
    # Use covariates as the "starting point" since that's what we're filtering
    starting_counts = []
    kept_counts = []
    for _, row in stats_df.iterrows():
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
    ax.set_title(f'Filtering Efficiency\n(Based on Covariates)', fontsize=14, fontweight='bold')
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
    # Figure 3: Overall summary (all cohorts combined)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    summary_data = [{
        'Total Genotype': stats_df['n_genotype'].sum(),
        'Total Covariates': stats_df['n_covariates'].sum(),
        'Total Expression': stats_df['n_expression'].sum(),
        'Total Final': stats_df['n_final'].sum(),
        'Total Samples': stats_df['n_samples_final'].sum()
    }]
    
    print(f"Summary data: {summary_data}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        width = 0.5
        
        ax.bar(1, summary_df['Total Genotype'], width, label='Genotype', alpha=0.8, color='#3498db')
        ax.bar(2, summary_df['Total Covariates'], width, label='Covariates', alpha=0.8, color='#2ecc71')
        ax.bar(3, summary_df['Total Expression'], width, label='Expression', alpha=0.8, color='#e74c3c')
        ax.bar(4, summary_df['Total Final'], width, label='Final Participants', alpha=0.8, color='#f39c12', edgecolor='black', linewidth=2)
        ax.bar(5, summary_df['Total Samples'], width, label='Final Samples', alpha=0.8, color='#9b59b6', edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Overall Summary: All Cohorts Combined', fontsize=14, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['Genotype Participants', 'Covariates Participants', 'Expression Participants', 'Final Participants', 'Final Samples'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        positions = [1, 2, 3, 4, 5]
        values = [summary_df['Total Genotype'].iloc[0], summary_df['Total Covariates'].iloc[0], 
                  summary_df['Total Expression'].iloc[0], summary_df['Total Final'].iloc[0], 
                  summary_df['Total Samples'].iloc[0]]
        for pos, val in zip(positions, values):
            if val > 0:
                ax.text(pos, val, f'{int(val):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overall_summary.pdf'), transparent=True, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: overall_summary.pdf")
    
    # ========================================================================
    # Figure 4: Overall summary split by cohort
    # ========================================================================
    cohort_summary = []
    for cohort in cohorts:
        cohort_summary.append({
            'Cohort': cohort,
            'Total Genotype': stats_df[stats_df['cohort'] == cohort]['n_genotype'].sum(),
            'Total Covariates': stats_df[stats_df['cohort'] == cohort]['n_covariates'].sum(),
            'Total Expression': stats_df[stats_df['cohort'] == cohort]['n_expression'].sum(),
            'Total Final': stats_df[stats_df['cohort'] == cohort]['n_final'].sum(),
            'Total Samples': stats_df[stats_df['cohort'] == cohort]['n_samples_final'].sum()
        })
    
    if cohort_summary:
        cohort_summary_df = pd.DataFrame(cohort_summary)
        x = np.arange(len(cohort_summary_df))
        width = 0.18
        # Scale width of figure with number of cohorts to keep labels readable and allow wider bars
        fig_width_cohort = max(14, len(cohort_summary_df) * 1.8)
        fig, ax = plt.subplots(figsize=(fig_width_cohort, 7))
        
        ax.bar(x - 2*width, cohort_summary_df['Total Genotype'], width, label='Genotype Participants', alpha=0.8, color='#3498db')
        ax.bar(x - width, cohort_summary_df['Total Covariates'], width, label='Covariates Participants', alpha=0.8, color='#2ecc71')
        ax.bar(x, cohort_summary_df['Total Expression'], width, label='Expression Participants', alpha=0.8, color='#e74c3c')
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

def normalize_cohort_name(cohort):
    if cohort == "NYGC ALS":
        return "NYGC"
    elif cohort == "Mayo Clinic":
        return "Mayo"
    else:
        return cohort
# Load file sheet
file_sheet = pd.read_csv(FILE_SHEET, sep="\t")

# Cohorts
COHORTS = []

# Build cohort mappings
COHORT_TO_GENOTYPE_FILE = {}
COHORT_TO_EXPRESSION_FILE = {}
NORMALIZED_TO_ORIGINAL_COHORT = {}  # Map normalized names to original names for filtering covariates

for cohort in file_sheet["Cohort"].unique():
    if pd.notna(cohort):
        normalized_cohort = normalize_cohort_name(cohort)
        COHORTS.append(normalized_cohort)
        NORMALIZED_TO_ORIGINAL_COHORT[normalized_cohort] = cohort
        geno_files = file_sheet[file_sheet["Cohort"] == cohort]["Genotype file"].dropna().unique().tolist()
        geno_files = [os.path.join(BASE_DIR, f) for f in geno_files if f and f.strip()]
        if geno_files:
            COHORT_TO_GENOTYPE_FILE[normalized_cohort] = geno_files
        expr_files = file_sheet[file_sheet["Cohort"] == cohort]["gene expression file not normalized (All ancestries)"].dropna().unique().tolist()
        processed_expr_files = []
        for f in expr_files:
            if f and f.strip():
                if os.path.isabs(f):
                    processed_expr_files.append(f)
                else:
                    processed_expr_files.append(os.path.join(EXPRESSION_DIR, f.split("/")[-1]))
        if processed_expr_files:
            COHORT_TO_EXPRESSION_FILE[normalized_cohort] = processed_expr_files



# Load covariates
COVARIATE_DATA = pd.read_csv(COVARIATE_FILE, sep="\t")

print(f"Covariates: \n\t{len(COVARIATE_DATA):,} samples, \n\t{COVARIATE_DATA['participant_id'].nunique():,} participants")
print(f"Cohorts: {COHORTS} ({len(COHORTS)} cohorts)")

# Create sample_id to participant_id mappings
sample_to_participant = dict(zip(
    COVARIATE_DATA['sample_id'].astype(str),
    COVARIATE_DATA['participant_id'].astype(str)
))



# ============================================================================
# PROCESS DATA BY COHORT
# ============================================================================

def process_cohort(cohort):
    missing_genotype_participants_cov = dict()
    missing_genotype_participants_expr = dict()
    
    if cohort not in COHORT_TO_GENOTYPE_FILE:
        print(f"  ⚠ No genotype file found for {cohort}")
        return
    
    genotype_path = COHORT_TO_GENOTYPE_FILE[cohort][0]
    print(f"  Genotype file: {genotype_path}")
    
    # Step 1: Get participant IDs from genotype
    try:
        geno_participant_ids = get_genotype_participant_ids(genotype_path)
        print(f"  Genotype participants: {len(geno_participant_ids):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    # Step 2: Get participant IDs from covariates
    cohort_cov = COVARIATE_DATA[COVARIATE_DATA['cohort'] == cohort].copy()
    cov_participant_ids = set(cohort_cov['participant_id'].astype(str))
    print(f"  Covariates participants: {len(cov_participant_ids):,}")
    
    # Save cov participant ids with missing genotypes
    missing_cov_set = cov_participant_ids - geno_participant_ids
    if missing_cov_set:        # Convert set to sorted list for JSON serialization
        missing_genotype_participants_cov[cohort] = sorted(list(missing_cov_set))
        print(f"  Missing genotype participants in covariates for {cohort}: {len(missing_cov_set):,}")
    
    
    # Step 3: Get participant IDs from expression files
    expr_participant_ids = None
    has_expression = False
    if cohort in COHORT_TO_EXPRESSION_FILE:
        all_expr_participants = set()
        for expr_file in COHORT_TO_EXPRESSION_FILE[cohort]:
            try:
                expr_pids, expr_sids = get_expression_participant_ids(expr_file, sample_to_participant)
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
    missing_expr_set = set()
    if expr_participant_ids is not None:
        missing_expr_set = expr_participant_ids - geno_participant_ids
    if missing_expr_set:        # Convert set to sorted list for JSON serialization
        missing_genotype_participants_expr[cohort] = sorted(list(missing_expr_set))
        print(f"  Missing genotype participants in expression for {cohort}: {len(missing_expr_set):,}")
    
    # Check if missing set is the same for both covariates and expression
    if missing_cov_set != missing_expr_set:
        print(f"  ⚠ WARNING: Missing genotype participants in covariates and expression are not the same for {cohort}")
        print(f"    Covariates: {len(missing_cov_set):,}")
        print(f"    Expression: {len(missing_expr_set):,}")

    # Save missing genotype participants in covariates
    if missing_cov_set:
        output_dir = os.path.join(OUTPUT_DIR, "missing")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{cohort}_missing_genotype_participants_cov.json")
        with open(output_file, 'w') as f:
            json.dump(missing_genotype_participants_cov[cohort], f)
        print(f"  ✓ Saved missing genotype participants in covariates: {output_file} ({len(missing_cov_set):,} participants)")

    # Save missing genotype participants in expression
    if missing_expr_set:
        output_dir = os.path.join(OUTPUT_DIR, "missing")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{cohort}_missing_genotype_participants_expr.json")
        with open(output_file, 'w') as f:
            json.dump(missing_genotype_participants_expr[cohort], f)
        print(f"  ✓ Saved missing genotype participants in expression: {output_file} ({len(missing_expr_set):,} participants)")    

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
        return
    
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
    
    FINAL_COVARIATES.append(final_covariates)
    print(f"  Final covariates: {len(final_covariates):,} samples, {final_covariates['participant_id'].nunique():,} participants")
    
    # Track statistics
    stats_key = f"{cohort}"
    STATS[stats_key] = {
        'cohort': cohort,
        'set': 'cohort',
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

# ============================================================================
# PROCESS DATASET
# ============================================================================
print("\n" + "=" * 80)
print("PROCESSING DATASET")
print("=" * 80)


FINAL_COVARIATES = []
STATS = {}

for cohort in COHORTS:
    print(f"\n--- Processing {cohort} ---")
    process_cohort(cohort)
   
    

# Save final covariates
if len(FINAL_COVARIATES) > 0:
    final_covariates_combined = pd.concat(FINAL_COVARIATES, ignore_index=True)
    output_cov_file = os.path.join(OUTPUT_DIR, "covariates.tsv")
    final_covariates_combined.to_csv(output_cov_file, sep="\t", index=False)
    print(f"\n✓ Saved final covariates: {len(final_covariates_combined):,} samples, {final_covariates_combined['participant_id'].nunique():,} participants")
    print(f"  File: {output_cov_file}")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nDataset:")
if FINAL_COVARIATES:
    cov_combined = pd.concat(FINAL_COVARIATES, ignore_index=True)
    print(f"  Total samples: {len(cov_combined):,}")
    print(f"  Total participants: {cov_combined['participant_id'].nunique():,}")
    print(f"  Cohorts: {sorted(cov_combined['cohort'].unique())}")

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
    create_filtering_visualizations(STATS, FIGURES_DIR)
    print("\n✓ All visualizations generated successfully!")
except Exception as e:
    print(f"\n⚠ Error generating visualizations: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)

