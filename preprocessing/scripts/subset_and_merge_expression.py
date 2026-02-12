#!/usr/bin/env python3
"""
Subset and merge expression matrices based on sample IDs in covariates files.

This script:
1. For each cohort, gets sample IDs from covariates
2. Subsets expression matrices to only include those sample IDs (columns)
3. Merges all subsetted expression matrices
4. Saves final merged expression matrices as tpm.tsv
"""

import pandas as pd
import os

# ============================================================================
# HARDCODED PATHS
# ============================================================================
FILE_SHEET = "/gpfs/commons/home/vmazeeva/BigBrain_files_sheet.tsv"
BASE_DIR = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain"
EXPRESSION_DIR = os.path.join(BASE_DIR, "Phenotypes_TPM")
COVARIATE_FILE = os.path.join(BASE_DIR, "Processed", "covariates.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SUBSET AND MERGE EXPRESSION MATRICES")
print("=" * 80)
print(f"Output: {OUTPUT_DIR}")

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

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load file sheet
file_sheet = pd.read_csv(FILE_SHEET, sep="\t")

# Build cohort to expression file mapping
COHORT_TO_EXPRESSION_FILE = {}

for cohort in file_sheet["Cohort"].unique():
    if pd.notna(cohort):
        norm_cohort = normalize_cohort_name(cohort)
        
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
covariates = pd.read_csv(COVARIATE_FILE, sep="\t")

print(f"Covariates: {len(covariates):,} samples, {covariates['participant_id'].nunique():,} participants")

# Create sample_id to participant_id mappings
sample_to_participant = dict(zip(
    covariates['sample_id'].astype(str),
    covariates['participant_id'].astype(str)
))

# ============================================================================
# PROCESS DATASET
# ============================================================================

print("\n" + "=" * 80)
print("PROCESSING DATASET")
print("=" * 80)

cohorts = sorted(covariates['cohort'].unique())
print(f"Cohorts: {cohorts}")

expression_matrices = []

for cohort in cohorts:
    print(f"\n--- Processing {cohort} ---")
    
    # Step 1: Get sample IDs from covariates
    cohort_cov = covariates[covariates['cohort'] == cohort].copy()
    cov_sample_ids = set(cohort_cov['sample_id'].astype(str))
    cov_participant_ids = set(cohort_cov['participant_id'].astype(str))
    
    print(f"  Covariates: {len(cov_sample_ids):,} samples, {len(cov_participant_ids):,} participants")
    
    # Step 2: Get expression file for this cohort
    if cohort not in COHORT_TO_EXPRESSION_FILE:
        print(f"  ⚠ No expression file found for {cohort}, skipping")
        continue
    
    expr_files = COHORT_TO_EXPRESSION_FILE[cohort]
    
    # Process each expression file (some cohorts may have multiple files)
    for expr_file in expr_files:
        if not os.path.exists(expr_file):
            print(f"  ⚠ Expression file not found: {expr_file}, skipping")
            continue
        
        print(f"  Expression file: {os.path.basename(expr_file)}")
        
        try:
            # Load expression matrix
            # Note: expression matrices have genes as rows, samples as columns
            expr_df = pd.read_csv(expr_file, sep="\t", index_col=0)
            
            # Get sample IDs from expression matrix (column names)
            expr_sample_ids = set(expr_df.columns.astype(str))
            print(f"  Full expression matrix: {expr_df.shape[0]:,} genes, {len(expr_sample_ids):,} samples")
            
            # Map expression samples to participants
            expr_participant_ids = set()
            for sample_id in expr_sample_ids:
                if sample_id in sample_to_participant:
                    expr_participant_ids.add(sample_to_participant[sample_id])
                elif sample_id in set(sample_to_participant.values()):
                    # Sample ID is already a participant ID
                    expr_participant_ids.add(sample_id)
            
            print(f"  Expression participants (from sample mapping): {len(expr_participant_ids):,}")
            
            # Step 3: Find intersection of sample IDs
            common_sample_ids = cov_sample_ids & expr_sample_ids
            print(f"  Common samples (covariates ∩ expression): {len(common_sample_ids):,}")
            
            if len(common_sample_ids) == 0:
                print(f"  ⚠ WARNING: No common samples! Skipping this expression file.")
                continue
            
            # Verify participants match
            common_participants_from_samples = set()
            for sample_id in common_sample_ids:
                if sample_id in sample_to_participant:
                    common_participants_from_samples.add(sample_to_participant[sample_id])
                elif sample_id in set(sample_to_participant.values()):
                    common_participants_from_samples.add(sample_id)
            
            print(f"  Participants from common samples: {len(common_participants_from_samples):,}")
            print(f"  Participants in covariates: {len(cov_participant_ids):,}")
            
            if common_participants_from_samples != cov_participant_ids:
                missing = cov_participant_ids - common_participants_from_samples
                extra = common_participants_from_samples - cov_participant_ids
                if missing:
                    print(f"  ⚠ NOTE: {len(missing)} participants in covariates but not in THIS expression file")
                    print(f"    (This is expected if cohort has multiple expression files - check final merge for complete picture)")
                    # Find which samples belong to missing participants
                    missing_samples = cohort_cov[cohort_cov['participant_id'].isin(missing)]['sample_id'].tolist()
                    print(f"    These participants have {len(missing_samples)} samples in covariates but none in this file")
                    # Check if any of these sample IDs exist in expression but weren't matched
                    missing_in_expr = [s for s in missing_samples if str(s) in expr_sample_ids]
                    if missing_in_expr:
                        print(f"    ⚠ NOTE: {len(missing_in_expr)} of these sample IDs actually exist in expression but weren't matched!")
                        print(f"    This suggests a sample ID mismatch (e.g., formatting differences)")
                        print(f"    Example mismatched sample IDs: {list(missing_in_expr)[:5]}")
                if extra:
                    print(f"  ⚠ WARNING: {len(extra)} participants in expression samples but not in covariates")
            
            # Step 4: Subset expression matrix
            # Only keep columns (samples) that are in common_sample_ids
            subset_expr_df = expr_df[[col for col in expr_df.columns if str(col) in common_sample_ids]].copy()
            
            print(f"  Subsetted expression matrix: {subset_expr_df.shape[0]:,} genes, {subset_expr_df.shape[1]:,} samples")
            
            # Verify no samples were lost
            if subset_expr_df.shape[1] != len(common_sample_ids):
                print(f"  ⚠ WARNING: Expected {len(common_sample_ids)} samples, got {subset_expr_df.shape[1]}")
            
            # Add cohort identifier to column names to avoid conflicts when merging
            # We'll keep original sample IDs for now, but add a prefix if needed during merge
            expression_matrices.append(subset_expr_df)
            print(f"  ✓ Subsetted expression matrix added (will merge later)")
            
        except Exception as e:
            print(f"  ERROR loading/subsetting expression file: {e}")
            import traceback
            traceback.print_exc()
            continue

# Step 5: Merge all expression matrices
print(f"\n--- Merging expression matrices ---")
if expression_matrices:
    print(f"  Number of expression matrices to merge: {len(expression_matrices)}")
    
    # Check for overlapping genes (should be the same across all)
    all_genes = set()
    for expr_df in expression_matrices:
        all_genes.update(expr_df.index)
    
    print(f"  Total unique genes across all matrices: {len(all_genes):,}")
    
    # Merge on gene index (outer join to keep all genes)
    merged_expr = expression_matrices[0]
    for i, expr_df in enumerate(expression_matrices[1:], 1):
        print(f"  Merging matrix {i+1}/{len(expression_matrices)}...")
        # Check for duplicate column names
        overlapping_cols = set(merged_expr.columns) & set(expr_df.columns)
        if overlapping_cols:
            print(f"    ⚠ WARNING: {len(overlapping_cols)} overlapping sample IDs found, will keep first occurrence")
        
        merged_expr = pd.concat([merged_expr, expr_df], axis=1, join='outer', sort=False)
        print(f"    After merge: {merged_expr.shape[0]:,} genes, {merged_expr.shape[1]:,} samples")
    
    # Fill NaN values with 0 (for genes not present in all cohorts)
    merged_expr = merged_expr.fillna(0)
    
    print(f"\n  Final merged expression matrix: {merged_expr.shape[0]:,} genes, {merged_expr.shape[1]:,} samples")
    
    # Verify sample counts match covariates
    merged_sample_ids = set(merged_expr.columns.astype(str))
    cov_sample_ids_all = set(covariates['sample_id'].astype(str))
    
    print(f"  Samples in merged expression: {len(merged_sample_ids):,}")
    print(f"  Samples in covariates: {len(cov_sample_ids_all):,}")
    
    if merged_sample_ids != cov_sample_ids_all:
        missing = cov_sample_ids_all - merged_sample_ids
        extra = merged_sample_ids - cov_sample_ids_all
        if missing:
            print(f"  ⚠ WARNING: {len(missing)} samples in covariates but not in merged expression")
        if extra:
            print(f"  ⚠ WARNING: {len(extra)} samples in merged expression but not in covariates")
    else:
        print(f"  ✓ All samples match between covariates and merged expression!")
    
    # Verify participant counts
    merged_participants = set()
    for sample_id in merged_sample_ids:
        if sample_id in sample_to_participant:
            merged_participants.add(sample_to_participant[sample_id])
        elif sample_id in set(sample_to_participant.values()):
            merged_participants.add(sample_id)
    
    cov_participants_all = set(covariates['participant_id'].astype(str))
    print(f"  Participants in merged expression: {len(merged_participants):,}")
    print(f"  Participants in covariates: {len(cov_participants_all):,}")
    
    if merged_participants != cov_participants_all:
        missing = cov_participants_all - merged_participants
        extra = merged_participants - cov_participants_all
        if missing:
            print(f"  ⚠ WARNING: {len(missing)} participants in covariates but not in merged expression")
        if extra:
            print(f"  ⚠ WARNING: {len(extra)} participants in merged expression but not in covariates")
    else:
        print(f"  ✓ All participants match between covariates and merged expression!")
    
    # Step 6: Save merged expression matrix
    output_file = os.path.join(OUTPUT_DIR, "tpm.tsv")
    merged_expr.to_csv(output_file, sep="\t")
    print(f"\n  ✓ Saved merged expression matrix: {output_file}")
else:
    print("  ⚠ No expression matrices to merge")



# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nDataset:")
print(f"  Covariates: {len(covariates):,} samples, {covariates['participant_id'].nunique():,} participants")
if expression_matrices:
    output_file = os.path.join(OUTPUT_DIR, "tpm.tsv")
    if os.path.exists(output_file):
        expr = pd.read_csv(output_file, sep="\t", index_col=0)
        print(f"  Expression: {expr.shape[0]:,} genes, {expr.shape[1]:,} samples")
        expr_samples = set(expr.columns.astype(str))
        expr_participants = set()
        for sample_id in expr_samples:
            if sample_id in sample_to_participant:
                expr_participants.add(sample_to_participant[sample_id])
            elif sample_id in set(sample_to_participant.values()):
                expr_participants.add(sample_id)
        print(f"  Expression participants: {len(expr_participants):,}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)

