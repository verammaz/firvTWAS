#!/usr/bin/env python3
"""
Subset and merge expression matrices based on sample IDs in covariates files.

This script:
1. For each cohort, gets sample IDs from covariates
2. Subsets expression matrices to only include those sample IDs (columns)
3. Merges all subsetted expression matrices for train and test sets
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
COVARIATE_FILE_TRAIN = os.path.join(BASE_DIR, "Processed", "Train", "covariates.tsv")
COVARIATE_FILE_TEST = os.path.join(BASE_DIR, "Processed", "Test", "covariates.tsv")
OUTPUT_DIR_TRAIN = os.path.join(BASE_DIR, "Processed", "Train")
OUTPUT_DIR_TEST = os.path.join(BASE_DIR, "Processed", "Test")

# Create output directories
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)

print("=" * 80)
print("SUBSET AND MERGE EXPRESSION MATRICES")
print("=" * 80)
print(f"Train output: {OUTPUT_DIR_TRAIN}")
print(f"Test output: {OUTPUT_DIR_TEST}")

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

train_expression_matrices = []

for cohort in train_cohorts:
    print(f"\n--- Processing {cohort} ---")
    
    # Step 1: Get sample IDs from covariates
    cohort_cov = covariates_train[covariates_train['cohort'] == cohort].copy()
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
                if sample_id in sample_to_participant_train:
                    expr_participant_ids.add(sample_to_participant_train[sample_id])
                elif sample_id in set(sample_to_participant_train.values()):
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
                if sample_id in sample_to_participant_train:
                    common_participants_from_samples.add(sample_to_participant_train[sample_id])
                elif sample_id in set(sample_to_participant_train.values()):
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
            train_expression_matrices.append(subset_expr_df)
            print(f"  ✓ Subsetted expression matrix added (will merge later)")
            
        except Exception as e:
            print(f"  ERROR loading/subsetting expression file: {e}")
            import traceback
            traceback.print_exc()
            continue

# Step 5: Merge all train expression matrices
print(f"\n--- Merging train expression matrices ---")
if train_expression_matrices:
    print(f"  Number of expression matrices to merge: {len(train_expression_matrices)}")
    
    # Check for overlapping genes (should be the same across all)
    all_genes = set()
    for expr_df in train_expression_matrices:
        all_genes.update(expr_df.index)
    
    print(f"  Total unique genes across all matrices: {len(all_genes):,}")
    
    # Merge on gene index (outer join to keep all genes)
    merged_expr = train_expression_matrices[0]
    for i, expr_df in enumerate(train_expression_matrices[1:], 1):
        print(f"  Merging matrix {i+1}/{len(train_expression_matrices)}...")
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
    cov_sample_ids_all = set(covariates_train['sample_id'].astype(str))
    
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
        if sample_id in sample_to_participant_train:
            merged_participants.add(sample_to_participant_train[sample_id])
        elif sample_id in set(sample_to_participant_train.values()):
            merged_participants.add(sample_id)
    
    cov_participants_all = set(covariates_train['participant_id'].astype(str))
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
    output_file = os.path.join(OUTPUT_DIR_TRAIN, "tpm.tsv")
    merged_expr.to_csv(output_file, sep="\t")
    print(f"\n  ✓ Saved merged train expression matrix: {output_file}")
else:
    print("  ⚠ No expression matrices to merge for train set")

# ============================================================================
# PROCESS TEST SET (ROSMAP DLPFC only)
# ============================================================================

print("\n" + "=" * 80)
print("PROCESSING TEST SET (ROSMAP DLPFC)")
print("=" * 80)

cohort = "ROSMAP"  # Test set only has ROSMAP DLPFC
print(f"\n--- Processing {cohort} (DLPFC) ---")

# Step 1: Get sample IDs from covariates (test set only)
cohort_cov = covariates_test[covariates_test['cohort'] == cohort].copy() # ROSMAP DLPFC only in covariates test file
cov_sample_ids = set(cohort_cov['sample_id'].astype(str))
cov_participant_ids = set(cohort_cov['participant_id'].astype(str))

print(f"  Covariates: {len(cov_sample_ids):,} samples, {len(cov_participant_ids):,} participants")

# Step 2: Get expression file for ROSMAP DLPFC
# Add ROSMAP DLPFC expression file if not already in mapping
rosmap_dlpfc_expr = os.path.join(EXPRESSION_DIR, "ROSMAP_genes_tpm.tsv")
if cohort not in COHORT_TO_EXPRESSION_FILE and os.path.exists(rosmap_dlpfc_expr):
    COHORT_TO_EXPRESSION_FILE[cohort] = [rosmap_dlpfc_expr]

if cohort not in COHORT_TO_EXPRESSION_FILE:
    print(f"  ⚠ No expression file found for {cohort}, skipping")
else:
    expr_files = COHORT_TO_EXPRESSION_FILE[cohort]
    test_expression_matrices = []
    
    # Process each expression file
    for expr_file in expr_files:
        if not os.path.exists(expr_file):
            print(f"  ⚠ Expression file not found: {expr_file}, skipping")
            continue
        
        print(f"  Expression file: {os.path.basename(expr_file)}")
        
        try:
            # Load expression matrix
            expr_df = pd.read_csv(expr_file, sep="\t", index_col=0)
            
            # Get sample IDs from expression matrix (column names)
            expr_sample_ids = set(expr_df.columns.astype(str))
            print(f"  Full expression matrix: {expr_df.shape[0]:,} genes, {len(expr_sample_ids):,} samples")
            
            # Map expression samples to participants
            expr_participant_ids = set()
            for sample_id in expr_sample_ids:
                if sample_id in sample_to_participant_test:
                    expr_participant_ids.add(sample_to_participant_test[sample_id])
                elif sample_id in set(sample_to_participant_test.values()):
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
                if sample_id in sample_to_participant_test:
                    common_participants_from_samples.add(sample_to_participant_test[sample_id])
                elif sample_id in set(sample_to_participant_test.values()):
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
            subset_expr_df = expr_df[[col for col in expr_df.columns if str(col) in common_sample_ids]].copy()
            
            print(f"  Subsetted expression matrix: {subset_expr_df.shape[0]:,} genes, {subset_expr_df.shape[1]:,} samples")
            
            # Verify no samples were lost
            if subset_expr_df.shape[1] != len(common_sample_ids):
                print(f"  ⚠ WARNING: Expected {len(common_sample_ids)} samples, got {subset_expr_df.shape[1]}")
            
            test_expression_matrices.append(subset_expr_df)
            print(f"  ✓ Subsetted expression matrix added (will merge later)")
            
        except Exception as e:
            print(f"  ERROR loading/subsetting expression file: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Merge all test expression matrices (only one file for ROSMAP DLPFC, but keep for consistency)
    print(f"\n--- Merging test expression matrices ---")
    if test_expression_matrices:
        print(f"  Number of expression matrices to merge: {len(test_expression_matrices)}")
        
        # Check for overlapping genes
        all_genes = set()
        for expr_df in test_expression_matrices:
            all_genes.update(expr_df.index)
        
        print(f"  Total unique genes across all matrices: {len(all_genes):,}")
        
        # Merge on gene index
        merged_expr = test_expression_matrices[0]
        for i, expr_df in enumerate(test_expression_matrices[1:], 1):
            print(f"  Merging matrix {i+1}/{len(test_expression_matrices)}...")
            overlapping_cols = set(merged_expr.columns) & set(expr_df.columns)
            if overlapping_cols:
                print(f"    ⚠ WARNING: {len(overlapping_cols)} overlapping sample IDs found, will keep first occurrence")
            
            merged_expr = pd.concat([merged_expr, expr_df], axis=1, join='outer', sort=False)
            print(f"    After merge: {merged_expr.shape[0]:,} genes, {merged_expr.shape[1]:,} samples")
        
        # Fill NaN values with 0
        merged_expr = merged_expr.fillna(0)
        
        print(f"\n  Final merged expression matrix: {merged_expr.shape[0]:,} genes, {merged_expr.shape[1]:,} samples")
        
        # Verify sample counts match covariates
        merged_sample_ids = set(merged_expr.columns.astype(str))
        cov_sample_ids_all = set(covariates_test['sample_id'].astype(str))
        
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
            if sample_id in sample_to_participant_test:
                merged_participants.add(sample_to_participant_test[sample_id])
            elif sample_id in set(sample_to_participant_test.values()):
                merged_participants.add(sample_id)
        
        cov_participants_all = set(covariates_test['participant_id'].astype(str))
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
        output_file = os.path.join(OUTPUT_DIR_TEST, "tpm.tsv")
        merged_expr.to_csv(output_file, sep="\t")
        print(f"\n  ✓ Saved merged test expression matrix: {output_file}")
    else:
        print("  ⚠ No expression matrices to merge for test set")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nTrain set:")
print(f"  Covariates: {len(covariates_train):,} samples, {covariates_train['participant_id'].nunique():,} participants")
if train_expression_matrices:
    train_output_file = os.path.join(OUTPUT_DIR_TRAIN, "tpm.tsv")
    if os.path.exists(train_output_file):
        train_expr = pd.read_csv(train_output_file, sep="\t", index_col=0)
        print(f"  Expression: {train_expr.shape[0]:,} genes, {train_expr.shape[1]:,} samples")
        train_expr_samples = set(train_expr.columns.astype(str))
        train_expr_participants = set()
        for sample_id in train_expr_samples:
            if sample_id in sample_to_participant_train:
                train_expr_participants.add(sample_to_participant_train[sample_id])
            elif sample_id in set(sample_to_participant_train.values()):
                train_expr_participants.add(sample_id)
        print(f"  Expression participants: {len(train_expr_participants):,}")

print(f"\nTest set:")
print(f"  Covariates: {len(covariates_test):,} samples, {covariates_test['participant_id'].nunique():,} participants")
test_output_file = os.path.join(OUTPUT_DIR_TEST, "tpm.tsv")
if os.path.exists(test_output_file):
    test_expr = pd.read_csv(test_output_file, sep="\t", index_col=0)
    print(f"  Expression: {test_expr.shape[0]:,} genes, {test_expr.shape[1]:,} samples")
    test_expr_samples = set(test_expr.columns.astype(str))
    test_expr_participants = set()
    for sample_id in test_expr_samples:
        if sample_id in sample_to_participant_test:
            test_expr_participants.add(sample_to_participant_test[sample_id])
        elif sample_id in set(sample_to_participant_test.values()):
            test_expr_participants.add(sample_id)
    print(f"  Expression participants: {len(test_expr_participants):,}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)

