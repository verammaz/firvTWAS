#!/usr/bin/env python3
"""
Generate ID mapping file for duplicating and renaming genotype entries.

For each cohort:
1. Load subsetted genotype file to get participant IDs
2. Get participant_id -> sample_id mapping from covariates
3. Generate mapping file with: old FID, old IID, new FID, new IID
"""

import pandas as pd
import os
import sys

# ============================================================================
# HARDCODED PATHS
# ============================================================================
BASE_DIR = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain"
GENOTYPE_DIR = os.path.join(BASE_DIR, "Genotypes")
GENOTYPE_SUBSETTED_DIR = os.path.join(GENOTYPE_DIR, "subsetted")
REMAPPING_DIR = os.path.join(GENOTYPE_DIR, "remapping")
COVARIATE_FILE = os.path.join(BASE_DIR, "Processed", "covariates.tsv")
COHORTS = ["NYGC", "ROSMAP", "MSBB", "Mayo", "GTEX", "AnswerALS"]


def get_genotype_participant_ids(base_path, chrom):
    """Get list of participant IDs (IIDs) from genotype file (PLINK 1.9 format)."""
    fam_file = f"{base_path}_chr{chrom}.fam"
    if not os.path.exists(fam_file):
        print(f"ERROR: FAM file not found: {fam_file}")
        sys.exit(1)
    # PLINK1 .fam format: FID, IID, [other columns]
    df = pd.read_csv(fam_file, sep='\s+', header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
    return dict(zip(df['IID'].astype(str), df['FID'].astype(str)))

def get_participant_to_samples_map(cohort, covariates):
    """Get mapping of participant_id -> list of sample_ids for a cohort."""
    cohort_cov = covariates[covariates['cohort'] == cohort].copy()
    if len(cohort_cov) == 0:
        print(f"ERROR: Cohort {cohort} not found in covariates")
        sys.exit(1)
    
    # Create mapping: participant_id -> list of sample_ids
    participant_to_samples = {}
    for _, row in cohort_cov.iterrows():
        participant_id = str(row['participant_id'])
        sample_id = str(row['sample_id'])
        
        if participant_id not in participant_to_samples:
            participant_to_samples[participant_id] = []
        participant_to_samples[participant_id].append(sample_id)
    
    return participant_to_samples

def generate_id_mapping(cohort, input_base, participant_to_samples):
    """Generate ID mapping file for duplicating and renaming genotype entries."""
    
    print(f"Generating ID mapping for cohort: {cohort}")
    print(f"Input genotype: {input_base}")
    print("")
    
    # Step 1: Get participant IDs from genotype file
    print("Step 1: Reading participant IDs from genotype files...")
    genotype_participant_ids = {}
    for chrom in range(1, 23):
        ids = get_genotype_participant_ids(input_base, chrom)
        print(f"  Found {len(ids)} participants in genotype file for chromosome {chrom}")
        genotype_participant_ids.update(ids)
    print(f"  Total participants: {len(genotype_participant_ids)}")
    print("")

    
    # Step 2: Create mapping
    print("Step 2: Creating ID mapping...")
    id_mapping = []  # Will store: old IID, new IID (FID not used)

    for participant_id, fid in genotype_participant_ids.items():
        participant_id = str(participant_id)
        
        if participant_id in participant_to_samples:
            sample_ids = participant_to_samples[participant_id]
            
            # For each sample, create a mapping entry
            for _, sample_id in enumerate(sample_ids):
                # Mapping: old IID, new IID (FID not used, will be set to '0' in bash script)
                id_mapping.append({
                    'old_iid': participant_id,
                    'new_iid': str(sample_id)
                })
            
        else:
            print(f"  WARNING: Participant {participant_id} not found in covariates mapping")
    
    # Step 3: Save mapping file
    print("Step 3: Saving mapping file...")
    os.makedirs(REMAPPING_DIR, exist_ok=True)
    mapping_file = os.path.join(REMAPPING_DIR, f"{cohort}_update_ids.txt")
    
    id_mapping_df = pd.DataFrame(id_mapping)
    id_mapping_df.to_csv(mapping_file, sep='\t', header=False, index=False)
    
    print(f"  File: {mapping_file}")
    print("")
    
    # Print summary
    n_participants = len(genotype_participant_ids)
    n_samples = len(id_mapping_df)
    participants_with_multiple = sum(1 for pid in genotype_participant_ids 
                                     if pid in participant_to_samples and 
                                     len(participant_to_samples[str(pid)]) > 1)
    
    print("Summary:")
    print(f"  Participants in genotype: {n_participants}")
    print(f"  Total samples (after duplication): {n_samples}")
    print(f"  Participants with multiple samples: {participants_with_multiple}")
    print("")
    
    return 

def main():
    # Load covariates
    print("Loading covariates files...")
    covariates = pd.read_csv(COVARIATE_FILE, sep="\t")
    print(f"  {len(covariates)} samples, {covariates['participant_id'].nunique():,} participants")
    print("")

    for cohort in COHORTS:
        
        print("=" * 80)
        print(f"GENERATE ID MAPPING: {cohort}")
        print("=" * 80)
        print("")
        
        # Get participant to samples mapping
        print("Creating participant -> samples mapping from covariates...")
        participant_to_samples = get_participant_to_samples_map(cohort, covariates)
        print(f"  Found {len(participant_to_samples)} participants in covariates")
        
        # Count samples
        total_samples = sum(len(samples) for samples in participant_to_samples.values())
        participants_with_multiple = sum(1 for samples in participant_to_samples.values() if len(samples) > 1)
        print(f"  Total samples: {total_samples}")
        print(f"  Participants with multiple samples: {participants_with_multiple}")
        print("")
        
        # Find input genotype file
        input_base = os.path.join(GENOTYPE_SUBSETTED_DIR, f"{cohort}_subset")
        
        # Generate mapping file
        generate_id_mapping(cohort, input_base, participant_to_samples)
        
    
       

if __name__ == "__main__":
    main()

