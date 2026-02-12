import os
import sys
import pandas as pd
import torch
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

COHORT = "ROSMAP"
TISSUE = "DLPFC"

if '--cohort' in sys.argv:
    COHORT = sys.argv[sys.argv.index('--cohort') + 1]
if '--tissue' in sys.argv:
    TISSUE = sys.argv[sys.argv.index('--tissue') + 1]

GENOTYPE_DIR = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/genotypes"
ANOTATION_DIR = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations"

EXPRESSION_MATRIX = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/tpm.tsv"
COVARIATES_FILE = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/covariates.tsv"

TRAIN_DIR = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/Train"
TEST_DIR = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/Test"

CHROM_NB = sys.argv[1]


if __name__ == "__main__":
    print("================================================")
    print(f"Cohort: {COHORT}")
    print(f"Tissue: {TISSUE}")
    print(f"Chromosome: {CHROM_NB}")
    print(f"Train Directory: {TRAIN_DIR}")
    print(f"Test Directory: {TEST_DIR}")
    print("================================================")
    
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # write holdout set name (cohort-tissue pair) to file
    with open(f"{TEST_DIR}/holdout_set.txt", "w") as f:
        f.write(f"{COHORT}_{TISSUE}")

    covariates = pd.read_csv(COVARIATES_FILE, sep="\t")
    print(f"Total number of samples: {len(covariates)}")
    
    # get holdout set
    holdout = covariates[covariates['cohort_tissue'].str.contains(TISSUE)]
    holdout = holdout[holdout['cohort'] == COHORT]

    # check if holdout set is empty
    if len(holdout) == 0:
        print(f"No holdout set found for {COHORT}_{TISSUE}")
        print(f"\tcohort-tissue pairs in covariates: {covariates['cohort_tissue'].unique()}")
        sys.exit(1)

    # get remaining samples for train set
    holdout_indices = holdout.index
    train_indices = covariates.index.difference(holdout_indices)
    train = covariates.loc[train_indices]

    # check if train set is empty
    if len(train) == 0:
        print(f"No train samples remaining after removing holdout set.")
        sys.exit(1)
    
    # print stats
    print(f"TRAIN SET:")
    print(f"\tTotal number of samples: {len(train)}")
    print(f"\tTotal number of participants: {train['participant_id'].nunique()}")
    print("================================================")
    print(f"TEST SET:")
    print(f"\tTotal number of samples: {len(holdout)}")
    print(f"\tTotal number of participants: {holdout['participant_id'].nunique()}")
    print("================================================")
    
    # get train and test sample IDs --> use to subset genotype matrices
    train_samples = train['sample_id'].tolist()
    test_samples = holdout['sample_id'].tolist()


    # only need to do this once, so do it with chrom 22
    if int(CHROM_NB) == 22:
        # save covariates --> relative order of samples is preserved
        holdout.set_index('sample_id', inplace=True)
        train.set_index('sample_id', inplace=True)

        print("================================================")
        print(f"Train Covariates: {os.path.join(TRAIN_DIR, 'covariates.tsv')}")
        print(f"Test Covariates: {os.path.join(TEST_DIR, 'covariates.tsv')}")
        print("================================================")

        holdout.to_csv(os.path.join(TRAIN_DIR, 'covariates.tsv'), sep="\t", index=True)
        train.to_csv(os.path.join(TEST_DIR, 'covariates.tsv'), sep="\t", index=True)

    # subset genotypes --> relative order of samples is preserved
    # (N all samples x variants) --> (N test samples x variants) + (N train samples x variants)
    genotype_train_dir = os.path.join(TRAIN_DIR, "genotypes", f"chr{CHROM_NB}")
    genotype_test_dir = os.path.join(TEST_DIR, "genotypes", f"chr{CHROM_NB}")
    print("================================================")
    print(f"Genotypes Training Directory: {genotype_train_dir}")
    print(f"Genotypes Test Directory: {genotype_test_dir}")
    print("================================================")
    os.makedirs(genotype_train_dir, exist_ok=True)
    os.makedirs(genotype_test_dir, exist_ok=True)
    
    # Collect all gene files to process
    gene_files = [f for f in os.listdir(os.path.join(GENOTYPE_DIR, f"chr{CHROM_NB}")) 
                  if f.endswith("_genotypes.tsv.gz")]
    n_files = len(gene_files)
    print(f"Splitting {n_files} genotype files for chromosome {CHROM_NB}...", flush=True)
    
    # Thread-safe counter and lock for progress reporting
    n_files_processed = 0
    progress_lock = Lock()
    
    def process_gene_file(gene_file):
        """Process a single gene file: read, subset, and save train/test splits."""
        gene = gene_file.split("_")[0]
        file_path = os.path.join(GENOTYPE_DIR, f"chr{CHROM_NB}", gene_file)
        
        # Read genotype file
        geno = pd.read_csv(file_path, sep="\t", index_col='IID')
        geno_train = geno.loc[train_samples]
        geno_test = geno.loc[test_samples]
        
        # Assertions
        assert geno_train.shape[0] == len(train_samples) # subset all samples
        assert geno_test.shape[0] == len(test_samples) # subset all samples
        assert geno_train.shape[1] == geno.shape[1] # didnt drop variants
        assert geno_test.shape[1] == geno.shape[1] # didnt drop variants

        # Save as tsv --> R compatibility
        geno_train.to_csv(os.path.join(genotype_train_dir, f"{gene}_genotypes.tsv.gz"), 
                         sep="\t", index=True, compression='gzip')
        geno_test.to_csv(os.path.join(genotype_test_dir, f"{gene}_genotypes.tsv.gz"), 
                        sep="\t", index=True, compression='gzip')

        # Save as torch tensor
        torch.save(torch.tensor(geno_train.values, dtype=torch.float32), 
                  os.path.join(genotype_train_dir, f"{gene}_genotypes.pt"))
        torch.save(torch.tensor(geno_test.values, dtype=torch.float32), 
                  os.path.join(genotype_test_dir, f"{gene}_genotypes.pt"))
        
    
    # Process files in parallel using ThreadPoolExecutor
    # max_workers=None uses min(32, os.cpu_count() + 4) as default
    # For I/O-bound tasks, we can use more workers
    max_workers = min(8, n_files)  # Use up to 8 threads, or fewer if fewer files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_gene_file, gene_file) for gene_file in gene_files]
        # Wait for all tasks to complete and handle any exceptions
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions that occurred
                n_files_processed += 1
                if n_files_processed % 10 == 0:
                    print(f"\t[{n_files_processed}/{n_files}] ✓ Completed", flush=True)
            except Exception as e:
                print(f"Error processing file: {e}", flush=True)
                raise
    
    print(f"\tCompleted all {n_files_processed}/{n_files} files", flush=True)


    # no need to subset annotations (variants x annotations)
    # just copy over from original file
    # annotation_train_dir = os.path.join(TRAIN_DIR, "annotations", f"chr{CHROM_NB}")
    # annotation_test_dir = os.path.join(TEST_DIR, "annotations", f"chr{CHROM_NB}")
    # os.makedirs(annotation_train_dir, exist_ok=True)
    # os.makedirs(annotation_test_dir, exist_ok=True)
    # print("================================================")
    # print(f"Annotations Training Directory: {annotation_train_dir}")
    # print(f"Annotations Test Directory: {annotation_test_dir}")
    # print("================================================")
    # for annotation_file in os.listdir(os.path.join(ANOTATION_DIR, f"chr{CHROM_NB}")):
    #     shutil.copy(os.path.join(ANOTATION_DIR, f"chr{CHROM_NB}", annotation_file), os.path.join(annotation_train_dir, annotation_file))
    #     shutil.copy(os.path.join(ANOTATION_DIR, f"chr{CHROM_NB}", annotation_file), os.path.join(annotation_test_dir, annotation_file))

    # subset expression matrix
    # only need to do this once, so do it with chrom 22
    if int(CHROM_NB) == 22:
        expression = pd.read_csv(EXPRESSION_MATRIX, sep="\t", header=0)
        expression_train = expression[train_samples] # will reorder to match covariates and genotypes
        expression_test = expression[test_samples] # will reorder to match covariates and genotypes
        assert expression_train.shape[1] == len(train_samples) # didnt drop samples
        assert expression_test.shape[1] == len(test_samples) # didnt drop samples
        assert expression_train.shape[0] == expression.shape[0] # didnt drop genes
        assert expression_test.shape[0] == expression.shape[0] # didnt drop genes
        
        print("================================================")
        print(f"Expression Training Matrix: ({expression_train.shape[0]} genes x {expression_train.shape[1]} samples)", flush=True)
        print(f"\tSaved to: {os.path.join(TRAIN_DIR, 'tpm.tsv')}", flush=True)
        print(f"Expression Test Matrix: ({expression_test.shape[0]} genes x {expression_test.shape[1]} samples)", flush=True)
        print(f"\tSaved to: {os.path.join(TEST_DIR, 'tpm.tsv')}", flush=True)
        print("================================================")

        # transpose --> samples x genes

        expression_train.to_csv(os.path.join(TRAIN_DIR, "tpm.tsv"), sep="\t", index=False)
        expression_test.to_csv(os.path.join(TEST_DIR, "tpm.tsv"), sep="\t", index=False)