#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
import torch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime


# --- CONFIG ---
CHR = sys.argv[1]
RAW_DIR = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

# Number of threads for parallel processing (default: 4, can be overridden with NUM_THREADS env var)
NUM_THREADS = int(os.environ.get("NUM_THREADS", "4"))

BASE_DIR = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations", f"chr{CHR}")
GENOTYPES_DIR = os.path.join(BASE_DIR, "genotypes", f"chr{CHR}")
COVARIATES_FILE = os.path.join(BASE_DIR, "covariates.tsv")

# Thread-safe locks for shared data structures
gene_variant_counts_lock = Lock()
variant_gene_counts_lock = Lock()

def process_gene(gene, file, samples, gene_variant_counts, variant_gene_counts):
    """
    Process a single gene. This function is designed to be thread-safe.
    
    Returns: (gene, success, n_variants, variant_ids) or (gene, False, 0, [])
    """
    try:
        print(f"Processing {gene} ...", flush=True)

        # --- Check if genotype file exists (ie gene passed plink extraction/recode) ---
        raw_file = os.path.join(RAW_DIR, f"{gene}.raw")
        if not os.path.exists(raw_file):
            raise ValueError(f"Missing genotype file for {gene}, skipping.")

        # --- Load annotations and sort by position ---
        annotations_file = os.path.join(ANNOTATIONS_DIR, file)
        annotations = pd.read_csv(annotations_file, sep="\t", index_col="variant_id")

        # --- Load genotype (.raw) file ---
        raw = pd.read_csv(raw_file, delim_whitespace=True)
        geno = raw.drop(columns=["FID", "PAT", "MAT", "SEX", "PHENOTYPE"])
        geno["IID"] = geno["IID"].astype(str)

        # --- Sanity check for sample consistency ---
        geno_iids = geno["IID"].tolist()

        if len(geno_iids) != len(set(geno_iids)):
            raise ValueError(f"{gene}: genotype file has duplicate IIDs!")

        missing_in_geno = set(samples) - set(geno_iids)
        extra_in_geno = set(geno_iids) - set(samples)

        if missing_in_geno:
            raise ValueError(f"{gene}: missing {len(missing_in_geno)} samples in genotype data.")
        if extra_in_geno:
            raise ValueError(f"{gene}: {len(extra_in_geno)} unexpected samples in genotype data.")

        # --- Reorder to match samples.txt ---
        geno = geno.set_index("IID").reindex(samples)
        print(f"{gene}: sample order aligned with covariates.tsv ({len(samples)} samples).", flush=True)

        # --- Variant column names ---
        variant_cols = [c for c in geno.columns if c != "IID"] 

        # PLINK --recode A include-alt creates: CHR:POS_A1_A2_COUNTED(/OTHER)
        # Format: 21:41698301_A_G_A(/G) where:
        #   - A1, A2 are from base file (may not be minor after merge)
        #   - COUNTED is the true minor allele in merged file (what PLINK counts)
        #   - (/OTHER) is the other allele in parentheses
        # Rename to: CHR:POS_COUNTED_OTHER
        col_rename_map = {}
        geno_lookup = {}
        for col in variant_cols:
            # Parse format: CHR:POS_A1_A2_COUNTED(/OTHER)
            m = re.match(rf"{CHR}:(\d+)_([ACGT])_([ACGT])_([ACGT])\(/([ACGT])\)$", col)
            if m:
                pos = m.group(1)
                a1_base = m.group(2)  # A1 from base file
                a2_base = m.group(3)  # A2 from base file
                counted = m.group(4)  # Counted (minor) allele
                other = m.group(5)    # Other allele from (/OTHER)
                
                # New format: CHR:POS_COUNTED_OTHER
                new_col = f"{CHR}:{pos}_{counted}_{other}"
                col_rename_map[col] = new_col

                # Keep track of variant ids to match with ids in annotation file (ref=other, alt=counted or vice versa)
                geno_lookup[(pos, counted, other)] = new_col
                geno_lookup[(pos, other, counted)] = new_col
        
        if col_rename_map:
            print(f"{gene}: Renaming {len(col_rename_map)} columns from CHR:POS_A1_A2_COUNTED(/OTHER) to CHR:POS_COUNTED_OTHER", flush=True)
            geno = geno.rename(columns=col_rename_map)
            variant_cols = [c for c in geno.columns if c != "IID"]

   
        # --- Align with annotation order ---
        # Genotype still may have multiple variants at same position
        variant_order = annotations.index.tolist() # chr:pos_ref_alt
        # Match annotation variants to genotype variants using pre-built lookup
        common_variants = []
        for var in variant_order:  # chr:pos_ref_alt
            pos, ref, alt =  var.split(":")[1].split("_")
          
            # Try both allele orderings: (pos, ref, alt) and (pos, alt, ref)
            key1 = (pos, ref, alt)
            key2 = (pos, alt, ref)
            
            if key1 in geno_lookup:
                common_variants.append(geno_lookup[key1])
            elif key2 in geno_lookup:
                common_variants.append(geno_lookup[key2])

        if not common_variants:
            raise ValueError(f"{gene}: no overlapping variants, skipping.")
        
        print(f"{gene}: Annotations have {len(variant_order)} variants, genotype has {len(variant_cols)} variants", flush=True)
        print(f"{gene}: {len(common_variants)} common variants", flush=True)

        geno_ordered = geno[common_variants]

        # --- Flip alleles if >50% are 2 --- 
        # PLINK --recode A recalculates minor allele and counts minor 0/1/2, so should already be sparse but just in case
        prop_2s = (geno_ordered == 2).sum(axis=0) / len(geno_ordered)
        cols_to_flip = prop_2s[prop_2s > 0.5].index
        geno_ordered[cols_to_flip] = geno_ordered[cols_to_flip].replace({0: 2, 2: 0})

        # --- Rename flipped columns ---
        if len(cols_to_flip) > 0:
            col_rename_map = {}
            for col in cols_to_flip:
                m = re.match(rf"{CHR}:(\d+)_([ACGT])_([ACGT])$", col)
                if m:
                    pos = m.group(1)
                    a1 = m.group(2)
                    a2 = m.group(3)
                    new_col = f"{CHR}:{pos}_{a2}_{a1}"
                    col_rename_map[col] = new_col
            geno_ordered = geno_ordered.rename(columns=col_rename_map)    

        variant_ids_original = geno_ordered.columns.tolist()

        # --- Handle missing values (drop >10%, impute others) ---
        missing_frac = geno_ordered.isna().mean(axis=0)
        cols_to_drop = missing_frac[missing_frac > 0.1].index
        geno_ordered = geno_ordered.drop(columns=cols_to_drop)
        geno_ordered = geno_ordered.apply(lambda col: col.fillna(col.mean()), axis=0)

        # --- Sync annotations (remove dropped variants, sort, re-save) ---
        remaining_positions = (
            pd.Series(geno_ordered.columns)
            .str.extract(r':(\d+)_([ACGT])_([ACGT])$')[0] # CHR:POS_A1_A2
            .astype(int)
        )
        annotations = annotations[annotations["pos"].isin(remaining_positions)]

        variant_ids = geno_ordered.columns 
        assert len(variant_ids) == len(annotations), \
            f"Mismatch: {len(variant_ids)} genotype variants vs {len(annotations)} annotations"


        # --- Resave annotations matrix ---
        annotations.to_csv(annotations_file, sep="\t", index=True, compression="gzip")
        tensor_out = os.path.join(ANNOTATIONS_DIR, f"{gene}_annotations.pt")
        torch.save(torch.tensor(annotations.values, dtype=torch.float32), tensor_out)


        # --- Save tensor and full matrix ---
        tensor_out = os.path.join(GENOTYPES_DIR, f"{gene}_genotypes.pt")
        geno_ordered.to_csv(os.path.join(GENOTYPES_DIR, f"{gene}_genotypes.tsv.gz"), sep="\t", index=True, compression="gzip")
        torch.save(torch.tensor(geno_ordered.values, dtype=torch.float32), tensor_out)
        
        # --- Save variant IDs (column names) ---
        variant_id_file = os.path.join(GENOTYPES_DIR, f"{gene}_variant_ids.txt")
        pd.Series(variant_ids).to_csv(variant_id_file, index=False, header=False)
        print(f"{gene}: Saved variant IDs → {variant_id_file}", flush=True)


        print(f"{gene}: {len(annotations)} variants after QC "
                f"(flipped {len(cols_to_flip)}, dropped {len(cols_to_drop)}).", flush=True)

        # --- Record QC summary stats (thread-safe) ---
        n_original = len(variant_ids_original)
        n_final = len(variant_ids)
        n_dropped = n_original - n_final
        assert(len(cols_to_drop) == n_dropped)

        with gene_variant_counts_lock:
            gene_variant_counts[gene] = len(variant_ids)

        # Count how many genes each variant maps to (after QC) - thread-safe
        with variant_gene_counts_lock:
            for v in variant_ids:
                variant_gene_counts[v] += 1

        return (gene, True, len(variant_ids), variant_ids)

    except Exception as e:
        print(f"Error processing {gene}: {e}", file=sys.stderr, flush=True)
        return (gene, False, 0, [])

def main():
    start_time = datetime.now()
    os.makedirs(GENOTYPES_DIR, exist_ok=True)

    # --- Load global sample list ---
    covariates = pd.read_csv(COVARIATES_FILE, sep="\t")
    samples = covariates['sample_id'].astype(str).tolist()
    n_total = len(samples)
    n_unique = len(set(samples))
    if n_total != n_unique:
        raise ValueError(f"covariates.tsv has {n_total} IDs but only {n_unique} unique ones (duplicates exist).")
    print(f"Loaded {n_total} unique samples from {COVARIATES_FILE}.")

    # Track number of variants per gene
    gene_variant_counts = {}

    # Track how many genes each variant maps to
    variant_gene_counts = defaultdict(int)

    # --- Collect all gene files to process ---
    print(f"Script started (using {NUM_THREADS} threads)", flush=True)
    gene_files = []
    for file in sorted(os.listdir(ANNOTATIONS_DIR)):
        if not file.endswith("_annotations.tsv.gz"):
            continue
        gene = re.sub(r"_annotations\.tsv\.gz$", "", file)
        gene_files.append((gene, file))
    
    print(f"Found {len(gene_files)} genes to process", flush=True)

    # --- Process genes in parallel using ThreadPoolExecutor ---
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit all tasks
        future_to_gene = {
            executor.submit(process_gene, gene, file, samples, gene_variant_counts, variant_gene_counts): (gene, file)
            for gene, file in gene_files
        }
        
        # Process completed tasks as they finish
        completed = 0
        for future in as_completed(future_to_gene):
            gene, file = future_to_gene[future]
            completed += 1
            try:
                result = future.result()
                gene_name, success, n_variants, variant_ids = result
                if success:
                    print(f"[{completed}/{len(gene_files)}] ✓ Completed {gene_name} ({n_variants} variants)", flush=True)
                else:
                    print(f"[{completed}/{len(gene_files)}] ✗ Failed {gene_name}", flush=True)
            except Exception as e:
                print(f"[{completed}/{len(gene_files)}] ✗ Exception for {gene}: {e}", flush=True)
    

    # --- Save per-gene summary ---
    print("\n")
    gene_summary = pd.DataFrame(list(gene_variant_counts.items()), columns=["gene", "n_variants"])
    gene_summary = gene_summary.sort_values("n_variants", ascending=False)
    gene_summary_file = os.path.join(GENOTYPES_DIR, f"gene_variant_counts.txt")
    gene_summary.to_csv(gene_summary_file, sep="\t", index=False)
    print(f"Saved per-gene variant counts → {gene_summary_file}")

    # --- Save per-variant summary ---
    variant_summary = pd.DataFrame(list(variant_gene_counts.items()), columns=["variant", "n_genes"])
    variant_summary = variant_summary.sort_values("n_genes", ascending=False)
    variant_summary_file = os.path.join(GENOTYPES_DIR, f"variant_gene_counts.txt")
    variant_summary.to_csv(variant_summary_file, sep="\t", index=False)
    print(f"Saved per-variant gene mapping counts → {variant_summary_file}")

    end_time = datetime.now()
    print(f"Total genotype processing time: {end_time - start_time}")


if __name__ == "__main__":
    main()