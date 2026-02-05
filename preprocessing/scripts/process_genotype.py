#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
import torch
from collections import defaultdict


# --- CONFIG ---
CHR = sys.argv[1]
SET = sys.argv[2]
RAW_DIR = sys.argv[3]
OUTPUT_DIR = sys.argv[4]

BASE_DIR = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/{SET}"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations", f"chr{CHR}")
GENOTYPES_DIR = os.path.join(BASE_DIR, "genotypes", f"chr{CHR}")
COVARIATES_FILE = os.path.join(BASE_DIR, "covariates.tsv")

def main():
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

    # --- Iterate over all genes in the chromosome ---
    print("Script started", flush=True)
    for file in sorted(os.listdir(ANNOTATIONS_DIR)):
        if not file.endswith("_annotations.tsv.gz"):
            continue

        gene = re.sub(r"_annotations\.tsv\.gz$", "", file)
        print(f"\nProcessing {gene} ...", flush=True)

        try:
            # --- Load annotations and sort by position ---
            annotations_file = os.path.join(ANNOTATIONS_DIR, file)
            annotations = pd.read_csv(annotations_file, sep="\t").sort_values("pos")

            # --- Load genotype (.raw) file ---
            raw_file = os.path.join(RAW_DIR, f"{gene}.raw")
            if not os.path.exists(raw_file):
                raise ValueError(f"Missing genotype file for {gene}, skipping.")

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
            geno = geno.set_index("IID").reindex(samples).reset_index()
            print(f"{gene}: sample order aligned with covariates.tsv ({len(samples)} samples).")

            # --- Variant column names ---
            variant_cols = [c for c in geno.columns if c != "IID"] 

            # PLINK --recode A include-alt creates: CHR:POS_A1_A2_COUNTED(/OTHER)
            # Format: 21:41698301_A_G_A(/G) where:
            #   - A1, A2 are from base file (may not be minor after merge)
            #   - COUNTED is the true minor allele in merged file (what PLINK counts)
            #   - (/OTHER) is the other allele in parentheses
            # Rename to: CHR:POS_COUNTED_OTHER
            col_rename_map = {}
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
            
            if col_rename_map:
                print(f"{gene}: Renaming {len(col_rename_map)} columns from CHR:POS_A1_A2_COUNTED(/OTHER) to CHR:POS_COUNTED_OTHER")
                geno = geno.rename(columns=col_rename_map)
                variant_cols = [c for c in geno.columns if c != "IID"]

            # --- Variant position extraction ---
            pos = (
                pd.Series(variant_cols)
                .str.extract(r':(\d+)_')[0]
            )
            geno_pos = pos.astype(int)

            # --- Align with annotation order ---
            variant_order = annotations["pos"].astype(int).tolist()
            common_positions = [pos for pos in variant_order if pos in geno_pos.values]
            if not common_positions:
                raise ValueError(f"{gene}: no overlapping variants, skipping.")
        

            pos_to_col = dict(zip(geno_pos.values, variant_cols))
            print(f"{gene}: {len(common_positions)} common positions, {len(pos_to_col)} mapped columns")

            # --- Order genotype columns by position order of annotations ---
            geno_ordered = geno[["IID"] + [pos_to_col[pos] for pos in common_positions]]

            # --- Flip alleles if >50% are 2 --- 
            # PLINK --recode A recalculates minor allele and counts minor 0/1/2, so should already be sparse but just in case
            geno_values = geno_ordered.drop(columns=["IID"])
            prop_2s = (geno_values == 2).sum(axis=0) / len(geno_values)
            cols_to_flip = prop_2s[prop_2s > 0.5].index
            geno_values[cols_to_flip] = geno_values[cols_to_flip].replace({0: 2, 2: 0})

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
                geno_values = geno_values.rename(columns=col_rename_map)    

            variant_ids_original = geno_values.columns.tolist()

            # --- Handle missing values (drop >10%, impute others) ---
            missing_frac = geno_values.isna().mean(axis=0)
            cols_to_drop = missing_frac[missing_frac > 0.1].index
            geno_values = geno_values.drop(columns=cols_to_drop)
            geno_values = geno_values.apply(lambda col: col.fillna(col.mean()), axis=0)

            # --- Sync annotations (remove dropped variants, sort, re-save) ---
            remaining_positions = (
                pd.Series(geno_values.columns)
                .str.extract(r':(\d+)_([ACGT])_([ACGT])$')[0] # CHR:POS_A1_A2
                .astype(int)
            )
            annotations = annotations[annotations["pos"].isin(remaining_positions)]

            variant_ids = geno_values.columns 
            assert len(variant_ids) == len(annotations), \
                f"Mismatch: {len(variant_ids)} genotype variants vs {len(annotations)} annotations"


            # --- Resave annotations matrix ---
            annotations.to_csv(annotations_file, sep="\t", index=False, compression="gzip")
            tensor_out = os.path.join(ANNOTATIONS_DIR, f"{gene}_annotations.pt")
            torch.save(torch.tensor(annotations.values, dtype=torch.float32), tensor_out)


            # --- Save tensor and full matrix ---
            tensor_out = os.path.join(GENOTYPES_DIR, f"{gene}_genotypes.pt")
            torch.save(torch.tensor(geno_values.values, dtype=torch.float32), tensor_out)
            
            # --- Save variant IDs (column names) ---
            variant_id_file = os.path.join(GENOTYPES_DIR, f"{gene}_variant_ids.txt")
            pd.Series(variant_ids).to_csv(variant_id_file, index=False, header=False)
            print(f"{gene}: Saved variant IDs → {variant_id_file}")


            print(f"{gene}: {len(annotations)} variants after QC "
                    f"(flipped {len(cols_to_flip)}, dropped {len(cols_to_drop)}).")

            # --- Record QC summary stats ---
            n_original = len(variant_ids_original)
            n_final = len(variant_ids)
            n_dropped = n_original - n_final
            assert(len(cols_to_drop) == n_dropped)

            gene_variant_counts[gene] = len(variant_ids)

            # Count how many genes each variant maps to (after QC)
            for v in variant_ids:
                variant_gene_counts[v] += 1

        except Exception as e:
            print(f"Error processing {gene}: {e}", file=sys.stderr)
            continue
    

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

        


if __name__ == "__main__":
    main()