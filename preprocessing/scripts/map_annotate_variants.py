import pandas as pd
import numpy as np
import os
from collections import defaultdict
import sys

import warnings
from pandas.errors import DtypeWarning

warnings.filterwarnings(
    "ignore",
    message="Pandas requires version.*numexpr.*"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DtypeWarning)



REMOVE_DUPS = '--remove_dups' in sys.argv
MATCH_ALLELES = '--match_alleles' in sys.argv
USE_REF_ALLELES = '--use_ref_alleles' in sys.argv

# Validate the presence of -chrom
if '-chrom' not in sys.argv:
    sys.exit("Error: Missing required -chrom argument (must be 1–22)")

# Get index of -chrom and try to read the next value
chrom_index = sys.argv.index('-chrom') + 1

if chrom_index >= len(sys.argv):
    sys.exit("Error: -chrom argument must be followed by a number (1–22)")

try:
    CHRO_NB = int(sys.argv[chrom_index])
    if CHRO_NB < 1 or CHRO_NB > 22:
        raise ValueError
except ValueError:
    sys.exit("Error: -chrom must be an integer between 1 and 22")

# Get train or test set
if '-set' in sys.argv:
    SET = sys.argv[sys.argv.index('-set') + 1]
else:
    sys.exit("Error: Specify Train or Test set with -set Train or -set Test.")

if SET not in ["Train", "Test"]:
    sys.exit("Error: Invalid set. Must be Train or Test.")

ANNOTATIONS_DIR = os.path.join(f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/{SET}", "annotations", f"chr{CHRO_NB}")
GENOTYPE_PATH=f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/{SET}/chroms"


noncoding_annotations = ['chr','pos','ref', 'alt', 'MAP20', 'phyloP17way_primate', 'phyloP30way_mammalian',
                         'phastCons30way_mammalian', 'phastCons17way_primate_rankscore',
                         'integrated_fitCons_score', 'H1-hESC_fitCons_score', 'bStatistic', 'GERP_RS',
                         'Roadmap_E074_GenoSkyline_Plus_score',
                       'Roadmap_E068_GenoSkyline_Plus_score',
                       'Roadmap_E069_GenoSkyline_Plus_score',
                       'Roadmap_E072_GenoSkyline_Plus_score',
                       'Roadmap_E067_GenoSkyline_Plus_score',
                       'Roadmap_E073_GenoSkyline_Plus_score',
                       'Roadmap_E070_GenoSkyline_Plus_score',
                       'Roadmap_E030_GenoSkyline_Plus_score',
                       'Roadmap_E050_GenoSkyline_Plus_score',
                       'Roadmap_E051_GenoSkyline_Plus_score',
                       'Roadmap_E124_GenoSkyline_Plus_score',
                      'funseq2_noncoding_score', 'fathmm-MKL_non-coding_score',
                   'fathmm-XF_score', 'CADD_raw', 'CADD_phred', 'DANN_score', 'Eigen-raw',
                   'Eigen-PC-raw', 'gnomAD_genomes_POPMAX_AF',
                   'gnomAD_genomes_AFR_AF', 'gnomAD_genomes_AMR_AF',
                   'gnomAD_genomes_NFE_AF', 'SpliceAI_DS_AG', 'SpliceAI_DS_AL',
                   'SpliceAI_DS_DG', 'SpliceAI_DS_DL', 'FANTOM5_CAGE_peak_robust', 'EnhancerFinder_brain_enhancer']


coding_annotations = ['chr','pos', 'ref', 'alt', 'MAP20', 'phyloP17way_primate', 'phyloP30way_mammalian',
                   'phastCons30way_mammalian', 'phastCons17way_primate_rankscore',
                   'integrated_fitCons_score', 'H1-hESC_fitCons_score', 'bStatistic', 'GERP_RS',
                   'fathmm-MKL_coding_score', 'fathmm-XF_score',
                   'CADD_raw', 'CADD_phred', 'DANN_score', 'Eigen-raw',
                   'Eigen-PC-raw','gnomAD_genomes_POPMAX_AF','gnomAD_genomes_AFR_AF', 'gnomAD_genomes_AMR_AF',
                   'gnomAD_genomes_NFE_AF','SpliceAI_DS_AG', 'SpliceAI_DS_AL','SpliceAI_DS_DG', 'SpliceAI_DS_DL']


def norm_chr(c):
    if pd.isna(c):
        return c
    c = str(c)
    c = c.replace("chr", "")
    return "chr" + c

def de_norm_chr(c):
    if pd.isna(c):
        return c
    c = str(c)
    c = c.replace("chr", "")
    return c



def load_reference():
    ref = pd.read_csv(
        f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/reference/chr{CHRO_NB}_ref.tsv.gz",
        sep="\t",
        compression="gzip",
        names=["chr", "pos", "ref"],
        dtype={"chr": str, "pos": int, "ref": str}
    )
    return ref

def load_gene_pos():
    gene_pos = pd.read_csv('/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/Alzheimer-RV/data/gene_positions38_ensemble.txt', sep = "\t")
    gene_pos = gene_pos[gene_pos['chr']==str(CHRO_NB)]
    genes = pd.read_csv("/gpfs/commons/groups/knowles_lab/vmazeeva/genes.bed", sep = "\t", header = None)
    genes = genes[[0,1,2,3]]
    genes.columns = ['chr','start','end','ensgene']
    genes = genes[genes['chr']==f'chr{CHRO_NB}']
    gene_pos['gencode'] = np.where(gene_pos['ensgene'].isin(list(genes['ensgene'])), 1, 0)
    gene_pos = gene_pos[(gene_pos['gencode']==1)]    
    gene_pos = gene_pos[(gene_pos['biotype']=='protein_coding')]
    gene_pos = gene_pos.drop_duplicates("ensgene")
    gene_pos = gene_pos.drop(['chr','start','end','entrez','gencode'], axis = 1).merge(genes, on = 'ensgene') # using gencode positions.
    gene_pos['symbol'] = np.where(gene_pos['symbol'].isna(), gene_pos['ensgene'], gene_pos['symbol'])
    return gene_pos


def map_genes(data, gene_pos):
    print("Mapping variants to genes...")
    all_variants = np.array(data['pos'].astype(int))  
    print(f"\tTotal Variants: {len(all_variants)} ({len(set(all_variants))} unique)")
    variant_genes = defaultdict(list)
    gene_variants = dict()
    for index, row in gene_pos.iterrows():
        start, end, gene = int(max(0, row['start'] - 100000)), int(row['end'] + 100000), row['ensgene'] # out of bounds end?
        variants = all_variants[(all_variants >= start) & (all_variants <= end)]
        if len(variants > 0):
            for var in variants:
                variant_genes[var].append(gene)
        gene_variants[gene] = list(variants)
    print("\tMapped Variants:", len(variant_genes), "\n")
    return variant_genes, gene_variants


def get_wgsa(bim, ref_table=None):
    print("# Coding annotations:", len(coding_annotations))
    print("# Noncoding annotations:",len(noncoding_annotations))
    columns_to_load = set(coding_annotations).union(set(noncoding_annotations))
    
    # columns_to_load.discard('ref')
    # columns_to_load.discard('alt')
    
    print("Total annotations:", len(columns_to_load), "\n")

    wgsa_file = f"/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/58K_preview_compact/WGSA/ADSP_chr{CHRO_NB}.snp.gz"
    wgsa = pd.read_csv(wgsa_file, sep = "\t", usecols=columns_to_load) # read full file

    print("WGSA # Variants:", len(wgsa))
    print("WGSA # Variants with unique positions:", len(set(wgsa['pos'])), "\n")
    
    print("BIM # Variants:", len(bim))
    print("BIM # Variants with unique positions:", len(set(bim[3])), "\n")

    data = wgsa[wgsa['pos'].isin(bim[3])] # wgsa has repeated positions
    
    print("WGSA # Variants in BIM by pos:", len(data))
    print("WGSA # Variants with unique positions in BIM by pos:", len(set(data['pos'])), "\n")

    if MATCH_ALLELES and ref_table is  None:
        print("\nFiltering by allele matches (either A1 or A2)...")
        # Rename bim columns for clarity
        bim = bim.rename(columns={0: "chr", 1: "id", 2: "cm", 3: "pos", 4: "a1", 5: "a2"})

        # Normalize chromosome naming
        bim["chr"] = bim["chr"].apply(norm_chr)
        data["chr"] = data["chr"].apply(norm_chr)

        # Merge on position (and chromosome if available)
        merged = pd.merge(data, bim, on=["pos", "chr"], how="inner")

        # Keep only rows where alleles match or are flipped
        allele_match = (
            ((merged["ref"] == merged["a1"]) & (merged["alt"] == merged["a2"])) |
            ((merged["ref"] == merged["a2"]) & (merged["alt"] == merged["a1"]))
        )

        # Filter
        merged = merged[allele_match]

        print("\tVariants after filtering:", len(merged))

        data = merged.drop(columns=['id', 'cm', 'a1', 'a2'])

    elif USE_REF_ALLELES and ref_table is not None:
        print("\nFiltering by reference allele...")
        # Rename bim columns for clarity
        bim = bim.rename(columns={0: "chr", 1: "id", 2: "cm", 3: "pos", 4: "a1", 5: "a2"})

        # Normalize chromosome naming
        bim["chr"] = bim["chr"].apply(norm_chr)
        data["chr"] = data["chr"].apply(norm_chr)
        ref_table["chr"] = ref_table["chr"].apply(norm_chr)

        # Merge WGSA with FASTA ref
        data = data.merge(
            ref_table,
            on=["chr", "pos"],
            how="inner",
            suffixes=("", "_fasta")
        )

        # Require WGSA ref == FASTA ref
        before = len(data)
        data = data[data["ref"] == data["ref_fasta"]]
        print(f"  Dropped {before - len(data)} WGSA variants with ref mismatch")

        # Merge with BIM
        data = data.merge(
            bim,
            on=["chr", "pos"],
            how="inner"
        )

        # Keep only variants where FASTA ref is in PLINK alleles
        allele_ok = (
            (data["ref_fasta"] == data["a1"]) |
            (data["ref_fasta"] == data["a2"])
        )
        data = data[allele_ok]

        # Determine which PLINK allele is ALT
        def which_alt(row):
            if row["alt"] == row["a1"]:
                return "A1"
            elif row["alt"] == row["a2"]:
                return "A2"
            else:
                return np.nan

        data["plink_alt"] = data.apply(which_alt, axis=1)
        data = data.dropna(subset=["plink_alt"])

        # Flag whether PLINK encoding is flipped
        data["plink_a1_ref"] = data["ref_fasta"] == data["a1"]

        # Print some stats
        n_total = len(data)

        n_a1_ref = data["plink_a1_ref"].sum()
        n_a2_ref = n_total - n_a1_ref

        alt_counts = data["plink_alt"].value_counts()

        print("PLINK allele reconciliation summary:")
        print(f"  Total variants after reconciliation: {n_total:,}")
        print("--------------------------------")
        print(f"    REF = A1: {n_a1_ref:,} ({n_a1_ref/n_total:.2%})")
        print(f"    REF = A2: {n_a2_ref:,} ({n_a2_ref/n_total:.2%})")
        print("--------------------------------")
        print(f"    ALT = A1: {alt_counts.get('A1', 0):,}")
        print(f"    ALT = A2: {alt_counts.get('A2', 0):,}\n")

        data = data.drop(columns=['id', 'cm', 'a1', 'a2', 'plink_a1_ref', 'plink_alt', 'ref_fasta'])
        data['chr'] = data['chr'].apply(de_norm_chr)
        
    else:
        print("\nNo allele filtering applied\n")
        data = data
    
    return data


def add_lof_missense(data):
    path = f"/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/36K_QC/VEP_annotations/chr{CHRO_NB}_LOF.txt"
    lof = pd.read_csv(path, sep = "\t", skiprows=4).reset_index().drop_duplicates("level_1")
    path = f"/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/36K_QC/VEP_annotations/chr{CHRO_NB}_missense.txt"
    missense = pd.read_csv(path, sep = "\t", header = None).drop_duplicates(1)
    variants = np.array(data['pos'].astype(int))
    data['lof'] = np.where(pd.DataFrame(variants).isin(list(lof['level_1'].str.split(":").str[1].astype(int))), 1, 0)
    data['missense'] = np.where(pd.DataFrame(variants).isin(list(missense[1].str.split(":").str[1].astype(int))), 1, 0)

    return data


def annotate(gene_pos, bim, ref_table=None):

    # Get WGSA annotations --> subset variants to only those in wgsa
    wgsa = get_wgsa(bim, ref_table)

    # Add LOF + Missense annotations
    data = add_lof_missense(wgsa)
    
    # Map variants to genes
    variant_genes, gene_variants = map_genes(data, gene_pos)

    variant_num_genes_file = os.path.join(ANNOTATIONS_DIR, "variant_gene_counts.txt")
    variants_summary_df = pd.DataFrame({
        'variant': list(variant_genes.keys()),
        'gene_count': [len(genes) for genes in variant_genes.values()], 
        'unique_gene_count': [len(set(genes)) for genes in variant_genes.values()]

    })

    gene_num_variants_file = os.path.join(ANNOTATIONS_DIR, "gene_variant_counts.txt")
    genes_summary_df = pd.DataFrame({
        'gene': list(gene_variants.keys()),
        'variant_count': [len(variants) for variants in gene_variants.values()],
        'unique_variant_count': [len(set(variants)) for variants in gene_variants.values()]

    })

    variants_summary_df.to_csv(variant_num_genes_file, index=False, sep="\t")
    genes_summary_df.to_csv(gene_num_variants_file, index=False, sep="\t")

    total_genes, genes_with_mapped_variants = 0, 0

    for gene, variants in gene_variants.items():
        total_genes += 1

        if len(variants) == 0 : 
            continue

        genes_with_mapped_variants += 1
        print(f"Getting {gene} variants...")
        print(f"\t ... {len(variants)} total variants ({len(set(variants))} unique)")
        gene_data = data[data['pos'].isin(variants)]
        anno = gene_data.replace(".", 0)
        
        anno[['SpliceAI_DS_AG',	'SpliceAI_DS_AL','SpliceAI_DS_DG', 'SpliceAI_DS_DL']] = anno[['SpliceAI_DS_AG',	'SpliceAI_DS_AL','SpliceAI_DS_DG', 'SpliceAI_DS_DL']].astype(str).applymap(lambda x: max(map(float, x.split(';'))))
        anno['splice'] = anno[['SpliceAI_DS_AG','SpliceAI_DS_AL','SpliceAI_DS_DG', 'SpliceAI_DS_DL']].max(axis = 1)

        gnomad_cols = ['gnomAD_genomes_POPMAX_AF', 'gnomAD_genomes_AFR_AF', 'gnomAD_genomes_AMR_AF', 'gnomAD_genomes_NFE_AF']
        anno[gnomad_cols] = anno[gnomad_cols].replace(".", np.NaN)
        
        anno[gnomad_cols] = anno[gnomad_cols].apply(lambda col: (col.astype(float).fillna(col.astype(float).min() / 2))) # if very rare, large weight.
        anno['EnhancerFinder_brain_enhancer'] = np.where(anno['EnhancerFinder_brain_enhancer']=="Y", 1, 0)
        anno['FANTOM5_CAGE_peak_robust'] = np.where(anno['FANTOM5_CAGE_peak_robust']=="Y", 1, 0)


        # there are some duplicates in wgsa with different splicing scores. keeping highest. TODO: is there a better way to drop duplicates?
        anno = anno.sort_values('splice', ascending=False).drop_duplicates('pos', keep='first').sort_values("pos")
        anno = anno.drop(columns=['SpliceAI_DS_AG', 'SpliceAI_DS_AL', 'SpliceAI_DS_DG', 'SpliceAI_DS_DL'] , axis = 1) 

        # remove non-numeric columns
        anno = anno.drop('ref', axis = 1)
        anno = anno.drop('alt', axis = 1)
        

        for column in anno.columns:
            try:
                anno[column] = anno[column].astype(float)
            except:
                anno = anno.drop(column, axis = 1)
                print(column, "\t COULD NOT BE REPRESENTED AS FLOAT AND IS BEING REMOVED")
        
        anno = anno.dropna(axis = 1).astype(float)
        
        # Determine transcription start site (TSS)
        gene_row = gene_pos[gene_pos['ensgene'] == gene]
        if gene_row.empty:
            print(f"\t Really shouldn't be hitting this case (gene {gene} row not found)!")
            valid = False
        else:
            strand = gene_row["strand"].iloc[0]
            if strand == '+' or strand == 1:
                valid = True
                tss = gene_row["start"].iloc[0]
            elif strand == '-' or strand == -1:
                valid = True
                tss = gene_row["end"].iloc[0]
            else:
                tss = False
                print(f"\t INVALID STRAND VALUE {gene_row['strand']}")
        anno['dist_to_TSS'] = anno['pos'] - tss if valid else 1e10

        # Calculate log10 of absolute distance, adding small epsilon to avoid log10(0) = -inf
        # When distance is 0 (variant at TSS), log10(0 + epsilon) gives a small negative value
        anno['dist_to_TSS'] = np.log10(np.abs(anno['dist_to_TSS']) + 1e-10)

        if len(anno) != 0: # dont save empty df
            anno.to_csv(os.path.join(ANNOTATIONS_DIR, f"{gene}_annotations.tsv.gz"), sep="\t", index=False, compression="gzip")

    print("\n")
    print(f"Total genes: {total_genes}")
    print(f"Genes with mapped variants: {genes_with_mapped_variants}")
    
    return 


def main():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    gene_pos = load_gene_pos()
    print(f"Chromosome {CHRO_NB} \n")
    
    if SET == "Train":
        bim = pd.read_csv(os.path.join(GENOTYPE_PATH, f'clean_chr{CHRO_NB}.bim'), delim_whitespace = True, header = None)
    elif SET == "Test":
        bim = pd.read_csv(os.path.join(GENOTYPE_PATH, f'merged_chr{CHRO_NB}.bim'), delim_whitespace = True, header = None)
    else:
        sys.exit("Error: Invalid set. Must be Train or Test.")

    if REMOVE_DUPS:
        print("Removing duplicate position variants from BIM...")
        total_variants = len(bim)
        duplicate_mask = bim[3].duplicated(keep=False)
        n_duplicates = duplicate_mask.sum()
        n_unique = total_variants - n_duplicates

        print(f"  Total variants: {total_variants}")
        print(f"  Duplicated positions: {n_duplicates} ({n_duplicates / total_variants * 100:.2f}%)")

        # Remove all variants that share a duplicated position
        bim = bim[~duplicate_mask]
        print(f"  Remaining after removing duplicates: {len(bim)}\n")
    
    if USE_REF_ALLELES:
        ref_table = load_reference()
        print("Loading reference table...\n")
    else:
        ref_table = None
        
    print("Annotating variants...\n\n")
    annotate(gene_pos, bim, ref_table)
    
    return
    

if __name__ == "__main__":
    main()





