import os
import pandas as pd

EXPRESSION_MATRIX = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/tpm.tsv"
GENOTYPE_DIR = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/genotypes"
SUBSET_EXPRESSION_MATRIX = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/tpm_genes_subset.tsv"

def main():
    # Load expression matrix
    tpm = pd.read_csv(EXPRESSION_MATRIX, sep='\t').set_index('feature')

    genes = [] # use genes with genotype matrices (instead of annotation matrices)
    # annotation matrices per gene created before MAF filtering --> some genes could be dropped due to no variats passing MAF and QC

    for chrom in range(1,23):
        g_genes = [f.split("_")[0] for f in os.listdir(os.path.join(GENOTYPE_DIR, f"chr{chrom}")) 
                  if f.endswith("_genotypes.tsv.gz")]
        genes.extend(g_genes)
    
    print(f"Number of genes in expression matrix: {len(tpm)}")
    print(f"Number of genes with genotype (and annotation) matrices: {len(genes)}")

    # Subset expression matrix to include only genes with genotype matrices
    tpm['gene'] = tpm.index.str.split('.').str[0]
    tpm.set_index('gene', inplace=True)
    tpm_filtered = tpm[tpm.index.isin(genes)]
    tpm_filtered.index.name = 'feature' # rename for consistency with original tpm matrix

    print(f"Number of genes in expression matrix after subsetting: {len(tpm_filtered)}")
    print(f"Subsetted expression matrix saved to: {SUBSET_EXPRESSION_MATRIX}")
    tpm_filtered.to_csv(SUBSET_EXPRESSION_MATRIX, sep='\t')

    return 

if __name__ == "__main__":
    main()