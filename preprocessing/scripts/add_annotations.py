import pandas as pd
import os
import sys


CHROMBPNET_DIR = "/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/58K_preview_compact/rare_variants/chrombpnet/variant_peak_pairs_scored"
def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: python add_annotations.py <genotype_file> <annotation_file>")
        sys.exit(1)
    genotype_file = args[0]
    annotation_file = args[1]
    genotype_df = pd.read_csv(genotype_file, sep="\t")
    annotation_df = pd.read_csv(annotation_file, sep="\t")
    print(genotype_df.head())