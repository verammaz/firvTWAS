import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml


def read_pergene_regression_outputs(outdir):
    results_list = []
    for chrom in range(1,23):
        file = os.path.join(outdir, f"chr{chrom}_gene_association_results.csv")
        df = pd.read_csv(file)
        results_list.append(df)

    results = pd.concat(results_list, axis=0)   # vertically
    return results


def get_genes(num_genes, results, random=False, top=100): 
    # if random = False, select top num_genes by pvalue
    # if random = True, select random subset of num_genes (from top or all)
    if not random:
        results = results.sort_values('pval', ascending=False)
        top_genes = results.head(num_genes)
        print(f"There were {len(results)} significant genes. Taking top {min(num_genes, len(top_genes))}.\n")
        genes = top_genes[['chrom', 'gene']]

    else:
        sample_size = min(num_genes, len(results))
        if top is not None:
            results = results.sort_values('pval', ascending=False)
            top_genes = results.head(top)
            random_genes = top_genes.sample(n=sample_size, replace=False, random_state=42)
            print(f"Randomly selected {sample_size} genes out of top {top}.\n")

        else:
            random_genes = results.sample(n=sample_size, replace=False, random_state=42)
            print(f"Randomly selected {sample_size} genes out of {len(results)}.\n")
        genes = random_genes[['chrom', 'gene']]

    return genes



def main():
    params_file = sys.argv[1]
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    
    outdir = os.path.join(params['output'], 'gene_association')
    regression_results = read_pergene_regression_outputs(outdir)
    num_genes = params.get('num_genes', 50)
    joint_analysis_genes = get_genes(num_genes, regression_results)
    joint_analysis_genes.to_csv(params['genes'], header=None, index=None, sep="\t")


if __name__ == "__main__":
    main()

