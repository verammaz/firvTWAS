import random
import os

N_GENES = 140

BRR_DIR = "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge/betas"
GENOTYPE_DIR = '/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/genotypes/'
ANNOTATION_DIR = '/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_minmax/'


if __name__ == "__main__":

    with open("genes_list_seed.txt", "r") as f:
        seed_genes = f.read().splitlines()

    # get genes from BRR directory
    genes = [] 

    for chrom in range(1,23):
        g_genes = [f"chr{chrom}/{f.split('.')[0]}" for f in os.listdir(os.path.join(BRR_DIR, f"chr{chrom}")) 
                    if f.endswith(".tsv.gz") and f not in seed_genes]

        genes.extend(g_genes)
   
    # remove seed genes from gene_list
    gene_list = [gene for gene in genes if gene not in seed_genes]

    # shuffle gene_list
    random.shuffle(gene_list)

    # take first N_GENES genes
    gene_list = gene_list[:N_GENES]
    

    # write gene_list to file
    with open("genes_list_seed_random.txt", "w") as f:
        for gene in gene_list:
            f.write(gene + "\n")
        for gene in seed_genes:
            f.write(gene + "\n")

