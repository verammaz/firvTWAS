import random
import os

N_GENES = 240

BRR_DIR_FULL = "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full/bayesian_ridge/betas"
BRR_DIR_LOG1P = "/gpfs/commons/home/adas/uTWAS/src/results/baseline_log1p/bayesian_ridge/betas"


if __name__ == "__main__":

    with open("genes_list_seed.txt", "r") as f:
        seed_genes = f.read().splitlines()

    # get genes from BRR directory
    genes_full = [] 
    genes_log1p = []

    for chrom in range(1,23):
        g_genes_full = [f"chr{chrom}/{f.split('.')[0]}" for f in os.listdir(os.path.join(BRR_DIR_FULL, f"chr{chrom}")) 
                    if f.endswith(".tsv.gz") and f not in seed_genes]
        g_genes_log1p = [f"chr{chrom}/{f.split('.')[0]}" for f in os.listdir(os.path.join(BRR_DIR_LOG1P, f"chr{chrom}")) 
                    if f.endswith(".tsv.gz") and f not in seed_genes]

        genes_full.extend(g_genes_full)
        genes_log1p.extend(g_genes_log1p)
   
    # remove seed genes from gene_list
    gene_list_full = [gene for gene in genes_full if gene not in seed_genes]
    gene_list_log1p = [gene for gene in genes_log1p if gene not in seed_genes]

    # save gene_list to file
    with open("gene_list_all_full.txt", "w") as f:
        for gene in gene_list_full:
            f.write(gene + "\n")
    with open("gene_list_all_log1p.txt", "w") as f:
        for gene in gene_list_log1p:
            f.write(gene + "\n")

    # shuffle gene_list
    random.shuffle(gene_list_full)
    random.shuffle(gene_list_log1p)

    # take first N_GENES genes
    gene_list_full = gene_list_full[:N_GENES]
    gene_list_log1p = gene_list_log1p[:N_GENES]
    
    total_genes = N_GENES + len(seed_genes)

    # write gene_list to file
    with open(f"{total_genes}genes_list_seed_random_full.txt", "w") as f:
        for gene in gene_list_full:
            f.write(gene + "\n")
        for gene in seed_genes:
            f.write(gene + "\n")

    # with open(f"{total_genes}genes_list_seed_random_log1p.txt", "w") as f:
    #     for gene in gene_list_log1p:
    #         f.write(gene + "\n")
    #     for gene in seed_genes:
    #         f.write(gene + "\n")

