import os
import yaml
import pandas as pd
from get_joint_analysis_genes import read_pergene_regression_outputs, get_genes

def write_genelist(path, series):
    series.to_csv(path, header=False, index=False, sep="\t")

def generate_runs(
    gene_assoc_dir,
    runs_root,
    template_config,
    num_genes=50,
    n_random_subsets=20,
    top=100
):
    os.makedirs(runs_root, exist_ok=True)

    # Load regression results
    results = read_pergene_regression_outputs(gene_assoc_dir)

    run_dirs = []

    # ----------- Run 0: All Top genes -----------
    top_genes = get_genes(top, results)
    run0 = os.path.join(runs_root, "joint_model_run_0")
    os.makedirs(run0, exist_ok=True)
    write_genelist(os.path.join(run0, "genes.txt"), top_genes)
    run_dirs.append(run0)

    # ----------- Run 1: Top 50 genes -----------
    top_genes = get_genes(num_genes, results)
    run1 = os.path.join(runs_root, "joint_model_run_1")
    os.makedirs(run1, exist_ok=True)
    write_genelist(os.path.join(run1, "genes.txt"), top_genes)
    run_dirs.append(run1)

    # ----------- Random subsets -----------
    for i in range(n_random_subsets):
        runi = os.path.join(runs_root, f"joint_model_run_{i+2}")
        os.makedirs(runi, exist_ok=True)

        # sample from the top pool
        sampled = get_genes(num_genes, results, random=True, top=top)
        write_genelist(os.path.join(runi, "genes.txt"), sampled)

        run_dirs.append(runi)

    # ----------- Write config for each run -----------
    with open(template_config) as f:
        base = yaml.safe_load(f)

    for i, r in enumerate(run_dirs):
        cfg = base.copy()
        cfg["genes"] = os.path.join(r, "genes.txt")
        cfg["output"] = os.path.join(r, "outputs")
        config_path = os.path.join(r, "config.yaml")

        with open(config_path, "w") as out:
            yaml.dump(cfg, out)

    print("Generated runs:")
    for r in run_dirs:
        print("   ", r)


if __name__ == "__main__":
    import sys
    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)  
    
    generate_runs(
        gene_assoc_dir=params['gene_assoc'],
        runs_root=sys.argv[2],
        template_config=params_file
    )
