import numpy as np
import pandas as pd
import os
import time
from dataclasses import dataclass, field
import torch
import utils
from tqdm import tqdm
from collections import defaultdict


def get_logger():
    return utils.get_logger()


def load_residualized_covariates(config, device):
    """
    Load covariates and expression, residualize expression on covariates.
    Uses config['covariates_path'] and config['expression_path'].
    Returns (covariates_scaled, residualized_Y, train_idx, test_idx).
    train_idx/test_idx are integer numpy arrays for subsetting.
    """
    logger = get_logger()
    covariates = pd.read_csv(config['covariates_path'], sep="\t").set_index("sample_id")
    tpm = pd.read_csv(config['expression_path'], sep="\t").set_index("feature")
    logger.info(f"# Genes overall: {len(tpm)}")
    tpm = tpm[covariates.index]
    chr_gene = utils.get_chr_gene(tpm, config['genes'])
    tpm = tpm.loc[chr_gene['feature']]
    logger.info(f"# Genes for fit: {len(tpm)}")

    # Train/test split: hold out ROSMAP DLPFC as test
    train_mask = ~(
        (covariates['cohort'] == "ROSMAP") &
        (covariates['tissue'] == "Dorsolateral Pre-frontal Cortex (DLPFC)")
    )
    test_mask = (
        (covariates['cohort'] == "ROSMAP") &
        (covariates['tissue'] == "Dorsolateral Pre-frontal Cortex (DLPFC)")
    )
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    logger.info(f"Train samples: {len(train_idx):,}")
    logger.info(f"Test samples:  {len(test_idx):,}")

    covariate_cols = [
        'biological_sex', 'eas_prob', 'afr_prob', 'amr_prob', 'sas_prob', 'eur_prob',
        'tissue', 'age', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'cohort',
        'rna_lib_prep_type', 'rna_strandedness', 'astrocyte', 'endothelial_cell',
        'excitatory_neuron', 'inhibitory_neuron', 'microglia', 'oligodendrocyte',
        'oligodendrocyte_progenitor_cell', 'others', 'pericyte'
    ]
    covariates_scaled = utils.preprocess_covariates(covariates, covariate_cols)
    tpm_scaled = utils.scale_tpm_matrix(tpm, median_filter=0, scale_center=config.get('scale_center', True))
    logger.info(f"Residualizing expression for {len(tpm_scaled)} genes...")
    residualized_Y = {}
    for idx, GENE in enumerate(tpm_scaled.index, 1):
        try:
            result = utils.residualize_expression_single_gene(tpm_scaled.loc[GENE], covariates_scaled, device)
            residualized_Y[GENE.split(".")[0]] = result
        except Exception as e:
            import traceback
            logger.error(f"ERROR residualizing gene {idx} ({GENE}): {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    logger.info(f"Completed residualization for {len(residualized_Y)} genes")
    return covariates_scaled, residualized_Y, train_idx, test_idx



# simple model -- trying to get negative annotations
def process_chrombpnet_dist_only(Z, config, logger):
    """
    Process chrombpnet and dist_to_TSS annotations for chrombpnet_dist_only mode.
    """

    if config.get('chrombpnet_dist_only', False):
        logger.info(f"Keeping only chrombpnet and dist_to_TSS annotations...")
    
    logger.info(f"Taking mean of chrombpnet annotations...")
    Z['chrombpnet'] = Z.filter(like="chrombpnet").mean(axis=1)
    keep_columns = ['chrombpnet', 'dist_to_TSS']
    Z = Z[keep_columns]

    ANNOTATIONS_RAW = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations/"
    ANNOTATIONS_MINMAX = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations_minmax/"

    if config.get('chrombpnet_dist_only_cfg_num', 1) == 1:
        assert config.get('annotation_dir', None) == ANNOTATIONS_RAW, "chrombpnet_dist_only_cfg_num 1 requires raw annotations directory"
        logger.info(f"Z-scoring chrombpnet...")
        Z['chrombpnet'] = (Z['chrombpnet'] - Z['chrombpnet'].mean()) / Z['chrombpnet'].std()        
        logger.info(f"Clipping chrombpnet at 10, keeping sign...")
        Z['chrombpnet'] = np.clip(Z['chrombpnet'], -10, 10)
        logger.info(f"Clipping dist_to_TSS at 0...")
        Z['dist_to_TSS'] = Z['dist_to_TSS'].clip(lower=0)
    elif config.get('chrombpnet_dist_only_cfg_num', 1) == 2:
        assert config.get('annotation_dir', None) == ANNOTATIONS_RAW, "chrombpnet_dist_only_cfg_num 2 requires raw annotations directory"
        logger.info(f"Z-scoring chrombpnet...")
        Z['chrombpnet'] = (Z['chrombpnet'] - Z['chrombpnet'].mean()) / Z['chrombpnet'].std()        
        logger.info(f"Clipping chrombpnet at 10, keeping sign...")
        Z['chrombpnet'] = np.clip(Z['chrombpnet'], -10, 10)
        logger.info(f"Taking 1/dist_to_TSS and minmax scaling...")
        Z['dist_to_TSS'] = 1/Z['dist_to_TSS']
        Z['dist_to_TSS'] = (Z['dist_to_TSS'] - Z['dist_to_TSS'].min()) / (Z['dist_to_TSS'].max() - Z['dist_to_TSS'].min())
    elif config.get('chrombpnet_dist_only_cfg_num', 1) == 3:
        assert config.get('annotation_dir', None) == ANNOTATIONS_MINMAX, "chrombpnet_dist_only_cfg_num 3 requires minmax scaled annotations directory"
        # already loaded minmax scaled annotations
    else:
        raise ValueError(f"chrombpnet_dist_only_cfg_num must be 1, 2, or 3, got {config.get('chrombpnet_dist_only_cfg_num', 1)}")
    
    return Z


def process_annotations(Z, config, logger):
    """
    Process annotations.
    """
    logger.info(f"Processing annotations...")
    annotations = config.get('annotations')
    Z_chrombonet_dist = process_chrombpnet_dist_only(Z.copy(), config, logger)
    Z_cols = []
    for annotation in annotations:
        logger.info(f"Processing annotation: {annotation}")
        if annotation == 'chrombpnet' or annotation == 'dist_to_TSS':
            continue
        elif annotation == 'ABC':
            Z_cols.append('ABC')
            Z['ABC'] = Z.filter(like="ABC").mean(axis=1)
            # these should be positive
            assert Z['ABC'].min() >= 0, "ABC annotations should be positive"
        elif annotation == 'CADD_raw':
            Z_cols.append('CADD_raw')
            # z-scale
            Z['CADD_raw'] = (Z['CADD_raw'] - Z['CADD_raw'].mean()) / Z['CADD_raw'].std()
        elif annotation == 'Eigen-raw':
            # z-scale
            Z_cols.append('Eigen-raw')
            Z['Eigen-raw'] = (Z['Eigen-raw'] - Z['Eigen-raw'].mean()) / Z['Eigen-raw'].std()
        elif annotation == 'enformer':
            # average _TF_delta_min over cell types
            Z_cols.append('enformer_min')
            Z['enformer_min'] = Z.filter(like="_TF_delta_min").mean(axis=1)
            # average _TF_delta_max over cell types
            Z_cols.append('enformer_max')
            Z['enformer_max'] = Z.filter(like="_TF_delta_max").mean(axis=1)
        elif annotation in ['lof', 'missense', 'splice', 'MAP20', 'alphamissense']:
            Z_cols.append(annotation)
            Z[annotation] = Z.filter(like=annotation).mean(axis=1)
    Z = Z[Z_cols]
    Z = pd.concat([Z_chrombonet_dist, Z], axis=1)
    return Z
        

def load_genes(config, genotype_dir=None, annotation_dir=None):
    """
    Load genotype and annotation matrices.
    Returns (G, Z, variant_ids_G, variant_ids_Z) where variant_ids are lists of
    raw variant ID strings from the genotype columns and annotation index respectively.

    INPUT:
        - config: configuration dictionary
            Option 1: genotype_path + annotation_path (pre-generated matrices)
            Option 2: genotype_dir + annotation_dir (per-gene files)
        - genotype_dir / annotation_dir: optional overrides for config values
    OUTPUT:
        - G: genotype DataFrame (N x P)
        - Z: annotation DataFrame (P x Q)
        - variant_ids_G: list of raw variant ID strings from G columns (chr:pos_a1_a2 format)
        - variant_ids_Z: list of raw variant ID strings from Z index (chr:pos_ref_alt format)
    """
    logger = get_logger()
    sample_col = config.get('sample_col', 'SampleID')

    if genotype_dir is None:
        genotype_dir = config.get('genotype_dir')
    if annotation_dir is None:
        annotation_dir = config.get('annotation_dir')

    variant_ids_G = []  # raw IDs from genotype columns
    variant_ids_Z = []  # raw IDs from annotation index

    # Option 1: pre-generated matrices
    if 'genotype_path' in config and 'annotation_path' in config:
        G = pd.read_csv(config['genotype_path'], sep='\t', index_col=0)
        Z = pd.read_csv(config['annotation_path'], sep='\t', index_col=0)
        variant_ids_G = G.columns.tolist()
        variant_ids_Z = Z.index.tolist()

    # Option 2: per-gene files
    elif genotype_dir is not None and annotation_dir is not None:
        G_list = []
        Z_list = []

        for gene in tqdm(config['genes']):
            try:
                g_path = os.path.join(genotype_dir, f"{gene}_genotypes.tsv.gz")
                g = pd.read_csv(g_path, sep='\t', index_col=0)
                g.index.name = 'IID'

                z_path = os.path.join(annotation_dir, f"{gene}_annotations.tsv.gz")
                z = pd.read_csv(z_path, sep='\t', index_col=0)
                # Handle malformed index (chr stored as column)
                if z.index.name == "chr":
                    z = z.reset_index()
                    z.index = z['chr'] + ":" + z['pos'].astype(str)
                    z.index.name = 'variant_id'
                z.index.name = 'variant_id'

                # Collect raw variant IDs before prefixing
                variant_ids_G.extend(g.columns.tolist())
                variant_ids_Z.extend(z.index.tolist())

                if len(g.columns) != len(z.index):
                    logger.warning(f"Skipping {gene}: genotype variants ({len(g.columns)}) "
                                   f"!= annotation rows ({len(z.index)})")
                    # Remove the IDs we just added since we're skipping
                    del variant_ids_G[-len(g.columns):]
                    del variant_ids_Z[-len(z.index):]
                    continue

                # Prefix variant names with gene for uniqueness
                g.columns = [f"{gene}_{col}" for col in g.columns]
                z.index = [f"{gene}_{idx}" for idx in z.index]

                G_list.append(g)
                Z_list.append(z)

            except Exception as e:
                logger.warning(f"Error loading gene {gene}: {e}. Skipping.")
                continue

        G = pd.concat(G_list, axis=1)
        Z = pd.concat(Z_list, axis=0)

    else:
        raise ValueError("Config must specify either (genotype_path, annotation_path) "
                         "or (genotype_dir, annotation_dir)")

    # Debug: check for duplicates before processing
    if Z.index.duplicated().sum() > 0:
        logger.warning(f"Z has {Z.index.duplicated().sum()} duplicate indices before cleaning")
    if G.columns.duplicated().sum() > 0:
        logger.warning(f"G has {G.columns.duplicated().sum()} duplicate columns before cleaning")

    # Strip allele suffixes and chr prefix: chr/GENE_chr:pos(_A1_A2) --> GENE_chr:pos
    G.columns = (G.columns.str.split("_").str[0:2].str.join("_")).str.split("/").str[-1]
    Z.index = (Z.index.str.split("_").str[0:2].str.join("_")).str.split("/").str[-1]

    logger.info(f"Genotype matrix: {G.shape}, Annotation matrix: {Z.shape}")

    # Verify alignment
    if set(G.columns) != set(Z.index):
        g_only = set(G.columns) - set(Z.index)
        z_only = set(Z.index) - set(G.columns)
        logger.error(f"Variants in G but not Z: {g_only}")
        logger.error(f"Variants in Z but not G: {z_only}")
        raise ValueError("Genotype columns must match annotation rows")

    Z = Z.loc[G.columns]  # reorder Z to match G column order

    # Feature engineering
    logger.info(f"Feature engineering for annotations...")
    if config.get('chrombpnet_dist_only', False): # only two annotations (chrombpnet and dist_to_TSS)
        Z = process_chrombpnet_dist_only(Z, config, logger)

    if config.get('annotations', []):
        Z = process_annotations(Z, config, logger)
    
    else: # keep all annotations 
        logger.info(f"Dropping promoter_3000 and promoter_2000 annotations...")
        Z = Z.drop(['promoter_3000', 'promoter_2000'], axis=1, errors='ignore')
        log_cols = Z.filter(like="log_counts").columns
        logger.info(f"Taking max of chrombpnet og_counts annotations...")
        Z["chromBPnet"] = Z[log_cols].max(axis=1)
        Z = Z.drop(columns=log_cols)
        enformer_min = Z.filter(like="TF_delta_min").columns
        logger.info(f"Taking max of TF_delta_min annotations...")
        Z["TF_delta_min"] = Z[enformer_min].max(axis=1)
        Z = Z.drop(columns=enformer_min)
        enformer_max = Z.filter(like="TF_delta_max").columns
        logger.info(f"Taking max of TF_delta_max annotations...")
        Z["TF_delta_max"] = Z[enformer_max].max(axis=1)
        Z = Z.drop(columns=enformer_max)

    G, Z = utils.impute_missing(G, Z)
    logger.info(f"Loaded {G.shape[0]} individuals, {G.shape[1]} variants")
    logger.info(f"Annotation matrix: {Z.shape[1]} annotations per variant")

    return G, Z, variant_ids_G, variant_ids_Z


def load_annotations_only(config, annotation_dir=None):
    """
    Load only the variant × annotation matrix Z (no genotype matrix G, no expression).

    Use for low-memory diagnostics (e.g. Z·τ, exp(Z·τ₂)) where genotypes are not needed.
    Mirrors `load_genes` annotation I/O and feature engineering, but skips reading genotype files.

    Does not apply BRR variant filtering (that requires G columns).

    Returns
    -------
    Z : pd.DataFrame
    gene_indices : dict
    gene_names : list
    variant_ids_Z : list
    """
    logger = get_logger()
    if annotation_dir is None:
        annotation_dir = config.get("annotation_dir")
    if annotation_dir is None:
        raise ValueError("annotation_dir required for load_annotations_only")

    variant_ids_Z = []
    Z_list = []
    gene_indices = {}
    gene_names = []
    cursor = 0

    for gene in tqdm(config["genes"]):
        try:
            z_path = os.path.join(annotation_dir, f"{gene}_annotations.tsv.gz")
            z = pd.read_csv(z_path, sep="\t", index_col=0)
            if z.index.name == "chr":
                z = z.reset_index()
                z.index = z["chr"] + ":" + z["pos"].astype(str)
                z.index.name = "variant_id"
            z.index.name = "variant_id"

            variant_ids_Z.extend(z.index.tolist())
            z.index = [f"{gene}_{idx}" for idx in z.index]
            Z_list.append(z)
            nvar = len(z)
            gene_indices[gene] = (cursor, cursor + nvar)
            cursor += nvar
            gene_names.append(gene)
        except Exception as e:
            logger.warning(f"Error loading annotations for gene {gene}: {e}. Skipping.")
            continue

    if not Z_list:
        raise ValueError("No annotation data loaded (empty gene list or all genes failed).")

    Z = pd.concat(Z_list, axis=0)

    if Z.index.duplicated().sum() > 0:
        logger.warning(f"Z has {Z.index.duplicated().sum()} duplicate indices before cleaning")

    Z.index = (Z.index.str.split("_").str[0:2].str.join("_")).str.split("/").str[-1]
    logger.info(f"Annotation-only matrix: {Z.shape[0]} variants × {Z.shape[1]} features")

    # no processing of annotations -- done by caller if needed

    if Z.isnull().any().any():
        n_missing_z = Z.isnull().sum().sum()
        Z = Z.fillna(0)
        logger.info(f"Imputed {n_missing_z} missing annotation values with zeros")

    logger.info(f"Annotation matrix (only): {Z.shape[1]} annotations per variant")
    return Z, gene_indices, gene_names, variant_ids_Z


@dataclass
class AnnotationTensors:
    """
    Minimal container for per-gene Z slices (no G, Y, or MAF). For memory-light diagnostics.
    """

    Z: torch.Tensor
    gene_indices: dict
    gene_names: list
    device: torch.device

    def get_gene_data(self, gene_name):
        start_idx, end_idx = self.gene_indices[gene_name]
        Z_gene = self.Z[start_idx:end_idx, :]
        empty = torch.zeros(0, dtype=torch.float32, device=self.device)
        return empty, Z_gene, empty


def load_brr_results(config):
    """
    Load Bayesian Ridge Regression results.
    Returns a dictionary of gene names and betas and alphas.
    """
    logger = get_logger()
    brr_results_dir = config.get('brr_results_dir')

    if not os.path.exists(brr_results_dir):
        logger.error(f"BRR results directory {brr_results_dir} does not exist")
        return None
    if not os.path.exists(os.path.join(brr_results_dir, 'betas')):
        logger.error(f"BRR betas directory {os.path.join(brr_results_dir, 'betas')} does not exist")
        return None

    brr_results = defaultdict(dict)
    for gene in tqdm(config['genes']):
        # assume gene is in the format chr/ENSG00000000000
        chr = gene.split("/")[0]
        ens = gene.split("/")[1]
        result_file = os.path.join(brr_results_dir, chr + ".tsv")
        beta_file = os.path.join(brr_results_dir, 'betas', gene + ".tsv.gz")
        brr_results['betas'][gene.split("/")[1]] = pd.read_csv(beta_file, sep="\t", index_col=0) # gene name without chr/ prefix to match G and Z matrices
        brr_df = pd.read_csv(result_file, sep="\t", index_col=0)
        brr_results['alphas'][gene.split("/")[1]] = brr_df.loc[ens]['alpha']

    logger.info(f"Total number of betas: {sum(len(brr_results['betas'][gene_name]) for gene_name in brr_results['betas'])}")
    logger.info(f"Total number of alphas: {len(brr_results['alphas'])}") # should be == number of genes
    assert len(config['genes']) == len(brr_results['betas']) == len(brr_results['alphas']), "Number of genes must match between config and BRR results"
 
    return brr_results


@dataclass
class DataTensors:
    """
    Container for all input data tensors required by Parmigiano model.
    Works for both joint (tau/threshold inferred) and per-gene (tau/threshold pre-loaded) modes.
    """
    G: torch.Tensor            # Genotype matrix (N x P)
    Z: torch.Tensor            # Annotation matrix (P x Q)
    X: torch.Tensor            # Covariate matrix (N x K)
    Y: torch.Tensor            # Phenotype: dict of tensors (joint) or single tensor (per-gene)
    maf_weights: torch.Tensor  # MAF-based variant weights (P,)
    gene_indices: dict         # Maps gene_name -> (start_idx, end_idx) for columns in G
    gene_names: list           # Ordered list of gene names
    device: torch.device
    num_anno: int = 0
    num_cov: int = 0
    num_genes: int = 0
    brr_betas: dict = None        # Bayesian Ridge Regression betas (gene_name -> pandas DataFrame)
    brr_alphas: dict = None        # Bayesian Ridge Regression alphas (gene_name -> float)
    # Per-gene specific: loaded externally from tauT files and alpha_dict
    tau: torch.Tensor = None       # Annotation weights (Q,) - set externally in per-gene mode
    threshold: torch.Tensor = None # Filter threshold - set externally in per-gene mode
    std: torch.Tensor = None       # Observation noise std (from alpha_dict) - per-gene mode

    def __post_init__(self):
        self.num_anno = self.Z.shape[1]
        self.num_cov = self.X.shape[1]
        self.num_genes = len(self.gene_names)

    def get_gene_data(self, gene_name):
        """
        Extract genotype and annotation data for a specific gene.
        Returns (G_gene, Z_gene, maf_weights_gene).
        """
        start_idx, end_idx = self.gene_indices[gene_name]
        G_gene = self.G[:, start_idx:end_idx]
        Z_gene = self.Z[start_idx:end_idx, :]
        maf_weights_gene = self.maf_weights[start_idx:end_idx]
        return G_gene, Z_gene, maf_weights_gene

    @staticmethod
    def from_pandas(G, Z, X, Y, brr_betas, brr_alphas, device, config):
        """
        Create DataTensors from pandas DataFrames.
        G columns must follow the pattern GENE_variantID (gene prefix separated by _).
        """
        logger = get_logger()
        assert set(X.index) == set(G.index), "Sample IDs must match between covariates and genotype files"
        assert set(G.columns) == set(Z.index), "Variant IDs must match between genotype columns and annotation index"

        G = G.loc[X.index]  # align sample order

        # TODO: make this part more efficient 

        # sync G and Z matrices with BRR results (keep only variants for which we have betas)
        if brr_betas is not None:
            keep_variants = []
            logger.info(f"Syncing G and Z matrices with BRR results (keep only variants for which we have betas)")
            # G columns: Variant ids --> GENE_chr:pos
            # BRR betas: Variant ids --> chr:pos_a1_a2
            for variant_id in tqdm(G.columns):
                gene = variant_id.split('_')[0]
                chr, pos = variant_id.split('_')[1].split(':') 
                brr_variants = brr_betas[gene].index.tolist()
                brr_variants_chr_pos = [variant.split('_')[0] for variant in brr_variants]
                brr_variant_pos = [variant.split(':')[1].split('_')[0] for variant in brr_variants]
                # shouldnt be a problem. if it is, need to be more careful matching up variants by id
                assert len(brr_variant_pos) == len(set(brr_variant_pos)), "Duplicate variant positions in BRR results" 
                if chr + ":" + pos in brr_variants_chr_pos:
                    keep_variants.append(variant_id)
            # subset G and Z matrices to keep only variants for which we have betas
            G = G[keep_variants]
            Z = Z.loc[keep_variants]
            


        # Parse gene names and indices from column names (GENE_variantID)
        gene_indices = {}
        gene_names = []
        current_gene = None
        start_idx = 0        
        
        for idx, col in enumerate(G.columns):
            gene = col.split('_')[0]
            if gene != current_gene:
                if current_gene is not None:
                    gene_indices[current_gene] = (start_idx, idx)
                    gene_names.append(current_gene)
                current_gene = gene
                start_idx = idx

        if current_gene is not None:
            gene_indices[current_gene] = (start_idx, len(G.columns))
            gene_names.append(current_gene)


        G_tensor = torch.as_tensor(G.values, dtype=torch.float32, device=device)

        return DataTensors(
            G=G_tensor,
            Z=torch.as_tensor(Z.values, dtype=torch.float32, device=device),
            X=torch.as_tensor(X.values, dtype=torch.float32, device=device),
            Y=Y,
            maf_weights=torch.as_tensor(
                utils.get_MAF_weights(G_tensor, device, config['beta']),
                dtype=torch.float32,
                device=device
            ),
            brr_betas=brr_betas,
            brr_alphas=brr_alphas,
            gene_indices=gene_indices,
            gene_names=gene_names,
            device=device
        )

# ---------------------------------------------------------------------------
# Chromosome gene discovery
# ---------------------------------------------------------------------------

def get_chr_genes(config):
    path = os.path.join(config['genotype_dir'], config['chromosome'])
    genes = []
    for i in os.listdir(path):
        if i.endswith("_genotypes.tsv.gz"):
            genes.append(config['chromosome'] + "/" + i.split("_")[0])
    return genes


# ---------------------------------------------------------------------------
# Load pre-computed tau and threshold
# ---------------------------------------------------------------------------

def load_tau_T(config, device, Z):
    """Load tau weights and filter threshold from a directory of per-run files.
    Supports both linear (tau.csv, threshold.csv) and NN (tau1.csv, tau2.csv, b1.csv) modes.
    """
    logger = utils.get_logger()
    path = config['tauT_path']

    def load_runs(filename):
        dfs = []
        for iteration in os.listdir(path):
            if "run" not in iteration: continue
            try:
                dfs.append(pd.read_csv(os.path.join(path, iteration, filename), index_col = 0))
            except Exception as e:
                print(f"Issue loading {filename} from {iteration}: {e}")
        return pd.concat(dfs, axis=0)

    if os.path.exists(os.path.join(path, "run_1", "tau_T.csv")):
        tauT_df = load_runs("tau_T.csv")
        threshold = torch.as_tensor(float(tauT_df['Filter Threshold'].values[0]), dtype=torch.float32, device=device)
        if config.get('tau12', False): # nonlinear mode
            tau1 = torch.as_tensor(tauT_df['Tau1'].groupby("annotation", sort = False).mean().values, dtype=torch.float32, device=device)
            tau2 = torch.as_tensor(tauT_df['Tau2'].groupby("annotation", sort = False).mean().values, dtype=torch.float32, device=device)
            return {'tau1': tau1, 'tau2': tau2, 'threshold': threshold}
        else:
            tau = torch.as_tensor(tauT_df['Tau'].groupby("annotation", sort = False).mean().values, dtype=torch.float32, device=device)
            return {'tau': tau, 'threshold': threshold}

    return None

