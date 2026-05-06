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


def variant_maf_series(G: pd.DataFrame) -> pd.Series:
    """
    Per-column MAF from dosage matrix G (samples × variants), matching
    torch-side convention: round dosages, mean/2, mirror for MAF > 0.5.
    """
    arr = np.asarray(G.values, dtype=np.float64)
    maf = np.round(arr).mean(axis=0) / 2.0
    maf = np.where(maf > 0.5, 1.0 - maf, maf)
    return pd.Series(maf, index=G.columns)


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
    tpm_scaled = utils.scale_tpm_matrix(tpm, median_filter=0)
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



def process_annotations(Z, config, logger):
    """
    Process annotations.
    Assumes annotations are already scaled.
    Clips dist_to_TSS to 0.
    Subsets to certain annotations only.
    Averages over cell types if needed.
    """

    logger.info(f"Processing annotations...")
    annotations = config.get('annotations')
    Z_cols = []
    for annotation in annotations:
        logger.info(f"Processing annotation: {annotation}")
        if annotation == 'chrombpnet':
            Z_cols.append('chrombpnet_ATAC')
            Z['chrombpnet_ATAC'] = Z.filter(like="chrombpnet_ATAC").mean(axis=1) #average over cell types (assume already scaled)
            Z_cols.append('chrombpnet_H3K27ac')
            Z['chrombpnet_H3K27ac'] = Z.filter(like="chrombpnet_H3K27ac").mean(axis=1) #average over cell types (assume already scaled)
        elif annotation == 'dist_to_TSS':
            Z_cols.append('dist_to_TSS')
            Z['dist_to_TSS'] = Z['dist_to_TSS'].clip(lower=0)
        elif annotation == 'ABC':
            Z_cols.append('ABC')
            Z['ABC'] = Z.filter(like="ABC").mean(axis=1) #average over cell types 
            # these should be positive
            assert Z['ABC'].min() >= 0, "ABC annotations should be positive"
        elif annotation == 'enformer':
            # average _TF_delta_min over cell types
            Z_cols.append('enformer_min')
            Z['enformer_min'] = Z.filter(like="_TF_delta_min").mean(axis=1)
            # average _TF_delta_max over cell types
            Z_cols.append('enformer_max')
            Z['enformer_max'] = Z.filter(like="_TF_delta_max").mean(axis=1)
        elif annotation.startswith('gnomAD'): 
            Z_cols.append('gnomAD_genomes_POPMAX_AF') # keep as is (globally scaled annotations should be passed in config['annotation_dir'])
        else:
            if annotation not in Z.columns:
                logger.warning(f"Annotation {annotation} not found in Z columns. Skipping.")
                continue
            Z_cols.append(annotation)
    Z = Z[Z_cols]
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
        assert set(variant_ids_G) == set(variant_ids_Z), "Variant IDs in G and Z do not match"

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
                    logger.error(f"{gene}: genotype variants ({len(g.columns)}) "
                                   f"!= annotation rows ({len(z.index)})... skipping gene")
                    # remove the variant ids that we just added
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
  
    if config.get('annotations', []):
        Z = process_annotations(Z, config, logger)
    
    else:
        logger.warning("No annotations provided. Using all annotations.")

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
        gene_key = gene.split("/")[1]

        # Missing BRR files should not crash training; downstream warm start can default to zero.
        if not os.path.exists(beta_file):
            logger.warning(f"Missing BRR beta file for {gene}; continuing without BRR beta warm start for this gene.")
            continue
        if not os.path.exists(result_file):
            logger.warning(f"Missing BRR summary file for {gene} at {result_file}; skipping BRR for this gene.")
            continue

        brr_results['betas'][gene_key] = pd.read_csv(beta_file, sep="\t", index_col=0) # gene name without chr/ prefix to match G and Z matrices
        brr_df = pd.read_csv(result_file, sep="\t", index_col=0)
        if ens in brr_df.index:
            brr_results['alphas'][gene_key] = brr_df.loc[ens]['alpha']
        else:
            logger.warning(f"Missing alpha row for {gene} in {result_file}; skipping BRR alpha for this gene.")

    logger.info(f"Total number of betas: {sum(len(brr_results['betas'][gene_name]) for gene_name in brr_results['betas'])}")
    logger.info(f"Total number of alphas: {len(brr_results['alphas'])}") # should be == number of genes
    if len(brr_results['betas']) == 0:
        logger.warning("No BRR beta files were found for requested genes; proceeding without BRR warm starts.")
        return None
 
    return brr_results


@dataclass
class DataTensors:
    """
    Container for all input data tensors required by Emmental model.
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
    variant_column_names: list = field(default_factory=list)  # G columns after BRR / MAF filters
    # Per-gene specific: loaded externally from tauT files and alpha_dict
    tau1: torch.Tensor = None       # Annotation weights (Q,) - set externally in per-gene mode
    tau2: torch.Tensor = None       # Annotation weights (Q,) - set externally in per-gene mode
    threshold: torch.Tensor = None # Filter threshold - set externally in per-gene mode

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
    def from_pandas(
        G,
        Z,
        X,
        Y,
        brr_betas,
        brr_alphas,
        device,
        config,
        train_G_for_maf_filter=None,
        forced_variant_columns=None,
    ):
        """
        Create DataTensors from pandas DataFrames.
        G columns must follow the pattern GENE_variantID (gene prefix separated by _).

        Parameters
        ----------
        train_G_for_maf_filter : pd.DataFrame, optional
            Training genotypes (same variant columns as G). When ``common_variants_only``
            is True, MAF is computed on this frame so test tensors do not use test-set MAF.
        forced_variant_columns : list, optional
            If set (e.g. from a training DataTensors run), subset to exactly these columns
            after BRR sync so train/test share the same variant set.
        """
        logger = get_logger()
        assert set(X.index) == set(G.index), "Sample IDs must match between covariates and genotype files"
        assert set(G.columns) == set(Z.index), "Variant IDs must match between genotype columns and annotation index"

        G = G.loc[X.index]  # align sample order

        # TODO: make this part more efficient 

        # Keep all variants; BRR is only used for warm initialization downstream.
        # Variants without BRR betas are initialized to zero there.
        if brr_betas is not None:
            n_total = len(G.columns)
            n_matched = 0
            logger.info("Checking BRR coverage over genotype variants (without dropping unmatched variants)")
            # G columns: GENE_chr:pos
            # BRR betas index: chr:pos_a1_a2
            for variant_id in tqdm(G.columns):
                gene = variant_id.split('_')[0]
                chr, pos = variant_id.split('_')[1].split(':') 
                if gene not in brr_betas:
                    continue
                brr_variants = brr_betas[gene].index.tolist()
                brr_variants_chr_pos = [variant.split('_')[0] for variant in brr_variants]
                brr_variant_pos = [variant.split(':')[1].split('_')[0] for variant in brr_variants]
                # shouldnt be a problem. if it is, need to be more careful matching up variants by id
                assert len(brr_variant_pos) == len(set(brr_variant_pos)), "Duplicate variant positions in BRR results" 
                if chr + ":" + pos in brr_variants_chr_pos:
                    n_matched += 1
            logger.info(
                f"BRR coverage: {n_matched}/{n_total} variants matched; "
                f"{n_total - n_matched} unmatched variants will be kept and zero-initialized."
            )

        if forced_variant_columns is not None:
            use_cols = [c for c in forced_variant_columns if c in G.columns]
            missing = set(forced_variant_columns) - set(G.columns)
            if missing:
                logger.warning(
                    f"forced_variant_columns: {len(missing)} variants not present in G after BRR sync; "
                    f"keeping {len(use_cols)}/{len(forced_variant_columns)}"
                )
            G = G[use_cols]
            Z = Z.loc[use_cols]

        maf_thr = config.get("maf_threshold", None)
        # Accept unset/null-like string values from CLI/YAML without crashing.
        if maf_thr is not None and not (isinstance(maf_thr, str) and maf_thr.strip().lower() in {"", "none", "null"}):
            thr = float(maf_thr)
            ref = train_G_for_maf_filter if train_G_for_maf_filter is not None else G
            shared = [c for c in G.columns if c in ref.columns]
            if len(shared) < len(G.columns):
                logger.warning(
                    f"common_variants_only: reference genotypes missing {len(G.columns) - len(shared)} columns; "
                    "using intersection for MAF."
                )
            ref_maf = ref.loc[:, [c for c in shared if c in ref.columns]]
            maf = variant_maf_series(ref_maf)
            keep_cols = maf[maf >= thr].index.tolist()
            dropped = len(shared) - len(keep_cols)
            G = G[keep_cols]
            Z = Z.loc[keep_cols]
            logger.info(
                f"MAF >= {thr} on training reference -> kept {len(keep_cols)} variants "
                f"({dropped} dropped as rare/low-MAF)."
            )
            
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
        variant_column_names = list(G.columns)

        return DataTensors(
            G=G_tensor,
            Z=torch.as_tensor(Z.values, dtype=torch.float32, device=device),
            X=torch.as_tensor(X.values, dtype=torch.float32, device=device),
            Y=Y,
            maf_weights=torch.as_tensor(
                utils.get_MAF_weights(G_tensor, device, config['maf_beta']),
                dtype=torch.float32,
                device=device
            ),
            brr_betas=brr_betas,
            brr_alphas=brr_alphas,
            gene_indices=gene_indices,
            gene_names=gene_names,
            device=device,
            variant_column_names=variant_column_names,
        )

# ---------------------------------------------------------------------------
# Chromosome gene discovery
# ---------------------------------------------------------------------------

def get_chr_genes(config):
    # for pergene mode, we need to load the genes for the specific chromosome
    logger = get_logger()
    genopath = os.path.join(config['genotype_dir'], f"chr{config['chromosome']}")
    anopath = os.path.join(config['annotation_dir'], f"chr{config['chromosome']}")
    genes = []
    for i in os.listdir(anopath):
        if i.endswith("_annotations.tsv.gz") and os.path.exists(os.path.join(genopath, i.replace("_annotations.tsv.gz", "_genotypes.tsv.gz"))):
            genes.append(f"chr{config['chromosome']}/{i.split('_')[0]}")
    logger.info(f"Found {len(genes)} genes for chromosome {config['chromosome']}")
    return genes


# ---------------------------------------------------------------------------
# Load pre-computed tau and threshold (joint outputs → per-gene stage)
# ---------------------------------------------------------------------------

def read_tau_T_csv(csv_path: str) -> pd.DataFrame:
    """Read ``tau_T.csv`` as written by ``save_results`` (``index=False``)."""
    return pd.read_csv(csv_path)


def discover_joint_run_directories(joint_root: str):
    """
    Return directories containing ``tau_T.csv`` for averaging.

    - If ``joint_root/run_*/tau_T.csv`` exist, return those ``run_*`` paths sorted by run index.
    - Else if ``joint_root/tau_T.csv`` exists, return ``[joint_root]``.
    """
    joint_root = os.path.abspath(joint_root)
    if not os.path.isdir(joint_root):
        raise FileNotFoundError(f"joint output directory not found: {joint_root}")

    run_dirs = []
    for name in sorted(os.listdir(joint_root)):
        if not name.startswith("run_"):
            continue
        suffix = name.split("_", 1)[-1]
        if not suffix.isdigit():
            continue
        rd = os.path.join(joint_root, name)
        if os.path.isdir(rd) and os.path.isfile(os.path.join(rd, "tau_T.csv")):
            run_dirs.append((int(suffix), rd))
    run_dirs = [rd for _, rd in sorted(run_dirs, key=lambda x: x[0])]

    if run_dirs:
        return run_dirs
    if os.path.isfile(os.path.join(joint_root, "tau_T.csv")):
        return [joint_root]
    raise FileNotFoundError(
        f"No tau_T.csv found under {joint_root} (expected run_*/tau_T.csv or a single tau_T.csv)."
    )


def aggregate_tau_t_from_joint_runs(joint_root: str) -> pd.DataFrame:
    """
    Load ``tau_T.csv`` from each joint refit and average ``Filter Threshold`` and τ columns.

    Assumes compatible row ordering across runs (same joint config). Uses nanmean for
    ``Tau2`` when an ``intercept`` row carries NaN in ``Tau2`` (nonlinear + tau1 intercept).
    """
    logger = get_logger()
    run_dirs = discover_joint_run_directories(joint_root)
    dfs = [read_tau_T_csv(os.path.join(rd, "tau_T.csv")) for rd in run_dirs]
    logger.info(f"Averaging τ / T over {len(dfs)} joint run(s) from {joint_root}")

    th = float(np.mean([float(d["Filter Threshold"].iloc[0]) for d in dfs]))

    ann = dfs[0]["Annotation"].astype(str)
    assert "intercept" == ann[0], "intercept must be the first annotation"
    tau1_stack = np.stack([d["Tau1"].to_numpy(dtype=np.float64) for d in dfs], axis=0)
    tau2_stack = np.stack([d["Tau2"].to_numpy(dtype=np.float64) for d in dfs], axis=0)
    tau1_mean = tau1_stack.mean(axis=0)
    tau2_mean = np.nanmean(tau2_stack, axis=0)
    assert np.isnan(tau2_mean[0]), "intercept for tau2 should be NaN"
    out = pd.DataFrame(
        {
            "Annotation": ann,
            "Tau1": tau1_mean,
            "Tau2": tau2_mean,
            "Filter Threshold": th,
        }
    )
    return out


def _tau_vectors_from_summary_df(mean_df: pd.DataFrame):
    """
    From a single summary ``tau_T`` dataframe, return numpy vectors for the generative model.

    Returns
    -------
    tau1 : ndarray or None  (includes intercept coefficient first when applicable)
    tau2 : ndarray or None  (length = number of annotation columns in Z, no intercept)
    """
    tau1 = mean_df["Tau1"].to_numpy(dtype=np.float32)
    ann = mean_df["Annotation"].astype(str)
    assert ann.str.lower().eq("intercept").any(), "expecting intercept for tau1"
    mask = ~ann.str.lower().eq("intercept")
    tau2 = mean_df.loc[mask, "Tau2"].to_numpy(dtype=np.float32)
    return tau1, tau2


def load_tau_threshold(config):
    """
    Load (or average over joint refits) global τ and threshold T.

    Uses ``config['joint_output_dir']`` as the directory that either
    contains ``run_*/tau_T.csv`` or a flat ``tau_T.csv``.

    Returns
    -------
    mean_df : pd.DataFrame
    threshold : float
    tau1 : ndarray or None
    tau2 : ndarray or None
    """
    root = config.get("joint_output_dir")
    if not root:
        raise ValueError("config must set joint_output_dir to load τ and T from joint outputs.")

    mean_df = aggregate_tau_t_from_joint_runs(root)
    th = float(mean_df["Filter Threshold"].iloc[0])
    tau1, tau2 = _tau_vectors_from_summary_df(mean_df)
    return mean_df, th, tau1, tau2


def load_tau_threshold_tensors(config, device):
    """Same as ``load_tau_threshold`` but returns torch tensors in a dict for training code."""
    _mean_df, th, tau1, tau2 = load_tau_threshold(config)
    threshold = torch.as_tensor(th, dtype=torch.float32, device=device)
    return {
            "tau1": torch.as_tensor(tau1, dtype=torch.float32, device=device),
            "tau2": torch.as_tensor(tau2, dtype=torch.float32, device=device),
            "threshold": threshold,
        }
    

