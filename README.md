# firvTWAS — Functionally-Informed Rare Variant TWAS

**emmental** (Expression Modification Model Encompassing NucleoTide ALterations) is a hierarchical Bayesian model for annotation-informed expression prediction that enables rare-variant inclusion in transcriptome-wide association studies (TWAS). It is the first step of the **firvTWAS** pipeline, which aims to discover Alzheimer's Disease (AD) risk genes by leveraging rare variants in the Alzheimer's Disease Sequencing Project (ADSP).

Related models from the Knowles Lab:
- **gruyere**: [`https://github.com/daklab/gruyere`](https://github.com/daklab/gruyere) — Bayesian RVAT framework for binary traits (used in preliminary analyses, see `prelim/`)
- **parmigiano**: [`https://github.com/daklab/parmigiano`](https://github.com/daklab/parmigiano) — extension of gruyere with two-stage global annotation weight learning; emmental adapts this framework to the expression-prediction setting

---

## Table of Contents
1. [Pipeline Overview](#1-pipeline-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Requirements and Dependencies](#3-system-requirements-and-dependencies)
4. [Data Requirements](#4-data-requirements)
5. [Preprocessing](#5-preprocessing)
6. [emmental: Expression Prediction](#6-emmental-expression-prediction)
   - [Configuration](#61-configuration)
   - [Parameters Reference](#62-parameters-reference)
   - [Stage 1 — Joint Model](#63-stage-1--joint-model)
   - [Stage 2 — Per-Gene Model](#64-stage-2--per-gene-model)
   - [Output Files](#65-output-files)
7. [Reproducing Results](#7-reproducing-results)
8. [Ongoing Work](#8-ongoing-work)

---

## 1. Pipeline Overview

The firvTWAS pipeline has three stages:

```
BigBrain (N=8K, paired WGS + RNA-seq)
         │
         ▼
┌──────────────────────────────────────────┐
│  Step 1: emmental expression prediction  │  ◄── THIS REPO (implemented)
│  Learn variant effect sizes β per gene   │
│  incl. annotation-informed priors for    │
│  unseen rare variants                    │
└──────────────────────────────────────────┘
         │  β̂ for all variants (seen + unseen)
         ▼
┌─────────────────────────────────────────┐
│  Step 2: Expression imputation in ADSP  │  ◄── ONGOING (see §8)
│  Apply β̂ to ADSP WGS (N=31K) to impute  │
│  genome-wide gene expression            │
└─────────────────────────────────────────┘
         │  Imputed expression per gene per individual
         ▼
┌──────────────────────────────────────────┐
│  Step 3: Rare-variant association tests  │  ◄── ONGOING (see §8)
│  Burden + SKAT-O against AD case/control │
└──────────────────────────────────────────┘
```

---

## 2. Repository Structure

```
firvTWAS/
├── README.md
├── prelim/                          # Preliminary analyses using gruyere (see gruyere repo)
├── preprocessing/
│   └── scripts/
│       ├── steps.md                 # Step-by-step preprocessing guide
│       ├── extract_common_participants.py
│       ├── subset_and_merge_expression.py
│       ├── subset_genotype_cohort.sh
│       ├── merge_cohorts_by_chromosome.sh
│       ├── annotate.sh              # Wrapper: map_annotate_variants.py
│       ├── map_annotate_variants.py # Variant annotation (WGSA, LoF, Enformer, ChromBPNet, AlphaMissense, ABC)
│       ├── genotype.sh              # Wrapper: process_genotype.py
│       ├── process_genotype.py      # Per-gene G matrix creation
│       ├── scale_annotations_global.py  # Global annotation scaling
│       ├── fix_annotations.py       # Annotation variant ID repair
│       ├── rename_genotype_mapping.py
│       ├── update_genotype_ids_cohort.sh
│       ├── train_test_split.py
│       ├── subset_expression.py
│       └── slurm/                   # SLURM job submission wrappers
└── emmental/
    └── src/
        ├── config_base.yaml         # Default configuration file
        ├── models.py                # EmmentalJoint and EmmentalPerGene model classes
        ├── load_data.py             # Data loading, annotation processing, DataTensors
        ├── utils.py                 # Logging, argument parsing, covariate preprocessing
        ├── save_outputs.py          # Result serialization (β, τ, R², tau history)
        ├── emmental_joint.py        # Stage 1: joint SVI fit for global τ₁, τ₂, T
        ├── emmental_pergene.py      # Stage 2: per-gene SVI fit for w_g, ρ_g, β_gj
        ├── run_emmental_joint.sh    # SLURM script for Stage 1
        ├── run_emmental_pergene.sh  # SLURM script for Stage 2 (array job, 1 job per chromosome)
        ├── scale_annotations_global.py
        └── genes/
            ├── genes_list_seed.txt          # Seed genes used for Stage 1 joint fit
            ├── 200genes_list_seed_random.txt  # Full gene list for Stage 1 (seed + random)
            └── generate_gene_list.py        # Script to generate gene lists
```

---

## 3. System Requirements and Dependencies

### Hardware
- Minimum 100 GB RAM for per-gene stage; 200 GB for joint stage
- 8 CPU cores recommended (used for parallelism in PLINK and Python)
- GPU optional (PyTorch will use CPU if no GPU is available)
- SLURM cluster recommended for full-genome per-gene runs

### Software
- Python ≥ 3.9
- PLINK 1.9 (for preprocessing)
- samtools (for preprocessing only)
- Conda environment (recommended)

### Python Dependencies

Install via conda or pip:

```bash
pip install torch pyro-ppl scikit-learn pandas numpy pyyaml tqdm
```

| Package | Version tested | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Tensor operations, autograd |
| `pyro-ppl` | ≥ 1.8 | Probabilistic programming, SVI inference |
| `scikit-learn` | ≥ 1.2 | Baseline models (LassoCV, RidgeCV, etc.) |
| `pandas` | ≥ 1.5 | Data manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `pyyaml` | any | Config file parsing |
| `tqdm` | any | Progress bars |

The full Conda environment can be activated on the Knowles Lab cluster:
```bash
source /gpfs/commons/groups/knowles_lab/software/anaconda3/bin/activate
```

---

## 4. Data Requirements

### BigBrain (reference dataset — expression + WGS)
The BigBrain resource must be pre-processed and stored in the following format:

| File | Description |
|---|---|
| `tpm_genes_subset.tsv` | Gene expression matrix (TPM), genes × samples, subset to genes with genotype matrices |
| `covariates.tsv` | Covariate matrix with columns: `sample_id`, `cohort`, `tissue`, `biological_sex`, `age`, ancestry probabilities (`eur_prob`, etc.), genotype PCs, RNA prep type, RNA strandedness, cell-type proportions |
| `genotypes/chr{1..22}/{ENSG}_genotypes.tsv.gz` | Per-gene genotype matrices (samples × variants), one file per gene. Variants within ±100kb of the gene body, biallelic SNPs, MAF > 0.0001 |
| `annotations_raw/chr{1..22}/{ENSG}_annotations.tsv.gz` | Per-gene annotation matrices (variants × 14 annotation features), **unscaled** |
| `annotations_scaled/chr{1..22}/{ENSG}_annotations.tsv.gz` | Per-gene annotation matrices, **globally scaled** (see §5) |
| `bayesian_ridge/` | Pre-computed Bayesian Ridge Regression β and α per gene, used for warm-starting and noise estimation |

BigBrain data access is controlled; see `preprocessing/scripts/steps.md` for the full pipeline to generate these files from raw cohort data.

### ADSP (target dataset — WGS only, step 2 onwards)
ADSP release 4 WGS is available through NIAGADS under controlled access. ADSP preprocessing mirrors BigBrain genotype processing (no expression data required for imputation).

---

## 5. Preprocessing

The full preprocessing pipeline is documented in `preprocessing/scripts/steps.md`. The key steps are:

1. **Extract common participants** across genotype, expression, and covariate files per cohort.
2. **Subset and rename genotype files** — filter to biallelic SNPs, split by chromosome, update participant IDs to sample IDs (duplicating participants with multiple RNA-seq samples).
3. **Merge cohorts** per chromosome into a single PLINK .bed/.bim/.fam file.
4. **Map and annotate variants** — for each gene, extract variants within ±100kb and compute 14 functional annotation scores (`map_annotate_variants.py`). Sources: WGSA (conservation, pathogenicity, splicing), LOFTEE (LoF/missense), AlphaMissense, Enformer, ChromBPNet, ABC.
5. **Build per-gene genotype matrices** (`process_genotype.py`) — align sample order to covariates.tsv, flip alleles if needed, impute missing values.
6. **Scale annotations globally** (`scale_annotations_global.py`) — signed annotations (e.g., Enformer TF-binding deltas) are z-scored; all others are min-max scaled to [0, 1]. Scaling parameters are saved to `minmax.json` and `zscore.json`.
7. **Subset expression** to genes with both genotype and annotation matrices (`subset_expression.py`).

---

## 6. emmental: Expression Prediction

### 6.1 Configuration

All parameters are set via a YAML config file. The default is `emmental/src/config_base.yaml`. Any parameter can be overridden at the command line.

```yaml
# emmental/src/config_base.yaml

# --- Data paths ---
expression_path: '/path/to/tpm_genes_subset.tsv'
covariates_path: '/path/to/covariates.tsv'
genotype_dir:    '/path/to/genotypes/'
annotation_dir:  '/path/to/annotations_scaled/'   
gene_list:       '/path/to/200genes_list_seed_random.txt'
brr_results_dir: '/path/to/bayesian_ridge/'

# --- Model ---
train_test: True          # Hold out ROSMAP DLPFC as test set
maf_beta: 1               # Beta distribution shape for MAF weights
lr: 0.01                  # Learning rate
epochs: 500               # SVI training epochs
n_posterior: 50           # Posterior samples for inference
refits: 1                 # Independent refits (averaged for τ in stage 2)

# --- Annotations (if annotation_dir points to raw, unscaled annotations) ---
annotations: ['chrombpnet', 'dist_to_TSS', 'ABC', 'CADD_raw', 'Eigen-raw',
              'enformer', 'lof', 'missense', 'splice', 'MAP20',
              'alphamissense', 'gnomAD_genomes_POPMAX_AF']

# --- Output ---
joint_output_dir:   'output/joint'
pergene_output_dir: 'output/pergene'
chromosome: 21    # Used by per-gene mode only
```

### 6.2 Parameters Reference

#### Data paths

| Parameter | Type | Description |
|---|---|---|
| `expression_path` | `str` | Path to the TPM gene expression matrix (genes × samples, TSV). |
| `covariates_path` | `str` | Path to the covariates file (samples × covariates, TSV). |
| `genotype_dir` | `str` | Directory containing per-gene genotype files, organized as `chr{N}/{ENSG}_genotypes.tsv.gz`. |
| `annotation_dir` | `str` | Directory containing per-gene annotation files, organized as `chr{N}/{ENSG}_annotations.tsv.gz`. Pass the raw (unscaled) directory if `annotations` is also specified; pass the scaled directory otherwise. |
| `gene_list` | `str` | Path to a text file with one gene per line in `chr{N}/ENSG{ID}` format, or a comma-separated string of gene names. |
| `brr_results_dir` | `str` \| `null` | Directory with pre-computed Bayesian Ridge Regression results. Used for (i) warm-starting β in SVI and (ii) estimating per-gene residual noise σ²_g. If `null`, SVI uses σ=0.5 and random initialization. |

#### Model parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_test` | `bool` | `False` | If `True`, holds out ROSMAP DLPFC (one-to-one sample-to-individual) as the test set. All other samples are used for training. |
| `maf_beta` | `int` | `1` | Shape parameter `b` for Beta(1, b) MAF weights. Higher values upweight rarer variants more aggressively. |
| `lr` | `float` | `0.01` | Learning rate for the ClippedAdam optimizer. |
| `clip_norm` | `float` | `10.0` | Gradient clipping norm for ClippedAdam. Reduces instability from large τ₂ gradients. |
| `epochs` | `int` | `500` | Number of SVI training epochs per run. |
| `n_posterior` | `int` | `50` | Number of posterior samples drawn for computing posterior statistics and R². |
| `refits` | `int` | `10` | Number of independent joint model refits (Stage 1 only). Global parameters τ₁, τ₂, T are averaged across refits before Stage 2. |
| `maf_threshold` | `float` \| `null` | `null` | If set, filters variants below this MAF threshold before fitting. Used to restrict to common variants only. `null` means all variants are retained. |
| `tau1_normal_prior` | `bool` | `False` | If `False` (default), τ₁ ~ Dirichlet(1_q) — constrained to a simplex. If `True`, τ₁ ~ N(0, 1) — allows negative components. |

#### Prior hyperparameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold_prior_alpha` | `float` | `2.0` | Alpha parameter for Beta(α, β) prior on T. Together with `threshold_prior_beta`, sets prior mean = α/(α+β) ≈ 0.09. |
| `threshold_prior_beta` | `float` | `20.0` | Beta parameter for Beta(α, β) prior on T. Higher values concentrate more mass near zero, encouraging a permissive gate. |

#### Annotation selection (for unscaled annotation directories)

| Parameter | Type | Description |
|---|---|---|
| `annotations` | `list[str]` | List of annotation categories to include. Recognized values: `chrombpnet` (expands to ATAC and H3K27ac channels), `enformer` (expands to min/max), `ABC`, `dist_to_TSS`, `CADD_raw`, `Eigen-raw`, `lof`, `missense`, `splice`, `MAP20`, `alphamissense`, `gnomAD_genomes_POPMAX_AF`. When this list is provided, the script sets `annotation_dir` to the raw annotations directory and applies feature engineering (averaging over cell types, clipping dist_to_TSS at 0). |

#### Output paths

| Parameter | Type | Description |
|---|---|---|
| `joint_output_dir` | `str` | Root output directory for Stage 1. Each refit is saved to `{joint_output_dir}/run_{N}/`. A shared `config.yaml` is saved in the root. |
| `pergene_output_dir` | `str` | Root output directory for Stage 2. Results are saved per chromosome at `{pergene_output_dir}/chr{N}/`. |
| `chromosome` | `int` | Chromosome number for Stage 2 (per-gene mode). Typically set via `$SLURM_ARRAY_TASK_ID`. |

### 6.3 Stage 1 — Joint Model

Stage 1 jointly learns the global annotation weights **τ₁, τ₂** and the filter threshold **T** across a set of seed genes.

**Run locally:**
```bash
cd emmental/src
python emmental_joint.py \
    --config config_base.yaml \
    --joint_output_dir output/joint \
    --gene_list genes/200genes_list_seed_random.txt \
    --refits 10 \
    --epochs 500
```

**Run on SLURM cluster:**
```bash
cd emmental/src
sbatch run_emmental_joint.sh \
    --joint_output_dir output/joint \
    --gene_list genes/200genes_list_seed_random.txt \
    --refits 10 \
    --epochs 500
```

**Outputs** (under `{joint_output_dir}/run_{N}/`):

| File | Description |
|---|---|
| `tau_T.csv` | Learned τ₁, τ₂ per annotation and threshold T. Header: `Annotation, Tau1, Tau2, Filter Threshold`. Intercept row for τ₁; NaN in Tau2 column for the intercept row. |
| `tau_history.csv` | τ₁ and τ₂ values at each epoch (for convergence diagnostics). |
| `train_r2_scores.csv` | Per-gene training R². |
| `test_r2_scores.csv` | Per-gene held-out R² (if `train_test: True`). |
| `beta_samples/` | Per-gene β posterior means and std (`{ENSG}_beta.csv.gz`). |
| `losses.txt` | ELBO loss per epoch. |
| `w_g.csv` / `rho_g.csv` | Posterior means and std for gene weights and burden/dispersion mixing parameter. |
| `posterior_stats.npz` | Full posterior statistics dictionary (all named sites). |
| `config.yaml` | Config used for this run (shared across refits). |

Stage 1 runs multiple independent refits (controlled by `--refits`). Averaged τ₁, τ₂, T across refits are computed automatically by Stage 2.

### 6.4 Stage 2 — Per-Gene Model

Stage 2 fixes the global τ₁, τ₂, T from Stage 1 and fits per-gene parameters (w_g, ρ_g, β_gj) for every gene on the relevant chromosome.

**Run locally (single chromosome):**
```bash
cd emmental/src
python emmental_pergene.py \
    --config config_base.yaml \
    --joint_output_dir output/joint \
    --pergene_output_dir output/pergene \
    --chromosome 21
```

**Run on SLURM cluster (all 22 chromosomes in parallel):**
```bash
cd emmental/src
sbatch run_emmental_pergene.sh \
    --joint_output_dir output/joint \
    --pergene_output_dir output/pergene \
    --epochs 500 \
    --lr 0.01
```

The SLURM script uses `--array=1-22`, launching one job per chromosome. The chromosome number is read from `$SLURM_ARRAY_TASK_ID`.

**Outputs** (under `{pergene_output_dir}/chr{N}/`):

| File | Description |
|---|---|
| `tau_T.csv` | Averaged τ₁, τ₂, T from Stage 1 (copied here for traceability). |
| `beta_samples/{ENSG}_beta.csv.gz` | Per-gene β posterior mean and std per variant. Columns: `variant_id_G`, `variant_id_Z`, `beta_mean`, `beta_std`. |
| `mean_samples/{ENSG}_mean.csv.gz` | Per-gene posterior imputed expression mean and std per individual. |
| `train_r2_scores.csv` | Per-gene training R². |
| `test_r2_scores.csv` | Per-gene held-out R². |
| `loss_time_by_gene.csv` | Per-gene, per-epoch ELBO loss and wall-clock time. |
| `config.yaml` | Config used for this chromosome. |

### 6.5 Output Files

#### Key output: `beta_samples/{ENSG}_beta.csv.gz`

This is the primary output of the emmental model. Each row is one variant; each file corresponds to one gene:

| Column | Description |
|---|---|
| `variant_id_G` | Variant ID from the genotype matrix (`chr:pos_counted_other` format) |
| `variant_id_Z` | Variant ID from the annotation matrix (`chr:pos_ref_alt` format) |
| `beta_mean` | Posterior mean of β_gj (the variant effect size on expression) |
| `beta_std` | Posterior std of β_gj |

For variants in ADSP that were absent from BigBrain training, β_gj is assigned via the annotation-derived prior mean (equation 7 in the report). These rows will have `variant_id_G` entries not present in the BigBrain genotype files.

---

## 7. Reproducing Results

The following steps reproduce the expression-prediction results reported in the course report (Table 1 and Figures 1–5 in the report).

### Quick test run (single chromosome, small gene list)

```bash
cd emmental/src

# Stage 1: fit τ on 3 seed genes, chromosome 21
python emmental_joint.py \
    --config config_base.yaml \
    --gene_list genes/gene_list_3.txt \
    --joint_output_dir output/test_joint \
    --epochs 100 \
    --refits 1 \
    --n_posterior 10

# Stage 2: fit per-gene for chromosome 21
python emmental_pergene.py \
    --config config_base.yaml \
    --joint_output_dir output/test_joint \
    --pergene_output_dir output/test_pergene \
    --chromosome 21 \
    --epochs 100
```

Expected outputs in `output/test_joint/run_1/` and `output/test_pergene/chr21/`. Runtime: ~5–10 minutes on CPU for 3 genes.

### Full genome reproduction (cluster)

```bash
cd emmental/src

# Stage 1: 200 seed genes, 10 refits (reproduces τ shown in Figure 1 of report)
sbatch run_emmental_joint.sh \
    --joint_output_dir output/joint \
    --gene_list genes/200genes_list_seed_random.txt \
    --refits 10 \
    --epochs 500 \
    --annotation_dir /path/to/annotations_scaled/ \
    --annotations chrombpnet dist_to_TSS ABC CADD_raw Eigen-raw enformer lof missense splice MAP20 alphamissense gnomAD_genomes_POPMAX_AF

# Stage 2: all chromosomes (reproduces R² results in Table 1 and Figures 2–5)
sbatch run_emmental_pergene.sh \
    --joint_output_dir output/joint \
    --pergene_output_dir output/pergene \
    --epochs 500
```

Estimated runtime: Stage 1 ~4 hours (200 GB RAM, 8 cores). Stage 2 ~6–10 hours per chromosome (100 GB RAM, 8 cores each), fully parallelized across chromosomes via SLURM array.

### Baseline models

Baseline expression-prediction models (Bayesian Ridge, LASSO, Ridge, ElasticNet) are fit as part of the preprocessing pipeline and their results are stored in `brr_results_dir`. Results used in Table 1 were generated with the settings in `config_base.yaml` using `scikit-learn` (LassoCV, RidgeCV, ElasticNetCV, BayesianRidge — see §3.4 of the report).

---

## 8. Ongoing Work

### Step 2 — Expression Imputation in ADSP

Once per-gene β̂ are available from Stage 2, expression is imputed for each ADSP individual as:

$$\hat{Y}_{ig} = \sum_j G_{ij} \hat{\beta}_{gj}$$

For ADSP variants absent from BigBrain, β̂_gj is set to the annotation-derived prior mean (μ_gj), enabling imputation even for the ~97% of ADSP rare variants not observed during training. This is the key rare-variant extension over standard TWAS methods.

**Status:** Implementation in progress. Requires:
- Aligning ADSP genotype variant IDs with the annotation coordinate system.
- Running annotation scoring pipeline for ADSP variants using the same 14 annotation categories and globally learned τ₁, τ₂, T.
- Matrix multiply G_ADSP × β̂ per gene.

### Step 3 — Rare-Variant Association Tests in ADSP

Imputed expression per gene will be tested for association with AD case/control status using rare-variant aggregation tests, following the approach established in gruyere/parmigiano:

- **Burden test**: sum-of-effects test, sensitive to consistent directional effects.
- **SKAT-O**: adaptive combination of burden and variance-component test, robust to mixed-direction effects.

Covariates: sex, age, APOE-ε4/ε2 genotypes, 20 ancestry PCs, common variant PRS, sequencing platform, sequencing center, cohort, and a sparse ancestry-adjusted GRM from FastSparseGRM.

**Status:** Pending Step 2 completion.

---

## Notes

- The `prelim/` directory contains preliminary analyses using the **gruyere** model applied directly to BigBrain expression data. This is separate from the emmental pipeline and uses the gruyere codebase ([`https://github.com/daklab/gruyere`](https://github.com/daklab/gruyere)).
- SLURM output and error logs from `run_emmental_joint.sh` are automatically moved to `{joint_output_dir}/` after the job completes. Logs from `run_emmental_pergene.sh` are moved to `{pergene_output_dir}/`.
- All Python scripts accept `--help` for a full list of CLI arguments.