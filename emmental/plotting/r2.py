import os
import glob
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from collections import defaultdict

BATCH_SIZE = 10

METRICS = [
    ("prop_train_gt_001", r"Train: frac $R^2>0.01$"),
    ("prop_train_gt_01", r"Train: frac $R^2>0.1$"),
    ("prop_test_gt_001", r"Test: frac $R^2>0.01$"),
    ("prop_test_gt_01", r"Test: frac $R^2>0.1$"),
]

MEAN_R2_METRICS = [
    ("mean_r2_train", "Mean train R² (across genes)"),
    ("mean_r2_test", "Mean test R² (across genes)"),
]

BASELINE_ROOTS = {
    "common01": "/gpfs/commons/home/adas/uTWAS/src/results/baseline_full_common01",
}
DEFAULT_TOP200_GENES_PATH = (
    "/gpfs/commons/home/vmazeeva/firvTWAS/emmental/src/genes/top200_BRR_genes.txt"
)
MODEL_NAMES = {
    "bayesian_ridge": "Bayesian Ridge",
    "lasso": "Lasso",
    "elasticnet": "Elastic Net",
    "ridge": "Ridge",
}
BASELINE_METHODS = ["ridge", "lasso", "bayesian_ridge", "elasticnet"]
BASELINE_BAR_LABELS = [MODEL_NAMES[m] for m in BASELINE_METHODS]
THRESHOLDS = [0, 0.01, 0.1]

# Post-train R² grid (matches emmental_post_train_betas.py)
COMMON_BETA_SOURCES = ("mu", "full", "common", "train")
RARE_BETA_SOURCES = ("mu", "full", "rare")
BETA_SOURCE_LABELS = {
    "mu": "μ (ρ·w·λ)",
    "full": "β_hat G_full",
    "common": "β_hat G_common",
    "train": "saved training β",
    "rare": "β_hat G_rare",
}
COMMON_BETA_SHORT = {
    "mu": "common: μ",
    "full": "common: G_full",
    "common": "common: G_common",
    "train": "common: saved β",
}
PERGENE_COMMON01_METHOD_ORDER = BASELINE_BAR_LABELS + ["Emmental pergene"]

# Colors for common-only vs +rare assemblies in panel proportion plots.
PANEL_VARIANT_COLORS: dict[str | None, str] = {
    None: "#8c96a8",
    "mu": "#4C72B0",
    "full": "#55A868",
    "rare": "#C44E52",
}


def _panel_variants() -> list[tuple[str | None, str]]:
    """(rare_beta, legend label) for common-only and +rare assemblies."""
    return [
        (None, "common only"),
        ("mu", f"+rare: {BETA_SOURCE_LABELS['mu']}"),
        ("full", f"+rare: {BETA_SOURCE_LABELS['full']}"),
        ("rare", f"+rare: {BETA_SOURCE_LABELS['rare']}"),
    ]


def post_train_r2_col(common_beta: str, rare_beta: str | None = None) -> str:
    """Column name in ``train|test_r2_scores.csv`` for one β-source choice."""
    common_beta = _validate_beta_source(common_beta, rare=False)
    if rare_beta is None:
        return f"r2_common_only__c_{common_beta}"
    rare_beta = _validate_beta_source(rare_beta, rare=True)
    return f"r2_common_plus_rare__c_{common_beta}__r_{rare_beta}"


def _validate_beta_source(source: str, *, rare: bool) -> str:
    allowed = RARE_BETA_SOURCES if rare else COMMON_BETA_SOURCES
    if source not in allowed:
        raise ValueError(
            f"Invalid {'rare' if rare else 'common'} beta source {source!r}; "
            f"expected one of {allowed}"
        )
    return source


_COMMON_BETA_RE = "(mu|full|common|train)"
_RARE_BETA_RE = "(mu|full|rare)"


def parse_post_train_r2_col(col: str) -> tuple[str, str, str | None]:
    mobj = re.fullmatch(rf"r2_common_only__c_{_COMMON_BETA_RE}", col)
    if mobj:
        return "common_only", mobj.group(1), None
    mobj = re.fullmatch(
        rf"r2_common_plus_rare__c_{_COMMON_BETA_RE}__r_{_RARE_BETA_RE}", col
    )
    if mobj:
        return "common_plus_rare", mobj.group(1), mobj.group(2)
    raise ValueError(f"Not a post-train R² column: {col!r}")


def emmental_model_key(common_beta: str, rare_beta: str | None = None) -> str:
    common_beta = _validate_beta_source(common_beta, rare=False)
    if rare_beta is None:
        return f"emmental__c_{common_beta}"
    rare_beta = _validate_beta_source(rare_beta, rare=True)
    return f"emmental__c_{common_beta}__r_{rare_beta}"


def emmental_panel_label(common_beta: str, rare_beta: str | None = None) -> str:
    """Short Emmental assembly label for axes (common-only vs common+rare)."""
    _validate_beta_source(common_beta, rare=False)
    if rare_beta is None:
        return "Emmental common-only"
    _validate_beta_source(rare_beta, rare=True)
    return "Emmental common+rare"


def emmental_beta_assembly_label(common_beta: str, rare_beta: str | None = None) -> str:
    """Human-readable β sources for legends / figure annotations."""
    c = BETA_SOURCE_LABELS[_validate_beta_source(common_beta, rare=False)]
    if rare_beta is None:
        return f"common: {c}"
    r = BETA_SOURCE_LABELS[_validate_beta_source(rare_beta, rare=True)]
    return f"common: {c}; rare: {r}"


def _emmental_legend_label_from_model_key(model_key: str) -> str:
    """Legend entry: panel name plus β sources, e.g. ``common-only (common: ...)``."""
    mobj = re.fullmatch(
        rf"emmental__c_{_COMMON_BETA_RE}(?:__r_{_RARE_BETA_RE})?", str(model_key)
    )
    if mobj is None:
        return _display_label(model_key)
    common_beta, rare_beta = mobj.group(1), mobj.group(2)
    return (
        f"{emmental_panel_label(common_beta, rare_beta)}"
        f" ({emmental_beta_assembly_label(common_beta, rare_beta)})"
    )


def _set_ax_title_with_subtitle(
    ax,
    title: str,
    subtitle: str | None = None,
    *,
    subtitle_scale: float = 0.82,
) -> None:
    """Main title plus smaller β-source subtitle above the axes."""
    if not subtitle:
        ax.set_title(title)
        return
    title_obj = ax.set_title(title, y=1.04, pad=8)
    ax.text(
        0.5,
        1.0,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=title_obj.get_fontsize() * subtitle_scale,
        color="#444444",
    )


def beta_source_subtitle(
    common_beta: str,
    rare_beta: str | None = None,
) -> str:
    """Subtitle string for Emmental β sources, e.g. ``β sources: c=common, rare=rare``."""
    if rare_beta is not None:
        return rf"$\beta$ sources: c={common_beta}, rare={rare_beta}"
    return rf"$\beta$ sources: c={common_beta}"


def _beta_source_from_model_keys(names: list[str]) -> str | None:
    """Build β-source subtitle from ``emmental__c_*`` model keys."""
    configs: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for name in names:
        mobj = re.fullmatch(
            rf"emmental__c_{_COMMON_BETA_RE}(?:__r_{_RARE_BETA_RE})?", str(name)
        )
        if mobj is None:
            continue
        cfg = (mobj.group(1), mobj.group(2))
        if cfg not in seen:
            seen.add(cfg)
            configs.append(cfg)
    if not configs:
        return None
    parts: list[str] = []
    seen_suffix: set[str] = set()
    for common_beta, rare_beta in configs:
        suffix = beta_source_subtitle(common_beta, rare_beta)
        if suffix not in seen_suffix:
            seen_suffix.add(suffix)
            parts.append(suffix)
    return " · ".join(parts)


def beta_assembly_legend_from_configs(
    configs: list[tuple[str, str | None]],
) -> str:
    """One-line β assembly annotation for multiple Emmental configs."""
    parts: list[str] = []
    seen: set[str] = set()
    for common_beta, rare_beta in configs:
        line = (
            f"{emmental_panel_label(common_beta, rare_beta)}"
            f" ({emmental_beta_assembly_label(common_beta, rare_beta)})"
        )
        if line not in seen:
            seen.add(line)
            parts.append(line)
    return " · ".join(parts)


def emmental_display_label(common_beta: str, rare_beta: str | None = None) -> str:
    """Short display label for one Emmental β assembly (alias of ``emmental_panel_label``)."""
    return emmental_panel_label(common_beta, rare_beta)


def prop_above(path: str, fname: str, thr: float) -> float | None:
    """Fraction of genes (CSV rows) with R² > thr for this run/split."""
    fp = os.path.join(path, fname)
    if not os.path.isfile(fp):
        return None
    try:
        df = pd.read_csv(fp)
        if "r2" not in df.columns:
            return None
        r = df["r2"].astype(float)
        if len(r) == 0:
            return None
        return float(np.mean(r > thr))
    except Exception:
        return None


def mean_r2(path: str, fname: str) -> float | None:
    """Mean per-gene R² in one split (train or test CSV)."""
    fp = os.path.join(path, fname)
    if not os.path.isfile(fp):
        return None
    try:
        df = pd.read_csv(fp)
        if "r2" not in df.columns:
            return None
        r = df["r2"].astype(float)
        if len(r) == 0:
            return None
        return float(r.mean())
    except Exception:
        return None


def run_level_metrics(run_path: str) -> dict | None:
    """Per run: threshold proportions + mean R² (train/test)."""
    t001_tr = prop_above(run_path, "train_r2_scores.csv", 0.01)
    t01_tr = prop_above(run_path, "train_r2_scores.csv", 0.1)
    t001_te = prop_above(run_path, "test_r2_scores.csv", 0.01)
    t01_te = prop_above(run_path, "test_r2_scores.csv", 0.1)
    m_tr = mean_r2(run_path, "train_r2_scores.csv")
    m_te = mean_r2(run_path, "test_r2_scores.csv")
    if any(
        x is None
        for x in (t001_tr, t01_tr, t001_te, t01_te, m_tr, m_te)
    ):
        return None
    return {
        "prop_train_gt_001": t001_tr,
        "prop_train_gt_01": t01_tr,
        "prop_test_gt_001": t001_te,
        "prop_test_gt_01": t01_te,
        "mean_r2_train": m_tr,
        "mean_r2_test": m_te,
    }


def load_train_test_r2_per_gene(run_path: str) -> pd.DataFrame | None:
    """Per run: columns gene, r2_train, r2_test."""
    tp = os.path.join(run_path, "train_r2_scores.csv")
    sp = os.path.join(run_path, "test_r2_scores.csv")
    if not (os.path.isfile(tp) and os.path.isfile(sp)):
        return None
    try:
        a = pd.read_csv(tp)
        b = pd.read_csv(sp)
        if "gene" not in a.columns or "r2" not in a.columns:
            return None
        if "gene" not in b.columns or "r2" not in b.columns:
            return None
        m = a[["gene", "r2"]].merge(b[["gene", "r2"]], on="gene", suffixes=("_train", "_test"))
        if m.empty:
            return None
        m["r2_train"] = m["r2_train"].astype(float)
        m["r2_test"] = m["r2_test"].astype(float)
        return m[["gene", "r2_train", "r2_test"]]
    except Exception:
        return None



def gather_per_gene_per_batch_averages(
    runs: list[str],
    exp_label: str,
    batch_size: int | None = None,
    load_fn=None,
    gene_list: set[str] | None = None,
) -> pd.DataFrame:
    """
    load_fn: optional loader (default ``load_train_test_r2_per_gene``).
    gene_list: if set, keep only these gene IDs.

    For each configuration: group runs into consecutive batches of up to
    ``batch_size``. The final batch may be smaller. Within each batch, for each
    gene, average train R² and test R² across those runs.

    Returns one row per (configuration, batch_id, gene) with columns
    mean_r2_train, mean_r2_test. With 100 runs and batch size 10 you get
  10 batches; with 5 runs and batch size 10 you get 1 batch of 5 runs.
    Genes missing in any run of the batch are dropped.
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if load_fn is None:
        load_fn = load_train_test_r2_per_gene
    rows = []
    for b_start in range(0, len(runs), batch_size):
        chunk = runs[b_start : b_start + batch_size]
        chunk_n = len(chunk)
        if chunk_n == 0:
            continue
        dfs = [load_fn(rp) for rp in chunk]
        if any(d is None for d in dfs):
            continue
        parts = []
        for i, d in enumerate(dfs):
            dd = d.copy()
            dd["_run"] = i
            parts.append(dd)
        long_df = pd.concat(parts, ignore_index=True)
        g = long_df.groupby("gene", as_index=False).agg(
            mean_r2_train=("r2_train", "mean"),
            mean_r2_test=("r2_test", "mean"),
            n=("r2_train", "count"),
        )
        g = g[g["n"] == chunk_n].drop(columns=["n"])
        if gene_list is not None:
            g = g[g["gene"].isin(gene_list)]
        g["batch_id"] = b_start // batch_size
        g["exp_label"] = exp_label
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def gather_run_level_proportion_batches(
    experiments: list[dict],
    r2_stats_full: dict,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Chunk runs in order into batches of up to ``batch_size``.

    Each row is one batch: **mean of per-run** gene fractions (``prop_*`` from
    ``run_level_metrics``). The final batch may be smaller. Batches with any
    failed run (``None``) are dropped, matching ``gather_per_gene_per_batch_averages``.
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    rows = []
    for exp in experiments:
        label = exp["label"]
        n_runs = len(exp["runs"])
        metrics = r2_stats_full[exp["joint_root"]]
        if len(metrics) != n_runs:
            raise ValueError(
                f"{len(metrics)} metrics vs {n_runs} runs"
            )
        for b_start in range(0, n_runs, batch_size):
            chunk = metrics[b_start : b_start + batch_size]
            if any(m is None for m in chunk):
                continue
            rows.append(
                {   "exp_label": label,
                    "batch_id": b_start // batch_size,
                    "prop_train_gt_001": float(
                        np.mean([m["prop_train_gt_001"] for m in chunk])
                    ),
                    "prop_train_gt_01": float(
                        np.mean([m["prop_train_gt_01"] for m in chunk])
                    ),
                    "prop_test_gt_001": float(
                        np.mean([m["prop_test_gt_001"] for m in chunk])
                    ),
                    "prop_test_gt_01": float(
                        np.mean([m["prop_test_gt_01"] for m in chunk])
                    ),
                }
            )
    return pd.DataFrame(rows)

def load_train_test_r2_per_gene_from_chr_tree(base_path: str) -> pd.DataFrame | None:
    """Per run or pergene root: merge ``chr*/train|test_r2_scores.csv`` if present, else single-dir CSVs."""
    import glob

    base_path = os.path.abspath(base_path)
    chr_dirs = sorted(
        d
        for d in glob.glob(os.path.join(base_path, "chr*"))
        if os.path.isdir(d)
        and os.path.isfile(os.path.join(d, "train_r2_scores.csv"))
        and os.path.isfile(os.path.join(d, "test_r2_scores.csv"))
    )
    if chr_dirs:
        parts = []
        for cd in chr_dirs:
            d = load_train_test_r2_per_gene(cd)
            if d is None:
                return None
            parts.append(d)
        out = pd.concat(parts, ignore_index=True)
        if out.empty:
            return None
        if out["gene"].duplicated().any():
            out = out.groupby("gene", as_index=False).agg(
                r2_train=("r2_train", "mean"),
                r2_test=("r2_test", "mean"),
            )
        return out
    return load_train_test_r2_per_gene(base_path)


def discover_pergene_run_paths(pergene_root: str) -> list[str]:
    """
    Refit layout: ``pergene/run_*`` (R² at run root and/or under ``run_*/chr*``).
    Single-fit layout: ``pergene/chr*`` only → returns ``[pergene_root]``.
    """
    root = os.path.abspath(pergene_root)
    if not os.path.isdir(root):
        return []

    runs: list[str] = []
    for name in sorted(os.listdir(root)):
        if not re.fullmatch(r"run_\d+", name):
            continue
        rd = os.path.join(root, name)
        if not os.path.isdir(rd):
            continue
        if os.path.isfile(os.path.join(rd, "train_r2_scores.csv")):
            runs.append(rd)
            continue
        if glob.glob(os.path.join(rd, "chr*", "train_r2_scores.csv")):
            runs.append(rd)
    if runs:
        return runs

    if glob.glob(os.path.join(root, "chr*", "train_r2_scores.csv")):
        return [root]
    return []


def gather_pergene_batch_df(
    pergene_root: str,
    exp_label: str,
    batch_size: int | None = None,
    gene_list: set[str] | None = None,
) -> pd.DataFrame:
    """
    Build the same gene-level batch table as joint ``gather_per_gene_per_batch_averages``,
    but discover runs from a pergene tree (chr-only single fit or ``run_*`` refits).
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    runs = discover_pergene_run_paths(pergene_root)
    if not runs:
        return pd.DataFrame()

    load_fn = load_train_test_r2_per_gene_from_chr_tree

    # Single refit aggregated across chromosomes → one batch (no run averaging).
    if len(runs) == 1:
        d = load_fn(runs[0])
        if d is None or d.empty:
            return pd.DataFrame()
        if gene_list is not None:
            d = d[d["gene"].isin(gene_list)]
        g = d.rename(columns={"r2_train": "mean_r2_train", "r2_test": "mean_r2_test"})
        g["batch_id"] = 0
        g["exp_label"] = exp_label
        return g[["gene", "mean_r2_train", "mean_r2_test", "batch_id", "exp_label"]]

    return gather_per_gene_per_batch_averages(
        runs, exp_label, batch_size=batch_size, load_fn=load_fn, gene_list=gene_list
    )


def batch_proportions_from_batch_df(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of genes above R² thresholds per (exp_label, batch_id)."""
    if batch_df.empty:
        return pd.DataFrame()
    _tmp = batch_df.copy()
    _tmp["_train_gt_001"] = _tmp["mean_r2_train"] > 0.01
    _tmp["_train_gt_01"] = _tmp["mean_r2_train"] > 0.1
    _tmp["_test_gt_001"] = _tmp["mean_r2_test"] > 0.01
    _tmp["_test_gt_01"] = _tmp["mean_r2_test"] > 0.1
    return _tmp.groupby(["exp_label", "batch_id"], as_index=False).agg(
        prop_train_gt_001=("_train_gt_001", "mean"),
        prop_train_gt_01=("_train_gt_01", "mean"),
        prop_test_gt_001=("_test_gt_001", "mean"),
        prop_test_gt_01=("_test_gt_01", "mean"),
        mean_r2_train=("mean_r2_train", "mean"),
        mean_r2_test=("mean_r2_test", "mean"),
    )

def get_r2_stats_batched_joint(experiments: list[dict], batch_size: int | None = None) -> pd.DataFrame:
    if batch_size is None:
        batch_size = BATCH_SIZE
    r2_stats_batched = dict()
    for exp in experiments:
        all_runs = [os.path.join(exp["joint_root"], run) for run in exp["runs"]]
        r2_stats_batched[exp["joint_root"]] = gather_per_gene_per_batch_averages(
            all_runs, exp["label"], batch_size=batch_size
        )
    batch_df = pd.concat(r2_stats_batched.values(), ignore_index=True)
    if batch_df.empty:
        return pd.DataFrame()
    return batch_proportions_from_batch_df(batch_df)

def load_gene_list(path: str | None) -> set[str] | None:
    """Load gene IDs from a one-column file (``chr/ENSG...`` or ``ENSG...``)."""
    if path is None:
        return None
    _raw = pd.read_csv(path, header=None)[0].astype(str)
    return {g.split("/")[-1] if "/" in g else g for g in _raw}

def get_r2_stats_batched_pergene(experiments: list[dict], batch_size: int | None = None, gene_list_path: str | None = None) -> pd.DataFrame:
    if batch_size is None:
        batch_size = BATCH_SIZE
    gene_list = None
    if gene_list_path is not None:
        gene_list = load_gene_list(gene_list_path)
    pergene_frames = []
    for exp in experiments:
        gdf = gather_pergene_batch_df(
            exp["pergene_root"],
            exp["label"],
            batch_size=batch_size,
            gene_list=gene_list,
        )
        if gdf.empty:
            print(f"skip {exp['label']}: no R² data")
            continue
        pergene_frames.append(gdf)

    if not pergene_frames:
        return pd.DataFrame()
    batch_df = pd.concat(pergene_frames, ignore_index=True)
    return batch_proportions_from_batch_df(batch_df)


def plot_train_test_bars(
    df: pd.DataFrame,
    figsize=(10, 5),
    thresh=0.1,
    train_color: str = "#4c78a8",
    test_color: str = "#f58518",
):
    if df.empty:
        return

    if thresh == 0.1:
        column_test = "prop_test_gt_01"
        column_train = "prop_train_gt_01"
        thr_label = r"0.1"
    else:
        column_test = "prop_test_gt_001"
        column_train = "prop_train_gt_001"
        thr_label = r"0.01"

    need = {"exp_label", column_train, column_test}
    if not need.issubset(df.columns):
        print(f"plot_train_test_bars: expected columns {need}; got {list(df.columns)}")
        return

    rows = df.sort_values(
        ["exp_label", "batch_id"] if "batch_id" in df.columns else ["exp_label"]
    )
    n = len(rows)
    y = np.arange(n)
    h = 0.36
    train_v = rows[column_train].to_numpy(dtype=float)
    test_v = rows[column_test].to_numpy(dtype=float)

    multibatch = "batch_id" in rows.columns and (
        rows["batch_id"].nunique() > 1
        or rows.groupby("exp_label").size().max() > 1
    )
    if multibatch:
        ylab = [
            f"{r.exp_label} (batch {int(r.batch_id)})" for _, r in rows.iterrows()
        ]
    else:
        ylab = rows["exp_label"].tolist()

    h_plot = figsize[1] if n <= 8 else max(figsize[1], 0.55 * n + 1.5)
    fig, ax = plt.subplots(figsize=(figsize[0], h_plot))

    train_bars = ax.barh(
        y - h / 2,
        train_v,
        height=h,
        color=train_color,
        alpha=0.9,
        label="train",
        edgecolor="black",
        linewidth=0.5,
    )
    test_bars = ax.barh(
        y + h / 2,
        test_v,
        height=h,
        color=test_color,
        alpha=0.9,
        label="test",
        edgecolor="black",
        linewidth=0.5,
    )

    def _label_barh(bars, vals, fmt="{:.3f}"):
        xmax = 0.0
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            xmax = max(xmax, float(v))
            ax.text(
                bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(v),
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )
        return xmax

    xmax = max(_label_barh(train_bars, train_v), _label_barh(test_bars, test_v))
    ax.set_xlim(0.0, max(1.02, xmax * 1.06 + 0.04))

    ax.set_yticks(y)
    ax.set_yticklabels(ylab)
    ax.set_xlabel(fr"Fraction of genes with $R^2 > {thr_label}$")
    ax.set_title(
        rf"Train vs test: fraction of genes above $R^2 = {thr_label}$ "
        rf"(batch-level proportions)"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    # move legend outside of plot area
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def plot_mean_r2_per_configuration(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    train_color: str = "#4c78a8",
    test_color: str = "#f58518",
    title: str | None = None,
):
    experiments = df["exp_label"].unique()
    def _agg(col):
        means = []
        for c in experiments:
            vals = df.loc[df["exp_label"] == c, col].dropna().values
            if len(vals) == 0:
                means.append(np.nan)
                continue
            means.append(float(np.mean(vals)))
            
        return np.array(means)

    train_mean = _agg("mean_r2_train")
    test_mean = _agg("mean_r2_test")

    n = len(experiments)
    if figsize is None:
        figsize = (max(8.0, 0.75 * n + 2.0), 5.0)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n)
    width = 0.38

    ax.bar(
        x - width / 2, train_mean, width,
        color=train_color, alpha=0.9, label="train",
        edgecolor="black", linewidth=0.5,
        capsize=3, error_kw=dict(elinewidth=0.8, ecolor="black"),
    )
    ax.bar(
        x + width / 2, test_mean, width,
        color=test_color, alpha=0.9, label="test",
        edgecolor="black", linewidth=0.5,
        capsize=3, error_kw=dict(elinewidth=0.8, ecolor="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=30, ha="right")
    ax.set_ylabel(r"Mean $R^2$")
    if title is None:
        title = r"Mean $R^2$ (avg over batches of {} runs)".format(BATCH_SIZE)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.show()


# --- Post-train R² loading + Emmental vs baseline --------------------------------


def ensg_from_gene(gene: str) -> str:
    g = str(gene).strip()
    return g.split("/")[-1] if "/" in g else g


def _gene_path(gene: str, chrom: int | None = None) -> str:
    g = str(gene).strip()
    if "/" in g:
        return g
    if chrom is None:
        return g
    return f"chr{chrom}/{g}"


def _iter_chr_dirs(root: str) -> list[str]:
    if os.path.isdir(os.path.join(root, "chr1")):
        return [os.path.join(root, f"chr{c}") for c in range(1, 23)]
    return [root]


def _chr_genes_from_gene_column(genes: pd.Series) -> dict[int, np.ndarray]:
    chr_genes: dict[int, list] = {}
    for gene in genes:
        mobj = re.match(r"^chr(\d+)/", str(gene))
        if mobj:
            chrom = int(mobj.group(1))
            chr_genes.setdefault(chrom, []).append(str(gene))
    return {c: np.unique(v) for c, v in chr_genes.items()}


def _display_label(model_key: str) -> str:
    key = str(model_key)
    if key.startswith("baseline__"):
        return MODEL_NAMES.get(key.split("__", 1)[1], key)
    if key == "emmental_pergene":
        return "Emmental pergene"
    mobj = re.fullmatch(
        rf"emmental__c_{_COMMON_BETA_RE}(?:__r_{_RARE_BETA_RE})?", key
    )
    if mobj:
        return emmental_display_label(mobj.group(1), mobj.group(2))
    return key


def _beta_assembly_legend_from_model_keys(names: list[str]) -> str:
    """Build β assembly annotation from ``emmental__c_*`` model keys."""
    configs: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for name in names:
        mobj = re.fullmatch(
            rf"emmental__c_{_COMMON_BETA_RE}(?:__r_{_RARE_BETA_RE})?", str(name)
        )
        if mobj is None:
            continue
        cfg = (mobj.group(1), mobj.group(2))
        if cfg not in seen:
            seen.add(cfg)
            configs.append(cfg)
    return beta_assembly_legend_from_configs(configs)


def _add_figure_beta_annotation(fig, annotation: str | None, *, bottom: float = 0.02) -> None:
    if not annotation:
        return
    fig.text(
        0.5,
        bottom,
        annotation,
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        wrap=True,
    )


def read_post_train_scores(post_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concat ``chr*/train|test_r2_scores.csv`` under a post-train root."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for cd in _iter_chr_dirs(post_root):
        if not os.path.isdir(cd):
            continue
        tr = os.path.join(cd, "train_r2_scores.csv")
        te = os.path.join(cd, "test_r2_scores.csv")
        if not (os.path.isfile(tr) and os.path.isfile(te)):
            continue
        train_parts.append(pd.read_csv(tr))
        test_parts.append(pd.read_csv(te))
    if not train_parts:
        raise FileNotFoundError(f"No post-train R² CSVs under {post_root}")
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def load_post_train_r2_long(
    post_root: str,
    *,
    common_beta: str,
    rare_beta: str | None = None,
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """One Emmental R² series (gene × train/test) from post-train score files."""
    if thresholds is None:
        thresholds = THRESHOLDS
    col = post_train_r2_col(common_beta, rare_beta)
    model = emmental_model_key(common_beta, rare_beta)
    train_wide, test_wide = read_post_train_scores(post_root)
    if col not in train_wide.columns or col not in test_wide.columns:
        raise KeyError(f"Column {col!r} not in post-train R² CSVs under {post_root}")

    m = train_wide[["gene", col]].merge(
        test_wide[["gene", col]], on="gene", suffixes=("_train", "_test")
    )
    m = m.rename(columns={f"{col}_train": "R2_train", f"{col}_test": "R2_test"})
    m["model"] = model
    m["ensg"] = m["gene"].map(ensg_from_gene)
    for thr in thresholds:
        m[f"R2_Train({thr})"] = m["R2_train"].astype(float) > thr
        m[f"R2_Test({thr})"] = m["R2_test"].astype(float) > thr
    return m, _chr_genes_from_gene_column(m["gene"])


def load_post_train_r2_long_multi(
    post_root: str,
    configs: list[tuple[str, str | None]],
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Load several Emmental β-source configurations from the same post-train root."""
    parts: list[pd.DataFrame] = []
    chr_genes: dict[int, np.ndarray] = {}
    for common_beta, rare_beta in configs:
        df, cg = load_post_train_r2_long(
            post_root,
            common_beta=common_beta,
            rare_beta=rare_beta,
            thresholds=thresholds,
        )
        parts.append(df)
        for chrom, genes in cg.items():
            if chrom in chr_genes:
                chr_genes[chrom] = np.unique(np.concatenate([chr_genes[chrom], genes]))
            else:
                chr_genes[chrom] = genes
    return pd.concat(parts, ignore_index=True), chr_genes


def summarize_post_train_r2_sources(post_root: str) -> pd.DataFrame:
    """Mean R² and threshold proportions for every post-train R² column."""
    train_wide, test_wide = read_post_train_scores(post_root)
    r2_cols = [c for c in train_wide.columns if c.startswith("r2_")]
    rows: list[dict] = []
    for col in r2_cols:
        try:
            panel, common_beta, rare_beta = parse_post_train_r2_col(col)
        except ValueError:
            continue
        for split, wide in (("train", train_wide), ("test", test_wide)):
            s = wide[col].astype(float)
            rows.append(
                {
                    "r2_col": col,
                    "panel": panel,
                    "common_beta": common_beta,
                    "rare_beta": rare_beta,
                    "split": split,
                    "mean_r2": float(s.mean()),
                    "prop_gt_0": float((s > 0).mean()),
                    "prop_gt_0.01": float((s > 0.01).mean()),
                    "prop_gt_0.1": float((s > 0.1).mean()),
                    "n_genes": int(s.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def compare_post_train_r2_sources(
    post_root: str,
    *,
    display_stats_table: bool = False,
) -> dict:
    """Summaries and per-gene correlations across all post-train R² columns."""
    summary = summarize_post_train_r2_sources(post_root)
    train_wide, test_wide = read_post_train_scores(post_root)
    r2_cols = [c for c in train_wide.columns if c.startswith("r2_")]
    out = {
        "post_root": post_root,
        "summary": summary,
        "corr_train": train_wide[r2_cols].corr(),
        "corr_test": test_wide[r2_cols].corr(),
        "per_gene_train": train_wide[["gene"] + r2_cols],
        "per_gene_test": test_wide[["gene"] + r2_cols],
    }
    if display_stats_table:
        _display_post_train_r2_summary(summary)
    return out


def _display_post_train_r2_summary(summary: pd.DataFrame) -> None:
    try:
        from IPython.display import display
    except ImportError:
        display = print
    display(
        summary.pivot_table(
            index=["panel", "common_beta", "rare_beta"],
            columns="split",
            values="mean_r2",
        ).round(4)
    )


def _display_prep_tables(prep: dict) -> None:
    try:
        from IPython.display import display
    except ImportError:
        display = print
    r2_col = prep.get("r2_col", "")
    col_line = f"\nr2_col: {r2_col}" if r2_col else ""
    print(
        f"\n=== {prep['title']} | n_genes={prep['n_genes']} ===\n"
        f"emmental: {prep['emmental_root']}\n"
        f"baseline: {prep['baseline_root']}{col_line}"
    )
    display(
        prep["props"]
        .pivot_table(index="method", columns=["threshold", "split"], values="prop")
        .round(3)
    )
    display(
        prep["means"]
        .pivot(index="method", columns="split", values="mean_r2")
        .round(4)
    )


def plot_post_train_r2_sources(
    summary: pd.DataFrame,
    *,
    title: str | None = None,
    show: bool = True,
) -> None:
    """Bar plot of mean per-gene R² for each β-source configuration."""
    if summary.empty:
        print("plot_post_train_r2_sources: empty summary")
        return

    def _label(row) -> str:
        if row["panel"] == "common_only":
            return f"c={row['common_beta']}"
        return f"c={row['common_beta']}, r={row['rare_beta']}"

    sub = summary.copy()
    sub["config"] = sub.apply(_label, axis=1)
    panels = ["common_only", "common_plus_rare"]
    n_panels = sum(sub["panel"].eq(p).any() for p in panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    colors = SPLIT_COLORS

    ax_idx = 0
    for panel in panels:
        psub = sub[sub["panel"] == panel]
        if psub.empty:
            continue
        ax = axes[0, ax_idx]
        configs = list(dict.fromkeys(psub["config"]))
        x = np.arange(len(configs))
        width = 0.36
        for i, split in enumerate(["train", "test"]):
            vals = []
            for cfg in configs:
                row = psub[(psub["config"] == cfg) & (psub["split"] == split)]
                vals.append(float(row["mean_r2"].iloc[0]) if len(row) else np.nan)
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=split, color=colors[split])
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=35, ha="right")
        ax.set_ylabel("Mean per-gene R²")
        ax.set_title(panel.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        ax_idx += 1

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def _prop_col_for_threshold(thr: float) -> str:
    mapping = {0: "prop_gt_0", 0.01: "prop_gt_0.01", 0.1: "prop_gt_0.1"}
    if thr in mapping:
        return mapping[thr]
    raise ValueError(f"Unsupported threshold {thr}; expected one of {sorted(mapping)}")


def _prop_label_decimals(ymax: float, thr: float) -> int:
    """Decimal places for bar-top proportion labels."""
    if thr >= 0.1:
        return 4
    # R² > 0.01: proportions are larger but nearby bars can differ by ~0.002
    return 3


def _format_prop_label(val: float, *, decimals: int) -> str:
    return f"{float(val):.{decimals}f}"


def _format_delta_pp_label(d_pp: float) -> str:
    """Percentage-point change with precision scaled to magnitude."""
    a = abs(float(d_pp))
    if a >= 10:
        return f"Δ{d_pp:+.1f}pp"
    return f"Δ{d_pp:+.2f}pp"


def _lookup_panel_prop(
    summary: pd.DataFrame,
    *,
    common_beta: str,
    rare_beta: str | None,
    split: str,
    prop_col: str,
) -> float:
    m = (summary["common_beta"] == common_beta) & (summary["split"] == split)
    if rare_beta is None:
        m &= summary["panel"] == "common_only"
    else:
        m &= (summary["panel"] == "common_plus_rare") & (summary["rare_beta"] == rare_beta)
    row = summary.loc[m]
    if row.empty:
        return float("nan")
    return float(row[prop_col].iloc[0])


def post_train_r2_panel_proportion_table(
    summary: pd.DataFrame,
    *,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Long table of threshold proportions for common-only vs +rare variants."""
    if thresholds is None:
        thresholds = THRESHOLDS
    rows: list[dict] = []
    variants = _panel_variants()
    for common_beta in COMMON_BETA_SOURCES:
        for rare_beta, variant_label in variants:
            for split in ("train", "test"):
                row = {
                    "common_beta": common_beta,
                    "common_label": BETA_SOURCE_LABELS[common_beta],
                    "rare_beta": rare_beta,
                    "variant": variant_label,
                    "panel": "common_only" if rare_beta is None else "common_plus_rare",
                    "split": split,
                }
                for thr in thresholds:
                    prop_col = _prop_col_for_threshold(thr)
                    row[f"prop_gt_{thr}"] = _lookup_panel_prop(
                        summary,
                        common_beta=common_beta,
                        rare_beta=rare_beta,
                        split=split,
                        prop_col=prop_col,
                    )
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and thresholds:
        co = (
            out[out["panel"] == "common_only"]
            .set_index(["common_beta", "split"])[[f"prop_gt_{t}" for t in thresholds]]
            .rename(columns=lambda c: c + "_common_only")
        )
        out = out.merge(co, left_on=["common_beta", "split"], right_index=True, how="left")
        for thr in thresholds:
            col = f"prop_gt_{thr}"
            out[f"delta_{thr}_vs_common_only"] = out[col] - out[f"{col}_common_only"]
    return out


def plot_post_train_r2_panel_proportions(
    summary: pd.DataFrame,
    *,
    split: str = "test",
    thresholds: list[float] | None = None,
    title: str | None = None,
    show_values: bool = True,
    show_delta: bool = False,
    delta_min_pp: float = 1.0,
    show: bool = True,
) -> pd.DataFrame:
    """
    Faceted bar plot: fraction of genes above each R² threshold.

    One separate figure per threshold. For each common β source (x-axis group),
    one bar per assembly: common-only, then +r μ / +r full / +r rare.

    Bar tops show the proportion (``show_values``). Optional ``show_delta`` adds
    a second line on +rare bars only when the change vs common-only within the
    same group is at least ``delta_min_pp`` percentage points.
    """
    if summary.empty:
        print("plot_post_train_r2_panel_proportions: empty summary")
        return pd.DataFrame()
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    if thresholds is None:
        thresholds = THRESHOLDS

    table = post_train_r2_panel_proportion_table(summary, thresholds=thresholds)
    table = table[table["split"] == split].copy()
    variants = _panel_variants()

    n_variants = len(variants)
    bar_w = 0.92
    variant_gap = 0.22
    group_gap = 1.35
    group_width = n_variants * bar_w + (n_variants - 1) * variant_gap
    group_pitch = group_width + group_gap
    fig_w = max(12.0, 3.4 * len(COMMON_BETA_SOURCES) + 2.5)

    legend_handles = [
        Patch(
            facecolor=PANEL_VARIANT_COLORS[rare_beta],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
        for rare_beta, label in variants
    ]

    for thr in thresholds:
        prop_col = _prop_col_for_threshold(thr)
        fig, ax = plt.subplots(figsize=(fig_w, 5.6))
        centers: list[float] = []
        xlabels: list[str] = []
        ymax = 0.0

        for gi, common_beta in enumerate(COMMON_BETA_SOURCES):
            for rare_beta, _ in variants:
                val = _lookup_panel_prop(
                    summary,
                    common_beta=common_beta,
                    rare_beta=rare_beta,
                    split=split,
                    prop_col=prop_col,
                )
                if np.isfinite(val):
                    ymax = max(ymax, float(val))

        prop_decimals = _prop_label_decimals(ymax, thr)

        for gi, common_beta in enumerate(COMMON_BETA_SOURCES):
            group_start = gi * group_pitch
            group_center = group_start + 0.5 * group_width
            centers.append(group_center)
            xlabels.append(COMMON_BETA_SHORT[common_beta])
            co_prop = _lookup_panel_prop(
                summary,
                common_beta=common_beta,
                rare_beta=None,
                split=split,
                prop_col=prop_col,
            )

            for vi, (rare_beta, _) in enumerate(variants):
                val = _lookup_panel_prop(
                    summary,
                    common_beta=common_beta,
                    rare_beta=rare_beta,
                    split=split,
                    prop_col=prop_col,
                )
                x_center = group_start + vi * (bar_w + variant_gap) + bar_w / 2
                if not np.isfinite(val):
                    continue
                ax.bar(
                    x_center,
                    val,
                    width=bar_w,
                    color=PANEL_VARIANT_COLORS[rare_beta],
                    edgecolor="black",
                    linewidth=0.4,
                    align="center",
                )
                label_y = float(val) + 0.008
                if show_values:
                    ax.text(
                        x_center,
                        label_y,
                        _format_prop_label(val, decimals=prop_decimals),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="0.15",
                    )
                    label_y += 0.032
                if (
                    show_delta
                    and rare_beta is not None
                    and np.isfinite(co_prop)
                ):
                    d_pp = 100.0 * (float(val) - float(co_prop))
                    if abs(d_pp) >= delta_min_pp:
                        ymax = max(ymax, float(val) + 0.06)
                        ax.text(
                            x_center,
                            label_y,
                            _format_delta_pp_label(d_pp),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="0.35",
                        )

        ax.set_xticks(centers)
        ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(fr"Fraction of genes with $R^2 > {thr}$")
        n_genes = summary[summary["split"] == split]["n_genes"].iloc[0]
        thr_title = f"{split} $R^2 > {thr} (n={n_genes})$"
        if title:
            ax.set_title(f"{title} | {thr_title}")
        else:
            ax.set_title(thr_title)
        ax.set_xlim(-0.4, len(COMMON_BETA_SOURCES) * group_pitch - group_gap + 0.4)
        ax.set_ylim(0, min(1.05, max(0.10, ymax + 0.10)))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            title="β assembly",
        )
        fig.subplots_adjust(bottom=0.20, right=0.78)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return table


def build_baseline_props_df(
    baseline_root: str,
    chr_genes: dict[int, np.ndarray],
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if thresholds is None:
        thresholds = THRESHOLDS
    rows: list[dict] = []
    for chrom in range(1, 23):
        genes = chr_genes.get(chrom, [])
        if len(genes) == 0:
            continue
        ensgs = [ensg_from_gene(g) for g in genes]
        for method in BASELINE_METHODS:
            path = os.path.join(baseline_root, method, f"chr{chrom}.tsv")
            if not os.path.isfile(path):
                continue
            bl = pd.read_csv(path, sep="\t", index_col=0)
            bl = bl.loc[bl.index.intersection(ensgs)]
            for ensg, row in bl.iterrows():
                rows.append(
                    {
                        "gene": f"chr{chrom}/{ensg}",
                        "ensg": ensg,
                        "model": method,
                        "R2_train": float(row["R2_train"]),
                        "R2_test": float(row["R2_test"]),
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No baseline rows loaded from {baseline_root}")

    props_df = pd.DataFrame(rows)
    for thr in thresholds:
        props_df[f"R2_Train({thr})"] = props_df["R2_train"] > thr
        props_df[f"R2_Test({thr})"] = props_df["R2_test"] > thr
    return props_df, props_df["ensg"].unique()


def _prep_vs_baseline_tables(
    emmental_root: str,
    baseline_root: str,
    *,
    emmental_long: pd.DataFrame,
    chr_genes: dict[int, np.ndarray],
    method_order: list[str],
    thresholds: list[float],
    gene_list: set[str] | None = None,
    title: str,
    r2_col: str,
) -> dict:
    bl_long = load_baseline_long(baseline_root, chr_genes, thresholds)
    long_df = pd.concat([emmental_long, bl_long], ignore_index=True)
    gene_set = np.intersect1d(
        emmental_long["ensg"].astype(str).unique(),
        bl_long["ensg"].astype(str).unique(),
    )
    if gene_list is not None:
        gene_set = np.intersect1d(gene_set, list(gene_list))
    if len(gene_set) == 0:
        raise ValueError(f"No overlapping genes for {title}")

    props = summarize_baseline_props_labeled(long_df, thresholds, gene_set=gene_set)
    means = summarize_baseline_means(long_df, gene_set=gene_set)
    return {
        "title": title,
        "emmental_root": emmental_root,
        "baseline_root": baseline_root,
        "r2_col": r2_col,
        "n_genes": int(len(gene_set)),
        "props": props,
        "means": means,
        "method_order": method_order,
        "thresholds": thresholds,
    }


def _resolve_pergene_chr_root(pergene_root: str) -> str:
    root = os.path.abspath(pergene_root)
    if glob.glob(os.path.join(root, "chr*", "train_r2_scores.csv")):
        return root
    nested = os.path.join(root, "pergene")
    if glob.glob(os.path.join(nested, "chr*", "train_r2_scores.csv")):
        return nested
    return root


def load_pergene_fit_r2_long(
    pergene_root: str,
    *,
    thresholds: list[float] | None = None,
    model_key: str = "emmental_pergene",
    r2_col: str = "r2",
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Pergene training R² (single ``r2`` column per chr)."""
    if thresholds is None:
        thresholds = THRESHOLDS
    root = _resolve_pergene_chr_root(pergene_root)
    parts: list[pd.DataFrame] = []
    for cd in _iter_chr_dirs(root):
        if not os.path.isdir(cd):
            continue
        mobj = re.match(r"chr(\d+)$", os.path.basename(cd))
        chrom = int(mobj.group(1)) if mobj else 1
        tr = os.path.join(cd, "train_r2_scores.csv")
        te = os.path.join(cd, "test_r2_scores.csv")
        if not (os.path.isfile(tr) and os.path.isfile(te)):
            continue
        tr_df = pd.read_csv(tr)
        te_df = pd.read_csv(te)
        if r2_col not in tr_df.columns or r2_col not in te_df.columns:
            continue
        m = tr_df[["gene", r2_col]].merge(
            te_df[["gene", r2_col]], on="gene", suffixes=("_train", "_test")
        )
        m = m.rename(columns={f"{r2_col}_train": "R2_train", f"{r2_col}_test": "R2_test"})
        m["model"] = model_key
        if len(m) and "/" not in str(m["gene"].iloc[0]):
            m["gene"] = m["gene"].map(lambda g: _gene_path(g, chrom))
        m["ensg"] = m["gene"].map(ensg_from_gene)
        for thr in thresholds:
            m[f"R2_Train({thr})"] = m["R2_train"].astype(float) > thr
            m[f"R2_Test({thr})"] = m["R2_test"].astype(float) > thr
        parts.append(m)

    if not parts:
        raise FileNotFoundError(f"No pergene R² CSVs found under {pergene_root}")

    long_df = pd.concat(parts, ignore_index=True)
    return long_df, _chr_genes_from_gene_column(long_df["gene"])


def prep_emmental_vs_baseline(
    post_root: str,
    baseline_root: str | None = None,
    *,
    common_beta: str = "common",
    rare_beta: str | None = None,
    include_common_only: bool = True,
    thresholds: list[float] | None = None,
    gene_list: set[str] | None = None,
    gene_list_path: str | None = None,
    title: str | None = None,
    display_stats_table: bool = False,
) -> dict:
    """
    Post-train Emmental vs ``baseline_full_common01``.

    ``common_beta``: one of ``mu``, ``full``, ``common``, ``train``.
    ``rare_beta``: if set, include a common+rare panel assembly. When set and
    ``include_common_only`` is True (default), both common-only and common+rare
    Emmental bars are shown alongside baselines.
    """
    if baseline_root is None:
        baseline_root = BASELINE_ROOTS["common01"]
    if thresholds is None:
        thresholds = THRESHOLDS
    if gene_list is None:
        gene_list = load_gene_list(gene_list_path)

    configs: list[tuple[str, str | None]] = []
    if rare_beta is None or include_common_only:
        configs.append((common_beta, None))
    if rare_beta is not None:
        configs.append((common_beta, rare_beta))
    if not configs:
        raise ValueError("No Emmental β configuration selected")

    r2_cols = [post_train_r2_col(c, r) for c, r in configs]
    emmental_labels = [emmental_panel_label(c, r) for c, r in configs]
    emmental_beta_legend = beta_assembly_legend_from_configs(configs)
    if title is None:
        if len(configs) == 1:
            title = (
                f"post-train ({r2_cols[0]}) vs {os.path.basename(baseline_root)}"
            )
        else:
            title = (
                f"post-train c={common_beta} vs {os.path.basename(baseline_root)}"
            )

    if len(configs) == 1:
        em_long, chr_genes = load_post_train_r2_long(
            post_root,
            common_beta=configs[0][0],
            rare_beta=configs[0][1],
            thresholds=thresholds,
        )
    else:
        em_long, chr_genes = load_post_train_r2_long_multi(
            post_root, configs, thresholds=thresholds
        )

    method_order = BASELINE_BAR_LABELS + emmental_labels
    prep = _prep_vs_baseline_tables(
        post_root,
        baseline_root,
        emmental_long=em_long,
        chr_genes=chr_genes,
        method_order=method_order,
        thresholds=thresholds,
        gene_list=gene_list,
        title=title,
        r2_col=", ".join(r2_cols),
    )
    prep["emmental_configs"] = configs
    prep["emmental_beta_legend"] = emmental_beta_legend
    prep["beta_source"] = beta_source_subtitle(common_beta, rare_beta)
    if display_stats_table:
        _display_prep_tables(prep)
    return prep


def prep_pergene_vs_baseline(
    pergene_root: str,
    baseline_root: str | None = None,
    *,
    thresholds: list[float] | None = None,
    gene_list: set[str] | None = None,
    gene_list_path: str | None = None,
    title: str | None = None,
    display_stats_table: bool = False,
) -> dict:
    """Pergene-fit Emmental (training ``r2`` column) vs ``baseline_full_common01``."""
    if baseline_root is None:
        baseline_root = BASELINE_ROOTS["common01"]
    if thresholds is None:
        thresholds = THRESHOLDS
    if gene_list is None:
        gene_list = load_gene_list(gene_list_path)
    root = _resolve_pergene_chr_root(pergene_root)
    if title is None:
        title = f"pergene fit vs {os.path.basename(baseline_root)}"

    em_long, chr_genes = load_pergene_fit_r2_long(root, thresholds=thresholds)
    prep = _prep_vs_baseline_tables(
        root,
        baseline_root,
        emmental_long=em_long,
        chr_genes=chr_genes,
        method_order=PERGENE_COMMON01_METHOD_ORDER,
        thresholds=thresholds,
        gene_list=gene_list,
        title=title,
        r2_col="r2",
    )
    if display_stats_table:
        _display_prep_tables(prep)
    return prep


# Backward-compatible names
prep_common01_vs_baseline = prep_emmental_vs_baseline
prep_pergene_common01_vs_baseline = prep_pergene_vs_baseline


def load_emmental_baseline_compare_long(
    post_root: str,
    baseline_root: str,
    *,
    emmental_configs: list[tuple[str, str | None]],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Emmental (one or more β configs) + baselines in one long table."""
    em_long, chr_genes = load_post_train_r2_long_multi(
        post_root, emmental_configs, thresholds=thresholds
    )
    bl_long = load_baseline_long(baseline_root, chr_genes, thresholds)
    return pd.concat([em_long, bl_long], ignore_index=True)


# --- Top-gene overlap (Emmental vs baselines) ---------------------------------

# Recommended β assemblies for overlap with common-panel baselines:
#   (common, None)     — common-only on MAF≥0.01 SNPs; baseline-fair (G_common β)
#   (common, rare)     — full panel; tests whether rare variants add top genes
# Train/test bar colors (reserved — do not reuse for Emmental assembly bars).
SPLIT_COLORS = {"train": "#4C72B0", "test": "#DD8452"}

# Extra colors for >2 Emmental assemblies in overlap plots (never train/test blue/orange).
_EXTRA_EMMENTAL_OVERLAP_COLORS = ["#55A868", "#8172B2", "#937860"]


def emmental_overlap_bar_color(model: str, *, index: int = 0) -> str:
    """Bar color for one Emmental assembly (common-only vs +rare), not train/test."""
    if model.startswith("emmental__c_") and "__r_" in model:
        rare_beta = model.rsplit("__r_", 1)[-1]
        return PANEL_VARIANT_COLORS.get(rare_beta, "#C44E52")
    if model.startswith("emmental__c_"):
        return PANEL_VARIANT_COLORS[None]
    return _EXTRA_EMMENTAL_OVERLAP_COLORS[index % len(_EXTRA_EMMENTAL_OVERLAP_COLORS)]


def default_overlap_emmental_configs(
    common_beta: str = "common",
    rare_beta: str | None = "rare",
    *,
    include_common_only: bool = True,
) -> list[tuple[str, str | None]]:
    """Default Emmental assemblies for top-gene overlap with common-panel baselines."""
    configs: list[tuple[str, str | None]] = []
    if include_common_only or rare_beta is None:
        configs.append((common_beta, None))
    if rare_beta is not None:
        configs.append((common_beta, rare_beta))
    seen: set[tuple[str, str | None]] = set()
    out: list[tuple[str, str | None]] = []
    for cfg in configs:
        if cfg not in seen:
            seen.add(cfg)
            out.append(cfg)
    return out if out else [(common_beta, None)]


def genes_above_r2_threshold(
    long_df: pd.DataFrame,
    model: str,
    threshold: float,
    split: str,
) -> set[str]:
    """Gene IDs (ENSG) with R² above ``threshold`` for one method and split."""
    col = "R2_train" if split == "train" else "R2_test"
    sub = long_df[long_df["model"] == model].copy()
    sub["ensg"] = sub["ensg"].astype(str)
    return set(sub.loc[sub[col].astype(float) > threshold, "ensg"])


def build_gene_sets_by_threshold(
    long_df: pd.DataFrame,
    thresholds: list[float],
    split: str,
) -> dict[float, dict[str, set[str]]]:
    models = sorted(long_df["model"].unique())
    return {
        thr: {m: genes_above_r2_threshold(long_df, m, thr, split) for m in models}
        for thr in thresholds
    }


def pairwise_gene_overlap_matrices(
    gene_sets: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = list(gene_sets.keys())
    n = len(names)
    cnt = np.zeros((n, n), dtype=int)
    jacc = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            inter = gene_sets[a] & gene_sets[b]
            cnt[i, j] = len(inter)
            union = gene_sets[a] | gene_sets[b]
            jacc[i, j] = len(inter) / len(union) if union else np.nan
    return cnt, jacc, names


def pairwise_gene_overlap_summary(
    gene_sets: dict[str, set[str]],
    *,
    label: str,
    threshold: float,
    split: str,
) -> pd.DataFrame:
    cnt, jacc, names = pairwise_gene_overlap_matrices(gene_sets)
    rows: list[dict] = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            rows.append(
                {
                    "label": label,
                    "threshold": threshold,
                    "split": split,
                    "method_a": _display_label(a),
                    "method_b": _display_label(b),
                    "model_a": a,
                    "model_b": b,
                    "n_a": len(gene_sets[a]),
                    "n_b": len(gene_sets[b]),
                    "n_intersection": int(cnt[i, j]),
                    "jaccard": float(jacc[i, j]),
                }
            )
    return pd.DataFrame(rows)


def emmental_panel_overlap_stats(
    gene_sets: dict[str, set[str]],
    emmental_models: list[str],
) -> list[dict]:
    """Within-Emmental overlap between common-only and common+rare top-gene sets."""
    rows: list[dict] = []
    for i, m_a in enumerate(emmental_models):
        for m_b in emmental_models[i + 1 :]:
            a, b = gene_sets.get(m_a, set()), gene_sets.get(m_b, set())
            rows.append(
                {
                    "method_a": _display_label(m_a),
                    "method_b": _display_label(m_b),
                    "n_a": len(a),
                    "n_b": len(b),
                    "n_intersection": len(a & b),
                    "n_a_only": len(a - b),
                    "n_b_only": len(b - a),
                    "jaccard": len(a & b) / len(a | b) if (a | b) else float("nan"),
                }
            )
    return rows


def emmental_rare_gain_genes(
    post_root: str,
    *,
    common_beta: str = "common",
    rare_beta: str = "rare",
    threshold: float = 0.01,
    split: str = "test",
    gene_list: set[str] | None = None,
    gene_list_path: str | None = None,
) -> pd.DataFrame:
    """
    Genes that cross ``threshold`` with common+rare but not with common-only.

    Panel comparison at a **fixed** common β source — not a β-source swap.
    Uses ``(common_beta, None)`` vs ``(common_beta, rare_beta)``.
    """
    if gene_list is None and gene_list_path is not None:
        gene_list = load_gene_list(gene_list_path)

    long_df, _ = load_post_train_r2_long_multi(
        post_root,
        [(common_beta, None), (common_beta, rare_beta)],
    )
    long_df["ensg"] = long_df["ensg"].astype(str)
    if gene_list is not None:
        long_df = long_df[long_df["ensg"].isin({str(g) for g in gene_list})]

    m_co = emmental_model_key(common_beta, None)
    m_cr = emmental_model_key(common_beta, rare_beta)
    co = long_df.loc[long_df["model"] == m_co, ["gene", "ensg", "R2_train", "R2_test"]].rename(
        columns={"R2_train": "r2_common_only_train", "R2_test": "r2_common_only_test"}
    )
    cr = long_df.loc[long_df["model"] == m_cr, ["gene", "ensg", "R2_train", "R2_test"]].rename(
        columns={"R2_train": "r2_common_plus_rare_train", "R2_test": "r2_common_plus_rare_test"}
    )
    merged = co.merge(cr, on=["gene", "ensg"], how="inner")
    merged["r2_common_only"] = merged[f"r2_common_only_{split}"].astype(float)
    merged["r2_common_plus_rare"] = merged[f"r2_common_plus_rare_{split}"].astype(float)
    merged["r2_delta"] = merged["r2_common_plus_rare"] - merged["r2_common_only"]

    gain = merged[
        (merged["r2_common_plus_rare"] > threshold)
        & (merged["r2_common_only"] <= threshold)
    ].copy()
    gain = gain.sort_values("r2_delta", ascending=False)
    gain.insert(0, "threshold", threshold)
    gain.insert(1, "split", split)
    gain.insert(2, "common_beta", common_beta)
    gain.insert(3, "rare_beta", rare_beta)
    return gain.reset_index(drop=True)


def plot_emmental_baseline_intersection_bars(
    gene_sets: dict[str, set[str]],
    *,
    emmental_models: list[str],
    threshold: float,
    split: str,
    title: str,
    beta_source: str | None = None,
    out_path: str | None = None,
    show: bool = True,
) -> None:
    """For each baseline, bar heights = |Emmental top genes ∩ baseline top genes|."""
    baselines = _ordered_baseline_keys(gene_sets)
    em_models = [m for m in emmental_models if m in gene_sets]
    if not baselines:
        print("plot_emmental_baseline_intersection_bars: no baseline models")
        return
    if not em_models:
        print("plot_emmental_baseline_intersection_bars: no Emmental models in gene_sets")
        return

    n_em = len(em_models)
    bar_width = 0.62
    bar_gap = 0.10
    group_spacing = max(2.0, 1.2 + 0.35 * n_em)
    cluster_w = n_em * bar_width + max(0, n_em - 1) * bar_gap

    fig, ax = plt.subplots(figsize=(max(11, 2.2 * len(baselines) + 2), 5.5))
    group_centers: list[float] = []
    xlabels: list[str] = []

    for gi, bl in enumerate(baselines):
        gc = gi * group_spacing
        group_centers.append(gc)
        xlabels.append(_display_label(bl))
        nb = len(gene_sets[bl])
        vals = [len(gene_sets.get(em, set()) & gene_sets[bl]) for em in em_models]
        ymax_local = max(max(vals, default=0), nb, 1)

        for ei, (em, val) in enumerate(zip(em_models, vals)):
            x_center = gc + (ei - (n_em - 1) / 2) * (bar_width + bar_gap)
            color = emmental_overlap_bar_color(em, index=ei)
            bar = ax.bar(
                x_center,
                val,
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                label=_display_label(em) if gi == 0 else None,
            )
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                bar[0].get_height() + ymax_local * 0.02,
                str(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        line_step = ymax_local * 0.06
        ax.text(
            gc,
            ymax_local + line_step * 1.8,
            f"baseline n={nb}",
            ha="center",
            fontsize=7,
        )
        em_ns = "/".join(str(len(gene_sets.get(em, set()))) for em in em_models)
        ax.text(gc, ymax_local + line_step * 0.7, f"emmental n={em_ns}", ha="center", fontsize=7)

    ymax = max(
        (
            len(gene_sets.get(em, set()) & gene_sets[bl])
            for bl in baselines
            for em in em_models
        ),
        default=1,
    )
    ax.set_xticks(group_centers)
    ax.set_xticklabels(xlabels, rotation=25, ha="center")
    ax.set_ylabel(f"|genes| with R² > {threshold} ({split})")
    _set_ax_title_with_subtitle(
        ax, f"{title}: Emmental ∩ baseline top genes", beta_source
    )
    ax.set_ylim(0, ymax * 1.25 + 2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    fig.subplots_adjust(right=0.78)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gene_overlap_heatmap(
    cnt: np.ndarray,
    names: list[str],
    title: str,
    *,
    beta_source: str | None = None,
    out_path: str | None = None,
    show: bool = True,
) -> None:
    labels = [_display_label(n) for n in names]
    if beta_source is None:
        beta_source = _beta_source_from_model_keys(names)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(names)), max(6, 0.5 * len(names))))
    cmap = plt.cm.YlOrRd
    vmin = int(cnt.min()) if cnt.size else 0
    vmax = int(cnt.max()) if cnt.size else 1
    norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1))
    im = ax.imshow(cnt, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            r, g, b, _ = cmap(norm(cnt[i, j]))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = "black" if lum > 0.62 else "white"
            ax.text(
                j,
                i,
                str(int(cnt[i, j])),
                ha="center",
                va="center",
                fontsize=7,
                color=txt_color,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, label="|intersection|")
    _set_ax_title_with_subtitle(ax, title, beta_source)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _ordered_baseline_keys(gene_sets: dict[str, set[str]]) -> list[str]:
    """Baseline keys in ``BASELINE_METHODS`` order (matches proportion bar plots)."""
    ordered = [f"baseline__{m}" for m in BASELINE_METHODS if f"baseline__{m}" in gene_sets]
    extra = sorted(
        m
        for m in gene_sets
        if str(m).startswith("baseline__") and m not in ordered
    )
    return ordered + extra


def _ordered_overlap_methods(
    gene_sets: dict[str, set[str]],
    methods: list[str] | None = None,
    *,
    emmental_models: list[str] | None = None,
) -> list[str]:
    """Baselines (ridge→lasso→BR→EN), then Emmental assemblies, then extras."""
    if methods is not None:
        return [m for m in methods if m in gene_sets]
    baselines = _ordered_baseline_keys(gene_sets)
    emmental = [m for m in (emmental_models or []) if m in gene_sets]
    extra = sorted(
        m for m in gene_sets if m not in baselines and m not in emmental
    )
    return baselines + emmental + extra


def _gene_sets_to_upset_data(
    gene_sets: dict[str, set[str]],
    methods: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Boolean membership frame (genes × methods) and display labels for UpSet."""
    if not methods:
        raise ValueError("No methods selected for UpSet plot")
    all_genes = sorted(set().union(*(gene_sets.get(m, set()) for m in methods)))
    labels = [_display_label(m) for m in methods]
    indicators = pd.DataFrame(
        {
            label: [g in gene_sets.get(model, set()) for g in all_genes]
            for label, model in zip(labels, methods)
        },
        index=all_genes,
    )
    indicators = indicators.loc[indicators.any(axis=1)]
    return indicators, labels


def _set_fig_title_with_subtitle(
    fig,
    title: str,
    subtitle: str | None = None,
    *,
    subtitle_scale: float = 0.82,
    title_fontsize: float = 11,
    line_dy: float = 0.045,
) -> None:
    """Figure-level title block + optional β-source subtitle (for UpSet plots)."""
    title_lines = [ln for ln in str(title).split("\n") if ln.strip()]
    if not title_lines:
        title_lines = [""]

    y = 0.99
    for line in title_lines:
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            va="top",
            fontsize=title_fontsize,
            transform=fig.transFigure,
        )
        y -= line_dy

    header_bottom = y + 0.004
    if subtitle:
        subtitle_dy = 0.038
        fig.text(
            0.5,
            header_bottom - 0.010,
            subtitle,
            ha="center",
            va="top",
            fontsize=title_fontsize * subtitle_scale,
            color="#444444",
            transform=fig.transFigure,
        )
        header_bottom -= subtitle_dy

    fig.subplots_adjust(top=max(0.70, header_bottom - 0.015))


def plot_gene_overlap_upset(
    gene_sets: dict[str, set[str]],
    *,
    methods: list[str] | None = None,
    emmental_models: list[str] | None = None,
    threshold: float,
    split: str,
    title: str,
    beta_source: str | None = None,
    out_path: str | None = None,
    show: bool = True,
    min_subset_size: int = 1,
    max_intersections: int | None = 40,
) -> pd.DataFrame | None:
    """
    UpSet plot of top-gene intersections across methods.

    Each row is a gene with R² above ``threshold``; bar height = number of genes
    with that exact membership pattern (which methods' top sets it belongs to).
  """
    try:
        from upsetplot import UpSet
    except ImportError:
        print(
            "plot_gene_overlap_upset: missing package upsetplot "
            "(pip install upsetplot)"
        )
        return None

    ordered = _ordered_overlap_methods(
        gene_sets, methods, emmental_models=emmental_models
    )
    if len(ordered) < 2:
        print("plot_gene_overlap_upset: need at least two methods")
        return None

    indicators, labels = _gene_sets_to_upset_data(gene_sets, ordered)
    if indicators.empty:
        print("plot_gene_overlap_upset: empty gene sets")
        return None

    upset_data = indicators.groupby(list(indicators.columns), sort=False).size()
    if min_subset_size > 1:
        upset_data = upset_data[upset_data >= min_subset_size]
    if upset_data.empty:
        print("plot_gene_overlap_upset: no intersections pass min_subset_size")
        return None

    if max_intersections is not None and len(upset_data) > max_intersections:
        upset_data = upset_data.sort_values(ascending=False).head(max_intersections)

    fig_h = max(5.5, 0.45 * len(labels) + 3.5)
    fig_w = max(10.0, 0.35 * len(upset_data) + 6.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    upset = UpSet(
        upset_data,
        sort_by="cardinality",
        sort_categories_by="input",
        show_counts=True,
        min_subset_size=None,
        element_size=32,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"upsetplot(\.|$)",
        )
        upset.plot(fig=fig)
    if beta_source is None:
        beta_source = _beta_source_from_model_keys(ordered)
    _set_fig_title_with_subtitle(
        fig,
        f"{title}\nTop genes: R² > {threshold} ({split})",
        beta_source,
    )
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.35)
    if show:
        plt.show()
    else:
        plt.close(fig)

    rows = []
    for idx, count in upset_data.items():
        if not isinstance(idx, tuple):
            idx = (idx,)
        rows.append(
            {
                "threshold": threshold,
                "split": split,
                "count": int(count),
                **{labels[i]: bool(idx[i]) for i in range(len(labels))},
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


UPSET_OVERLAP_MODES = ("all", "common_only", "common_plus_rare")


def _resolve_overlap_thresholds(
    overlap: dict,
    threshold: float | list[float] | None,
) -> list[float]:
    if threshold is None:
        return list(overlap["thresholds"])
    if isinstance(threshold, (list, tuple)):
        return [float(t) for t in threshold]
    return [float(threshold)]


def _overlap_thr_slug(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def _overlap_beta_subtitle(overlap: dict) -> str:
    return beta_source_subtitle(overlap["common_beta"], overlap.get("rare_beta"))


def _upset_methods_for_mode(
    overlap: dict,
    gene_sets: dict[str, set[str]],
    mode: str,
) -> list[str] | None:
    """Method list for one UpSet mode; ``None`` means all methods (ordered)."""
    baselines = _ordered_baseline_keys(gene_sets)
    common_beta = overlap["common_beta"]
    rare_beta = overlap.get("rare_beta")
    m_common_only = emmental_model_key(common_beta, None)
    m_common_rare = (
        emmental_model_key(common_beta, rare_beta) if rare_beta is not None else None
    )
    if mode == "all":
        return None
    if mode == "common_only":
        return baselines + [m_common_only]
    if mode == "common_plus_rare":
        return baselines + ([m_common_rare] if m_common_rare is not None else [])
    raise ValueError(
        f"Unknown upset mode {mode!r}; expected one of {UPSET_OVERLAP_MODES}"
    )


def prep_gene_overlap_vs_baseline(
    post_root: str,
    baseline_root: str | None = None,
    *,
    label: str = "",
    emmental_configs: list[tuple[str, str | None]] | None = None,
    common_beta: str = "common",
    rare_beta: str | None = "rare",
    include_common_only: bool = True,
    thresholds: list[float] | None = None,
    split: str = "test",
    gene_list: set[str] | None = None,
    gene_list_path: str | None = None,
    out_dir: str | None = None,
    save_tables: bool = False,
    compute_rare_gain: bool = True,
) -> dict:
    """
    Load and summarize top-gene overlap (Emmental vs baselines); no figures.

    Returns a dict for ``plot_gene_overlap_*`` helpers. Keys include
    ``gene_sets_by_threshold``, ``summary``, ``long_df``, ``emmental_models``.
    """
    if baseline_root is None:
        baseline_root = BASELINE_ROOTS["common01"]
    if thresholds is None:
        thresholds = THRESHOLDS
    if gene_list is None:
        gene_list = load_gene_list(gene_list_path)
    if emmental_configs is None:
        emmental_configs = default_overlap_emmental_configs(
            common_beta,
            rare_beta,
            include_common_only=include_common_only,
        )
    if not label:
        label = os.path.basename(os.path.dirname(os.path.abspath(post_root)))

    emmental_models = [emmental_model_key(c, r) for c, r in emmental_configs]

    long_df = load_emmental_baseline_compare_long(
        post_root,
        baseline_root,
        emmental_configs=emmental_configs,
        thresholds=thresholds,
    )
    long_df["ensg"] = long_df["ensg"].astype(str)
    if gene_list is not None:
        long_df = long_df[long_df["ensg"].isin({str(g) for g in gene_list})]

    em_ensg = long_df.loc[~long_df["model"].str.startswith("baseline__"), "ensg"].unique()
    bl_ensg = long_df.loc[long_df["model"].str.startswith("baseline__"), "ensg"].unique()
    common_ensg = np.intersect1d(em_ensg, bl_ensg)
    long_df = long_df[long_df["ensg"].isin(common_ensg)]

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(post_root)), "plots", "top_gene_overlap"
        )
    if save_tables:
        os.makedirs(out_dir, exist_ok=True)

    gene_sets_by_thr = build_gene_sets_by_threshold(long_df, thresholds, split)
    summary_parts: list[pd.DataFrame] = []
    panel_stats_by_thr: dict[float, list[dict]] = {}

    for thr in thresholds:
        gsets = gene_sets_by_thr[thr]
        summary_parts.append(
            pairwise_gene_overlap_summary(
                gsets, label=label, threshold=thr, split=split
            )
        )
        panel_stats_by_thr[thr] = emmental_panel_overlap_stats(gsets, emmental_models)

    rare_gain_by_thr: dict[float, pd.DataFrame] = {}
    if (
        compute_rare_gain
        and include_common_only
        and rare_beta is not None
        and (common_beta, None) in emmental_configs
        and (common_beta, rare_beta) in emmental_configs
    ):
        for thr in thresholds:
            rare_gain_by_thr[thr] = emmental_rare_gain_genes(
                post_root,
                common_beta=common_beta,
                rare_beta=rare_beta,
                threshold=thr,
                split=split,
                gene_list=set(long_df["ensg"].unique()),
            )
            if save_tables:
                rare_gain_by_thr[thr].to_csv(
                    os.path.join(
                        out_dir,
                        f"rare_gain_genes_thr{_overlap_thr_slug(thr)}_{split}.csv",
                    ),
                    index=False,
                )

    summary_df = pd.concat(summary_parts, ignore_index=True)
    if save_tables:
        summary_df.to_csv(
            os.path.join(out_dir, f"pairwise_overlap_{split}.csv"), index=False
        )

    return {
        "label": label,
        "post_root": post_root,
        "baseline_root": baseline_root,
        "emmental_configs": emmental_configs,
        "emmental_models": emmental_models,
        "common_beta": common_beta,
        "rare_beta": rare_beta,
        "include_common_only": include_common_only,
        "split": split,
        "thresholds": thresholds,
        "n_genes": int(len(common_ensg)),
        "gene_sets_by_threshold": gene_sets_by_thr,
        "emmental_panel_stats_by_threshold": panel_stats_by_thr,
        "rare_gain_genes_by_threshold": rare_gain_by_thr,
        "summary": summary_df,
        "long_df": long_df,
        "out_dir": out_dir,
    }


def plot_gene_overlap_intersection_bars(
    overlap: dict,
    threshold: float | list[float] | None = None,
    *,
    show: bool = True,
    out_dir: str | None = None,
    save: bool = True,
) -> None:
    """Intersection bar plots: |Emmental top genes ∩ each baseline| per threshold."""
    out_dir = out_dir or overlap["out_dir"]
    beta_sub = _overlap_beta_subtitle(overlap)
    label = overlap["label"]
    split = overlap["split"]
    emmental_models = overlap["emmental_models"]

    for thr in _resolve_overlap_thresholds(overlap, threshold):
        gsets = overlap["gene_sets_by_threshold"][thr]
        plot_emmental_baseline_intersection_bars(
            gsets,
            emmental_models=emmental_models,
            threshold=thr,
            split=split,
            title=label,
            beta_source=beta_sub,
            out_path=os.path.join(
                out_dir,
                f"intersection_baselines_thr{_overlap_thr_slug(thr)}_{split}.png",
            )
            if save
            else None,
            show=show,
        )


def plot_gene_overlap_heatmaps(
    overlap: dict,
    threshold: float | list[float] | None = None,
    *,
    show: bool = True,
    out_dir: str | None = None,
    save: bool = True,
) -> None:
    """Pairwise top-gene intersection heatmaps per threshold."""
    out_dir = out_dir or overlap["out_dir"]
    beta_sub = _overlap_beta_subtitle(overlap)
    label = overlap["label"]
    split = overlap["split"]

    for thr in _resolve_overlap_thresholds(overlap, threshold):
        gsets = overlap["gene_sets_by_threshold"][thr]
        cnt, _, names = pairwise_gene_overlap_matrices(gsets)
        plot_gene_overlap_heatmap(
            cnt,
            names,
            f"{label}: pairwise top-gene overlap (R²>{thr}, {split})",
            beta_source=beta_sub,
            out_path=os.path.join(
                out_dir, f"overlap_heatmap_thr{_overlap_thr_slug(thr)}_{split}.png"
            )
            if save
            else None,
            show=show,
        )


def plot_gene_overlap_upsets(
    overlap: dict,
    threshold: float | list[float] | None = None,
    *,
    upset_modes: tuple[str, ...] = UPSET_OVERLAP_MODES,
    show: bool = True,
    out_dir: str | None = None,
    save: bool = True,
) -> dict[float, dict[str, pd.DataFrame]]:
    """
    UpSet plots per threshold and mode.

    Modes (``upset_modes``):
      - ``all``: 4 baselines + both Emmental assemblies
      - ``common_only``: baselines + Emmental common-only
      - ``common_plus_rare``: baselines + Emmental common+rare

    Returns ``{threshold: {mode: membership-count DataFrame}}``.
    """
    out_dir = out_dir or overlap["out_dir"]
    if save:
        os.makedirs(out_dir, exist_ok=True)

    beta_sub = _overlap_beta_subtitle(overlap)
    label = overlap["label"]
    split = overlap["split"]
    emmental_models = overlap["emmental_models"]
    common_beta = overlap["common_beta"]
    rare_beta = overlap.get("rare_beta")
    m_common_only = emmental_model_key(common_beta, None)
    m_common_rare = (
        emmental_model_key(common_beta, rare_beta) if rare_beta is not None else None
    )

    upset_tables_by_thr: dict[float, dict[str, pd.DataFrame]] = {}

    for thr in _resolve_overlap_thresholds(overlap, threshold):
        gsets = overlap["gene_sets_by_threshold"][thr]
        thr_slug = _overlap_thr_slug(thr)
        upset_thr: dict[str, pd.DataFrame] = {}

        for mode in upset_modes:
            if mode == "common_only" and m_common_only not in gsets:
                continue
            if mode == "common_plus_rare" and (
                m_common_rare is None or m_common_rare not in gsets
            ):
                continue

            upset_methods = _upset_methods_for_mode(overlap, gsets, mode)
            plot_title = (
                label
                if mode == "all"
                else f"{label} ({mode.replace('_', ' ')})"
            )
            upset_df = plot_gene_overlap_upset(
                gsets,
                methods=upset_methods,
                emmental_models=emmental_models if upset_methods is None else None,
                threshold=thr,
                split=split,
                title=plot_title,
                beta_source=beta_sub,
                out_path=os.path.join(
                    out_dir,
                    f"upset_{mode}_thr{thr_slug}_{split}.png",
                )
                if save
                else None,
                show=show,
            )
            if upset_df is not None:
                upset_thr[mode] = upset_df
                if save:
                    upset_df.to_csv(
                        os.path.join(
                            out_dir,
                            f"upset_{mode}_thr{thr_slug}_{split}.csv",
                        ),
                        index=False,
                    )

        if upset_thr:
            upset_tables_by_thr[thr] = upset_thr

    return upset_tables_by_thr


def _display_gene_overlap_summary(summary_df: pd.DataFrame) -> None:
    try:
        from IPython.display import display
    except ImportError:
        display = print
    if summary_df.empty:
        return
    display(
        summary_df.pivot_table(
            index=["method_a", "method_b"],
            columns="threshold",
            values="n_intersection",
            aggfunc="first",
        )
    )


def plot_vs_baseline(
    prep: dict,
    *,
    display_stats_table: bool = False,
    show: bool = True,
) -> None:
    """Bar plots from ``prep_emmental_vs_baseline`` or ``prep_pergene_vs_baseline``."""
    if display_stats_table:
        _display_prep_tables(prep)

    plot_emmental_baseline_props_and_means(
        prep["props"],
        prep["means"],
        f"{prep['title']} (n={prep['n_genes']})",
        prep["method_order"],
        thresholds=prep.get("thresholds", THRESHOLDS),
        beta_source=prep.get("beta_source"),
        show=show,
    )


def prep_emmental_baseline_comparison(
    emmental_specs: list[dict],
    *,
    post_root_key: str = "post_pergene",
    common_beta: str = "common",
    rare_beta: str | None = None,
    thresholds: list[float] | None = None,
    top200_genes_path: str | None = DEFAULT_TOP200_GENES_PATH,
    gene_sets: tuple[str, ...] = ("all", "top200"),
) -> list[dict]:
    results: list[dict] = []
    gene_list_top200 = load_gene_list(top200_genes_path) if "top200" in gene_sets else None
    for spec in emmental_specs:
        label = spec["label"]
        post_root = spec[post_root_key]
        for gene_set_name, gene_list in [("all", None), ("top200", gene_list_top200)]:
            if gene_set_name not in gene_sets:
                continue
            try:
                prep = prep_emmental_vs_baseline(
                    post_root,
                    thresholds=thresholds,
                    gene_list=gene_list,
                    common_beta=common_beta,
                    rare_beta=rare_beta,
                    title=f"{label} | {gene_set_name}",
                )
                prep["gene_set"] = gene_set_name
                results.append(prep)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f"skip {label} ({gene_set_name}): {exc}")
    return results


def plot_emmental_baseline_comparison(
    results: list[dict],
    *,
    display_stats_table: bool = False,
    show_plots: bool = True,
) -> None:
    for prep in results:
        plot_vs_baseline(prep, display_stats_table=display_stats_table, show=show_plots)


def load_baseline_long(
    baseline_root: str,
    chr_genes: dict[int, np.ndarray],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    bl, _ = build_baseline_props_df(baseline_root, chr_genes, thresholds)
    bl = bl.copy()
    bl["model"] = "baseline__" + bl["model"].astype(str)
    return bl


def summarize_baseline_means(
    long_df: pd.DataFrame, gene_set: set[str] | np.ndarray | None = None
) -> pd.DataFrame:
    df = long_df.copy()
    df["ensg"] = df["ensg"].astype(str)
    if gene_set is not None:
        gene_set = {str(g) for g in gene_set}
        df = df[df["ensg"].isin(gene_set)]
    rows: list[dict] = []
    for model, sub in df.groupby("model", sort=False):
        label = _display_label(model)
        rows.append(
            {
                "method": label,
                "split": "train",
                "mean_r2": float(sub["R2_train"].mean()),
            }
        )
        rows.append(
            {
                "method": label,
                "split": "test",
                "mean_r2": float(sub["R2_test"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_baseline_props_labeled(
    long_df: pd.DataFrame,
    thresholds: list[float] | None = None,
    gene_set: set[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = THRESHOLDS
    df = long_df.copy()
    df["ensg"] = df["ensg"].astype(str)
    if gene_set is not None:
        gene_set = {str(g) for g in gene_set}
        df = df[df["ensg"].isin(gene_set)]
    rows: list[dict] = []
    for model_key, sub in df.groupby("model", sort=False):
        label = _display_label(model_key)
        for thr in thresholds:
            rows.append(
                {
                    "method": label,
                    "model_key": str(model_key),
                    "threshold": thr,
                    "split": "train",
                    "prop": float(sub[f"R2_Train({thr})"].mean()),
                    "n_genes": int(sub[f"R2_Train({thr})"].sum()),
                    "n_genes_total": int(sub["ensg"].nunique()),
                }
            )
            rows.append(
                {
                    "method": label,
                    "model_key": str(model_key),
                    "threshold": thr,
                    "split": "test",
                    "prop": float(sub[f"R2_Test({thr})"].mean()),
                    "n_genes": int(sub[f"R2_Test({thr})"].sum()),
                    "n_genes_total": int(sub["ensg"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def plot_baseline_props_bar(
    props_df: pd.DataFrame,
    thr: float,
    out_path: str | None = None,
    title: str | None = None,
    show: bool = True,
) -> None:
    sub = props_df[props_df["threshold"] == thr].copy()
    method_labels = [MODEL_NAMES[m] for m in MODEL_NAMES]
    splits = ["train", "test"]
    x = np.arange(len(MODEL_NAMES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(method_labels)), 5.5))
    colors = SPLIT_COLORS

    for i, split in enumerate(splits):
        vals = []
        for label in method_labels:
            row = sub[(sub["method"] == label) & (sub["split"] == split)]
            vals.append(float(row["prop"].iloc[0]) if len(row) else float("nan"))
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=split.capitalize(), color=colors[split])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha="center")
    ax.set_ylabel(f"Fraction of genes with R² > {thr}")
    ax.set_ylim(0, min(1.05, max(0.15, ax.get_ylim()[1] + 0.08)))
    ax.legend(
        title="Split",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(right=0.82)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_emmental_baseline_props_and_means(
    props_df: pd.DataFrame,
    means_df: pd.DataFrame,
    title_prefix: str,
    method_order: list[str],
    thresholds: list[float] | None = None,
    beta_source: str | None = None,
    show: bool = True,
) -> None:
    if thresholds is None:
        thresholds = THRESHOLDS
    colors = SPLIT_COLORS.copy()

    for thr in thresholds:
        sub = props_df[props_df["threshold"] == thr]
        fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(method_order)), 5.5))
        x = np.arange(len(method_order))
        width = 0.36
        for i, split in enumerate(["train", "test"]):
            vals = []
            n_genes = []
            for lab in method_order:
                row = sub[(sub["method"] == lab) & (sub["split"] == split)]
                vals.append(float(row["prop"].iloc[0]) if len(row) else np.nan)
                n_genes.append(int(row["n_genes"].iloc[0]) if len(row) else np.nan)
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width, label=split.capitalize(), color=colors[split]
            )
            for b, v, g in zip(bars, vals, n_genes):
                if np.isfinite(v):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.004,
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        max(b.get_height() * 0.5, 0.01),
                        f"({g})",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if b.get_height() > 0.06 else "black",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=45, ha="right")
        ax.set_ylabel(f"Fraction of genes with R² > {thr}")
        ymax = float(np.nanmax(sub["prop"])) if len(sub) else 0.12
        ax.set_ylim(0, min(1.05, max(0.12, ymax + 0.08)))
        ax.legend(title="Split", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        _set_ax_title_with_subtitle(
            ax, f"{title_prefix}: R² proportion > {thr}", beta_source
        )
        ax.grid(axis="y", alpha=0.3)
        fig.subplots_adjust(right=0.82)
        if show:
            plt.show()
        else:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(method_order)), 5.5))
    x = np.arange(len(method_order))
    width = 0.36
    for i, split in enumerate(["train", "test"]):
        vals = []
        for lab in method_order:
            row = means_df[(means_df["method"] == lab) & (means_df["split"] == split)]
            vals.append(float(row["mean_r2"].iloc[0]) if len(row) else np.nan)
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, vals, width, label=split.capitalize(), color=colors[split]
        )
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(method_order, rotation=45, ha="right")
    ax.set_ylabel("Mean per-gene R²")
    ax.legend(title="Split", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _set_ax_title_with_subtitle(ax, f"{title_prefix}: mean R²", beta_source)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(right=0.82)
    if show:
        plt.show()
    else:
        plt.close(fig)


def load_emmental_r2_long(
    post_root: str,
    mode: str,
    configs: list[tuple[str, str | None]],
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame | None, dict[int, np.ndarray]]:
    """Notebook helper: load several post-train β configs (``mode`` ignored)."""
    del mode
    if thresholds is None:
        thresholds = THRESHOLDS
    if not os.path.isdir(post_root):
        return None, {}
    try:
        return load_post_train_r2_long_multi(post_root, configs, thresholds=thresholds)
    except (FileNotFoundError, KeyError):
        return None, {}


def load_emmental_pergene_fit_r2_long(
    pergene_root: str,
    model_key: str,
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame | None, dict[int, np.ndarray]]:
    del model_key
    if thresholds is None:
        thresholds = THRESHOLDS
    if not os.path.isdir(pergene_root):
        return None, {}
    try:
        return load_pergene_fit_r2_long(pergene_root, thresholds=thresholds)
    except FileNotFoundError:
        return None, {}


def compare_pergene_vs_post_train(
    experiment_root: str,
    *,
    common_beta: str = "common",
    rare_beta: str | None = None,
    split: str = "test",
) -> pd.DataFrame:
    """Merge pergene training R² with one post-train β-source column."""
    experiment_root = os.path.abspath(experiment_root)
    pergene_root = _resolve_pergene_chr_root(os.path.join(experiment_root, "pergene"))
    post_root = os.path.join(experiment_root, "post_pergene")
    col = post_train_r2_col(common_beta, rare_beta)
    fname = f"{split}_r2_scores.csv"

    parts: list[pd.DataFrame] = []
    for post_chr in sorted(glob.glob(os.path.join(post_root, "chr*"))):
        chrom = os.path.basename(post_chr)
        pg_path = os.path.join(pergene_root, chrom, fname)
        po_path = os.path.join(post_chr, fname)
        if not (os.path.isfile(pg_path) and os.path.isfile(po_path)):
            continue
        pg = pd.read_csv(pg_path).rename(columns={"r2": "r2_pergene"})
        if "/" not in str(pg["gene"].iloc[0]) if len(pg) else False:
            pg["gene"] = pg["gene"].map(lambda g: f"{chrom}/{g}")
        po = pd.read_csv(po_path)
        if col not in po.columns:
            continue
        parts.append(pg.merge(po[["gene", col]], on="gene", how="inner"))

    if not parts:
        raise FileNotFoundError(
            f"No matched pergene/post {fname} ({col}) under {experiment_root}"
        )

    df = pd.concat(parts, ignore_index=True)
    summary = {
        "split": split,
        "r2_col": col,
        "n_genes": len(df),
        "mean_r2_pergene": float(df["r2_pergene"].mean()),
        "mean_r2_post": float(df[col].mean()),
        "corr": float(df["r2_pergene"].corr(df[col])) if len(df) > 1 else float("nan"),
    }
    df.attrs["summary"] = summary
    return df


# Backward-compatible alias
compare_r2_sources = compare_pergene_vs_post_train


def summarize_r2_baseline_comparison_guidance(
    experiment_root: str,
    *,
    common_beta: str = "common",
    rare_beta: str | None = None,
) -> dict:
    """Quick pergene vs post-train R² comparison for picking β sources."""
    out: dict = {}
    for split in ("train", "test"):
        try:
            df = compare_pergene_vs_post_train(
                experiment_root,
                common_beta=common_beta,
                rare_beta=rare_beta,
                split=split,
            )
            out[split] = df.attrs["summary"]
        except FileNotFoundError as exc:
            out[split] = {"error": str(exc)}
    out["post_train_summary"] = summarize_post_train_r2_sources(
        os.path.join(experiment_root, "post_pergene")
    ).to_dict(orient="records")
    out["recommendation"] = (
        "Baseline comparison: common_beta='common' (G_common decoupled) for common-only; "
        "rare_beta='rare' with common_beta='common' or 'full' for full panel. "
        "Use compare_post_train_r2_sources() to inspect all saved assemblies."
    )
    return out
