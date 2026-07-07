"""Parameter / beta diagnostic plots for Emmental post-train outputs."""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

COMMON_MASK_COL = "is_common_maf_ge_threshold"
SAVED_COMMON_BETA_COL = "beta_common_train_posterior"
LEGACY_SAVED_COMMON_BETA_COLS = (
    "beta_common_posterior",
    "beta_train_posterior",
)
PERGENE_BETA_COL = SAVED_COMMON_BETA_COL
POSTERIOR_BETA_COL = "beta_hat"
DECOUPLED_POSTERIOR_BETA_COL = "beta_hat_common_decoupled"
G_FULL_POSTERIOR_BETA_COL = POSTERIOR_BETA_COL
G_COMMON_POSTERIOR_BETA_COL = DECOUPLED_POSTERIOR_BETA_COL
PANEL_COLS = [COMMON_MASK_COL, SAVED_COMMON_BETA_COL, POSTERIOR_BETA_COL]
GENE_FROM_PANEL_RE = re.compile(r"chr(\d+)_(ENSG\d+)_full_panel_beta\.csv\.gz$")

COMMON_COLOR = "#4c78a8"
RARE_COLOR = "#f58518"
SAVED_COLOR = "#4c78a8"
POST_COLOR = "#f58518"
G_COMMON_COLOR = "#e45756"
DIFF_COLOR = "#54a24b"
CHUNK = 500_000

CommonBetaSource = Literal["posterior", "saved", "recomputed", "pergene"]
VARIANT_RARE_COL = "is_rare"

BETA_COL_ALIASES = {
    "beta_hat": POSTERIOR_BETA_COL,
    "beta_full": POSTERIOR_BETA_COL,
    "G_full": POSTERIOR_BETA_COL,
    "assembled": POSTERIOR_BETA_COL,
    "beta_hat_common_decoupled": DECOUPLED_POSTERIOR_BETA_COL,
    "G_common": DECOUPLED_POSTERIOR_BETA_COL,
    "G_common_decoupled": DECOUPLED_POSTERIOR_BETA_COL,
    "g_common_decoupled": DECOUPLED_POSTERIOR_BETA_COL,
    "common_decoupled": DECOUPLED_POSTERIOR_BETA_COL,
    "saved_common": SAVED_COMMON_BETA_COL,
    "train_posterior": SAVED_COMMON_BETA_COL,
    "beta_common_posterior": SAVED_COMMON_BETA_COL,
    "beta_common_train_posterior": SAVED_COMMON_BETA_COL,
    "beta_train_posterior": SAVED_COMMON_BETA_COL,
    "mu": "mu_rho_w_lambda",
}


def resolve_computed_beta_col(computed_col: str | None) -> str:
    """Map CLI aliases to post-train panel column for computed common β."""
    if not computed_col:
        return DECOUPLED_POSTERIOR_BETA_COL
    key = computed_col.strip()
    return BETA_COL_ALIASES.get(key, key)


def resolve_beta_col(beta_col: str | None) -> str:
    """Map CLI aliases to a panel / variant-level CSV column name."""
    if not beta_col:
        return POSTERIOR_BETA_COL
    key = beta_col.strip()
    return BETA_COL_ALIASES.get(key, key)


def beta_col_to_common_source(beta_col: str | None) -> str:
    col = resolve_beta_col(beta_col)
    if col in (SAVED_COMMON_BETA_COL, *LEGACY_SAVED_COMMON_BETA_COLS):
        return "saved"
    return "posterior"


def discover_beta_panel_paths(post_pergene_root: str) -> list[str]:
    root = os.path.abspath(post_pergene_root)
    patterns = [
        os.path.join(root, "chr*", "full_beta_panel", "*_full_panel_beta.csv.gz"),
        os.path.join(root, "full_beta_panel", "*_full_panel_beta.csv.gz"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(p for p in set(paths) if os.path.isfile(p))


def load_post_train_manifest(post_root: str | None) -> dict:
    if not post_root:
        return {}
    root = os.path.abspath(post_root)
    candidates = [os.path.join(root, "manifest.json")]
    candidates.extend(sorted(glob.glob(os.path.join(root, "chr*", "manifest.json"))))
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f) or {}
            except (json.JSONDecodeError, OSError):
                continue
    return {}


def _resolve_saved_common_beta_col(columns: Iterable[str]) -> str:
    cols = set(columns)
    if SAVED_COMMON_BETA_COL in cols:
        return SAVED_COMMON_BETA_COL
    for legacy in LEGACY_SAVED_COMMON_BETA_COLS:
        if legacy in cols:
            return legacy
    raise ValueError(
        f"Panel missing saved common β column "
        f"({SAVED_COMMON_BETA_COL!r} or legacy {LEGACY_SAVED_COMMON_BETA_COLS!r})"
    )


def normalize_common_beta_source(source: str) -> str:
    key = str(source).strip().lower()
    if key in (
        "saved",
        "pergene",
        "train",
        SAVED_COMMON_BETA_COL,
        *LEGACY_SAVED_COMMON_BETA_COLS,
    ):
        return "saved"
    if key in ("posterior", "recomputed", "beta_hat", "hat"):
        return "posterior"
    raise ValueError(
        f"Unknown common_source={source!r}; use 'posterior'/'recomputed' or 'saved'/'pergene'"
    )


def _mode_label(mode: str, *, role: str) -> str:
    if mode == "posterior":
        return rf"{role} $\beta$ (collapsed posterior)"
    return rf"{role} $\mu$ (prior mean)"


def beta_distribution_xlabel(
    manifest: dict,
    *,
    common_source: str = "posterior",
) -> str:
    """X-axis label describing how common vs rare β were chosen."""
    common_src = normalize_common_beta_source(common_source)
    common_mode = str(manifest.get("common_beta_mode", "posterior"))
    rare_mode = str(manifest.get("rare_beta_mode", "posterior"))

    if common_src == "saved":
        common_txt = r"common: saved pergene $\bar{\beta}$"
    elif common_mode == "posterior":
        common_txt = r"common: post-train $\beta_\mathrm{posterior}$"
    else:
        common_txt = r"common: post-train $\mu$"

    if rare_mode == "posterior":
        rare_txt = r"rare: post-train $\beta_\mathrm{posterior}$"
    else:
        rare_txt = r"rare: post-train $\mu$"

    return rf"$\beta$ ({common_txt}; {rare_txt})"


def _filter_chromosome_paths(
    paths: Iterable[str], chromosomes: list[int] | None
) -> list[str]:
    if not chromosomes:
        return list(paths)
    chrom_set = {int(c) for c in chromosomes}
    out: list[str] = []
    for path in paths:
        mobj = re.search(r"/chr(\d+)/", path.replace(os.sep, "/"))
        if mobj and int(mobj.group(1)) in chrom_set:
            out.append(path)
    return out


def _gene_from_panel_path(path: str) -> tuple[int | None, str | None]:
    mobj = GENE_FROM_PANEL_RE.search(os.path.basename(path))
    if not mobj:
        return None, None
    return int(mobj.group(1)), mobj.group(2)


def _percentile_limits(values: Iterable[np.ndarray], q: float) -> tuple[float, float]:
    parts = [v for v in values if len(v)]
    if not parts:
        return -1.0, 1.0
    a = np.concatenate(parts)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return -1.0, 1.0
    hi = float(np.quantile(np.abs(a), q))
    return -hi, hi


def _accumulate_stratum_histograms_from_paths(
    paths: Iterable[str],
    bins: np.ndarray,
    *,
    common_source: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Histogram common/rare β per panel without concatenating all variants."""
    common_src = normalize_common_beta_source(common_source)
    common_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    rare_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    n_common = n_rare = 0

    for path in paths:
        if not os.path.isfile(path):
            continue
        header = pd.read_csv(path, nrows=0, compression="infer").columns
        saved_col = _resolve_saved_common_beta_col(header)
        df = pd.read_csv(
            path,
            usecols=[COMMON_MASK_COL, saved_col, POSTERIOR_BETA_COL],
            compression="infer",
        )
        common_mask = df[COMMON_MASK_COL].astype(bool).to_numpy()
        beta_hat = df[POSTERIOR_BETA_COL].astype(np.float64).to_numpy()
        if common_src == "saved":
            common_beta = df[saved_col].astype(np.float64).to_numpy()
            c_ok = common_mask & np.isfinite(common_beta)
            common_vals = common_beta[c_ok]
        else:
            c_ok = common_mask & np.isfinite(beta_hat)
            common_vals = beta_hat[c_ok]
        r_ok = (~common_mask) & np.isfinite(beta_hat)
        rare_vals = beta_hat[r_ok]

        if len(common_vals):
            hc, _ = np.histogram(common_vals, bins=bins)
            common_counts += hc
            n_common += len(common_vals)
        if len(rare_vals):
            hr, _ = np.histogram(rare_vals, bins=bins)
            rare_counts += hr
            n_rare += len(rare_vals)

    return common_counts, rare_counts, n_common, n_rare


def _approx_percentile_limits_from_panels(
    paths: Iterable[str],
    q: float,
    *,
    common_source: str,
    max_values: int = 2_000_000,
    max_panels: int = 800,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate symmetric |β| percentile limits by subsampling across panels."""
    common_src = normalize_common_beta_source(common_source)
    path_list = [p for p in paths if os.path.isfile(p)]
    if len(path_list) > max_panels:
        rng_paths = np.random.default_rng(seed)
        path_list = list(
            rng_paths.choice(path_list, size=max_panels, replace=False)
        )

    rng = np.random.default_rng(seed)
    sample = np.empty(max_values, dtype=np.float64)
    filled = 0
    seen = 0

    for path in path_list:
        header = pd.read_csv(path, nrows=0, compression="infer").columns
        saved_col = _resolve_saved_common_beta_col(header)
        df = pd.read_csv(
            path,
            usecols=[COMMON_MASK_COL, saved_col, POSTERIOR_BETA_COL],
            compression="infer",
        )
        common_mask = df[COMMON_MASK_COL].astype(bool).to_numpy()
        beta_hat = df[POSTERIOR_BETA_COL].astype(np.float64).to_numpy()
        if common_src == "saved":
            common_beta = df[saved_col].astype(np.float64).to_numpy()
            vals = np.concatenate(
                [
                    np.abs(common_beta[common_mask & np.isfinite(common_beta)]),
                    np.abs(beta_hat[(~common_mask) & np.isfinite(beta_hat)]),
                ]
            )
        else:
            vals = np.abs(beta_hat[np.isfinite(beta_hat)])
        if len(vals) == 0:
            continue

        for v in vals:
            seen += 1
            if filled < max_values:
                sample[filled] = v
                filled += 1
            else:
                j = rng.integers(0, seen)
                if j < max_values:
                    sample[j] = v

    if filled == 0:
        return -1.0, 1.0
    hi = float(np.quantile(sample[:filled], q))
    return -hi, hi


def _load_stratum_beta_arrays(
    paths: Iterable[str],
    *,
    common_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load common and rare β arrays from panel CSVs (one read per gene)."""
    common_src = normalize_common_beta_source(common_source)
    common_parts: list[np.ndarray] = []
    rare_parts: list[np.ndarray] = []

    for path in paths:
        if not os.path.isfile(path):
            continue
        header = pd.read_csv(path, nrows=0, compression="infer").columns
        saved_col = _resolve_saved_common_beta_col(header)
        df = pd.read_csv(
            path,
            usecols=[COMMON_MASK_COL, saved_col, POSTERIOR_BETA_COL],
            compression="infer",
        )
        common_mask = df[COMMON_MASK_COL].astype(bool).to_numpy()
        rare_mask = ~common_mask
        beta_hat = df[POSTERIOR_BETA_COL].astype(np.float64).to_numpy()
        if common_src == "saved":
            common_beta = df[saved_col].astype(np.float64).to_numpy()
            c_ok = common_mask & np.isfinite(common_beta)
            common_parts.append(common_beta[c_ok])
        else:
            c_ok = common_mask & np.isfinite(beta_hat)
            common_parts.append(beta_hat[c_ok])
        r_ok = rare_mask & np.isfinite(beta_hat)
        rare_parts.append(beta_hat[r_ok])

    common = np.concatenate(common_parts) if common_parts else np.array([], dtype=np.float64)
    rare = np.concatenate(rare_parts) if rare_parts else np.array([], dtype=np.float64)
    return common, rare


def _iter_panel_beta_values(
    paths: Iterable[str],
    *,
    common_source: str,
    chromosomes: list[int] | None = None,
    max_variants: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (common_beta, rare_beta) arrays for histograms."""
    paths = _filter_chromosome_paths(paths, chromosomes)
    common, rare = _load_stratum_beta_arrays(paths, common_source=common_source)
    rng = np.random.default_rng(seed)
    if max_variants is not None:
        if len(common) > max_variants:
            common = rng.choice(common, size=max_variants, replace=False)
        if len(rare) > max_variants:
            rare = rng.choice(rare, size=max_variants, replace=False)
    return common, rare


def accumulate_stratum_histograms(
    post_pergene_root: str,
    bins: np.ndarray,
    *,
    common_source: str = "posterior",
    chromosomes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    paths = discover_beta_panel_paths(post_pergene_root)
    if not paths:
        raise FileNotFoundError(f"No full_beta_panel CSVs under {post_pergene_root}")

    paths = _filter_chromosome_paths(paths, chromosomes)
    return _accumulate_stratum_histograms_from_paths(
        paths, bins, common_source=common_source
    )


def _render_stratum_histogram(
    ax,
    bins: np.ndarray,
    common_counts: np.ndarray,
    rare_counts: np.ndarray,
    n_common: int,
    n_rare: int,
    *,
    density: bool,
    xlabel: str,
    title: str | None,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    if density:
        common_y = common_counts / (n_common * width) if n_common else common_counts
        rare_y = rare_counts / (n_rare * width) if n_rare else rare_counts
        ylabel = "Density"
    else:
        common_y = common_counts
        rare_y = rare_counts
        ylabel = "Count"

    ax.bar(
        centers,
        common_y,
        width=width * 0.95,
        color=COMMON_COLOR,
        alpha=0.55,
        label=f"common (n={n_common:,})",
        edgecolor="black",
        linewidth=0.2,
    )
    ax.bar(
        centers,
        rare_y,
        width=width * 0.95,
        color=RARE_COLOR,
        alpha=0.55,
        label=f"rare (n={n_rare:,})",
        edgecolor="black",
        linewidth=0.2,
    )
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_title(title or "β distribution by variant stratum")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def plot_beta_distribution_by_stratum(
    post_pergene_root: str,
    *,
    common_source: CommonBetaSource = "posterior",
    chromosomes: list[int] | None = None,
    n_bins: int = 120,
    percentile_cap: float = 0.999,
    density: bool = True,
    log_y: bool = False,
    title: str | None = None,
    out_path: str | None = None,
    show: bool = True,
    ax=None,
) -> dict:
    """
    Histogram of β by common vs rare (train MAF threshold).

    ``common_source``:
      - ``posterior`` / ``recomputed``: common uses ``beta_hat`` (post-train assembled β)
      - ``saved`` / ``pergene``: common uses ``beta_common_posterior`` (pergene ``beta_mean``)

    Rare variants always use ``beta_hat`` (per ``rare_beta_mode`` in the post-train manifest).
    """
    root = os.path.abspath(post_pergene_root)
    paths = discover_beta_panel_paths(root)
    if not paths:
        raise FileNotFoundError(f"No full_beta_panel CSVs under {root}")

    manifest = load_post_train_manifest(root)
    paths = _filter_chromosome_paths(paths, chromosomes)
    common_src = normalize_common_beta_source(common_source)

    lo, hi = _approx_percentile_limits_from_panels(
        paths, percentile_cap, common_source=common_source
    )
    bins = np.linspace(lo, hi, n_bins + 1)
    common_counts, rare_counts, n_common, n_rare = _accumulate_stratum_histograms_from_paths(
        tqdm(paths, desc="beta panels", leave=False),
        bins,
        common_source=common_src,
    )
    xlab = beta_distribution_xlabel(manifest, common_source=common_source)

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(9, 5))

    chrom_txt = f"chr{','.join(map(str, chromosomes))}" if chromosomes else "genome-wide"
    default_title = (
        f"β by common vs rare ({chrom_txt}; common={normalize_common_beta_source(common_source)})"
    )
    _render_stratum_histogram(
        ax,
        bins,
        common_counts.astype(np.int64),
        rare_counts.astype(np.int64),
        n_common,
        n_rare,
        density=density,
        xlabel=xlab,
        title=title or default_title,
    )
    if log_y:
        ax.set_yscale("log")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")

    if show and created_fig:
        plt.show()
    elif created_fig:
        plt.close(ax.figure)

    return {
        "post_pergene_root": root,
        "common_source": normalize_common_beta_source(common_source),
        "manifest": manifest,
        "n_common": n_common,
        "n_rare": n_rare,
        "xlabel": xlab,
        "ax": ax,
    }


def _iter_saved_common_beta_values(
    paths: Iterable[str],
    *,
    chromosomes: list[int] | None = None,
    max_variants: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    paths = _filter_chromosome_paths(paths, chromosomes)
    parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for path in paths:
        if not os.path.isfile(path):
            continue
        header = pd.read_csv(path, nrows=0, compression="infer").columns
        saved_col = _resolve_saved_common_beta_col(header)
        df = pd.read_csv(path, usecols=[COMMON_MASK_COL, saved_col], compression="infer")
        common_mask = df[COMMON_MASK_COL].astype(bool).to_numpy()
        beta = df[saved_col].astype(np.float64).to_numpy()
        ok = common_mask & np.isfinite(beta)
        if ok.any():
            parts.append(beta[ok])

    out = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
    if max_variants is not None and len(out) > max_variants:
        out = rng.choice(out, size=max_variants, replace=False)
    return out


def plot_saved_common_beta_distribution(
    post_pergene_root: str,
    *,
    chromosomes: list[int] | None = None,
    n_bins: int = 120,
    percentile_cap: float = 0.999,
    density: bool = True,
    log_y: bool = False,
    title: str | None = None,
    out_path: str | None = None,
    show: bool = True,
    ax=None,
) -> dict:
    """Histogram of saved pergene common β (``beta_common_posterior``) only."""
    root = os.path.abspath(post_pergene_root)
    paths = discover_beta_panel_paths(root)
    if not paths:
        raise FileNotFoundError(f"No full_beta_panel CSVs under {root}")

    values = _iter_saved_common_beta_values(paths, chromosomes=chromosomes)
    if len(values) == 0:
        raise ValueError("No finite saved common β values to plot")

    lo, hi = _percentile_limits([values], percentile_cap)
    bins = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=bins)
    n_total = int(counts.sum())

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(9, 5))

    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    y = counts / (n_total * width) if density and n_total else counts
    ax.bar(
        centers,
        y,
        width=width * 0.95,
        color=SAVED_COLOR,
        alpha=0.7,
        label=f"saved common β (n={n_total:,})",
        edgecolor="black",
        linewidth=0.2,
    )
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"Common $\beta_\mathrm{saved}$ (pergene mean)")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_xlim(bins[0], bins[-1])
    if log_y:
        ax.set_yscale("log")

    chrom_txt = f"chr{','.join(map(str, chromosomes))}" if chromosomes else "genome-wide"
    ax.set_title(title or f"Saved common β distribution ({chrom_txt})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")

    if show and created_fig:
        plt.show()
    elif created_fig:
        plt.close(ax.figure)

    return {
        "post_pergene_root": root,
        "chromosomes": chromosomes,
        "n_variants": n_total,
        "ax": ax,
    }


def _percentile_limits_from_csv(
    path: str,
    col: str,
    q: float,
    *,
    max_rows: int = 2_000_000,
) -> tuple[float, float]:
    vals: list[np.ndarray] = []
    seen = 0
    for chunk in pd.read_csv(path, usecols=[col], chunksize=CHUNK, compression="infer"):
        v = np.abs(chunk[col].astype(np.float64).to_numpy())
        v = v[np.isfinite(v)]
        if len(v):
            vals.append(v)
        seen += len(chunk)
        if seen >= max_rows:
            break
    return _percentile_limits(vals, q)


def plot_variant_level_beta_distribution(
    variant_level_path: str,
    *,
    beta_col: str | None = None,
    n_bins: int = 120,
    percentile_cap: float = 0.999,
    density: bool = True,
    log_y: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    out_path: str | None = None,
    show: bool = True,
    ax=None,
) -> dict:
    """Histogram of β by common vs rare from a pooled ``variant_level.csv.gz``."""
    path = os.path.abspath(variant_level_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    col = resolve_beta_col(beta_col)
    header = pd.read_csv(path, nrows=0, compression="infer").columns.tolist()
    if col not in header and POSTERIOR_BETA_COL in header:
        col = POSTERIOR_BETA_COL
    if col not in header:
        raise ValueError(f"Column {col!r} not in {path}")

    lo, hi = _percentile_limits_from_csv(path, col, percentile_cap)
    bins = np.linspace(lo, hi, n_bins + 1)
    rare_col = VARIANT_RARE_COL if VARIANT_RARE_COL in header else COMMON_MASK_COL

    common_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    rare_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    n_common = n_rare = 0
    for chunk in pd.read_csv(
        path, usecols=[col, rare_col], chunksize=CHUNK, compression="infer"
    ):
        beta = chunk[col].astype(np.float64).to_numpy()
        if rare_col == VARIANT_RARE_COL:
            rare_mask = chunk[rare_col].astype(bool).to_numpy()
        else:
            rare_mask = ~chunk[rare_col].astype(bool).to_numpy()
        ok = np.isfinite(beta)
        beta, rare_mask = beta[ok], rare_mask[ok]
        if len(beta) == 0:
            continue
        hc, _ = np.histogram(beta[~rare_mask], bins=bins)
        hr, _ = np.histogram(beta[rare_mask], bins=bins)
        common_counts += hc
        rare_counts += hr
        n_common += int((~rare_mask).sum())
        n_rare += int(rare_mask.sum())

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(9, 5))

    _render_stratum_histogram(
        ax,
        bins,
        common_counts,
        rare_counts,
        n_common,
        n_rare,
        density=density,
        xlabel=xlabel or r"Full-panel $\beta$",
        title=title,
    )
    if log_y:
        ax.set_yscale("log")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")

    if show and created_fig:
        plt.show()
    elif created_fig:
        plt.close(ax.figure)

    return {
        "variant_level_path": path,
        "beta_col": col,
        "n_common": n_common,
        "n_rare": n_rare,
        "ax": ax,
    }


def load_common_beta_pairs(
    post_pergene_root: str,
    *,
    chromosomes: list[int] | None = None,
    posterior_col: str = DECOUPLED_POSTERIOR_BETA_COL,
    computed_col: str | None = None,
) -> pd.DataFrame:
    root = os.path.abspath(post_pergene_root)
    posterior_col = resolve_computed_beta_col(computed_col or posterior_col)
    paths = _filter_chromosome_paths(discover_beta_panel_paths(root), chromosomes)
    if not paths:
        raise FileNotFoundError(f"No full-panel beta CSVs found under {root}")

    usecols = [COMMON_MASK_COL, posterior_col]
    parts: list[pd.DataFrame] = []
    for path in paths:
        chrom, ensg = _gene_from_panel_path(path)
        header = pd.read_csv(path, nrows=0, compression="infer").columns
        saved_col = _resolve_saved_common_beta_col(header)
        df = pd.read_csv(
            path, usecols=[*usecols, saved_col], compression="infer"
        )
        if posterior_col not in df.columns:
            continue
        mask = (
            df[COMMON_MASK_COL].astype(bool)
            & df[saved_col].notna()
            & df[posterior_col].notna()
        )
        if not mask.any():
            continue
        sub = pd.DataFrame(
            {
                "beta_pergene": df.loc[mask, saved_col].astype(float).to_numpy(),
                "beta_posterior": df.loc[mask, posterior_col].astype(float).to_numpy(),
                "chrom": chrom,
                "ensg": ensg,
                "gene": f"chr{chrom}/{ensg}" if chrom and ensg else os.path.basename(path),
                "panel_path": path,
                "posterior_col": posterior_col,
            }
        )
        parts.append(sub)

    if not parts:
        raise ValueError(
            f"No aligned common variants under {root} "
            f"(posterior_col={posterior_col!r}; re-run post-train if decoupled column missing)"
        )
    out = pd.concat(parts, ignore_index=True)
    out["beta_diff"] = out["beta_pergene"] - out["beta_posterior"]
    return out


def summarize_common_beta_pairs(pairs: pd.DataFrame) -> dict:
    x = pairs["beta_pergene"].astype(float)
    y = pairs["beta_posterior"].astype(float)
    diff = pairs["beta_diff"].astype(float) if "beta_diff" in pairs.columns else x - y
    return {
        "n_variants": int(len(pairs)),
        "n_genes": int(pairs["gene"].nunique()) if "gene" in pairs.columns else None,
        "pearson": float(x.corr(y)) if len(pairs) > 1 else float("nan"),
        "spearman": float(x.corr(y, method="spearman")) if len(pairs) > 1 else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "median_abs_diff": float(np.abs(diff).median()),
        "mean_pergene": float(x.mean()),
        "mean_posterior": float(y.mean()),
        "mean_diff": float(diff.mean()),
    }


def _subsample_array(values: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=max_points, replace=False)


def plot_common_beta_distribution_comparison(
    post_pergene_root: str,
    *,
    chromosomes: list[int] | None = None,
    posterior_col: str = DECOUPLED_POSTERIOR_BETA_COL,
    computed_col: str | None = None,
    n_bins: int = 100,
    percentile_cap: float = 0.999,
    max_points: int | None = 500_000,
    sample_seed: int = 0,
    density: bool = True,
    out_path: str | None = None,
    show: bool = True,
) -> dict:
    """
    Common-only β distributions: saved pergene mean vs post-train posterior, plus their difference.

    Three panels: (1) overlaid saved vs posterior, (2) difference histogram, (3) optional stats in title.
    """
    pairs = load_common_beta_pairs(
        post_pergene_root,
        chromosomes=chromosomes,
        posterior_col=posterior_col,
        computed_col=computed_col,
    )
    stats = summarize_common_beta_pairs(pairs)
    post_label = (
        "post G_common β"
        if posterior_col == DECOUPLED_POSTERIOR_BETA_COL
        else "post G_full β"
    )

    saved = _subsample_array(pairs["beta_pergene"].to_numpy(), max_points, sample_seed)
    post = _subsample_array(pairs["beta_posterior"].to_numpy(), max_points, sample_seed + 1)
    diff = _subsample_array(pairs["beta_diff"].to_numpy(), max_points, sample_seed + 2)

    lo, hi = _percentile_limits([saved, post, diff], percentile_cap)
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for data, color, label in [
        (saved, SAVED_COLOR, "saved pergene"),
        (post, POST_COLOR, post_label),
    ]:
        axes[0].hist(
            data,
            bins=bins,
            density=density,
            alpha=0.55,
            color=color,
            label=label,
            edgecolor="black",
            linewidth=0.15,
        )
    axes[0].axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[0].set_xlabel(r"Common $\beta$")
    axes[0].set_ylabel("Density" if density else "Count")
    axes[0].set_title(f"Saved vs post-train posterior (n={len(pairs):,} variants)")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].hist(
        diff,
        bins=bins,
        density=density,
        alpha=0.7,
        color=DIFF_COLOR,
        edgecolor="black",
        linewidth=0.15,
    )
    axes[1].axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[1].set_xlabel(r"$\beta_\mathrm{saved} - \beta_\mathrm{posterior}$")
    axes[1].set_ylabel("Density" if density else "Count")
    axes[1].set_title(
        f"Difference (median |Δ|={stats['median_abs_diff']:.4g}, r={stats['pearson']:.3f})"
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    chrom_txt = f"chr{','.join(map(str, chromosomes))}" if chromosomes else "genome-wide"
    fig.suptitle(f"Common β alignment ({chrom_txt})", y=1.02)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pairs": pairs, "stats": stats, "fig": fig}


def _subsample_pairs(pairs: pd.DataFrame, max_points: int | None, seed: int) -> pd.DataFrame:
    if max_points is None or len(pairs) <= max_points:
        return pairs
    return pairs.sample(n=max_points, random_state=seed)


def plot_common_beta_scatter(
    post_pergene_root: str,
    *,
    chromosomes: list[int] | None = None,
    posterior_col: str = DECOUPLED_POSTERIOR_BETA_COL,
    computed_col: str | None = None,
    max_points: int | None = 500_000,
    sample_seed: int = 0,
    plot_style: str = "auto",
    alpha: float = 0.12,
    point_size: float = 4.0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    out_path: str | None = None,
    show: bool = True,
    ax=None,
) -> dict:
    root = os.path.abspath(post_pergene_root)
    col = resolve_computed_beta_col(computed_col or posterior_col)
    pairs = load_common_beta_pairs(
        root, chromosomes=chromosomes, posterior_col=col, computed_col=col
    )
    stats = summarize_common_beta_pairs(pairs)
    plot_df = _subsample_pairs(pairs, max_points, sample_seed)
    ylab_default = (
        r"Post-train $\beta$ (G_common posterior)"
        if col == DECOUPLED_POSTERIOR_BETA_COL
        else r"Post-train posterior $\beta$ (G_full)"
    )

    manifest = load_post_train_manifest(root)
    common_mode = str(manifest.get("common_beta_mode", "posterior"))
    rare_mode = str(manifest.get("rare_beta_mode", "posterior"))
    if common_mode != "posterior":
        import warnings

        warnings.warn(
            f"common_beta_mode={common_mode!r}: beta_hat on common variants is not "
            "the collapsed posterior.",
            stacklevel=2,
        )

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    x = plot_df["beta_pergene"].to_numpy()
    y = plot_df["beta_posterior"].to_numpy()
    style = "hexbin" if plot_style == "auto" and len(plot_df) > 200_000 else plot_style
    if style == "auto":
        style = "scatter"

    if style == "hexbin":
        hb = ax.hexbin(x, y, gridsize=80, cmap="Blues", mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="count")
    elif style == "scatter":
        ax.scatter(x, y, s=point_size, alpha=alpha, c="#4c78a8", edgecolors="none", rasterized=True)
    else:
        raise ValueError(f"Unknown plot_style={plot_style!r}")

    lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) or 1.0
    pad = lim * 1.05
    ax.plot([-pad, pad], [-pad, pad], color="#888888", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlab or r"Pergene saved $\beta$ (common variants)")
    ax.set_ylabel(ylab or ylab_default)
    if title is None:
        chrom_txt = f"chr{','.join(map(str, chromosomes))}" if chromosomes else "genome-wide"
        beta_kind = "G_common" if col == DECOUPLED_POSTERIOR_BETA_COL else "G_full"
        title = (
            f"Common β: pergene vs post-train ({beta_kind}, {chrom_txt})\n"
            f"n={stats['n_variants']:,} variants | r={stats['pearson']:.3f}, RMSE={stats['rmse']:.4g}"
        )
    ax.set_title(title)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        ax.figure.savefig(out_path, dpi=160, bbox_inches="tight")

    if show and created_fig:
        plt.show()
    elif created_fig:
        plt.close(ax.figure)

    return {
        "post_pergene_root": root,
        "chromosomes": chromosomes,
        "pairs": pairs,
        "plot_pairs": plot_df,
        "stats": stats,
        "manifest": manifest,
        "common_beta_mode": common_mode,
        "rare_beta_mode": rare_mode,
        "posterior_col": col,
        "computed_col": col,
        "ax": ax,
    }


def load_common_posterior_pairs(
    post_pergene_root: str,
    *,
    chromosomes: list[int] | None = None,
    full_col: str = G_FULL_POSTERIOR_BETA_COL,
    g_common_col: str = G_COMMON_POSTERIOR_BETA_COL,
) -> pd.DataFrame:
    """Aligned common-variant β pairs: G_full posterior vs G_common-only posterior."""
    root = os.path.abspath(post_pergene_root)
    paths = _filter_chromosome_paths(discover_beta_panel_paths(root), chromosomes)
    if not paths:
        raise FileNotFoundError(f"No full-panel beta CSVs found under {root}")

    parts: list[pd.DataFrame] = []
    for path in paths:
        chrom, ensg = _gene_from_panel_path(path)
        header = pd.read_csv(path, nrows=0, compression="infer").columns.tolist()
        if full_col not in header or g_common_col not in header:
            continue
        df = pd.read_csv(
            path,
            usecols=[COMMON_MASK_COL, full_col, g_common_col],
            compression="infer",
        )
        mask = (
            df[COMMON_MASK_COL].astype(bool)
            & df[full_col].notna()
            & df[g_common_col].notna()
        )
        if not mask.any():
            continue
        sub = pd.DataFrame(
            {
                "beta_g_full": df.loc[mask, full_col].astype(float).to_numpy(),
                "beta_g_common": df.loc[mask, g_common_col].astype(float).to_numpy(),
                "chrom": chrom,
                "ensg": ensg,
                "gene": f"chr{chrom}/{ensg}" if chrom and ensg else os.path.basename(path),
                "panel_path": path,
                "full_col": full_col,
                "g_common_col": g_common_col,
            }
        )
        parts.append(sub)

    if not parts:
        raise ValueError(
            f"No aligned common posterior pairs under {root} "
            f"(need {full_col!r} and {g_common_col!r}; re-run post-train if decoupled missing)"
        )
    out = pd.concat(parts, ignore_index=True)
    out["beta_diff"] = out["beta_g_full"] - out["beta_g_common"]
    return out


def summarize_common_posterior_pairs(pairs: pd.DataFrame) -> dict:
    x = pairs["beta_g_full"].astype(float)
    y = pairs["beta_g_common"].astype(float)
    diff = pairs["beta_diff"].astype(float) if "beta_diff" in pairs.columns else x - y
    return {
        "n_variants": int(len(pairs)),
        "n_genes": int(pairs["gene"].nunique()) if "gene" in pairs.columns else None,
        "pearson": float(x.corr(y)) if len(pairs) > 1 else float("nan"),
        "spearman": float(x.corr(y, method="spearman")) if len(pairs) > 1 else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "median_abs_diff": float(np.abs(diff).median()),
        "mean_g_full": float(x.mean()),
        "mean_g_common": float(y.mean()),
        "mean_diff": float(diff.mean()),
    }


def plot_common_posterior_scatter(
    post_pergene_root: str,
    *,
    chromosomes: list[int] | None = None,
    full_col: str = G_FULL_POSTERIOR_BETA_COL,
    g_common_col: str = G_COMMON_POSTERIOR_BETA_COL,
    max_points: int | None = 500_000,
    sample_seed: int = 0,
    plot_style: str = "auto",
    alpha: float = 0.12,
    point_size: float = 4.0,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    out_path: str | None = None,
    show: bool = True,
    ax=None,
) -> dict:
    """Scatter of common β: G_full posterior vs G_common-only posterior."""
    root = os.path.abspath(post_pergene_root)
    pairs = load_common_posterior_pairs(
        root,
        chromosomes=chromosomes,
        full_col=full_col,
        g_common_col=g_common_col,
    )
    stats = summarize_common_posterior_pairs(pairs)
    plot_df = _subsample_pairs(pairs, max_points, sample_seed)

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    x = plot_df["beta_g_full"].to_numpy()
    y = plot_df["beta_g_common"].to_numpy()
    style = "hexbin" if plot_style == "auto" and len(plot_df) > 200_000 else plot_style
    if style == "auto":
        style = "scatter"

    if style == "hexbin":
        hb = ax.hexbin(x, y, gridsize=80, cmap="Oranges", mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="count")
    elif style == "scatter":
        ax.scatter(x, y, s=point_size, alpha=alpha, c=G_COMMON_COLOR, edgecolors="none", rasterized=True)
    else:
        raise ValueError(f"Unknown plot_style={plot_style!r}")

    lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) or 1.0
    pad = lim * 1.05
    ax.plot([-pad, pad], [-pad, pad], color="#888888", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlab or r"Common $\beta$ (G_full posterior)")
    ax.set_ylabel(ylab or r"Common $\beta$ (G_common posterior)")
    if title is None:
        chrom_txt = f"chr{','.join(map(str, chromosomes))}" if chromosomes else "genome-wide"
        title = (
            f"Common β: G_full vs G_common posterior ({chrom_txt})\n"
            f"n={stats['n_variants']:,} variants | r={stats['pearson']:.3f}, RMSE={stats['rmse']:.4g}"
        )
    ax.set_title(title)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        ax.figure.savefig(out_path, dpi=160, bbox_inches="tight")

    if show and created_fig:
        plt.show()
    elif created_fig:
        plt.close(ax.figure)

    return {
        "post_pergene_root": root,
        "chromosomes": chromosomes,
        "pairs": pairs,
        "plot_pairs": plot_df,
        "stats": stats,
        "full_col": full_col,
        "g_common_col": g_common_col,
        "ax": ax,
    }


def plot_common_posterior_distribution_comparison(
    post_pergene_root: str,
    *,
    chromosomes: list[int] | None = None,
    full_col: str = G_FULL_POSTERIOR_BETA_COL,
    g_common_col: str = G_COMMON_POSTERIOR_BETA_COL,
    n_bins: int = 100,
    percentile_cap: float = 0.999,
    max_points: int | None = 500_000,
    sample_seed: int = 0,
    density: bool = True,
    out_path: str | None = None,
    show: bool = True,
) -> dict:
    """Common-only β distributions: G_full vs G_common posterior, plus their difference."""
    pairs = load_common_posterior_pairs(
        post_pergene_root,
        chromosomes=chromosomes,
        full_col=full_col,
        g_common_col=g_common_col,
    )
    stats = summarize_common_posterior_pairs(pairs)

    g_full = _subsample_array(pairs["beta_g_full"].to_numpy(), max_points, sample_seed)
    g_common = _subsample_array(pairs["beta_g_common"].to_numpy(), max_points, sample_seed + 1)
    diff = _subsample_array(pairs["beta_diff"].to_numpy(), max_points, sample_seed + 2)

    lo, hi = _percentile_limits([g_full, g_common, diff], percentile_cap)
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for data, color, label in [
        (g_full, POST_COLOR, "G_full posterior"),
        (g_common, G_COMMON_COLOR, "G_common posterior"),
    ]:
        axes[0].hist(
            data,
            bins=bins,
            density=density,
            alpha=0.55,
            color=color,
            label=label,
            edgecolor="black",
            linewidth=0.15,
        )
    axes[0].axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[0].set_xlabel(r"Common $\beta$")
    axes[0].set_ylabel("Density" if density else "Count")
    axes[0].set_title(f"G_full vs G_common posterior (n={len(pairs):,} variants)")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].hist(
        diff,
        bins=bins,
        density=density,
        alpha=0.7,
        color=DIFF_COLOR,
        edgecolor="black",
        linewidth=0.15,
    )
    axes[1].axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[1].set_xlabel(r"$\beta_\mathrm{G\_full} - \beta_\mathrm{G\_common}$")
    axes[1].set_ylabel("Density" if density else "Count")
    axes[1].set_title(
        f"Difference (median |Δ|={stats['median_abs_diff']:.4g}, r={stats['pearson']:.3f})"
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    chrom_txt = f"chr{','.join(map(str, chromosomes))}" if chromosomes else "genome-wide"
    fig.suptitle(f"Common posterior β coupling ({chrom_txt})", y=1.02)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"pairs": pairs, "stats": stats, "fig": fig}
