#!/usr/bin/env python3
"""CLI for β distribution plots (implementation in ``param_plots.py``)."""
from __future__ import annotations

import argparse
import os

from param_plots import (
    beta_col_to_common_source,
    discover_beta_panel_paths,
    plot_beta_distribution_by_stratum,
    plot_saved_common_beta_distribution,
    plot_variant_level_beta_distribution,
    resolve_beta_col,
)


def plot_beta_distribution_from_panels(
    post_pergene_root: str,
    out_path: str,
    *,
    n_bins: int = 120,
    percentile_cap: float = 0.999,
    density: bool = True,
    log_y: bool = False,
    title: str | None = None,
    beta_col: str | None = None,
    xlabel: str | None = None,
) -> None:
    """Write common vs rare β histogram from per-gene panel CSVs."""
    common_source = beta_col_to_common_source(beta_col)
    plot_beta_distribution_by_stratum(
        post_pergene_root,
        common_source=common_source,
        n_bins=n_bins,
        percentile_cap=percentile_cap,
        density=density,
        log_y=log_y,
        title=title,
        out_path=out_path,
        show=False,
    )
    col = resolve_beta_col(beta_col)
    n_panels = len(discover_beta_panel_paths(post_pergene_root))
    print(
        f"wrote {out_path}  (common_source={common_source}, beta_col={col}, "
        f"n_panels={n_panels})"
    )


def plot_saved_common_beta_distribution_from_panels(
    post_pergene_root: str,
    out_path: str,
    *,
    n_bins: int = 120,
    percentile_cap: float = 0.999,
    density: bool = True,
    log_y: bool = False,
    title: str | None = None,
) -> None:
    """Write saved common β histogram from per-gene panel CSVs."""
    plot_saved_common_beta_distribution(
        post_pergene_root,
        n_bins=n_bins,
        percentile_cap=percentile_cap,
        density=density,
        log_y=log_y,
        title=title,
        out_path=out_path,
        show=False,
    )
    print(f"wrote {out_path}")


def plot_beta_distribution(
    variant_level_path: str,
    out_path: str,
    *,
    n_bins: int = 120,
    percentile_cap: float = 0.999,
    density: bool = True,
    log_y: bool = False,
    title: str | None = None,
    beta_col: str | None = None,
    xlabel: str | None = None,
) -> None:
    """Write common vs rare β histogram from a pooled variant-level table."""
    result = plot_variant_level_beta_distribution(
        variant_level_path,
        beta_col=beta_col,
        n_bins=n_bins,
        percentile_cap=percentile_cap,
        density=density,
        log_y=log_y,
        title=title,
        xlabel=xlabel,
        out_path=out_path,
        show=False,
    )
    print(
        f"wrote {out_path}  (beta_col={result['beta_col']}, "
        f"n_common={result['n_common']:,}, n_rare={result['n_rare']:,})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant_level",
        default=None,
        help="Pooled variant_level.csv.gz (from diagnose_posttrain_rare.py).",
    )
    p.add_argument(
        "--post_pergene_root",
        default=None,
        help="Read full_beta_panel CSVs directly under post_pergene/chr*/.",
    )
    p.add_argument("--out", default=None)
    p.add_argument(
        "--experiment_roots",
        nargs="+",
        default=None,
        help="Batch: {root}/post_pergene -> {root}/diagnostics/post_pergene_variant_only/",
    )
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--percentile", type=float, default=0.999)
    p.add_argument("--log_y", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument(
        "--beta_col",
        default=None,
        help=(
            "CSV column to plot (default: beta_hat). "
            "Aliases: beta_hat, beta_full, assembled, saved_common, train_posterior, mu."
        ),
    )
    p.add_argument("--xlabel", default=None, help="Override x-axis label (default: from manifest).")
    args = p.parse_args()

    beta_col = args.beta_col

    if args.experiment_roots:
        for root in args.experiment_roots:
            root = os.path.abspath(root)
            label = os.path.basename(root.rstrip("/"))
            out_dir = os.path.join(root, "diagnostics", "post_pergene_variant_only")
            post_root = os.path.join(root, "post_pergene")
            for log_y, suffix in [(False, ""), (True, "_logy")]:
                plot_beta_distribution_from_panels(
                    post_root,
                    os.path.join(out_dir, f"beta_distribution_by_rare{suffix}.png"),
                    n_bins=args.bins,
                    percentile_cap=args.percentile,
                    log_y=log_y,
                    title=f"{label}: β by common vs rare (x clipped to |β| p{100*args.percentile:g})",
                    beta_col=beta_col,
                    xlabel=args.xlabel,
                )
        return

    if args.post_pergene_root:
        out = args.out or os.path.join(
            os.path.dirname(os.path.abspath(args.post_pergene_root)),
            "diagnostics",
            "post_pergene_variant_only",
            "beta_distribution_by_rare.png",
        )
        plot_beta_distribution_from_panels(
            args.post_pergene_root,
            out,
            n_bins=args.bins,
            percentile_cap=args.percentile,
            log_y=args.log_y,
            title=args.title,
            beta_col=beta_col,
            xlabel=args.xlabel,
        )
        return

    variant_level = args.variant_level or (
        "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01/full/"
        "diagnostics/post_pergene_variant_only/variant_level.csv.gz"
    )
    out = args.out or (
        "/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common01/full/"
        "diagnostics/post_pergene_variant_only/beta_distribution_by_rare.png"
    )
    plot_beta_distribution(
        variant_level,
        out,
        n_bins=args.bins,
        percentile_cap=args.percentile,
        log_y=args.log_y,
        title=args.title,
        beta_col=beta_col,
        xlabel=args.xlabel,
    )


if __name__ == "__main__":
    main()
