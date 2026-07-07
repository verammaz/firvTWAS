#!/usr/bin/env python3
"""Compare saved pergene common β vs post-train recomputed β (panel CSVs)."""
from __future__ import annotations

import argparse
import json
import os

from param_plots import (
    DECOUPLED_POSTERIOR_BETA_COL,
    POSTERIOR_BETA_COL,
    load_common_beta_pairs,
    plot_common_beta_scatter,
    resolve_computed_beta_col,
    summarize_common_beta_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Correlation of saved pergene common β vs post-train computed β "
            "(from full_beta_panel CSVs)."
        )
    )
    parser.add_argument(
        "post_pergene_root",
        help="Post-train root (e.g. .../train_common01_normG_collapse/post_pergene)",
    )
    parser.add_argument(
        "--computed-col",
        "--computed_col",
        default="G_common_decoupled",
        help=(
            "Post-train panel column for computed common β "
            f"(default: G_common_decoupled -> {DECOUPLED_POSTERIOR_BETA_COL}; "
            f"aliases: G_full, beta_hat)"
        ),
    )
    parser.add_argument(
        "--chromosomes",
        nargs="*",
        type=int,
        default=None,
        help="Optional chr list (default: all chromosomes with panels)",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Optional scatter plot output path",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Subsample variants for scatter only (stats use all variants)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path to write summary stats JSON",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display matplotlib window",
    )
    args = parser.parse_args()

    computed_col = resolve_computed_beta_col(args.computed_col)
    pairs = load_common_beta_pairs(
        args.post_pergene_root,
        chromosomes=args.chromosomes,
        computed_col=computed_col,
    )
    stats = summarize_common_beta_pairs(pairs)

    print(f"post_pergene_root: {os.path.abspath(args.post_pergene_root)}")
    print(f"computed_col:      {computed_col}")
    print(f"n_variants:        {stats['n_variants']:,}")
    print(f"n_genes:           {stats['n_genes']:,}")
    print(f"pearson:           {stats['pearson']:.6f}")
    print(f"spearman:          {stats['spearman']:.6f}")
    print(f"rmse:              {stats['rmse']:.6g}")
    print(f"median_abs_diff:   {stats['median_abs_diff']:.6g}")

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)) or ".", exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(
                {
                    "post_pergene_root": os.path.abspath(args.post_pergene_root),
                    "computed_col": computed_col,
                    **stats,
                },
                f,
                indent=2,
            )
        print(f"wrote stats: {args.json_path}")

    if args.out_path:
        plot_common_beta_scatter(
            args.post_pergene_root,
            chromosomes=args.chromosomes,
            computed_col=computed_col,
            max_points=args.max_points,
            out_path=args.out_path,
            show=not args.no_show,
        )
        print(f"wrote plot: {args.out_path}")


if __name__ == "__main__":
    main()
