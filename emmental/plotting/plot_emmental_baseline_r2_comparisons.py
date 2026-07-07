#!/usr/bin/env python3
"""CLI wrapper around ``r2`` Emmental vs baseline comparison helpers."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2

BASELINE_ROOT = r2.BASELINE_ROOTS["common01"]
THRESHOLDS = r2.THRESHOLDS
OUT_DIR = "plots"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emmental-post-root", type=str, required=True)
    parser.add_argument("--baseline-root", type=str, default=BASELINE_ROOT)
    parser.add_argument("--common-beta", type=str, default="common", choices=r2.COMMON_BETA_SOURCES)
    parser.add_argument(
        "--rare-beta",
        type=str,
        default=None,
        choices=list(r2.RARE_BETA_SOURCES),
        help="If set, compare common+rare panel R²; else common-only.",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    thresholds = [float(t) for t in args.thresholds]
    prep = r2.prep_emmental_vs_baseline(
        args.emmental_post_root,
        args.baseline_root,
        common_beta=args.common_beta,
        rare_beta=args.rare_beta,
        thresholds=thresholds,
    )
    slug = r2.post_train_r2_col(args.common_beta, args.rare_beta).replace("__", "_")
    prep["props"].to_csv(
        os.path.join(out_dir, f"emmental_vs_baseline_{slug}_props.csv"),
        index=False,
    )
    prep["means"].to_csv(
        os.path.join(out_dir, f"emmental_vs_baseline_{slug}_means.csv"),
        index=False,
    )

    for thr in thresholds:
        thr_slug = str(thr).replace(".", "p")
        sub = prep["props"][prep["props"]["threshold"] == thr]
        r2.plot_baseline_props_bar(
            sub,
            thr,
            os.path.join(out_dir, f"emmental_vs_baseline_{slug}_gt_{thr_slug}.png"),
            title=(
                f"Emmental vs baseline: fraction with R² > {thr} "
                f"(n={prep['n_genes']} genes, {prep['r2_col']})"
            ),
            show=False,
        )

    print(f"Wrote summary and plots to {out_dir} (n_genes={prep['n_genes']})")
    print(
        prep["props"]
        .pivot_table(index="method", columns=["threshold", "split"], values="prop")
        .round(3)
    )


if __name__ == "__main__":
    main()
