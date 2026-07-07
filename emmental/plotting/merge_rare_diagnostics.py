#!/usr/bin/env python3
"""Merge per-chromosome rare-diagnostics outputs into one genome-wide summary."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from diagnose_posttrain_rare import (  # noqa: E402
    BASELINE_ROOT_DEFAULT,
    _plot_gate_effect,
    _plot_r2_decomposition,
    build_methods_comparison_table,
    summarize_gate_comparison,
    summarize_lambda_zero_by_annotation,
    summarize_mu_maf_lambda,
    summarize_r2_decomposition,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chr_dirs_glob",
        required=True,
        help="Glob for chr subdirs, e.g. .../diagnostics/post_pergene/chr*/",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--baseline_root", default=BASELINE_ROOT_DEFAULT)
    p.add_argument("--gene_list", default=None)
    args = p.parse_args()

    chr_dirs = sorted(glob.glob(args.chr_dirs_glob))
    if not chr_dirs:
        raise SystemExit(f"No dirs match {args.chr_dirs_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    r2_parts, gate_parts, var_parts = [], [], []
    for d in chr_dirs:
        r2p = os.path.join(d, "r2_decomposition_per_gene.csv")
        if os.path.isfile(r2p):
            r2_parts.append(pd.read_csv(r2p))
        gp = os.path.join(d, "gate_comparison_variants.csv.gz")
        if os.path.isfile(gp):
            gate_parts.append(pd.read_csv(gp, compression="infer"))
        vp = os.path.join(d, "variant_level.csv.gz")
        if os.path.isfile(vp):
            var_parts.append(pd.read_csv(vp, compression="infer"))

    if var_parts:
        var_df = pd.concat(var_parts, ignore_index=True)
        var_df.to_csv(os.path.join(args.out_dir, "variant_level.csv.gz"), index=False, compression="gzip")
        maf_summary, stratum_summary = summarize_mu_maf_lambda(var_df)
        maf_summary.to_csv(os.path.join(args.out_dir, "abs_mu_vs_maf_binned.csv"), index=False)
        stratum_summary.to_csv(os.path.join(args.out_dir, "mu_lambda_by_stratum.csv"), index=False)

    if r2_parts:
        r2_df = pd.concat(r2_parts, ignore_index=True)
        r2_ok = r2_df[r2_df.get("error", pd.Series(index=r2_df.index)).isna()] if "error" in r2_df.columns else r2_df
        r2_df.to_csv(os.path.join(args.out_dir, "r2_decomposition_per_gene.csv"), index=False)
        summarize_r2_decomposition(r2_ok).to_csv(
            os.path.join(args.out_dir, "r2_decomposition_summary.csv"), index=False
        )
        _plot_r2_decomposition(r2_ok, os.path.join(args.out_dir, "r2_decomposition_boxplot.png"))

    if gate_parts:
        gate_df = pd.concat(gate_parts, ignore_index=True)
        gate_df.to_csv(
            os.path.join(args.out_dir, "gate_comparison_variants.csv.gz"), index=False, compression="gzip"
        )
        summarize_gate_comparison(gate_df).to_csv(
            os.path.join(args.out_dir, "gate_comparison_summary.csv"), index=False
        )
        summarize_lambda_zero_by_annotation(gate_df).to_csv(
            os.path.join(args.out_dir, "lambda_zero_by_dominant_annotation.csv"), index=False
        )
        _plot_gate_effect(gate_df, os.path.join(args.out_dir, "gate_mu_rare_scatter.png"))

    if r2_parts and var_parts:
        if args.gene_list and os.path.isfile(args.gene_list):
            with open(args.gene_list) as f:
                gene_ensgs = [
                    line.strip().split("/")[-1]
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        else:
            gene_ensgs = None
        build_methods_comparison_table(
            r2_ok, var_df, args.baseline_root, gene_ensgs=gene_ensgs
        ).to_csv(os.path.join(args.out_dir, "methods_comparison.csv"), index=False)

    manifest = {
        "merged_from": chr_dirs,
        "n_chr_dirs": len(chr_dirs),
        "n_genes_r2": int(len(r2_parts) and len(r2_df)),
        "n_variants_pooled": int(len(var_parts) and len(var_df)),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Merged {len(chr_dirs)} chr dirs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
