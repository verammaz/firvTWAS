#!/usr/bin/env python3
"""Merge per-chr expression tail outputs into one genome-wide directory."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_expression_tails import _genome_summary  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Merge chr*/ tail analysis outputs.")
    p.add_argument("--root", required=True, help="Parent dir containing chr1/ ... chr22/")
    p.add_argument("--splits", default="test")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    chr_dirs = sorted(glob.glob(os.path.join(root, "chr*")))
    if not chr_dirs:
        raise SystemExit(f"No chr* subdirs under {root}")

    gene_parts = []
    ind_parts = []
    for d in chr_dirs:
        gpath = os.path.join(d, "gene_tail_summary.csv")
        ipath = os.path.join(d, "individual_gene_residuals.csv.gz")
        if os.path.isfile(gpath):
            gene_parts.append(pd.read_csv(gpath))
        if os.path.isfile(ipath):
            ind_parts.append(pd.read_csv(ipath, compression="infer"))

    if not gene_parts:
        raise SystemExit("No gene_tail_summary.csv files found")

    gene_df = pd.concat(gene_parts, ignore_index=True)
    gene_df.to_csv(os.path.join(root, "gene_tail_summary.csv"), index=False)

    ind_df = pd.concat(ind_parts, ignore_index=True) if ind_parts else pd.DataFrame()
    if not ind_df.empty:
        ind_df.to_csv(
            os.path.join(root, "individual_gene_residuals.csv.gz"),
            index=False,
            compression="gzip",
        )

    ok = gene_df[~gene_df["error"].notna()] if "error" in gene_df.columns else gene_df
    manifest = {
        "merged_from": chr_dirs,
        "n_genes": int(len(gene_df)),
        "n_individual_rows": int(len(ind_df)),
    }
    for split in splits:
        manifest[f"genome_{split}"] = _genome_summary(ok, split=split)

    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Merged {len(gene_df)} genes from {len(chr_dirs)} chr dirs -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
