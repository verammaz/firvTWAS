#!/usr/bin/env python3
"""
Scale ChromBPNet log-count-diff columns in per-gene annotation matrices.

Each file is {gene}_annotations.tsv.gz under <annotation_root>/chr{N}/.
Columns log_counts_diff_chrombpnet_<cell> are replaced with z-scores using the
same joblib scalers as chrombpnet_scaling (StandardScaler with_mean=False on the
reference rare-variant distribution). All other columns are unchanged.

Scalers are keyed by (cell_type, assay). Per-gene matrices store one value per
cell type (no assay dimension); by default we use the ATAC scaler for each cell.
Override with --assay H3K27ac or H3K4me3 if your upstream values match that assay.

Example:
  python scale_per_gene_chrombpnet_annotations.py \\
    --annotation-root /path/to/annotations \\
    --output-root /path/to/annotations_chrombpnet_zscore \\
    --chromosomes 1-22
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_SCALERS = (
    "/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/"
    "annotations_hg38/merged_annotations_ADSP_v2/chrombpnet/"
    "variant_peak_pairs_scored/zscore_scaler/zscore_scalers.joblib"
)

CELL_TYPES = ["microglia", "astrocyte", "neuron", "oligodendrocyte"]
ASSAYS = ["ATAC", "H3K27ac", "H3K4me3"]
SHORT_NAME = "log_counts_diff"


def chrombpnet_columns() -> list[str]:
    return [f"log_counts_diff_chrombpnet_{ct}" for ct in CELL_TYPES]


def parse_chromosomes(spec: str) -> list[int]:
    """Parse '1', '1,2,3', or '1-22' into a sorted unique list of ints."""
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def scaler_key(cell: str, assay: str) -> str:
    return f"{cell}__{assay}__{SHORT_NAME}"


def scale_column(values: np.ndarray, scaler) -> np.ndarray:
    """Apply scaler; non-finite inputs stay NaN; finite values transformed in-place logic."""
    x = np.asarray(values, dtype=np.float64)
    out = x.copy()
    finite = np.isfinite(x)
    if not finite.any():
        return out
    col = x[finite].reshape(-1, 1)
    out[finite] = scaler.transform(col).ravel()
    return out


def process_file(path: str, scalers: dict, assay: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="gzip")
    missing = [c for c in chrombpnet_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    for cell in CELL_TYPES:
        col = f"log_counts_diff_chrombpnet_{cell}"
        key = scaler_key(cell, assay)
        if key not in scalers:
            raise KeyError(f"Scaler not found: {key!r} (available cell/assay combo?)")
        df[col] = scale_column(df[col].to_numpy(), scalers[key])

    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--annotation-root",
        required=True,
        help="Root folder containing chr1, chr2, ... subdirectories",
    )
    p.add_argument(
        "--output-root",
        required=True,
        help="Destination root (same chr*/gene_annotations.tsv.gz layout)",
    )
    p.add_argument(
        "--chromosomes",
        default="1-22",
        help="Comma list and/or ranges, e.g. 1-22 or 1,3,5 (default: 1-22)",
    )
    p.add_argument(
        "--scalers",
        default=DEFAULT_SCALERS,
        help="Path to zscore_scalers.joblib",
    )
    p.add_argument(
        "--assay",
        choices=ASSAYS,
        default="ATAC",
        help="Which assay scaler to use for the single per-cell column (default: ATAC)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and transform one file per chromosome only; do not write",
    )
    p.add_argument(
        "--limit-genes",
        type=int,
        default=0,
        help="If >0, process at most this many genes per chromosome (debug)",
    )
    args = p.parse_args()

    chroms = parse_chromosomes(args.chromosomes)
    print(f"Loading scalers from {args.scalers} ...", file=sys.stderr)
    scalers = joblib.load(args.scalers)

    for assay in ASSAYS:
        for cell in CELL_TYPES:
            k = scaler_key(cell, assay)
            if k not in scalers:
                print(f"Warning: missing scaler key {k!r}", file=sys.stderr)

    for chrom in chroms:
        sub_in = os.path.join(args.annotation_root, f"chr{chrom}")
        sub_out = os.path.join(args.output_root, f"chr{chrom}")
        if not os.path.isdir(sub_in):
            print(f"Skip missing input dir: {sub_in}", file=sys.stderr)
            continue

        os.makedirs(sub_out, exist_ok=True)
        files = sorted(
            f
            for f in os.listdir(sub_in)
            if f.endswith("_annotations.tsv.gz")
            and re.match(r"^.+_annotations\.tsv\.gz$", f)
        )
        if args.limit_genes:
            files = files[: args.limit_genes]

        if args.dry_run and files:
            sample = os.path.join(sub_in, files[0])
            print(f"Dry-run sample chr{chrom}: {sample}", file=sys.stderr)
            process_file(sample, scalers, args.assay)
            continue

        for fname in tqdm(files, desc=f"chr{chrom}"):
            in_path = os.path.join(sub_in, fname)
            out_path = os.path.join(sub_out, fname)
            df = process_file(in_path, scalers, args.assay)
            df.to_csv(out_path, sep="\t", index=False, compression="gzip")

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
