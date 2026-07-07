#!/usr/bin/env python3
"""
Backfill ``tau_T.csv`` under joint ``run_*`` dirs when it was missing (e.g. collapsed joint).

Prefers ``posterior_stats.npz``; falls back to last row of ``tau_history.csv`` plus
threshold from npz or ``--default-threshold``.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from save_outputs import (
    tau_T_from_tau_history_last_row,
    write_tau_T_csv,
)


def _annotations_from_tau_history(tau_history_path: str) -> list:
    hist = pd.read_csv(tau_history_path, nrows=0)
    return [c[len("Tau2_") :] for c in hist.columns if c.startswith("Tau2_")]


def _load_posterior_stats(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k].item() for k in data.files}


def backfill_run(run_dir: str, default_threshold: float, force: bool) -> bool:
    out_path = os.path.join(run_dir, "tau_T.csv")
    if os.path.isfile(out_path) and not force:
        print(f"SKIP {run_dir} (tau_T.csv exists)")
        return False

    npz_path = os.path.join(run_dir, "posterior_stats.npz")
    hist_path = os.path.join(run_dir, "tau_history.csv")
    annotations = None
    if os.path.isfile(hist_path):
        annotations = _annotations_from_tau_history(hist_path)

    if os.path.isfile(npz_path):
        stats = _load_posterior_stats(npz_path)
        if annotations is None:
            tau2_len = len(np.asarray(stats["tau2"]["mean"]).ravel())
            annotations = [f"anno_{i}" for i in range(tau2_len)]
        write_tau_T_csv(run_dir, stats, annotations)
        print(f"Wrote {out_path} from posterior_stats.npz")
        return True

    if not os.path.isfile(hist_path):
        print(f"SKIP {run_dir} (no posterior_stats.npz or tau_history.csv)")
        return False

    threshold = default_threshold
    tau_df = tau_T_from_tau_history_last_row(hist_path, threshold)
    tau_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} from tau_history.csv (T={threshold})")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--joint-root",
        required=True,
        help="Joint output dir containing run_*/ subdirs",
    )
    p.add_argument("--default-threshold", type=float, default=0.05)
    p.add_argument("--force", action="store_true", help="Overwrite existing tau_T.csv")
    args = p.parse_args()

    joint_root = os.path.abspath(args.joint_root)
    run_dirs = sorted(
        [d for d in glob.glob(os.path.join(joint_root, "run_*")) if os.path.isdir(d)],
        key=lambda x: int(re.search(r"run_(\d+)", x).group(1)),
    )
    if not run_dirs:
        raise SystemExit(f"No run_* under {joint_root}")

    n = sum(
        backfill_run(rd, args.default_threshold, args.force)
        for rd in run_dirs
    )
    print(f"Done: wrote tau_T.csv in {n}/{len(run_dirs)} run dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
