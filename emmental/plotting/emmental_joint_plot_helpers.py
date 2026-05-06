"""
Helper functions for plotting parmigiano_joint experiments.

This file exists to avoid editing a possibly corrupted `plot.ipynb`.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_tau_trajectories_grouped_by_three(
    runs_df: pd.DataFrame,
    base_dir: str,
    batch_size: int = 10,
    group_size: int = 3,
    load_joint_run_files=None,
    discover_joint_runs=None,
):
    """
    Render tau trajectories grouped by 3 subplots per figure.

    Parameters
    ----------
    runs_df:
        DataFrame with columns: experiment, run_id, run_path.
    load_joint_run_files:
        Callable(run_path, what=...) -> dict of loaded arrays/frames.
    """
    if load_joint_run_files is None:
        raise ValueError("load_joint_run_files must be provided.")

    if runs_df is None or runs_df.empty:
        print("No runs provided.")
        return

    for exp in sorted(runs_df["experiment"].unique()):
        sub = runs_df[runs_df["experiment"] == exp].sort_values("run_id")
        run_paths = list(sub["run_path"])
        if not run_paths:
            continue

        # Find annotation columns from first available tau_history
        cols = None
        for rp in run_paths:
            files = load_joint_run_files(rp, what=("tau_history",))
            th = files.get("tau_history")
            if th is not None and not th.empty:
                cols = list(th.columns)
                break
        if cols is None:
            continue

        batches = [run_paths[i : i + batch_size] for i in range(0, len(run_paths), batch_size)]

        for start in range(0, len(cols), group_size):
            group = cols[start : start + group_size]
            n_sub = len(group)
            fig, axes = plt.subplots(1, n_sub, figsize=(6 * n_sub, 4), squeeze=False)

            for j, ann in enumerate(group):
                ax = axes[0, j]
                for b_idx, batch in enumerate(batches):
                    trajs = []
                    for rp in batch:
                        files = load_joint_run_files(rp, what=("tau_history",))
                        th = files.get("tau_history")
                        if th is None or ann not in th.columns:
                            continue
                        trajs.append(th[ann].values.astype(float))
                    if not trajs:
                        continue
                    arr = np.vstack(trajs)  # (runs, iters)
                    mean_traj = arr.mean(axis=0)
                    iters = np.arange(len(mean_traj))
                    ax.plot(iters, mean_traj, label=f"batch {b_idx + 1} (n={arr.shape[0]})")

                ax.set_xlabel("Iteration")
                ax.set_ylabel(f"Tau ({ann})")
                ax.set_title(str(ann))
                ax.grid(alpha=0.3)

            fig.suptitle(f"Tau trajectories - {exp} (batched by {batch_size} runs)", y=1.02)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

