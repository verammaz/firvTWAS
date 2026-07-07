import sys
import os
import pandas as pd
import numpy as np
import torch
import yaml
import pickle

def _tau_labels(annotations, tau1_values, tau2_values):
    """
    Build row labels for nonlinear tau outputs.
    Supports tau1 having an optional intercept while tau2 does not.
    """
    n_anno = len(annotations)
    n_tau1 = len(np.asarray(tau1_values).ravel())
    n_tau2 = len(np.asarray(tau2_values).ravel())

    if n_tau1 == n_anno and n_tau2 == n_anno:
        return annotations, False
    if n_tau1 == n_anno + 1 and n_tau2 == n_anno:
        return ["intercept"] + annotations, True

    raise ValueError(
        f"Unexpected nonlinear tau dimensions: tau1={n_tau1}, tau2={n_tau2}, annotations={n_anno}"
    )


def _posterior_mean_vec(stat_entry) -> np.ndarray:
    """Posterior mean as 1d vector (handles AutoDelta (d,) and sampled (1, d))."""
    return np.asarray(stat_entry["mean"], dtype=float).ravel()


def _posterior_mean_scalar(stat_entry) -> float:
    return float(np.asarray(stat_entry["mean"], dtype=float).ravel()[0])


def wg_rhog_delta_sites(config: dict) -> list:
    """Global sites inferred with AutoDelta when ``wg_rhog_delta_guide`` is set."""
    if not config.get("wg_rhog_delta_guide", False):
        return []
    sites = []
    if not config.get("no_wg", False):
        sites.append("w_g")
    if not config.get("no_rhog", False):
        sites.append("rho_g")
    return sites


def summarize_predictive_tensor(values, *, point_estimate: bool = False) -> dict:
    """
    Reduce predictive samples to mean/std vectors.

    Handles AutoDiagonalNormal ``(n_samples, ...)`` and AutoDelta point maps
    ``(...)`` without a leading sample dimension.
    """
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)

    if arr.ndim == 0:
        mean = np.asarray([float(arr)], dtype=float)
    elif arr.ndim == 1:
        mean = arr.astype(float, copy=False)
    else:
        mean = arr.mean(axis=0).astype(float, copy=False)

    mean = np.atleast_1d(np.asarray(mean, dtype=float))

    if point_estimate:
        std = np.full(mean.shape, np.nan, dtype=float)
    elif arr.ndim <= 1:
        std = np.zeros(mean.shape, dtype=float)
    else:
        std = arr.std(axis=0).astype(float, copy=False)
        std = np.atleast_1d(std)

    return {
        "mean": mean,
        "std": std,
        "point_estimate": bool(point_estimate),
    }


def build_posterior_stats(samples: dict, *, delta_sites=None) -> dict:
    """Summarize all predictive sites; mark AutoDelta globals as point estimates."""
    delta = set(delta_sites or [])
    return {
        k: summarize_predictive_tensor(v, point_estimate=(k in delta))
        for k, v in samples.items()
    }


def write_tau_T_csv(output_dir, posterior_stats, annotations) -> str:
    """
    Write ``tau_T.csv`` from posterior_stats (requires tau1, tau2, threshold).
    Returns path written.
    """
    required = {"tau1", "tau2", "threshold"}
    if not required.issubset(posterior_stats.keys()):
        raise KeyError(f"posterior_stats missing {required - set(posterior_stats.keys())}")

    tau1_vals = _posterior_mean_vec(posterior_stats["tau1"])
    tau2_vals = _posterior_mean_vec(posterior_stats["tau2"])
    threshold = _posterior_mean_scalar(posterior_stats["threshold"])
    row_labels, has_tau1_intercept = _tau_labels(annotations, tau1_vals, tau2_vals)

    tau_df = pd.DataFrame()
    tau_df["Annotation"] = row_labels
    tau_df["Tau1"] = tau1_vals
    tau_df["Filter Threshold"] = threshold
    if has_tau1_intercept:
        tau_df["Tau2"] = np.concatenate(([np.nan], tau2_vals))
    else:
        tau_df["Tau2"] = tau2_vals

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tau_T.csv")
    tau_df.to_csv(out_path, index=False)
    return out_path


def tau_T_from_tau_history_last_row(tau_history_path: str, threshold: float) -> pd.DataFrame:
    """Build tau_T dataframe from final epoch of ``tau_history.csv``."""
    hist = pd.read_csv(tau_history_path)
    if hist.empty:
        raise ValueError(f"Empty tau history: {tau_history_path}")
    last = hist.iloc[-1]
    tau1_cols = [c for c in hist.columns if c.startswith("Tau1_")]
    tau2_cols = [c for c in hist.columns if c.startswith("Tau2_")]
    if not tau1_cols or not tau2_cols:
        raise ValueError(f"No Tau1_/Tau2_ columns in {tau_history_path}")

    tau1_labels = [c[len("Tau1_") :] for c in tau1_cols]
    tau2_labels = [c[len("Tau2_") :] for c in tau2_cols]
    tau1_vals = last[tau1_cols].to_numpy(dtype=float)
    tau2_vals = last[tau2_cols].to_numpy(dtype=float)

    row_labels, has_tau1_intercept = _tau_labels(tau2_labels, tau1_vals, tau2_vals)
    tau_df = pd.DataFrame()
    tau_df["Annotation"] = row_labels
    tau_df["Tau1"] = tau1_vals
    tau_df["Filter Threshold"] = threshold
    if has_tau1_intercept:
        tau_df["Tau2"] = np.concatenate(([np.nan], tau2_vals))
    else:
        tau_df["Tau2"] = tau2_vals
    return tau_df


def save_results(output_dir, losses, times, posterior_stats, annotations, 
                 data = None,
                 beta_samples=None, 
                 mu_samples=None,
                 mean_samples=None,
                 train_r2=None, test_r2=None,
                 tau_history=None,
                 variant_ids_G=None, variant_ids_Z=None,
                 gene_indices=None,
                 mean_sample_ids=None,
                 save_full_samples=False):

    os.makedirs(output_dir, exist_ok=True)

    # --- Basic training diagnostics ---
    np.savetxt(os.path.join(output_dir, "losses.txt"), losses)
    np.savetxt(os.path.join(output_dir, "times.txt"), times)

    # --- Tau and threshold summary ---
    try:
        write_tau_T_csv(output_dir, posterior_stats, annotations)
    except Exception:
        # Expected for workflows that use fixed tau/T loaded from disk (e.g. per-gene mode).
        pass

    # --- w_g and rho_g ---
    try:
        scalar_stats = {}
        for param in ['w_g', 'rho_g']:
            if param not in posterior_stats:
                continue
            entry = posterior_stats[param]
            mean_vals = np.asarray(entry['mean'], dtype=float).ravel()
            std_vals = np.asarray(entry['std'], dtype=float).ravel()
            is_point = bool(entry.get('point_estimate', False))
            if len(mean_vals) == 1:
                # Per-gene: single value, save as before
                scalar_stats[f'{param}_mean'] = float(mean_vals[0])
                scalar_stats[f'{param}_std'] = (
                    np.nan if is_point else float(std_vals[0])
                )
                if is_point:
                    scalar_stats[f'{param}_map'] = float(mean_vals[0])
            else:
                # Joint: one row per gene
                out = {
                    'gene': data.gene_names,
                    f'{param}_mean': mean_vals.tolist(),
                    f'{param}_std': [
                        (np.nan if is_point else float(s)) for s in std_vals
                    ],
                }
                if is_point:
                    out[f'{param}_map'] = mean_vals.tolist()
                df = pd.DataFrame(out)
                df.to_csv(os.path.join(output_dir, f"{param}.csv"), index=False)
    
        if scalar_stats:
            pd.DataFrame([scalar_stats]).to_csv(
                os.path.join(output_dir, "w_g_rho_g.csv"), index=False
            )
    except Exception as e:
        print(f"Could not save w_g and/or rho_g.csv: {e}")

    # --- Tau history across epochs ---
    if tau_history is not None:
        valid_tau_history = [tau for tau in tau_history if tau[0] is not None and tau[1] is not None]
        if len(valid_tau_history) > 0:
            tau1_history_array = np.stack([np.asarray(p[0], dtype=float).ravel() for p in valid_tau_history])
            tau2_history_array = np.stack([np.asarray(p[1], dtype=float).ravel() for p in valid_tau_history])
            n_tau1 = tau1_history_array.shape[1]
            n_tau2 = tau2_history_array.shape[1]

            if n_tau1 == len(annotations) and n_tau2 == len(annotations):
                tau1_labels = annotations
            elif n_tau1 == len(annotations) + 1 and n_tau2 == len(annotations):
                tau1_labels = ["intercept"] + annotations
            else:
                raise ValueError(
                    f"Unexpected tau history dimensions: tau1={n_tau1}, tau2={n_tau2}, annotations={len(annotations)}"
                )

            cols1 = [f"Tau1_{a}" for a in tau1_labels]
            cols2 = [f"Tau2_{a}" for a in annotations]
            tau_history_df = pd.concat(
                [
                    pd.DataFrame(tau1_history_array, columns=cols1),
                    pd.DataFrame(tau2_history_array, columns=cols2),
                ],
                axis=1,
            )
            tau_history_df.insert(0, "epoch", np.arange(len(valid_tau_history), dtype=int))
            tau_history_df.to_csv(os.path.join(output_dir, 'tau_history.csv'), index=False)
      
  
    # --- Beta samples: CSV per gene with variant IDs (saved in output_dir directly) ---
    if beta_samples is not None:
        os.makedirs(os.path.join(output_dir, 'beta_samples'), exist_ok=True)
        for gene_name, beta_gene in beta_samples.items():
            beta_mean = np.atleast_1d(beta_gene.mean(axis=0))
            beta_std  = np.atleast_1d(beta_gene.std(axis=0))

            variant_ids_gene_G, variant_ids_gene_Z = _get_variant_ids_for_gene(
                gene_name, gene_indices, variant_ids_G, variant_ids_Z
            )
            variant_ids_gene_G, variant_ids_gene_Z = _variant_ids_match_length(
                variant_ids_gene_G, variant_ids_gene_Z, beta_mean.size
            )

            if variant_ids_gene_G is not None and variant_ids_gene_Z is not None:
                beta_df = pd.DataFrame({
                    'variant_id_G': variant_ids_gene_G,
                    'variant_id_Z': variant_ids_gene_Z,
                    'beta_mean': beta_mean,
                    'beta_std': beta_std
                })
            else:
                beta_df = pd.DataFrame({'beta_mean': beta_mean, 'beta_std': beta_std})

            safe_gene_name = gene_name.replace('/', '_')
            beta_df.to_csv(os.path.join(output_dir, 'beta_samples', f'{safe_gene_name}_beta.csv.gz'),
                           index=False, compression="gzip")

        if save_full_samples:
            np.savez_compressed(os.path.join(output_dir, 'beta_samples', 'all_beta_samples.npz'), **beta_samples)

    # --- Mu samples: CSV per gene with variant IDs (saved in output_dir directly) ---
    if mu_samples is not None:
        os.makedirs(os.path.join(output_dir, 'mu_samples'), exist_ok=True)
        for gene_name, mu_gene in mu_samples.items():
            mu_mean = np.atleast_1d(mu_gene.mean(axis=0))
            mu_std  = np.atleast_1d(mu_gene.std(axis=0))

            variant_ids_gene_G, variant_ids_gene_Z = _get_variant_ids_for_gene(
                gene_name, gene_indices, variant_ids_G, variant_ids_Z
            )
            variant_ids_gene_G, variant_ids_gene_Z = _variant_ids_match_length(
                variant_ids_gene_G, variant_ids_gene_Z, mu_mean.size
            )

            if variant_ids_gene_G is not None and variant_ids_gene_Z is not None:
                mu_df = pd.DataFrame({
                    'variant_id_G': variant_ids_gene_G,
                    'variant_id_Z': variant_ids_gene_Z,
                    'mu_mean': mu_mean,
                    'mu_std': mu_std
                })
            else:
                mu_df = pd.DataFrame({'mu_mean': mu_mean, 'mu_std': mu_std})

            safe_gene_name = gene_name.replace('/', '_')
            mu_df.to_csv(os.path.join(output_dir, 'mu_samples', f'{safe_gene_name}_mu.csv.gz'),
                         index=False, compression="gzip")

        if save_full_samples:
            np.savez_compressed(os.path.join(output_dir, 'mu_samples', 'all_mu_samples.npz'), **mu_samples)

    # --- Mean samples (per-individual predicted expression, not per-variant) ---
    if mean_samples is not None:
        os.makedirs(os.path.join(output_dir, 'mean_samples'), exist_ok=True)
        for gene_name, mean_gene in mean_samples.items():
            mean_mean = np.atleast_1d(mean_gene.mean(axis=0))
            mean_std  = np.atleast_1d(mean_gene.std(axis=0))

            row_data = {'mean_mean': mean_mean, 'mean_std': mean_std}
            if mean_sample_ids is not None and len(mean_sample_ids) == mean_mean.size:
                row_data = {'sample_id': list(mean_sample_ids), **row_data}
            mean_df = pd.DataFrame(row_data)

            safe_gene_name = gene_name.replace('/', '_')
            mean_df.to_csv(os.path.join(output_dir, 'mean_samples', f'{safe_gene_name}_mean.csv.gz'),
                           index=False, compression="gzip")

        if save_full_samples:
            np.savez_compressed(os.path.join(output_dir, 'mean_samples', 'all_mean_samples.npz'), **mean_samples)

    # --- R2 scores ---
    if train_r2 is not None:
        r2_df = pd.DataFrame({'gene': list(train_r2.keys()), 'r2': list(train_r2.values())})
        r2_df.to_csv(os.path.join(output_dir, 'train_r2_scores.csv'), index=False)

    if test_r2 is not None:
        r2_df = pd.DataFrame({'gene': list(test_r2.keys()), 'r2': list(test_r2.values())})
        r2_df.to_csv(os.path.join(output_dir, 'test_r2_scores.csv'), index=False)
    np.savez(os.path.join(output_dir, 'posterior_stats.npz'), **posterior_stats)
    return


def _get_variant_ids_for_gene(gene_name, gene_indices, variant_ids_G, variant_ids_Z):
    """Helper to slice variant ID lists for a given gene using gene_indices."""
    variant_ids_gene_G = None
    variant_ids_gene_Z = None
    if gene_indices is not None and gene_name in gene_indices:
        start_idx, end_idx = gene_indices[gene_name]
        if variant_ids_G is not None:
            variant_ids_gene_G = variant_ids_G[start_idx:end_idx]
        if variant_ids_Z is not None:
            variant_ids_gene_Z = variant_ids_Z[start_idx:end_idx]
    return variant_ids_gene_G, variant_ids_gene_Z


def _variant_ids_match_length(variant_ids_gene_G, variant_ids_gene_Z, n_expected):
    """Return variant ID slices only when both align with posterior dimension."""
    if variant_ids_gene_G is None or variant_ids_gene_Z is None:
        return None, None
    if len(variant_ids_gene_G) != n_expected or len(variant_ids_gene_Z) != n_expected:
        return None, None
    return variant_ids_gene_G, variant_ids_gene_Z