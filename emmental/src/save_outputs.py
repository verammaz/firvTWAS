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
    
def save_results(output_dir, losses, times, posterior_stats, annotations, 
                 data = None,
                 beta_samples=None, 
                 mu_samples=None,
                 mean_samples=None,
                 train_r2=None, test_r2=None,
                 tau_history=None,
                 variant_ids_G=None, variant_ids_Z=None,
                 gene_indices=None,
                 save_full_samples=False):

    os.makedirs(output_dir, exist_ok=True)

    # --- Basic training diagnostics ---
    np.savetxt(os.path.join(output_dir, "losses.txt"), losses)
    np.savetxt(os.path.join(output_dir, "times.txt"), times)

    # --- Tau and threshold summary ---
    try:
        required_tau_keys = {"tau1", "tau2", "threshold"}
        if not required_tau_keys.issubset(set(posterior_stats.keys())):
            raise KeyError("tau1/tau2/threshold missing from posterior_stats")

        tau1_vals = np.asarray(posterior_stats['tau1']['mean'][0]).ravel()
        tau2_vals = np.asarray(posterior_stats['tau2']['mean'][0]).ravel()
        row_labels, has_tau1_intercept = _tau_labels(
            annotations, tau1_vals, tau2_vals
        )

        tau_df = pd.DataFrame()
        tau_df['Annotation'] = row_labels
        tau_df['Tau1'] = tau1_vals
        tau_df['Filter Threshold'] = posterior_stats['threshold']['mean'][0]
        if has_tau1_intercept:
            tau_df['Tau2'] = np.concatenate(([np.nan], tau2_vals))
        else:
            tau_df['Tau2'] = tau2_vals
        tau_df.to_csv(os.path.join(output_dir, "tau_T.csv"), index=False)
    except Exception:
        # Expected for workflows that use fixed tau/T loaded from disk (e.g. per-gene mode).
        pass

    # --- w_g and rho_g ---
    try:
        scalar_stats = {}
        for param in ['w_g', 'rho_g']:
            if param in posterior_stats:
                mean_vals = posterior_stats[param]['mean'].flatten()
                std_vals = posterior_stats[param]['std'].flatten()
                if len(mean_vals) == 1:
                    # Per-gene: single value, save as before
                    scalar_stats[f'{param}_mean'] = float(mean_vals[0])
                    scalar_stats[f'{param}_std'] = float(std_vals[0])
                else:
                    # Joint: one row per gene
                    df = pd.DataFrame({
                        'gene': data.gene_names,
                        f'{param}_mean': mean_vals.tolist(),
                        f'{param}_std': std_vals.tolist(),
                    })
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

    # --- Mean samples ---
    if mean_samples is not None:
        os.makedirs(os.path.join(output_dir, 'mean_samples'), exist_ok=True)
        for gene_name, mean_gene in mean_samples.items():
            mean_mean = np.atleast_1d(mean_gene.mean(axis=0))
            mean_std  = np.atleast_1d(mean_gene.std(axis=0))

            variant_ids_gene_G, variant_ids_gene_Z = _get_variant_ids_for_gene(
                gene_name, gene_indices, variant_ids_G, variant_ids_Z
            )

            if variant_ids_gene_G is not None and variant_ids_gene_Z is not None:
                mean_df = pd.DataFrame({
                    'variant_id_G': variant_ids_gene_G,
                    'variant_id_Z': variant_ids_gene_Z,
                    'mean_mean': mean_mean,
                    'mean_std': mean_std
                })
            else:
                mean_df = pd.DataFrame({'mean_mean': mean_mean, 'mean_std': mean_std})

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