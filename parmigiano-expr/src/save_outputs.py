import sys
import os
import pandas as pd
import numpy as np
import torch
import yaml
import pickle

def simulations_to_samples(simulated_parameters, gene_names):
    """
    Convert simulate_expression output into beta_samples/mu_samples dicts
    compatible with save_results format (each value is array of shape [1, num_variants]).
    """
    beta_samples_sim = {}
    mu_samples_sim = {}
    for gene_name in gene_names:
        if gene_name not in simulated_parameters:
            continue
        beta = simulated_parameters[gene_name]['beta']
        mu   = simulated_parameters[gene_name]['mu']
        # Add a sample dimension so mean/std calls in save_results work (shape: [1, num_variants])
        beta_samples_sim[gene_name] = beta.detach().cpu().numpy()[None, :]
        mu_samples_sim[gene_name]   = mu.detach().cpu().numpy()[None, :]
    return beta_samples_sim, mu_samples_sim
    
def save_results(losses, times, posterior_stats, config, annotations, 
                 data = None,
                 simulations=None,
                 beta_samples=None, mu_samples=None,
                 train_r2=None, test_r2=None,
                 tau_history=None,
                 variant_ids_G=None, variant_ids_Z=None,
                 gene_indices=None,
                 save_full_samples=False):

    os.makedirs(config['output_dir'], exist_ok=True)

    # --- Basic training diagnostics ---
    np.savetxt(os.path.join(config['output_dir'], "losses.txt"), losses)
    np.savetxt(os.path.join(config['output_dir'], "times.txt"), times)

    # --- Tau and threshold summary ---
    try:
        tau_df = pd.DataFrame()
        tau_df['Annotation'] = annotations
        tau_df['Tau'] = posterior_stats['tau']['mean'][0]
        tau_df['Filter Threshold'] = posterior_stats['threshold']['mean'][0]
        tau_df.to_csv(os.path.join(config['output_dir'], "tau_T.csv"), index=False)
    except Exception as e:
        print(f"Could not save tau_T.csv: {e}")

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
                    df.to_csv(os.path.join(config['output_dir'], f"{param}.csv"), index=False)
    
        if scalar_stats:
            pd.DataFrame([scalar_stats]).to_csv(
                os.path.join(config['output_dir'], "w_g_rho_g.csv"), index=False
            )
    except Exception as e:
        print(f"Could not save w_g_rho_g.csv: {e}")

    # --- Tau history across epochs ---
    if tau_history is not None:
        valid_tau_history = [tau for tau in tau_history if tau is not None]
        if len(valid_tau_history) > 0:
            tau_history_array = np.array(valid_tau_history)
            tau_history_df = pd.DataFrame(tau_history_array, columns=annotations)
            tau_history_df.insert(0, 'epoch', range(len(valid_tau_history)))
            tau_history_df.to_csv(os.path.join(config['output_dir'], 'tau_history.csv'), index=False)

    # --- Simulated parameters ---
    beta_samples_sim, mu_samples_sim = {}, {}
    if simulations is not None:
        beta_samples_sim, mu_samples_sim = simulations_to_samples(simulations, data.gene_names)
        torch.save(simulations, os.path.join(config['output_dir'], 'simulations.pt'))
        
    # --- Beta samples: CSV per gene with variant IDs (saved in output_dir directly) ---
    if beta_samples is not None:
        os.makedirs(os.path.join(config['output_dir'], 'beta_samples'), exist_ok=True)
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
            if gene_name in beta_samples_sim:
                beta_sim = np.atleast_1d(beta_samples_sim[gene_name].flatten())
                beta_df['beta_simulated'] = beta_sim

            safe_gene_name = gene_name.replace('/', '_')
            beta_df.to_csv(os.path.join(config['output_dir'], 'beta_samples', f'{safe_gene_name}_beta.csv.gz'),
                           index=False, compression="gzip")

        if save_full_samples:
            np.savez_compressed(os.path.join(config['output_dir'], 'beta_samples', 'all_beta_samples.npz'), **beta_samples)

    # --- Mu samples: CSV per gene with variant IDs (saved in output_dir directly) ---
    if mu_samples is not None:
        os.makedirs(os.path.join(config['output_dir'], 'mu_samples'), exist_ok=True)
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
            if gene_name in mu_samples_sim:
                mu_sim = np.atleast_1d(mu_samples_sim[gene_name].flatten())
                mu_df['mu_simulated'] = mu_sim

            safe_gene_name = gene_name.replace('/', '_')
            mu_df.to_csv(os.path.join(config['output_dir'], 'mu_samples', f'{safe_gene_name}_mu.csv.gz'),
                         index=False, compression="gzip")

        if save_full_samples:
            np.savez_compressed(os.path.join(config['output_dir'], 'mu_samples', 'all_mu_samples.npz'), **mu_samples)

    # --- R2 scores ---
    if train_r2 is not None:
        r2_df = pd.DataFrame({'gene': list(train_r2.keys()), 'r2': list(train_r2.values())})
        r2_df.to_csv(os.path.join(config['output_dir'], 'train_r2_scores.csv'), index=False)

    if test_r2 is not None:
        r2_df = pd.DataFrame({'gene': list(test_r2.keys()), 'r2': list(test_r2.values())})
        r2_df.to_csv(os.path.join(config['output_dir'], 'test_r2_scores.csv'), index=False)
    np.savez(os.path.join(config['output_dir'], 'posterior_stats.npz'), **posterior_stats)
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