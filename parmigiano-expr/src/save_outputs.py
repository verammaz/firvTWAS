import sys, os
import pandas as pd
import numpy as np
import torch
import yaml
import pickle


def save_results(losses, times, posterior_stats, config, annotations, 
                    beta_samples=None, mu_samples=None, 
                    train_r2=None, test_r2=None, tau_history=None, 
                    variant_ids_G=None, variant_ids_Z=None, 
                    gene_indices=None):
    '''
    Save learned parmigiano parameters and statistics.
    '''
    os.makedirs(config['output_dir'], exist_ok=True)
    np.savetxt(os.path.join(config['output_dir'], "losses.txt"), losses)
    np.savetxt(os.path.join(config['output_dir'], "times.txt"), times)
    tau_df = pd.DataFrame()
    tau_df['Annotation'] = annotations
    tau_df['Tau'] = posterior_stats['tau']['mean'][0]
    tau_df['Filter Threshold'] = posterior_stats['threshold']['mean'][0]
    tau_df.to_csv(os.path.join(config['output_dir'], "tau_T.csv"), index = False)
    np.savez(os.path.join(config['output_dir'], 'posterior_stats.npz'), **posterior_stats)
    
   
    # Save beta samples if provided
    if beta_samples is not None:
        beta_dir = os.path.join(config['output_dir'], 'beta_samples')
        os.makedirs(beta_dir, exist_ok=True)
        for gene_name, beta_gene in beta_samples.items():
            # Save mean and std of beta
            beta_mean = beta_gene.mean(axis=0)
            beta_std = beta_gene.std(axis=0)
            
           # Get variant IDs for this gene
            variant_ids_gene_G = None
            variant_ids_gene_Z = None
            if variant_ids_G is not None and gene_indices is not None and gene_name in gene_indices:
                start_idx, end_idx = gene_indices[gene_name]
                variant_ids_gene_G = variant_ids_G[start_idx:end_idx]
            if variant_ids_Z is not None and gene_indices is not None and gene_name in gene_indices:
                start_idx, end_idx = gene_indices[gene_name]
                variant_ids_gene_Z = variant_ids_Z[start_idx:end_idx]
            
            # Create DataFrame with variant IDs if available
            if variant_ids_gene_G is not None and variant_ids_gene_Z is not None:
                beta_df = pd.DataFrame({
                    'variant_id_G': variant_ids_gene_G,
                    'variant_id_Z': variant_ids_gene_Z,
                    'beta_mean': beta_mean,
                    'beta_std': beta_std
                })
            else:
                # Fallback if variant IDs not available
                beta_df = pd.DataFrame({
                    'beta_mean': beta_mean,
                    'beta_std': beta_std
                })
            
            # Clean gene name for filename
            safe_gene_name = gene_name.replace('/', '_')
            beta_df.to_csv(os.path.join(beta_dir, f'{safe_gene_name}_beta.csv'), index=False)
        # Also save all samples as npz
        np.savez(os.path.join(beta_dir, 'all_beta_samples.npz'), **beta_samples)
    
    # Save mu samples if provided
    if mu_samples is not None:
        mu_dir = os.path.join(config['output_dir'], 'mu_samples')
        os.makedirs(mu_dir, exist_ok=True)
        for gene_name, mu_gene in mu_samples.items():
            # Save mean and std of mu
            mu_mean = mu_gene.mean(axis=0)
            mu_std = mu_gene.std(axis=0)
            # Ensure arrays are 1D (handle scalar case)
            mu_mean = np.atleast_1d(mu_mean)
            mu_std = np.atleast_1d(mu_std)
            
            # Get variant IDs for this gene
            variant_ids_gene_G = None
            variant_ids_gene_Z = None
            if variant_ids_G is not None and gene_indices is not None and gene_name in gene_indices:
                start_idx, end_idx = gene_indices[gene_name]
                variant_ids_gene_G = variant_ids_G[start_idx:end_idx]
            if variant_ids_Z is not None and gene_indices is not None and gene_name in gene_indices:
                start_idx, end_idx = gene_indices[gene_name]
                variant_ids_gene_Z = variant_ids_Z[start_idx:end_idx]
            
            # Create DataFrame with variant IDs if available
            if variant_ids_gene_G is not None and variant_ids_gene_Z is not None:
                mu_df = pd.DataFrame({
                    'variant_id_G': variant_ids_gene_G,
                    'variant_id_Z': variant_ids_gene_Z,
                    'mu_mean': mu_mean,
                    'mu_std': mu_std
                })
            else:
                # Fallback if variant IDs not available
                mu_df = pd.DataFrame({
                    'mu_mean': mu_mean,
                    'mu_std': mu_std
                })
            
            # Clean gene name for filename
            safe_gene_name = gene_name.replace('/', '_')
            mu_df.to_csv(os.path.join(mu_dir, f'{safe_gene_name}_mu.csv'), index=False)
        # Also save all samples as npz
        np.savez(os.path.join(mu_dir, 'all_mu_samples.npz'), **mu_samples)
    
    # Save train R2 scores if provided
    if train_r2 is not None:
        r2_df = pd.DataFrame({
            'gene': list(train_r2.keys()),
            'r2': list(train_r2.values())
        })
        r2_df.to_csv(os.path.join(config['output_dir'], 'train_r2_scores.csv'), index=False)
    
    # Save test R2 scores if provided
    if test_r2 is not None:
        r2_df = pd.DataFrame({
            'gene': list(test_r2.keys()),
            'r2': list(test_r2.values())
        })
        r2_df.to_csv(os.path.join(config['output_dir'], 'test_r2_scores.csv'), index=False)
    
    # Save tau history across epochs if provided
    if tau_history is not None:
        # Filter out None values (epochs where tau wasn't accessible)
        valid_tau_history = [tau for tau in tau_history if tau is not None]
        if len(valid_tau_history) > 0:
            # Create DataFrame with epochs as rows and annotations as columns
            tau_history_array = np.array(valid_tau_history)  # Shape: (n_epochs, n_annotations)
            tau_history_df = pd.DataFrame(
                tau_history_array,
                columns=annotations
            )
            tau_history_df.insert(0, 'epoch', range(len(valid_tau_history)))
            tau_history_df.to_csv(os.path.join(config['output_dir'], 'tau_history.csv'), index=False)
    
    return



    