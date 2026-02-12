import sys, os
import pandas as pd
import numpy as np
import torch
import yaml
import pickle


def save_results(losses, times, posterior_stats, config, annotations):
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
    return



    