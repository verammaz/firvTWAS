###### This script runs genome-wide gruyere ######

from datetime import datetime
import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, AutoMultivariateNormal, AutoNormal
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.infer import Predictive
from torch.distributions import constraints
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os, sys
from dataclasses import dataclass, field
import yaml

# Loading other scripts
import utils
import load_data
import data_class
import models
import performance



def fit(data, params):
    '''
    Fit gruyere model jointly
    '''
    
    model = models.gruyere()
    guide = AutoGuideList(model)
    to_optimize = ["tau"] # can have "tau" here if want point estimates (Delta guide)
    guide.add(AutoNormal(poutine.block(model, hide = to_optimize)))
    guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))
    adam = pyro.optim.Adam({"lr": params['lr']})
    svi = SVI(model, guide = guide, optim = adam, loss=Trace_ELBO()) 
    pyro.clear_param_store()
    losses = []
    taus = []
    for j in tqdm(range(params['epochs'])): 
        if j > 0: 
            try:
                current_tau = pyro.param("tau").detach().cpu().numpy().copy()
            except KeyError:
                # sometimes pyro stores parameter names with prefixes (e.g., guide.tau)
                current_tau = {name: pyro.param(name).detach().cpu().numpy().copy()
                            for name in pyro.get_param_store().keys() if "tau" in name}
            taus.append(current_tau['AutoGuideList.1.tau'])
        loss = svi.step(data, params)
        losses.append(float(loss))
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=params.get('num_samples', 50)) 
    samples = predictive(data, params)
    posterior_stats = {k:{'mean':np.array(torch.mean(v, 0)[0]),
                         'std': np.array(torch.std(v,0)[0])} for k,v in samples.items() if "obs" not in k}
    print("Finished training")
    return posterior_stats, losses, taus


def write_outputs(posterior_stats, losses, taus, params, data, train_perf, test_perf):  
    if not os.path.exists(params['output']):
        os.mkdir(params['output'])
    params['output'] = os.path.join(params['output'], 'joint_model')
    if not os.path.exists(params['output']):
        os.mkdir(params['output'])
    np.savetxt(os.path.join(params['output'], "losses.txt"), losses)
    df = pd.DataFrame.from_dict(posterior_stats['tau'], orient = 'index')
    df.columns = data.annotations
    df.T.to_csv(os.path.join(params['output'], "final_tau.csv"))
    
    alphas = {key.split('_')[1]: value['mean'] for key, value in posterior_stats.items() if key.startswith('alpha_')}
    alphas = pd.DataFrame(alphas)
    alphas.index = data.covariates
    alphas.to_csv(os.path.join(params['output'], 'alpha.csv')) 
    
    df = pd.DataFrame.from_dict(posterior_stats['w_g'], orient = 'index')
    df.columns = data.genes
    df = df.T
    df.columns = ['wg', 'wg_std']
    df.to_csv(os.path.join(params['output'], "wg.csv"))

    df = pd.DataFrame(taus, columns=data.annotations)
    df.to_csv(os.path.join(params['output'], 'taus.csv'), index=False)


    pd.DataFrame([train_perf]).to_csv(os.path.join(params['output'], 'train_performance.csv'))
    if test_perf is not None:
        pd.DataFrame([test_perf]).to_csv(os.path.join(params['output'], 'test_performance.csv'))
    
    return
    
    
def run_gruyere(params):
    start_time = datetime.now()
    print("Loading data...", end="", flush=True)

    X = load_data.load_covariates(params)
    Y = load_data.load_phenotypes(params)
    Gs, Zs = load_data.load_genotypes_annotations(params)
    data = data_class.GenomeWide.from_pandas(Gs, Zs, X, Y, params)
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Done ({elapsed})")


    print("\tgenes: ", data.genes)
    print("\tY train shape: ", data.Y['train'].shape)
    print("\tX train shape: ", data.X['train'].shape)
    print("\tG train shape: ", data.G['train'].shape)
    print("\tZ shape: ", data.Z.size())
    print("\n")

    start_time = datetime.now()
    print("Fitting joint model ...", end="", flush=True)
    posterior_stats, losses, taus = fit(data, params)
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Done ({elapsed})\n")

    with open(os.path.join(params['output'], 'joint_model', 'fit_time.txt'), 'w') as f:
        #num_genes  fit_time (seconds)
        f.write(f"{len(data.genes)}\t{round(elapsed.total_seconds(), 2)}")
        
    start_time = datetime.now()
    print("Evaluating model performance...", end="", flush=True)
    train_perf = performance.predict_joint(data, params, posterior_stats, 'train')
    if params['test_prop'] != 0:
        test_perf = performance.predict_joint(data, params, posterior_stats, 'test')
    else: test_perf = None
    write_outputs(posterior_stats, losses, taus, params, data, train_perf, test_perf)
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Done ({elapsed})\n")
    

    return
    
    
if __name__ == "__main__":
    # Load YAML input arguments
    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)  
    run_gruyere(params)