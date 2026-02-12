import torch
import pyro
from pyro import poutine
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm import tqdm
import time

import utils
import load_data
from save_outputs import save_results
from models import parmigiano_expression


def fit_parmigiano(data, config):
    """
    Fit parmigiano model using stochastic variational inference
    INPUT:
        - data: DataTensors object containing all input data
        - config: configuration dictionary
    OUTPUT:
        - losses: list of loss values per epoch
        - times: list of training times per epoch
        - posterior_stats: dictionary of posterior statistics for each parameter
    """
    
    model = parmigiano_expression() # Initialize model and guide
    to_optimize = ['tau', 'threshold'] # Define paramater-specific guide optimizations
    guide = AutoGuideList(model)
    guide.add(AutoDiagonalNormal(poutine.block(model, hide=to_optimize))) 
    guide.add(AutoDelta(poutine.block(model, expose=to_optimize)))    
    
    adam = pyro.optim.Adam({"lr": config['lr']}) # Setup optimizer and SVI
    svi = SVI(model, guide=guide, optim=adam, loss=Trace_ELBO()) 
    pyro.clear_param_store()
    
    # Training loop
    losses = []
    times = []
    
    print(f"Training for {config['epochs']} epochs...")
    for epoch in tqdm(range(config['epochs'])): 
        start = time.time()
        loss = svi.step(data, config)
        times.append(time.time() - start)
        losses.append(float(loss))
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Generate posterior samples
    print("Generating posterior samples...")
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=config['n_posterior']) 
    samples = predictive(data, config)
    
    # Compute posterior statistics
    posterior_stats = {
        k: {
            'mean': v.mean(0).cpu().numpy(),
            'std': v.std(0).cpu().numpy()
        } 
        for k, v in samples.items()
    }
    
    print("Training complete!")
    return losses, times, posterior_stats


def main():
    """Main execution function"""
    # Load configuration
    args = utils.parse_args()
    yaml_config = None
    if args.config:
        yaml_config = utils.load_yaml(args.config)
    config = utils.fill_defaults(args, yaml_config)
    
    print("=" * 50)
    print("PARMIGIANO - Bayesian Hierarchical Gene Analysis")
    print("=" * 50)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    if isinstance(config['gene_list'], str):  # Path to file
        with open(config['gene_list'], 'r') as f:
            config['genes'] = [line.strip() for line in f if line.strip()]
    else:  # List provided directly
        config['genes'] = config['gene_list']
    # Load data
    print("Loading phenotype and covariate data...")
    X, Y = load_data.load_residualized_covariates(config, device)
    
    print("\nLoading genotype and annotation data...")
    G, Z = load_data.load_genes(config)
    
    # Create data tensors
    print("\nCreating data tensors...")
    data = load_data.DataTensors.from_pandas(G, Z, X, Y, device, config)
    
    print(f"\nData summary:")
    print(f"  Samples: {data.G.shape[0]}")
    print(f"  Variants: {data.G.shape[1]}")
    print(f"  Genes: {data.num_genes}")
    print(f"  Annotations: {data.num_anno}")
    print(f"  Covariates: {data.num_cov}")
    print()
    
    # Fit model
    losses, times, posterior_stats = fit_parmigiano(data, config)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average time per epoch: {sum(times)/len(times):.2f}s")
    print(f"Total training time: {sum(times):.2f}s")
    annotations = list(Z.columns)
    if config.get('output_dir'):
        print(f"\nSaving results to {config['output_dir']}...")
        save_results(
            losses=losses,
            times=times,
            posterior_stats=posterior_stats,
            config=config, 
            annotations=annotations
        )
        print("Results saved!")
    print("\nDone!")


if __name__ == '__main__':
    main()