import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def predict_pergene(data, params, posterior_stats, group):
    """
    Compute per-gene prediction performance (R² and MSE) for continuous expression traits.
    """
    performance = {'R2': {}, 'MSE': {}}

    tau = torch.tensor(posterior_stats['tau']['mean'], dtype=torch.float32)
    w_g = torch.tensor(posterior_stats['w_g']['mean'], dtype=torch.float32)
    num_indivs = data.G[group].shape[0]

    for gene in range(data.num_genes):
        gene_name = data.genes[gene]

        # Coefficients
        alpha = torch.tensor(posterior_stats[f'alpha_{gene_name}']['mean'], dtype=torch.float32)
        beta = (
            (data.Z[data.gene_indices == gene].T * data.maf_weights[data.gene_indices == gene])
            .T.matmul(tau)
        ) * w_g[gene]

        # Predictions
        Gbeta = data.G[group][:, data.gene_indices == gene].matmul(beta)
        preds = torch.matmul(data.X[group], alpha).reshape(-1, 1) + Gbeta.reshape(-1, 1)

        # True and predicted expression
        y_true = data.Y[group][:, gene].detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy().flatten()

        # Compute metrics
        performance['R2'][gene_name] = r2_score(y_true, y_pred)
        performance['MSE'][gene_name] = mean_squared_error(y_true, y_pred)

    return performance


def predict_joint(data, params, posterior_stats, group):
    """
    Compute joint prediction performance across all genes for continuous expression.
    """
    tau = torch.tensor(posterior_stats['tau']['mean'], dtype=torch.float32)
    w_g = torch.tensor(posterior_stats['w_g']['mean'], dtype=torch.float32)
    num_indivs = data.G[group].shape[0]

    preds_all = []

    for gene in range(data.num_genes):
        gene_name = data.genes[gene]

        alpha = torch.tensor(posterior_stats[f'alpha_{gene_name}']['mean'], dtype=torch.float32)
        beta = (
            (data.Z[data.gene_indices == gene].T * data.maf_weights[data.gene_indices == gene])
            .T.matmul(tau)
        ) * w_g[gene]

        Gbeta = data.G[group][:, data.gene_indices == gene].matmul(beta)
        preds = torch.matmul(data.X[group], alpha).reshape(-1, 1) + Gbeta.reshape(-1, 1)
        preds_all.append(preds)

    # Combine predictions across genes
    preds_all = torch.cat(preds_all, dim=1)
    y_true = data.Y[group].detach().cpu().numpy()
    y_pred = preds_all.detach().cpu().numpy()

    # Global metrics across all genes
    joint_r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    joint_mse = mean_squared_error(y_true, y_pred)

    performance = {'R2': joint_r2, 'MSE': joint_mse}
    return performance
