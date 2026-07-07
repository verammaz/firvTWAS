#!/usr/bin/env python3
"""
Build genome-wide train/test_r2_scores.csv from post-joint full_panel_beta tables.

Uses saved beta_full_common_csv_rare_mu and is_common_maf_ge_threshold per variant.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import load_data  # noqa: E402
import utils  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402


def _load_covariates_scaled(config):
    cov = pd.read_csv(config["covariates_path"], sep="\t").set_index("sample_id")
    covariate_cols = [
        "biological_sex", "eas_prob", "afr_prob", "amr_prob", "sas_prob", "eur_prob",
        "tissue", "age", "pc1", "pc2", "pc3", "pc4", "pc5", "cohort",
        "rna_lib_prep_type", "rna_strandedness", "astrocyte", "endothelial_cell",
        "excitatory_neuron", "inhibitory_neuron", "microglia", "oligodendrocyte",
        "oligodendrocyte_progenitor_cell", "others", "pericyte",
    ]
    return utils.preprocess_covariates(cov, covariate_cols)


class _ExpressionResidualizer:
    def __init__(self, config, cov_scaled, device):
        tpm = pd.read_csv(config["expression_path"], sep="\t").set_index("feature")
        self.tpm = tpm[cov_scaled.index]
        self.cov_scaled = cov_scaled
        self.device = device

    def residualize(self, gene_name: str) -> np.ndarray:
        chr_gene = utils.get_chr_gene(self.tpm, [gene_name])
        feat = chr_gene["feature"].iloc[0]
        expr = utils.scale_tpm_matrix(self.tpm.loc[[feat]]).loc[feat]
        resid_out = utils.residualize_expression_single_gene(
            expr, self.cov_scaled, device=self.device
        )
        if isinstance(resid_out, torch.Tensor):
            idx = expr.index.intersection(self.cov_scaled.index)
            resid = pd.Series(resid_out.detach().cpu().numpy(), index=idx)
        else:
            resid = resid_out
        return resid.reindex(self.cov_scaled.index).to_numpy(dtype=np.float64)


def _r2_predict(G, y, beta, mask):
    if mask.sum() < 1 or np.var(y) < 1e-12:
        return float("nan")
    cols = G.columns[mask]
    b = np.asarray(beta, dtype=np.float64)[mask]
    if not np.isfinite(b).all():
        b = np.nan_to_num(b, nan=0.0)
    pred = G[cols].values @ b
    return float(r2_score(y, pred))


def _train_test_idx(covariates_path: str):
    cov = pd.read_csv(covariates_path, sep="\t").set_index("sample_id")
    train_mask = ~(
        (cov["cohort"] == "ROSMAP")
        & (cov["tissue"] == "Dorsolateral Pre-frontal Cortex (DLPFC)")
    )
    test_mask = (
        (cov["cohort"] == "ROSMAP")
        & (cov["tissue"] == "Dorsolateral Pre-frontal Cortex (DLPFC)")
    )
    return np.where(train_mask)[0], np.where(test_mask)[0]


def _gene_r2_row(
    gene_name: str,
    panel_path: str,
    cfg: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    expr_cache: _ExpressionResidualizer,
) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(panel_path):
        return None

    panel = pd.read_csv(panel_path)
    gcfg = dict(cfg)
    gcfg["genes"] = [gene_name]
    gcfg["maf_threshold"] = None

    G, _, _, _ = load_data.load_genes(gcfg)
    if G.shape[1] == 0 or len(panel) != G.shape[1]:
        return None

    beta = panel["beta_full_common_csv_rare_mu"].to_numpy(dtype=np.float64)
    common_mask = panel["is_common_maf_ge_threshold"].to_numpy(dtype=bool)
    all_mask = np.ones(len(common_mask), dtype=bool)

    y = expr_cache.residualize(gene_name)
    G_tr, G_te = G.iloc[train_idx], G.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    return {
        "gene": gene_name,
        "r2_train_common_only": _r2_predict(G_tr, y_tr, beta, common_mask),
        "r2_test_common_only": _r2_predict(G_te, y_te, beta, common_mask),
        "r2_train_common_plus_rare": _r2_predict(G_tr, y_tr, beta, all_mask),
        "r2_test_common_plus_rare": _r2_predict(G_te, y_te, beta, all_mask),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--postjoint-dir",
        default="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common_01_only/joint/postjoint_rare_beta",
    )
    p.add_argument(
        "--config",
        default="/gpfs/commons/home/vmazeeva/firvTWAS_myout/train_common_01_only/joint/config.yaml",
    )
    p.add_argument("--max-genes", type=int, default=None, help="Limit genes (debug)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if train/test_r2_scores.csv already exist",
    )
    args = p.parse_args()

    postjoint = os.path.abspath(args.postjoint_dir)
    summary_path = os.path.join(postjoint, "summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)

    tr_out = os.path.join(postjoint, "train_r2_scores.csv")
    te_out = os.path.join(postjoint, "test_r2_scores.csv")
    if not args.force and os.path.isfile(tr_out) and os.path.isfile(te_out):
        print(f"Exists: {tr_out} — skip (delete to rebuild)")
        return 0

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["joint_output_dir"] = os.path.dirname(postjoint)

    import logging

    logging.getLogger().setLevel(logging.WARNING)
    utils.get_logger().setLevel(logging.WARNING)

    train_idx, test_idx = _train_test_idx(cfg["covariates_path"])
    expr_cache = _ExpressionResidualizer(cfg, _load_covariates_scaled(cfg), torch.device("cpu"))

    summary = pd.read_csv(summary_path)
    if args.max_genes:
        summary = summary.head(args.max_genes)

    rows: List[Dict[str, Any]] = []
    checkpoint_every = 250

    def _flush_tables() -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        train_df = df[["gene", "r2_train_common_only", "r2_train_common_plus_rare"]].rename(
            columns={
                "r2_train_common_only": "r2_common_only",
                "r2_train_common_plus_rare": "r2_common_plus_rare",
            }
        )
        test_df = df[["gene", "r2_test_common_only", "r2_test_common_plus_rare"]].rename(
            columns={
                "r2_test_common_only": "r2_common_only",
                "r2_test_common_plus_rare": "r2_common_plus_rare",
            }
        )
        train_df.to_csv(tr_out, index=False)
        test_df.to_csv(te_out, index=False)

    for i, (_, row) in enumerate(
        tqdm(summary.iterrows(), total=len(summary), desc="R² from full panel"), start=1
    ):
        gene = row["gene"]
        panel_path = row.get("out_csv") or ""
        try:
            r = _gene_r2_row(gene, panel_path, cfg, train_idx, test_idx, expr_cache)
            if r:
                rows.append(r)
        except Exception as e:
            tqdm.write(f"SKIP {gene}: {e}")
        if i % 50 == 0:
            gc.collect()
        if i % checkpoint_every == 0:
            _flush_tables()

    if not rows:
        print("No genes processed")
        return 1

    _flush_tables()
    print(f"Wrote {len(rows)} genes -> {tr_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
