"""
BioFusionFormer - Pathway Selection Module
===========================================
Unsupervised pathway filtering based on Reactome hierarchy and feature variance.
Filters pathways to:
  - 10 <= member features <= 300
  - depth <= 4 in the Reactome hierarchy
  - Top 150-250 pathways ranked by cohort expression variance
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp


def select_unsupervised_pathways(
    data_dir: Path,
    top_k: int = 192,
    min_members: int = 10,
    max_members: int = 300,
    max_depth: int = 4,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Selects top pathways satisfying size and depth constraints, sorted by variance.
    Returns list of pathway IDs and summary DataFrame.
    """
    data_dir = Path(data_dir)
    nodes_csv = data_dir / "interim" / "reactome" / "pathway_nodes.csv"
    nodes = pd.read_csv(nodes_csv).set_index("id")

    mask_gene_npz = data_dir / "interim" / "priors" / "gene_pathway_mask.npz"
    mask_gene = sp.load_npz(mask_gene_npz)
    pways_json = data_dir / "interim" / "priors" / "gene_pathway_mask_pathways.json"
    with open(pways_json) as f:
        all_pways = json.load(f)

    n_genes_per_pway = np.array(mask_gene.sum(axis=0)).flatten()
    pway_gene_counts = pd.Series(n_genes_per_pway, index=all_pways)

    df = pd.DataFrame({"n_genes": pway_gene_counts}).join(nodes[["depth", "name"]])
    filtered = df[(df["depth"] <= max_depth) & (df["n_genes"] >= min_members) & (df["n_genes"] <= max_members)]

    # Compute variance from pathway features or mRNA features
    pway_feats_path = data_dir / "processed" / "pathway_features_fold_agnostic.parquet"
    pway_feats = pd.read_parquet(pway_feats_path)

    pway_vars = {}
    for p in filtered.index:
        col = f"mrna_pway_{p}"
        if col in pway_feats.columns:
            pway_vars[p] = pway_feats[col].var()

    s_vars = pd.Series(pway_vars).sort_values(ascending=False)
    selected_pways = s_vars.head(top_k).index.tolist()
    summary_df = filtered.loc[selected_pways].copy()
    summary_df["variance"] = s_vars[selected_pways]

    return selected_pways, summary_df
