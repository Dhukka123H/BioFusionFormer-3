"""
src/pathways/bio_bias.py
========================
Unified Biological Attention Prior Matrix Construction.

Canonical implementation from scripts/run_e1.py (CV benchmark run).
"""

from typing import List, Optional
import numpy as np
import scipy.sparse as sp
import torch


def build_bio_bias_matrix(
    pathway_indices: List[int],
    n_modalities: int = 3,
    adj_sparse: Optional[sp.spmatrix] = None,
    same_pathway_bias: float = 1.0,
    parent_child_bias: float = 0.5,
    cls_bias: float = 0.5,
) -> torch.Tensor:
    """
    Constructs the (S x S) biological attention prior bias matrix for BioFusionFormer
    where S = 1 + n_modalities * n_pathways.

    Canonical algorithm (from scripts/run_e1.py):
      1. [CLS] token has bidirectional prior bias `cls_bias` (0.5) to all pathway tokens; B[0, 0] = 0.0.
      2. Subgraph adjacency between selected pathways is symmetrized:
         sym_adj = ((sub_adj + sub_adj.T) > 0).astype(float)
      3. For all pairs of modalities (m1, m2):
         - If m1 != m2, tokens corresponding to the identical pathway receive `same_pathway_bias` (1.0).
         - Across all modality blocks (m1, m2), parent-child hierarchical edges receive `parent_child_bias` (0.5).
      4. Self-attention diagonal is set to 0.0 via np.fill_diagonal(B, 0.0).

    Args:
        pathway_indices: list of indices into full pathway graph
        n_modalities: number of input modalities (default 3: mRNA, miRNA, meth)
        adj_sparse: sparse pathway adjacency matrix (P_full x P_full)
        same_pathway_bias: cross-modality bias for the identical pathway (default 1.0)
        parent_child_bias: hierarchical DAG graph prior bias (default 0.5)
        cls_bias: [CLS] token attention prior bias (default 0.5)

    Returns:
        torch.Tensor of shape (1 + n_modalities * P, 1 + n_modalities * P) with dtype torch.float32.
    """
    P = len(pathway_indices)
    M = n_modalities
    S = 1 + M * P
    B = np.zeros((S, S), dtype=np.float32)

    # 1. [CLS] bi-directional attention
    B[0, :] = cls_bias
    B[:, 0] = cls_bias
    B[0, 0] = 0.0

    # Subgraph adjacency between selected pathways
    if adj_sparse is not None:
        sub_adj = adj_sparse[pathway_indices, :][:, pathway_indices].toarray()
        sym_adj = ((sub_adj + sub_adj.T) > 0).astype(np.float32)
    else:
        sym_adj = np.zeros((P, P), dtype=np.float32)

    for m1 in range(M):
        for m2 in range(M):
            offset1 = 1 + m1 * P
            offset2 = 1 + m2 * P
            if m1 != m2:
                for p in range(P):
                    B[offset1 + p, offset2 + p] += same_pathway_bias
            B[offset1 : offset1 + P, offset2 : offset2 + P] += sym_adj * parent_child_bias

    np.fill_diagonal(B, 0.0)
    return torch.tensor(B, dtype=torch.float32)
