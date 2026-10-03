"""
BioFusionFormer - Pathway-Topology Attention Bias
==================================================
Computes additive attention bias from the Reactome pathway hierarchy.

Two types of structural relationships increase attention weight:
  1. Parent-child (hierarchical): pathway i is a parent of pathway j in Reactome DAG.
  2. Modality-sharing: pathways i and j are both annotated to the same gene set
     across multiple modalities (strong co-regulation signal).

These biases are fixed (non-trainable) — they encode prior biological knowledge
without being fitted to the dataset labels.

Usage
-----
    builder = PathwayBiasBuilder(hierarchy_json, modality_masks)
    bias = builder.build()   # (n_pathways, n_pathways) float tensor
    # Add bias to attention logits before softmax:
    # attn_logits = attn_logits + bias.unsqueeze(0)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


class PathwayBiasBuilder:
    """
    Builds a fixed attention-bias matrix from pathway topology.

    Parameters
    ----------
    hierarchy_path : Path or str
        JSON file mapping pathway ID -> list of parent pathway IDs.
        (Produced by phase_a2_pathway_graph.py)
    pathway_order  : list[str]
        Ordered list of pathway IDs defining the row/column ordering.
        Must match the column ordering of the feature matrices.
    modality_masks : dict[str, torch.BoolTensor], optional
        Dict keyed by modality name, each value is a (n_pathways, n_features)
        mask. If provided, adds a co-annotation bias for pathways that share
        at least one gene in every modality.
    parent_bias    : float
        Added to logit when pathway_j is a direct child of pathway_i.
    coannot_bias   : float
        Added to logit when pathways i and j share >= min_shared genes
        across all modalities.
    min_shared     : int
        Minimum number of shared features to trigger co-annotation bias.
    """

    def __init__(
        self,
        hierarchy_path: Optional[Path],
        pathway_order: List[str],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        parent_bias: float = 1.0,
        coannot_bias: float = 0.5,
        min_shared: int = 3,
    ):
        self.pathway_order = pathway_order
        self.P = len(pathway_order)
        self.idx = {pw: i for i, pw in enumerate(pathway_order)}
        self.hierarchy_path = hierarchy_path
        self.modality_masks = modality_masks
        self.parent_bias = parent_bias
        self.coannot_bias = coannot_bias
        self.min_shared = min_shared

    def _parent_child_bias(self) -> torch.Tensor:
        """Return (P, P) tensor; B[i,j] = parent_bias if i is parent of j."""
        B = torch.zeros(self.P, self.P)
        if self.hierarchy_path is None or not Path(self.hierarchy_path).exists():
            return B
        with open(self.hierarchy_path) as f:
            hierarchy = json.load(f)
        # hierarchy: {child_id: [parent_id, ...]}
        for child, parents in hierarchy.items():
            j = self.idx.get(child)
            if j is None:
                continue
            for parent in parents:
                i = self.idx.get(parent)
                if i is not None:
                    B[i, j] += self.parent_bias
        return B

    def _coannot_bias(self) -> torch.Tensor:
        """Return (P, P) tensor; B[i,j] = coannot_bias if i,j share >= min_shared features."""
        B = torch.zeros(self.P, self.P)
        if self.modality_masks is None:
            return B
        # Sum the shared feature count across all modalities
        shared_total = None
        for name, mask in self.modality_masks.items():
            # mask: (P, F_mod), bool
            mask_f = mask.float()
            # shared[i,j] = number of features annotated to BOTH i and j
            shared = (mask_f @ mask_f.T)  # (P, P)
            if shared_total is None:
                shared_total = shared
            else:
                shared_total = shared_total + shared
        if shared_total is not None:
            B = (shared_total >= self.min_shared).float() * self.coannot_bias
            # Zero diagonal (self-attention not biased)
            B.fill_diagonal_(0.0)
        return B

    def build(self) -> torch.Tensor:
        """
        Returns fixed attention bias of shape (P, P).
        Downstream: add as bias before softmax in attention.
        """
        B = self._parent_child_bias() + self._coannot_bias()
        return B
