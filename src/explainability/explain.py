"""
BioFusionFormer - Explainability Module
=======================================
Provides biological explainability tools:
  1. Modality Gate Weight extraction per patient and stratified by class
  2. Attention Rollout from [CLS] token to pathway tokens
  3. Captum Integrated Gradients at the pathway-token level
  4. Stability analysis across cross-validation folds (pairwise Jaccard and recurrent pathways)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def compute_attention_rollout(attns: List[torch.Tensor]) -> torch.Tensor:
    """
    Computes attention rollout across L transformer layers.
    attns: list of (B, H, S, S) tensors
    Returns: (B, S, S) rollout matrix where rollout[b, 0, 1:] is attention from [CLS] to pathway tokens.
    """
    # Average across attention heads: (B, S, S)
    mean_attns = [a.mean(dim=1) for a in attns]
    B, S, _ = mean_attns[0].shape
    eye = torch.eye(S, device=mean_attns[0].device).unsqueeze(0).expand(B, -1, -1)

    # Rollout formula: R_l = 0.5 * (A_l + I) @ R_{l-1}
    rollout = eye
    for a in mean_attns:
        a_norm = 0.5 * (a + eye)
        rollout = torch.bmm(a_norm, rollout)
    return rollout


def extract_gate_weights(model: nn.Module, batch_inputs: Dict[str, torch.Tensor]) -> np.ndarray:
    """
    Returns (B, M) array of modality gate weights.
    """
    model.eval()
    with torch.no_grad():
        out = model(batch_inputs)
        return out["gate"].cpu().numpy()


def compute_pathway_jaccard(top_pathway_lists: List[List[str]]) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Computes mean pairwise Jaccard similarity across folds and identifies pathways
    appearing in >= 80% of folds.
    """
    n_folds = len(top_pathway_lists)
    jaccards = []
    for i in range(n_folds):
        set_i = set(top_pathway_lists[i])
        for j in range(i + 1, n_folds):
            set_j = set(top_pathway_lists[j])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            if union > 0:
                jaccards.append(inter / union)

    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    # Frequency across folds
    from collections import Counter
    counts = Counter()
    for p_list in top_pathway_lists:
        for p in set(p_list):
            counts[p] += 1

    recurrent = [(p, count / n_folds) for p, count in counts.most_common() if count / n_folds >= 0.80]
    return mean_jaccard, recurrent
