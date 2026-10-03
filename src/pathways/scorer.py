"""
BioFusionFormer - Pathway Scorer
Computes pathway activity scores as mean z-scores of member features,
fitted strictly on training folds to prevent data leakage.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PathwayZScorer(BaseEstimator, TransformerMixin):
    """
    Fits feature-level mean and std strictly on training fold,
    then transforms any matrix by computing pathway scores:
      z_g = (x_g - mu_g) / sigma_g
      score_p = (1 / |G_p|) * sum_{g in G_p} z_g
    """
    def __init__(self, pathway_to_features: dict[str, list[str]], eps=1e-6):
        self.pathway_to_features = pathway_to_features
        self.eps = eps
        self.feature_means_ = {}
        self.feature_stds_ = {}
        self.valid_pathways_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PathwayZScorer expects a pandas DataFrame with feature column names.")

        avail_cols = set(X.columns)
        self.valid_pathways_ = []

        # Find all features that belong to at least one pathway
        needed_features = set()
        for pway, f_list in self.pathway_to_features.items():
            common = [f for f in f_list if f in avail_cols]
            if len(common) >= 5:  # Require at least 5 features present
                self.valid_pathways_.append((pway, common))
                needed_features.update(common)

        # Compute mean and std for each needed feature
        for feat in needed_features:
            vals = X[feat].values
            m = np.nanmean(vals)
            s = np.nanstd(vals)
            if s < self.eps or np.isnan(s):
                s = 1.0
            self.feature_means_[feat] = m
            self.feature_stds_[feat] = s

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PathwayZScorer expects a pandas DataFrame.")

        scores = {}
        for pway, common_features in self.valid_pathways_:
            # Compute standardized z-scores for common features
            z_list = []
            for feat in common_features:
                x = X[feat].values
                m = self.feature_means_.get(feat, 0.0)
                s = self.feature_stds_.get(feat, 1.0)
                z = (x - m) / s
                z_list.append(z)

            # Stack and average across features: shape (n_features, n_samples)
            z_mat = np.array(z_list)
            scores[pway] = np.nanmean(z_mat, axis=0)

        out_df = pd.DataFrame(scores, index=X.index)
        return out_df


def load_pathway_hierarchy_mapping(inter_dir: Path) -> dict:
    """
    Builds and returns pathway hierarchy depth and parent/child mappings
    from Reactome pathway graph.
    """
    nodes_csv = inter_dir / "reactome" / "pathway_nodes.csv"
    edges_csv = inter_dir / "reactome" / "pathway_edges.csv"

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    parents_map = {}
    children_map = {}
    for _, row in edges_df.iterrows():
        p, c = row["parent"], row["child"]
        parents_map.setdefault(c, []).append(p)
        children_map.setdefault(p, []).append(c)

    hierarchy = {}
    for _, row in nodes_df.iterrows():
        pid = row["id"]
        hierarchy[pid] = {
            "name": row["name"],
            "depth": int(row["depth"]),
            "is_root": bool(row["is_root"]),
            "is_leaf": bool(row["is_leaf"]),
            "parents": parents_map.get(pid, []),
            "children": children_map.get(pid, []),
        }

    return hierarchy
