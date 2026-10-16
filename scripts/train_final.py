#!/usr/bin/env python3
"""
scripts/train_final.py
======================
Final model training and artifact generator for BioFusionFormer (Stage 6.1).

Trains the frozen-config E1 on all 121 patients with 3 random seeds (42, 123, 999).
Saves to models/final/ (git-ignored):
  - model weights per seed (model_seed42.pt, model_seed123.pt, model_seed999.pt)
  - fitted SimpleImputer and StandardScaler transformers (preprocessors.pt)
  - feature name lists (feature_names.json)
  - selected pathways list (pathways.json)
  - biological masks (masks.pt) and attention bias matrix (bio_bias.pt)
  - frozen configuration copy (config.yaml)
  - small manifest (manifest.json) containing SHA256 hashes, git commit hash, seeds, and metadata.
"""

import datetime
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pathways.selection import select_unsupervised_pathways
from src.biofusionformer.full_model import BioFusionFormerModel
from src.heads.classification import LabelSmoothedBCEWithLogitsLoss
from src.explainability.explain import compute_attention_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_final")
torch.set_num_threads(2)

DATA_DIR = ROOT / "data"
MODELS_FINAL_DIR = ROOT / "models" / "final"
CONFIG_PATH = ROOT / "configs" / "default.yaml"
SEEDS = [42, 123, 999]
HP = dict(lr=3e-4, weight_decay=1e-2, max_epochs=100, patience=20, batch_size=16)


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_bio_bias_matrix(pathway_indices, n_modalities=3, adj_sparse=None):
    P = len(pathway_indices)
    S = 1 + n_modalities * P
    bias = np.zeros((S, S), dtype=np.float32)

    # CLS token bi-directional bias
    bias[0, :] = 0.5
    bias[:, 0] = 0.5
    bias[0, 0] = 0.0

    if adj_sparse is not None:
        sub_adj = adj_sparse[pathway_indices, :][:, pathway_indices].toarray()
        sub_adj = (sub_adj + sub_adj.T > 0).astype(np.float32) * 0.5
    else:
        sub_adj = np.zeros((P, P), dtype=np.float32)

    for m in range(n_modalities):
        s_m = 1 + m * P
        e_m = 1 + (m + 1) * P
        bias[s_m:e_m, s_m:e_m] += sub_adj

    for m1 in range(n_modalities):
        for m2 in range(m1 + 1, n_modalities):
            s1 = 1 + m1 * P
            s2 = 1 + m2 * P
            for p in range(P):
                bias[s1 + p, s2 + p] += 1.0
                bias[s2 + p, s1 + p] += 1.0

    return torch.tensor(bias, dtype=torch.float32)


def main():
    MODELS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Starting Final Model Training on all 121 cohort patients...")

    # 1. Load data
    labels_df = pd.read_csv(DATA_DIR / "processed" / "labels.csv")
    y_all = labels_df["histology_label"].values
    pos_weight = float((y_all == 0).sum() / (y_all == 1).sum())

    mrna_df = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet")
    mirna_df = pd.read_parquet(DATA_DIR / "processed" / "mirna_unique.parquet")
    meth_df = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet")

    feature_names = {
        "mrna": mrna_df.columns.tolist(),
        "mirna": mirna_df.columns.tolist(),
        "meth": meth_df.columns.tolist(),
    }

    # 2. Fit imputers and scalers on all 121 patients
    preprocessors = {}
    X_scaled = {}
    for mod, df in [("mrna", mrna_df), ("mirna", mirna_df), ("meth", meth_df)]:
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        imp_vals = imp.fit_transform(df.values)
        scaled_vals = sc.fit_transform(imp_vals)
        preprocessors[mod] = {"imputer": imp, "scaler": sc}
        X_scaled[mod] = torch.tensor(scaled_vals, dtype=torch.float32)

    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    # 3. Pathways and masks
    selected_pways, pway_summary = select_unsupervised_pathways(DATA_DIR, top_k=192)
    P = len(selected_pways)
    pri = DATA_DIR / "interim" / "priors"

    prior_pathways = json.load(open(pri / "gene_pathway_mask_pathways.json"))
    pway_map = {p: i for i, p in enumerate(prior_pathways)}
    pway_idx = [pway_map[p] for p in selected_pways]

    mask_gene_full = sp.load_npz(pri / "gene_pathway_mask.npz")
    gene_names = json.load(open(pri / "gene_pathway_mask_genes.json"))
    gene_map = {g: i for i, g in enumerate(gene_names)}
    mask_mrna = mask_gene_full[[gene_map[c] for c in mrna_df.columns], :][:, pway_idx].T.toarray()
    mask_meth = mask_gene_full[[gene_map[c] for c in meth_df.columns], :][:, pway_idx].T.toarray()

    sub_mask_strong = sp.load_npz(pri / "mirna_unique_pathway_mask_strong.npz")[:, pway_idx].toarray()
    zero_strong = sub_mask_strong.sum(axis=1) == 0
    sub_mask_strong[zero_strong, P - 1] = True
    mask_mirna_strong = sub_mask_strong.T

    masks = {
        "mrna": torch.tensor(mask_mrna, dtype=torch.bool),
        "mirna": torch.tensor(mask_mirna_strong, dtype=torch.bool),
        "meth": torch.tensor(mask_meth, dtype=torch.bool),
    }

    adj_full = sp.load_npz(pri / "pathway_adjacency.npz")
    bio_bias = build_bio_bias_matrix(pway_idx, n_modalities=3, adj_sparse=adj_full)

    feat_dims = {"mrna": mrna_df.shape[1], "mirna": mirna_df.shape[1], "meth": meth_df.shape[1]}

    # 4. Train with 3 random seeds
    trained_models = {}
    N = len(y_all)
    batch_size = HP["batch_size"]
    n_batches = int(math.ceil(N / batch_size))

    for seed in SEEDS:
        log.info(f"Training final model with seed {seed}...")
        set_seed(seed)
        model = BioFusionFormerModel(
            modality_features=feat_dims,
            modality_masks=masks,
            n_pathways=P,
            d_model=32, n_heads=4, n_layers=2, d_ff=128,
            dropout=0.3, attn_dropout=0.2, modality_dropout=0.2,
            use_pathway_bias=True, use_gating=True, bio_bias=bio_bias,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP["max_epochs"], eta_min=HP["lr"] * 0.05)
        criterion = LabelSmoothedBCEWithLogitsLoss(eps=0.05, pos_weight=pos_weight)

        for epoch in range(HP["max_epochs"]):
            model.train()
            perm = torch.randperm(N)
            for b in range(n_batches):
                b_idx = perm[b * batch_size : (b + 1) * batch_size]
                b_x = {m: X_scaled[m][b_idx] for m in X_scaled}
                b_y = y_tensor[b_idx]

                optimizer.zero_grad()
                out = model(b_x)
                loss = criterion(out["logit"], b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        # Save weights
        pt_path = MODELS_FINAL_DIR / f"model_seed{seed}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, pt_path)
        log.info(f"Saved {pt_path}")

    # 5. Save all metadata and artifacts
    torch.save(preprocessors, MODELS_FINAL_DIR / "preprocessors.pt")
    torch.save(masks, MODELS_FINAL_DIR / "masks.pt")
    torch.save(bio_bias, MODELS_FINAL_DIR / "bio_bias.pt")

    with open(MODELS_FINAL_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(MODELS_FINAL_DIR / "pathways.json", "w", encoding="utf-8") as f:
        json.dump(selected_pways, f, indent=2)

    import shutil
    shutil.copy(CONFIG_PATH, MODELS_FINAL_DIR / "config.yaml")

    # 6. Build and write manifest.json
    commit_hash = get_git_commit_hash()
    files_to_hash = [
        "model_seed42.pt", "model_seed123.pt", "model_seed999.pt",
        "preprocessors.pt", "masks.pt", "bio_bias.pt",
        "feature_names.json", "pathways.json", "config.yaml",
    ]
    hashes = {}
    for fname in files_to_hash:
        fpath = MODELS_FINAL_DIR / fname
        if fpath.exists():
            hashes[fname] = sha256_file(fpath)

    manifest = {
        "model_name": "BioFusionFormer_E1_Final",
        "cohort": "TARGET-WT (N=121)",
        "task": "wilms_tumor_histologic_subtype",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": commit_hash,
        "seeds": SEEDS,
        "n_pathways": P,
        "feature_counts": {k: len(v) for k, v in feature_names.items()},
        "file_hashes_sha256": hashes,
    }

    manifest_path = MODELS_FINAL_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Final model artifacts and manifest saved to {MODELS_FINAL_DIR}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
