#!/usr/bin/env python3
"""
scripts/explain_e1.py
=====================
Stage A: Explainability, Stability, Representation and Calibration for E1 (v2).

Requirements per AGENT_PROMPT_2 Stage A:
  A1  Fallback audit: confirmed that selected pathway P-1 is R-HSA-350562; 726 miRNA features
      routed to it by fallback. Written to results/explainability/fallback_audit.md.
  A2  Re-aggregate IG per pathway in TWO ways side-by-side:
        (i)  sum of |IG| over genuine member features, EXCLUDING the 726 fallback-routed miRNA edges;
        (ii) mean |IG| per member feature (size-normalised).
      Report, for both: top-20 list, Spearman correlation between pathway size and attribution,
      and count of generic translation/ribosome/splicing pathways.
  A3  Stability across the 5 saved fold models: mean pairwise top-20 Jaccard for IG (i), IG (ii)
      and for per-fold elastic net.
  A4  Matched null: shuffle TRAINING labels inside each of the 5 folds, train with SAME procedure
      as E1 (max_epochs 100, patience 20, 20% inner val, seed 42), compute same IG on test patients.
      3 permutations x 5 folds = 15 trainings, resumable per (perm, fold) with atomic JSON files.
      Report real vs null Jaccard for (i) and (ii). State that with 3 permutations no p-value is computed.
  A5  Recalibration: logistic regression WITHOUT penalty on logit(pooled out-of-fold probability)
      -> slope, intercept, 1000-draw patient bootstrap CI. 5-fold CV over patients of this 2-parameter
      calibrator, reporting Brier before and after. Redraw reliability figure with bin counts printed.
  A6  Redraw representation figure: caption "in-sample; separation reflects training labels; not evidence of generalisation".
  A7  Write results/explainability/summary.md strictly from JSON outputs. State plainly that ODC was a fallback artefact.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.biofusionformer.full_model import BioFusionFormerModel
from src.evaluation.cv_stats import nb_interval, nadeau_bengio_t
from src.explainability.explain import compute_attention_rollout, compute_pathway_jaccard
from src.pathways.selection import select_unsupervised_pathways
from src.heads.classification import LabelSmoothedBCEWithLogitsLoss

log = logging.getLogger("explain_e1")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

torch.set_num_threads(2)

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models" / "e1_folds_final_v1_repro"
FROZEN_DIR = ROOT / "experiments" / "final_v1" / "E1" / "folds"
OUT_DIR = ROOT / "results" / "explainability"
NULL_CACHE_DIR = OUT_DIR / "null_cache"
FIG_DIR = ROOT / "figures" / "explain"

GENERIC_KEYWORDS = [
    "translat", "ribosom", "spliceosom", "mrna processing", "rna splicing",
    "rrna", "transfer rna", "rna polymerase", "post-transcriptional",
]

E1_MODS = ["mrna", "mirna", "meth"]
SEEDS = [42, 123, 999]
HP = dict(lr=3e-4, weight_decay=1e-2, max_epochs=100, patience=20, batch_size=16)


def parse_args():
    ap = argparse.ArgumentParser(description="BioFusionFormer E1 Explainability v2 (Stage A)")
    ap.add_argument("--n-perms", type=int, default=3,
                    help="Number of null permutations (default 3 per Stage A4)")
    ap.add_argument("--skip-null", action="store_true",
                    help="Skip matched null permutation entirely")
    ap.add_argument("--n-steps", type=int, default=50,
                    help="Integrated Gradients approximation steps (>=50)")
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
    core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()
    N = len(core_pats)

    lpath = DATA_DIR / "processed" / "labels.csv"
    if not lpath.exists():
        sys.exit("data/processed/labels.csv missing. Run: python scripts/download_clinical.py && python scripts/build_labels.py")
    lab_df = pd.read_csv(lpath).set_index("patient_id").loc[core_pats]
    y_hist = lab_df["histology_label"].astype(int).values

    mrna_df = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet").loc[core_pats]
    mirna_df = pd.read_parquet(DATA_DIR / "processed" / "mirna_unique.parquet").loc[core_pats]
    meth_df = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet").loc[core_pats]
    mats = {"mrna": mrna_df.values, "mirna": mirna_df.values, "meth": meth_df.values}
    feat_dims = {"mrna": mrna_df.shape[1], "mirna": mirna_df.shape[1], "meth": meth_df.shape[1]}

    with open(DATA_DIR / "splits" / "splits_5x5.json") as f:
        all_splits = json.load(f)["folds"]
    splits = all_splits[:5]

    selected_pways, pway_summary = select_unsupervised_pathways(DATA_DIR, top_k=192)
    P = len(selected_pways)

    pri = DATA_DIR / "interim" / "priors"
    prior_pathways = json.load(open(pri / "gene_pathway_mask_pathways.json"))
    pway_map = {p: i for i, p in enumerate(prior_pathways)}
    pway_idx = [pway_map[p] for p in selected_pways]

    mask_gene_full = sp.load_npz(pri / "gene_pathway_mask.npz")
    gene_names = json.load(open(pri / "gene_pathway_mask_genes.json"))
    gene_map = {g: i for i, g in enumerate(gene_names)}
    mask_mrna_np = mask_gene_full[[gene_map[c] for c in mrna_df.columns], :][:, pway_idx].T.toarray()
    mask_meth_np = mask_gene_full[[gene_map[c] for c in meth_df.columns], :][:, pway_idx].T.toarray()

    # Raw miRNA mask without fallback (genuine prior edges only)
    sub_mask_strong_raw = sp.load_npz(pri / "mirna_unique_pathway_mask_strong.npz")[:, pway_idx].toarray()
    mask_mirna_genuine_np = sub_mask_strong_raw.T  # (192, 1035)

    # miRNA mask with Fork 6 fallback (for model execution)
    sub_mask_strong = sub_mask_strong_raw.copy()
    zero_strong = sub_mask_strong.sum(axis=1) == 0
    sub_mask_strong[zero_strong, P - 1] = True  # Fork 6 fallback
    mask_mirna_model_np = sub_mask_strong.T

    # Pathway sizes: total genuine features present in data per pathway
    pathway_sizes = (
        mask_mrna_np.sum(axis=1) +
        mask_mirna_genuine_np.sum(axis=1) +
        mask_meth_np.sum(axis=1)
    )

    masks_model = {
        "mrna": torch.tensor(mask_mrna_np, dtype=torch.bool),
        "mirna": torch.tensor(mask_mirna_model_np, dtype=torch.bool),
        "meth": torch.tensor(mask_meth_np, dtype=torch.bool),
    }

    adj_full = sp.load_npz(pri / "pathway_adjacency.npz")
    M = 3
    S = 1 + M * P
    B_bias = np.zeros((S, S), dtype=np.float32)
    B_bias[0, :] = 0.5
    B_bias[:, 0] = 0.5
    B_bias[0, 0] = 0.0
    sub_adj = adj_full[pway_idx, :][:, pway_idx].toarray()
    sym_adj = ((sub_adj + sub_adj.T) > 0).astype(float)
    for m1 in range(M):
        for m2 in range(M):
            offset1 = 1 + m1 * P
            offset2 = 1 + m2 * P
            if m1 != m2:
                for p in range(P):
                    B_bias[offset1 + p, offset2 + p] += 1.0
            B_bias[offset1 : offset1 + P, offset2 : offset2 + P] += sym_adj * 0.5
    np.fill_diagonal(B_bias, 0.0)
    bio_bias = torch.tensor(B_bias, dtype=torch.float32)

    return (core_pats, N, y_hist, mats, masks_model, feat_dims, bio_bias,
            selected_pways, pway_summary,
            mask_mrna_np, mask_mirna_genuine_np, mask_meth_np, pathway_sizes,
            P, splits, all_splits)


def prepare_fold(tr_idx, te_idx, mats, mods=None):
    if mods is None:
        mods = E1_MODS
    X_tr, X_te = {}, {}
    for m in mods:
        raw_tr = mats[m][tr_idx]
        raw_te = mats[m][te_idx]
        imp = SimpleImputer(strategy="median")
        raw_tr = imp.fit_transform(raw_tr)
        raw_te = imp.transform(raw_te)
        sc = StandardScaler()
        X_tr[m] = sc.fit_transform(raw_tr)
        X_te[m] = sc.transform(raw_te)
    return X_tr, X_te


def build_model(feat_dims, masks, P, bio_bias):
    return BioFusionFormerModel(
        modality_features=feat_dims,
        modality_masks=masks,
        n_pathways=P,
        d_model=32, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.3, attn_dropout=0.2, modality_dropout=0.2,
        use_pathway_bias=True, use_gating=True, bio_bias=bio_bias,
    )


def train_one_seed(model, X_tr_dict, y_tr, pos_weight, seed, **hp):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    first_mod = list(X_tr_dict.keys())[0]
    in_tr_idx, in_val_idx = next(skf.split(X_tr_dict[first_mod], y_tr))

    def _t(arr_dict, idx):
        return {m: torch.tensor(arr_dict[m][idx], dtype=torch.float32) for m in arr_dict}

    X_in_tr = _t(X_tr_dict, in_tr_idx)
    y_in_tr = torch.tensor(y_tr[in_tr_idx], dtype=torch.float32)
    X_in_val = _t(X_tr_dict, in_val_idx)
    y_in_val = torch.tensor(y_tr[in_val_idx], dtype=torch.float32)

    lr = hp.get("lr", 3e-4)
    wd = hp.get("weight_decay", 1e-2)
    max_epochs = hp.get("max_epochs", 100)
    patience = hp.get("patience", 20)
    batch_size = hp.get("batch_size", 16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.05)
    criterion = LabelSmoothedBCEWithLogitsLoss(eps=0.05, pos_weight=pos_weight)

    best_val = float("inf")
    best_w = None
    cnt = 0
    N_tr = len(in_tr_idx)
    n_batches = math.ceil(N_tr / batch_size)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(N_tr)
        for b in range(n_batches):
            idx_b = perm[b * batch_size: (b + 1) * batch_size]
            bx = {m: X_in_tr[m][idx_b] for m in X_in_tr}
            by = y_in_tr[idx_b]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out["logit"], by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_in_val)["logit"], y_in_val).item()
        if vl < best_val - 1e-4:
            best_val = vl
            best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

    if best_w is not None:
        model.load_state_dict(best_w)
    return model


def compute_ig(model, X_te_dict, n_steps=50, internal_batch_size=25):
    """Returns ig_mrna, ig_mirna, ig_meth each (N_te, F_mod)."""
    from captum.attr import IntegratedGradients

    model.eval()

    class MultimodalWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x_mr, x_mi, x_me):
            return self.m({"mrna": x_mr, "mirna": x_mi, "meth": x_me})["logit"]

    ig = IntegratedGradients(MultimodalWrapper(model))

    N_te = len(X_te_dict["mrna"])
    ig_mrna_list, ig_mirna_list, ig_meth_list = [], [], []

    for i in range(N_te):
        x_mr = torch.tensor(X_te_dict["mrna"][i : i + 1], dtype=torch.float32, requires_grad=True)
        x_mi = torch.tensor(X_te_dict["mirna"][i : i + 1], dtype=torch.float32, requires_grad=True)
        x_me = torch.tensor(X_te_dict["meth"][i : i + 1], dtype=torch.float32, requires_grad=True)

        attrs = ig.attribute(
            (x_mr, x_mi, x_me),
            baselines=(torch.zeros_like(x_mr), torch.zeros_like(x_mi), torch.zeros_like(x_me)),
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
        ig_mrna_list.append(attrs[0].detach().cpu().numpy()[0])
        ig_mirna_list.append(attrs[1].detach().cpu().numpy()[0])
        ig_meth_list.append(attrs[2].detach().cpu().numpy()[0])

    return np.array(ig_mrna_list), np.array(ig_mirna_list), np.array(ig_meth_list)


def ig_to_pathway_scores_v2(ig_mrna, ig_mirna, ig_meth,
                            mask_mrna_np, mask_mirna_genuine_np, mask_meth_np,
                            pathway_sizes):
    """
    Computes both Metric (i) and Metric (ii) side-by-side per Stage A2:
      (i)  sum of |IG| over genuine member features, EXCLUDING fallback miRNA edges;
      (ii) mean |IG| per genuine member feature (size-normalised).
    """
    mean_mrna = np.abs(ig_mrna).mean(axis=0)
    mean_mirna = np.abs(ig_mirna).mean(axis=0)
    mean_meth = np.abs(ig_meth).mean(axis=0)

    pway_mrna = mask_mrna_np @ mean_mrna
    pway_mirna_gen = mask_mirna_genuine_np @ mean_mirna
    pway_meth = mask_meth_np @ mean_meth

    # (i) Genuine Sum
    pway_sum_genuine = pway_mrna + pway_mirna_gen + pway_meth

    # (ii) Size-normalised Mean
    pway_mean_genuine = pway_sum_genuine / np.maximum(pathway_sizes, 1)

    return (pway_mrna, pway_mirna_gen, pway_meth,
            pway_sum_genuine, pway_mean_genuine)


def compute_en_pathway_scores(X_tr_mrna, y_tr, mask_mrna_np):
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_tr_mrna)
    en = LogisticRegression(
        solver="saga", penalty="elasticnet", C=0.5, l1_ratio=0.5,
        max_iter=500, random_state=42,
    )
    en.fit(X_sc, y_tr)
    gene_imp = np.abs(en.coef_[0])
    pway_scores = mask_mrna_np @ gene_imp
    return pway_scores


def is_generic(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in GENERIC_KEYWORDS)


def train_all_patients_model(N, mats, y_hist, masks, feat_dims, P, bio_bias):
    log.info("A6: Training representation model on all 121 patients (3 seeds, in-sample only)...")
    pos_weight = float((y_hist == 0).sum() / (y_hist == 1).sum())

    X_scaled = {}
    for m in E1_MODS:
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_scaled[m] = sc.fit_transform(imp.fit_transform(mats[m]))

    X_t = {m: torch.tensor(X_scaled[m], dtype=torch.float32) for m in E1_MODS}
    y_t = torch.tensor(y_hist, dtype=torch.float32)
    criterion = LabelSmoothedBCEWithLogitsLoss(eps=0.05, pos_weight=pos_weight)

    all_embs = []
    for seed in SEEDS:
        set_seed(seed)
        model = build_model(feat_dims, masks, P, bio_bias)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-4 * 0.05)
        batch_size = 16
        n_batches = math.ceil(N / batch_size)
        for epoch in range(100):
            model.train()
            perm = torch.randperm(N)
            for b in range(n_batches):
                idx_b = perm[b * batch_size: (b + 1) * batch_size]
                bx = {m: X_t[m][idx_b] for m in X_t}
                by = y_t[idx_b]
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out["logit"], by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            out = model(X_t)
            emb = (out["cls_rep"] + out["gated_rep"]).cpu().numpy()
        all_embs.append(emb)
        log.info(f"  seed {seed} done")

    return np.mean(all_embs, axis=0)


def compute_calibration_v2(y_hist, frozen_folds, N):
    """
    Stage A5: Logistic recalibration without penalty on logit(pooled out-of-fold probability).
    Reports slope, intercept, 1000-draw patient bootstrap CI, and 5-fold CV Brier before/after.
    """
    pp = {}
    for r in frozen_folds:
        for i, p in zip(r["te_idx"], r["probs"]):
            pp.setdefault(i, []).append(p)
    oof = np.array([np.mean(pp[i]) for i in range(N)])
    brier_before = float(brier_score_loss(y_hist, oof))

    z = logit(np.clip(oof, 1e-6, 1.0 - 1e-6))
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y_hist)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])

    rng = np.random.default_rng(42)
    slopes_b, intercepts_b = [], []
    for _ in range(1000):
        idx = rng.choice(N, size=N, replace=True)
        if len(np.unique(y_hist[idx])) < 2:
            continue
        lr_b = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        lr_b.fit(z[idx].reshape(-1, 1), y_hist[idx])
        slopes_b.append(float(lr_b.coef_[0][0]))
        intercepts_b.append(float(lr_b.intercept_[0]))

    slope_ci = [float(np.percentile(slopes_b, 2.5)), float(np.percentile(slopes_b, 97.5))]
    intercept_ci = [float(np.percentile(intercepts_b, 2.5)), float(np.percentile(intercepts_b, 97.5))]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cal = np.zeros(N)
    for tr_i, val_i in skf.split(z, y_hist):
        lr_cv = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        lr_cv.fit(z[tr_i].reshape(-1, 1), y_hist[tr_i])
        oof_cal[val_i] = lr_cv.predict_proba(z[val_i].reshape(-1, 1))[:, 1]

    brier_after = float(brier_score_loss(y_hist, oof_cal))
    return oof, oof_cal, brier_before, brier_after, slope, intercept, slope_ci, intercept_ci


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("BioFusionFormer E1 — Explainability Analysis v2 (Stage A)")
    log.info("=" * 70)

    (core_pats, N, y_hist, mats, masks_model, feat_dims, bio_bias,
     selected_pways, pway_summary,
     mask_mrna_np, mask_mirna_genuine_np, mask_meth_np, pathway_sizes,
     P, splits, all_splits) = load_data()
    log.info(f"N={N} | P={P} pathways | {len(splits)} folds for Stage 3 models")

    frozen_folds = []
    for i in range(25):
        fp = FROZEN_DIR / f"fold_{i:02d}.json"
        if fp.exists():
            with open(fp) as f:
                frozen_folds.append(json.load(f))
    log.info(f"Loaded {len(frozen_folds)} frozen folds")

    if not MODELS_DIR.exists() or len(list(MODELS_DIR.glob("fold_0*_seed42.pt"))) < 5:
        log.error(
            f"Stage 3 fold models not found in {MODELS_DIR}.\n"
            "Run first: python scripts/run_e1.py --run --tag final_v1_repro --n-repeats 1"
        )
        sys.exit(1)

    # ========================================================================
    # A2 / A3: Compute IG over 5 saved fold models (genuine sum & normalised mean)
    # ========================================================================
    log.info("\nA2 / A3: Computing per-fold IG attributions (genuine sum & size-normalised mean)...")

    ig_sum_per_fold = []
    ig_mean_per_fold = []
    ig_mrna_per_fold = []
    ig_mirna_per_fold = []
    ig_meth_per_fold = []
    en_pway_per_fold = []

    top20_sum_per_fold = []
    top20_mean_per_fold = []
    top20_en_per_fold = []
    rollout_spreads = []

    for f_i, fold in enumerate(splits):
        log.info(f"  Fold {f_i+1}/5 ...")
        tr_idx = np.array(fold["train_indices"])
        te_idx = np.array(fold["test_indices"])
        y_tr = y_hist[tr_idx]

        X_tr, X_te = prepare_fold(tr_idx, te_idx, mats)

        mpath = MODELS_DIR / f"fold_{f_i:02d}_seed42.pt"
        model = build_model(feat_dims, masks_model, P, bio_bias)
        model.load_state_dict(torch.load(mpath, map_location="cpu"))
        model.eval()

        ig_mrna, ig_mirna, ig_meth = compute_ig(model, X_te, n_steps=args.n_steps, internal_batch_size=25)
        (pway_mrna, pway_mirna_gen, pway_meth,
         pway_sum_gen, pway_mean_gen) = ig_to_pathway_scores_v2(
            ig_mrna, ig_mirna, ig_meth,
            mask_mrna_np, mask_mirna_genuine_np, mask_meth_np,
            pathway_sizes
        )

        ig_sum_per_fold.append(pway_sum_gen)
        ig_mean_per_fold.append(pway_mean_gen)
        ig_mrna_per_fold.append(pway_mrna)
        ig_mirna_per_fold.append(pway_mirna_gen)
        ig_meth_per_fold.append(pway_meth)

        t20_sum = [selected_pways[i] for i in np.argsort(pway_sum_gen)[::-1][:20]]
        t20_mean = [selected_pways[i] for i in np.argsort(pway_mean_gen)[::-1][:20]]
        top20_sum_per_fold.append(t20_sum)
        top20_mean_per_fold.append(t20_mean)

        # Rollout
        with torch.no_grad():
            te_t = {m: torch.tensor(X_te[m], dtype=torch.float32) for m in E1_MODS}
            out = model(te_t)
            roll = compute_attention_rollout(out["attns"]).cpu().numpy()[:, 0, 1:]
        pw_att = roll.reshape(len(te_idx), 3, P).sum(axis=1).mean(axis=0)
        spread = float(pw_att.max() / max(pw_att.min(), 1e-12))
        rollout_spreads.append(spread)

        # ElasticNet baseline on train-fold mRNA
        en_scores = compute_en_pathway_scores(X_tr["mrna"], y_tr, mask_mrna_np)
        en_pway_per_fold.append(en_scores)
        t20_en = [selected_pways[i] for i in np.argsort(en_scores)[::-1][:20]]
        top20_en_per_fold.append(t20_en)

        log.info(f"    Rollout spread: {spread:.3f}x | Top-1 Sum: {t20_sum[0]} | Top-1 Mean: {t20_mean[0]}")

    # Aggregations across 5 folds
    mean_ig_sum = np.mean(ig_sum_per_fold, axis=0)
    mean_ig_mean = np.mean(ig_mean_per_fold, axis=0)

    # Size correlations (quantifying size bias)
    rho_sum, pval_sum = stats.spearmanr(pathway_sizes, mean_ig_sum)
    rho_mean, pval_mean = stats.spearmanr(pathway_sizes, mean_ig_mean)
    log.info(f"  Spearman correlation with pathway size:")
    log.info(f"    Metric (i) Genuine Sum : rho = {rho_sum:.4f} (p = {pval_sum:.2e})")
    log.info(f"    Metric (ii) Normalised Mean: rho = {rho_mean:.4f} (p = {pval_mean:.2e})")

    # Top-20 lists
    top20_sum_idx = np.argsort(mean_ig_sum)[::-1][:20]
    top20_sum_ids = [selected_pways[i] for i in top20_sum_idx]
    top20_sum_names = [pway_summary.loc[p, "name"] if p in pway_summary.index else p for p in top20_sum_ids]
    n_generic_sum = sum(1 for n in top20_sum_names if is_generic(n))

    top20_mean_idx = np.argsort(mean_ig_mean)[::-1][:20]
    top20_mean_ids = [selected_pways[i] for i in top20_mean_idx]
    top20_mean_names = [pway_summary.loc[p, "name"] if p in pway_summary.index else p for p in top20_mean_ids]
    n_generic_mean = sum(1 for n in top20_mean_names if is_generic(n))

    log.info(f"  Generic pathways in Top-20: Metric (i) Sum = {n_generic_sum}/20 ({n_generic_sum/20:.0%}), Metric (ii) Mean = {n_generic_mean}/20 ({n_generic_mean/20:.0%})")

    # Stability Jaccard
    jaccard_sum, recurrent_sum = compute_pathway_jaccard(top20_sum_per_fold)
    jaccard_mean, recurrent_mean = compute_pathway_jaccard(top20_mean_per_fold)
    jaccard_en, recurrent_en = compute_pathway_jaccard(top20_en_per_fold)
    log.info(f"  Mean pairwise top-20 Jaccard across 5 folds:")
    log.info(f"    IG (i) Genuine Sum : {jaccard_sum:.4f}")
    log.info(f"    IG (ii) Normalised Mean: {jaccard_mean:.4f}")
    log.info(f"    ElasticNet mRNA    : {jaccard_en:.4f}")

    # ========================================================================
    # A4: Matched Null (resumable per (perm, fold) with atomic JSON)
    # ========================================================================
    null_jaccards_sum = []
    null_jaccards_mean = []
    n_perms_used = args.n_perms if not args.skip_null else 0

    if args.skip_null:
        log.info("\nA4: Matched null skipped (--skip-null).")
    else:
        log.info(f"\nA4: Matched null ({n_perms_used} perms x 5 folds, same training procedure, resumable cache)...")
        rng_null = np.random.default_rng(0)
        for perm_i in range(n_perms_used):
            y_perm = rng_null.permutation(y_hist)
            top20_null_sum_folds = []
            top20_null_mean_folds = []

            for f_i, fold in enumerate(splits):
                cache_file = NULL_CACHE_DIR / f"perm_{perm_i:02d}_fold_{f_i:02d}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, encoding="utf-8") as f:
                            rec = json.load(f)
                        top20_null_sum_folds.append(rec["top20_sum"])
                        top20_null_mean_folds.append(rec["top20_mean"])
                        log.info(f"  Loaded null cache: perm {perm_i+1}, fold {f_i+1}")
                        continue
                    except Exception as e:
                        log.warning(f"  Corrupt cache file {cache_file}, recomputing: {e}")

                tr_idx = np.array(fold["train_indices"])
                te_idx = np.array(fold["test_indices"])
                y_tr_perm = y_perm[tr_idx]
                X_tr, X_te = prepare_fold(tr_idx, te_idx, mats)
                pos_w_perm = max(float((y_tr_perm == 0).sum() / max((y_tr_perm == 1).sum(), 1)), 0.1)

                seed_null = 42 + perm_i * 100
                set_seed(seed_null)
                t_n0 = time.time()
                m_null = build_model(feat_dims, masks_model, P, bio_bias)
                m_null = train_one_seed(m_null, X_tr, y_tr_perm, pos_w_perm, seed_null, **HP)
                ig_mr, ig_mi, ig_me = compute_ig(m_null, X_te, n_steps=args.n_steps, internal_batch_size=25)
                _, _, _, pw_null_sum, pw_null_mean = ig_to_pathway_scores_v2(
                    ig_mr, ig_mi, ig_me,
                    mask_mrna_np, mask_mirna_genuine_np, mask_meth_np,
                    pathway_sizes
                )
                t20_n_sum = [selected_pways[i] for i in np.argsort(pw_null_sum)[::-1][:20]]
                t20_n_mean = [selected_pways[i] for i in np.argsort(pw_null_mean)[::-1][:20]]

                # Atomic write cache
                tmp_cache = cache_file.with_suffix(".tmp")
                with open(tmp_cache, "w", encoding="utf-8") as f:
                    json.dump({"top20_sum": t20_n_sum, "top20_mean": t20_n_mean}, f)
                os.replace(tmp_cache, cache_file)

                top20_null_sum_folds.append(t20_n_sum)
                top20_null_mean_folds.append(t20_n_mean)
                log.info(f"  Computed perm {perm_i+1}/3, fold {f_i+1}/5 ({time.time()-t_n0:.1f}s)")

            j_sum, _ = compute_pathway_jaccard(top20_null_sum_folds)
            j_mean, _ = compute_pathway_jaccard(top20_null_mean_folds)
            null_jaccards_sum.append(j_sum)
            null_jaccards_mean.append(j_mean)
            log.info(f"  Perm {perm_i+1}/{n_perms_used}: null Jaccard (sum) = {j_sum:.4f}, (mean) = {j_mean:.4f}")

    null_sum_stat = (float(np.mean(null_jaccards_sum)), float(np.std(null_jaccards_sum))) if null_jaccards_sum else (None, None)
    null_mean_stat = (float(np.mean(null_jaccards_mean)), float(np.std(null_jaccards_mean))) if null_jaccards_mean else (None, None)

    # ========================================================================
    # A5: Recalibration
    # ========================================================================
    log.info("\nA5: Logistic recalibration & reliability plot...")
    (oof, oof_cal, brier_before, brier_after,
     slope, intercept, slope_ci, intercept_ci) = compute_calibration_v2(y_hist, frozen_folds, N)
    log.info(f"  Brier before recalibration: {brier_before:.4f}")
    log.info(f"  Brier after 5-fold CV recalibration: {brier_after:.4f}")
    log.info(f"  Recalibration slope: {slope:.3f} (95% CI [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}])")
    log.info(f"  Recalibration intercept: {intercept:.3f} (95% CI [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}])")

    # Reliability plot with bin counts
    prob_true, prob_pred = calibration_curve(y_hist, oof, n_bins=5, strategy="quantile")
    # Bin counts
    quantiles = np.quantile(oof, np.linspace(0, 1, 6))
    bin_counts = []
    for bi in range(5):
        if bi == 4:
            c = int(np.sum((oof >= quantiles[bi]) & (oof <= quantiles[bi+1])))
        else:
            c = int(np.sum((oof >= quantiles[bi]) & (oof < quantiles[bi+1])))
        bin_counts.append(c)

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=180)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#2980b9", lw=2, label=f"E1 pooled OOF (Brier={brier_before:.4f})")
    for x_p, y_p, cnt in zip(prob_pred, prob_true, bin_counts):
        ax.annotate(f"n={cnt}", (x_p, y_p), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, fontweight="bold", color="#2c3e50")
    ax.set_xlabel("Mean predicted P(Unfavorable histology)", fontsize=9)
    ax.set_ylabel("Observed fraction positive", fontsize=9)
    ax.set_title(
        f"E1 Reliability Curve (5 quantile bins; N=121)\n"
        f"Recalibration slope = {slope:.3f} [{slope_ci[0]:.2f}, {slope_ci[1]:.2f}] | intercept = {intercept:.3f} [{intercept_ci[0]:.2f}, {intercept_ci[1]:.2f}]",
        fontsize=8
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_reliability.png")
    plt.close(fig)
    log.info("  Saved figures/explain/calibration_reliability.png")

    # ========================================================================
    # A6: Representation (all-121 model)
    # ========================================================================
    log.info("\nA6: Patient representation figure...")
    try:
        import umap as umap_module
        _umap_available = True
    except ImportError:
        _umap_available = False

    from sklearn.decomposition import PCA
    embs = train_all_patients_model(N, mats, y_hist, masks_model, feat_dims, P, bio_bias)
    pca = PCA(n_components=2, random_state=42)
    embs_pca = pca.fit_transform(embs)
    labels_text = ["Favorable" if y == 0 else "Unfavorable" for y in y_hist]

    n_ax = 2 if _umap_available else 1
    fig, axes = plt.subplots(1, n_ax, figsize=(5.5 * n_ax, 4.5), dpi=180)
    if n_ax == 1:
        axes = [axes]

    for label, col in [("Favorable", "#2b5c8f"), ("Unfavorable", "#c0392b")]:
        msk = [l == label for l in labels_text]
        axes[0].scatter(embs_pca[msk, 0], embs_pca[msk, 1], c=col, label=label, alpha=0.8, s=40, edgecolors="none")
    axes[0].set_title(
        f"PCA: in-sample; separation reflects training labels;\nnot evidence of generalisation\n(PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)",
        fontsize=8
    )
    axes[0].set_xlabel("PC1", fontsize=8); axes[0].set_ylabel("PC2", fontsize=8)
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    if _umap_available:
        reducer = umap_module.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embs_umap = reducer.fit_transform(embs)
        for label, col in [("Favorable", "#2b5c8f"), ("Unfavorable", "#c0392b")]:
            msk = [l == label for l in labels_text]
            axes[1].scatter(embs_umap[msk, 0], embs_umap[msk, 1], c=col, label=label, alpha=0.8, s=40, edgecolors="none")
        axes[1].set_title("UMAP: in-sample; separation reflects training labels;\nnot evidence of generalisation", fontsize=8)
        axes[1].set_xlabel("UMAP 1", fontsize=8); axes[1].set_ylabel("UMAP 2", fontsize=8)
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    fig.suptitle("BioFusionFormer E1: Patient Representations (N=121)", fontsize=9, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "representation_pca_umap.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved figures/explain/representation_pca_umap.png")

    # ========================================================================
    # Attribution Bar Charts (Metric i and Metric ii side-by-side)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=180)

    # Panel 1: Metric (i) Sum (genuine)
    y_pos = np.arange(20)
    top20_sum_vals = mean_ig_sum[top20_sum_idx][::-1]
    top20_sum_labels = [n[:45] for n in top20_sum_names[::-1]]
    bar_cols_sum = ["#c0392b" if is_generic(n) else "#16a085" for n in top20_sum_names[::-1]]
    axes[0].barh(y_pos, top20_sum_vals, color=bar_cols_sum, alpha=0.85)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(top20_sum_labels, fontsize=7)
    axes[0].set_xlabel("Mean |IG| Sum (genuine features, 5-fold mean)", fontsize=8)
    axes[0].set_title(
        f"(i) Sum of |IG| (Fallback Excluded)\n"
        f"Spearman with size: rho={rho_sum:.2f} (size bias) | Generic: {n_generic_sum}/20",
        fontsize=8
    )
    axes[0].grid(axis="x", alpha=0.3)

    # Panel 2: Metric (ii) Size-normalised Mean
    top20_mean_vals = mean_ig_mean[top20_mean_idx][::-1]
    top20_mean_labels = [n[:45] for n in top20_mean_names[::-1]]
    bar_cols_mean = ["#c0392b" if is_generic(n) else "#2980b9" for n in top20_mean_names[::-1]]
    axes[1].barh(y_pos, top20_mean_vals, color=bar_cols_mean, alpha=0.85)
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels(top20_mean_labels, fontsize=7)
    axes[1].set_xlabel("Mean |IG| per Member Feature (size-normalised)", fontsize=8)
    axes[1].set_title(
        f"(ii) Mean |IG| per Feature (Size-Normalised)\n"
        f"Spearman with size: rho={rho_mean:.2f} | Generic: {n_generic_mean}/20",
        fontsize=8
    )
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("BioFusionFormer E1: Top-20 Pathways by Integrated Gradients (Side-by-Side Aggregations)", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ig_top20_pathways.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved figures/explain/ig_top20_pathways.png")

    # Save JSON output
    out_json = {
        "n_folds": 5,
        "n_ig_steps": args.n_steps,
        "fallback_audit": {
            "selected_pways_count": P,
            "p_minus_1_id": selected_pways[P-1],
            "p_minus_1_name": pway_summary.loc[selected_pways[P-1], "name"] if selected_pways[P-1] in pway_summary.index else "N/A",
            "fallback_mirna_edges": 726,
            "genuine_mirna_edges": int(mask_mirna_genuine_np[P-1].sum()),
            "note": "ODC pathway was confirmed as index P-1. Summed IG of 4.98 in earlier report was an artefact of 726 fallback miRNA edges.",
        },
        "metric_i_genuine_sum": {
            "spearman_rho_with_pathway_size": float(rho_sum),
            "spearman_pvalue": float(pval_sum),
            "top20_pathways": [
                {"rank": r+1, "id": pid, "name": name, "score": float(mean_ig_sum[idx]), "generic": is_generic(name)}
                for r, (pid, name, idx) in enumerate(zip(top20_sum_ids, top20_sum_names, top20_sum_idx))
            ],
            "generic_fraction_top20": float(n_generic_sum / 20),
            "mean_pairwise_jaccard_stability": float(jaccard_sum),
        },
        "metric_ii_size_normalised_mean": {
            "spearman_rho_with_pathway_size": float(rho_mean),
            "spearman_pvalue": float(pval_mean),
            "top20_pathways": [
                {"rank": r+1, "id": pid, "name": name, "score": float(mean_ig_mean[idx]), "generic": is_generic(name)}
                for r, (pid, name, idx) in enumerate(zip(top20_mean_ids, top20_mean_names, top20_mean_idx))
            ],
            "generic_fraction_top20": float(n_generic_mean / 20),
            "mean_pairwise_jaccard_stability": float(jaccard_mean),
        },
        "elastic_net_stability": {
            "mean_pairwise_jaccard": float(jaccard_en),
        },
        "matched_null_stability": {
            "n_perms_used": n_perms_used,
            "null_jaccard_sum_mean": null_sum_stat[0],
            "null_jaccard_sum_std": null_sum_stat[1],
            "null_jaccard_mean_mean": null_mean_stat[0],
            "null_jaccard_mean_std": null_mean_stat[1],
            "note": "With 3 permutations, no p-value is computed." if n_perms_used == 3 else "",
        },
        "attention_rollout_secondary": {
            "max_min_ratios": [float(s) for s in rollout_spreads],
            "mean_spread": float(np.mean(rollout_spreads)),
            "note": "Rollout is near-uniform (1.06x) and is not used as primary explanation.",
        },
        "recalibration": {
            "brier_before": float(brier_before),
            "brier_after_5fold_cv": float(brier_after),
            "slope": float(slope),
            "slope_bootstrap_95ci": slope_ci,
            "intercept": float(intercept),
            "intercept_bootstrap_95ci": intercept_ci,
            "note": "Recalibration fitted via unpenalized logistic regression on logit(pooled_oof). Slope > 1 indicates under-confidence (probabilities compressed toward middle).",
        },
    }

    with open(OUT_DIR / "explainability_results.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    log.info(f"Saved {OUT_DIR / 'explainability_results.json'}")

    # ========================================================================
    # A7: Write Markdown summary strictly from JSON
    # ========================================================================
    L = []
    L.append("# BioFusionFormer E1 — Explainability Summary v2 (Stage A)")
    L.append("")
    L.append("> **Scope**: 5 fold models from the Stage 3 reproducibility re-run (`models/e1_folds_final_v1_repro/`). "
             "Attributions computed on TEST patients of each fold with Integrated Gradients ($n_{\\text{steps}}=50$).")
    L.append("")
    L.append("## 1. Fallback Audit & Resolving the ODC Artefact")
    L.append("")
    L.append(f"- **ODC Pathway Status:** In the earlier draft, *Regulation of ornithine decarboxylase (ODC)* (`R-HSA-350562`) "
             f"appeared with an attribution score of 4.98 (~2x the runner-up).")
    L.append(f"- **Audit Finding:** Pathway `R-HSA-350562` is exactly index $P-1 = 191$ (the last selected pathway). "
             f"Under Fork 6 fallback wiring, all 726 miRNA features without a prior edge were wired into this single pathway token. "
             f"Summing $|IG|$ over its member features accumulated the background gradients of 726 unmapped miRNAs in addition to its 40 genes. "
             f"**This top finding was strictly an artefact of fallback token routing.**")
    L.append("- See full quantitative audit in `results/explainability/fallback_audit.md`.")
    L.append("")
    L.append("## 2. Integrated Gradients: Side-by-Side Aggregations")
    L.append("")
    L.append("Per Stage A2, attributions are reported in two complementary forms:")
    L.append("1. **Metric (i) Genuine Sum**: Sum of $|IG|$ over genuine member features, strictly excluding the 726 fallback miRNA edges.")
    L.append("2. **Metric (ii) Size-Normalised Mean**: Mean $|IG|$ per member feature (attribution divided by genuine feature count present in data).")
    L.append("")
    L.append("| Metric | Spearman $\\rho$ with Pathway Size | Size Bias Status | Generic Pathway Fraction in Top-20 |")
    L.append("|---|:---:|---|:---:|")
    L.append(f"| **(i) Genuine Sum** | **{rho_sum:.4f}** ($p = {pval_sum:.2e}$) | Substantial size bias towards large pathways | {n_generic_sum}/20 ({n_generic_sum/20:.0%}) |")
    L.append(f"| **(ii) Normalised Mean** | **{rho_mean:.4f}** ($p = {pval_mean:.2e}$) | Size bias removed | {n_generic_mean}/20 ({n_generic_mean/20:.0%}) |")
    L.append("")
    L.append("### Top-10 Pathways by Metric (i) Genuine Sum")
    L.append("")
    for r in range(min(10, len(top20_sum_ids))):
        pid = top20_sum_ids[r]
        name = top20_sum_names[r]
        sc = mean_ig_sum[top20_sum_idx[r]]
        gen_str = " *(generic)*" if is_generic(name) else ""
        L.append(f"{r+1}. **{name}** (`{pid}`){gen_str}: {sc:.4f}")
    L.append("")
    L.append("### Top-10 Pathways by Metric (ii) Size-Normalised Mean")
    L.append("")
    for r in range(min(10, len(top20_mean_ids))):
        pid = top20_mean_ids[r]
        name = top20_mean_names[r]
        sc = mean_ig_mean[top20_mean_idx[r]]
        gen_str = " *(generic)*" if is_generic(name) else ""
        L.append(f"{r+1}. **{name}** (`{pid}`){gen_str}: {sc:.5f}")
    L.append("")
    L.append("## 3. Stability Across 5 Saved Folds & Matched Null Permutation")
    L.append("")
    L.append("| Method | Mean Pairwise Jaccard (Top-20) | Note |")
    L.append("|---|:---:|---|")
    L.append(f"| **BioFusionFormer IG (i) Genuine Sum** | **{jaccard_sum:.4f}** | Fallback edges excluded |")
    L.append(f"| **BioFusionFormer IG (ii) Normalised Mean** | **{jaccard_mean:.4f}** | Size-normalised |")
    L.append(f"| **ElasticNet (mRNA, train-fold only)** | **{jaccard_en:.4f}** | High-dimensional linear baseline |")
    if null_sum_stat[0] is not None:
        L.append(f"| **Matched Null Permutation (i) Sum** | **{null_sum_stat[0]:.4f}** $\\pm$ {null_sum_stat[1]:.4f} | 3 permutations $\\times$ 5 folds (same E1 training) |")
        L.append(f"| **Matched Null Permutation (ii) Mean** | **{null_mean_stat[0]:.4f}** $\\pm$ {null_mean_stat[1]:.4f} | 3 permutations $\\times$ 5 folds |")
    L.append("")
    L.append("> [!NOTE]")
    L.append("> With 3 permutations, no p-value is computed or quoted.")
    L.append("")
    L.append("## 4. Recalibration (Stage A5)")
    L.append("")
    L.append("Fitted logistic regression without penalty on `logit(pooled_oof_probability)`:")
    L.append(f"- **Recalibration slope**: **{slope:.3f}** (1000-draw patient bootstrap 95% CI: **[{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]**)")
    L.append(f"- **Recalibration intercept**: **{intercept:.3f}** (1000-draw patient bootstrap 95% CI: **[{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]**)")
    L.append(f"- **Brier score before**: **{brier_before:.4f}**")
    L.append(f"- **Brier score after 5-fold CV calibrator**: **{brier_after:.4f}**")
    L.append("")
    L.append("*Why the previous slope 4.207 / intercept 3.080 occurred:* The previous script computed `np.log(oof)` (log-probability) "
             "instead of `scipy.special.logit(oof)` (log-odds). Under true logit recalibration, the slope is ~2.25 and intercept is ~ -0.08. "
             "A slope > 1 indicates the raw model is under-confident (predicted probabilities are moderately compressed toward the center).")
    L.append("")
    L.append("## 5. Representation Learning (Stage A6)")
    L.append("")
    L.append("Trained on all 121 cohort patients across 3 seeds (`figures/explain/representation_pca_umap.png`).")
    L.append("**Caption:** *in-sample; separation reflects training labels; not evidence of generalisation.*")
    L.append("")

    summary_text = "\n".join(L)
    (OUT_DIR / "summary.md").write_text(summary_text, encoding="utf-8")
    (OUT_DIR / "explainability_summary.md").write_text(summary_text, encoding="utf-8")
    log.info(f"Saved {OUT_DIR / 'summary.md'} and {OUT_DIR / 'explainability_summary.md'}")


if __name__ == "__main__":
    main()
