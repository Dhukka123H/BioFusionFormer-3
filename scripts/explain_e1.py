#!/usr/bin/env python3
"""
scripts/explain_e1.py
=====================
Stage 4: Explainability, Stability, Representation and Calibration for E1.

This script uses the 5 saved fold models from the Stage 3 reproducibility re-run
(models/e1_folds_final_v1_repro/fold_XX_seed42.pt) and the frozen 25-fold results.

Requirements per AGENT_PROMPT Stage 4:
  4.1  Primary attribution = Integrated Gradients (captum) over ALL THREE modalities,
       TEST patients of each fold, n_steps >= 50, baseline = zeros in standardised space
       (training-fold mean = 0 after z-score). Aggregate to 192 pathways. Print
       per-modality and summed scores. Attention rollout is secondary; print max/min ratio.
  4.2  Per-fold elastic-net baseline: fitted on TRAINING fold mRNA only (fixed C=0.5,
       l1_ratio=0.5). Map coefficients to pathways via mask. Never fit on all 121 patients.
  4.3  Stability: top-20 pathways per fold -> mean pairwise Jaccard across 5 folds, for
       (a) IG, (b) elastic net, (c) null permutation.
  4.4  Matched null: train E1 on shuffled labels, SAME procedure as E1 (max_epochs 100,
       patience 20, same inner validation, same seed rule), 5 permutations x 5 folds,
       then same IG pipeline. If compute too high, use 3 permutations and say so.
  4.4b Per-pathway stability frequency across folds; fraction of top-20 that are
       translation/ribosome/splicing pathways (generic high-variance). State caveat.
  4.5  IG vs EN agreement per fold: Jaccard of top-20 and Spearman over all 192. Report
       mean and range.
  4.6  Representation: train ONE model on all 121 patients (3 seeds, average embeddings),
       PCA + UMAP. Label "in-sample, visualisation only". No silhouette claims.
  4.7  Calibration: pooled OOF probs (from frozen 25 folds), Brier, logistic recalibration
       slope/intercept. If no nested-CV baseline predictions exist per patient, omit the
       baseline curve.
  4.8  Outputs: results/explainability/*.md/*.json and figures/explain/*.png

Usage:
  python scripts/explain_e1.py
  python scripts/explain_e1.py --n-perms 3      # use 3 null permutations
  python scripts/explain_e1.py --skip-null       # skip null entirely (saves ~1h of CPU)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
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
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
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
FIG_DIR = ROOT / "figures" / "explain"

# Generic high-variance pathway keywords (translation, ribosome, splicing)
GENERIC_KEYWORDS = [
    "translat", "ribosom", "spliceosom", "mrna processing", "rna splicing",
    "rrna", "transfer rna", "rna polymerase", "post-transcriptional",
]

# E1 config (frozen)
E1_MODS = ["mrna", "mirna", "meth"]
SEEDS = [42, 123, 999]
HP = dict(lr=3e-4, weight_decay=1e-2, max_epochs=100, patience=20, batch_size=16)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="BioFusionFormer E1 Explainability (Stage 4)")
    ap.add_argument("--n-perms", type=int, default=5,
                    help="Number of null permutations (default 5; use 3 if compute is limited)")
    ap.add_argument("--skip-null", action="store_true",
                    help="Skip matched null permutation entirely")
    ap.add_argument("--n-steps", type=int, default=50,
                    help="Integrated Gradients approximation steps (>=50)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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
    splits = all_splits[:5]  # Stage 3 re-run used n-repeats=1

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

    sub_mask_strong = sp.load_npz(pri / "mirna_unique_pathway_mask_strong.npz")[:, pway_idx].toarray()
    zero_strong = sub_mask_strong.sum(axis=1) == 0
    sub_mask_strong[zero_strong, P - 1] = True  # frozen v1 fallback (Fork 6)
    mask_mirna_np = sub_mask_strong.T

    masks = {
        "mrna": torch.tensor(mask_mrna_np, dtype=torch.bool),
        "mirna": torch.tensor(mask_mirna_np, dtype=torch.bool),
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

    return (core_pats, N, y_hist, mats, masks, feat_dims, bio_bias,
            selected_pways, pway_summary,
            mask_mrna_np, mask_mirna_np, mask_meth_np,
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


# ---------------------------------------------------------------------------
# Training loop (for null permutations; mirrors run_e1.py)
# ---------------------------------------------------------------------------
def train_one_seed(model, X_tr_dict, y_tr, pos_weight, seed, **hp):
    from sklearn.model_selection import StratifiedKFold
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


# ---------------------------------------------------------------------------
# Integrated Gradients over all 3 modalities
# ---------------------------------------------------------------------------
def compute_ig(model, X_te_dict, n_steps=50, internal_batch_size=10):
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

    # Process patient-by-patient with internal_batch_size to keep memory bounded (<100 MB)
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


def ig_to_pathway_scores(ig_mrna, ig_mirna, ig_meth, mask_mrna_np, mask_mirna_np, mask_meth_np):
    mean_mrna = np.abs(ig_mrna).mean(axis=0)
    mean_mirna = np.abs(ig_mirna).mean(axis=0)
    mean_meth = np.abs(ig_meth).mean(axis=0)
    pway_mrna = mask_mrna_np @ mean_mrna
    pway_mirna = mask_mirna_np @ mean_mirna
    pway_meth = mask_meth_np @ mean_meth
    pway_sum = pway_mrna + pway_mirna + pway_meth
    return pway_mrna, pway_mirna, pway_meth, pway_sum


# ---------------------------------------------------------------------------
# Elastic-net baseline (per fold, train fold mRNA only)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Generic pathway detection
# ---------------------------------------------------------------------------
def is_generic(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in GENERIC_KEYWORDS)


# ---------------------------------------------------------------------------
# Stage 4.6: all-121 representation model
# ---------------------------------------------------------------------------
def train_all_patients_model(N, mats, y_hist, masks, feat_dims, P, bio_bias):
    log.info("4.6: Training representation model on all 121 patients (3 seeds, in-sample only)...")
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


# ---------------------------------------------------------------------------
# Calibration (4.7)
# ---------------------------------------------------------------------------
def compute_calibration(y_hist, frozen_folds, N):
    pp = {}
    for r in frozen_folds:
        for i, p in zip(r["te_idx"], r["probs"]):
            pp.setdefault(i, []).append(p)
    oof = np.array([np.mean(pp[i]) for i in range(N)])

    brier = float(brier_score_loss(y_hist, oof))

    logit_oof = np.log(np.clip(oof, 1e-7, 1 - 1e-7))
    lr_cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr_cal.fit(logit_oof.reshape(-1, 1), y_hist)
    slope = float(lr_cal.coef_[0][0])
    intercept = float(lr_cal.intercept_[0])

    return brier, slope, intercept, oof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("BioFusionFormer E1 — Explainability, Stability, Representation & Calibration")
    log.info("=" * 70)

    (core_pats, N, y_hist, mats, masks, feat_dims, bio_bias,
     selected_pways, pway_summary,
     mask_mrna_np, mask_mirna_np, mask_meth_np,
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

    pos_weight = float((y_hist == 0).sum() / (y_hist == 1).sum())

    # ========================================================================
    # 4.1 + 4.2 + 4.5 per fold
    # ========================================================================
    log.info("\n4.1 / 4.2: Computing per-fold IG and elastic-net attributions...")

    ig_pway_per_fold = []
    ig_pway_per_fold_mrna = []
    ig_pway_per_fold_mirna = []
    ig_pway_per_fold_meth = []
    en_pway_per_fold = []
    top20_ig_per_fold = []
    top20_en_per_fold = []
    rollout_spreads = []

    for f_i, fold in enumerate(splits):
        log.info(f"  Fold {f_i+1}/5 ...")
        tr_idx = np.array(fold["train_indices"])
        te_idx = np.array(fold["test_indices"])
        y_tr = y_hist[tr_idx]

        X_tr, X_te = prepare_fold(tr_idx, te_idx, mats)

        mpath = MODELS_DIR / f"fold_{f_i:02d}_seed42.pt"
        model = build_model(feat_dims, masks, P, bio_bias)
        model.load_state_dict(torch.load(mpath, map_location="cpu"))
        model.eval()

        ig_mrna, ig_mirna, ig_meth = compute_ig(model, X_te, n_steps=args.n_steps)
        pway_mrna, pway_mirna, pway_meth, pway_sum = ig_to_pathway_scores(
            ig_mrna, ig_mirna, ig_meth, mask_mrna_np, mask_mirna_np, mask_meth_np
        )
        ig_pway_per_fold.append(pway_sum)
        ig_pway_per_fold_mrna.append(pway_mrna)
        ig_pway_per_fold_mirna.append(pway_mirna)
        ig_pway_per_fold_meth.append(pway_meth)
        top20_ig = [selected_pways[i] for i in np.argsort(pway_sum)[::-1][:20]]
        top20_ig_per_fold.append(top20_ig)

        with torch.no_grad():
            te_t = {m: torch.tensor(X_te[m], dtype=torch.float32) for m in E1_MODS}
            out = model(te_t)
            roll = compute_attention_rollout(out["attns"]).cpu().numpy()[:, 0, 1:]
        pw_att = roll.reshape(len(te_idx), 3, P).sum(axis=1).mean(axis=0)
        spread = float(pw_att.max() / max(pw_att.min(), 1e-12))
        rollout_spreads.append(spread)

        en_scores = compute_en_pathway_scores(X_tr["mrna"], y_tr, mask_mrna_np)
        en_pway_per_fold.append(en_scores)
        top20_en = [selected_pways[i] for i in np.argsort(en_scores)[::-1][:20]]
        top20_en_per_fold.append(top20_en)

        log.info(f"    Rollout spread: {spread:.3f}x | Top-3 IG: {top20_ig[:3]}")

    log.info(f"  Rollout mean max/min: {np.mean(rollout_spreads):.3f} "
             f"range=[{min(rollout_spreads):.3f}, {max(rollout_spreads):.3f}]")

    # ========================================================================
    # 4.3: Stability
    # ========================================================================
    log.info("\n4.3: Stability (mean pairwise Jaccard)...")
    ig_jaccard, ig_recurrent = compute_pathway_jaccard(top20_ig_per_fold)
    en_jaccard, en_recurrent = compute_pathway_jaccard(top20_en_per_fold)
    log.info(f"  IG Jaccard: {ig_jaccard:.4f} | EN Jaccard: {en_jaccard:.4f}")

    # ========================================================================
    # 4.4b: Frequency + generic fraction
    # ========================================================================
    log.info("\n4.4b: Pathway stability frequency and generic pathway check...")
    freq_counter = Counter()
    for t20 in top20_ig_per_fold:
        for p in set(t20):
            freq_counter[p] += 1
    freq_list = [(p, cnt / 5) for p, cnt in freq_counter.most_common()]

    n_generic = sum(
        1 for p in top20_ig_per_fold[0]
        if is_generic(pway_summary.loc[p, "name"] if p in pway_summary.index else "")
    )
    generic_fraction = n_generic / 20
    log.info(f"  Generic fraction fold-0 top-20: {generic_fraction:.2f}")
    log.info("  NOTE: translation/ribosome/splicing may reflect proliferation, tumour content "
             "or quality, not histology-specific biology.")

    # ========================================================================
    # 4.4: Matched null
    # ========================================================================
    null_jaccards = []
    if args.skip_null:
        log.info("\n4.4: Null permutation skipped (--skip-null).")
        n_perms_used = 0
    else:
        n_perms_used = args.n_perms
        if n_perms_used < 5:
            log.warning(f"Using {n_perms_used} permutations (< 5): will state this in the report.")
        log.info(f"\n4.4: Matched null ({n_perms_used} perms x 5 folds, same training procedure)...")
        rng_null = np.random.default_rng(0)
        for perm_i in range(n_perms_used):
            y_perm = rng_null.permutation(y_hist)
            top20_null_this_perm = []
            for f_i, fold in enumerate(splits):
                tr_idx = np.array(fold["train_indices"])
                te_idx = np.array(fold["test_indices"])
                y_tr_perm = y_perm[tr_idx]
                X_tr, X_te = prepare_fold(tr_idx, te_idx, mats)
                pos_w_perm = max(float((y_tr_perm == 0).sum() / max((y_tr_perm == 1).sum(), 1)), 0.1)
                seed_null = SEEDS[0] + perm_i * 100
                set_seed(seed_null)
                m_null = build_model(feat_dims, masks, P, bio_bias)
                m_null = train_one_seed(m_null, X_tr, y_tr_perm, pos_w_perm, seed_null, **HP)
                ig_mr, ig_mi, ig_me = compute_ig(m_null, X_te, n_steps=args.n_steps)
                _, _, _, pway_null = ig_to_pathway_scores(
                    ig_mr, ig_mi, ig_me, mask_mrna_np, mask_mirna_np, mask_meth_np
                )
                top20_null = [selected_pways[i] for i in np.argsort(pway_null)[::-1][:20]]
                top20_null_this_perm.append(top20_null)
            null_j, _ = compute_pathway_jaccard(top20_null_this_perm)
            null_jaccards.append(null_j)
            log.info(f"  Perm {perm_i+1}/{n_perms_used}: null Jaccard = {null_j:.4f}")
        if null_jaccards:
            log.info(f"  Null Jaccard: mean={np.mean(null_jaccards):.4f} ± {np.std(null_jaccards):.4f}")

    # ========================================================================
    # 4.5: IG vs EN agreement
    # ========================================================================
    log.info("\n4.5: IG vs EN agreement per fold...")
    ig_en_jaccards = []
    ig_en_spearmans = []
    for f_i in range(5):
        t20_ig = set(top20_ig_per_fold[f_i])
        t20_en = set(top20_en_per_fold[f_i])
        inter = len(t20_ig & t20_en)
        union = len(t20_ig | t20_en)
        j = inter / union if union > 0 else 0.0
        ig_en_jaccards.append(j)
        rho, _ = stats.spearmanr(ig_pway_per_fold[f_i], en_pway_per_fold[f_i])
        ig_en_spearmans.append(float(rho))
        log.info(f"  Fold {f_i+1}: Jaccard={j:.3f} | Spearman={rho:.3f}")
    log.info(f"  Mean Jaccard: {np.mean(ig_en_jaccards):.3f}  Mean Spearman: {np.mean(ig_en_spearmans):.3f}")

    # ========================================================================
    # 4.6: Representation
    # ========================================================================
    log.info("\n4.6: Patient representation (in-sample only, visualisation)...")
    try:
        import umap as umap_module
        _umap_available = True
    except ImportError:
        log.warning("umap-learn not installed; skipping UMAP.")
        _umap_available = False

    from sklearn.decomposition import PCA
    embs = train_all_patients_model(N, mats, y_hist, masks, feat_dims, P, bio_bias)

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
        f"PCA — in-sample, visualisation only\n"
        f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
        fontsize=8
    )
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    if _umap_available:
        reducer = umap_module.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embs_umap = reducer.fit_transform(embs)
        for label, col in [("Favorable", "#2b5c8f"), ("Unfavorable", "#c0392b")]:
            msk = [l == label for l in labels_text]
            axes[1].scatter(embs_umap[msk, 0], embs_umap[msk, 1], c=col, label=label, alpha=0.8, s=40, edgecolors="none")
        axes[1].set_title("UMAP — in-sample, visualisation only", fontsize=8)
        axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    fig.suptitle(
        "BioFusionFormer E1: Patient Representations\n"
        "(in-sample, all 121 patients, 3-seed average, coloured by histologic subtype)",
        fontsize=8, y=1.01
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "representation_pca_umap.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved figures/explain/representation_pca_umap.png")

    # ========================================================================
    # 4.7: Calibration
    # ========================================================================
    log.info("\n4.7: Calibration (pooled OOF from 25 frozen folds)...")
    brier, slope, intercept, oof = compute_calibration(y_hist, frozen_folds, N)
    log.info(f"  Brier: {brier:.4f} | Recalibration slope: {slope:.3f}, intercept: {intercept:.3f}")

    pt, pp_cal = calibration_curve(y_hist, oof, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=180)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(pp_cal, pt, "s-", color="#2980b9", lw=2, label=f"E1 pooled OOF (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted P(Unfavorable histology)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("E1 Reliability Curve (5-quantile bins; N=121 is noisy)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_reliability.png")
    plt.close(fig)
    log.info("  Saved figures/explain/calibration_reliability.png")

    # ========================================================================
    # Pathway attribution bar charts
    # ========================================================================
    mean_ig_pway = np.mean(ig_pway_per_fold, axis=0)
    top20_idx = np.argsort(mean_ig_pway)[::-1][:20]
    top20_ids = [selected_pways[i] for i in top20_idx]
    top20_names_raw = [
        pway_summary.loc[p, "name"][:55] if p in pway_summary.index else p for p in top20_ids
    ]
    top20_names = top20_names_raw[::-1]
    top20_vals = mean_ig_pway[top20_idx][::-1]
    bar_colors = ["#c0392b" if is_generic(n) else "#16a085" for n in top20_names]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    y_pos = np.arange(len(top20_names))
    ax.barh(y_pos, top20_vals, color=bar_colors, alpha=0.85)
    ax.set_yticks(y_pos); ax.set_yticklabels(top20_names, fontsize=8)
    ax.set_xlabel("Mean |IG| pathway attribution (summed over modalities, 5-fold average)")
    ax.set_title(
        "Top 20 Pathways by Integrated Gradients (mean over 5 folds)\n"
        "Red = generic (translation/ribosome/splicing); may reflect proliferation or quality",
        fontsize=9
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ig_top20_pathways.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved figures/explain/ig_top20_pathways.png")

    mean_en_pway = np.mean(en_pway_per_fold, axis=0)
    common_set = (
        set(np.argsort(mean_ig_pway)[::-1][:15].tolist()) |
        set(np.argsort(mean_en_pway)[::-1][:15].tolist())
    )
    common_idx = sorted(list(common_set))[:20]
    c_names_raw = [
        pway_summary.loc[selected_pways[i], "name"][:50] if selected_pways[i] in pway_summary.index
        else selected_pways[i] for i in common_idx
    ]
    c_names = c_names_raw[::-1]
    ig_v = (mean_ig_pway[common_idx] / max(mean_ig_pway[common_idx].max(), 1e-12))[::-1]
    en_v = (mean_en_pway[common_idx] / max(mean_en_pway[common_idx].max(), 1e-12))[::-1]
    w = 0.35
    y_pos_c = np.arange(len(c_names))

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    ax.barh(y_pos_c - w / 2, ig_v, w, color="#2980b9", alpha=0.85, label="BioFusionFormer IG")
    ax.barh(y_pos_c + w / 2, en_v, w, color="#e67e22", alpha=0.85, label="ElasticNet (train-fold mRNA only)")
    ax.set_yticks(y_pos_c); ax.set_yticklabels(c_names, fontsize=8)
    ax.set_xlabel("Normalised attribution score")
    ax.set_title(
        f"IG vs ElasticNet Pathway Attribution (5-fold mean)\n"
        f"Jaccard top-20: mean={np.mean(ig_en_jaccards):.3f} | Spearman: mean={np.mean(ig_en_spearmans):.3f}",
        fontsize=9
    )
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ig_vs_en_pathways.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved figures/explain/ig_vs_en_pathways.png")

    # ========================================================================
    # JSON output
    # ========================================================================
    out_json = {
        "n_folds": 5,
        "n_ig_steps": args.n_steps,
        "ig_stability": {
            "mean_pairwise_jaccard": float(ig_jaccard),
            "recurrent_pathways_80pct": [(p, float(fq)) for p, fq in ig_recurrent],
        },
        "en_stability": {
            "mean_pairwise_jaccard": float(en_jaccard),
            "recurrent_pathways_80pct": [(p, float(fq)) for p, fq in en_recurrent],
        },
        "null_stability": {
            "n_perms_used": n_perms_used,
            "mean_jaccard": float(np.mean(null_jaccards)) if null_jaccards else None,
            "std_jaccard": float(np.std(null_jaccards)) if null_jaccards else None,
            "note": (
                f"Matched null: same E1 training procedure (max_epochs=100, patience=20). "
                f"{n_perms_used} permutation(s) used."
                + (" NOTE: fewer than 5 perms used; state in report." if 0 < n_perms_used < 5 else "")
            ),
        },
        "ig_vs_en_per_fold": {
            "jaccard_top20": [float(j) for j in ig_en_jaccards],
            "spearman_all192": [float(r) for r in ig_en_spearmans],
            "mean_jaccard": float(np.mean(ig_en_jaccards)),
            "mean_spearman": float(np.mean(ig_en_spearmans)),
        },
        "attention_rollout_secondary": {
            "max_min_ratios_per_fold": [float(s) for s in rollout_spreads],
            "mean": float(np.mean(rollout_spreads)),
            "note": "Rollout near 1x indicates near-uniform attention; not used as primary explanation.",
        },
        "calibration": {
            "brier_pooled_oof_25folds": float(brier),
            "logistic_recalibration_slope": float(slope),
            "logistic_recalibration_intercept": float(intercept),
            "note": (
                "Calibration uses all 25 frozen fold OOF predictions averaged over repeats. "
                "Logistic recalibration (Platt scaling) slope: >1 = overconfident, <1 = underconfident."
            ),
        },
        "top20_ig_per_fold": [list(t) for t in top20_ig_per_fold],
        "top20_en_per_fold": [list(t) for t in top20_en_per_fold],
        "pathway_freq_across_5_folds": freq_list[:30],
        "generic_fraction_fold0_top20": float(generic_fraction),
        "generic_note": (
            "Pathways flagged as generic (translation/ribosome/splicing keywords). "
            "May reflect proliferation rate, tumour cell content, or sequencing quality "
            "rather than histology-specific biology."
        ),
    }

    with open(OUT_DIR / "explainability_results.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    log.info("Saved results/explainability/explainability_results.json")

    # ========================================================================
    # Markdown summary
    # ========================================================================
    L = []
    L.append("# BioFusionFormer E1 — Explainability Summary (Stage 4)")
    L.append("")
    L.append("> **Scope**: 5 fold models from the Stage 3 reproducibility re-run "
             "(seed 42, `models/e1_folds_final_v1_repro/`). "
             "Attributions computed on TEST patients of each fold only.")
    L.append("")
    L.append("## 4.1 Integrated Gradients (Primary Attribution)")
    L.append("")
    L.append(f"- n_steps = {args.n_steps}; baseline = zero vector = training-fold mean in standardised space.")
    L.append("- Computed independently for all three modalities; aggregated to 192 pathways via the same prior masks as the model.")
    L.append("")
    L.append("### Top-20 pathways by mean |IG| (summed over modalities, 5-fold mean)")
    L.append("")
    for rank_i, (pg_i, pid) in enumerate(zip(top20_idx, top20_ids), 1):
        pname = pway_summary.loc[pid, "name"] if pid in pway_summary.index else pid
        gen_tag = " *(generic)*" if is_generic(pname) else ""
        L.append(f"{rank_i}. {pname} (`{pid}`){gen_tag}: {mean_ig_pway[pg_i]:.5f}")
    L.append("")
    L.append("### Per-modality top-3 pathways (5-fold mean)")
    L.append("")
    for mod_name, plist in [("mRNA", ig_pway_per_fold_mrna),
                              ("miRNA", ig_pway_per_fold_mirna),
                              ("Methylation", ig_pway_per_fold_meth)]:
        mean_m = np.mean(plist, axis=0)
        top3ids = [selected_pways[i] for i in np.argsort(mean_m)[::-1][:3]]
        top3names = [pway_summary.loc[p, "name"][:60] if p in pway_summary.index else p for p in top3ids]
        L.append(f"**{mod_name}**: " + "; ".join(top3names))
    L.append("")
    L.append("### Attention rollout (secondary reference only)")
    L.append("")
    L.append(f"Mean max/min ratio: **{np.mean(rollout_spreads):.3f}x** "
             f"(range [{min(rollout_spreads):.3f}, {max(rollout_spreads):.3f}]). "
             "A ratio near 1 means near-uniform attention; rollout is not a reliable explanation here.")
    L.append("")
    L.append("## 4.2 Elastic-Net Baseline")
    L.append("")
    L.append("Logistic regression (saga, C=0.5, l1_ratio=0.5, max_iter=500) fitted on "
             "mRNA training fold only. Coefficients mapped to pathways via the same mask. "
             "**Never fitted on all 121 patients.**")
    L.append("")
    L.append("## 4.3 Stability")
    L.append("")
    L.append("| Attribution | Mean pairwise Jaccard (top-20, 5 folds) |")
    L.append("|---|---|")
    L.append(f"| IG (BioFusionFormer) | {ig_jaccard:.4f} |")
    L.append(f"| ElasticNet (mRNA, train-fold) | {en_jaccard:.4f} |")
    if null_jaccards:
        L.append(f"| Matched null (permuted labels) | {np.mean(null_jaccards):.4f} ± {np.std(null_jaccards):.4f} |")
    else:
        L.append("| Matched null | skipped |")
    L.append("")
    L.append("## 4.4 Matched Null Permutation")
    L.append("")
    if args.skip_null:
        L.append("Null permutation was skipped (`--skip-null`). Re-run without this flag to include.")
    elif n_perms_used < 5:
        L.append(f"**{n_perms_used} permutation(s)** used (< 5) due to compute constraints; "
                 "explicitly stated here.")
        if null_jaccards:
            L.append(f"Null mean Jaccard = {np.mean(null_jaccards):.4f} ± {np.std(null_jaccards):.4f}.")
    else:
        L.append(f"5 permutations × 5 folds; same E1 training procedure (max_epochs=100, patience=20).")
        L.append(f"Null mean Jaccard = {np.mean(null_jaccards):.4f} ± {np.std(null_jaccards):.4f}.")
    L.append("")
    L.append("## 4.4b Generic Pathway Fraction")
    L.append("")
    L.append(f"Fraction of fold-0 top-20 classified as generic (translation/ribosome/splicing): "
             f"**{generic_fraction:.0%}**.")
    L.append("These pathways may reflect global proliferation rate, tumour cell content, or "
             "sequencing quality rather than histology-specific biology. They are retained in ranked "
             "lists but flagged with *(generic)* above.")
    L.append("")
    L.append("## 4.5 IG vs ElasticNet Agreement")
    L.append("")
    L.append("| Fold | Jaccard (top-20) | Spearman ρ (all 192) |")
    L.append("|---|---|---|")
    for fi in range(5):
        L.append(f"| {fi+1} | {ig_en_jaccards[fi]:.3f} | {ig_en_spearmans[fi]:.3f} |")
    L.append(f"| **Mean** | **{np.mean(ig_en_jaccards):.3f}** | **{np.mean(ig_en_spearmans):.3f}** |")
    L.append(f"| **Range** | [{min(ig_en_jaccards):.3f}, {max(ig_en_jaccards):.3f}] "
             f"| [{min(ig_en_spearmans):.3f}, {max(ig_en_spearmans):.3f}] |")
    L.append("")
    L.append("## 4.6 Patient Representation")
    L.append("")
    L.append("One model trained on all 121 patients (3 seeds, embeddings averaged). "
             "**In-sample, visualisation only. No silhouette or separation claims.**")
    L.append("See `figures/explain/representation_pca_umap.png`.")
    L.append("")
    L.append("## 4.7 Calibration")
    L.append("")
    L.append("Pooled OOF predictions from the 25 frozen E1 folds "
             "(averaged over repeats where a patient appears more than once).")
    L.append("")
    L.append(f"- **Brier score**: {brier:.4f}")
    L.append(f"- **Logistic recalibration (Platt scaling)**: slope = {slope:.3f}, "
             f"intercept = {intercept:.3f}")
    L.append("  - Slope > 1 = overconfident; slope < 1 = underconfident; ideal = 1.0.")
    L.append("  - With N = 121 recalibration estimates are themselves uncertain.")
    L.append("")
    L.append("See `figures/explain/calibration_reliability.png`.")
    L.append("")
    L.append("## Known Limitations")
    L.append("")
    L.append("- Only 5 fold models (1 repeat) used; attribution estimates are noisy (N_test ≈ 25).")
    L.append("- Jaccard stability values below 0.3 are common at this sample size.")
    L.append("- E1 was not distinguishable from the elastic-net baseline (Nadeau-Bengio corrected test).")
    L.append("- Fork 6: 726/1035 miRNA features connect to pathway P-1 (last Reactome pathway), "
             "not a dedicated unmapped token.")
    L.append("")

    (OUT_DIR / "explainability_summary.md").write_text("\n".join(L), encoding="utf-8")
    log.info("Saved results/explainability/explainability_summary.md")

    log.info("\n" + "=" * 70)
    log.info("Stage 4 COMPLETE. Outputs written to:")
    log.info(f"  {OUT_DIR / 'explainability_results.json'}")
    log.info(f"  {OUT_DIR / 'explainability_summary.md'}")
    log.info(f"  {FIG_DIR / 'ig_top20_pathways.png'}")
    log.info(f"  {FIG_DIR / 'ig_vs_en_pathways.png'}")
    log.info(f"  {FIG_DIR / 'representation_pca_umap.png'}")
    log.info(f"  {FIG_DIR / 'calibration_reliability.png'}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
