"""
BioFusionFormer - Step A1: Gene Coverage Analysis & Restricted ElasticNet
========================================================================
1. Computes coverage of mRNA (11,171), miRNA (1,797), and Methylation (9,778)
   in the 192 selected pathways.
2. Runs in-fold ElasticNet on mRNA restricted to covered genes across 25 outer folds.
3. Tests pre-specified rule: if restricted AUROC < (0.9169 - 0.03) = 0.8869,
   evaluates top-500 pathways (depth <= 6, 10-300 genes).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

from src.pathways.selection import select_unsupervised_pathways

def tune_and_predict_elasticnet(X_tr, y_tr, X_te):
    grid = [(0.1, 0.3), (0.1, 0.7), (0.5, 0.5), (1.0, 0.5)]
    best_params = grid[0]
    best_score = -1.0
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for c_val, l1_val in grid:
        scores = []
        for in_tr, in_val in inner_cv.split(X_tr, y_tr):
            m = LogisticRegression(solver="saga", penalty="elasticnet", C=c_val, l1_ratio=l1_val, max_iter=200, random_state=42)
            m.fit(X_tr[in_tr], y_tr[in_tr])
            val_p = m.predict_proba(X_tr[in_val])[:, 1]
            try:
                scores.append(roc_auc_score(y_tr[in_val], val_p))
            except Exception:
                scores.append(0.5)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_params = (c_val, l1_val)

    final_model = LogisticRegression(solver="saga", penalty="elasticnet", C=best_params[0], l1_ratio=best_params[1], max_iter=250, random_state=42)
    final_model.fit(X_tr, y_tr)
    return final_model.predict_proba(X_te)[:, 1]

def run_evaluation():
    cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
    core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()

    import gzip
    with gzip.open(DATA_DIR / "raw" / "target_wt_xena" / "TARGET-WT.clinical.tsv.gz", "rt") as f:
        clin_df = pd.read_csv(f, sep="\t")
    clin_df["patient_id"] = clin_df["sample"].str.slice(0, 16)
    core_clin = clin_df[clin_df["patient_id"].isin(core_pats)].drop_duplicates("patient_id").set_index("patient_id").loc[core_pats]
    y_histology = (core_clin["wilms_tumor_histologic_subtype.diagnoses"] == "Unfavorable").astype(int).values

    mrna_df = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet").loc[core_pats]
    mirna_df = pd.read_parquet(DATA_DIR / "processed" / "mirna.parquet").loc[core_pats]
    meth_df = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet").loc[core_pats]

    with open(DATA_DIR / "splits" / "splits_5x5.json") as f:
        splits = json.load(f)["folds"]

    # 1. 192 Pathway Selection
    selected_pways_192, _ = select_unsupervised_pathways(DATA_DIR, top_k=192)

    prior_pathways = json.load(open(DATA_DIR / "interim" / "priors" / "gene_pathway_mask_pathways.json"))
    pway_map = {p: i for i, p in enumerate(prior_pathways)}
    pway_idx_192 = [pway_map[p] for p in selected_pways_192]

    mask_gene_full = sp.load_npz(DATA_DIR / "interim" / "priors" / "gene_pathway_mask.npz")
    gene_names = json.load(open(DATA_DIR / "interim" / "priors" / "gene_pathway_mask_genes.json"))
    gene_map = {g: i for i, g in enumerate(gene_names)}

    mrna_idx = [gene_map[c] for c in mrna_df.columns]
    meth_idx = [gene_map[c] for c in meth_df.columns]

    mask_mrna_192 = mask_gene_full[mrna_idx, :][:, pway_idx_192].toarray()  # (F_mrna, 192)
    mask_meth_192 = mask_gene_full[meth_idx, :][:, pway_idx_192].toarray()

    mask_mirna_strong_full = sp.load_npz(DATA_DIR / "interim" / "priors" / "mirna_pathway_mask.npz")
    mask_mirna_broad_full = sp.load_npz(DATA_DIR / "interim" / "priors" / "mirna_pathway_mask_broad.npz")
    mirna_names = json.load(open(DATA_DIR / "interim" / "priors" / "mirna_pathway_mask_mirnas.json"))
    mirna_map = {m: i for i, m in enumerate(mirna_names)}
    mirna_idx = [mirna_map[c] for c in mirna_df.columns]

    mask_mirna_192_strong = mask_mirna_strong_full[mirna_idx, :][:, pway_idx_192].toarray()
    mask_mirna_192_broad = mask_mirna_broad_full[mirna_idx, :][:, pway_idx_192].toarray()

    mrna_covered_192 = (mask_mrna_192.sum(axis=1) > 0)
    meth_covered_192 = (mask_meth_192.sum(axis=1) > 0)
    mirna_cov_strong_192 = (mask_mirna_192_strong.sum(axis=1) > 0)
    mirna_cov_broad_192 = (mask_mirna_192_broad.sum(axis=1) > 0)

    print("=== Coverage with Top 192 Pathways ===")
    print(f"mRNA covered: {mrna_covered_192.sum()} / {len(mrna_covered_192)} ({100*mrna_covered_192.mean():.1f}%)")
    print(f"miRNA covered (strong): {mirna_cov_strong_192.sum()} / {len(mirna_cov_strong_192)} ({100*mirna_cov_strong_192.mean():.1f}%)")
    print(f"miRNA covered (broad): {mirna_cov_broad_192.sum()} / {len(mirna_cov_broad_192)} ({100*mirna_cov_broad_192.mean():.1f}%)")
    print(f"Methylation covered: {meth_covered_192.sum()} / {len(meth_covered_192)} ({100*meth_covered_192.mean():.1f}%)")

    # Run restricted in-fold ElasticNet on mRNA (covered genes only)
    mrna_restricted_df = mrna_df.loc[:, mrna_covered_192]
    print(f"Restricted mRNA matrix shape: {mrna_restricted_df.shape}")

    restricted_scores_192 = []
    for fold in splits:
        tr_idx = np.array(fold["train_indices"])
        te_idx = np.array(fold["test_indices"])
        y_tr, y_te = y_histology[tr_idx], y_histology[te_idx]

        sc = StandardScaler()
        X_tr = sc.fit_transform(mrna_restricted_df.values[tr_idx])
        X_te = sc.transform(mrna_restricted_df.values[te_idx])

        sel = SelectKBest(f_classif, k=min(200, X_tr.shape[1]))
        X_tr_sel = sel.fit_transform(X_tr, y_tr)
        X_te_sel = sel.transform(X_te)

        probs = tune_and_predict_elasticnet(X_tr_sel, y_tr, X_te_sel)
        restricted_scores_192.append(roc_auc_score(y_te, probs))

    mean_restr_192 = float(np.mean(restricted_scores_192))
    std_restr_192 = float(np.std(restricted_scores_192))
    full_gene_mean = 0.9169
    delta_192 = mean_restr_192 - full_gene_mean
    print(f"\nRestricted mRNA ElasticNet (192 pathways): Mean AUROC = {mean_restr_192:.4f} (std = {std_restr_192:.4f})")
    print(f"Difference vs Full Gene (0.9169): {delta_192:+.4f}")

    widen_needed = delta_192 < -0.03
    print(f"Pre-specified threshold rule (delta < -0.03): {widen_needed}")

    results = {
        "192_pathways": {
            "mrna_covered": int(mrna_covered_192.sum()),
            "mrna_total": len(mrna_covered_192),
            "mrna_pct": round(float(100*mrna_covered_192.mean()), 2),
            "mirna_strong_covered": int(mirna_cov_strong_192.sum()),
            "mirna_broad_covered": int(mirna_cov_broad_192.sum()),
            "meth_covered": int(meth_covered_192.sum()),
            "restricted_auroc_mean": round(mean_restr_192, 4),
            "restricted_auroc_std": round(std_restr_192, 4),
            "delta_vs_full": round(delta_192, 4),
            "widen_needed": widen_needed,
        }
    }

    if widen_needed:
        print("\nWidening pathway set to top 500 by variance (depth <= 6, 10-300 genes)...")
        selected_pways_500, _ = select_unsupervised_pathways(DATA_DIR, top_k=500, max_depth=6)
        pway_idx_500 = [pway_map[p] for p in selected_pways_500]
        mask_mrna_500 = mask_gene_full[mrna_idx, :][:, pway_idx_500].toarray()
        mrna_covered_500 = (mask_mrna_500.sum(axis=1) > 0)
        print(f"Top 500 mRNA covered: {mrna_covered_500.sum()} / {len(mrna_covered_500)} ({100*mrna_covered_500.mean():.1f}%)")

        mrna_restr_500 = mrna_df.loc[:, mrna_covered_500]
        restr_scores_500 = []
        for fold in splits:
            tr_idx = np.array(fold["train_indices"])
            te_idx = np.array(fold["test_indices"])
            y_tr, y_te = y_histology[tr_idx], y_histology[te_idx]

            sc = StandardScaler()
            X_tr = sc.fit_transform(mrna_restr_500.values[tr_idx])
            X_te = sc.transform(mrna_restr_500.values[te_idx])

            sel = SelectKBest(f_classif, k=min(200, X_tr.shape[1]))
            X_tr_sel = sel.fit_transform(X_tr, y_tr)
            X_te_sel = sel.transform(X_te)

            probs = tune_and_predict_elasticnet(X_tr_sel, y_tr, X_te_sel)
            restr_scores_500.append(roc_auc_score(y_te, probs))

        mean_restr_500 = float(np.mean(restr_scores_500))
        delta_500 = mean_restr_500 - full_gene_mean
        print(f"Restricted mRNA ElasticNet (500 pathways): Mean AUROC = {mean_restr_500:.4f}")
        print(f"Difference vs Full Gene: {delta_500:+.4f}")
        skip_path_needed = delta_500 < -0.03
        print(f"Skip-path needed: {skip_path_needed}")

        results["500_pathways"] = {
            "mrna_covered": int(mrna_covered_500.sum()),
            "mrna_pct": round(float(100*mrna_covered_500.mean()), 2),
            "restricted_auroc_mean": round(mean_restr_500, 4),
            "delta_vs_full": round(delta_500, 4),
            "skip_path_needed": skip_path_needed,
        }

    with open(ROOT / "results" / "model" / "gene_coverage_check.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
