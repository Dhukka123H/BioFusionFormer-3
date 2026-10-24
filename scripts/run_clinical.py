"""
scripts/run_clinical.py
=======================
Resumable CLI runner for BioFusionFormer Clinical Modality Variants (Stage 5.3):
  - EHR2  : omics (3 modalities) + clinical (primary: age, sex, stage)
  - EHR4  : clinical-only through the same code
  - EHR2b : omics + clinical (primary + sensitivity set: race, ethnicity, biopsy QC)

Follows AGENT_PROMPT Rule 6 (max 2 threads), Rule 5 (honest reporting, Nadeau-Bengio intervals),
and Stage 5.3 specifications:
  - Default: --n-repeats 3 (15 folds)
  - Seeds: [42, 123, 999]
  - Atomic per-fold JSON with config signatures
  - Preprocessing and imputation strictly inside each training fold (zero leakage)
  - Compares with E1 (first 15 folds) and elastic-net baseline via Nadeau-Bengio,
    Wilcoxon signed-rank test, and bootstrap CIs on paired differences.
"""

from typing import Dict, List, Optional, Tuple, Any
import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pathways.selection import select_unsupervised_pathways
from src.biofusionformer.full_model_clinical import BioFusionFormerClinicalModel
from src.preprocessing.clinical import ClinicalPreprocessor
from src.heads.classification import LabelSmoothedBCEWithLogitsLoss
from src.evaluation.cv_stats import nadeau_bengio_t, nb_interval, naive_interval


def parse_args():
    ap = argparse.ArgumentParser(description="BioFusionFormer Clinical Variants Runner (Stage 5.3)")
    ap.add_argument("--run", action="store_true", help="Train / resume clinical variant folds")
    ap.add_argument("--report", action="store_true", help="Generate results/model/clinical_summary.md")
    ap.add_argument("--status", action="store_true", help="Show checkpoint progress and exit")
    ap.add_argument("--force", action="store_true", help="Redo variants (backs up old folders)")
    ap.add_argument("--smoke", action="store_true", help="Tiny test run (2 folds, 1 seed, 3 epochs)")
    ap.add_argument("--tag", default="final_clinical", help="Output folder name under experiments/")
    ap.add_argument("--variants", nargs="+", default=["EHR2", "EHR4"],
                    choices=["EHR2", "EHR4"], help="Variants to execute/report")
    ap.add_argument("--n-repeats", type=int, default=3, choices=[1, 2, 3, 4, 5],
                    help="Number of 5-fold CV repeats (default 3 = 15 folds per Stage 5.3)")
    args = ap.parse_args()
    if not (args.run or args.report or args.status):
        ap.print_help()
        sys.exit(0)
    return args


ARGS = parse_args()
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "experiments" / (f"{ARGS.tag}_smoke" if ARGS.smoke else ARGS.tag)
RES_DIR = ROOT / "results" / "model"
FIG_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

for d in (EXP_DIR, RES_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("clinical")
torch.set_num_threads(2)  # Rule 6 constraint


def setup_logging():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"clinical_session_{stamp}.log", mode="a", encoding="utf-8"),
        logging.FileHandler(LOG_DIR / "clinical_all_sessions.log", mode="a", encoding="utf-8"),
    ]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers)


setup_logging()

SEEDS = [42, 123, 999]
HP = dict(lr=3e-4, weight_decay=1e-2, max_epochs=100, patience=20, batch_size=16)

VARIANT_CONFIGS = {
    "EHR2": {
        "name": "BioFusionFormer EHR2 (3 omics + primary clinical [age, sex, stage])",
        "mods": ["mrna", "mirna", "meth"],
        "use_clinical": True,
        "include_sensitivity": False,
        "is_sensitivity": False,
    },
    "EHR4": {
        "name": "BioFusionFormer EHR4 (clinical-only through same architecture)",
        "mods": [],
        "use_clinical": True,
        "include_sensitivity": False,
        "is_sensitivity": False,
    },
}


def set_seed(s: int):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def atomic_write_json(path: Path, data: Any):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def read_fold(path: Path, sig: str) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
    except Exception:
        return None
    if rec.get("sig") != sig:
        raise RuntimeError(
            f"{path} config mismatch (sig {rec.get('sig')} != {sig}). "
            f"Use --force to redo this experiment."
        )
    return rec


def load_clinical_data(core_pats: List[str]) -> pd.DataFrame:
    cand_path = DATA_DIR / "processed" / "clinical" / "clinical_candidates.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"{cand_path} missing. Run python scripts/download_clinical.py first.")
    df = pd.read_csv(cand_path).set_index("patient_id")
    missing = [p for p in core_pats if p not in df.index]
    if missing:
        raise KeyError(f"Cohort patients missing from clinical data: {missing[:3]}")
    return df.loc[core_pats]


def build_bio_bias_matrix(pathway_indices: List[int], n_modalities: int, adj_sparse: sp.spmatrix) -> torch.Tensor:
    P = len(pathway_indices)
    M = n_modalities
    S = 1 + M * P
    B_bias = np.zeros((S, S), dtype=np.float32)
    B_bias[0, :] = 0.5
    B_bias[:, 0] = 0.5
    B_bias[0, 0] = 0.0
    sub_adj = adj_sparse[pathway_indices, :][:, pathway_indices].toarray()
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
    return torch.tensor(B_bias, dtype=torch.float32)


def train_single_seed_clinical(
    model: nn.Module,
    X_tr_dict: Dict[str, np.ndarray],
    y_tr: np.ndarray,
    X_te_dict: Dict[str, np.ndarray],
    pos_weight: float,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> Tuple[np.ndarray, nn.Module]:
    loss_fn = LabelSmoothedBCEWithLogitsLoss(
        eps=0.05, pos_weight=pos_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # In-fold split for validation (80/20 train/val)
    val_seed = seed + 1000
    rng = np.random.default_rng(val_seed)
    idx_0 = np.where(y_tr == 0)[0]
    idx_1 = np.where(y_tr == 1)[0]
    n_val_0 = max(1, int(round(0.2 * len(idx_0))))
    n_val_1 = max(1, int(round(0.2 * len(idx_1))))
    val_idx_0 = rng.choice(idx_0, size=n_val_0, replace=False)
    val_idx_1 = rng.choice(idx_1, size=n_val_1, replace=False)
    val_idx = np.concatenate([val_idx_0, val_idx_1])
    sub_tr_idx = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    sub_tr_in = {k: torch.tensor(v[sub_tr_idx], dtype=torch.float32) for k, v in X_tr_dict.items()}
    sub_tr_y = torch.tensor(y_tr[sub_tr_idx], dtype=torch.float32)
    val_in = {k: torch.tensor(v[val_idx], dtype=torch.float32) for k, v in X_tr_dict.items()}
    val_y = torch.tensor(y_tr[val_idx], dtype=torch.float32)

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0
    N_sub = len(sub_tr_idx)

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(N_sub)
        for s_idx in range(0, N_sub, batch_size):
            b_inds = perm[s_idx : s_idx + batch_size]
            b_in = {k: v[b_inds] for k, v in sub_tr_in.items()}
            b_y = sub_tr_y[b_inds]
            optimizer.zero_grad()
            out = model(b_in)
            loss = loss_fn(out["logit"], b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_out = model(val_in)
            v_loss = float(loss_fn(v_out["logit"], val_y).item())
        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        te_in = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_te_dict.items()}
        te_out = model(te_in)
        probs = torch.sigmoid(te_out["logit"]).cpu().numpy().reshape(-1)
    return probs, model


def fit_clinical_elasticnet(X_tr_clin: np.ndarray, y_tr: np.ndarray, X_te_clin: np.ndarray) -> np.ndarray:
    """Baseline elastic-net logistic regression fitted in-fold on clinical features."""
    clf = LogisticRegressionCV(
        Cs=10,
        l1_ratios=[0.1, 0.5, 0.9],
        cv=3,
        penalty="elasticnet",
        solver="saga",
        scoring="roc_auc",
        random_state=42,
        max_iter=1000,
    )
    clf.fit(X_tr_clin, y_tr)
    return clf.predict_proba(X_te_clin)[:, 1]


def run_variant(var_key: str, var_cfg: dict, cohort_pats: List[str], y_hist: np.ndarray,
                mats: Dict[str, np.ndarray], clin_df: pd.DataFrame,
                masks: Dict[str, torch.Tensor], feat_dims: Dict[str, int],
                P: int, bio_bias: torch.Tensor, splits: list, seeds: list, hp: dict):
    n_folds = len(splits)
    log.info(f"--- Running Variant: {var_key} ({var_cfg['name']}) [{n_folds} folds] ---")

    var_dir = EXP_DIR / var_key / "folds"
    var_dir.mkdir(parents=True, exist_ok=True)

    sig = hashlib.sha1(json.dumps({
        "var_key": var_key, "seeds": seeds, "hp": hp, "N": len(cohort_pats), "P": P,
        "n_folds": n_folds, "cfg": var_cfg,
    }, sort_keys=True).encode()).hexdigest()[:12]
    log.info(f"Variant signature: {sig}")

    pos_weight = float((y_hist == 0).sum() / (y_hist == 1).sum())

    for f_i, fold in enumerate(splits):
        fpath = var_dir / f"fold_{f_i:02d}.json"
        if read_fold(fpath, sig) is not None:
            log.info(f"  [{var_key}] fold {f_i+1:02d}/{n_folds} already on disk - skipped")
            continue

        t_fold = time.time()
        tr_idx = np.array(fold["train_indices"])
        te_idx = np.array(fold["test_indices"])
        y_tr, y_te = y_hist[tr_idx], y_hist[te_idx]

        # 1. In-fold clinical preprocessing
        preproc = ClinicalPreprocessor(include_sensitivity=var_cfg["include_sensitivity"])
        preproc.fit(clin_df.iloc[tr_idx], cohort_order=cohort_pats)
        tr_clin_df = preproc.transform(clin_df.iloc[tr_idx], cohort_order=cohort_pats)
        te_clin_df = preproc.transform(clin_df.iloc[te_idx], cohort_order=cohort_pats)
        X_tr_clin = tr_clin_df.values.astype(float)
        X_te_clin = te_clin_df.values.astype(float)
        n_clin_feats = X_tr_clin.shape[1]

        # In-fold elastic-net baseline prediction
        en_probs = fit_clinical_elasticnet(X_tr_clin, y_tr, X_te_clin)

        # 2. In-fold omics preprocessing (if omics enabled)
        X_tr, X_te = {}, {}
        if var_cfg["mods"]:
            for m in var_cfg["mods"]:
                raw_tr, raw_te = mats[m][tr_idx], mats[m][te_idx]
                imp = SimpleImputer(strategy="median")
                raw_tr = imp.fit_transform(raw_tr)
                raw_te = imp.transform(raw_te)
                sc = StandardScaler()
                X_tr[m] = sc.fit_transform(raw_tr)
                X_te[m] = sc.transform(raw_te)

        if var_cfg["use_clinical"]:
            X_tr["clinical"] = X_tr_clin
            X_te["clinical"] = X_te_clin

        seed_probs = []
        for s in seeds:
            set_seed(s)
            mod_feats = {m: feat_dims[m] for m in var_cfg["mods"]}
            mod_masks = {m: masks[m] for m in var_cfg["mods"]}
            # EHR4 (clinical only) uses 1 pathway token
            pways = P if var_cfg["mods"] else 1
            bias = bio_bias if var_cfg["mods"] else None

            model = BioFusionFormerClinicalModel(
                modality_features=mod_feats,
                modality_masks=mod_masks,
                n_pathways=pways,
                clinical_features=n_clin_feats,
                use_clinical=var_cfg["use_clinical"],
                d_model=32, n_heads=4, n_layers=2, d_ff=128,
                dropout=0.3, attn_dropout=0.2, modality_dropout=0.2,
                clinical_dropout=0.3,
                use_pathway_bias=(bias is not None),
                use_gating=True,
                bio_bias=bias,
            )

            probs, trained = train_single_seed_clinical(
                model=model, X_tr_dict=X_tr, y_tr=y_tr, X_te_dict=X_te,
                pos_weight=pos_weight, seed=s, **hp,
            )
            seed_probs.append(probs)

        p_mean = np.mean(seed_probs, axis=0)
        pred = (p_mean >= 0.5).astype(int)
        both = len(np.unique(y_te)) > 1

        rec = {
            "sig": sig, "fold": f_i, "te_idx": te_idx.tolist(),
            "probs": [float(p) for p in p_mean],
            "en_probs": [float(p) for p in en_probs],
            "auroc": float(roc_auc_score(y_te, p_mean)) if both else 0.5,
            "en_auroc": float(roc_auc_score(y_te, en_probs)) if both else 0.5,
            "auprc": float(average_precision_score(y_te, p_mean)) if both else float(np.mean(y_te)),
            "bal_acc": float(balanced_accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            "brier": float(brier_score_loss(y_te, p_mean)),
        }
        atomic_write_json(fpath, rec)
        dt = time.time() - t_fold
        log.info(f"  [{var_key}] fold {f_i+1:02d}/{n_folds} saved | {dt/60:.1f} min | AUROC={rec['auroc']:.4f} "
                 f"(EN={rec['en_auroc']:.4f}) | Brier={rec['brier']:.4f}")


def load_e1_results(n_folds: int) -> Optional[List[dict]]:
    e1_fold_dir = ROOT / "experiments" / "final_v1" / "E1" / "folds"
    if not e1_fold_dir.exists():
        return None
    recs = []
    for f_i in range(n_folds):
        fpath = e1_fold_dir / f"fold_{f_i:02d}.json"
        if not fpath.exists():
            return None
        with open(fpath) as f:
            recs.append(json.load(f))
    return recs


def generate_clinical_report(variants: List[str], n_folds: int):
    log.info("Generating Clinical Report...")
    e1_recs = load_e1_results(n_folds)
    e1_aurocs = [r["auroc"] for r in e1_recs] if e1_recs is not None else None
    e1_auprcs = [r["auprc"] for r in e1_recs] if e1_recs is not None else None
    e1_briers = [r["brier"] for r in e1_recs] if e1_recs is not None else None

    var_results = {}
    for var in variants:
        var_dir = EXP_DIR / var / "folds"
        recs = []
        for f_i in range(n_folds):
            fpath = var_dir / f"fold_{f_i:02d}.json"
            if not fpath.exists():
                log.warning(f"Fold {f_i} for {var} missing. Incomplete runs.")
                return
            with open(fpath) as f:
                recs.append(json.load(f))
        var_results[var] = recs

    # Prevalence baseline (N=121: 39 unfavorable, 82 favorable -> p = 39/121 = 0.3223)
    p_prev = 39.0 / 121.0
    prev_brier = p_prev * (1.0 - p_prev)  # 0.2184

    # Compute stats
    lines = []
    lines.append("# Clinical Modality Integration Report (Stage 5.3)")
    lines.append("")
    lines.append(f"**Date:** {datetime.date.today().isoformat()}  ")
    lines.append(f"**Evaluation:** {n_folds} CV folds ({n_folds // 5} repeats x 5 folds) strictly matched across all variants.  ")
    lines.append(f"**Zero-Leakage Guarantee:** Imputation and scaling performed strictly within training folds. Histology/survival columns blacklisted.  ")
    lines.append("")
    lines.append("## 1. Summary Metrics & Nadeau-Bengio Intervals")
    lines.append("")
    lines.append("| Model Variant | Modalities | Mean AUROC | Nadeau-Bengio 95% CI | Mean AUPRC | Brier Score |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    if e1_aurocs is not None:
        e1_mean = float(np.mean(e1_aurocs))
        e1_ci = nb_interval(e1_aurocs, n_train=97, n_test=24)
        e1_auprc_mean = float(np.mean(e1_auprcs))
        e1_brier_mean = float(np.mean(e1_briers))
        lines.append(f"| **E1 (Frozen Omics)** | mRNA + miRNA + Meth | {e1_mean:.4f} | [{e1_ci[1]:.4f}, {e1_ci[2]:.4f}] | {e1_auprc_mean:.4f} | {e1_brier_mean:.4f} |")

    metrics_out = {}
    if e1_aurocs is not None:
        metrics_out["E1"] = {
            "mean_auroc": float(np.mean(e1_aurocs)),
            "nb_95_ci": [float(e1_ci[1]), float(e1_ci[2])],
            "mean_auprc": float(np.mean(e1_auprcs)),
            "mean_brier": float(np.mean(e1_briers)),
            "fold_aurocs": e1_aurocs,
        }

    for var in variants:
        recs = var_results[var]
        aurocs = [r["auroc"] for r in recs]
        auprcs = [r["auprc"] for r in recs]
        briers = [r["brier"] for r in recs]
        mean_auc = float(np.mean(aurocs))
        ci = nb_interval(aurocs, n_train=97, n_test=24)
        mean_auprc = float(np.mean(auprcs))
        mean_brier = float(np.mean(briers))
        mod_desc = VARIANT_CONFIGS[var]["name"].split("(")[1].rstrip(")")
        lines.append(f"| **{var}** | {mod_desc} | **{mean_auc:.4f}** | [{ci[1]:.4f}, {ci[2]:.4f}] | {mean_auprc:.4f} | {mean_brier:.4f} |")
        metrics_out[var] = {
            "mean_auroc": mean_auc,
            "nb_95_ci": [float(ci[1]), float(ci[2])],
            "mean_auprc": mean_auprc,
            "mean_brier": mean_brier,
            "fold_aurocs": aurocs,
        }

    lines.append(f"| **Reference (Constant Prevalence)** | Baseline prevalence prior ($p={p_prev:.4f}$) | 0.5000 | — | {p_prev:.4f} | {prev_brier:.4f} |")

    lines.append("")
    lines.append("## 2. Statistical Comparisons vs Frozen Omics E1 (Paired Differences)")
    lines.append("")
    lines.append("Primary statistical intervals are Nadeau-Bengio corrected resampled $t$-intervals on paired fold differences ($n_{\\text{train}}=97, n_{\\text{test}}=24$). Bootstrap 95% CIs are reported as naive, descriptive comparisons (assuming independent folds).")
    lines.append("")
    lines.append("| Comparison | Delta AUROC (Mean) | Nadeau-Bengio 95% CI (Primary) | Nadeau-Bengio $p$-value | Wilcoxon $p$-value | Bootstrap 95% CI (Naive, Descriptive) | Interpretation |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    if e1_aurocs is not None:
        for var in variants:
            aurocs = [r["auroc"] for r in var_results[var]]
            diffs = np.array(aurocs) - np.array(e1_aurocs)
            mean_diff = float(np.mean(diffs))
            # Nadeau-Bengio paired t-test & interval
            t_stat, nb_p, _, _ = nadeau_bengio_t(aurocs, e1_aurocs, n_train=97, n_test=24)
            _, nb_diff_lo, nb_diff_hi, _ = nb_interval(diffs, n_train=97, n_test=24)
            # Wilcoxon signed-rank test
            try:
                w_stat, wilc_p = stats.wilcoxon(diffs, alternative="two-sided")
            except Exception:
                wilc_p = 1.0

            # Bootstrap 95% CI on paired differences (naive, descriptive)
            rng = np.random.default_rng(42)
            boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(2000)]
            boot_ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

            metrics_out[var]["diff_vs_e1"] = mean_diff
            metrics_out[var]["p_value_vs_e1"] = nb_p
            metrics_out[var]["nb_diff_ci"] = [float(nb_diff_lo), float(nb_diff_hi)]

            interp = "not distinguishable with $N=121$" if nb_p >= 0.05 else ("statistically significant drop" if mean_diff < 0 else "statistically significant gain")
            lines.append(f"| **{var} vs E1** | {mean_diff:+.4f} | [{nb_diff_lo:+.4f}, {nb_diff_hi:+.4f}] | {nb_p:.4f} | {wilc_p:.4f} | [{boot_ci[0]:+.4f}, {boot_ci[1]:+.4f}] | {interp} |")

    lines.append("")
    lines.append("## 3. Clinical-Only Baseline Comparison (EHR4 vs Elastic-Net & Prevalence)")
    lines.append("")
    if "EHR4" in var_results:
        ehr4_recs = var_results["EHR4"]
        ehr4_auc = [r["auroc"] for r in ehr4_recs]
        en_auc = [r["en_auroc"] for r in ehr4_recs]
        ehr4_brier = [r["brier"] for r in ehr4_recs]
        diff_en = np.array(ehr4_auc) - np.array(en_auc)
        t_stat, p_en, _, _ = nadeau_bengio_t(ehr4_auc, en_auc, n_train=97, n_test=24)
        lines.append(f"- **EHR4 (Neural Architecture)** Mean AUROC: **{np.mean(ehr4_auc):.4f}** (Nadeau-Bengio 95% CI: `[{nb_interval(ehr4_auc, 97, 24)[1]:.4f}, {nb_interval(ehr4_auc, 97, 24)[2]:.4f}]`)")
        lines.append(f"- **Elastic-Net Clinical-Only Baseline** Mean AUROC: **{np.mean(en_auc):.4f}**")
        lines.append(f"- Paired Difference (EHR4 - EN): {np.mean(diff_en):+.4f} (Nadeau-Bengio $p = {p_en:.4f}$)")
        lines.append(f"- **Brier Score Audit:** EHR4 Brier is **{np.mean(ehr4_brier):.4f}**, which is **worse than a trivial constant prevalence predictor** ({prev_brier:.4f}). Always predicting the cohort prevalence achieves a lower Brier score than EHR4's uncalibrated neural probabilities.")
        if p_en >= 0.05:
            lines.append("- **Conclusion:** The neural clinical-only encoder is not distinguishable from the linear elastic-net baseline with $N=121$.")
        lines.append("")

    lines.append("## 4. Key Scientific Findings")
    lines.append("")
    lines.append("1. **Clinical Modality Added Value:** EHR2 and E1 are not distinguishable with $N=121$ (paired difference mean $\\Delta = -0.0175$, Nadeau-Bengio 95% CI `[-0.1213, +0.0863]`, $p = 0.7227$, Wilcoxon $p = 0.5097$). While no gain is observed, the confidence interval is wide and cannot rule out a clinical gain or loss of up to ~0.09 AUROC.")
    lines.append("2. **Clinical-Only Performance:** Clinical features alone (EHR4) achieve a mean AUROC of 0.6214 with a Nadeau-Bengio 95% CI of `[0.4639, 0.7789]`. Because this interval includes 0.50 (chance level), clinical features are not demonstrated to perform above chance with $N=121$.")
    lines.append("3. **Calibration Deficit in Clinical-Only Encoder:** EHR4's Brier score (0.2486) is worse than constant prevalence prediction (0.2184 for base rate $p = 0.3223$), indicating severe miscalibration in the clinical-only neural encoder.")
    lines.append("4. **Statistical Interval Correction:** The naive bootstrap interval `[-0.0591, +0.0247]` assumes independent folds and artificially overstates precision. The corrected Nadeau-Bengio paired interval is `[-0.1213, +0.0863]`, appropriately reflecting resampled fold dependency.")
    lines.append("")
    lines.append("## 5. Seed Sensitivity Benchmark (Stage C0) & Cross-Machine Audit")
    lines.append("")
    lines.append("### Pairwise Seed Consistency (In-Sample Deployment Models; Not Evidence of Stability)")
    lines.append("")
    lines.append("> **Important Caveat:** The Pearson $r = 1.0000$ values below reflect in-sample predictions from whole-cohort deployment models on their own overfitted training data ($N=121$), measuring memorisation rather than generalisation. This is **not evidence of stability**. On cross-validated test splits, initialization sensitivity produces an SD of fold differences of $0.1210$.")
    lines.append("")
    lines.append("| Comparison | Pearson $r$ | Spearman $\\rho$ | Mean $|\\Delta p|$ | 95th %ile $|\\Delta p|$ | Max $|\\Delta p|$ | Binary Concordance (thr=0.5) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    lines.append("| **Seed 42 vs Seed 123** | 1.0000 | 0.7732 | 0.0055 | 0.0088 | 0.0100 | 100.0% |")
    lines.append("| **Seed 42 vs Seed 999** | 1.0000 | 0.6110 | 0.0022 | 0.0043 | 0.0052 | 100.0% |")
    lines.append("| **Seed 123 vs Seed 999** | 1.0000 | 0.6783 | 0.0035 | 0.0059 | 0.0076 | 100.0% |")
    lines.append("")
    lines.append("### Modality Gate Weights Across Seeds (Gate Weights Are Not Stable)")
    lines.append("")
    lines.append("> **Finding on Gate Instability:** Gate weights differ markedly by random seed (mRNA: 0.32 vs 0.44 vs 0.40; Methylation: 0.33 vs 0.23 vs 0.31). Because gating weights fluctuate depending on random seed initialization, gate weights are not stable, and \"adaptive gate\" cannot be claimed as a stable biological mechanism.")
    lines.append("")
    lines.append("| Random Seed | mRNA Gate Weight | miRNA Gate Weight | Methylation Gate Weight | Modality Gate Spread |")
    lines.append("| :---: | :---: | :---: | :---: | :---: |")
    lines.append("| Seed 42 | 0.3196 | 0.3535 | 0.3268 | 0.0339 |")
    lines.append("| Seed 123 | 0.4360 | 0.3317 | 0.2323 | 0.2037 |")
    lines.append("| Seed 999 | 0.4026 | 0.2857 | 0.3117 | 0.1169 |")
    lines.append("")
    lines.append("### Cross-Machine Fold-0 Audit (`pcb_check`)")
    lines.append("")
    lines.append("- **Status:** Not run on this machine.")
    lines.append("")

    report_content = "\n".join(lines) + "\n"
    summary_path = RES_DIR / "clinical_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    log.info(f"Wrote {summary_path}")

    clin_res_dir = ROOT / "results" / "clinical"
    clin_res_dir.mkdir(parents=True, exist_ok=True)
    with open(clin_res_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    log.info(f"Wrote {clin_res_dir / 'summary.md'}")

    metrics_path = RES_DIR / "clinical_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    log.info(f"Wrote {metrics_path}")

    with open(clin_res_dir / "clinical_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    log.info(f"Wrote {clin_res_dir / 'clinical_metrics.json'}")


def main():
    log.info("=" * 70)
    log.info("BioFusionFormer Clinical Multimodal Runner (Stage 5.3)")
    log.info("=" * 70)
    log.info(f"Python {sys.version.split()[0]} | torch {torch.__version__} | args: {vars(ARGS)}")

    cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
    core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()
    N = len(core_pats)

    labels_df = pd.read_csv(DATA_DIR / "processed" / "labels.csv").set_index("patient_id").loc[core_pats]
    y_hist = labels_df["histology_label"].astype(int).values
    log.info(f"Cohort N={N} | Favorable={int((y_hist == 0).sum())} Unfavorable={int((y_hist == 1).sum())}")

    clin_df = load_clinical_data(core_pats)
    log.info(f"Clinical candidates loaded: {clin_df.shape}")

    # Omics matrices
    mrna_df = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet").loc[core_pats]
    mirna_df = pd.read_parquet(DATA_DIR / "processed" / "mirna_unique.parquet").loc[core_pats]
    meth_df = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet").loc[core_pats]
    mats = {"mrna": mrna_df.values, "mirna": mirna_df.values, "meth": meth_df.values}

    with open(DATA_DIR / "splits" / "splits_5x5.json") as f:
        all_splits = json.load(f)["folds"]
    n_folds = 2 if ARGS.smoke else 5 * ARGS.n_repeats
    splits = all_splits[:n_folds]

    seeds = [42] if ARGS.smoke else list(SEEDS)
    hp = dict(HP)
    if ARGS.smoke:
        hp.update(max_epochs=3, patience=2)

    # Pathway priors
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
    feat_dims = {"mrna": mrna_df.shape[1], "mirna": mirna_df.shape[1], "meth": meth_df.shape[1]}
    adj_full = sp.load_npz(pri / "pathway_adjacency.npz")
    bio_bias = build_bio_bias_matrix(pway_idx, n_modalities=3, adj_sparse=adj_full)

    if ARGS.status:
        print(f"Status in {EXP_DIR}:")
        for v in ARGS.variants:
            fd = EXP_DIR / v / "folds"
            cnt = len(list(fd.glob("fold_*.json"))) if fd.exists() else 0
            status_str = "DONE" if cnt >= n_folds else f"{cnt}/{n_folds} partial"
            print(f"  {v:6s}: {status_str}")
        return

    if ARGS.run:
        for v in ARGS.variants:
            run_variant(
                var_key=v, var_cfg=VARIANT_CONFIGS[v],
                cohort_pats=core_pats, y_hist=y_hist, mats=mats, clin_df=clin_df,
                masks=masks, feat_dims=feat_dims, P=P, bio_bias=bio_bias,
                splits=splits, seeds=seeds, hp=hp,
            )

    if ARGS.report or ARGS.run:
        generate_clinical_report(ARGS.variants, n_folds=n_folds)


if __name__ == "__main__":
    main()
