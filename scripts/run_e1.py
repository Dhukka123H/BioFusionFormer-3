"""
BioFusionFormer - E1 only (frozen v1 configuration), CLI
=========================================================
E1 = Full BioFusionFormer: mRNA + miRNA + methylation tokens, pathway attention bias ON,
     gated fusion ON, strong-evidence miRNA mask.  Target: Wilms histology (FH=0 vs UH=1).

This script replaces the old multi-experiment runner (E1-E10). It trains ONLY E1.

Usage (from the repo root, inside the venv)
-------------------------------------------
  python scripts/run_e1.py --status                 # show checkpoint progress (fast)
  python scripts/run_e1.py --smoke --run            # 2 folds / 1 seed / 3 epochs -> experiments/final_v1_smoke
  python scripts/run_e1.py --run                    # train/resume E1 on all 25 outer folds (5 repeats x 5-fold)
  python scripts/run_e1.py --run --n-repeats 2      # only the first 10 folds (decided BEFORE looking at results)
  python scripts/run_e1.py --report                 # E1_summary.md + figures from finished folds (no training)
  python scripts/run_e1.py --run --tag final_v1_repro --n-repeats 1   # reproducibility re-run (5 folds) in a separate folder
  python scripts/run_e1.py --run --force            # redo E1 (old folder is renamed E1.bak_<time>, never deleted)

Design rules
------------
* Every finished fold is written atomically to experiments/final_v1/E1/folds/fold_XX.json.
  Ctrl+C / power loss costs at most the fold that was running. Re-run the same command to resume.
* The config signature is identical to the one used by the earlier runner, so the 25 existing E1
  folds are recognised and NOT retrained (unless --force).
* set_seed(seed) is called BEFORE the model is built -> initialisation is reproducible.
* Labels come from data/processed/labels.csv (built from GDC clinical by scripts/build_labels.py).
* Confidence intervals in the report use the Nadeau-Bengio corrected resampled t-interval.
  The naive mean +/- 1.96*SD/sqrt(K) interval is also printed, only to show how much too narrow it is.
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser(description="BioFusionFormer E1-only benchmark (frozen v1 config, resumable)")
    ap.add_argument("--run", action="store_true", help="Train / resume E1")
    ap.add_argument("--report", action="store_true", help="Write results/model/E1_summary.md and figures from finished folds")
    ap.add_argument("--status", action="store_true", help="Show checkpoint progress and exit")
    ap.add_argument("--force", action="store_true", help="Redo E1 (old folder is renamed, not deleted)")
    ap.add_argument("--smoke", action="store_true", help="Tiny test run into experiments/final_v1_smoke")
    ap.add_argument("--tag", default="final_v1",
                    help="Output folder name under experiments/ (use e.g. final_v1_repro for a reproducibility re-run "
                         "so the frozen results are never touched)")
    ap.add_argument("--n-repeats", type=int, default=5, choices=[1, 2, 3, 4, 5],
                    help="Use the first n-repeats x 5 outer folds of splits_5x5.json (default 5 = all 25). "
                         "Decide this BEFORE looking at any result.")
    ap.add_argument("--seeds", default="42,123,999",
                    help="Comma-separated random seeds (default: '42,123,999')")
    args = ap.parse_args()
    if not (args.run or args.report or args.status):
        ap.print_help()
        sys.exit(0)
    return args


ARGS = parse_args()

from src.pathways.selection import select_unsupervised_pathways
from src.biofusionformer.full_model import BioFusionFormerModel
from src.heads.classification import LabelSmoothedBCEWithLogitsLoss
from src.explainability.explain import compute_attention_rollout
from src.evaluation.cv_stats import nadeau_bengio_t, nb_interval, naive_interval

DATA_DIR = ROOT / "data"
if ARGS.smoke:
    EXP_DIR = ROOT / "experiments" / f"{ARGS.tag}_smoke"
    RES_DIR = ROOT / "results_smoke" / "model"
    FIG_DIR = ROOT / "figures_smoke"
else:
    EXP_DIR = ROOT / "experiments" / ARGS.tag
    RES_DIR = ROOT / "results" / ("model" if ARGS.tag == "final_v1" else f"model_{ARGS.tag}")
    FIG_DIR = ROOT / "figures" if ARGS.tag == "final_v1" else ROOT / "figures" / ARGS.tag
LOG_DIR = ROOT / "logs"
for d in (EXP_DIR, RES_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("e1")
torch.set_num_threads(2)  # Use 2 CPU threads per AGENT_PROMPT Rule 6


def setup_logging():
    """One new log file per session (never overwritten) + one cumulative append-only file."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"e1_session_{stamp}.log", mode="x", encoding="utf-8"),
        logging.FileHandler(LOG_DIR / "e1_all_sessions.log", mode="a", encoding="utf-8"),
    ]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_bio_bias_matrix(
    pathway_indices: List[int],
    n_modalities: int,
    adj_sparse: sp.spmatrix,
    same_pathway_bias: float = 1.0,
    parent_child_bias: float = 0.5,
    cls_bias: float = 0.5,
) -> torch.Tensor:
    P = len(pathway_indices)
    M = n_modalities
    S = 1 + M * P
    B = np.zeros((S, S), dtype=np.float32)

    # 1. [CLS] bi-directional attention
    B[0, :] = cls_bias
    B[:, 0] = cls_bias
    B[0, 0] = 0.0

    # Subgraph adjacency between selected pathways
    sub_adj = adj_sparse[pathway_indices, :][:, pathway_indices].toarray()
    sym_adj = ((sub_adj + sub_adj.T) > 0).astype(float)

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

def train_single_seed_v1(
    model: BioFusionFormerModel,
    X_train_dict: Dict[str, np.ndarray],
    y_train: np.ndarray,
    X_test_dict: Dict[str, np.ndarray],
    pos_weight: float,
    seed: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    max_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 16,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Trains one model on one seed with 20% stratified inner validation early stopping.
    Returns: (test_probs, test_embeddings, test_attns)
    """
    set_seed(seed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    first_mod = list(X_train_dict.keys())[0]
    train_in_idx, val_in_idx = next(skf.split(X_train_dict[first_mod], y_train))

    X_in_tr = {m: torch.tensor(X_train_dict[m][train_in_idx], dtype=torch.float32) for m in X_train_dict}
    y_in_tr = torch.tensor(y_train[train_in_idx], dtype=torch.float32)
    X_in_val = {m: torch.tensor(X_train_dict[m][val_in_idx], dtype=torch.float32) for m in X_train_dict}
    y_in_val = torch.tensor(y_train[val_in_idx], dtype=torch.float32)
    X_te = {m: torch.tensor(X_test_dict[m], dtype=torch.float32) for m in X_test_dict}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.05)
    criterion = LabelSmoothedBCEWithLogitsLoss(eps=0.05, pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    N_tr = len(train_in_idx)
    n_batches = int(math.ceil(N_tr / batch_size))

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(N_tr)

        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            b_x = {m: X_in_tr[m][b_idx] for m in X_in_tr}
            b_y = y_in_tr[b_idx]

            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out["logit"], b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_in_val)
            val_loss = criterion(val_out["logit"], y_in_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        test_out = model(X_te)
        test_probs = torch.sigmoid(test_out["logit"]).cpu().numpy()
        test_emb = (test_out["cls_rep"] + test_out["gated_rep"]).cpu().numpy()

    return test_probs, test_emb, model

def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

class RunLock:
    """Prevents two processes from writing the same experiment folder at once."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"{self.path} exists: another process is running this experiment. "
                f"If you are sure none is (e.g. after a crash/power cut), delete that file and re-run."
            )
        os.write(fd, f"pid={os.getpid()} started={datetime.datetime.now()}".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

def read_fold(path, sig):
    """Return the checkpoint dict, or None if missing/corrupt. Raises if it was made with a different config."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning(f"Corrupt checkpoint {path.name}; it will be recomputed.")
        return None
    if rec.get("sig") != sig:
        raise RuntimeError(
            f"{path} was produced with a different configuration (sig {rec.get('sig')} != {sig}). "
            f"Refusing to mix results. Use --force to redo this experiment (old folder is kept as a backup)."
        )
    return rec



# -------------------------------------------------------------
# The only experiment in this repository
# -------------------------------------------------------------
E1 = {
    "id": "E1",
    "name": "Full BioFusionFormer (3 omics, bias ON, gate ON, strong mask)",
    "mods": ["mrna", "mirna", "meth"],
    "bias": True,
    "gate": True,
    "mask_type": "strong",
    "target": "histology",
}
SEEDS = [42, 123, 999]
HP = dict(lr=3e-4, weight_decay=1e-2, max_epochs=100, patience=20, batch_size=16)


def load_labels(core_pats):
    """Histology label (Unfavorable=1) from data/processed/labels.csv (see scripts/build_labels.py)."""
    p = DATA_DIR / "processed" / "labels.csv"
    if not p.exists():
        log.error(f"{p} not found. Run: python scripts/download_clinical.py && python scripts/build_labels.py")
        sys.exit(2)
    df = pd.read_csv(p).set_index("patient_id")
    missing = [x for x in core_pats if x not in df.index]
    if missing:
        log.error(f"{len(missing)} cohort patients missing in labels.csv, e.g. {missing[:3]}")
        sys.exit(2)
    df = df.loc[core_pats]
    return df["histology_label"].astype(int).values


def print_status():
    n_exp_folds = 2 if ARGS.smoke else 5 * ARGS.n_repeats
    print(f"Checkpoint status in {EXP_DIR}")
    fd = EXP_DIR / "E1" / "folds"
    n = len(list(fd.glob("fold_*.json"))) if fd.exists() else 0
    flag = "DONE" if n >= n_exp_folds else ("running/partial" if n else "not started")
    lock = " [LOCK present]" if (EXP_DIR / "E1" / ".lock").exists() else ""
    print(f"  E1: {n:2d}/{n_exp_folds} folds  {flag}{lock}")


def main():
    log.info("=" * 70)
    log.info("BioFusionFormer E1 (frozen v1 config) - resumable CLI runner")
    log.info("=" * 70)
    log.info(f"Python {sys.version.split()[0]} | torch {torch.__version__} | args: {vars(ARGS)}")

    # ---- 1. cohort + labels
    cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
    core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()
    N = len(core_pats)
    y_hist = load_labels(core_pats)
    log.info(f"Cohort N={N} | Favorable={int((y_hist == 0).sum())} Unfavorable={int((y_hist == 1).sum())}")

    # ---- 2. feature matrices (patient order is asserted)
    mrna_df = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet").loc[core_pats]
    mirna_df = pd.read_parquet(DATA_DIR / "processed" / "mirna_unique.parquet").loc[core_pats]
    meth_df = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet").loc[core_pats]
    for name, df in (("mrna", mrna_df), ("mirna", mirna_df), ("meth", meth_df)):
        assert list(df.index) == core_pats, f"{name} patient order differs from cohort_patients.csv"
    log.info(f"mRNA {mrna_df.shape} | miRNA(unique) {mirna_df.shape} | methylation {meth_df.shape}")

    # ---- 3. splits (first n_repeats x 5 folds; signature always uses the full file)
    with open(DATA_DIR / "splits" / "splits_5x5.json") as f:
        all_splits = json.load(f)["folds"]
    splits = all_splits[: 5 * ARGS.n_repeats]
    seeds = [int(s.strip()) for s in ARGS.seeds.split(",") if s.strip()]
    hp = dict(HP)
    if ARGS.smoke:
        splits = all_splits[:2]
        seeds = [42]
        hp.update(max_epochs=3, patience=2)
        log.warning("SMOKE MODE: 2 folds, 1 seed, 3 epochs. Results are meaningless.")
    n_folds = len(splits)
    sig_splits = splits if ARGS.smoke else all_splits
    split_hash = hashlib.sha1(
        json.dumps([[f["train_indices"], f["test_indices"]] for f in sig_splits]).encode()
    ).hexdigest()[:12]
    n_train = int(round(np.mean([len(f["train_indices"]) for f in splits])))
    n_test = int(round(np.mean([len(f["test_indices"]) for f in splits])))

    # ---- 4. unsupervised pathway selection + masks (identical to the frozen v1 code)
    selected_pways, pway_summary = select_unsupervised_pathways(DATA_DIR, top_k=192)
    P = len(selected_pways)
    log.info(f"Selected {P} unsupervised Reactome pathways")

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
    sub_mask_strong[zero_strong, P - 1] = True  # no-prior fallback token (frozen v1 behaviour)
    mask_mirna_strong = sub_mask_strong.T
    log.info(f"miRNA fallback active: {int(zero_strong.sum())} features mapped to the fallback token")

    masks = {
        "mrna": torch.tensor(mask_mrna, dtype=torch.bool),
        "mirna": torch.tensor(mask_mirna_strong, dtype=torch.bool),
        "meth": torch.tensor(mask_meth, dtype=torch.bool),
    }
    feat_dims = {"mrna": mrna_df.shape[1], "mirna": mirna_df.shape[1], "meth": meth_df.shape[1]}
    adj_full = sp.load_npz(pri / "pathway_adjacency.npz")
    bio_bias = build_bio_bias_matrix(pway_idx, n_modalities=3, adj_sparse=adj_full)
    pos_weight = float((y_hist == 0).sum() / (y_hist == 1).sum())
    mats = {"mrna": mrna_df.values, "mirna": mirna_df.values, "meth": meth_df.values}

    def prepare_fold_data(tr_idx, te_idx, mods):
        """Median-impute + z-score, fitted on the training fold only."""
        X_tr, X_te = {}, {}
        for m in mods:
            raw_tr, raw_te = mats[m][tr_idx], mats[m][te_idx]
            imp = SimpleImputer(strategy="median")
            raw_tr = imp.fit_transform(raw_tr)
            raw_te = imp.transform(raw_te)
            sc = StandardScaler()
            X_tr[m] = sc.fit_transform(raw_tr)
            X_te[m] = sc.transform(raw_te)
        return X_tr, X_te

    def build_model():
        return BioFusionFormerModel(
            modality_features={m: feat_dims[m] for m in E1["mods"]},
            modality_masks=masks,
            n_pathways=P,
            d_model=32, n_heads=4, n_layers=2, d_ff=128,
            dropout=0.3, attn_dropout=0.2, modality_dropout=0.2,
            use_pathway_bias=E1["bias"], use_gating=E1["gate"], bio_bias=bio_bias,
        )

    # NOTE: this signature is byte-identical to the one of the earlier runner so that existing folds stay valid.
    sig = hashlib.sha1(json.dumps({
        "exp": E1, "seeds": seeds, "hp": hp, "N": N, "P": P, "splits": split_hash,
        "arch": [32, 4, 2, 128, 0.3, 0.2, 0.2],
    }, sort_keys=True).encode()).hexdigest()[:12]
    log.info(f"E1 config signature: {sig}")

    run_dir = EXP_DIR / "E1"
    fold_dir = run_dir / "folds"

    def load_recs():
        recs = []
        for f_i in range(n_folds):
            rec = read_fold(fold_dir / f"fold_{f_i:02d}.json", sig)
            if rec is None:
                return None
            recs.append(rec)
        return recs

    def run_folds():
        t0 = time.time()
        times = []
        for f_i, fold in enumerate(splits):
            fpath = fold_dir / f"fold_{f_i:02d}.json"
            if read_fold(fpath, sig) is not None:
                log.info(f"  fold {f_i+1:02d}/{n_folds} already on disk - skipped")
                continue
            t_fold = time.time()
            tr_idx = np.array(fold["train_indices"])
            te_idx = np.array(fold["test_indices"])
            y_tr, y_te = y_hist[tr_idx], y_hist[te_idx]
            X_tr, X_te = prepare_fold_data(tr_idx, te_idx, E1["mods"])

            seed_probs = []
            gate = cls_roll = None
            for s in seeds:
                set_seed(s)  # BEFORE building the model
                model = build_model()
                probs, _embs, trained = train_single_seed_v1(
                    model=model, X_train_dict=X_tr, y_train=y_tr, X_test_dict=X_te,
                    pos_weight=pos_weight, seed=s, **hp,
                )
                seed_probs.append(probs)
                if s == seeds[0]:
                    trained.eval()
                    with torch.no_grad():
                        te_in = {m: torch.tensor(X_te[m], dtype=torch.float32) for m in X_te}
                        out = trained(te_in)
                        gate = out["gate"].cpu().numpy()
                        cls_roll = compute_attention_rollout(out["attns"]).cpu().numpy()[:, 0, 1:]
                    # one model per fold (seed 0 of the list) for later attribution work; git-ignored
                    mdir = ROOT / "models" / f"e1_folds_{ARGS.tag}"
                    mdir.mkdir(parents=True, exist_ok=True)
                    atomic_torch_save({k: v.cpu() for k, v in trained.state_dict().items()},
                                      mdir / f"fold_{f_i:02d}_seed{s}.pt")

            p_mean = np.mean(seed_probs, axis=0)
            pred = (p_mean >= 0.5).astype(int)
            both = len(np.unique(y_te)) > 1
            rec = {
                "sig": sig, "fold": f_i, "te_idx": te_idx.tolist(),
                "probs": [float(p) for p in p_mean],
                "auroc": float(roc_auc_score(y_te, p_mean)) if both else 0.5,
                "auprc": float(average_precision_score(y_te, p_mean)) if both else float(np.mean(y_te)),
                "bal_acc": float(balanced_accuracy_score(y_te, pred)),
                "f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
                "brier": float(brier_score_loss(y_te, p_mean)),
                "gate": np.asarray(gate, dtype=float).tolist(),
                "cls_rollout": np.asarray(cls_roll, dtype=float).tolist(),
            }
            atomic_write_json(fpath, rec)  # written last: existence == fold fully finished
            dt = time.time() - t_fold
            times.append(dt)
            remaining = sum(1 for k in range(n_folds) if not (fold_dir / f"fold_{k:02d}.json").exists())
            try:
                import psutil
                peak_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                ram_str = f" | Peak RAM={peak_ram_mb:.1f} MB"
            except Exception:
                ram_str = ""
            log.info(f"  fold {f_i+1:02d}/{n_folds} saved | {dt/60:.1f} min | AUROC={rec['auroc']:.4f} | "
                     f"Brier={rec['brier']:.4f}{ram_str} | ETA ~{remaining*np.mean(times)/60:.0f} min")
        log.info(f"session time {(time.time()-t0)/60:.1f} min")

    if ARGS.run:
        if ARGS.force and run_dir.exists():
            bak = EXP_DIR / f"E1.bak_{datetime.datetime.now():%Y%m%d_%H%M%S}"
            run_dir.rename(bak)
            log.warning(f"--force: previous E1 folder moved to {bak.name}")
        fold_dir.mkdir(parents=True, exist_ok=True)
        recs = load_recs()
        if recs is not None:
            log.info(f"all {n_folds} folds already on disk -> not retraining")
        else:
            with RunLock(run_dir / ".lock"):
                run_folds()

    if not ARGS.report:
        return
    recs = load_recs()
    if recs is None:
        log.error(f"--report needs all {n_folds} folds finished. Run --run first.")
        sys.exit(1)
    write_report(recs, core_pats, y_hist, selected_pways, pway_summary, n_train, n_test, sig, n_folds)


def write_report(recs, core_pats, y_hist, selected_pways, pway_summary, n_train, n_test, sig, n_folds):
    metrics = {k: [r[k] for r in recs] for k in ("auroc", "auprc", "bal_acc", "f1", "brier")}
    auc = metrics["auroc"]
    m, lo, hi, se = nb_interval(auc, n_train, n_test)
    nm, nlo, nhi = naive_interval(auc)
    rep_means = [float(np.mean(auc[i:i + 5])) for i in range(0, len(auc), 5)]

    # pooled out-of-fold predictions, averaged over repeats (ensemble effect -> better Brier than the fold mean)
    pp = {}
    for r in recs:
        for i, p in zip(r["te_idx"], r["probs"]):
            pp.setdefault(i, []).append(p)
    oof = np.array([np.mean(pp[i]) for i in range(len(core_pats))])
    brier_pooled = float(brier_score_loss(y_hist, oof))
    brier_foldmean = float(np.mean(metrics["brier"]))

    # baselines (same folds)
    base_rows = []
    bpath = ROOT / "results" / "baselines" / "baseline_metrics_histology.json"
    if bpath.exists():
        base = json.load(open(bpath))
        for name in ("mRNA (ElasticNet)", "Early Fusion (ElasticNet)", "miRNA (ElasticNet)",
                     "Methylation (ElasticNet)", "Clinical-only (Age + Gender)"):
            if name in base:
                sc = base[name]["auroc"]["raw_folds"][:n_folds]
                t, p, d, s_ = nadeau_bengio_t(auc, sc, n_train=n_train, n_test=n_test)
                base_rows.append((name, float(np.mean(sc)), d, t, p))

    # gate + rollout
    gates = np.concatenate([np.array(r["gate"]) for r in recs], axis=0)
    labels = np.concatenate([y_hist[np.array(r["te_idx"])] for r in recs])
    roll = np.concatenate([np.array(r["cls_rollout"]) for r in recs], axis=0)
    P = len(selected_pways)
    pw_att = roll.reshape(len(roll), 3, P).sum(axis=1).mean(axis=0)
    spread = float(pw_att.max() / max(pw_att.min(), 1e-12))
    top = np.argsort(pw_att)[::-1][:10]

    # figures
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    pt, pq = calibration_curve(y_hist, oof, n_bins=5, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(pq, pt, "s-", label=f"E1 pooled OOF (Brier {brier_pooled:.3f})")
    ax.set_xlabel("Mean predicted P(unfavorable)"); ax.set_ylabel("Observed fraction")
    ax.set_title("E1 reliability (5 quantile bins, N=121: noisy)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_calibration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    names = ["mRNA", "miRNA", "Methylation"]
    x = np.arange(3)
    for off, (lab, col, msk) in zip((-0.18, 0.18), (("Favorable", "#2b5c8f", labels == 0), ("Unfavorable", "#c0392b", labels == 1))):
        ax.bar(x + off, gates[msk].mean(0), 0.36, yerr=gates[msk].std(0), capsize=3, label=lab, color=col, alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Gate weight")
    ax.set_title("E1 modality gate weights (mean +/- SD across patients)")
    ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_gate_weights.png"); plt.close(fig)

    L = []
    L.append("# E1 summary (frozen v1 configuration)")
    L.append("")
    L.append(f"- Config signature: `{sig}`; folds used: **{n_folds}** (n_train~{n_train}, n_test~{n_test}); seeds per fold: 3")
    L.append("- Source of numbers: per-fold checkpoints in `experiments/final_v1/E1/folds/` (nothing is re-tuned here).")
    L.append("")
    L.append("## Discrimination")
    L.append("")
    L.append("| Metric | Mean over folds |")
    L.append("|---|---|")
    L.append(f"| AUROC | **{m:.4f}** |")
    L.append(f"| AUPRC (prevalence {float(np.mean(y_hist)):.3f}) | {np.mean(metrics['auprc']):.4f} |")
    L.append(f"| Balanced accuracy | {np.mean(metrics['bal_acc']):.4f} |")
    L.append(f"| Macro-F1 | {np.mean(metrics['f1']):.4f} |")
    L.append(f"| Brier (mean of per-fold) | {brier_foldmean:.4f} |")
    L.append(f"| Brier (pooled out-of-fold, predictions averaged over repeats) | {brier_pooled:.4f} |")
    L.append("")
    L.append(f"AUROC 95% interval, **Nadeau-Bengio corrected**: [{lo:.4f}, {hi:.4f}] (half-width {m-lo:.4f}).")
    L.append(f"Naive interval that treats overlapping folds as independent: [{nlo:.4f}, {nhi:.4f}] "
             f"(half-width {nm-nlo:.4f}) - **too narrow; do not quote it**.")
    L.append(f"Per-repeat mean AUROC: {', '.join(f'{v:.3f}' for v in rep_means)} "
             f"(SD across repeats {np.std(rep_means, ddof=1) if len(rep_means) > 1 else float('nan'):.3f}).")
    L.append("")
    if base_rows:
        L.append("## Comparison with baselines on the same folds (Nadeau-Bengio corrected paired test)")
        L.append("")
        L.append("| Baseline | Baseline AUROC | E1 - baseline | t | p |")
        L.append("|---|---|---|---|---|")
        for name, bm, d, t, p in base_rows:
            L.append(f"| {name} | {bm:.4f} | {d:+.4f} | {t:.2f} | {p:.4f} |")
        L.append("")
        L.append("p >= 0.05 means 'not distinguishable with N=121', not 'equal'.")
        L.append("")
    L.append("## Modality gate (seed-42 model of each fold, test patients)")
    L.append("")
    L.append("| Modality | Favorable | Unfavorable |")
    L.append("|---|---|---|")
    for j, nme in enumerate(names):
        L.append(f"| {nme} | {gates[labels == 0][:, j].mean():.3f} +/- {gates[labels == 0][:, j].std():.3f} | "
                 f"{gates[labels == 1][:, j].mean():.3f} +/- {gates[labels == 1][:, j].std():.3f} |")
    L.append("")
    L.append("## Attention rollout from [CLS] (summed over modalities, mean over test patients)")
    L.append("")
    L.append(f"Largest / smallest pathway weight = **{spread:.2f}x**. If this is close to 1, attention does not "
             "discriminate between pathways and should not be used as an explanation; use gradient-based attribution instead.")
    L.append("")
    for i in top:
        L.append(f"- {pway_summary.loc[selected_pways[i], 'name']} ({selected_pways[i]}): {pw_att[i]:.5f}")
    L.append("")
    L.append("## Limitations that must accompany any quoted number")
    L.append("")
    L.append("- N = 121 patients from one enriched, high-risk cohort; no independent test set.")
    L.append("- Omics matrices are Xena-derived (see data/DATA_CARD.json) unless the GDC-native rebuild has been completed.")
    L.append("- Pathway selection (top-variance, unsupervised) used all 121 patients' omics, no labels.")
    L.append("- E1 was first run once with an unseeded initialisation (AUROC 0.846, discarded) and then re-run "
             "with seeding (this result). See results/model/forking_paths.md.")
    (RES_DIR / "E1_summary.md").write_text("\n".join(L), encoding="utf-8")
    atomic_write_json(RES_DIR / "E1_metrics.json", {
        "signature": sig, "n_folds": n_folds, "n_train": n_train, "n_test": n_test,
        "auroc_mean": m, "auroc_nb_ci": [lo, hi], "auroc_naive_ci": [nlo, nhi],
        "per_repeat_means": rep_means, "brier_foldmean": brier_foldmean, "brier_pooled": brier_pooled,
        "folds": metrics, "baselines": [dict(zip(("name", "mean", "diff", "t", "p"), r)) for r in base_rows],
        "rollout_max_over_min": spread,
    })
    log.info(f"Wrote {RES_DIR / 'E1_summary.md'}")
    log.info(f"E1 AUROC {m:.4f}  NB 95% CI [{lo:.4f}, {hi:.4f}]  (naive [{nlo:.4f}, {nhi:.4f}])")


if __name__ == "__main__":
    if ARGS.status:
        print_status()
        sys.exit(0)
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted. Finished folds are saved on disk; re-run the same command to resume.")
        sys.exit(130)
