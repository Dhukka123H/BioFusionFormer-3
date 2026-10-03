"""
BioFusionFormer - Dataset and Data Loaders
==========================================
Provides:
  - OmicsDataset: PyTorch Dataset for aligned multi-omics + labels
  - build_fold_loaders: Returns train/test DataLoaders for one outer CV fold

All label-dependent steps (standard scaling of features) happen INSIDE this
loader, using only training-fold statistics. Test folds never see fit statistics.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Helper: column-mean imputation
# ---------------------------------------------------------------------------

def _impute(arr: np.ndarray, fill_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Impute NaN with column means (computed from arr if fill_values is None).
    Returns (imputed_arr, col_means) tuple.
    """
    if fill_values is None:
        fill_values = np.nanmean(arr, axis=0)
        # If an entire column is NaN, fill with 0
        fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    inds = np.where(np.isnan(arr))
    arr = arr.copy()
    arr[inds] = np.take(fill_values, inds[1])
    return arr, fill_values


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OmicsDataset(Dataset):
    """
    Multi-omics dataset for one split (train or test).

    Parameters
    ----------
    features : dict[str, np.ndarray]  (N, F_mod) float32 arrays
    labels   : np.ndarray             (N,) int64 binary labels {0, 1}
    patient_ids : list[str]           for bookkeeping / explainability
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
        patient_ids: List[str],
    ):
        self.features    = {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}
        self.labels      = torch.tensor(labels, dtype=torch.float32)
        self.patient_ids = patient_ids
        self._N = labels.shape[0]

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, str]:
        feat = {k: v[idx] for k, v in self.features.items()}
        return feat, self.labels[idx], self.patient_ids[idx]


# ---------------------------------------------------------------------------
# Build train/test loaders for one outer fold
# ---------------------------------------------------------------------------

def build_fold_loaders(
    splits_path: Path,
    fold_idx: int,
    mrna_path: Path,
    mirna_path: Path,
    meth_path: Path,
    label_col: str = "wilms_tumor_histologic_subtype.diagnoses",
    label_map: Optional[Dict[str, int]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train + test DataLoaders for one outer CV fold.

    Preprocessing steps:
      1. Load feature matrices (patients are rows, aligned).
      2. NaN imputation with TRAINING fold column means (applied to both splits).
      3. Standard scaling with TRAINING fold mean/std (applied to both splits).
      These label-dependent steps use ONLY the training fold.

    Parameters
    ----------
    splits_path : Path to splits_5x5.json
    fold_idx    : int, index into splits["folds"]
    *_path      : Path to processed .parquet files
    label_col   : which clinical column to use as label
    label_map   : dict mapping string labels to ints (default: Favorable=0, Unfavorable=1)
    batch_size  : int
    num_workers : int (0 recommended for Windows)
    seed        : random seed for DataLoader shuffling

    Returns
    -------
    train_loader, test_loader, patient_ids_train
    """
    if label_map is None:
        label_map = {"Favorable": 0, "Unfavorable": 1}

    # ---- Load splits ----
    with open(splits_path) as f:
        splits = json.load(f)
    fold = splits["folds"][fold_idx]
    train_idx = np.array(fold["train_indices"])
    test_idx  = np.array(fold["test_indices"])
    patient_ids = fold["train_patient_ids"] + fold["test_patient_ids"]
    # Reconstruct full ordered list
    all_patient_ids = _reconstruct_patient_order(splits)

    # ---- Load feature matrices ----
    mrna  = pd.read_parquet(mrna_path)
    mirna = pd.read_parquet(mirna_path)
    meth  = pd.read_parquet(meth_path)

    # All matrices share the same patient order (verified by check G)
    patient_order = mrna.index.tolist()

    # ---- Load labels from splits JSON ----
    labels_arr = _load_labels_from_splits(splits, fold_idx, patient_order, label_map)

    # ---- Extract train/test subsets ----
    def _subset(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        return df.iloc[idx].values.astype(np.float32)

    train_mrna  = _subset(mrna,  train_idx)
    train_mirna = _subset(mirna, train_idx)
    train_meth  = _subset(meth,  train_idx)
    test_mrna   = _subset(mrna,  test_idx)
    test_mirna  = _subset(mirna, test_idx)
    test_meth   = _subset(meth,  test_idx)

    train_labels = labels_arr[train_idx].astype(np.int64)
    test_labels  = labels_arr[test_idx].astype(np.int64)

    train_pids = [patient_order[i] for i in train_idx]
    test_pids  = [patient_order[i] for i in test_idx]

    # ---- NaN imputation (fit on train only) ----
    train_mrna,  means_mrna  = _impute(train_mrna)
    train_mirna, means_mirna = _impute(train_mirna)
    train_meth,  means_meth  = _impute(train_meth)
    test_mrna,  _ = _impute(test_mrna,  means_mrna)
    test_mirna, _ = _impute(test_mirna, means_mirna)
    test_meth,  _ = _impute(test_meth,  means_meth)

    # ---- Standard scaling (fit on train only) ----
    def _scale(train: np.ndarray, test: np.ndarray):
        mu  = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True) + 1e-8
        return (train - mu) / std, (test - mu) / std

    train_mrna,  test_mrna  = _scale(train_mrna,  test_mrna)
    train_mirna, test_mirna = _scale(train_mirna, test_mirna)
    train_meth,  test_meth  = _scale(train_meth,  test_meth)

    # ---- Build datasets ----
    train_ds = OmicsDataset(
        features={"mrna": train_mrna, "mirna": train_mirna, "meth": train_meth},
        labels=train_labels,
        patient_ids=train_pids,
    )
    test_ds = OmicsDataset(
        features={"mrna": test_mrna, "mirna": test_mirna, "meth": test_meth},
        labels=test_labels,
        patient_ids=test_pids,
    )

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=len(test_ds),  # one batch for test
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, train_pids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reconstruct_patient_order(splits: dict) -> List[str]:
    """Reconstruct full patient list from splits metadata."""
    if "metadata" in splits and "patient_ids" in splits["metadata"]:
        return splits["metadata"]["patient_ids"]
    # Fallback: union of all patient IDs preserving order
    seen = {}
    for fold in splits["folds"]:
        for pid in fold["train_patient_ids"] + fold["test_patient_ids"]:
            if pid not in seen:
                seen[pid] = True
    return list(seen.keys())


def _load_labels_from_splits(
    splits: dict,
    fold_idx: int,
    patient_order: List[str],
    label_map: Dict[str, int],
) -> np.ndarray:
    """
    Extract per-patient labels from splits JSON.
    Uses train/test histology counts to build a label vector.
    Falls back to reading clinical file if counts not available.
    """
    N = len(patient_order)
    labels = np.full(N, -1, dtype=np.int64)

    # Try to rebuild from fold metadata if histology per-patient is stored
    fold = splits["folds"][fold_idx]
    if "train_patient_ids" in fold and "test_patient_ids" in fold:
        # Check if the splits contain per-patient label info
        # If not, load from clinical file
        pass

    # Load from raw clinical data
    import gzip
    clin_path = Path("data/raw/target_wt_xena/TARGET-WT.clinical.tsv.gz")
    if not clin_path.exists():
        clin_path = Path("../data/raw/target_wt_xena/TARGET-WT.clinical.tsv.gz")

    with gzip.open(clin_path, "rt") as f:
        clin = pd.read_csv(f, sep="\t")
    clin["patient_id"] = clin["sample"].str[:16]
    clin = clin.sort_values("sample").drop_duplicates("patient_id", keep="first")
    clin = clin.set_index("patient_id")

    label_col = "wilms_tumor_histologic_subtype.diagnoses"
    pid_to_label = {}
    for pid in clin.index:
        val = clin.loc[pid, label_col]
        if val in label_map:
            pid_to_label[pid] = label_map[val]

    for i, pid in enumerate(patient_order):
        if pid in pid_to_label:
            labels[i] = pid_to_label[pid]
        else:
            warnings.warn(f"Patient {pid!r} missing label; setting to -1.")

    return labels
