"""
BioFusionFormer - Training Infrastructure
=========================================
Training harness supporting:
  - AdamW optimizer with cosine learning rate schedule
  - Positive class weighting for histology imbalance
  - Label smoothing BCE loss
  - Gradient norm clipping
  - Stratified 20% inner validation split for early stopping
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from src.heads.classification import LabelSmoothedBCEWithLogitsLoss


class BioFusionFormerTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.05,
        grad_clip: float = 1.0,
        max_epochs: int = 150,
        patience: int = 20,
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.max_epochs = max_epochs
        self.patience = patience

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=lr * 0.05)
        self.criterion = LabelSmoothedBCEWithLogitsLoss(eps=label_smoothing, pos_weight=pos_weight)

    def fit_with_inner_val(
        self,
        X_train_dict: Dict[str, np.ndarray],
        y_train: np.ndarray,
        batch_size: int = 16,
        seed: int = 42,
    ) -> Dict[str, float]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        first_mod = next(iter(X_train_dict.keys()))
        tr_in, val_in = next(skf.split(X_train_dict[first_mod], y_train))

        X_tr = {m: torch.tensor(X_train_dict[m][tr_in], dtype=torch.float32) for m in X_train_dict}
        y_tr = torch.tensor(y_train[tr_in], dtype=torch.float32)
        X_val = {m: torch.tensor(X_train_dict[m][val_in], dtype=torch.float32) for m in X_train_dict}
        y_val = torch.tensor(y_train[val_in], dtype=torch.float32)

        best_loss = float("inf")
        patience_ctr = 0
        N = len(tr_in)
        n_batches = int(math.ceil(N / batch_size))

        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(N)
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                bx = {m: X_tr[m][idx] for m in X_tr}
                by = y_tr[idx]
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out["logit"], by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val)
                val_loss = self.criterion(val_out["logit"], y_val).item()

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        return {"best_inner_val_loss": best_loss, "epochs_trained": epoch + 1}
