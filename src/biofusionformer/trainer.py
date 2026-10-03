"""
BioFusionFormer - Training Engine
==================================
Trains BioFusionFormer for one outer CV fold with:
  - Binary cross-entropy loss with optional class weighting
  - AdamW optimizer with cosine annealing LR schedule
  - Gradient clipping
  - Early stopping on validation loss
  - Deterministic seeding

All fitting is strictly inside the training fold.
"""

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trains a BioFusionFormer model for one outer CV fold.

    Parameters
    ----------
    model        : BioFusionFormer
    device       : torch.device (CPU for this project)
    lr           : float   initial learning rate
    weight_decay : float   L2 regularization
    n_epochs     : int     max training epochs
    patience     : int     early stopping patience (epochs w/o val loss improvement)
    grad_clip    : float   gradient norm clipping threshold
    pos_weight   : float   BCE positive class weight (for class imbalance)
    seed         : int
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        patience: int = 15,
        grad_clip: float = 1.0,
        pos_weight: Optional[float] = None,
        seed: int = 42,
    ):
        self.model        = model.to(device)
        self.device       = device
        self.n_epochs     = n_epochs
        self.patience     = patience
        self.grad_clip    = grad_clip
        self.seed         = seed

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )

        pw = torch.tensor([pos_weight], device=device) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def _to_device(self, batch):
        feat, labels, pids = batch
        feat   = {k: v.to(self.device) for k, v in feat.items()}
        labels = labels.to(self.device)
        return feat, labels, pids

    def _forward(self, feat: Dict, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out    = self.model(feat)
        logits = out["logit"]
        loss   = self.criterion(logits, labels)
        return loss, logits

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            feat, labels, _ = self._to_device(batch)
            self.optimizer.zero_grad()
            loss, _ = self._forward(feat, labels)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []

        for batch in loader:
            feat, labels, _ = self._to_device(batch)
            _, logits = self._forward(feat, labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy().astype(int)
        probs  = 1 / (1 + np.exp(-logits))   # sigmoid

        loss = float(
            nn.BCEWithLogitsLoss()(
                torch.tensor(logits), torch.tensor(labels, dtype=torch.float32)
            ).item()
        )
        try:
            auroc = roc_auc_score(labels, probs)
        except Exception:
            auroc = 0.5

        return {"loss": loss, "auroc": auroc, "probs": probs, "labels": labels}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict:
        """
        Train until patience exhausted or n_epochs reached.
        Returns dict of training history and best validation metrics.
        """
        set_seed(self.seed)
        best_val_loss = math.inf
        best_state    = None
        patience_ctr  = 0
        history       = {"train_loss": [], "val_loss": [], "val_auroc": []}

        for epoch in range(1, self.n_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.evaluate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_auroc"].append(val_metrics["auroc"])

            if verbose and epoch % 10 == 0:
                log.info(
                    f"  Epoch {epoch:3d}  "
                    f"train_loss={train_metrics['loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_auroc={val_metrics['auroc']:.4f}"
                )

            if val_metrics["loss"] < best_val_loss - 1e-5:
                best_val_loss = val_metrics["loss"]
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    if verbose:
                        log.info(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        final_val = self.evaluate(val_loader)
        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_auroc": final_val["auroc"],
            "best_val_probs": final_val["probs"].tolist(),
            "best_val_labels": final_val["labels"].tolist(),
            "n_epochs_run": len(history["train_loss"]),
        }
