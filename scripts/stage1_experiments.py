"""
BioFusionFormer - Experiment Runner
=====================================
Defines 8 experiment configurations (E1-E8) and runs the full 25-fold
5x5 nested CV benchmark.

Experiment Configurations
--------------------------
E1  BioFusionFormer (all 3 omics, topology bias ON)         [primary]
E2  BioFusionFormer (all 3 omics, topology bias OFF)        [ablation: no structure]
E3  BioFusionFormer (mRNA + methylation only, bias ON)      [ablation: no miRNA]
E4  BioFusionFormer (mRNA only, bias ON)                    [ablation: unimodal]
E5  BioFusionFormer (miRNA + meth only, bias ON)            [ablation: no mRNA]
E6  BioFusionFormer (all 3 omics, no gating, bias ON)       [ablation: no modality gate]
E7  BioFusionFormer (all 3 omics, unmasked projection)      [ablation: no pathway mask]
E8  BioFusionFormer (all 3 omics, RANDOM topology bias)     [ablation: random structure]

Results are compared using the Nadeau-Bengio corrected resampled t-test.
"""

import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.stats as stats
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.biofusionformer.model import BioFusionFormer
from src.biofusionformer.mask_builder import MaskBuilder
from src.biofusionformer.dataset import build_fold_loaders, _load_labels_from_splits
from src.biofusionformer.trainer import Trainer, set_seed
from src.biofusionformer.pathway_bias import PathwayBiasBuilder

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results" / "model"
RES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage1_experiments.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nadeau-Bengio corrected t-test
# ---------------------------------------------------------------------------

def nadeau_bengio_t(scores_a: List[float], scores_b: List[float], n_train: int, n_test: int):
    J = len(scores_a)
    diffs = np.array(scores_a) - np.array(scores_b)
    mean_diff = np.mean(diffs)
    var_diffs  = np.var(diffs, ddof=1)
    rho = n_test / (n_train + n_test)
    corrected_var = (1 / J + rho / (1 - rho)) * var_diffs
    t_stat = mean_diff / np.sqrt(max(corrected_var, 1e-12))
    p_val  = 2 * stats.t.sf(abs(t_stat), df=J - 1)
    return float(t_stat), float(p_val)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    config_name: str,
    modality_features: Dict[str, int],
    modality_masks: Dict[str, torch.BoolTensor],
    pathway_order: List[str],
    topology_bias: Optional[torch.Tensor],
    n_pathways: int,
    d_model: int = 32,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    dropout: float = 0.1,
) -> BioFusionFormer:

    # Which modalities to include
    if config_name == "E3":
        mods = ["mrna", "meth"]
    elif config_name == "E4":
        mods = ["mrna"]
    elif config_name == "E5":
        mods = ["mirna", "meth"]
    else:
        mods = ["mrna", "mirna", "meth"]

    feat_subset = {m: modality_features[m] for m in mods if m in modality_features}
    mask_subset = {m: modality_masks[m]     for m in mods if m in modality_masks}

    # Topology bias
    if config_name in ("E2",):
        bias = None                        # no topology bias
    elif config_name == "E8":
        # random topology bias (same sparsity as real, random entries)
        if topology_bias is not None:
            rng_bias = torch.zeros_like(topology_bias)
            nonzero = (topology_bias != 0)
            rng_bias[nonzero] = torch.randn(nonzero.sum()) * topology_bias[nonzero].abs().mean()
            bias = rng_bias
        else:
            bias = None
    else:
        bias = topology_bias

    # Unmasked projection (E7): replace masks with all-ones
    if config_name == "E7":
        mask_subset = {
            m: torch.ones(n_pathways, modality_features[m], dtype=torch.bool)
            for m in mods if m in modality_features
        }

    # No gating (E6): handled via a subclass override (use equal weights)
    model = BioFusionFormer(
        modality_features=feat_subset,
        modality_masks=mask_subset,
        n_pathways=n_pathways,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        topology_bias=bias,
    )

    if config_name == "E6":
        # Freeze gate projection, set to constant equal weights
        # (simulate no modality gate by fixing gate to uniform)
        with torch.no_grad():
            model.fusion.gate_proj.weight.fill_(0.0)
            model.fusion.gate_proj.bias.fill_(0.0)
        for p in model.fusion.gate_proj.parameters():
            p.requires_grad = False

    return model


# ---------------------------------------------------------------------------
# Single fold runner
# ---------------------------------------------------------------------------

def run_fold(
    fold_idx: int,
    config_name: str,
    modality_features: Dict[str, int],
    modality_masks: Dict[str, torch.BoolTensor],
    pathway_order: List[str],
    topology_bias: Optional[torch.Tensor],
    n_pathways: int,
    splits_path: Path,
    hparams: Dict,
    seed: int = 42,
) -> float:
    """Train one fold of one experiment. Returns test AUROC."""
    set_seed(seed + fold_idx)

    train_loader, test_loader, _ = build_fold_loaders(
        splits_path=splits_path,
        fold_idx=fold_idx,
        mrna_path=ROOT / "data/processed/mrna_gene.parquet",
        mirna_path=ROOT / "data/processed/mirna.parquet",
        meth_path=ROOT / "data/processed/meth_gene.parquet",  # use gene-level meth
        batch_size=hparams.get("batch_size", 32),
        seed=seed + fold_idx,
    )

    model = build_model(
        config_name=config_name,
        modality_features=modality_features,
        modality_masks=modality_masks,
        pathway_order=pathway_order,
        topology_bias=topology_bias,
        n_pathways=n_pathways,
        d_model=hparams.get("d_model", 32),
        n_heads=hparams.get("n_heads", 4),
        n_layers=hparams.get("n_layers", 2),
        d_ff=hparams.get("d_ff", 128),
        dropout=hparams.get("dropout", 0.1),
    )

    # Class imbalance weighting: Unfavorable (minority) gets higher weight
    # (ratio: Favorable=82/121, Unfavorable=39/121 approx)
    pos_weight = hparams.get("pos_weight", 82.0 / 39.0)

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=hparams.get("lr", 3e-4),
        weight_decay=hparams.get("weight_decay", 1e-4),
        n_epochs=hparams.get("n_epochs", 150),
        patience=hparams.get("patience", 20),
        grad_clip=hparams.get("grad_clip", 1.0),
        pos_weight=pos_weight,
        seed=seed + fold_idx,
    )

    result = trainer.fit(train_loader, test_loader, verbose=False)
    return result["best_val_auroc"]


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
EXPERIMENT_NAMES = {
    "E1": "BioFusionFormer (full, bias ON)",
    "E2": "BioFusionFormer (full, bias OFF)",
    "E3": "BioFusionFormer (mRNA+meth, bias ON)",
    "E4": "BioFusionFormer (mRNA only)",
    "E5": "BioFusionFormer (miRNA+meth, bias ON)",
    "E6": "BioFusionFormer (full, no gate, bias ON)",
    "E7": "BioFusionFormer (full, unmasked projection)",
    "E8": "BioFusionFormer (full, random topology bias)",
}


def main():
    log.info("=" * 70)
    log.info("BioFusionFormer Stage 1 - Experiment Benchmarking (E1-E8)")
    log.info("=" * 70)

    SPLITS_PATH = ROOT / "data/splits/splits_5x5.json"
    REACTOME_DIR = ROOT / "data/interim/reactome"
    MIRNA2GENE_PATH = ROOT / "data/interim/mirna2gene_strong.parquet"
    HIER_PATH = ROOT / "data/processed/pathway_hierarchy_mapping.json"

    # ---- Load feature matrices to get dimensions and feature IDs ----
    import pandas as pd
    mrna  = pd.read_parquet(ROOT / "data/processed/mrna_gene.parquet")
    mirna = pd.read_parquet(ROOT / "data/processed/mirna.parquet")
    meth  = pd.read_parquet(ROOT / "data/processed/meth_gene.parquet")
    pathway_feat = pd.read_parquet(ROOT / "data/processed/pathway_features_fold_agnostic.parquet")

    # Pathway order from the pathway feature matrix
    pathway_order = pathway_feat.columns.tolist()
    n_pathways    = len(pathway_order)
    log.info(f"Pathway order: {n_pathways} pathways")
    log.info(f"mRNA features: {mrna.shape[1]}")
    log.info(f"miRNA features: {mirna.shape[1]}")
    log.info(f"Meth gene features: {meth.shape[1]}")

    # ---- Build masks ----
    log.info("Building pathway masks ...")
    builder = MaskBuilder(reactome_dir=REACTOME_DIR, mirna2gene_path=MIRNA2GENE_PATH)
    modality_masks = builder.build_all_masks(
        pathway_order=pathway_order,
        mrna_features=mrna.columns.tolist(),
        mirna_features=mirna.columns.tolist(),
        meth_features=meth.columns.tolist(),
    )

    # Verify mask coverage
    for mod, mask in modality_masks.items():
        covered_pathways = (mask.any(dim=1)).sum().item()
        covered_features = (mask.any(dim=0)).sum().item()
        log.info(f"  {mod}: {covered_pathways}/{n_pathways} pathways covered, "
                 f"{covered_features}/{mask.shape[1]} features covered")

    modality_features = {
        "mrna":  mrna.shape[1],
        "mirna": mirna.shape[1],
        "meth":  meth.shape[1],
    }

    # ---- Build topology bias ----
    log.info("Building topology bias ...")
    topo_builder = PathwayBiasBuilder(
        hierarchy_path=HIER_PATH,
        pathway_order=pathway_order,
        modality_masks={k: v for k, v in modality_masks.items()},
        parent_bias=1.0,
        coannot_bias=0.5,
        min_shared=3,
    )
    topology_bias = topo_builder.build()
    n_nonzero = int((topology_bias != 0).sum().item())
    log.info(f"  Topology bias: {n_nonzero} nonzero entries / {n_pathways**2} total")

    # ---- Hyperparameters ----
    hparams = {
        "d_model":     32,
        "n_heads":     4,
        "n_layers":    2,
        "d_ff":        128,
        "dropout":     0.1,
        "lr":          3e-4,
        "weight_decay": 1e-4,
        "n_epochs":    150,
        "patience":    20,
        "grad_clip":   1.0,
        "batch_size":  32,
        "pos_weight":  82.0 / 39.0,
    }

    # Load splits to get N
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    n_folds = len(splits["folds"])
    n_total = 121
    n_test  = round(n_total / 5)
    n_train = n_total - n_test
    log.info(f"Running {n_folds} folds across {len(EXPERIMENT_CONFIGS)} experiments")

    # ---- Run experiments ----
    all_results: Dict[str, List[float]] = {}

    for config in EXPERIMENT_CONFIGS:
        log.info(f"\n--- {config}: {EXPERIMENT_NAMES[config]} ---")
        fold_aurocs = []
        t0 = time.time()

        for fold_idx in range(n_folds):
            auroc = run_fold(
                fold_idx=fold_idx,
                config_name=config,
                modality_features=modality_features,
                modality_masks=modality_masks,
                pathway_order=pathway_order,
                topology_bias=topology_bias,
                n_pathways=n_pathways,
                splits_path=SPLITS_PATH,
                hparams=hparams,
                seed=42,
            )
            fold_aurocs.append(auroc)
            log.info(f"  fold {fold_idx+1:2d}/{n_folds}: AUROC={auroc:.4f}")

        all_results[config] = fold_aurocs
        elapsed = time.time() - t0
        log.info(
            f"  {config} DONE  mean={np.mean(fold_aurocs):.4f}  "
            f"std={np.std(fold_aurocs):.4f}  time={elapsed/60:.1f}min"
        )

        # Save intermediate results
        _save_results(all_results, RES_DIR, hparams, n_train, n_test)

    # ---- Compute Nadeau-Bengio corrected t-tests vs E1 ----
    log.info("\n--- Nadeau-Bengio corrected t-test (vs E1) ---")
    nb_results = {}
    e1_scores = all_results["E1"]
    for config in EXPERIMENT_CONFIGS:
        if config == "E1":
            continue
        t_stat, p_val = nadeau_bengio_t(e1_scores, all_results[config], n_train, n_test)
        sig = "p<0.05" if p_val < 0.05 else "ns"
        log.info(f"  E1 vs {config}: t={t_stat:.3f}, p={p_val:.4f} [{sig}]")
        nb_results[config] = {"t_stat": round(t_stat, 4), "p_val": round(p_val, 4), "significant": p_val < 0.05}

    _save_results(all_results, RES_DIR, hparams, n_train, n_test, nb_results=nb_results)
    _write_summary_md(all_results, nb_results, RES_DIR, hparams, n_folds)


def _save_results(all_results, RES_DIR, hparams, n_train, n_test, nb_results=None):
    out = {
        "fold_scores": all_results,
        "summary": {
            config: {
                "mean_auroc": round(float(np.mean(scores)), 4),
                "std_auroc":  round(float(np.std(scores)), 4),
                "min_auroc":  round(float(np.min(scores)), 4),
                "max_auroc":  round(float(np.max(scores)), 4),
            }
            for config, scores in all_results.items()
        },
        "hparams": hparams,
        "n_train": n_train,
        "n_test": n_test,
    }
    if nb_results:
        out["nadeau_bengio"] = nb_results
    (RES_DIR / "experiment_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


def _write_summary_md(all_results, nb_results, RES_DIR, hparams, n_folds):
    lines = [
        "# BioFusionFormer Experiment Results (E1-E8)",
        "",
        f"**Evaluation:** {n_folds}-fold repeated CV (5x5) on TARGET-WT, N=121",
        f"**Metric:** AUROC on held-out test fold",
        f"**Statistics:** Nadeau-Bengio corrected resampled paired t-test",
        "",
        "## Hyperparameters",
        "",
        "| Param | Value |",
        "|-------|-------|",
    ]
    for k, v in hparams.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Results",
        "",
        "| Config | Name | Mean AUROC | Std | vs E1 t-stat | vs E1 p-val | Sig? |",
        "|--------|------|-----------|-----|-------------|------------|------|",
    ]
    e1_mean = np.mean(all_results.get("E1", [0.0]))
    for config in EXPERIMENT_CONFIGS:
        if config not in all_results:
            continue
        scores = all_results[config]
        mean_a = np.mean(scores)
        std_a  = np.std(scores)
        if config == "E1":
            lines.append(f"| **{config}** | **{EXPERIMENT_NAMES[config]}** | **{mean_a:.4f}** | {std_a:.4f} | — | — | — |")
        else:
            nb = nb_results.get(config, {})
            t = nb.get("t_stat", "—")
            p = nb.get("p_val", "—")
            sig = "✅" if nb.get("significant") else "—"
            lines.append(f"| {config} | {EXPERIMENT_NAMES[config]} | {mean_a:.4f} | {std_a:.4f} | {t} | {p} | {sig} |")

    lines += [
        "",
        "## Model Architecture",
        "",
        "`",
        "BioFusionFormer(d_model=32, n_heads=4, n_layers=2, d_ff=128)",
        "  ├── ModalityProjection × 3  (MaskedLinear -> LN -> GELU -> Dropout)",
        "  ├── pathway_embed           (Linear 1 -> 32)",
        "  ├── modality_embed          (Embedding M=3 x 32)",
        "  ├── GatedModalityFusion     (softmax gate over 3 modalities)",
        "  ├── PathwayTransformerLayer × 2  (BiasedMHA + FFN, pre-LN)",
        "  └── Classifier MLP          (32 -> 16 -> 1)",
        "`",
        "",
    ]

    (RES_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote results to {RES_DIR}/summary.md")


if __name__ == "__main__":
    main()
