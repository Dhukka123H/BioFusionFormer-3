"""
src/inference/predict.py
========================
Inference pipeline for BioFusionFormer (Stage 6.2).

API function:
    predict(mrna_df, mirna_df, meth_df, model=None, priors_dir=None) -> pd.DataFrame

Columns returned:
    ['patient_id', 'probability_unfavorable', 'gate_mrna', 'gate_mirna', 'gate_meth', 'top_pathways_IG', 'warnings']

Design rules:
  - All three omics are REQUIRED (refuses with clear error message if any is missing).
  - Strict validation: Ensembl IDs without version suffix, non-negative log2(count+1) for mRNA,
    non-negative log2(RPM+1) for miRNA, beta in [0, 1] for methylation.
  - Coverage check: refuses if < 90% of expected training features are present.
  - Aligns feature order strictly to training specifications.
  - Never saves uploaded data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent


def validate_omics_inputs(
    mrna_df: Optional[pd.DataFrame],
    mirna_df: Optional[pd.DataFrame],
    meth_df: Optional[pd.DataFrame],
    expected_features: Dict[str, List[str]],
    min_feature_coverage: float = 0.90,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Validates the three input DataFrames.
    Returns:
        (aligned_dfs_dict, warnings_list)
    """
    warnings = []

    # 1. Require all three omics
    missing_mods = []
    if mrna_df is None or not isinstance(mrna_df, pd.DataFrame) or mrna_df.empty:
        missing_mods.append("mRNA (mrna_df)")
    if mirna_df is None or not isinstance(mirna_df, pd.DataFrame) or mirna_df.empty:
        missing_mods.append("miRNA (mirna_df)")
    if meth_df is None or not isinstance(meth_df, pd.DataFrame) or meth_df.empty:
        missing_mods.append("DNA Methylation (meth_df)")

    if missing_mods:
        raise ValueError(
            f"Missing required modality: {', '.join(missing_mods)}. "
            f"BioFusionFormer requires all three omics (mRNA, miRNA, methylation) to predict."
        )

    dfs = {"mrna": mrna_df, "mirna": mirna_df, "meth": meth_df}

    # Verify matching patient indices / IDs across modalities
    pats = list(mrna_df.index)
    if list(mirna_df.index) != pats or list(meth_df.index) != pats:
        # Check intersection
        common_pats = [p for p in pats if p in mirna_df.index and p in meth_df.index]
        if len(common_pats) == 0:
            raise ValueError("No common patient IDs found across mRNA, miRNA, and methylation tables.")
        if len(common_pats) < len(pats):
            warnings.append(f"Subset of {len(common_pats)}/{len(pats)} patients matched across all 3 modalities.")
        # Re-index to common patients
        dfs = {m: d.loc[common_pats] for m, d in dfs.items()}

    # 2. Check Ensembl ID format (no version suffix)
    for mod in ["mrna", "meth"]:
        versioned = [c for c in dfs[mod].columns if "." in str(c) and str(c).startswith("ENSG")]
        if len(versioned) > 0:
            sample_ver = versioned[:3]
            raise ValueError(
                f"Invalid feature identifiers in {mod}: Ensembl gene IDs must NOT include version suffixes "
                f"(found e.g. {sample_ver}). Please strip '.XX' version extensions (e.g. 'ENSG00000141510.15' -> 'ENSG00000141510')."
            )

    # 3. Check value ranges
    # mRNA: log2(count + 1) -> >= 0
    mrna_min = dfs["mrna"].values.min()
    if mrna_min < -0.1:
        raise ValueError(f"mRNA contains negative values (min = {mrna_min:.3f}). Expected log2(STAR_count + 1) >= 0.")

    # miRNA: log2(RPM + 1) -> >= 0
    mirna_min = dfs["mirna"].values.min()
    if mirna_min < -0.1:
        raise ValueError(f"miRNA contains negative values (min = {mirna_min:.3f}). Expected log2(RPM + 1) >= 0.")

    # Methylation: beta values in [0, 1]
    meth_min, meth_max = dfs["meth"].values.min(), dfs["meth"].values.max()
    if meth_min < -0.05 or meth_max > 1.05:
        raise ValueError(
            f"Methylation beta values out of range: min={meth_min:.3f}, max={meth_max:.3f}. "
            f"Expected promoter gene beta values bounded in [0.0, 1.0]."
        )

    # 4. Feature presence and alignment (refuse < 90% coverage)
    aligned_dfs = {}
    for mod, df in dfs.items():
        exp = expected_features[mod]
        n_exp = len(exp)
        present = [f for f in exp if f in df.columns]
        frac = len(present) / n_exp
        if frac < min_feature_coverage:
            missing_count = n_exp - len(present)
            raise ValueError(
                f"Insufficient feature coverage for {mod}: {len(present)}/{n_exp} ({frac:.1%}) features present. "
                f"BioFusionFormer requires >= {min_feature_coverage:.0%} coverage ({missing_count} missing). "
                f"Missing features include: {exp[:3]}..."
            )
        if frac < 1.0:
            warnings.append(f"{mod}: {len(present)}/{n_exp} features present ({frac:.1%}). Missing filled with 0.")

        # Align columns
        aligned = df.reindex(columns=exp, fill_value=0.0)
        aligned_dfs[mod] = aligned

    return aligned_dfs, warnings


def predict(
    mrna_df: pd.DataFrame,
    mirna_df: pd.DataFrame,
    meth_df: pd.DataFrame,
    model: Optional[nn.Module] = None,
    weights_path: Optional[Union[str, Path]] = None,
    priors_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Predicts unfavorable Wilms tumor histology probabilities for a cohort of patients.

    Returns:
        pd.DataFrame with columns:
        ['patient_id', 'probability_unfavorable', 'gate_mrna', 'gate_mirna', 'gate_meth', 'top_pathways_IG', 'warnings']
    """
    if priors_dir is None:
        priors_dir = ROOT / "data" / "interim" / "priors"
    else:
        priors_dir = Path(priors_dir)

    processed_dir = ROOT / "data" / "processed"

    # Load expected feature lists
    if (processed_dir / "mrna_gene.parquet").exists():
        exp_mrna = pd.read_parquet(processed_dir / "mrna_gene.parquet").columns.tolist()
        exp_mirna = pd.read_parquet(processed_dir / "mirna_unique.parquet").columns.tolist()
        exp_meth = pd.read_parquet(processed_dir / "meth_gene.parquet").columns.tolist()
    else:
        # Fallback to json gene names
        exp_mrna = json.load(open(priors_dir / "gene_pathway_mask_genes.json"))
        exp_mirna = [f"mirna_{i}" for i in range(1035)]
        exp_meth = exp_mrna

    expected_features = {
        "mrna": exp_mrna,
        "mirna": exp_mirna,
        "meth": exp_meth,
    }

    # Validate
    aligned_dfs, warnings = validate_omics_inputs(mrna_df, mirna_df, meth_df, expected_features)
    patient_ids = list(aligned_dfs["mrna"].index)
    N = len(patient_ids)

    # Impute missing values (median) + z-score
    norm_mats = {}
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    for mod, df in aligned_dfs.items():
        vals = df.values.astype(np.float32)
        imp = SimpleImputer(strategy="median")
        vals_imp = imp.fit_transform(vals)
        sc = StandardScaler()
        norm_mats[mod] = sc.fit_transform(vals_imp)

    # If model is not supplied, build it
    if model is None:
        from src.biofusionformer.full_model import BioFusionFormerModel
        from src.pathways.selection import select_unsupervised_pathways

        selected_pways, _ = select_unsupervised_pathways(ROOT / "data", top_k=192)
        P = len(selected_pways)
        prior_pathways = json.load(open(priors_dir / "gene_pathway_mask_pathways.json"))
        pway_map = {p: i for i, p in enumerate(prior_pathways)}
        pway_idx = [pway_map[p] for p in selected_pways]

        mask_gene_full = sp.load_npz(priors_dir / "gene_pathway_mask.npz")
        gene_names = json.load(open(priors_dir / "gene_pathway_mask_genes.json"))
        gene_map = {g: i for i, g in enumerate(gene_names)}
        mask_mrna = mask_gene_full[[gene_map[c] for c in exp_mrna], :][:, pway_idx].T.toarray()
        mask_meth = mask_gene_full[[gene_map[c] for c in exp_meth], :][:, pway_idx].T.toarray()

        sub_mask_strong = sp.load_npz(priors_dir / "mirna_unique_pathway_mask_strong.npz")[:, pway_idx].toarray()
        zero_strong = sub_mask_strong.sum(axis=1) == 0
        sub_mask_strong[zero_strong, P - 1] = True
        mask_mirna_strong = sub_mask_strong.T

        masks = {
            "mrna": torch.tensor(mask_mrna, dtype=torch.bool),
            "mirna": torch.tensor(mask_mirna_strong, dtype=torch.bool),
            "meth": torch.tensor(mask_meth, dtype=torch.bool),
        }
        feat_dims = {"mrna": len(exp_mrna), "mirna": len(exp_mirna), "meth": len(exp_meth)}

        model = BioFusionFormerModel(
            modality_features=feat_dims,
            modality_masks=masks,
            n_pathways=P,
            d_model=32, n_heads=4, n_layers=2, d_ff=128,
            dropout=0.3, attn_dropout=0.2, modality_dropout=0.2,
            use_pathway_bias=True, use_gating=True,
        )

        # Load weights if provided or if saved model exists
        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            # Check for final model or existing fold_00 model
            final_pt = ROOT / "models" / "final" / "model_seed42.pt"
            fold_pt = ROOT / "models" / "e1_folds_final_v1_repro" / "fold_00_seed42.pt"
            chosen_pt = final_pt if final_pt.exists() else (fold_pt if fold_pt.exists() else None)
            if chosen_pt is not None:
                state = torch.load(chosen_pt, map_location="cpu")
                model.load_state_dict(state, strict=False)

    model.eval()
    inputs = {m: torch.tensor(norm_mats[m], dtype=torch.float32) for m in norm_mats}

    with torch.no_grad():
        out = model(inputs)
        probs = torch.sigmoid(out["logit"]).cpu().numpy()
        gates = out["gate"].cpu().numpy()

    # If 1D prob array
    if probs.ndim == 0:
        probs = np.array([probs])

    # Build output dataframe
    warn_str = "; ".join(warnings) if warnings else "None"
    records = []
    for i, pid in enumerate(patient_ids):
        p_val = float(probs[i])
        g_mrna = float(gates[i, 0])
        g_mirna = float(gates[i, 1])
        g_meth = float(gates[i, 2])
        records.append({
            "patient_id": pid,
            "probability_unfavorable": p_val,
            "gate_mrna": g_mrna,
            "gate_mirna": g_mirna,
            "gate_meth": g_meth,
            "top_pathways_IG": "N/A (CLI mode)",
            "warnings": warn_str,
        })

    return pd.DataFrame(records)
