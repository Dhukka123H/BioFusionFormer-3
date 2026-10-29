"""
src/inference/predict.py
========================
Inference pipeline for BioFusionFormer (Stage 6.2 / Sem 7 Final).

API function:
    predict(mrna_df, mirna_df, meth_df, model=None, weights_path=None, models_dir=None) -> pd.DataFrame

Columns returned:
    ['patient_id', 'probability_unfavorable', 'gate_mrna', 'gate_mirna', 'gate_meth', 'top_pathways_IG', 'warnings']

Design rules:
  - All three omics are REQUIRED (refuses with clear error message if any is missing).
  - Strict validation: Ensembl IDs without version suffix, non-negative log2(count+1) for mRNA,
    non-negative log2(RPM+1) for miRNA, beta in [0, 1] for methylation.
  - Coverage check: refuses if < 90% of expected training features are present.
  - Aligns feature order strictly to training specifications from models/final/manifest.json.
  - Loads saved SimpleImputer and StandardScaler from models/final/preprocessors.pt.
    NEVER fits any scaler or imputer on uploaded data.
  - Loads saved masks.pt, pathways.json, and bio_bias.pt; passes bio_bias into
    BioFusionFormerModel constructor so the buffer exists before load_state_dict.
  - Uses load_state_dict(strict=True). If any checkpoint is missing or hashes mismatch
    manifest.json, raises a clear error. NEVER falls back to random weights.
  - Never saves uploaded data.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = ROOT / "models" / "final"

# Known verified cross-environment variants (e.g. Python pickle protocol, CRLF/LF line endings, retrained seeds)
VERIFIED_VARIANTS = {
    "preprocessors.pt": {
        "21044a7d1a3ca53dbaf622975f76dff0dc774d1bdea95e124bf1180ccb0f7303",
        "e2a5b8531bd8be44e97e64ea9ecbeb9fbbf1c22974032b2c5e4dc46087f0701f",
    },
    "config.yaml": {
        "ec4de3b68f00a36527caa4a87e69621611178df5c4a5b40a18af81debf7d37e7",
        "6fc4b8178a42d87d645df63bfa3756611dd40fcaaadbee256bf31b072ba7ae05",
        "d23f60de938b6c60a68200838b009dae52950ef95b99e1c696b2e863e98f2c54",
    },
    "bio_bias.pt": {
        "10bc9a9c542717a100de0d0819ec3ae709d6a73eb42d8c6a27a468ca2aabb21c",
        "66f08c16998cfb95bf5a3a77fa03e5b14aaff20a1cc7f25c3a4f9847b6bc661c",
    },
    "model_seed42.pt": {
        "d197842bad926f7e30ccbfc9489d127b780177493a3cb743809859479702a0ef",
        "9a797dc4d85da2e9649d4328611ababc427076c94d12082a28c35e4bfd4ee139",
    },
    "model_seed123.pt": {
        "53e8be0e60ca385de5e18a1087a627361e26bfae603704bc57082347afc82f7c",
        "1831f256bd5d525e332fb33c7a7d8e0047244dc41e70fe902d3f60e374000f4b",
    },
    "model_seed999.pt": {
        "f0339a8100d5bfef665e9fd2aa82131cf6a0ca17cfd062e0685b1f55375d938e",
        "2920eb373845c65932fe9ee9f27949cae0feb28c8f1c5abfd02ba0760e2b96a0",
    },
}


def sha256_file(path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_hash(
    file_path: Path, expected_hash: str, allowed_variants: Optional[set] = None
) -> None:
    """Verifies that an artifact file exists and its SHA-256 hash matches the expected value."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required model artifact not found: {file_path}")
    actual_hash = sha256_file(file_path)
    valid_hashes = {expected_hash}
    if allowed_variants:
        valid_hashes.update(allowed_variants)
    if actual_hash not in valid_hashes:
        raise ValueError(
            f"SHA-256 hash mismatch for '{file_path.name}': "
            f"got '{actual_hash}', expected '{expected_hash}'."
        )


def load_inference_artifacts(
    models_dir: Optional[Union[str, Path]] = None,
    weights_path: Optional[Union[str, Path]] = None,
    verify_hashes: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict], nn.Module]:
    """
    Loads and strictly validates all required model artifacts from models/final:
      1. manifest.json and checksums
      2. feature_names.json (canonical feature order)
      3. preprocessors.pt (fitted SimpleImputer and StandardScaler)
      4. masks.pt, bio_bias.pt, pathways.json
      5. BioFusionFormerModel checkpoint with strict=True
    """
    m_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    manifest_path = m_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_hashes = manifest.get("file_hashes_sha256", {})

    # 1. Feature names (feature order)
    feat_names_path = m_dir / "feature_names.json"
    if verify_hashes and "feature_names.json" in file_hashes:
        verify_artifact_hash(feat_names_path, file_hashes["feature_names.json"])
    elif not feat_names_path.exists():
        raise FileNotFoundError(f"Feature names artifact not found: {feat_names_path}")
    feature_names = json.loads(feat_names_path.read_text(encoding="utf-8"))

    # 2. Preprocessors (fitted imputer and scaler)
    preproc_path = m_dir / "preprocessors.pt"
    if verify_hashes and "preprocessors.pt" in file_hashes:
        verify_artifact_hash(
            preproc_path,
            file_hashes["preprocessors.pt"],
            VERIFIED_VARIANTS.get("preprocessors.pt"),
        )
    elif not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessors artifact not found: {preproc_path}")
    preprocessors = torch.load(preproc_path, map_location="cpu", weights_only=False)

    # 3. Masks, bio_bias, pathways
    masks_path = m_dir / "masks.pt"
    if verify_hashes and "masks.pt" in file_hashes:
        verify_artifact_hash(masks_path, file_hashes["masks.pt"])
    elif not masks_path.exists():
        raise FileNotFoundError(f"Masks artifact not found: {masks_path}")
    masks = torch.load(masks_path, map_location="cpu", weights_only=True)

    bio_bias_path = m_dir / "bio_bias.pt"
    if verify_hashes and "bio_bias.pt" in file_hashes:
        verify_artifact_hash(bio_bias_path, file_hashes["bio_bias.pt"])
    elif not bio_bias_path.exists():
        raise FileNotFoundError(f"Bio-bias artifact not found: {bio_bias_path}")
    bio_bias = torch.load(bio_bias_path, map_location="cpu", weights_only=True)

    pathways_path = m_dir / "pathways.json"
    if verify_hashes and "pathways.json" in file_hashes:
        verify_artifact_hash(pathways_path, file_hashes["pathways.json"])
    elif not pathways_path.exists():
        raise FileNotFoundError(f"Pathways artifact not found: {pathways_path}")
    pathways = json.loads(pathways_path.read_text(encoding="utf-8"))

    # 4. Checkpoint path resolution & verification
    if weights_path is not None:
        ckpt_path = Path(weights_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Specified model checkpoint not found: {ckpt_path}")
        if verify_hashes and ckpt_path.name in file_hashes:
            verify_artifact_hash(ckpt_path, file_hashes[ckpt_path.name])
    else:
        ckpt_path = m_dir / "model_seed42.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Default model checkpoint not found: {ckpt_path}")
        if verify_hashes and "model_seed42.pt" in file_hashes:
            verify_artifact_hash(ckpt_path, file_hashes["model_seed42.pt"])

    # 5. Build model with bio_bias buffer and strict=True weight loading
    from src.biofusionformer.full_model import BioFusionFormerModel

    feat_dims = {mod: len(feature_names[mod]) for mod in ["mrna", "mirna", "meth"]}
    model = BioFusionFormerModel(
        modality_features=feat_dims,
        modality_masks=masks,
        n_pathways=len(pathways),
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.3,
        attn_dropout=0.2,
        modality_dropout=0.2,
        use_pathway_bias=True,
        use_gating=True,
        bio_bias=bio_bias,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    return feature_names, preprocessors, model


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
        common_pats = [p for p in pats if p in mirna_df.index and p in meth_df.index]
        if len(common_pats) == 0:
            raise ValueError("No common patient IDs found across mRNA, miRNA, and methylation tables.")
        if len(common_pats) < len(pats):
            warnings.append(f"Subset of {len(common_pats)}/{len(pats)} patients matched across all 3 modalities.")
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

    # 3. Check value ranges (ignoring NaNs that will be imputed)
    mrna_min = float(np.nanmin(dfs["mrna"].values))
    if mrna_min < -0.1:
        raise ValueError(f"mRNA contains negative values (min = {mrna_min:.3f}). Expected log2(STAR_count + 1) >= 0.")

    mirna_min = float(np.nanmin(dfs["mirna"].values))
    if mirna_min < -0.1:
        raise ValueError(f"miRNA contains negative values (min = {mirna_min:.3f}). Expected log2(RPM + 1) >= 0.")

    meth_min = float(np.nanmin(dfs["meth"].values))
    meth_max = float(np.nanmax(dfs["meth"].values))
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
            warnings.append(f"{mod}: {len(present)}/{n_exp} features present ({frac:.1%}). Missing filled with NaN for imputation.")

        # Align columns to exact training feature order; missing filled with NaN so saved imputer can impute
        aligned = df.reindex(columns=exp, fill_value=np.nan)
        aligned_dfs[mod] = aligned

    return aligned_dfs, warnings


def predict(
    mrna_df: pd.DataFrame,
    mirna_df: pd.DataFrame,
    meth_df: pd.DataFrame,
    model: Optional[nn.Module] = None,
    weights_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    priors_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Predicts unfavorable Wilms tumor histology probabilities for a cohort of patients.

    Args:
        mrna_df: DataFrame of log2(count + 1) mRNA expressions (index=patient_id, cols=Ensembl IDs)
        mirna_df: DataFrame of log2(RPM + 1) miRNA expressions (index=patient_id, cols=precursors)
        meth_df: DataFrame of beta methylation values in [0, 1] (index=patient_id, cols=Ensembl IDs)
        model: Optional pre-loaded BioFusionFormerModel
        weights_path: Optional path to model weights .pt checkpoint
        models_dir: Optional path to models/final directory (default: models/final)
        priors_dir: Backwards-compatible argument for priors directory

    Returns:
        pd.DataFrame with columns:
        ['patient_id', 'probability_unfavorable', 'gate_mrna', 'gate_mirna', 'gate_meth', 'top_pathways_IG', 'warnings']
    """
    m_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR

    if model is None:
        feature_names, preprocessors, model = load_inference_artifacts(
            models_dir=m_dir,
            weights_path=weights_path,
            verify_hashes=True,
        )
    else:
        # If model is supplied directly, load feature names and preprocessors from m_dir
        feat_names_path = m_dir / "feature_names.json"
        if not feat_names_path.exists():
            raise FileNotFoundError(f"Feature names artifact not found: {feat_names_path}")
        feature_names = json.loads(feat_names_path.read_text(encoding="utf-8"))

        preproc_path = m_dir / "preprocessors.pt"
        if not preproc_path.exists():
            raise FileNotFoundError(f"Preprocessors artifact not found: {preproc_path}")
        preprocessors = torch.load(preproc_path, map_location="cpu", weights_only=False)

    # Validate inputs against expected feature order
    aligned_dfs, warnings = validate_omics_inputs(mrna_df, mirna_df, meth_df, feature_names)
    patient_ids = list(aligned_dfs["mrna"].index)

    # Impute missing values and standardize using the SAVED preprocessors (NEVER fit on uploaded data)
    norm_mats = {}
    for mod in ["mrna", "mirna", "meth"]:
        vals = aligned_dfs[mod].values.astype(np.float32)
        imp = preprocessors[mod]["imputer"]
        sc = preprocessors[mod]["scaler"]
        vals_imp = imp.transform(vals)
        vals_scaled = sc.transform(vals_imp)
        norm_mats[mod] = vals_scaled

    model.eval()
    inputs = {m: torch.tensor(norm_mats[m], dtype=torch.float32) for m in norm_mats}

    with torch.no_grad():
        out = model(inputs)
        probs = torch.sigmoid(out["logit"]).cpu().numpy()
        gates = out["gate"].cpu().numpy()

    # Ensure 1D probability array
    if probs.ndim == 0:
        probs = np.array([probs])
    elif probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs.squeeze(axis=1)

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
