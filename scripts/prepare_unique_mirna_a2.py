"""
BioFusionFormer - Step A2: Collapse Duplicate miRNA Features & Build Union Masks
================================================================================
1. Identifies and collapses the 762 duplicate miRNA expression profiles across 121 patients
   into 1,035 unique precursor-level features.
2. Constructs the union mask for each unique feature across all constituent mature miRNAs
   for both strong-evidence and broad-evidence Reactome pathway priors.
3. Audits pathway link counts for strong and broad evidence.
4. Implements a no-miRNA-prior fallback: features with no pathway edges are assigned
   to an explicit 'unmapped_mirna' token index, preventing silent data loss.
5. Saves:
   - data/processed/mirna_unique.parquet
   - data/interim/priors/mirna_unique_pathway_mask_strong.npz
   - data/interim/priors/mirna_unique_pathway_mask_broad.npz
   - data/interim/priors/mirna_unique_features.json
   - data/interim/priors/mirna_unique_mapping.json
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

def run_mirna_collapse():
    print("Loading processed miRNA matrix and priors...")
    mirna_df = pd.read_parquet(DATA_DIR / "processed" / "mirna.parquet")
    print(f"Original miRNA shape: {mirna_df.shape}")

    # 1. Group columns by identical expression profile across all 121 samples
    col_profiles = [tuple(mirna_df[c].values) for c in mirna_df.columns]
    profile_to_cols = defaultdict(list)
    for col_name, prof in zip(mirna_df.columns, col_profiles):
        profile_to_cols[prof].append(col_name)

    print(f"Unique expression profiles: {len(profile_to_cols)}")

    # For each unique profile, define representative name:
    # If multiple mature miRNAs share the profile, join them or pick the stem-loop / primary name
    unique_feature_names = []
    mapping_dict = {}
    unique_cols_data = {}

    for i, (prof, col_list) in enumerate(profile_to_cols.items()):
        # Primary name: first column name (e.g. hsa-miR-125b-5p)
        rep_name = col_list[0] if len(col_list) == 1 else f"{col_list[0]}+coll_{len(col_list)}"
        unique_feature_names.append(rep_name)
        mapping_dict[rep_name] = col_list
        unique_cols_data[rep_name] = mirna_df[col_list[0]].values

    mirna_unique_df = pd.DataFrame(unique_cols_data, index=mirna_df.index)
    print(f"Collapsed miRNA matrix shape: {mirna_unique_df.shape}")

    # Save unique miRNA matrix
    out_parquet = DATA_DIR / "processed" / "mirna_unique.parquet"
    mirna_unique_df.to_parquet(out_parquet)
    print(f"Saved {out_parquet}")

    # 2. Build union masks
    # Load original mature miRNA masks
    mask_strong_full = sp.load_npz(DATA_DIR / "interim" / "priors" / "mirna_pathway_mask.npz")
    mask_broad_full = sp.load_npz(DATA_DIR / "interim" / "priors" / "mirna_pathway_mask_broad.npz")
    mirna_names = json.load(open(DATA_DIR / "interim" / "priors" / "mirna_pathway_mask_mirnas.json"))
    mirna_map = {m: i for i, m in enumerate(mirna_names)}

    N_pathways = mask_strong_full.shape[1]
    N_unique = len(unique_feature_names)

    # We construct union masks: (N_unique, N_pathways + 1)
    # where the last column index (N_pathways) is the 'unmapped_mirna' fallback token!
    mask_unique_strong = np.zeros((N_unique, N_pathways + 1), dtype=bool)
    mask_unique_broad = np.zeros((N_unique, N_pathways + 1), dtype=bool)

    for u_idx, rep_name in enumerate(unique_feature_names):
        constituent_cols = mapping_dict[rep_name]
        # Union across all constituent mature miRNAs
        c_indices = [mirna_map[c] for c in constituent_cols if c in mirna_map]
        if len(c_indices) > 0:
            strong_sub = mask_strong_full[c_indices, :].toarray()
            broad_sub = mask_broad_full[c_indices, :].toarray()

            union_strong = (strong_sub.sum(axis=0) > 0)
            union_broad = (broad_sub.sum(axis=0) > 0)

            mask_unique_strong[u_idx, :N_pathways] = union_strong
            mask_unique_broad[u_idx, :N_pathways] = union_broad

        # No-miRNA-prior fallback: if feature has no pathway link in strong mask, connect to unmapped token
        if mask_unique_strong[u_idx, :N_pathways].sum() == 0:
            mask_unique_strong[u_idx, N_pathways] = True  # fallback token

        # If feature has no pathway link in broad mask, connect to unmapped token
        if mask_unique_broad[u_idx, :N_pathways].sum() == 0:
            mask_unique_broad[u_idx, N_pathways] = True

    # Audit pathway link counts
    has_strong_link = (mask_unique_strong[:, :N_pathways].sum(axis=1) > 0)
    has_broad_link = (mask_unique_broad[:, :N_pathways].sum(axis=1) > 0)
    unmapped_strong = mask_unique_strong[:, N_pathways]
    unmapped_broad = mask_unique_broad[:, N_pathways]

    print("\n=== Unique miRNA Pathway Link Audit ===")
    print(f"Total unique features: {N_unique}")
    print(f"Features with >=1 Reactome pathway link (Strong): {has_strong_link.sum()} / {N_unique} ({100*has_strong_link.mean():.1f}%)")
    print(f"Features mapped to 'unmapped' fallback (Strong): {unmapped_strong.sum()} / {N_unique} ({100*unmapped_strong.mean():.1f}%)")
    print(f"Features with >=1 Reactome pathway link (Broad): {has_broad_link.sum()} / {N_unique} ({100*has_broad_link.mean():.1f}%)")
    print(f"Features mapped to 'unmapped' fallback (Broad): {unmapped_broad.sum()} / {N_unique} ({100*unmapped_broad.mean():.1f}%)")

    # Save new masks and mappings
    sp.save_npz(DATA_DIR / "interim" / "priors" / "mirna_unique_pathway_mask_strong.npz", sp.csr_matrix(mask_unique_strong))
    sp.save_npz(DATA_DIR / "interim" / "priors" / "mirna_unique_pathway_mask_broad.npz", sp.csr_matrix(mask_unique_broad))

    with open(DATA_DIR / "interim" / "priors" / "mirna_unique_features.json", "w") as f:
        json.dump(unique_feature_names, f, indent=2)

    with open(DATA_DIR / "interim" / "priors" / "mirna_unique_mapping.json", "w") as f:
        json.dump(mapping_dict, f, indent=2)

    print("All unique miRNA artifacts and fallback masks saved successfully.")

if __name__ == "__main__":
    run_mirna_collapse()
