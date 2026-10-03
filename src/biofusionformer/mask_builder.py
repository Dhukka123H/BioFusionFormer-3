"""
BioFusionFormer - Mask Builder
==============================
Constructs binary (P, F_mod) masks from Reactome gene-set membership data.

For each modality:
  - mRNA / meth-gene: feature columns are Ensembl IDs; gene-set lists Ensembl IDs
  - miRNA: feature columns are mature miRNA names; use miRNA-target gene mapping
    (mirna2gene_strong.parquet) then pathway membership of those target genes

All masks are derived from prior annotation, never from dataset labels.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


class MaskBuilder:
    """
    Builds modality-specific pathway masks.

    Parameters
    ----------
    reactome_dir : Path
        Directory containing the Reactome gene-set files
        (e.g. data/interim/reactome/).
    mirna2gene_path : Path
        Parquet file with columns ['miRNA', 'Target Gene']
        mapping mature miRNA names to Ensembl gene IDs.
    """

    def __init__(self, reactome_dir: Path, mirna2gene_path: Path):
        self.reactome_dir    = Path(reactome_dir)
        self.mirna2gene_path = Path(mirna2gene_path)
        self._gene_sets: Optional[Dict[str, List[str]]] = None
        self._mirna2genes: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # Internal loaders (cached)
    # ------------------------------------------------------------------

    def _load_gene_sets(self) -> Dict[str, List[str]]:
        """Return {pathway_id: [ensembl_gene_id, ...]}."""
        if self._gene_sets is not None:
            return self._gene_sets
        # Prefer a pre-built parquet; fall back to TSV
        gs_parquet = self.reactome_dir / "pathway_gene_sets.parquet"
        gs_tsv = self.reactome_dir / "ReactomePathways.gmt"
        if gs_parquet.exists():
            df = pd.read_parquet(gs_parquet)
            # Expected columns: pathway_id, ensembl_ids (list column) or long-form
            if "ensembl_id" in df.columns and "pathway_id" in df.columns:
                self._gene_sets = df.groupby("pathway_id")["ensembl_id"].apply(list).to_dict()
            elif "pathway_id" in df.columns:
                # Wide form: pathway_id, then gene columns
                gene_cols = [c for c in df.columns if c != "pathway_id"]
                self._gene_sets = {
                    row["pathway_id"]: [row[c] for c in gene_cols if pd.notna(row[c])]
                    for _, row in df.iterrows()
                }
        elif gs_tsv.exists():
            gene_sets = {}
            with open(gs_tsv) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        gene_sets[parts[0]] = parts[2:]
            self._gene_sets = gene_sets
        else:
            # Try any available parquet or csv in reactome_dir
            for p in self.reactome_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(p)
                    if "pathway_id" in df.columns and "ensembl_id" in df.columns:
                        self._gene_sets = df.groupby("pathway_id")["ensembl_id"].apply(list).to_dict()
                        break
                except Exception:
                    continue
        if self._gene_sets is None:
            raise FileNotFoundError(
                f"Cannot locate Reactome gene-set file in {self.reactome_dir}"
            )
        return self._gene_sets

    def _load_mirna2genes(self) -> Dict[str, List[str]]:
        """Return {mirna_name: [ensembl_gene_id, ...]}."""
        if self._mirna2genes is not None:
            return self._mirna2genes
        df = pd.read_parquet(self.mirna2gene_path)
        # Determine column names
        mirna_col = next((c for c in df.columns if "mirna" in c.lower() or "mirna" in c.lower()), df.columns[0])
        gene_col  = next((c for c in df.columns if "gene" in c.lower() or "target" in c.lower()), df.columns[1])
        self._mirna2genes = df.groupby(mirna_col)[gene_col].apply(list).to_dict()
        return self._mirna2genes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_mask(
        self,
        pathway_order: List[str],
        feature_ids: List[str],
        modality: str,
    ) -> torch.BoolTensor:
        """
        Build a (P, F) boolean mask.

        Parameters
        ----------
        pathway_order : list[str]  ordered pathway IDs
        feature_ids   : list[str]  ordered feature column names for the modality
        modality      : str        one of "mrna", "meth", "mirna"

        Returns
        -------
        mask : BoolTensor of shape (P, F)
        """
        gene_sets = self._load_gene_sets()
        P = len(pathway_order)
        F = len(feature_ids)
        mask = torch.zeros(P, F, dtype=torch.bool)

        feat_idx = {fid: i for i, fid in enumerate(feature_ids)}

        if modality in ("mrna", "meth"):
            # Feature IDs are Ensembl gene IDs; direct lookup
            for pi, pw_id in enumerate(pathway_order):
                genes_in_pathway = set(gene_sets.get(pw_id, []))
                for gene in genes_in_pathway:
                    fi = feat_idx.get(gene)
                    if fi is not None:
                        mask[pi, fi] = True

        elif modality == "mirna":
            # Feature IDs are mature miRNA names; map to target genes then to pathways
            mirna2genes = self._load_mirna2genes()
            for pi, pw_id in enumerate(pathway_order):
                genes_in_pathway = set(gene_sets.get(pw_id, []))
                for mirna_name, target_genes in mirna2genes.items():
                    fi = feat_idx.get(mirna_name)
                    if fi is None:
                        continue
                    if genes_in_pathway & set(target_genes):
                        mask[pi, fi] = True

        else:
            raise ValueError(f"Unknown modality: {modality!r}")

        n_active = int(mask.sum().item())
        density  = n_active / (P * F) * 100
        print(f"  [{modality}] mask: {P}x{F}, active={n_active} ({density:.2f}%)")

        # Fallback: if mask is all-zero (e.g. Ensembl ID mismatch), warn prominently
        if n_active == 0:
            import warnings
            warnings.warn(
                f"[MaskBuilder] {modality} mask is entirely zero! "
                "Check Ensembl ID format in gene_sets vs feature_ids.",
                RuntimeWarning,
            )

        return mask

    def build_all_masks(
        self,
        pathway_order: List[str],
        mrna_features: Optional[List[str]] = None,
        mirna_features: Optional[List[str]] = None,
        meth_features: Optional[List[str]] = None,
    ) -> Dict[str, torch.BoolTensor]:
        """
        Build all masks at once. Any None modality is skipped.
        """
        masks = {}
        if mrna_features is not None:
            masks["mrna"] = self.build_mask(pathway_order, mrna_features, "mrna")
        if mirna_features is not None:
            masks["mirna"] = self.build_mask(pathway_order, mirna_features, "mirna")
        if meth_features is not None:
            masks["meth"] = self.build_mask(pathway_order, meth_features, "meth")
        return masks
