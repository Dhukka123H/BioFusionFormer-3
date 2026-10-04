# BioFusionFormer - Priors & Sparse Masks Report

Generated: 2026-10-03 06:29:31 UTC

## 1. Executive Summary
This report summarizes the construction of hierarchical prior masks connecting multi-modal features (mRNA genes, miRNA expressions, DNA methylation CpG probes) to Reactome biological pathways and the pathway directed acyclic graph (DAG).

All masks are constructed as memory-efficient **SciPy Compressed Sparse Row (CSR)** matrices (`.npz`) accompanied by deterministic `.json` string index maps.

---

## 2. Sparse Prior Masks Summary Table

| Mask Name | Dimensions (Rows x Cols) | Non-Zero Entries (nnz) | Density (%) | Unlinked Entities | Unlinked (%) | Storage Format |
|---|---|---|---|---|---|---|
| **`pathway_adjacency`** | 2,883 x 2,883 | 2,899 | 0.0349% | 0 pathways (DAG verified) | 0.00% | `.npz` + `.json` |
| **`gene_pathway_mask`** (Unified) | 20,264 x 2,883 | 158,206 | 0.2708% | 7,191 genes | 35.49% | `.npz` + `.json` |
| **`gene_pathway_mask`** (Reactome Only) | 13,073 x 2,883 | 158,206 | 0.4198% | 0 genes | 0.00% | `.npz` |
| **`mirna_pathway_mask`** (Strong Evidence) | 3,124 x 2,883 | 129,271 | 1.4353% | 2,149 miRNAs | 68.79% | `.npz` + `.json` |
| **`cpg_gene_mask`** (HM450 -> ENSG) | 482,421 x 20,264 | 347,232 | 0.0036% | 162,091 probes (no gene) / 273,094 (no pathway) | 33.60% / 56.61% | `.npz` + `.json` |

---

## 3. Structural Analysis & Biological Findings

### A. Pathway Adjacency DAG (`pathway_adjacency.npz`)
- **Nodes**: 2,883 Reactome pathways.
- **Edges**: 2,899 directed parent -> child relationships.
- **Topological Integrity**: Verified cycle-free DAG (0 cycles, 0 orphans).
- **Function in BioFusionFormer**: Guides hierarchical graph pooling and attention between parent and child pathway modules.

### B. Gene-Pathway Mask (`gene_pathway_mask.npz`)
- **Gene Universe**: 20,264 ENSG genes (union of Reactome all-levels genes and HM450 mapped target genes).
- **Pathways**: 2,883 Reactome pathways.
- **Active Connections**: 158,206 gene-pathway links derived from Reactome all-levels expansion.
- **Unlinked Genes**: 7,191 genes (35.49%) are known genomic loci represented on methylation arrays that do not currently have annotated pathways in Reactome.

### C. miRNA Prior Slimming & Pathway Mask (`mirna_pathway_mask.npz`)
- **miRTarBase Prior Pruning**:
  - Filtered raw miRTarBase 10.0 (4,004,967 rows) down to high-confidence **Strong Experimental Evidence** rows (`Support Type == 'Functional MTI'` validated by Reporter Assays, Western Blots, and qRT-PCR).
  - High-throughput weak evidence (CLIP-seq, PAR-CLIP, Microarrays) accounted for 99.4% of raw records and was isolated into `data/interim/mirna2gene_all.parquet`.
  - Strong evidence yielded 18,130 validated interactions across 1,052 miRNAs and 3,084 target genes.
- **miRNA Universe**: 3,124 human miRNAs (miRBase `hsa-` mature miRNAs + strong miRTarBase).
- **Coverage**:
  - Linked to pathways: 975 miRNAs (31.21%).
  - Unlinked miRNAs: 2,149 miRNAs (68.79%) lack strong low-throughput experimental targets in Reactome pathways.

### D. CpG-Gene Mask (`cpg_gene_mask.npz`)
- **Probes**: 482,421 Illumina HumanMethylation450 probes.
- **Target Genes**: 20,264 ENSG genes mapped via official Illumina manifest gene symbols and HGNC symbol-to-ENSG dictionary.
- **Coverage**:
  - Probes mapped to >= 1 gene: 320,330 (66.40%).
  - Intergenic probes without gene annotation: 162,091 (33.60%).
  - Probes linking through to a Reactome pathway: 209,327 (43.39%).
- **Hierarchical Composition**:
  - Downstream models compute CpG-to-Pathway routing directly via sparse matrix multiplication:
    $$\mathbf{M}_{\text{CpG}\to\text{Pathway}} = \mathbf{M}_{\text{CpG}\to\text{Gene}} \times \mathbf{M}_{\text{Gene}\to\text{Pathway}}$$

---

## 4. Prior Files Created
All files are saved in `data/interim/priors/`:
- `pathway_adjacency.npz` & `pathway_adjacency_pathways.json`
- `gene_pathway_mask.npz` & `gene_pathway_mask_genes.json` & `gene_pathway_mask_pathways.json`
- `gene_pathway_mask_reactome_only.npz`
- `mirna_pathway_mask.npz` & `mirna_pathway_mask_mirnas.json` & `mirna_pathway_mask_pathways.json`
- `cpg_gene_mask.npz` & `cpg_gene_mask_cpgs.json` & `cpg_gene_mask_genes.json`
