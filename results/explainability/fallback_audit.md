# Fallback Audit: Pathway P-1 and miRNA Fallback Wiring

**Date**: 2026-10-19  
**File**: `results/explainability/fallback_audit.md`  
**Related**: Fork 6 (`results/model/forking_paths.md`), Finding F1 (`docs/AGENT_PROMPT_2.md`)

---

## 1. Executive Summary

In Stage 4 explainability analysis, the top pathway identified by Integrated Gradients attribution was **"Regulation of ornithine decarboxylase (ODC)"** (`R-HSA-350562`), with an attribution score of **4.98**, approximately double the runner-up pathway.

This audit confirms that this top finding is an **artefact of the Fork 6 fallback wiring**. Pathway `R-HSA-350562` happens to be index $P-1$ (the 192nd and final selected Reactome pathway). In the frozen E1 architecture, all 726 miRNA features lacking a prior edge to the selected 192 pathways were wired into pathway token $P-1$. Consequently, summing $|IG|$ over its member features accumulated attributions from 726 fallback-routed miRNAs in addition to genuine member features.

---

## 2. Quantitative Verification

| Parameter | Observed Value | Expected Value | Status |
|:---|:---:|:---:|:---:|
| Total selected pathways ($P$) | **192** | 192 | Verified |
| Selected pathway index $P-1$ ID | **`R-HSA-350562`** | `R-HSA-350562` | Verified |
| Pathway $P-1$ Reactome Name | **Regulation of ornithine decarboxylase (ODC)** | Regulation of ornithine decarboxylase (ODC) | Verified |
| Total miRNA features in dataset | **1,035** | 1,035 | Verified |
| miRNA features without prior edge | **726** | 726 | Verified |
| miRNA features routed by fallback to $P-1$ | **726** | 726 | Verified |
| Genuine miRNA edges to $P-1$ (prior) | **2** | 2 | Verified |
| Total miRNA edges to $P-1$ in model mask | **728** | 728 | Verified |
| mRNA member genes present in dataset | **38** | ~40 | Verified |
| DNA methylation member genes in dataset | **37** | ~40 | Verified |
| **Total genuine member features** | **77** | ~78 | Verified |
| **Total features with fallback wiring** | **803** | 803 | Verified (10.4× genuine) |

---

## 3. Mechanism of the Artefact

1. **Fallback Implementation (Fork 6)**:
   In `scripts/explain_e1.py` (and the model definition pipeline):
   ```python
   sub_mask_strong = sp.load_npz(pri / "mirna_unique_pathway_mask_strong.npz")[:, pway_idx].toarray()
   zero_strong = sub_mask_strong.sum(axis=1) == 0  # 726 unmapped miRNAs
   sub_mask_strong[zero_strong, P - 1] = True     # wired into pathway index 191
   ```
2. **Attribution Aggregation**:
   The primary Integrated Gradients attribution summed $|IG|$ over all features connected to each pathway:
   $$\text{Attribution}(p) = \sum_{g \in \text{mRNA}} |IG_g| \cdot M_{p, g}^{\text{mRNA}} + \sum_{m \in \text{miRNA}} |IG_m| \cdot M_{p, m}^{\text{miRNA}} + \sum_{c \in \text{meth}} |IG_c| \cdot M_{p, c}^{\text{meth}}$$
3. **Impact**:
   For genuine pathways ($p < P-1$), the miRNA term sums over 0 to ~10 genuine target miRNAs. For pathway $P-1$, the miRNA term summed over **728 miRNAs** (726 fallback + 2 genuine). The resulting attribution of 4.98 was therefore dominated by the summed background attribution of 726 non-specific miRNA features.

---

## 4. Size Bias in Pathway Attribution

In addition to the fallback artefact on $P-1$, pathway attribution by raw summation is inherently biased toward large pathways:
- **Median genuine pathway size**: 72.5 features (across mRNA, miRNA, meth; median mRNA genes = 26).
- Pathways like mRNA splicing (255 genes) and translation initiation (140 genes) lie in the 97th–99th percentiles of pathway size.
- Raw sum $|IG|$ correlates strongly with pathway size, making large housekeeping pathways appear dominant regardless of biological specificity.

---

## 5. Corrective Action (Implemented in Stage A)

In accordance with Stage A requirements:
1. **Exclude Fallback Edges**: When aggregating $|IG|$ by summation (Metric i), the 726 fallback-routed miRNA edges are strictly excluded; only the 2 genuine miRNA edges are retained.
2. **Size Normalisation**: Metric (ii) reports the mean $|IG|$ per genuine member feature ($\frac{\text{sum } |IG|}{\text{genuine feature count}}$) alongside Metric (i).
3. **Correlation Reporting**: Both metrics report Spearman correlation with pathway size to quantify remaining size bias.
4. **Documentation**: All explainability reports, model cards, and documentation state plainly that ODC regulation was an artefact of fallback token routing.
