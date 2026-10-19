# Fallback Audit: Pathway P-1 and miRNA Fallback Wiring

**Date:** 2026-10-17  
**Artifact Target:** `results/explainability/fallback_audit.md`  
**Related Findings:** Review Finding F1, Decision Fork 6  

---

## 1. Executive Summary

An audit of the feature-to-pathway wiring in frozen BioFusionFormer E1 confirms that the #1 attributed pathway in the initial Integrated Gradients report—**"Regulation of ornithine decarboxylase (ODC)" (`R-HSA-350562`, score: 4.981)**—is an artifact of the fallback wiring implementation (Fork 6).

In the frozen E1 architecture, the 726 unique miRNA features that lacked prior edges to the 192 selected Reactome pathways were wired to pathway token index $P - 1 = 191$ (`sub_mask_strong[zero_strong, P - 1] = True`). This pathway happened to be `R-HSA-350562`. When pathway attribution was computed by summing absolute Integrated Gradients across all connected features, this token aggregated attributions across **728 miRNA features** (726 fallback + 2 genuine) plus 38 mRNA and 37 methylation features (total: **803 input features**). This artificial 803-feature funnel roughly doubled its attribution score over genuine pathways, masquerading as a major biological finding.

---

## 2. Quantitative Verification

| Attribute | Pathway $P-1$ (`R-HSA-350562`) | Median Selected Pathway |
| :--- | :---: | :---: |
| **Pathway Index** | 191 (the 192nd selected pathway) | — |
| **Reactome ID** | `R-HSA-350562` | — |
| **Pathway Name** | Regulation of ornithine decarboxylase (ODC) | — |
| **Reactome Hierarchy Depth** | 3 | $\le 4$ |
| **Reactome Total Genes** | 40 | 26 |
| **Genuine mRNA Features Present** | 38 | 25 |
| **Genuine Methylation Features Present** | 37 | 24 |
| **Genuine miRNA Features Present** | **2** | 1 |
| **Fallback Routed miRNA Features** | **726** | **0** |
| **Total miRNA Features Wired** | **728** | 1 |
| **Total Features Connected to Token** | **803** | ~50 |

---

## 3. Mechanism of the Artifact

1. **Prior Selection:** 192 Reactome pathways were unsupervisedly selected by variance ranking among pathways with $10 \le \text{genes} \le 300$ and $\text{depth} \le 4$.
2. **miRNA Prior Sparsity:** Out of 1,035 unique precursor-level miRNA features, only 309 had genuine target edges in the selected 192 pathways under the strong-evidence mask (`mirna_unique_pathway_mask_strong.npz`). The remaining 726 features had zero edges.
3. **The Fallback Routing:** In `src/biofusionformer/full_model.py` and `scripts/explain_e1.py`, fallback assignment was coded as:
   ```python
   sub_mask_strong = sp.load_npz(pri / "mirna_unique_pathway_mask_strong.npz")[:, pway_idx].toarray()
   zero_strong = sub_mask_strong.sum(axis=1) == 0
   sub_mask_strong[zero_strong, P - 1] = True  # Fork 6 fallback
   ```
   Instead of allocating a dedicated 193rd "unmapped_mirna" token, the 726 orphan features were injected directly into token $P - 1$.
4. **Attribution Aggregation:** In initial explainability analysis (Stage 4 of prompt 1), pathway attribution was defined as:
   $$\text{Attr}(p) = \sum_{m \in \text{mods}} \sum_{j \in \text{members}(m, p)} |\text{IG}_{m, j}|$$
   Because pathway 191 had 803 connected features (instead of ~50), it accumulated 16-fold more feature attributions, driving its raw score to 4.981 (vs. 2.507 for mRNA Splicing and 2.213 for Eukaryotic Translation Initiation).

---

## 4. Remediation Required in Stage A

1. **Side-by-Side Dual Attribution Reporting (A2):**
   - **Variant (i) Genuine Sum:** Sum of $|IG|$ over genuine member features only, strictly excluding the 726 fallback miRNA edges.
   - **Variant (ii) Size-Normalized Mean:** Mean $|IG|$ per member feature ($\frac{1}{|\text{members}|} \sum_{j} |IG_j|$), removing the pathway size bias.
2. **Explicit Disclaimer:** All project documentation and UI displays must plainly state that ODC was an artifact of the fallback wiring.
3. **Frozen Model Integrity:** Per Rule 2, the weights of frozen E1 are preserved. The artifact was purely an attribution accounting error, not a change to model weights.
