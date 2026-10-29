# Semester 7 Final Verification Report — BioFusionFormer-3

**Repository**: BioFusionFormer-3  
**Evaluation Environment**: Frozen E1 / Semester 7 Final  
**Verification Status**: **ALL OBJECTIVES VERIFIED FROM CODE AND DATA**  

---

## Objective 1: Hierarchical Multi-Modal Feature Extraction

### 1.1 Per-Modality Raw Features & Pathway Tokens

| Modality | Raw Features ($F_{mod}$) | Pathway Tokens ($P$) | Active Prior Connections | Total Dense Elements | Sparsity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MRNA** | 11,171 | 192 | 7,148 | 2,144,832 | 99.667% |
| **MIRNA** | 1,035 | 192 | 5,768 | 198,720 | 97.097% |
| **METH** | 9,778 | 192 | 5,949 | 1,877,376 | 99.683% |
| **Total / Overall** | **21,984** | **192** | **18,865** | **4,220,928** | **99.553%** |

- Each modality projection $\mathbf{W}_{mod} \in \mathbb{R}^{192 \times F_{mod}}$ is constrained by a biological mask $\mathbf{M}_{mod} \in \{0, 1\}^{192 \times F_{mod}}$.
- Total active biological edges: **18,865** out of 4,220,928 possible dense connections (**99.553% parameter reduction**).

### 1.2 Reactome Pathway Hierarchy Mapping (`pathway_hierarchy_mapping.json`)

- **Total selected pathways**: 192
- **Pathways mapped in Reactome hierarchy**: 192
- **Root pathways** (top-level biological systems): 0
- **Internal pathways** (intermediate biological subsystems): 52
- **Leaf pathways** (terminal reaction cascades): 140
- **Hierarchy Depth**: min = 1, max = 4, mean = 3.07

### 1.3 Masked Weight Invariance Proof Under Optimizer Steps

| Metric | Value | Status |
| :--- | :---: | :---: |
| Initial weights outside mask ($|\mathbf{W}_{\sim M}|_{1}$) | 0.0 | **Zero** |
| Gradients outside mask ($|\nabla_{\mathbf{W}_{\sim M}} \mathcal{L}|_{1}$) | 0.0 | **Zero** |
| Weights outside mask post-AdamW step | 0.0 | **Zero** |
| **Verification Outcome** | **STRICTLY ZERO (PROVEN)** | ✅ |

---

## Objective 2: Pathway-Aware Cross-Modal Transformer

### 2.1 Biological Prior Bias Matrix (`bio_bias.pt`)

- **Matrix Shape**: `577 × 577` ($S = 1 + 3 \times 192 = 577$)
- **Nonzero Entries**: 2,646 / 332,929 (0.795%)
- **Integration in Attention**: Added directly to scaled dot-product attention logits:
  $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{B}_{\text{bio}}\right)$$

### 2.2 Token Layout (Sequence Length $S = 577$)

| Token Range | Token Count | Identity / Modality | Purpose |
| :---: | :---: | :--- | :--- |
| Index `0` | 1 | `[CLS]` Token | Cross-modal pooled summary representation |
| Indices `1..192` | 192 | mRNA Pathway Tokens | Gene expression pathway activity states |
| Indices `193..384` | 192 | miRNA Pathway Tokens | Post-transcriptional pathway target states |
| Indices `385..576` | 192 | DNA Methylation Pathway Tokens | Epigenetic promoter regulation states |

### 2.3 Transformer Architecture & Forward Shape Trace

- **Layers**: 2 | **Heads**: 4 | **$d_{model}$**: 32 | **$d_{ff}$**: 128 | **$d_{head}$**: 8
- **Attention Dropout**: 0.2 | **FFN Dropout**: 0.3

**Shape Trace Across Forward Pass** (Batch Size $B=2$):

- **1. Raw Inputs**: `{'mrna': [2, 11171], 'mirna': [2, 1035], 'meth': [2, 9778]}`
- **2. Masked Encoders Output**: `{'mrna': [2, 192, 32], 'mirna': [2, 192, 32], 'meth': [2, 192, 32]}`
- **3. Transformer Sequence Input**: `[2, 577, 32]`
- **4. Multi-Head Attention Maps (per layer)**: `[[2, 4, 577, 577], [2, 4, 577, 577]]`
- **5. Adaptive Gate Weights Output**: `[2, 3]`
- **6. Classification Logit Output**: `[2]`

---

## Objective 3: Adaptive Multi-Modal Fusion

### 3.1 Patient-Specific Gate Weights on Demo Cohort

| Patient ID | Predicted P(Unfavorable) | Gate mRNA | Gate miRNA | Gate Methylation | Sum |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `TARGET-50-CAAAAB` | **0.995** | 0.3128 | 0.3216 | 0.3656 | **1.0** |
| `TARGET-50-CAAAAC` | **0.04** | 0.3254 | 0.3693 | 0.3053 | **1.0** |
| `TARGET-50-CAAAAH` | **0.9943** | 0.3149 | 0.322 | 0.3631 | **1.0** |

- **Sum-to-One Invariance**: PASSED (all patient gates sum to exactly 1.0)
- **Patient Dynamism**: PASSED (gate weights adapt per patient based on omics evidence)
- **Modality Dropout Configuration**: Rate = `0.2`
  *During training, each modality is dropped with probability 0.2 (with at least 1 modality retained and remaining active modalities renormalized to sum to 1.0). During evaluation/inference, dropout is disabled and patient-specific gating weights all 3 modalities.*

---

## Read-Only Published E1 Benchmark Metrics

*Source: `experiments/final_v1/E1/metrics.json` (Signature: `a42642001739`)*

| Metric | 25-Fold Mean | Std Dev | 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **AUROC** | **0.8716** | ±0.0613 | [0.8475, 0.8956] |
| **AUPRC** | **0.8332** | — | — |
| **Balanced Accuracy** | **0.7653** | — | — |
| **Macro F1** | **0.7697** | — | — |
| **Brier Score** | **0.1436** | — | — |

> **Factual Reporting Notice**: All figures above are extracted verbatim from the frozen experimental run records. No comparative claims of superiority over external baselines are asserted beyond verified empirical confidence intervals.
