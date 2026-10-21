# BioFusionFormer E1 — Parameter Count

Generated from `BioFusionFormerModel.count_parameters()` and `count_effective_parameters()` with the frozen E1 shapes.

## Architecture Shapes (frozen E1)

| Modality | Features |
|---|---|
| mRNA | 11,171 |
| miRNA (unique) | 1,035 |
| Methylation | 9,778 |
| Pathways (P) | 192 |

Transformer: d_model=32, n_heads=4, n_layers=2, d_ff=128.

## Results

| Count type | Value |
|---|---|
| `count_parameters()["total_trainable"]` — **dense** total trainable parameters | **4,249,284** (~4.25M) |
| `count_effective_parameters()` — **effective free** (non-zero by mask) | **47,221** (~47.2k) |
| Encoders (effective free parameters) | 20,785 |
| Modality embeddings (3 × 32) | 96 |
| [CLS] token embedding | 32 |
| Transformer backbone (2 layers, 4 heads, d_model=32, d_ff=128) | 25,218 |
| Fusion gate (AdaptiveModalityGate) | 545 |
| Classification head | 545 |
| **Total effective free parameters** | **47,221** |

## Per-Encoder Parameter Breakdown

Each modality encoder consists of a sparse `MaskedLinear` projection followed by bias, LayerNorm, and token projection:
- **Per-encoder overhead:** 640 parameters
  - Bias: 192
  - LayerNorm: 384
  - Token projection: 64

| Modality Encoder | Mask Non-Zeros | Encoder Overhead (Bias + LN + Proj) | Total Effective Free Parameters | Dense Parameters |
|---|:---:|:---:|:---:|:---:|
| **mRNA Encoder** | 7,148 | 640 | **7,788** | 2,145,024 |
| **miRNA Encoder** (incl. 726 fallback edges) | 5,768 | 640 | **6,408** | 198,912 |
| **Methylation Encoder** | 5,949 | 640 | **6,589** | 1,877,568 |
| **Total Encoders** | **18,865** | **1,920** | **20,785** | **4,221,504** |

### Dense vs. Free Parameters

`count_parameters()` sums the full dense weight matrices of every `MaskedLinear` layer:
- mRNA encoder masked weight: 192 × 11,171 = 2,144,832 entries
- miRNA encoder masked weight: 192 × 1,035 = 198,720 entries
- Methylation encoder masked weight: 192 × 9,778 = 1,877,376 entries
- Total dense encoder weights: 4,220,928 entries (plus 576 bias + 1,152 LN + 192 proj = 4,222,848 total encoder parameters)

Of these, only 18,865 entries are allowed to be non-zero (those inside the Reactome/miRTarBase mask).
All other entries are permanently zero and receive no gradient.

### Note on Historical 62,450 Figure

62,450 appeared in an earlier draft and could not be reproduced. The verified free-parameter count is **47,221**, and the dense trainable parameter count is **4,249,284**.
