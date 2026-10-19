# BioFusionFormer E1 — Parameter Count

Generated from `BioFusionFormerModel.count_parameters()` and `count_effective_parameters()` with the frozen E1 architecture.

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

## Component Breakdown

| Component | Dense Total Parameters | Mask Non-Zeros | Extra Parameters | Effective Free Parameters | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **mRNA Modality Encoder** | 2,145,472 | 7,148 | 640 | 7,788 | 192 pathways $\times$ 11,171 genes (MaskedLinear + extra 640) |
| **miRNA Modality Encoder** | 199,360 | 5,768 | 640 | 6,408 | 192 pathways $\times$ 1,035 miRNAs (incl. 726 fallback edges + extra 640) |
| **Methylation Modality Encoder** | 1,878,016 | 5,949 | 640 | 6,589 | 192 pathways $\times$ 9,778 CpGs (MaskedLinear + extra 640) |
| **[CLS] Token** | 32 | — | — | 32 | Learnable classification token ($1 \times 1 \times 32$) |
| **Modality Embedding** | 96 | — | — | 96 | Modality type embeddings ($3 \times 32$) |
| **Cross-Modal Transformer** | 25,218 | — | — | 25,218 | 2 layers, 4 heads, $d_{\text{model}}=32$, $d_{\text{ff}}=128$ |
| **Adaptive Modality Gate** | 545 | — | — | 545 | Modality-specific gating network |
| **Classification Head** | 545 | — | — | 545 | 2-layer MLP ($32 \rightarrow 16 \rightarrow 1$) |
| **Total** | **4,249,284** | **18,865** | **1,920** | **47,221** | Trainable parameters |

*Note on per-encoder extra parameters (640 each):*
- `masked_linear.bias`: 192
- `norm.weight` (LayerNorm): 192
- `norm.bias` (LayerNorm): 192
- `token_proj.weight`: 32
- `token_proj.bias`: 32
- Total per encoder = $192 + 192 + 192 + 32 + 32 = 640$.

### Why two numbers?

`count_parameters()` sums the full dense weight matrices of every layer:
- mRNA encoder masked weight: 192 $\times$ 11,171 = 2,144,832 entries
- miRNA encoder masked weight: 192 $\times$ 1,035 = 198,720 entries
- Methylation encoder masked weight: 192 $\times$ 9,778 = 1,877,376 entries
- Dense encoder weights: 4,220,928 entries (+ 1,920 bias/norm/proj = 4,222,848)

Of these, only 18,865 weight entries are allowed to be non-zero (those inside the Reactome/miRTarBase mask).
All other entries are permanently zero and receive no gradient.

### Note on earlier drafts
62,450 appeared in an earlier draft and could not be reproduced. The verified free-parameter count is 47,221. Both the dense count (4,249,284) and free count (47,221) are quoted together throughout this repository.
