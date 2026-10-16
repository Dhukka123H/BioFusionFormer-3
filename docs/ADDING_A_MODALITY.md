# Adding a Modality — Architectural Guide (Stage 6.5)

## Overview

This document specifies the architectural protocol for extending **BioFusionFormer** beyond the three frozen omics modalities (mRNA, miRNA, DNA methylation) to support future radiographic imaging modalities (such as volumetric **CT** or multi-parametric **MRI**).

> [!IMPORTANT]
> **No Imaging Performance is Claimed**: The TARGET-WT cohort has no matched public diagnostic CT or MRI imaging series linked to these omics matrices. The implementation in `src/modalities/registry.py` provides the extensible registry, interface specifications, and opt-in missing-modality transformer wrapper, validated via synthetic image token unit tests (`tests/test_modality_registry.py`).

---

## 1. Modality Registry (`src/modalities/registry.py`)

All modalities (core omics, clinical EHR, and future imaging) are registered with metadata and token specifications:

```python
from src.modalities.registry import ModalityRegistry, ModalitySpec, ImagingFeatureEncoder

registry = ModalityRegistry()
registry.register(ModalitySpec(name="mrna", token_count=192, modality_type="omics"))
registry.register(ModalitySpec(name="mirna", token_count=192, modality_type="omics"))
registry.register(ModalitySpec(name="meth", token_count=192, modality_type="omics"))
registry.register(ModalitySpec(name="clinical", token_count=1, modality_type="clinical"))
registry.register(ModalitySpec(name="ct_imaging", token_count=16, modality_type="imaging"))
```

---

## 2. Feature Extraction & `ImagingFeatureEncoder`

### Step A: External Pre-extraction (Frozen Backbone)
Raw medical imaging volumes (DICOM / NIfTI) must **not** be trained end-to-end with the transformer given cohort size constraints ($N=121$). Instead, patch embeddings or volumetric feature vectors must be pre-extracted using an external frozen backbone (e.g., a 3D ResNet, Swin UNETR, or self-supervised foundation model like RadImageNet):
- Output representation: $T$ patch feature vectors of dimension $D_{\text{in}}$ (e.g. $T=16$ regional patches, $D_{\text{in}}=512$).

### Step B: In-Repository Projection
In `src/modalities/registry.py`, the `ImagingFeatureEncoder` projects pre-extracted vectors into the common $d_{\text{model}}$ space ($d_{\text{model}}=32$):
```python
class ImagingFeatureEncoder(nn.Module):
    def __init__(self, in_dim=512, n_patches=16, d_model=32, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout),
        )
```

The encoder returns:
1. Projected tokens: `(B, T, d_model)`
2. Modality availability mask: `(B,)` boolean tensor (`True` = present, `False` = missing).

---

## 3. Handling Missing Modalities (Opt-In Key-Padding Mask)

In clinical cohorts, imaging is frequently missing for subsets of patients. The frozen E1 model cannot handle missing modalities (documented in `results/model/forking_paths.md` Fork 8).

To solve this for future multimodal pipelines without changing the frozen E1 code, `RobustCrossModalTransformerWrapper` provides an opt-in key-padding mask:
- For absent modalities, zero tokens `(B, T, d_model)` are inserted.
- The `key_padding_mask` marks the corresponding sequence positions as `True`.
- In cross-modal attention, the attention logits for padded keys are masked with $-10^4$ prior to softmax:
  $$\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{scaled\_bias} + M_{\text{padding}}\right)$$
- Attention from $[CLS]$ and other modalities to absent tokens is strictly $0.0$, preventing hallucinated cross-modal leakage.

---

## 4. Verification & Testing

The imaging extension is unit-tested in `tests/test_modality_registry.py`:
- `test_imaging_encoder_present_and_absent`: Verifies projection shapes and availability masks.
- `test_robust_transformer_with_imaging_modality_present_and_absent`: Verifies that a 5th modality (imaging) plugs in and that attention to missing modality tokens is strictly masked out (< $10^{-4}$).
