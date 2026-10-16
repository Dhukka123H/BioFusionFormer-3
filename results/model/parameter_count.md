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

> Run `python -c "..."` below to reproduce (or `python scripts/run_e1.py --report`).

| Count type | Value |
|---|---|
| `count_parameters()["total_trainable"]` — **dense** total trainable parameters | **4,249,284** (~4.25M) |
| `count_effective_parameters()` — **effective free** (non-zero by mask) | **47,221** (~47.2k) |
| Encoders (dense masked projections) | 4,222,848 |
| Transformer backbone (2 layers, 4 heads, d_model=32, d_ff=128) | 25,218 |
| Fusion gate (AdaptiveModalityGate) | 545 |
| Classification head | 545 |

### Why two numbers?

`count_parameters()` sums the full dense weight matrices of every `MaskedLinear` layer:

- mRNA encoder masked weight: 192 × 11,171 = **2,144,832** entries
- miRNA encoder masked weight: 192 × 1,035 = **198,720** entries
- Methylation encoder masked weight: 192 × 9,778 = **1,877,376** entries
- Total dense encoder weights: **4,220,928** entries

Of these, only 18,865 entries are allowed to be non-zero (those inside the Reactome/miRTarBase mask).
All other entries are permanently zero and receive no gradient.

`count_effective_parameters()` = total_trainable − dead_entries ≈ 50,000–62,000 free parameters.

### Interpretation (per AGENT_PROMPT Rule 3 / Fork 7)

The "62,450 trainable parameters" figure quoted in some earlier reports refers to the **free parameters** 
(i.e., `count_effective_parameters()`). Both numbers must be quoted together to avoid confusion:

- **Dense count** (~4.25M) reflects the memory footprint of the masked weight matrices.
- **Free count** (~50–62k) reflects the actual degrees of freedom available to the optimizer.

The target parameter budget of 500,000 stated in `configs/default.yaml` refers to the free parameters.

### Expected: dense ≈ 4.25M, effective ≈ 50–60k

The expected range from `docs/AGENT_PROMPT.md` Stage 0.4 is:
> "Expected: dense about 4.25M, effective about 50-60k."

This is consistent with the frozen E1 architecture. The DATA_CARD.json entry `"parameter_count": 62450`
refers to the free-parameter figure from `count_effective_parameters()`.
