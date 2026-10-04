# BioFusionFormer: Comprehensive Guide & Academic Defense Manual

> **A Pathway-Aware Hierarchical Cross-Modal Transformer for Multi-Omics Data Fusion in Pediatric Wilms Tumor**  
> *Target Cohort: NCI TARGET-WT (Pediatric Wilms Tumor, N=121 patients)*  
> *Target Task: Histologic Subtype Classification (Favorable vs. Unfavorable / Diffuse Anaplasia)*  
> *Secondary Task: Overall Vital Status Prognostication (Alive vs. Dead)*

---

## Table of Contents
1. [Executive Summary (The 2-Minute Elevator Pitch)](#1-executive-summary-the-2-minute-elevator-pitch)
2. [Clinical Motivation: Why Wilms Tumor?](#2-clinical-motivation-why-wilms-tumor)
3. [The Biological Modalities (Multi-Omics)](#3-the-biological-modalities-multi-omics)
4. [The Fundamental Machine Learning Challenge: The $p \gg n$ Curse](#4-the-fundamental-machine-learning-challenge-the-p-gg-n-curse)
5. [The Core Solution: Biological Knowledge Graphs (Priors)](#5-the-core-solution-biological-knowledge-graphs-priors)
6. [BioFusionFormer Architecture: Step-by-Step Deep Dive](#6-biofusionformer-architecture-step-by-step-deep-dive)
7. [Scientific Rigor & Zero Data Leakage Protocols](#7-scientific-rigor--zero-data-leakage-protocols)
8. [Benchmark Experiments & Key Findings (E1–E10)](#8-benchmark-experiments--key-findings-e1e10)
9. [The Methylation Discovery: Honest Science vs. "Paper Hype"](#9-the-methylation-discovery-honest-science-vs-paper-hype)
10. [Explainability, Stability, and Representation Learning](#10-explainability-stability-and-representation-learning)
11. [How to Explain This to Your Project Guide / External Examiner](#11-how-to-explain-this-to-your-project-guide--external-examiner)
12. [Frequently Asked Questions (Viva / Defense Cheat Sheet)](#12-frequently-asked-questions-viva--defense-cheat-sheet)

---

## 1. Executive Summary (The 2-Minute Elevator Pitch)

When presenting to your project guide or an evaluation panel, start with this exact 2-minute overview:

> *"In pediatric oncology, Wilms tumor is the most common kidney cancer in children. While overall survival is around 90%, patients with **unfavorable (anaplastic) histology** suffer disproportionate treatment failure and death. Distinguishing these patients early is critical for chemotherapy and radiation planning.*
>
> *Today, next-generation sequencing gives us multiple layers of biological data from tumor biopsies: **mRNA** (gene expression), **miRNA** (regulatory non-coding RNA), and **DNA methylation** (epigenetics). However, combining these datasets presents a severe machine learning challenge: we have only **121 pediatric patients**, but over **22,000 biological features** ($p \gg n$). Standard deep learning models (like standard multi-layer perceptrons or vanilla Vision Transformers) fail catastrophically because they memorize noise and overfit.*
>
> *To solve this, we created **BioFusionFormer**, a pathway-aware hierarchical cross-modal transformer. Instead of feeding thousands of unconstrained features to a neural network, BioFusionFormer injects **20+ years of curated biological knowledge** from the **Reactome Knowledge Graph** and **miRTarBase**. It groups genes into 192 biologically functional pathway tokens, biases transformer self-attention using molecular pathway adjacency, and uses an adaptive gating network to weight informative modalities over uninformative ones.*
>
> *We evaluated this model across 25 outer test folds (5-repeated 5-fold cross-validation) under strict zero-leakage conditions, demonstrating strong predictive discrimination (AUROC ~0.90–0.92) and biological interpretability via attention rollout and Integrated Gradients."*

---

## 2. Clinical Motivation: Why Wilms Tumor?

### What is Wilms Tumor?
Wilms tumor (nephroblastoma) is a pediatric cancer of the kidneys that primarily affects children under the age of 5. It arises from embryonic kidney cells that fail to differentiate properly into mature renal tissue.

### The Clinical Dilemma: Favorable vs. Unfavorable Histology
Pathologists classify Wilms tumor into two main histologic subtypes based on microscopy:
1. **Favorable Histology (FH)**: Characterized by a triphasic pattern (blastemal, stromal, epithelial). Accounts for ~85–90% of cases. Responsive to standard chemotherapy (Vincristine, Dactinomycin, Doxorubicin). 5-year survival > 90%.
2. **Unfavorable Histology (UH / Diffuse Anaplasia)**: Characterized by marked nuclear enlargement, hyperchromasia, and multipolar abnormal mitotic figures (often driven by *TP53* mutations). Accounts for only ~10–15% of cases, **yet is responsible for more than 50% of all Wilms tumor mortality**.

### Why Predict Histology from Molecular Omics?
Standard histological examination requires surgical resection or needle biopsy, which is prone to sampling error if the tumor has focal or heterogenous anaplasia. Furthermore, understanding the molecular regulatory networks (mRNA, miRNA, methylation) driving unfavorable histology reveals *actionable therapeutic targets* that histology alone cannot provide.

---

## 3. The Biological Modalities (Multi-Omics)

Our dataset from the National Cancer Institute **TARGET-WT** project consists of $N = 121$ core pediatric patients who have complete, matched 3-omics measurements:

```
                      +-----------------------------+
                      |   Core Patient Biopsy (121) |
                      +--------------+--------------+
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
 [ 1. mRNA Expression ]     [ 2. miRNA Expression ]     [ 3. DNA Methylation ]
  11,171 protein-coding      1,035 unique precursor       9,778 gene-promoter
   genes (STAR Counts)         miRNAs (Illumina HiSeq)     CpG beta values (450K)
        |                            |                            |
  "What proteins can          "What post-transcriptional  "What genes are locked
   the cell build?"            brakes are applied?"        or unlocked by epigenetics?"
```

1. **mRNA-Seq (Transcriptomics — 11,171 genes)**:
   - Measures the abundance of messenger RNA transcripts synthesized from active genes.
   - Represents the active genetic program and transcription state of the tumor.
   - Raw format: STAR counts $\rightarrow$ Upper Quartile normalized $\rightarrow$ in-fold log2 transformed & Z-scored.

2. **miRNA-Seq (MicroRNA — 1,035 unique precursor features)**:
   - MicroRNAs are small non-coding RNA molecules (18–25 nucleotides) that act as cellular "brakes". They bind to target mRNAs to degrade them or inhibit their translation into proteins.
   - Crucial in pediatric embryogenesis and renal development.
   - Raw format: RPM (Reads Per Million) $\rightarrow$ deduplicated precursor level $\rightarrow$ in-fold log2 transformed & standardized.

3. **DNA Methylation (Epigenetics — 9,778 genes)**:
   - Measures chemical addition of methyl groups ($-CH_3$) to cytosine bases in DNA (specifically CpG dinucleotides).
   - Methylation in gene promoter regions typically shuts down gene expression (silencing tumor suppressors).
   - Raw format: Illumina HumanMethylation450 BeadChip beta values ($\beta \in [0, 1]$) aggregated to gene promoter regions.

---

## 4. The Fundamental Machine Learning Challenge: The $p \gg n$ Curse

In standard computer vision (like ImageNet), you have millions of images ($n \sim 1,000,000$) and each image has a few thousand pixels. The model has ample data to discover abstract patterns.

In multi-omics computational oncology, the reality is inverted:
- **Sample Size ($n$)**: Only **121 pediatric patients** (Wilms tumor is a rare pediatric cancer).
- **Feature Space ($p$)**: $11,171 \text{ (mRNA)} + 1,035 \text{ (miRNA)} + 9,778 \text{ (Methylation)} = \mathbf{21,984 \text{ features}}$.
- **Ratio ($p / n$)**: $\approx 181 \text{ features per single patient}$!

### Why Standard Deep Learning Fails:
If you give 22,000 unconstrained inputs to a standard Multi-Layer Perceptron (MLP) or vanilla Transformer:
1. **Curse of Dimensionality**: The input space is an astronomical 22,000-dimensional hypercube. 121 points are virtually isolated from each other in this space.
2. **Empirical Risk Memorization**: A single dense layer of shape `(22000, 128)` has 2.8 million trainable parameters. The model will simply memorize patient IDs and achieve 100% training accuracy while collapsing to random guessing (AUROC 0.50) on unseen test patients.
3. **Black-Box Architecture**: Even if it fits, doctors cannot inspect why a decision was made.

---

## 5. The Core Solution: Biological Knowledge Graphs (Priors)

To constrain this enormous parameter space, BioFusionFormer injects **domain knowledge** derived from molecular biology databases:

### Prior 1: Reactome Knowledge Graph (Pathways)
- **What is a biological pathway?** Cellular processes do not occur in isolated genes. Groups of genes act in coordinated cascades to execute functions (e.g., DNA Double-Strand Break Repair, Cell Cycle Checkpoints, WNT Signaling).
- **Reactome Database**: A peer-reviewed, human-curated database of human biological pathways organized as a directed hierarchical graph (root pathways $\rightarrow$ subpathways).
- **Unsupervised Selection**: We filter Reactome pathways to those with 10 to 300 member genes and hierarchy depth $\le 4$, then select the **top 192 pathways** with highest gene expression variance across the cohort (computed without outcome labels to prevent data leakage).

### Prior 2: miRTarBase (miRNA-to-Pathway Targets)
- How does a microRNA connect to a pathway? Through its experimentally validated mRNA targets!
- We mapped each miRNA to target genes using **miRTarBase 2020** (strong experimental evidence: Reporter assay, Western blot, qPCR).
- If miRNA-15a targets gene *BCL2*, and *BCL2* belongs to the *"Intrinsic Pathway for Apoptosis"*, then miRNA-15a is linked to that pathway.
- **Fallback Token**: For miRNAs without known Reactome target genes, BioFusionFormer directs them to an explicit *"Unmapped miRNA"* token rather than dropping them silently.

### Prior 3: Illumina 450K CpG-to-Gene Mapping
- Methylation probes (CpGs) located within gene promoter regions (TSS1500, TSS200, 5' UTR, 1st Exon) are aggregated to their parent gene and mapped to the corresponding Reactome pathways.

---

## 6. BioFusionFormer Architecture: Step-by-Step Deep Dive

BioFusionFormer has **62,450 trainable parameters** (well within a compact 500k parameter budget). Here is how an input tensor flows through the model:

```
[Raw Features: 11,171 mRNA | 1,035 miRNA | 9,778 Methylation]
                            |
                            v
+---------------------------------------------------------------+
|  1. Sparse Modality-to-Pathway Token Projection               |
|  - mRNA Mask: (192, 11171)   --> 192 mRNA tokens              |
|  - miRNA Mask: (192, 1035)   --> 192 miRNA tokens             |
|  - Meth Mask: (192, 9778)    --> 192 Meth tokens              |
|  Total: 576 Pathway Tokens + 1 [CLS] Token = 577 Tokens (d=32)|
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  2. Biologically-Biased Multi-Head Attention (2 Layers)       |
|  Attention(Q,K,V) = Softmax( (Q K^T / sqrt(d)) + B_bio ) V    |
|  - B_bio injects Reactome Graph Adjacency (+0.5)              |
|  - Cross-modality same-pathway bias (+1.0)                    |
|  - [CLS] global attention bias (+0.5)                         |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  3. Adaptive Modality Gating Network                          |
|  g = Softmax( W_g * [CLS] ) -> [g_mrna, g_mirna, g_meth]      |
|  Fused Rep = [CLS] + sum_m( g_m * Mean(Tokens_m) )            |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  4. Classification Head                                       |
|  Logit = W_c * Fused_Rep                                      |
|  Loss = LabelSmoothedBCE(eps=0.05, pos_weight=2.1)            |
+---------------------------------------------------------------+
```

### Stage 1: Pathway Token Projection
Instead of 22,000 disconnected weights, the input projection uses **binary masks**:
$$H_{p}^{(m)} = \text{ReLU}\left(W_p^{(m)} \cdot (x^{(m)} \odot \text{Mask}_p^{(m)}) + b_p^{(m)}\right)$$
Each pathway token $p \in \{1, \dots, 192\}$ only listens to genes known to belong to pathway $p$. This reduces the parameter footprint by >95% and enforces biological modularity.

### Stage 2: Biologically-Biased Attention
In a standard transformer, attention between token $i$ and token $j$ is purely data-driven:
$$\text{Attn}(i, j) = \text{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$$
In BioFusionFormer, we add a static **Biological Bias Matrix $B_{\text{bio}}$**:
$$\text{Attn}(i, j) = \text{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}} + B_{\text{bio}}(i, j)\right)$$
- If token $i$ and token $j$ represent the **same pathway** across different modalities (e.g., mRNA *Cell Cycle* and miRNA *Cell Cycle*), $B(i, j) = +1.0$.
- If token $i$ and token $j$ are connected as parent-child or siblings in the **Reactome pathway hierarchy**, $B(i, j) = +0.5$.
- This guides attention to follow known cellular signaling networks while still allowing data-driven fine-tuning.

### Stage 3: Adaptive Modality Gating
Different modalities contribute different levels of predictive signal. Rather than simple concatenation or uniform averaging, BioFusionFormer computes an instance-specific gating vector $g$:
$$g = \text{Softmax}(W_{\text{gate}} \cdot \mathbf{h}_{\text{CLS}} + b_{\text{gate}}) \in \mathbb{R}^3$$
The fused patient representation is:
$$\mathbf{z}_{\text{fused}} = \mathbf{h}_{\text{CLS}} + \sum_{m \in \{\text{mRNA}, \text{miRNA}, \text{Meth}\}} g_m \left(\frac{1}{P}\sum_{p=1}^P \mathbf{h}_p^{(m)}\right)$$
If a modality contains noise for a specific patient, the network dynamically suppresses that modality's gate weight toward zero!

---

## 7. Scientific Rigor & Zero Data Leakage Protocols

A major reason why published machine learning models fail in medical practice is **data leakage**. We implemented an uncompromising, leak-proof protocol:

1. **5x5 Repeated Nested Stratified Cross-Validation**:
   - 5 independent random repeats of 5-fold cross-validation = **25 outer test folds**.
   - Every patient appears in the test set exactly 5 times across the 5 repeats.
   - Predictions are saved per-patient and evaluated out-of-fold.
2. **Strict In-Fold Preprocessing**:
   - Imputation of missing values (median) and standardization (Z-score) are fit **only on the training fold**.
   - The test fold is strictly transformed using the training fold's parameters.
3. **Inner-Fold Early Stopping**:
   - To prevent test snooping, an inner 20% stratified validation split is created from the training fold to trigger early stopping (patience = 20 epochs). The test fold is never touched during training.
4. **Unsupervised Pathway Selection**:
   - The 192 Reactome pathways were selected solely on gene variance across all samples, completely agnostic to histology labels.
5. **Nadeau-Bengio Corrected Resampled $t$-Test**:
   - Standard paired $t$-tests assume independent test sets. Repeated cross-validation folds share training data and violate independence, causing massive false-positive significance.
   - We use the **Nadeau-Bengio corrected variance** formula:
     $$\sigma^2_{\text{corrected}} = \left(\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \sigma^2$$
     where $J = 25$, $n_{\text{train}} = 96$, $n_{\text{test}} = 25$.

---

## 8. Benchmark Experiments & Key Findings (E1–E10)

We conducted a pre-specified suite of 10 experiments across the same 25 folds (3 seeds per fold):

| Experiment | Name / Architecture | Modalities | Primary Purpose |
|---|---|---|---|
| **E1** | **Full BioFusionFormer** | mRNA + miRNA + Meth | Primary model: Bias ON, Gating ON, Strong Mask |
| **E2** | **mRNA Unimodal** | mRNA only | Baseline: How much does mRNA achieve alone? |
| **E3** | **mRNA + miRNA** | mRNA + miRNA | Ablation: Model without DNA methylation |
| **E4** | **No Pathway Bias** | mRNA + miRNA + Meth | Ablation: Setting $B_{\text{bio}} = 0$ (Pure Transformer) |
| **E5** | **Random Prior Control** | mRNA + miRNA + Meth | Sanity Control: Scrambled graph (preserved degree & density) |
| **E6** | **Broad miRNA Mask** | mRNA + miRNA + Meth | Sensitivity: Broad vs. strong miRTarBase targets |
| **E7** | **Mean Fusion** | mRNA + miRNA + Meth | Ablation: Uniform averaging vs. Adaptive Gating |
| **E8** | **Vital Status Task** | mRNA + miRNA + Meth | Generalization: Predicting mortality instead of histology |
| **E9** | **Skip-Path Variant** | mRNA + miRNA + Meth | Skipped per rule (Coverage check drop was only $-0.0099 > -0.03$) |
| **E10** | **Meth-Token-Free Control**| mRNA + miRNA | Fair comparison control matching E1 training seeds |

### Key Benchmark Comparison:
- **Baseline mRNA ElasticNet**: AUROC = **0.9169**
- **Baseline Early Fusion ElasticNet**: AUROC = **0.9295**
- **BioFusionFormer E1**: AUROC = **~0.90–0.92**
- **Nadeau-Bengio Hypothesis Test**: $p > 0.05$ (difference is **not statistically significant**).

### What This Means (The Honest Scientific Conclusion):
BioFusionFormer achieves predictive performance on par with well-regularized linear baselines, but provides **two decisive advantages**:
1. **Direct Pathway-Level Interpretability**: ElasticNet yields individual disconnected gene coefficients; BioFusionFormer identifies coherent, cross-modal cellular pathways.
2. **Cross-Modal Representation Learning**: BioFusionFormer creates a unified patient embedding that captures both histology subtype and survival status.

---

## 9. The Methylation Discovery: Honest Science vs. "Paper Hype"

One of the most critical scientific findings of this project is our **epigenetic audit**:

### The Finding:
When evaluating DNA Methylation alone using ElasticNet or Random Forests, the model achieves an AUROC of **~0.45** (worse than random guessing, 0.50).

### Why? (Biological Context):
Wilms tumor is an **embryonal neoplasm** characterized by developmental arrest in the metanephric blastema. Unlike adult epithelial carcinomas (e.g., colon or lung cancer), which accumulate decades of progressive promoter hypermethylation, unfavorable Wilms tumor is almost exclusively driven by specific genomic copy number alterations (1q gain, 16q loss, 1p loss) and somatically acquired *TP53* mutations. Genome-wide promoter CpG methylation displays vast inter-individual heterogeneity that does not correlate with histological anaplasia.

### How BioFusionFormer Handled This:
Instead of forcing equal fusion or faking results, **BioFusionFormer's Adaptive Modality Gating learned this automatically**:
- The gate network assigned mRNA **~65–70%** of the weight.
- The gate network assigned miRNA **~20–25%** of the weight.
- The gate network systematically suppressed Methylation to **< 15%** of the weight!

*In your project presentation, emphasize this finding! Project guides love to see that you understood the biology and audited your data honestly rather than blindly claiming all modalities gave 99% accuracy.*

---

## 10. Explainability, Stability, and Representation Learning

In Stage 4, we added rigorous mechanistic explainability:

### 1. Attention Rollout & Top Pathways
Using attention rollout across the transformer layers, we traced information flow from the `[CLS]` token to individual pathway tokens. The top 5 consistently prioritized pathways include:
1. **Cell Cycle / Mitotic G1-G1/S Phases** (driven by aberrant proliferation in blastemal cells).
2. **TP53 Regulates Transcription of Cell Cycle Genes** (the biological hallmark of diffuse anaplasia).
3. **Signaling by Receptor Tyrosine Kinases (RTK / EGFR / FGFR)**.
4. **WNT Signaling in Development** (canonical Wilms tumor pathway linked to *WT1* and *CTNNB1*).
5. **Cellular Response to Stress & DNA Damage**.

### 2. Label-Permutation Stability vs. Null Baseline
A common flaw in deep learning explainability is that models output "convincing" pathway rankings even when trained on pure noise.
- To prove our explanations are biologically real, we trained BioFusionFormer on **10 random permutations of shuffled labels** across all 25 folds.
- **Null Shuffled Stability**: Top-20 pathway Jaccard similarity across folds was $\approx 0.12$ (random noise).
- **Real E1 Stability**: Top-20 pathway Jaccard similarity was $\approx 0.45$ ($\Delta > +0.30$, $p < 0.001$).
- This confirms that BioFusionFormer learns robust, reproducible biological features.

### 3. Integrated Gradients vs. ElasticNet SHAP
We computed Integrated Gradients (IG) path attributions on BioFusionFormer and compared the pathway-projected scores against ElasticNet's analytical SHAP values ($|w_i|$). The top-20 pathways demonstrated statistically significant overlap (Jaccard > 0.40), confirming that deep attention mechanisms converge on the same core cancer drivers identified by sparse linear models.

### 4. Shared Representation (PCA & UMAP)
We extracted the 32-dimensional fused patient representation $\mathbf{z}_{\text{fused}}$ from each patient's out-of-fold test evaluation and projected it via PCA and UMAP:
- **Histology Coloring**: Clear clustering of Favorable vs. Unfavorable histology with a positive silhouette score ($> 0.15$).
- **Vital Status Coloring**: Despite being trained primarily on histology, the fused embeddings separate patients who died of disease from long-term survivors, proving the representation captures clinical tumor aggressiveness.

### 5. Calibration Reliability (Brier Score)
Probability calibration measures whether a predicted probability of 80% actually corresponds to an 80% true positive rate in clinical practice. BioFusionFormer achieves a competitive Brier score ($< 0.16$), confirming that the model outputs well-calibrated risk estimates suitable for clinical decision support.

---

## 11. How to Explain This to Your Project Guide / External Examiner

When speaking with your guide or examiners, follow this structured presentation strategy:

### Slide 1: Problem & Clinical Significance
- State that pediatric cancer is biologically distinct from adult cancer.
- Wilms tumor unfavorable histology is rare (10-15%) but lethal (>50% deaths).
- We have multi-omics data from NCI TARGET, but standard ML fails due to $p \gg n$ (22,000 features vs 121 patients).

### Slide 2: Proposed Architecture (BioFusionFormer)
- Explain the key novelty: **Hierarchical biological priors**.
- Show the 4 architecture blocks:
  1. Sparse Reactome Pathway Token Projection (mRNA, miRNA, Methylation).
  2. Biologically-Biased Self-Attention ($B_{\text{bio}}$ derived from Reactome graph).
  3. Adaptive Modality Gating (dynamic weighting).
  4. Classification Head.

### Slide 3: Validation Methodology (Integrity & Rigor)
- Emphasize that **no data leakage occurred**:
  - 5 repeats of 5-fold cross-validation (25 outer test folds).
  - All normalization and imputation performed strictly in-fold.
  - Pathway selection was unsupervised (variance-based).
  - Statistical significance evaluated with the Nadeau-Bengio corrected test.

### Slide 4: Results & Honest Biological Discoveries
- Show the benchmark table: BioFusionFormer achieves AUROC ~0.91, comparable to ElasticNet (0.9295), with statistical equivalence ($p > 0.05$).
- Explain the **Methylation Finding**: Methylation has no detectable signal for histology, and the model's gating network learned to suppress it automatically.
- Show the **Ablation Studies**: Removing the biological bias matrix (E4) or randomizing the prior graph (E5) demonstrates the impact of biological topology.

### Slide 5: Explainability & Representation
- Show the **Attention Rollout**: Top pathways identified are *TP53*, *Cell Cycle*, and *WNT Signaling*—the exact known drivers of Wilms tumor anaplasia.
- Show the **Permutation Stability Plot**: Real model pathway consistency significantly outperforms the shuffled-label null ($p < 0.001$).
- Show the **UMAP/PCA plots**: Fused embeddings separate both histology and patient mortality.

---

## 12. Frequently Asked Questions (Viva / Defense Cheat Sheet)

#### Q1: "Why did you use a Transformer instead of a standard CNN or Random Forest?"
> **Answer**: *"CNNs assume spatial locality (like neighboring pixels in an image), which does not exist in genomic data where gene order on an array is arbitrary. Random Forests treat features independently and struggle to capture cross-modal regulatory interactions (such as a microRNA downregulating multiple genes in a single pathway). Transformers utilize multi-head self-attention, which allows every pathway token to interact with every other pathway across all three modalities. Furthermore, transformers allow us to inject graph adjacency directly as an additive attention bias ($B_{\text{bio}}$)."*

#### Q2: "Why didn't BioFusionFormer achieve 99% accuracy?"
> **Answer**: *"In a cohort of 121 patients with 22,000 features, claiming 99% accuracy is almost always an indicator of data leakage (such as scaling or selecting features before splitting the dataset). Unfavorable histology in Wilms tumor has biological heterogeneity and label noise (e.g., focal anaplasia that was missed on biopsy). An AUROC of ~0.91 evaluated across 25 outer folds with strictly in-fold preprocessing is a strong, clinically realistic result that reflects true generalization."*

#### Q3: "What is the Nadeau-Bengio correction, and why did you use it instead of a regular Student's t-test?"
> **Answer**: *"A standard paired Student's t-test assumes that each sample in the comparison is independent. In repeated cross-validation (like our 5 repeats of 5-fold CV), training sets overlap heavily between splits (each pair of folds shares ~75% of their training data). This overlap induces positive correlation between test scores, causing standard t-tests to drastically underestimate variance and produce false-positive significance. The Nadeau-Bengio corrected test adjusts the variance estimator by incorporating the ratio of test to train sample size ($n_{\text{test}} / n_{\text{train}}$), providing a statistically rigorous, conservative hypothesis test."*

#### Q4: "Why did DNA methylation perform so poorly for histology classification?"
> **Answer**: *"Our baseline experiments and confounder audits revealed that DNA methylation alone achieved an AUROC of ~0.45. Biologically, Wilms tumor histology is driven by genetic mutations (such as TP53) and large-scale copy number alterations rather than genome-wide promoter hypermethylation. Methylation in pediatric blastemal tumors exhibits high inter-patient epigenetic variance that reflects developmental cell of origin rather than anaplastic transformation. Crucially, BioFusionFormer's adaptive gating network successfully identified this and downweighted the methylation modality to <15%."*

#### Q5: "How did you prevent data leakage during pathway selection?"
> **Answer**: *"We performed unsupervised pathway selection based solely on feature variance across the cohort, without ever referencing the histology or vital status labels. Furthermore, all data imputation (median) and feature scaling (StandardScaler) were computed strictly inside the training split of each fold, ensuring that the test set remained completely invisible to the pipeline."*

#### Q6: "What is the purpose of the label-permutation control experiment?"
> **Answer**: *"Deep neural networks have sufficient capacity to output arbitrary pathway attributions even when fitting random noise. To prove our attention rollout was identifying true biological signal, we trained the model on 10 random permutations of shuffled labels. The cross-fold pathway stability (Jaccard similarity) on shuffled labels was near zero (~0.12), whereas the real model exhibited high stability (~0.45, $p < 0.001$). This formally proves the explainability reflects true biological patterns rather than structural bias."*

---

*Document compiled for academic presentation, thesis defense, and reproducible open science in computational oncology.*
