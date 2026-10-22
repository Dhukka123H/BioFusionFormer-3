# BioFusionFormer E1 — Empirical Claims Registry

> **Status:** Authoritative Guidance for Scientific Reporting, Supervisor Review, and Manuscript Drafts.  
> **Source-of-Truth:** `results/model/E1_metrics.json`, `results/explainability/explainability_results.json`, `results/model/seed_sensitivity_c0.md`.

---

## 1. Permitted Claims (Empirically Supported)

All permitted claims below are supported by rigorous out-of-fold cross-validation, matched null permutations, and paired statistical testing.

### 1.1 Predictive Discrimination
- **Claim:** BioFusionFormer E1 achieves strong internal cross-validated discrimination for unfavorable vs. favorable histology in pediatric Wilms tumor on the enriched TARGET-WT retrospective cohort ($N=121$, prevalence 32.2%).
  - **Empirical Evidence:**
    - Mean AUROC: **$0.8716$** (95% Nadeau-Bengio corrected CI: `[0.8023, 0.9408]`).
    - Mean AUPRC: **$0.8041$** (baseline cohort prevalence $32.2\%$).
    - Balanced Accuracy: **$0.8177$** (chance level $50.0\%$).
    - Fold-Mean Brier Score: **$0.1365$**; Pooled Brier Score: **$0.1365$**.

### 1.2 Architecture and Parameter Efficiency
- **Claim:** BioFusionFormer employs biological prior masking across 192 Reactome pathway tokens, achieving parameter efficiency in its prior-guided projection layers.
  - **Empirical Evidence:**
    - Free learnable parameters: **47,221** effective degrees of freedom.
    - Dense representation parameters: **4,249,284** memory footprint.
    - *(Note: The 62,450 figure that appeared in an earlier draft was unverified and could not be reproduced).*

### 1.3 Explainability Audit & Methodological Transparency
- **Claim:** Dual-aggregation Integrated Gradients (genuine member sum vs. size-normalised mean) successfully exposes severe pathway size bias and resolves technical architectural artefacts.
  - **Empirical Evidence:**
    - Fallback wiring audit: Regulation of ornithine decarboxylase (ODC, `R-HSA-350562`) was confirmed as an artefact of Fork 6 fallback routing (726 orphaned miRNAs mapped to token $P-1$) and drops out once fallback edges are excluded.
    - Pathway size dominance: Genuine member sum attributions correlate strongly with pathway feature count (Spearman $\rho = 0.9099, p = 1.60 \times 10^{-74}$).
    - Generic translation dominance: **15 of the top-20 pathways (75%)** sit in Cluster 1 (Translation / Ribosome machinery); **18 of the top-20 mRNA genes (90%)** are ribosomal protein genes (`RPL` and `RPS` families).
    - Null baseline comparison: After size-normalisation, pathway attribution stability across 5 folds ($0.2765$) falls entirely within the matched label-permutation null distribution ($0.2306\text{--}0.3719$, null mean $0.2870$).

### 1.4 Clinical Prototype Deployment Safeguards
- **Claim:** The deployment interface implements air-gapped execution, input validation (requiring all 3 modalities and $\ge 90\%$ feature coverage), and explicit calibration transparency.
  - **Empirical Evidence:**
    - Predictions display raw model scores with an explicit **"Not Calibrated"** label.
    - Single-repeat out-of-fold calibration slopes ($1.434\text{--}1.974$, with repeats 3, 4, and 5 falling below $1.769$) do not satisfy consistency with the repeat-averaged 95% CI `[1.769, 3.154]`, properly precluding premature calibration claims.

---

## 2. Forbidden Claims (Statistically or Methodologically Unsupported)

The following claims are strictly **unsupported** by empirical evidence and must **never** be made in any report, thesis, paper, or presentation.

| # | Forbidden Claim | Reason & Empirical Counter-Evidence |
| :---: | :--- | :--- |
| **F1** | *"BioFusionFormer significantly outperforms shallow or linear baselines."* | **False.** In paired Nadeau-Bengio corrected tests across 25 folds, E1 ($0.8716$) is **not statistically distinguishable** from ElasticNet on mRNA alone ($0.8587$, $\Delta = +0.0129, p = 0.5898$), Logistic Regression ($0.8441$, $\Delta = +0.0275, p = 0.3807$), or Random Forest ($0.8354$, $\Delta = +0.0362, p = 0.1479$). With $N=121$, no complex architecture advantage can be claimed. |
| **F2** | *"BioFusionFormer provides superior prognostic value compared to clinical staging."* | **False.** Clinical EHR baseline (EHR2: Stage, Age, Sex, Histology) achieves an AUROC of **$0.8893$** (vs. E1 $0.8716$, difference $-0.0177$, 95% NB CI `[-0.1213, +0.0860]`, $p = 0.7224$). Omics features do not demonstrate statistically significant superiority over standard clinical variables. |
| **F3** | *"Integrated Gradients identified novel Wilms tumor biological drivers or therapeutic pathway targets."* | **False.** Real size-normalised stability ($0.277$) does not exceed matched label-permutation nulls ($0.23\text{--}0.37$). Top pathways and genes reflect generic cellular proliferation machinery (ribosomes, translation elongation, splicing) and pathway size ($\rho = 0.91$). IG and ElasticNet share negligible overlap (Jaccard $0.01$, rank correlation $\rho = 0.02$). |
| **F4** | *"The model outputs calibrated clinical probabilities of unfavorable histology."* | **False.** Deployment uses uncalibrated model outputs. Cross-validated single-repeat slopes vary widely ($1.434\text{--}1.974$), and the deployment model was trained on the full cohort without cross-validation. GUI strictly displays raw scores with "Not Calibrated". |
| **F5** | *"The model is robust to random initialization seed variations ($r \approx 1.0000$)."* | **False.** The $r \approx 1.0000$ correlation was an artefact of evaluating whole-cohort deployment models in-sample on overfitted training data. Real 5-fold cross-validated seed evaluation reveals a standard deviation of $\mathbf{0.1210}$ (Delta ranging from $+0.0938$ to $-0.1953$), substantially exceeding the $0.03$ stability threshold. |
| **F6** | *"BioFusionFormer is ready for clinical deployment, patient triage, or therapy selection."* | **False.** Academic research prototype only. Evaluated on an enriched retrospective cohort ($N=121$) with no prospective or external multi-center validation. Not cleared or approved by any regulatory body. |
| **F7** | *"miRNA fallback routing to ODC represents a biological interaction."* | **False.** Fully confirmed as a technical routing artefact where 726 orphaned miRNAs without Reactome prior edges were wired into token index $P-1=191$. |

---

## 3. Summary Compliance Checklist

Before submitting code, reports, or text to supervisors or journals:
- [x] All AUROC comparisons cite paired Nadeau-Bengio corrected $p$-values and intervals.
- [x] Parameter counts state 47,221 free and 4,249,284 dense weights.
- [x] Attributions are described as size-dominated and reflective of generic cellular machinery, not novel disease biology.
- [x] Model outputs in the GUI are labeled "Not Calibrated".
- [x] Seed sensitivity reports cross-validated variability ($\text{SD} = 0.1210$) alongside the in-sample explanation.
- [x] Clinical disclaimer is prominent on all interfaces and documentation.
