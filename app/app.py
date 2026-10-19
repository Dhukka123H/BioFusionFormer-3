"""
app/app.py
==========
BioFusionFormer E1 — Interactive Clinical Research Interface (Streamlit)
Local research prototype for multimodal Wilms tumor histology classification.

Design:
  - 5 Navigation Sections:
    1. About & Limitations
    2. Data Upload & Validation
    3. Patient Predictions & Pathway Attributions
    4. Cross-Validated Model Performance
    5. Export & Audit Log
  - Strict privacy & safeguards: local in-memory processing only, no telemetry, no disk persistence.
  - Memory limit safeguard: Peak RSS < 2 GB.
"""

import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.predict import predict, validate_omics_inputs

# Configure page layout and visual theme
st.set_page_config(
    page_title="BioFusionFormer E1 — Clinical Research Interface",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, professional typography and layout
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .badge-prototype {
        background-color: #fee2e2;
        color: #991b1b;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid #f87171;
    }
    .badge-verified {
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid #38bdf8;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.75rem 0;
        color: #92400e;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_current_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def check_calibration_deployability() -> bool:
    """Only if the slope is consistent (within earlier CI) across single repeats may GUI show calibrated probability."""
    exp_json = ROOT / "results" / "explainability" / "explainability_results.json"
    if exp_json.exists():
        try:
            with open(exp_json, "r", encoding="utf-8") as f:
                d = json.load(f)
            return d.get("recalibration", {}).get("deployment_single_repeat_check", {}).get("is_consistent_with_earlier_ci", False)
        except Exception:
            return False
    return False


def apply_logistic_recalibration(p: float, slope: float = 2.254, intercept: float = -0.076) -> float:
    """Applies Platt scaling recalibration to raw cross-validated probabilities."""
    p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit_val = np.log(p_clipped / (1.0 - p_clipped))
    cal_logit = slope * logit_val + intercept
    return float(1.0 / (1.0 + np.exp(-cal_logit)))



# Session state initialization
if "predictions_df" not in st.session_state:
    st.session_state["predictions_df"] = None
if "uploaded_mrna" not in st.session_state:
    st.session_state["uploaded_mrna"] = None
if "uploaded_mirna" not in st.session_state:
    st.session_state["uploaded_mirna"] = None
if "uploaded_meth" not in st.session_state:
    st.session_state["uploaded_meth"] = None
if "demo_loaded" not in st.session_state:
    st.session_state["demo_loaded"] = False


# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧬 BioFusionFormer E1")
    st.markdown("<span class='badge-prototype'>RESEARCH PROTOTYPE</span>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "1. About & Limitations",
            "2. Data Upload & Validation",
            "3. Patient Predictions & Attributions",
            "4. Model Performance Benchmark",
            "5. Export Predictions",
        ],
        index=1 if st.session_state["predictions_df"] is not None else 0,
    )

    st.markdown("---")
    ram_now = get_current_ram_mb()
    st.caption(f"Host Peak RAM: **{ram_now:.1f} MB** / 2048 MB")
    if ram_now > 1800:
        st.warning("Memory utilization approaching 2 GB safety bound.")


# ==============================================================================
# Page 1: About & Limitations
# ==============================================================================
if page == "1. About & Limitations":
    st.markdown("<div class='main-header'>About BioFusionFormer E1</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>A Pathway-Aware Cross-Modal Transformer for Multi-Omics Data Fusion in Pediatric Wilms Tumor</div>",
        unsafe_allow_html=True,
    )

    st.error(
        """
        **MANDATORY CLINICAL DISCLAIMER:**
        This software is a research prototype developed for academic evaluation.
        - **NOT FOR CLINICAL USE:** It has not been approved by the FDA, EMA, or any regulatory body.
        - **NO EXTERNAL VALIDATION:** The model was trained and evaluated on an enriched, single-cohort retrospective dataset ($N=121$, TARGET-WT).
        - **DO NOT USE FOR TREATMENT DECISIONS:** Under no circumstances should output probabilities replace histopathologic review by a certified pathologist.
        """
    )

    st.markdown("### Architectural Principles")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            **1. Hierarchical Prior Masks**
            - 192 Reactome pathways selected unsupervised by feature variance.
            - Features mapped to pathways via genuine Reactome and miRTarBase interactions.
            """
        )
    with col2:
        st.markdown(
            """
            **2. Cross-Modal Attention**
            - Multi-head self-attention with biological graph adjacency bias ($B_{\\text{bio}}$).
            - Joint transformer space ($d_{\\text{model}}=32$, 2 layers, 4 heads).
            """
        )
    with col3:
        st.markdown(
            """
            **3. Modality Gating**
            - Gating network dynamically weights contributions from mRNA, miRNA, and methylation.
            - All three omics are strictly required.
            """
        )

    st.markdown("---")
    st.markdown("### Documented Limitations & Safeguards")
    st.markdown(
        r"""
        1. **All Three Omics Required:** BioFusionFormer cannot operate with a missing modality. Uploading partial omics triggers an immediate validation stop.
        2. **Coverage Bound ($\ge 90\%$):** Inputs missing more than 10% of expected gene/miRNA features are refused to protect predictive integrity.
        3. **Enriched Cohort Bias:** High-risk TARGET-WT cohort contains 32.2% unfavorable histology (population incidence is ~10%).
        4. **In-Memory Privacy:** No uploaded patient matrices are persisted to disk or sent across any network connection.
        """
    )


# ==============================================================================
# Page 2: Data Upload & Validation
# ==============================================================================
elif page == "2. Data Upload & Validation":
    st.markdown("<div class='main-header'>Data Upload & Schema Validation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Upload patient-aligned mRNA, miRNA, and DNA Methylation CSV matrices</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "**Required Schema:** Rows = Patients (index column: `patient_id`), Columns = Features.\n"
        "- **mRNA:** Ensembl gene IDs (without version suffix, e.g., `ENSG00000141510`), $\\log_2(\\text{count} + 1) \\ge 0$.\n"
        "- **miRNA:** miRBase precursor IDs (1,035 unique), $\\log_2(\\text{RPM} + 1) \\ge 0$.\n"
        "- **Methylation:** Promoter mean beta values bounded in $[0.0, 1.0]$."
    )

    # Demo Loader Button
    col_demo, col_status = st.columns([1, 3])
    with col_demo:
        if st.button("📥 Load Demo Data (3 Patients)", help="Loads pre-validated cohort training samples"):
            demo_dir = ROOT / "data" / "demo"
            if (demo_dir / "demo_mrna.csv").exists():
                st.session_state["uploaded_mrna"] = pd.read_csv(demo_dir / "demo_mrna.csv", index_col=0)
                st.session_state["uploaded_mirna"] = pd.read_csv(demo_dir / "demo_mirna.csv", index_col=0)
                st.session_state["uploaded_meth"] = pd.read_csv(demo_dir / "demo_meth.csv", index_col=0)
                st.session_state["demo_loaded"] = True
                st.success("Loaded 3 demo patients successfully.")
            else:
                st.error("Demo files not found in data/demo/.")

    if st.session_state["demo_loaded"]:
        st.markdown(
            "<div class='warning-box'>⚠️ <b>Demo Notice:</b> Loaded patients are from the retrospective training cohort. "
            "Predictions on these patients reflect in-sample inference and are NOT evidence of generalisation.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. mRNA Expression CSV**")
        file_mrna = st.file_uploader("Upload mRNA", type=["csv"], key="uploader_mrna")
        if file_mrna is not None:
            st.session_state["uploaded_mrna"] = pd.read_csv(file_mrna, index_col=0)
            st.session_state["demo_loaded"] = False
        if st.session_state["uploaded_mrna"] is not None:
            st.caption(f"Shape: {st.session_state['uploaded_mrna'].shape[0]} patients, {st.session_state['uploaded_mrna'].shape[1]} genes")

    with c2:
        st.markdown("**2. miRNA Expression CSV**")
        file_mirna = st.file_uploader("Upload miRNA", type=["csv"], key="uploader_mirna")
        if file_mirna is not None:
            st.session_state["uploaded_mirna"] = pd.read_csv(file_mirna, index_col=0)
            st.session_state["demo_loaded"] = False
        if st.session_state["uploaded_mirna"] is not None:
            st.caption(f"Shape: {st.session_state['uploaded_mirna'].shape[0]} patients, {st.session_state['uploaded_mirna'].shape[1]} miRNAs")

    with c3:
        st.markdown("**3. DNA Methylation Beta CSV**")
        file_meth = st.file_uploader("Upload Methylation", type=["csv"], key="uploader_meth")
        if file_meth is not None:
            st.session_state["uploaded_meth"] = pd.read_csv(file_meth, index_col=0)
            st.session_state["demo_loaded"] = False
        if st.session_state["uploaded_meth"] is not None:
            st.caption(f"Shape: {st.session_state['uploaded_meth'].shape[0]} patients, {st.session_state['uploaded_meth'].shape[1]} CpGs")

    st.markdown("---")
    if st.button("🚀 Validate & Execute Predictions", type="primary"):
        df_mrna = st.session_state["uploaded_mrna"]
        df_mirna = st.session_state["uploaded_mirna"]
        df_meth = st.session_state["uploaded_meth"]

        if df_mrna is None or df_mirna is None or df_meth is None:
            st.error("All three modalities (mRNA, miRNA, methylation) are required. Please upload all files.")
        else:
            with st.spinner("Executing zero-leakage preprocessing and BioFusionFormer inference..."):
                try:
                    res_df = predict(df_mrna, df_mirna, df_meth)
                    is_deployable = check_calibration_deployability()
                    if is_deployable:
                        res_df["calibrated_probability"] = res_df["probability_unfavorable"].apply(
                            apply_logistic_recalibration
                        )
                    else:
                        res_df["calibrated_probability"] = None
                    st.session_state["predictions_df"] = res_df
                    st.success("Inference complete! Navigate to '3. Patient Predictions & Attributions' to view results.")
                except Exception as e:
                    st.error(f"Validation / Inference Failed: {str(e)}")


# ==============================================================================
# Page 3: Patient Predictions & Attributions
# ==============================================================================
elif page == "3. Patient Predictions & Attributions":
    st.markdown("<div class='main-header'>Patient Predictions & Explanations</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Individual histologic risk probabilities, modality gate weights, and pathway attributions</div>",
        unsafe_allow_html=True,
    )

    res_df = st.session_state.get("predictions_df")
    if res_df is None:
        st.warning("No predictions available. Please upload data or load demo data in '2. Data Upload & Validation'.")
    else:
        st.markdown("### Patient Predictions Table")

        is_deployable = check_calibration_deployability()
        if is_deployable and "calibrated_probability" in res_df and res_df["calibrated_probability"].iloc[0] is not None:
            display_cols = [
                "patient_id",
                "probability_unfavorable",
                "calibrated_probability",
                "gate_mrna",
                "gate_mirna",
                "gate_meth",
                "warnings",
            ]
            show_df = res_df[display_cols].copy()
            show_df.columns = [
                "Patient ID",
                "Raw Model Score",
                "Calibrated Prob (approx)",
                "mRNA Gate",
                "miRNA Gate",
                "Meth Gate",
                "Data Warnings",
            ]
            st.dataframe(
                show_df.style.format(
                    {
                        "Raw Model Score": "{:.4f}",
                        "Calibrated Prob (approx)": "{:.4f}",
                        "mRNA Gate": "{:.3f}",
                        "miRNA Gate": "{:.3f}",
                        "Meth Gate": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )
            st.caption(
                "*Note on calibration:* Calibrated probabilities apply Platt scaling ($s=2.254, b=-0.076$) "
                "fitted on cross-validated out-of-fold predictions. Labelled as approximate."
            )
        else:
            display_cols = [
                "patient_id",
                "probability_unfavorable",
                "gate_mrna",
                "gate_mirna",
                "gate_meth",
                "warnings",
            ]
            show_df = res_df[display_cols].copy()
            show_df.columns = [
                "Patient ID",
                "Raw Model Score (Not Calibrated)",
                "mRNA Gate",
                "miRNA Gate",
                "Meth Gate",
                "Data Warnings",
            ]
            st.dataframe(
                show_df.style.format(
                    {
                        "Raw Model Score (Not Calibrated)": "{:.4f}",
                        "mRNA Gate": "{:.3f}",
                        "miRNA Gate": "{:.3f}",
                        "Meth Gate": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )
            st.caption(
                "⚠️ **Deployment Recalibration Audit:** Single-repeat deployment calibration slopes (1.434–1.974, pooled 1.731 [1.352, 2.286]) "
                "deviate from the repeat-averaged ensemble CI [1.769, 3.154]. In accordance with the deployment validation protocol, "
                "probabilities are reported as **Raw Model Score (Not Calibrated)**."
            )

        st.markdown("---")
        st.markdown("### Multimodal Fusion Weights")
        c1, c2, c3 = st.columns(3)
        mean_mrna_g = res_df["gate_mrna"].mean()
        mean_mirna_g = res_df["gate_mirna"].mean()
        mean_meth_g = res_df["gate_meth"].mean()

        with c1:
            st.metric("Mean mRNA Weight", f"{mean_mrna_g:.1%}")
            st.progress(float(mean_mrna_g))
        with c2:
            st.metric("Mean miRNA Weight", f"{mean_mirna_g:.1%}")
            st.progress(float(mean_mirna_g))
        with c3:
            st.metric("Mean Methylation Weight", f"{mean_meth_g:.1%}")
            st.progress(float(mean_meth_g))

        st.markdown("---")
        st.markdown("### Top Biological Pathways (Integrated Gradients)")
        st.markdown(
            "<div class='warning-box'>⚠️ <b>Biological Attribution Caveat:</b> "
            "Generic translation, ribosome, and splicing pathways may reflect cellular proliferation rate, "
            "tumor cell content, or technical factors rather than histology-specific Wilms tumor driver biology. "
            "Both genuine member feature sums and size-normalised attributions are shown side-by-side.</div>",
            unsafe_allow_html=True,
        )

        tab_sum, tab_norm = st.tabs(["(i) Genuine Member Sum (Fallback Excluded)", "(ii) Size-Normalised Mean |IG|"])

        with tab_sum:
            st.markdown("**Top Pathways by Total Attribution Sum (excluding 726 orphaned miRNA edges):**")
            sum_data = [
                {"Rank": 1, "Pathway": "mRNA Splicing", "ID": "R-HSA-72172", "Score": 2.507, "Generic": True},
                {"Rank": 2, "Pathway": "mRNA Splicing - Major Pathway", "ID": "R-HSA-72163", "Score": 2.482, "Generic": True},
                {"Rank": 3, "Pathway": "Eukaryotic Translation Initiation", "ID": "R-HSA-72613", "Score": 2.213, "Generic": True},
                {"Rank": 4, "Pathway": "Cap-dependent Translation Initiation", "ID": "R-HSA-72737", "Score": 2.213, "Generic": True},
                {"Rank": 5, "Pathway": "Cell Cycle", "ID": "R-HSA-1640170", "Score": 1.945, "Generic": False},
                {"Rank": 6, "Pathway": "TP53 Regulates Transcription of Genes", "ID": "R-HSA-6804116", "Score": 1.832, "Generic": False},
            ]
            st.dataframe(pd.DataFrame(sum_data), use_container_width=True)

        with tab_norm:
            st.markdown("**Top Pathways by Size-Normalised Mean |IG| (per member feature):**")
            norm_data = [
                {"Rank": 1, "Pathway": "Transcriptional regulation by RUNX1", "ID": "R-HSA-8878171", "Score": 0.0412, "Generic": False},
                {"Rank": 2, "Pathway": "TP53 Regulates Transcription of Genes", "ID": "R-HSA-6804116", "Score": 0.0385, "Generic": False},
                {"Rank": 3, "Pathway": "WNT Signaling Pathway", "ID": "R-HSA-195721", "Score": 0.0351, "Generic": False},
                {"Rank": 4, "Pathway": "Cellular Senescence", "ID": "R-HSA-2559583", "Score": 0.0318, "Generic": False},
                {"Rank": 5, "Pathway": "Eukaryotic Translation Initiation", "ID": "R-HSA-72613", "Score": 0.0158, "Generic": True},
            ]
            st.dataframe(pd.DataFrame(norm_data), use_container_width=True)


# ==============================================================================
# Page 4: Model Performance Benchmark
# ==============================================================================
elif page == "4. Model Performance Benchmark":
    st.markdown("<div class='main-header'>Cross-Validated Model Performance</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Repeated Stratified 5-Fold Cross-Validation (25 outer test folds) on TARGET-WT (N=121)</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cross-Validated AUROC", "0.8716", help="Nadeau-Bengio 95% CI: [0.8023, 0.9408]")
        st.caption("95% CI: **[0.8023, 0.9408]**")
    with col2:
        st.metric("Cross-Validated AUPRC", "0.8332", help="Prevalence 32.2%")
        st.caption("95% CI: **[0.7434, 0.9230]**")
    with col3:
        st.metric("Balanced Accuracy", "0.7653")
        st.caption("95% CI: **[0.6821, 0.8486]**")
    with col4:
        st.metric("Brier Score (Pooled)", "0.1365", help="Ensemble-pooled OOF predictions")
        st.caption("Fold-mean: **0.1436**")

    st.markdown("---")
    st.markdown("### Benchmark Comparison Against Baselines")
    st.info(
        "**Statistical Interpretation:** Differences evaluated with the paired Nadeau-Bengio corrected resampled $t$-test. "
        "Differences with $p \\ge 0.05$ indicate the model is **not statistically distinguishable** with $N=121$."
    )

    baseline_data = [
        {"Model": "Early Fusion ElasticNet", "AUROC": 0.9295, "Difference (E1 - Base)": -0.0580, "p-value": 0.1778, "Status": "Not distinguishable (p >= 0.05)"},
        {"Model": "mRNA ElasticNet", "AUROC": 0.9169, "Difference (E1 - Base)": -0.0453, "p-value": 0.2996, "Status": "Not distinguishable (p >= 0.05)"},
        {"Model": "BioFusionFormer E1", "AUROC": 0.8716, "Difference (E1 - Base)": 0.0000, "p-value": 1.0000, "Status": "Primary Architecture"},
        {"Model": "miRNA ElasticNet", "AUROC": 0.8481, "Difference (E1 - Base)": 0.0235, "p-value": 0.6209, "Status": "Not distinguishable (p >= 0.05)"},
        {"Model": "Clinical-only (Age + Sex)", "AUROC": 0.6296, "Difference (E1 - Base)": 0.2419, "p-value": 0.0022, "Status": "Statistically distinguishable"},
        {"Model": "Methylation ElasticNet", "AUROC": 0.4601, "Difference (E1 - Base)": 0.4114, "p-value": "< 0.0001", "Status": "Statistically distinguishable"},
    ]
    st.dataframe(pd.DataFrame(baseline_data), use_container_width=True)

    fig_cal = ROOT / "figures" / "explain" / "calibration_reliability.png"
    if fig_cal.exists():
        st.markdown("### Reliability & Calibration Plot")
        st.image(str(fig_cal), caption="E1 Calibration Reliability Curve (5 quantile bins with sample counts)", width=550)


# ==============================================================================
# Page 5: Export Predictions
# ==============================================================================
elif page == "5. Export Predictions":
    st.markdown("<div class='main-header'>Export Predictions & Audit Log</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Download prediction records as CSV for audit, research recordkeeping, or validation</div>",
        unsafe_allow_html=True,
    )

    res_df = st.session_state.get("predictions_df")
    if res_df is None:
        st.warning("No predictions available for export. Please run inference first.")
    else:
        csv_buffer = io.StringIO()
        res_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="💾 Download Predictions CSV",
            data=csv_data,
            file_name="biofusionformer_predictions.csv",
            mime="text/csv",
            type="primary",
        )

        st.markdown("### Preview of Export Data")
        st.dataframe(res_df.head(10), use_container_width=True)
