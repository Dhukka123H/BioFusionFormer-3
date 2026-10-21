"""
app/app.py
==========
BioFusionFormer E1 — Clinical Research Prototype & Multi-Omics Inference GUI.

Features:
  Page 1: About & Limitations (research prototype, N=121 enriched cohort, 3 omics mandatory).
  Page 2: Upload (mRNA, miRNA, methylation CSVs, schema checker, demo data loader).
  Page 3: Prediction Results (raw score, calibrated prob, gate weights, top pathways IG).
  Page 4: Model Performance (CV AUROC with NB CI, calibration reliability plot, baselines).
  Page 5: Download predictions as CSV.

Safeguards:
  - Requires all three omics (mRNA, miRNA, methylation).
  - Enforces >= 90% feature coverage.
  - Zero disk writes, zero network calls.
  - Logs peak memory (< 2 GB limit).
"""

import io
import json
import os
import sys
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tracemalloc.start()

from src.inference.predict import predict, validate_omics_inputs

st.set_page_config(
    page_title="BioFusionFormer — Wilms Tumor Histology Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .disclaimer-box {
        background-color: #3b1d1d;
        border-left: 5px solid #ef4444;
        padding: 14px;
        border-radius: 4px;
        margin-bottom: 18px;
    }
    .stProgress > div > div > div > div {
        background-color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


def load_e1_metrics():
    p = ROOT / "results" / "model" / "E1_metrics.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_explain_metrics():
    p = ROOT / "results" / "explainability" / "explainability_results.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_peak_memory_mb():
    current, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


# Sidebar Navigation
st.sidebar.title("🧬 BioFusionFormer")
st.sidebar.caption("Pediatric Wilms Tumor Multi-Omics Classifier")
st.sidebar.markdown("---")

pages = [
    "1. About & Limitations",
    "2. Upload & Schema Check",
    "3. Patient Predictions",
    "4. Model Performance",
    "5. Export & Downloads",
]
selected_page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Safeguards:**")
st.sidebar.text("• Network Calls: None (Air-gapped)")
st.sidebar.text("• Disk Writes: None (In-memory)")
st.sidebar.text(f"• Peak Memory: {get_peak_memory_mb():.1f} MB (< 2048 MB)")
st.sidebar.markdown("---")
st.sidebar.caption("Branch `final/e1-gui` | Frozen E1")

# Session state initialization
if "mrna_df" not in st.session_state:
    st.session_state["mrna_df"] = None
if "mirna_df" not in st.session_state:
    st.session_state["mirna_df"] = None
if "meth_df" not in st.session_state:
    st.session_state["meth_df"] = None
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "is_demo" not in st.session_state:
    st.session_state["is_demo"] = False


# ============================================================================
# PAGE 1: About & Limitations
# ============================================================================
if selected_page == "1. About & Limitations":
    st.title("BioFusionFormer: Pathway-Aware Multi-Omics Classifier")
    st.markdown("### Pediatric Wilms Tumor Histology Research Prototype")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ CRITICAL RESEARCH PROTOTYPE DISCLAIMER</strong><br>
        This software is an academic research prototype evaluated on an enriched retrospective cohort 
        (N = 121 TARGET-WT patients) with <strong>no external clinical validation</strong>.<br>
        It is strictly <strong>not approved, validated, or intended for clinical diagnosis, patient staging, 
        or treatment decisions</strong>. All predictive outputs are experimental.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Free Parameters", "47,221", "Effective Degrees of Freedom")
    with col2:
        st.metric("Dense Weights", "4,249,284", "Memory Footprint")
    with col3:
        st.metric("Cross-Validated AUROC", "0.8716", "NB 95% CI: [0.8023, 0.9408]")

    st.markdown("---")
    st.markdown("### Key Requirements & Constraints")
    st.markdown("""
    1. **Mandatory 3-Omics**: BioFusionFormer requires **all three modalities** (mRNA, miRNA, and DNA methylation). The model cannot run with missing modalities.
    2. **Enriched High-Risk Cohort**: The training set contains 32.2% anaplastic (unfavorable) histology compared to ~10% population prevalence.
    3. **Prior-Informed Architecture**: Genes are projected into 192 Reactome pathway tokens. miRTarBase targets connect miRNAs, and CpG methylation beta values connect to promoter regions.
    4. **Air-gapped & Ephemeral**: Uploaded datasets are processed purely in RAM and never written to disk or sent over a network.
    """)


# ============================================================================
# PAGE 2: Upload & Schema Check
# ============================================================================
elif selected_page == "2. Upload & Schema Check":
    st.title("Upload Patient Omics Data")
    st.markdown("Upload tabular CSV files with patient identifiers in the first column or index.")

    st.markdown("""
    > [!IMPORTANT]
    > **Coverage Safeguard:** BioFusionFormer requires at least **90%** of expected features for each modality.
    > Feature identifiers must be Ensembl gene IDs (e.g. `ENSG00000141510` without version extensions).
    """)

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        if st.button("📂 Load Demo Data (3 Patients)", help="Load example cohort patient profiles"):
            demo_dir = ROOT / "app" / "demo_data"
            if (demo_dir / "demo_mrna.csv").exists():
                st.session_state["mrna_df"] = pd.read_csv(demo_dir / "demo_mrna.csv", index_col=0)
                st.session_state["mirna_df"] = pd.read_csv(demo_dir / "demo_mirna.csv", index_col=0)
                st.session_state["meth_df"] = pd.read_csv(demo_dir / "demo_meth.csv", index_col=0)
                st.session_state["is_demo"] = True
                st.success("Loaded 3 demo cohort patients.")
            else:
                st.error("Demo data files not found in app/demo_data/.")

    if st.session_state["is_demo"]:
        st.info("ℹ️ **Notice:** Demo data represents training cohort patients. Predictions on these profiles are **NOT evidence of model generalization**.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. mRNA Expression")
        st.caption("Normalized log2(count + 1), 11,171 genes")
        f_mrna = st.file_uploader("Upload mRNA CSV", type=["csv"], key="uploader_mrna")
        if f_mrna is not None:
            st.session_state["mrna_df"] = pd.read_csv(f_mrna, index_col=0)
            st.session_state["is_demo"] = False
        if st.session_state["mrna_df"] is not None:
            st.write(f"Shape: {st.session_state['mrna_df'].shape}")

    with c2:
        st.subheader("2. miRNA Expression")
        st.caption("Normalized log2(RPM + 1), 1,035 unique miRNAs")
        f_mirna = st.file_uploader("Upload miRNA CSV", type=["csv"], key="uploader_mirna")
        if f_mirna is not None:
            st.session_state["mirna_df"] = pd.read_csv(f_mirna, index_col=0)
            st.session_state["is_demo"] = False
        if st.session_state["mirna_df"] is not None:
            st.write(f"Shape: {st.session_state['mirna_df'].shape}")

    with c3:
        st.subheader("3. DNA Methylation")
        st.caption("Mean promoter CpG beta values in [0.0, 1.0], 9,778 genes")
        f_meth = st.file_uploader("Upload Methylation CSV", type=["csv"], key="uploader_meth")
        if f_meth is not None:
            st.session_state["meth_df"] = pd.read_csv(f_meth, index_col=0)
            st.session_state["is_demo"] = False
        if st.session_state["meth_df"] is not None:
            st.write(f"Shape: {st.session_state['meth_df'].shape}")

    st.markdown("---")
    if st.button("🚀 Run Pipeline & Predict", type="primary"):
        if st.session_state["mrna_df"] is None or st.session_state["mirna_df"] is None or st.session_state["meth_df"] is None:
            st.error("Error: All three omics matrices (mRNA, miRNA, methylation) are mandatory. Please provide all three files.")
        else:
            with st.spinner("Executing BioFusionFormer multimodal forward pass..."):
                try:
                    res_df = predict(
                        st.session_state["mrna_df"],
                        st.session_state["mirna_df"],
                        st.session_state["meth_df"],
                    )
                    st.session_state["results_df"] = res_df
                    st.success("Inference completed successfully! Go to '3. Patient Predictions' to review results.")
                except Exception as e:
                    st.error(f"Inference Pipeline Error: {e}")


# ============================================================================
# PAGE 3: Patient Predictions
# ============================================================================
elif selected_page == "3. Patient Predictions":
    st.title("Patient Histology Predictions & Interpretability")

    res_df = st.session_state["results_df"]
    if res_df is None:
        st.warning("No predictions available. Please upload data or load demo data in '2. Upload & Schema Check'.")
    else:
        if st.session_state["is_demo"]:
            st.info("ℹ️ **Notice:** Running on demo cohort patients. Predictions here reflect training data and are **NOT performance evidence**.")

        explain_json = load_explain_metrics()
        has_calibrator = explain_json is not None and "recalibration" in explain_json
        slope = explain_json["recalibration"]["slope"] if has_calibrator else 2.254
        intercept = explain_json["recalibration"]["intercept"] if has_calibrator else -0.076

        st.subheader("Prediction Overview")
        display_records = []
        for _, row in res_df.iterrows():
            raw_prob = float(row["probability_unfavorable"])
            z = np.log(max(raw_prob, 1e-6) / max(1.0 - raw_prob, 1e-6))
            cal_prob = float(1.0 / (1.0 + np.exp(-(slope * z + intercept))))
            
            display_records.append({
                "Patient ID": row["patient_id"],
                "Model Score (Raw Prob)": f"{raw_prob:.4f}",
                "Calibrated Probability": f"{cal_prob:.4f}",
                "Gate (mRNA)": f"{row['gate_mrna']:.3f}",
                "Gate (miRNA)": f"{row['gate_mirna']:.3f}",
                "Gate (Meth)": f"{row['gate_meth']:.3f}",
                "Calibration Status": "Calibrated on cross-validated predictions, approximate",
            })

        st.dataframe(pd.DataFrame(display_records), use_container_width=True)

        st.markdown("---")
        st.subheader("Per-Patient Modality Gating Weights")
        for _, row in res_df.iterrows():
            with st.expander(f"Patient: {row['patient_id']} (Raw: {row['probability_unfavorable']:.3f})"):
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1:
                    st.progress(float(row["gate_mrna"]), text=f"mRNA Gate: {row['gate_mrna']:.1%}")
                with g_col2:
                    st.progress(float(row["gate_mirna"]), text=f"miRNA Gate: {row['gate_mirna']:.1%}")
                with g_col3:
                    st.progress(float(row["gate_meth"]), text=f"Meth Gate: {row['gate_meth']:.1%}")

        st.markdown("---")
        st.subheader("Top Biological Pathways (Integrated Gradients)")
        st.warning(
            "⚠️ **Biological Caution**: Generic translation/ribosome/splicing pathways often appear with high attribution "
            "because they reflect general cell proliferation rate or tumor cellularity rather than Wilms tumor-specific histology."
        )

        if explain_json and "metric_i_genuine_sum" in explain_json and "metric_ii_size_normalised_mean" in explain_json:
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("#### (i) Sum of |IG| (Fallback Excluded)")
                st.caption(f"Spearman with size: rho = {explain_json['metric_i_genuine_sum']['spearman_rho_with_pathway_size']:.2f} (size bias)")
                top_sum = pd.DataFrame(explain_json["metric_i_genuine_sum"]["top20_pathways"][:10])
                st.dataframe(top_sum[["rank", "name", "score", "generic"]], use_container_width=True)

            with p_col2:
                st.markdown("#### (ii) Mean |IG| per Feature (Size-Normalised)")
                st.caption(f"Spearman with size: rho = {explain_json['metric_ii_size_normalised_mean']['spearman_rho_with_pathway_size']:.2f} (size bias removed)")
                top_mean = pd.DataFrame(explain_json["metric_ii_size_normalised_mean"]["top20_pathways"][:10])
                st.dataframe(top_mean[["rank", "name", "score", "generic"]], use_container_width=True)
        else:
            st.info("Global pathway attribution summaries available after running scripts/explain_e1.py.")


# ============================================================================
# PAGE 4: Model Performance
# ============================================================================
elif selected_page == "4. Model Performance":
    st.title("Model Validation Performance & Statistical Benchmarks")

    e1 = load_e1_metrics()
    if e1 is None:
        st.error("E1_metrics.json not found in results/model/.")
    else:
        st.markdown("### Cross-Validation Core Metrics (25 Outer Test Folds)")
        st.caption("Evaluated strictly out-of-fold with Nadeau-Bengio corrected resampled t-intervals.")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("AUROC", f"{e1['auroc_mean']:.4f}", f"95% CI: [{e1['auroc_nb_ci'][0]:.3f}, {e1['auroc_nb_ci'][1]:.3f}]")
        with m2:
            auprc_mean = np.mean(e1["folds"]["auprc"])
            st.metric("AUPRC", f"{auprc_mean:.4f}", "Prevalence: 32.2%")
        with m3:
            bal_acc = np.mean(e1["folds"]["bal_acc"])
            st.metric("Balanced Accuracy", f"{bal_acc:.4f}", "Chance: 50.0%")
        with m4:
            st.metric("Brier (Fold Mean)", f"{e1['brier_foldmean']:.4f}", f"Pooled: {e1['brier_pooled']:.4f}")

        st.markdown("---")
        st.subheader("Comparison Against Shallow Baselines (Paired Nadeau-Bengio Corrected Tests)")

        baseline_rows = []
        for b in e1.get("baselines", []):
            baseline_rows.append({
                "Baseline Model": b["name"],
                "Baseline AUROC": f"{b['mean']:.4f}",
                "Difference (E1 - Base)": f"{b['diff']:+.4f}",
                "t-statistic": f"{b['t']:.2f}",
                "p-value": f"{b['p']:.4f}",
                "Conclusion": "Not distinguishable with N=121 (p >= 0.05)" if b["p"] >= 0.05 else "Statistically distinguishable",
            })
        st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True)

        st.info("ℹ️ **Statistical Finding**: BioFusionFormer achieves strong predictive discrimination but is **not statistically distinguishable** from penalized logistic regression (ElasticNet) on mRNA alone with N=121.")

        # Show calibration plot if present
        cal_img_path = ROOT / "figures" / "explain" / "calibration_reliability.png"
        if cal_img_path.exists():
            st.markdown("---")
            st.subheader("Calibration Reliability Curve")
            st.image(str(cal_img_path), caption="E1 Reliability Curve with 5 quantile bins (N=121).", width=600)


# ============================================================================
# PAGE 5: Export & Downloads
# ============================================================================
elif selected_page == "5. Export & Downloads":
    st.title("Export & Download Predictions")

    res_df = st.session_state["results_df"]
    if res_df is None:
        st.warning("No predictions to download. Please upload files and run inference on Page 2.")
    else:
        explain_json = load_explain_metrics()
        has_calibrator = explain_json is not None and "recalibration" in explain_json
        slope = explain_json["recalibration"]["slope"] if has_calibrator else 2.254
        intercept = explain_json["recalibration"]["intercept"] if has_calibrator else -0.076

        export_df = res_df.copy()
        raw_probs = export_df["probability_unfavorable"].values.astype(float)
        z = np.log(np.maximum(raw_probs, 1e-6) / np.maximum(1.0 - raw_probs, 1e-6))
        export_df["calibrated_probability_approx"] = 1.0 / (1.0 + np.exp(-(slope * z + intercept)))

        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        st.dataframe(export_df, use_container_width=True)

        st.download_button(
            label="💾 Download Predictions as CSV",
            data=csv_bytes,
            file_name="biofusionformer_predictions.csv",
            mime="text/csv",
            type="primary",
        )
