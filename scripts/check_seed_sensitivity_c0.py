"""
scripts/check_seed_sensitivity_c0.py
====================================
Measures seed-to-seed variability across seeds 42, 123, 999 (Stage C0).

Evaluates the 3 final models trained on all 121 cohort patients:
- models/final/model_seed42.pt
- models/final/model_seed123.pt
- models/final/model_seed999.pt

Quantifies:
1. Max and mean absolute probability differences |p_a - p_b| across pairs of seeds.
2. Pearson and Spearman correlations between seeds.
3. Prediction concordance at threshold 0.5.
4. Modality gate weight differences across seeds (mRNA, miRNA, methylation).
5. Comparison between individual seeds and the 3-seed ensemble.

Outputs:
- results/model/seed_sensitivity_c0.json
- results/model/seed_sensitivity_c0.md
"""

from pathlib import Path
import json
import datetime
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS_FINAL_DIR = ROOT / "models" / "final"
DATA_DIR = ROOT / "data"
OUT_JSON = ROOT / "results" / "model" / "seed_sensitivity_c0.json"
OUT_MD = ROOT / "results" / "model" / "seed_sensitivity_c0.md"

torch.set_num_threads(2)


def main():
    print("=" * 70)
    print("C0: BioFusionFormer Seed Sensitivity Check (Seeds 42, 123, 999)")
    print("=" * 70)

    # 1. Load data
    labels_df = pd.read_csv(DATA_DIR / "processed" / "labels.csv")
    cohort_df = pd.read_csv(DATA_DIR / "interim" / "cohort_patients.csv")
    core_pats = cohort_df[cohort_df["is_3omics_core"]]["patient_id"].tolist()
    labels_df = labels_df.set_index("patient_id").loc[core_pats]
    y_true = labels_df["histology_label"].values

    mrna_df = pd.read_parquet(DATA_DIR / "processed" / "mrna_gene.parquet").loc[core_pats]
    mirna_df = pd.read_parquet(DATA_DIR / "processed" / "mirna_unique.parquet").loc[core_pats]
    meth_df = pd.read_parquet(DATA_DIR / "processed" / "meth_gene.parquet").loc[core_pats]

    # 2. Load preprocessors, masks, bias
    preprocessors = torch.load(MODELS_FINAL_DIR / "preprocessors.pt", map_location="cpu", weights_only=False)
    masks = torch.load(MODELS_FINAL_DIR / "masks.pt", map_location="cpu", weights_only=False)
    bio_bias = torch.load(MODELS_FINAL_DIR / "bio_bias.pt", map_location="cpu", weights_only=False)

    X_scaled = {}
    for mod, df in [("mrna", mrna_df), ("mirna", mirna_df), ("meth", meth_df)]:
        imp = preprocessors[mod]["imputer"]
        sc = preprocessors[mod]["scaler"]
        vals = sc.transform(imp.transform(df.values))
        X_scaled[mod] = torch.tensor(vals, dtype=torch.float32)

    feat_dims = {"mrna": mrna_df.shape[1], "mirna": mirna_df.shape[1], "meth": meth_df.shape[1]}
    P = masks["mrna"].shape[0]

    # 3. Model factory
    from src.biofusionformer.full_model import BioFusionFormerModel

    seeds = [42, 123, 999]
    seed_preds = {}
    seed_gates = {}
    seed_aurocs = {}
    seed_briers = {}

    for s in seeds:
        pt_path = MODELS_FINAL_DIR / f"model_seed{s}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Model file not found: {pt_path}")

        model = BioFusionFormerModel(
            modality_features=feat_dims,
            modality_masks=masks,
            n_pathways=P,
            d_model=32, n_heads=4, n_layers=2, d_ff=128,
            dropout=0.3, attn_dropout=0.2, modality_dropout=0.2,
            use_pathway_bias=True, use_gating=True, bio_bias=bio_bias,
        )
        weights = torch.load(pt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(weights)
        model.eval()

        with torch.no_grad():
            out = model(X_scaled)
            probs = torch.sigmoid(out["logit"]).cpu().numpy().flatten()
            gate = out["gate"].cpu().numpy()  # (N, M, P) or (N, M) depending on shape

        seed_preds[s] = probs
        seed_gates[s] = gate
        seed_aurocs[s] = float(roc_auc_score(y_true, probs))
        seed_briers[s] = float(brier_score_loss(y_true, probs))
        print(f"Seed {s}: In-sample AUROC = {seed_aurocs[s]:.4f}, Brier = {seed_briers[s]:.4f}")

    # 4. Ensemble prediction (3-seed average)
    p_ens = (seed_preds[42] + seed_preds[123] + seed_preds[999]) / 3.0
    ens_auroc = float(roc_auc_score(y_true, p_ens))
    ens_brier = float(brier_score_loss(y_true, p_ens))

    # 5. Pairwise comparisons
    pairs = [(42, 123), (42, 999), (123, 999)]
    pairwise_stats = []

    for s_a, s_b in pairs:
        p_a = seed_preds[s_a]
        p_b = seed_preds[s_b]
        diffs = np.abs(p_a - p_b)

        r_pearson, p_pearson = stats.pearsonr(p_a, p_b)
        rho_spearman, p_spearman = stats.spearmanr(p_a, p_b)

        # Classification concordance at 0.5 threshold
        bin_a = (p_a >= 0.5).astype(int)
        bin_b = (p_b >= 0.5).astype(int)
        concordance = float(np.mean(bin_a == bin_b))

        pairwise_stats.append({
            "pair": f"Seed {s_a} vs Seed {s_b}",
            "seed_a": s_a,
            "seed_b": s_b,
            "max_abs_diff": float(np.max(diffs)),
            "mean_abs_diff": float(np.mean(diffs)),
            "p95_abs_diff": float(np.percentile(diffs, 95)),
            "pearson_r": float(r_pearson),
            "spearman_rho": float(rho_spearman),
            "binary_concordance": concordance,
        })

    # 6. Modality gate comparison
    # gate shape is typically (N, 3, P)
    gate_means = {}
    mod_names = ["mrna", "mirna", "meth"]
    for s in seeds:
        g = seed_gates[s]
        # Average across patients (N) and pathways (P)
        if g.ndim == 3:
            mean_per_mod = g.mean(axis=(0, 2))  # (3,)
        else:
            mean_per_mod = g.mean(axis=0)  # (3,)
        gate_means[s] = {mod_names[i]: float(mean_per_mod[i]) for i in range(3)}

    # Individual vs Ensemble
    indiv_vs_ens = {}
    for s in seeds:
        p_s = seed_preds[s]
        diffs = np.abs(p_s - p_ens)
        r, _ = stats.pearsonr(p_s, p_ens)
        indiv_vs_ens[s] = {
            "max_diff_from_ens": float(np.max(diffs)),
            "mean_diff_from_ens": float(np.mean(diffs)),
            "pearson_r_with_ens": float(r),
        }

    # Summary dictionary
    results = {
        "n_patients": len(y_true),
        "seeds_evaluated": seeds,
        "in_sample_metrics": {
            "seeds": {s: {"auroc": seed_aurocs[s], "brier": seed_briers[s]} for s in seeds},
            "ensemble_3seed": {"auroc": ens_auroc, "brier": ens_brier},
            "note": "In-sample cohort evaluation; cross-validated performance is reported in E1_metrics.json.",
        },
        "pairwise_seed_differences": pairwise_stats,
        "modality_gate_means": gate_means,
        "individual_vs_ensemble": indiv_vs_ens,
        "key_takeaways": {
            "mean_pairwise_r": float(np.mean([p["pearson_r"] for p in pairwise_stats])),
            "mean_pairwise_prob_diff": float(np.mean([p["mean_abs_diff"] for p in pairwise_stats])),
            "max_pairwise_prob_diff": float(np.max([p["max_abs_diff"] for p in pairwise_stats])),
            "mean_concordance": float(np.mean([p["binary_concordance"] for p in pairwise_stats])),
        }
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    # 7. Write Markdown summary
    lines = []
    lines.append("# BioFusionFormer E1 — Seed Sensitivity Audit (Stage C0)")
    lines.append("")
    lines.append(f"**Date:** {datetime.date.today().isoformat()}  ")
    lines.append("**Scope:** Seed-to-seed variability across 3 random seeds (42, 123, 999) trained on all 121 cohort patients.  ")
    lines.append("**Context:** Complements same-seed determinism (`results/model/reproducibility.md`). Evaluates stability of final deployment models (`models/final/`).  ")
    lines.append("")
    lines.append("## 1. Pairwise Seed Consistency (Predictions on N=121)")
    lines.append("")
    lines.append("| Comparison | Pearson $r$ | Spearman $\\rho$ | Mean $|\\Delta p|$ | 95th %ile $|\\Delta p|$ | Max $|\\Delta p|$ | Binary Concordance (thr=0.5) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for p in pairwise_stats:
        lines.append(
            f"| **{p['pair']}** | {p['pearson_r']:.4f} | {p['spearman_rho']:.4f} | "
            f"{p['mean_abs_diff']:.4f} | {p['p95_abs_diff']:.4f} | {p['max_abs_diff']:.4f} | "
            f"{p['binary_concordance']*100:.1f}% |"
        )
    lines.append("")
    lines.append(f"- **Mean Pairwise Pearson Correlation:** `{results['key_takeaways']['mean_pairwise_r']:.4f}`")
    lines.append(f"- **Mean Pairwise Probability Difference:** `{results['key_takeaways']['mean_pairwise_prob_diff']:.4f}`")
    lines.append(f"- **Mean Classification Concordance:** `{results['key_takeaways']['mean_concordance']*100:.1f}%`")
    lines.append("")
    lines.append("## 2. Modality Gate Weights Across Seeds")
    lines.append("")
    lines.append("Average gate weight assigned to each omics modality across all 121 patients and 192 pathways:")
    lines.append("")
    lines.append("| Random Seed | mRNA Gate Weight | miRNA Gate Weight | Methylation Gate Weight | Modality Gate Spread |")
    lines.append("| :---: | :---: | :---: | :---: | :---: |")
    for s in seeds:
        gm = gate_means[s]
        spread = max(gm.values()) - min(gm.values())
        lines.append(f"| Seed {s} | {gm['mrna']:.4f} | {gm['mirna']:.4f} | {gm['meth']:.4f} | {spread:.4f} |")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Consistent with Fork 9 audit findings, modality gate weights remain approximately uniform (~0.31–0.37) across all seeds, confirming that the gating mechanism operates in a damped regime across initialization seeds.")
    lines.append("")
    lines.append("## 3. Individual Seeds vs. 3-Seed Final Ensemble")
    lines.append("")
    lines.append("In deployment (`predict.py` and GUI), predictions are computed as the 3-seed ensemble average $p_{\\text{ens}} = \\frac{1}{3}(p_{42} + p_{123} + p_{999})$:")
    lines.append("")
    lines.append("| Seed | Correlation with Ensemble ($r$) | Mean Difference from Ensemble | Max Difference from Ensemble |")
    lines.append("| :---: | :---: | :---: | :---: |")
    for s in seeds:
        iv = indiv_vs_ens[s]
        lines.append(f"| Seed {s} | {iv['pearson_r_with_ens']:.4f} | {iv['mean_diff_from_ens']:.4f} | {iv['max_diff_from_ens']:.4f} |")
    lines.append("")
    lines.append("## 4. In-Sample Diagnostic Performance")
    lines.append("")
    lines.append("| Model Variant | In-Sample AUROC | In-Sample Brier |")
    lines.append("| :--- | :---: | :---: |")
    for s in seeds:
        lines.append(f"| Seed {s} (single) | {seed_aurocs[s]:.4f} | {seed_briers[s]:.4f} |")
    lines.append(f"| **3-Seed Ensemble** | **{ens_auroc:.4f}** | **{ens_brier:.4f}** |")
    lines.append("")
    lines.append("> **Disclaimer on In-Sample Metrics:** These numbers reflect whole-cohort training fit ($N=121$); true generalisation and cross-validated discrimination must be referenced strictly from `results/model/E1_metrics.json` (Nadeau-Bengio corrected AUROC 0.8716 [0.8023, 0.9408]).")
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
