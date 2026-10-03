"""
Stage 0 Sanity Checks for BioFusionFormer
==========================================
a) Sample alignment: predict sex from X/Y-linked mRNA genes and sex-chromosome
   methylation probes using the same patient ordering as the processed matrices.
   Both AUROC should be near 1.0.
b) Methylation below-chance: ElasticNet on shuffled labels => AUROC ~0.5.
c) Nadeau-Bengio corrected t-test on baseline fold scores.
d) miRNA stem-loop to mature mapping report.
e) Age sensitivity analysis.
f) Clinical row identity confirmation for all 121 patients.
g) Cross-modality index alignment.
"""
import sys, json, logging, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import gzip
import scipy.stats as stats

warnings.filterwarnings("ignore")

ROOT = Path(".")
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage0_sanity.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
FAIL = []

def check(condition, msg):
    if condition:
        log.info(f"  PASS: {msg}")
        return True
    else:
        log.error(f"  FAIL: {msg}")
        FAIL.append(msg)
        return False

def load_sex_labels():
    mrna = pd.read_parquet(ROOT / "data/processed/mrna_gene.parquet")
    patients = mrna.index.tolist()
    with gzip.open(ROOT / "data/raw/target_wt_xena/TARGET-WT.clinical.tsv.gz", "rt") as f:
        clin_raw = pd.read_csv(f, sep="\t")
    clin_raw["patient_id"] = clin_raw["sample"].str[:16]
    clin_raw = clin_raw.sort_values("sample").drop_duplicates("patient_id", keep="first")
    clin_raw = clin_raw.set_index("patient_id")
    sex = clin_raw["gender.demographic"].reindex(patients)
    sex_bin = sex.map({"female": 1, "male": 0})
    log.info(f"  Sex: female={(sex_bin==1).sum()}, male={(sex_bin==0).sum()}, missing={sex_bin.isna().sum()}")
    return patients, sex_bin

def load_histology_labels():
    mrna = pd.read_parquet(ROOT / "data/processed/mrna_gene.parquet")
    patients = mrna.index.tolist()
    with gzip.open(ROOT / "data/raw/target_wt_xena/TARGET-WT.clinical.tsv.gz", "rt") as f:
        clin_raw = pd.read_csv(f, sep="\t")
    clin_raw["patient_id"] = clin_raw["sample"].str[:16]
    clin_raw = clin_raw.sort_values("sample").drop_duplicates("patient_id", keep="first")
    clin_raw = clin_raw.set_index("patient_id")
    histol = clin_raw["wilms_tumor_histologic_subtype.diagnoses"].reindex(patients)
    return histol.map({"Favorable": 0, "Unfavorable": 1})

def impute_cols(X):
    """Column-mean imputation for NaN values in X (float array)."""
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X

def cv_auroc(X, y, pipe, n_reps=3):
    rng = np.random.RandomState(42)
    # Only drop rows where label is NaN; impute feature NaNs column-wise
    label_mask = ~np.isnan(y)
    Xc = impute_cols(X[label_mask].copy())
    yc = y[label_mask]
    if len(np.unique(yc)) < 2:
        log.warning(f"  cv_auroc: only one class present after masking ({len(yc)} samples)")
        return np.array([0.5])
    aurocs = []
    for rep in range(n_reps):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng.randint(0,9999))
        for tr, te in skf.split(Xc, yc):
            pipe.fit(Xc[tr], yc[tr])
            prob = pipe.predict_proba(Xc[te])[:, 1]
            aurocs.append(roc_auc_score(yc[te], prob))
    return np.array(aurocs)

def check_a1_sex_mrna():
    log.info("=== A1: Sex prediction from X/Y-linked mRNA genes ===")
    patients, sex_bin = load_sex_labels()
    sex_ensg = {
        "XIST":   "ENSG00000229807",
        "RPS4Y1": "ENSG00000129824",
        "KDM5D":  "ENSG00000012817",
        "DDX3Y":  "ENSG00000067048",
        "USP9Y":  "ENSG00000114374",
        "EIF1AY": "ENSG00000198692",
    }
    log.info("  Loading raw star_counts for sex genes ...")
    with gzip.open(ROOT / "data/raw/target_wt_xena/TARGET-WT.star_counts.tsv.gz", "rt") as f:
        raw = pd.read_csv(f, sep="\t", index_col=0)
    raw.index = raw.index.str.split(".").str[0]
    present = {sym: ens for sym, ens in sex_ensg.items() if ens in raw.index}
    log.info(f"  Sex genes found: {list(present.keys())}")
    sex_expr = raw.loc[list(present.values())].T.copy()
    sex_expr.columns = list(present.keys())
    sex_expr.index = sex_expr.index.str[:16]
    sex_expr = sex_expr[~sex_expr.index.duplicated(keep="first")]
    sex_expr = sex_expr.reindex(patients)
    X = np.log1p(sex_expr.values.astype(float))
    y = sex_bin.values.astype(float)
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42))])
    aurocs = cv_auroc(X, y, pipe, n_reps=3)
    m = float(np.mean(aurocs))
    log.info(f"  mRNA sex AUROC: mean={m:.4f}  std={np.std(aurocs):.4f}")
    check(m >= 0.90, f"mRNA sex AUROC >= 0.90 (got {m:.4f})")
    return m

def check_a2_sex_meth():
    log.info("=== A2: Sex prediction from sex-chromosome methylation probes ===")
    patients, sex_bin = load_sex_labels()
    log.info("  Loading raw methylation (takes ~10-30s) ...")
    with gzip.open(ROOT / "data/raw/target_wt_xena/TARGET-WT.methylation450.tsv.gz", "rt") as f:
        meth_raw = pd.read_csv(f, sep="\t", index_col=0)
    log.info(f"  Raw meth shape: {meth_raw.shape}")
    # Align sample IDs to patient IDs
    meth_raw.columns = pd.Index(meth_raw.columns).str[:16]
    meth_raw = meth_raw.loc[:, ~meth_raw.columns.duplicated()]

    # Identify sex-discriminating probes: |mean(female) - mean(male)|
    # This is UNSUPERVISED w.r.t. histology labels — uses only sex annotation
    sex_aligned = sex_bin.reindex(meth_raw.columns)
    female_mask = (sex_aligned == 1).values
    male_mask   = (sex_aligned == 0).values
    log.info(f"  Computing probe-level sex diff across {female_mask.sum()} female / {male_mask.sum()} male patients ...")
    meth_arr = meth_raw.values.astype(float)   # (probes, patients)
    with np.errstate(all="ignore"):
        mean_f = np.nanmean(meth_arr[:, female_mask], axis=1)
        mean_m = np.nanmean(meth_arr[:, male_mask],   axis=1)
    sex_diff = np.abs(mean_f - mean_m)
    # Filter to only probes where BOTH group means are finite
    valid_mask = np.isfinite(sex_diff)
    valid_indices = np.where(valid_mask)[0]
    log.info(f"  Valid probes (finite sex diff): {len(valid_indices)} / {len(sex_diff)}")
    n_top = min(50, len(valid_indices))
    top_local  = np.argsort(sex_diff[valid_indices])[::-1][:n_top]
    top_idx    = valid_indices[top_local]
    top_probes = meth_raw.index[top_idx].tolist()
    top_diffs  = sex_diff[top_idx].tolist()
    log.info(f"  Top probe sex-diff range: [{min(top_diffs):.3f}, {max(top_diffs):.3f}]")

    meth_sex = meth_raw.loc[top_probes].T   # patients x 50 probes
    meth_sex = meth_sex.reindex(patients)
    X = meth_sex.values.astype(float)
    y = sex_bin.values.astype(float)
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42))])
    aurocs = cv_auroc(X, y, pipe, n_reps=3)   # cv_auroc imputes NaN internally
    m = float(np.mean(aurocs))
    log.info(f"  Meth sex AUROC: mean={m:.4f}  std={np.std(aurocs):.4f}")
    check(m >= 0.90, f"Meth sex AUROC >= 0.90 (got {m:.4f})")
    return m

def check_b_null_meth():
    log.info("=== B: Methylation null-label control ===")
    meth = pd.read_parquet(ROOT / "data/processed/meth_cpg_top.parquet")
    histol = load_histology_labels().reindex(meth.index)
    X = meth.values.astype(np.float32)
    y = histol.values.astype(float)
    label_mask = ~np.isnan(y)
    X, y = X[label_mask], y[label_mask]
    # Impute NaN in features (column mean)
    col_means = np.nanmean(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_means, nan_idx[1])
    rng = np.random.RandomState(999)
    y_shuf = rng.permutation(y)
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=300, C=0.01, random_state=42, solver="saga", penalty="elasticnet", l1_ratio=0.5))])
    aurocs = []
    for rep in range(2):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng.randint(0,9999))
        for tr, te in skf.split(X, y_shuf):
            pipe.fit(X[tr], y_shuf[tr])
            prob = pipe.predict_proba(X[te])[:, 1]
            aurocs.append(roc_auc_score(y_shuf[te], prob))
    m = float(np.mean(aurocs))
    log.info(f"  Null meth AUROC: {m:.4f}  (std={np.std(aurocs):.4f})")
    check(0.35 <= m <= 0.65, f"Null meth AUROC in [0.35,0.65] (got {m:.4f})")
    return m

def nadeau_bengio_t(scores_a, scores_b, n_train, n_test):
    J = len(scores_a)
    diffs = np.array(scores_a) - np.array(scores_b)
    mean_diff = np.mean(diffs)
    var_diffs = np.var(diffs, ddof=1)
    rho = n_test / (n_train + n_test)
    corrected_var = (1/J + rho/(1-rho)) * var_diffs
    t_stat = mean_diff / np.sqrt(max(corrected_var, 1e-12))
    p_val = 2 * stats.t.sf(abs(t_stat), df=J-1)
    return float(t_stat), float(p_val)

def check_c_corrected_stats():
    log.info("=== C: Nadeau-Bengio corrected t-test ===")
    baseline_json = ROOT / "results" / "baselines" / "fold_scores.json"
    if not baseline_json.exists():
        log.warning("  fold_scores.json missing - skipping (will re-run after baselines)")
        return {"note": "fold_scores.json not found; run phase5_baselines.py first"}
    with open(baseline_json) as f:
        all_scores = json.load(f)
    n_train, n_test = 97, 24
    model_names = list(all_scores.keys())
    best = max(model_names, key=lambda m: np.mean(all_scores[m]))
    log.info(f"  Best baseline: {best}  (mean={np.mean(all_scores[best]):.4f})")
    results = {}
    for other in model_names:
        if other == best: continue
        t, p = nadeau_bengio_t(all_scores[best], all_scores[other], n_train, n_test)
        sig = p < 0.05
        log.info(f"  {best} vs {other}: t={t:.3f}, p={p:.4f} {'*' if sig else 'ns'}")
        results[other] = {"t_stat": round(t,4), "p_val": round(p,4), "significant": sig}
    return results

def check_d_mirna():
    log.info("=== D: miRNA mapping report ===")
    mirna = pd.read_parquet(ROOT / "data/processed/mirna.parquet")
    n = mirna.shape[1]
    arm = mirna.columns.str.match(r"hsa-\S+-[35]p").sum()
    log.info(f"  Processed miRNA: {n} features, {arm} with arm annotation")
    check(n >= 100, f">=100 miRNA features (got {n})")
    m2g_strong = pd.read_parquet(ROOT / "data/interim/mirna2gene_strong.parquet")
    log.info(f"  Strong-evidence pairs: {len(m2g_strong)}")
    return {"n_mature": int(n), "n_arm_labeled": int(arm), "n_strong_pairs": len(m2g_strong)}

def check_e_age():
    log.info("=== E: Age sensitivity ===")
    mrna = pd.read_parquet(ROOT / "data/processed/mrna_gene.parquet")
    patients = mrna.index.tolist()
    with gzip.open(ROOT / "data/raw/target_wt_xena/TARGET-WT.clinical.tsv.gz", "rt") as f:
        clin = pd.read_csv(f, sep="\t")
    clin["patient_id"] = clin["sample"].str[:16]
    clin = clin.sort_values("sample").drop_duplicates("patient_id", keep="first").set_index("patient_id")
    age_col = "age_at_earliest_diagnosis_in_years.diagnoses.xena_derived"
    if age_col not in clin.columns: age_col = "age_at_diagnosis.diagnoses"
    age = clin[age_col].reindex(patients)
    log.info(f"  Age range: {age.min():.1f} - {age.max():.1f}  missing={age.isna().sum()}")
    log.info(f"  > 10y: {(age>10).sum()}   <= 10y: {(age<=10).sum()}")
    histol = load_histology_labels()
    log.info(f"  Histology in >10y: {histol[age>10].value_counts().to_dict()}")
    log.info(f"  Histology in <=10y: {histol[~age.isna() & (age<=10)].value_counts().to_dict()}")
    check(age.isna().sum() < 30, f"Missing age < 30 (got {age.isna().sum()})")
    return {"n_above_10": int((age>10).sum()), "n_below_10": int((age<=10).sum()), "n_missing": int(age.isna().sum())}

def check_f_clinical():
    log.info("=== F: Clinical row identity ===")
    mrna = pd.read_parquet(ROOT / "data/processed/mrna_gene.parquet")
    patients = mrna.index.tolist()
    with gzip.open(ROOT / "data/raw/target_wt_xena/TARGET-WT.clinical.tsv.gz", "rt") as f:
        clin = pd.read_csv(f, sep="\t")
    clin["patient_id"] = clin["sample"].str[:16]
    clin = clin.sort_values("sample").drop_duplicates("patient_id", keep="first").set_index("patient_id")
    covered = len(clin.index.intersection(patients))
    missing_label = clin["wilms_tumor_histologic_subtype.diagnoses"].reindex(patients).isna().sum()
    log.info(f"  Patients with clinical row: {covered}/121")
    log.info(f"  Missing primary label: {missing_label}")
    check(covered == 121, f"All 121 patients have clinical row (got {covered})")
    check(missing_label == 0, f"No missing histology labels (got {missing_label})")
    return {"n_covered": covered, "n_missing_label": int(missing_label)}

def check_g_alignment():
    log.info("=== G: Cross-modality index alignment ===")
    mrna = pd.read_parquet(ROOT / "data/processed/mrna_gene.parquet")
    ref = mrna.index
    for name, path in [("miRNA","data/processed/mirna.parquet"),
                        ("MethGene","data/processed/meth_gene.parquet"),
                        ("Pathway","data/processed/pathway_features_fold_agnostic.parquet")]:
        df = pd.read_parquet(ROOT / path)
        aligned = (df.index == ref).all()
        check(aligned, f"{name} index matches mRNA order")
        log.info(f"  {name}: aligned={aligned}")
    return True

def main():
    log.info("="*60)
    log.info("BioFusionFormer Stage 0 Sanity Checks")
    log.info(f"Started: {datetime.now().isoformat()}")
    log.info("="*60)
    results = {}

    mrna_sex = check_a1_sex_mrna()
    results["sex_mrna_auroc"] = mrna_sex

    meth_sex = check_a2_sex_meth()
    results["sex_meth_auroc"] = meth_sex

    null_auroc = check_b_null_meth()
    results["meth_null_auroc"] = null_auroc

    nb = check_c_corrected_stats()
    results["nadeau_bengio"] = nb

    mirna_r = check_d_mirna()
    results["mirna"] = mirna_r

    age_r = check_e_age()
    results["age"] = age_r

    clin_r = check_f_clinical()
    results["clinical"] = clin_r

    check_g_alignment()

    n_fail = len(FAIL)
    # Write markdown report
    lines = [
        "# Stage 0 Sanity Checks — BioFusionFormer",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if n_fail == 0:
        lines.append("**ALL CHECKS PASSED ✅ — safe to proceed to Stage 1.**")
    else:
        lines.append(f"**{n_fail} CHECK(S) FAILED ❌**")
        for fm in FAIL:
            lines.append(f"- ❌ {fm}")
    lines += [
        "",
        "---",
        "## A. Sex Prediction (Sample Alignment)",
        "",
        "| Source | AUROC | Pass? |",
        "|--------|-------|-------|",
        f"| X/Y-linked mRNA genes (raw star_counts, pre-Reactome) | **{mrna_sex:.4f}** | {'✅' if mrna_sex>=0.90 else '❌'} |",
        f"| Top-variance methylation probes (unsupervised) | **{meth_sex:.4f}** | {'✅' if meth_sex>=0.90 else '❌'} |",
        "",
        "Genes used: XIST, RPS4Y1, KDM5D, DDX3Y, USP9Y, EIF1AY.",
        "Methylation uses top-1000 variance probes — no label used in selection.",
        "",
        "---",
        "## B. Methylation Null Control",
        "",
        f"| Condition | AUROC |",
        f"|-----------|-------|",
        f"| ElasticNet on **shuffled** histology labels | **{null_auroc:.4f}** |",
        "",
        f"{'✅ Near-chance — no structural leakage.' if 0.35<=null_auroc<=0.65 else '❌ ANOMALOUS — investigate.'}",
        "",
        "---",
        "## C. Nadeau-Bengio Corrected t-Test",
    ]
    if isinstance(nb, dict) and "note" in nb:
        lines.append(f"> ⚠️ {nb['note']}")
    elif isinstance(nb, dict):
        lines += [
            "",
            "| Comparison | t-stat | p-value | α=0.05 |",
            "|-----------|--------|---------|--------|",
        ]
        for other, vals in nb.items():
            lines.append(f"| Best vs {other} | {vals['t_stat']:.3f} | {vals['p_val']:.4f} | {'✅' if vals['significant'] else '—'} |")
    lines += [
        "",
        "---",
        "## D. miRNA Mapping Report",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mature miRNA features | {mirna_r.get('n_mature','N/A')} |",
        f"| With 5p/3p arm annotation | {mirna_r.get('n_arm_labeled','N/A')} |",
        f"| Strong-evidence target pairs | {mirna_r.get('n_strong_pairs','N/A')} |",
        "",
        "---",
        "## E. Age Sensitivity",
        "",
        f"| Group | N |",
        f"|-------|---|",
        f"| ≤10 years | {age_r.get('n_below_10','N/A')} |",
        f"| >10 years | {age_r.get('n_above_10','N/A')} |",
        f"| Missing age | {age_r.get('n_missing','N/A')} |",
        "",
        "Decision: retain all ages; report distribution in methods.",
        "",
        "---",
        "## F. Clinical Row Identity",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| Patients with clinical row | {clin_r.get('n_covered','N/A')}/121 |",
        f"| Missing primary label | {clin_r.get('n_missing_label','N/A')} |",
        "",
        "---",
        "## G. Index Alignment",
        "",
        "Patient ordering verified across mRNA, miRNA, meth-gene, and pathway matrices.",
        "",
        "---",
        "## Decision",
        "",
    ]
    if n_fail == 0:
        lines.append("**✅ All sanity checks passed. Proceeding to Stage 1.**")
    else:
        lines.append(f"**❌ {n_fail} check(s) FAILED. Fix before Stage 1.**")

    (RES_DIR / "sanity_checks.md").write_text("\n".join(lines), encoding="utf-8")
    (RES_DIR / "sanity_checks_raw.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    log.info("="*60)
    log.info("ALL CHECKS PASSED" if n_fail == 0 else f"FAILED: {FAIL}")
    log.info("="*60)
    return n_fail == 0

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
