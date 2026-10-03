"""
Phase B2-B7 - GDC metadata discovery for TARGET-WT
====================================================
Queries the GDC API (no molecular file downloads):
  B2: Facets (data_category, data_type, experimental_strategy,
              workflow_type, platform, data_format)
  B3: Propose modality filters (mRNA, miRNA, methylation) with counts
  B4: Cases / clinical data
  B5: file_id -> sample_id -> case_id manifest
  B6: Patient overlap table
  B7: Cohort report -> results/cohort_report.md
"""
import sys
import json
import logging
import csv
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
FACET_D = INTERIM / "gdc_facets"
CLIN_D  = ROOT / "data" / "raw" / "target_wt" / "clinical"
RESULTS = ROOT / "results"
LOG     = ROOT / "logs" / "phase_b.log"

for d in [INTERIM, FACET_D, CLIN_D, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("PhaseB")

GDC_BASE = "https://api.gdc.cancer.gov"
UA = "Mozilla/5.0 (compatible; BioFusionFormer/1.0; academic)"
HEADERS = {"Content-Type": "application/json", "User-Agent": UA}
PROJECT = "TARGET-WT"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def gdc_get(endpoint: str, params: dict | None = None,
            retries: int = 4, timeout: int = 60) -> dict | None:
    url = f"{GDC_BASE}/{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS,
                             timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("GET %s attempt %d/%d: %s", endpoint, attempt, retries, exc)
    return None


def gdc_post(endpoint: str, payload: dict,
             retries: int = 4, timeout: int = 60) -> dict | None:
    url = f"{GDC_BASE}/{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("POST %s attempt %d/%d: %s", endpoint, attempt, retries, exc)
    return None


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("  Saved: %s", path)


# ─── B2: Facets ───────────────────────────────────────────────────────────────

FACET_FIELDS = [
    "data_category", "data_type", "experimental_strategy",
    "workflow_type", "platform", "data_format",
]

def fetch_facets() -> dict:
    """Query /files facets for TARGET-WT open-access files."""
    log.info("B2: Fetching facets for %s ...", PROJECT)
    facets_str = ",".join(FACET_FIELDS)
    params = {
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id",
                                         "value": PROJECT}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ]
        }),
        "facets": facets_str,
        "size": "0",
        "format": "json",
    }
    data = gdc_get("files", params)
    if data is None:
        log.error("Facet query failed")
        return {}

    facets = data.get("data", {}).get("aggregations", {})
    save_json(facets, FACET_D / "target_wt_facets.json")
    save_json(data,   FACET_D / "target_wt_facets_full.json")
    return facets


def print_facets(facets: dict) -> None:
    print("\n=== B2: FACETS (TARGET-WT, open access) ===")
    for field in FACET_FIELDS:
        buckets = facets.get(field, {}).get("buckets", [])
        print(f"\n  [{field}]")
        for b in buckets:
            print(f"    {b['key']:50s}  n={b['doc_count']}")


# ─── B3: Modality filters ─────────────────────────────────────────────────────

def build_modality_filters() -> dict[str, dict]:
    """
    Returns proposed filters for each modality based on facet values.
    Values verified against live API above.
    """
    base = [
        {"op": "=", "content": {"field": "cases.project.project_id",  "value": PROJECT}},
        {"op": "=", "content": {"field": "access",                      "value": "open"}},
    ]

    mrna_filter = {
        "op": "and", "content": base + [
            {"op": "=", "content": {"field": "data_category",
                                    "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type",
                                    "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "experimental_strategy",
                                    "value": "RNA-Seq"}},
            # Accept any workflow that produces gene-level counts/FPKM
            {"op": "in", "content": {"field": "workflow_type",
                                     "value": [
                                         "STAR - Counts",
                                         "HTSeq - Counts",
                                         "HTSeq - FPKM",
                                         "HTSeq - FPKM-UQ",
                                         "STAR - FPKM",
                                     ]}},
        ]
    }

    mirna_filter = {
        "op": "and", "content": base + [
            {"op": "=", "content": {"field": "data_category",
                                    "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type",
                                    "value": "Isoform Expression Quantification"}},
            {"op": "=", "content": {"field": "experimental_strategy",
                                    "value": "miRNA-Seq"}},
        ]
    }
    # Also try miRNA expression quantification (not isoform)
    mirna_filter2 = {
        "op": "and", "content": base + [
            {"op": "=", "content": {"field": "data_category",
                                    "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type",
                                    "value": "miRNA Expression Quantification"}},
            {"op": "=", "content": {"field": "experimental_strategy",
                                    "value": "miRNA-Seq"}},
        ]
    }

    meth_filter = {
        "op": "and", "content": base + [
            {"op": "=", "content": {"field": "data_category",
                                    "value": "DNA Methylation"}},
            {"op": "=", "content": {"field": "data_type",
                                    "value": "Methylation Beta Value"}},
            {"op": "in", "content": {"field": "platform",
                                     "value": [
                                         "Illumina Human Methylation 450",
                                         "Illumina Human Methylation 27",
                                     ]}},
        ]
    }

    return {
        "mrna":   mrna_filter,
        "mirna":  mirna_filter,
        "mirna2": mirna_filter2,
        "methylation": meth_filter,
    }


def count_files(filter_obj: dict, label: str) -> tuple[int, list[dict]]:
    """Returns (total count, first page of hits)."""
    params = {
        "filters": json.dumps(filter_obj),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id,"
                  "cases.samples.sample_id,cases.samples.sample_type,"
                  "cases.samples.submitter_id,file_size,md5sum,"
                  "data_type,workflow_type,platform",
        "size": "2000",
        "format": "json",
    }
    data = gdc_get("files", params)
    if data is None:
        log.warning("File count query failed for %s", label)
        return 0, []
    total = data.get("data", {}).get("pagination", {}).get("total", 0)
    hits  = data.get("data", {}).get("hits", [])
    log.info("  %s: %d files", label, total)
    return total, hits


# ─── B4: Cases / clinical ─────────────────────────────────────────────────────

CLINICAL_FIELDS = [
    "case_id", "submitter_id",
    "demographic.gender", "demographic.race", "demographic.ethnicity",
    "demographic.vital_status", "demographic.days_to_death",
    "demographic.days_to_birth",
    "diagnoses.age_at_diagnosis",
    "diagnoses.primary_diagnosis",
    "diagnoses.tumor_stage", "diagnoses.tumor_grade",
    "diagnoses.morphology",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_recurrence",
    "diagnoses.vital_status",
    "diagnoses.last_known_disease_status",
    "diagnoses.progression_or_recurrence",
    "diagnoses.prior_malignancy",
    "exposures.alcohol_history",
    "samples.sample_id", "samples.sample_type", "samples.submitter_id",
]


def fetch_cases() -> list[dict]:
    """Fetch all TARGET-WT cases with clinical fields."""
    log.info("B4: Fetching cases for %s ...", PROJECT)
    filter_obj = {
        "op": "=",
        "content": {"field": "project.project_id", "value": PROJECT}
    }
    all_hits = []
    page_size = 500
    from_idx  = 0

    while True:
        params = {
            "filters": json.dumps(filter_obj),
            "fields":  ",".join(CLINICAL_FIELDS),
            "size":    str(page_size),
            "from":    str(from_idx),
            "format":  "json",
        }
        data = gdc_get("cases", params)
        if data is None:
            log.error("Cases query failed at from=%d", from_idx)
            break
        hits  = data.get("data", {}).get("hits", [])
        total = data.get("data", {}).get("pagination", {}).get("total", 0)
        all_hits.extend(hits)
        log.info("  Cases fetched: %d / %d", len(all_hits), total)
        if len(all_hits) >= total or not hits:
            break
        from_idx += page_size

    return all_hits


def flatten_case(case: dict) -> dict:
    """Flatten a nested GDC case dict into a single-row dict."""
    row = {
        "case_id":       case.get("case_id", ""),
        "submitter_id":  case.get("submitter_id", ""),
    }
    # Demographic
    dem = case.get("demographic", {}) or {}
    for k in ["gender", "race", "ethnicity", "vital_status",
              "days_to_death", "days_to_birth"]:
        row[f"demo_{k}"] = dem.get(k, "")

    # Diagnoses (take first)
    diags = case.get("diagnoses", []) or []
    d = diags[0] if diags else {}
    for k in ["age_at_diagnosis", "primary_diagnosis", "tumor_stage",
              "tumor_grade", "morphology", "days_to_last_follow_up",
              "days_to_recurrence", "vital_status", "last_known_disease_status",
              "progression_or_recurrence", "prior_malignancy"]:
        row[f"diag_{k}"] = d.get(k, "")

    # Exposures (take first)
    exps = case.get("exposures", []) or []
    e = exps[0] if exps else {}
    row["exp_alcohol_history"] = e.get("alcohol_history", "")

    # Samples (list all)
    samples = case.get("samples", []) or []
    row["n_samples"]    = len(samples)
    row["sample_ids"]   = "|".join(s.get("sample_id", "") for s in samples)
    row["sample_types"] = "|".join(s.get("sample_type", "") for s in samples)

    return row


# ─── B5 + B6: Manifest and overlap ───────────────────────────────────────────

def build_manifest_and_overlap(
    hits_dict: dict[str, list[dict]],
    cases_list: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    hits_dict: {modality_label: list of file hits}
    Returns (manifest_df, overlap_df)
    """
    # Build file manifest
    manifest_rows = []
    for modality, hits in hits_dict.items():
        for h in hits:
            file_id   = h.get("file_id", "")
            file_name = h.get("file_name", "")
            file_size = h.get("file_size", 0)
            md5       = h.get("md5sum", "")
            wf        = h.get("workflow_type", "")
            platform  = h.get("platform", "")
            for case_obj in h.get("cases", []):
                case_id     = case_obj.get("case_id", "")
                submitter   = case_obj.get("submitter_id", "")
                for samp in case_obj.get("samples", []):
                    manifest_rows.append({
                        "modality":   modality,
                        "file_id":    file_id,
                        "file_name":  file_name,
                        "file_size":  file_size,
                        "md5sum":     md5,
                        "workflow":   wf,
                        "platform":   platform,
                        "case_id":    case_id,
                        "submitter_id": submitter,
                        "sample_id":  samp.get("sample_id", ""),
                        "sample_type": samp.get("sample_type", ""),
                    })

    manifest_df = pd.DataFrame(manifest_rows)

    # Build per-case overlap
    all_case_ids = {c.get("case_id", "") for c in cases_list}

    # For each modality, which case_ids have files?
    case_modality: dict[str, dict[str, bool]] = {cid: {} for cid in all_case_ids}
    for modality, hits in hits_dict.items():
        for h in hits:
            for case_obj in h.get("cases", []):
                cid = case_obj.get("case_id", "")
                if cid:
                    if cid not in case_modality:
                        case_modality[cid] = {}
                    case_modality[cid][modality] = True

    # Build per-case clinical lookup
    clinical_lookup = {c.get("case_id", ""): flatten_case(c) for c in cases_list}

    overlap_rows = []
    for cid, mods in case_modality.items():
        cl = clinical_lookup.get(cid, {})
        # Survival: need vital_status + time
        vital  = cl.get("demo_vital_status") or cl.get("diag_vital_status", "")
        dtd    = cl.get("demo_days_to_death", "")
        dtlf   = cl.get("diag_days_to_last_follow_up", "")
        has_survival = bool(vital and (dtd or dtlf))

        # Pick best survival time
        survival_time = dtd if dtd else dtlf

        overlap_rows.append({
            "case_id":         cid,
            "submitter_id":    cl.get("submitter_id", ""),
            "has_mrna":        mods.get("mrna", False),
            "has_mirna":       mods.get("mirna", False),
            "has_methylation": mods.get("methylation", False),
            "has_clinical":    cid in clinical_lookup,
            "has_survival":    has_survival,
            "vital_status":    vital,
            "survival_time_days": survival_time,
            "age_at_diagnosis": cl.get("diag_age_at_diagnosis", ""),
            "tumor_stage":     cl.get("diag_tumor_stage", ""),
            "tumor_grade":     cl.get("diag_tumor_grade", ""),
            "morphology":      cl.get("diag_morphology", ""),
            "progression":     cl.get("diag_progression_or_recurrence", ""),
            "prior_malignancy": cl.get("diag_prior_malignancy", ""),
            "gender":          cl.get("demo_gender", ""),
            "n_samples":       cl.get("n_samples", 0),
            "sample_types":    cl.get("sample_types", ""),
        })

    overlap_df = pd.DataFrame(overlap_rows)
    return manifest_df, overlap_df


# ─── B7: Cohort Report ────────────────────────────────────────────────────────

def generate_cohort_report(
    overlap_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    facets: dict,
    modality_counts: dict[str, int],
) -> str:
    o = overlap_df

    n_mrna  = o["has_mrna"].sum()
    n_mirna = o["has_mirna"].sum()
    n_meth  = o["has_methylation"].sum()
    n_clin  = o["has_clinical"].sum()
    n_surv  = o["has_survival"].sum()

    n_mrna_mirna   = (o["has_mrna"] & o["has_mirna"]).sum()
    n_mrna_meth    = (o["has_mrna"] & o["has_methylation"]).sum()
    n_mirna_meth   = (o["has_mirna"] & o["has_methylation"]).sum()
    n_all_3        = (o["has_mrna"] & o["has_mirna"] & o["has_methylation"]).sum()
    n_full_cohort  = (o["has_mrna"] & o["has_mirna"] & o["has_methylation"]
                      & o["has_clinical"]).sum()
    n_full_surv    = (o["has_mrna"] & o["has_mirna"] & o["has_methylation"]
                      & o["has_clinical"] & o["has_survival"]).sum()

    # Candidate labels
    full = o[o["has_mrna"] & o["has_mirna"] & o["has_methylation"] & o["has_clinical"]]

    def label_summary(col: str) -> str:
        if col not in full.columns:
            return "N/A"
        vals = full[col].replace("", pd.NA).dropna()
        miss_pct = 100 * (1 - len(vals) / max(len(full), 1))
        counts = vals.value_counts()
        top = "; ".join(f"{k}:{v}" for k, v in counts.head(8).items())
        return f"{top}  [missing {miss_pct:.0f}%]"

    # Sample type breakdown
    stype_rows = []
    for _, row in o[o["has_mrna"] | o["has_mirna"] | o["has_methylation"]].iterrows():
        stypes = str(row.get("sample_types", "")).split("|")
        for st in stypes:
            st = st.strip()
            if st:
                stype_rows.append(st)
    stype_counts = pd.Series(stype_rows).value_counts().to_dict()

    # Multi-sample patients
    multi_sample = o[(o["has_mrna"] | o["has_mirna"] | o["has_methylation"])
                     & (o["n_samples"].astype(str).str.isnumeric())
                     & (o["n_samples"].astype(float) > 1)]

    # Estimated download size for full cohort
    if not manifest_df.empty and "file_size" in manifest_df.columns:
        fc_cases = set(full["case_id"].tolist())
        fc_files = manifest_df[manifest_df["case_id"].isin(fc_cases)]
        # Deduplicate file_id
        fc_files = fc_files.drop_duplicates(subset="file_id")
        total_bytes = fc_files["file_size"].astype(float).sum()
        total_gb    = total_bytes / 1e9
    else:
        total_gb = float("nan")

    # Recommendation
    if n_full_cohort >= 200:
        rec = ("N is adequate for deep learning. Both survival regression "
               "(C-index) and classification tasks are defensible.")
    elif n_full_cohort >= 100:
        rec = ("N is marginal (~100-200). A survival task is defensible; "
               "classification may be under-powered unless classes are balanced. "
               "Consider cross-validation with careful stratification.")
    elif n_full_cohort >= 50:
        rec = ("N < 100 for 3-modal intersection. Deep learning is risky. "
               "Strongly consider: (1) relaxing to 2-modal + clinical, "
               "(2) expanding to TCGA-WT (Wilms), or (3) combining TARGET + TCGA "
               "cohorts for sufficient N.")
    else:
        rec = ("N < 50 for full 3-modal cohort. Deep multi-modal modelling is "
               "NOT defensible. Must either: (a) relax to 2 omics, (b) use TCGA "
               "kidney cohorts, or (c) reconsider the cancer type.")

    lines = [
        "# TARGET-WT Cohort Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Project: {PROJECT}  |  Access: open",
        "",
        "## 1. Files per Modality (from GDC API)",
        "",
        f"| Modality        | File count (API) |",
        f"|-----------------|-----------------|",
    ]
    for k, v in modality_counts.items():
        lines.append(f"| {k:15s}  | {v:>15,} |")

    lines += [
        "",
        "## 2. Patients per Modality",
        "",
        f"| Group                              | N patients |",
        f"|------------------------------------|-----------|",
        f"| mRNA                               | {n_mrna:>9,} |",
        f"| miRNA                              | {n_mirna:>9,} |",
        f"| Methylation                        | {n_meth:>9,} |",
        f"| Clinical                           | {n_clin:>9,} |",
        f"| Survival data                      | {n_surv:>9,} |",
        "",
        "## 3. Pairwise and Full Intersections",
        "",
        f"| Intersection                       | N patients |",
        f"|------------------------------------|-----------|",
        f"| mRNA ∩ miRNA                       | {n_mrna_mirna:>9,} |",
        f"| mRNA ∩ Methylation                 | {n_mrna_meth:>9,} |",
        f"| miRNA ∩ Methylation                | {n_mirna_meth:>9,} |",
        f"| mRNA ∩ miRNA ∩ Methylation         | {n_all_3:>9,} |",
        f"| + Clinical (full cohort)           | {n_full_cohort:>9,} |",
        f"| + Survival (vital + time)          | {n_full_surv:>9,} |",
        "",
        "## 4. Candidate Classification Labels",
        "",
        f"  vital_status    : {label_summary('vital_status')}",
        f"  tumor_stage     : {label_summary('tumor_stage')}",
        f"  tumor_grade     : {label_summary('tumor_grade')}",
        f"  morphology      : {label_summary('morphology')}",
        f"  progression     : {label_summary('progression')}",
        f"  prior_malignancy: {label_summary('prior_malignancy')}",
        "",
        "## 5. Sample Type Breakdown",
        "",
        "| Sample Type                        | Count |",
        "|------------------------------------|-------|",
    ]
    for st, cnt in sorted(stype_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {st:34s} | {cnt:5d} |")

    lines += [
        "",
        f"Patients with >1 sample per modality: {len(multi_sample)}",
        "(Strategy: prefer 'Primary Tumor' sample; flag multi-sample patients for sensitivity analysis.)",
        "",
        "## 6. Estimated Download Size (Full Cohort Files Only)",
        "",
        f"  ~{total_gb:.1f} GB  (sum of file_size for {n_full_cohort} patients across 3 modalities)",
        "",
        "## 7. Recommendation",
        "",
        rec,
    ]

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # B1 - Check GDC status
    log.info("B1: Checking GDC status ...")
    status_data = gdc_get("status")
    if status_data is None:
        log.error("GDC is unreachable.")
        print("\nGDC UNREACHABLE — launching background poller.")
        sys.exit(2)

    gdc_status = status_data.get("status", "unknown")
    log.info("GDC status: %s  tag=%s  version=%s",
             gdc_status,
             status_data.get("tag", ""),
             status_data.get("version", ""))

    if gdc_status != "OK":
        log.warning("GDC not OK (status=%s). Launching poller.", gdc_status)
        print(f"\nGDC status = {gdc_status}. Run scripts/wait_for_gdc.py to poll.")
        sys.exit(2)

    print(f"\nGDC is UP: status={gdc_status}, "
          f"tag={status_data.get('tag')}, "
          f"version={status_data.get('version')}")

    # B2 - Facets
    facets = fetch_facets()
    if facets:
        print_facets(facets)
    else:
        log.warning("No facet data — using empty dict")

    # B3 - Propose filters and count files
    modality_filters = build_modality_filters()
    log.info("B3: Counting files per modality ...")

    hits_dict: dict[str, list[dict]]  = {}
    modality_counts: dict[str, int]   = {}

    # mRNA
    n_mrna, hits_mrna = count_files(modality_filters["mrna"], "mRNA")
    # If zero, it might be because workflow_type values differ; relax
    if n_mrna == 0:
        log.warning("mRNA count=0 with workflow filter; relaxing workflow_type ...")
        mrna_relaxed = {
            "op": "and", "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
                {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                {"op": "=", "content": {"field": "experimental_strategy", "value": "RNA-Seq"}},
            ]
        }
        n_mrna, hits_mrna = count_files(mrna_relaxed, "mRNA (relaxed)")
        save_json(mrna_relaxed, FACET_D / "filter_mrna_relaxed.json")

    save_json(modality_filters["mrna"], FACET_D / "filter_mrna.json")
    modality_counts["mRNA"]   = n_mrna
    hits_dict["mrna"]         = hits_mrna

    # miRNA
    n_mirna, hits_mirna = count_files(modality_filters["mirna"], "miRNA-isoform")
    if n_mirna == 0:
        n_mirna, hits_mirna = count_files(modality_filters["mirna2"], "miRNA-expr")
        save_json(modality_filters["mirna2"], FACET_D / "filter_mirna.json")
    else:
        save_json(modality_filters["mirna"], FACET_D / "filter_mirna.json")
    modality_counts["miRNA"]  = n_mirna
    hits_dict["mirna"]        = hits_mirna

    # Methylation
    n_meth, hits_meth = count_files(modality_filters["methylation"], "Methylation")
    save_json(modality_filters["methylation"], FACET_D / "filter_methylation.json")
    modality_counts["Methylation"] = n_meth
    hits_dict["methylation"]       = hits_meth

    log.info("File counts: mRNA=%d, miRNA=%d, Methylation=%d",
             n_mrna, n_mirna, n_meth)

    print("\n=== B3: PROPOSED FILTERS & COUNTS ===")
    for label, count in modality_counts.items():
        print(f"  {label:15s}: {count:,} files")

    # B4 - Cases
    cases_list = fetch_cases()
    log.info("Total cases fetched: %d", len(cases_list))

    # Save raw JSON
    save_json({"cases": cases_list}, CLIN_D / "clinical_cases.json")

    # Flatten and save CSV
    flat_rows = [flatten_case(c) for c in cases_list]
    clin_df   = pd.DataFrame(flat_rows)
    clin_df.to_csv(CLIN_D / "clinical_cases.csv", index=False)
    log.info("Saved clinical_cases.csv (%d rows)", len(clin_df))

    # B5+B6 - Manifest and overlap
    log.info("B5/B6: Building manifest and overlap tables ...")
    manifest_df, overlap_df = build_manifest_and_overlap(hits_dict, cases_list)
    manifest_df.to_csv(INTERIM / "target_wt_file_manifest_all.csv", index=False)
    overlap_df.to_csv(INTERIM / "target_wt_overlap.csv", index=False)
    log.info("Saved manifest (%d rows) and overlap (%d rows)",
             len(manifest_df), len(overlap_df))

    # B7 - Report
    log.info("B7: Generating cohort report ...")
    report_text = generate_cohort_report(
        overlap_df, manifest_df, facets, modality_counts
    )
    report_path = RESULTS / "cohort_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Saved: %s", report_path)

    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
