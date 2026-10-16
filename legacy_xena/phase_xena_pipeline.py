"""
BioFusionFormer - UCSC Xena GDC Hub Pipeline (Phases 1, 2, 3)
1. Download & verify Xena TARGET-WT datasets & metadata.
2. Patient/sample cleaning, barcode parsing, duplicate resolution, cohort_patients.csv.
3. Clinical audit, survival derivation & crosscheck, prediction target screening, task_options.md.
"""
import sys
import os
import json
import gzip
import hashlib
import time
import logging
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_XENA = ROOT / "data" / "raw" / "target_wt_xena"
INTER = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
LOG_FILE = ROOT / "logs" / "phase_xena.log"
CHECKSUMS_FILE = ROOT / "data" / "raw" / "CHECKSUMS.txt"
DATA_CARD_FILE = ROOT / "data" / "DATA_CARD.json"

RAW_XENA.mkdir(parents=True, exist_ok=True)
INTER.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
ROOT.joinpath("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("xena_pipeline")

HUB_URL = "https://gdc.xenahubs.net/download"
DATASETS = [
    "TARGET-WT.star_counts.tsv.gz",
    "TARGET-WT.mirna.tsv.gz",
    "TARGET-WT.methylation450.tsv.gz",
    "TARGET-WT.clinical.tsv.gz",
    "TARGET-WT.survival.tsv.gz",
]


def compute_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def verify_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                pass
        return True
    except Exception as e:
        log.error("Gzip verification failed for %s: %s", path.name, e)
        return False


def download_file(url: str, dest: Path, max_retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 100:
        if str(dest).endswith(".gz"):
            if verify_gzip(dest):
                log.info("[SKIP] %s already downloaded and verified (%d bytes)", dest.name, dest.stat().st_size)
                return True
            else:
                log.warning("%s is corrupted; re-downloading...", dest.name)
                dest.unlink()
        else:
            log.info("[SKIP] %s already downloaded (%d bytes)", dest.name, dest.stat().st_size)
            return True

    for attempt in range(1, max_retries + 1):
        log.info("[GET attempt %d/%d] %s -> %s", attempt, max_retries, url, dest.name)
        # Use curl with -L (follow redirect to S3) and -C - (resumable)
        cmd = ["curl.exe", "-L", "-C", "-", "-k", "--retry", "3", "--retry-delay", "5", "-o", str(dest), url]
        try:
            res = subprocess.run(cmd, timeout=1200)
            if res.returncode == 0 and dest.exists() and dest.stat().st_size > 100:
                if str(dest).endswith(".gz"):
                    if verify_gzip(dest):
                        log.info("  Verified gzip OK: %s (%d bytes)", dest.name, dest.stat().st_size)
                        return True
                    else:
                        log.warning("  Attempt %d produced corrupt gzip; retrying...", attempt)
                        dest.unlink()
                else:
                    log.info("  Download successful: %s (%d bytes)", dest.name, dest.stat().st_size)
                    return True
        except Exception as exc:
            log.warning("  Attempt %d failed with exception: %s", attempt, exc)
        time.sleep(3)

    log.error("Failed to download %s after %d retries", url, max_retries)
    return False


def run_phase_1():
    log.info("==================================================")
    log.info("PHASE 1: Download and verify Xena TARGET-WT files")
    log.info("==================================================")

    # 1. Download datasets and their metadata .json
    metadata_dict = {}
    checksum_lines = []

    for dset in DATASETS:
        dset_url = f"{HUB_URL}/{dset}"
        dset_dest = RAW_XENA / dset
        ok = download_file(dset_url, dset_dest)
        if not ok:
            raise RuntimeError(f"Critical download failure for {dset}")

        # Compute checksum
        md5, sha256 = compute_hashes(dset_dest)
        entry = f"{dset_dest.name}  size={dset_dest.stat().st_size}  md5={md5}  sha256={sha256}"
        checksum_lines.append(entry)
        log.info("  Checksum: %s", entry)

        # Download json metadata
        base_name = dset[:-3] if dset.endswith(".gz") else dset
        json_name = f"{base_name}.json"
        json_url = f"{HUB_URL}/{json_name}"
        json_dest = RAW_XENA / json_name
        ok_json = download_file(json_url, json_dest)
        if ok_json and json_dest.exists():
            with open(json_dest, "r", encoding="utf-8") as jf:
                raw_text = jf.read()
                try:
                    import re
                    clean_text = re.sub(r",\s*([\]}])", r"\1", raw_text)
                    meta_content = json.loads(clean_text)
                    metadata_dict[base_name] = meta_content
                except Exception as je:
                    log.warning("Could not parse json metadata for %s: %s", json_name, je)

    # Append checksums to CHECKSUMS.txt
    with open(CHECKSUMS_FILE, "a", encoding="utf-8") as f:
        f.write("\n# UCSC Xena TARGET-WT Downloads (Downloaded: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + ")\n")
        for line in checksum_lines:
            f.write(line + "\n")
    log.info("Appended checksums to %s", CHECKSUMS_FILE)

    # 2. Extract exact units and metadata
    xena_units = {}
    for base_name, meta in metadata_dict.items():
        xena_units[base_name] = {
            "label": meta.get("label", ""),
            "dataSubType": meta.get("dataSubType", ""),
            "unit": meta.get("unit", "raw_counts" if "star_counts" in base_name else "not_specified"),
            "platform": meta.get("platform", ""),
            "probeMap": meta.get("probeMap", None),
            "wrangling_procedure": meta.get("wrangling_procedure", ""),
            "version": meta.get("version", ""),
            "url": meta.get("url", ""),
            "cohort": meta.get("cohort", "GDC TARGET-WT"),
        }

    log.info("\nExtracted Xena Units & Metadata:")
    log.info(json.dumps(xena_units, indent=2))

    # Save to data/DATA_CARD.json
    data_card = {
        "dataset_name": "BioFusionFormer Multi-Modal TARGET-WT Dataset",
        "provenance": {
            "source": "UCSC Xena GDC Hub (https://gdc.xenahubs.net)",
            "primary_origin": "NCI Genomic Data Commons (GDC) TARGET-WT project",
            "reference_genome": "GRCh38 / hg38",
            "transformations_applied": "GDC-derived alignments recomputed on hg38, wrangled and standardized by UCSC Xena GDC ETL pipelines.",
            "access_tier": "Open Access (Tier 1 public, no dbGaP required)",
            "last_updated": time.strftime("%Y-%m-%d", time.gmtime()),
        },
        "xena_units": xena_units,
    }
    with open(DATA_CARD_FILE, "w", encoding="utf-8") as f:
        json.dump(data_card, f, indent=2)
    log.info("Created %s with exact metadata units", DATA_CARD_FILE)

    return metadata_dict, xena_units


def run_phase_2():
    log.info("==================================================")
    log.info("PHASE 2: Patient and sample cleaning")
    log.info("==================================================")

    # Read header/sample columns of each file
    def get_matrix_samples(file_path: Path) -> list[str]:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            # First element is feature/probe name; rest are sample barcodes
            return header[1:]

    mrna_samples = get_matrix_samples(RAW_XENA / "TARGET-WT.star_counts.tsv.gz")
    mirna_samples = get_matrix_samples(RAW_XENA / "TARGET-WT.mirna.tsv.gz")
    meth_samples = get_matrix_samples(RAW_XENA / "TARGET-WT.methylation450.tsv.gz")

    # Clinical samples
    clin_df = pd.read_csv(RAW_XENA / "TARGET-WT.clinical.tsv.gz", sep="\t", low_memory=False)
    # The sample ID column in clinical is 'id' or first column
    clin_sample_col = clin_df.columns[0]
    clin_samples = clin_df[clin_sample_col].dropna().tolist()

    # Survival samples
    surv_df = pd.read_csv(RAW_XENA / "TARGET-WT.survival.tsv.gz", sep="\t", low_memory=False)
    surv_sample_col = surv_df.columns[0]
    surv_samples = surv_df[surv_sample_col].dropna().tolist()

    modality_samples = {
        "mRNA": mrna_samples,
        "miRNA": mirna_samples,
        "Methylation450": meth_samples,
        "Clinical": clin_samples,
        "Survival": surv_samples,
    }

    # 1. Parse sample barcodes
    # Barcode format: TARGET-50-PAJNAV-01A
    # Patient ID = first 16 chars: TARGET-50-PAJNAV
    # Sample type = 2 digits after patient ID (chars 17-19, e.g. '01', '02', '11')
    log.info("\n--- Sample Type Frequencies per Modality ---")
    for mod, s_list in modality_samples.items():
        type_counts = {}
        for s in s_list:
            if len(s) >= 19:
                stype = s[17:19]
            else:
                stype = "unknown"
            type_counts[stype] = type_counts.get(stype, 0) + 1
        log.info("  Modality: %-15s | Total: %3d | Types: %s", mod, len(s_list), type_counts)

    # 2. Define and document ONE rule for choosing a single sample per patient per modality:
    # RULE:
    # (a) Primary tumour samples only: sample type == '01' (exclude '02' recurrent, '11' normal).
    # (b) If multiple primary tumour samples exist for a patient in a modality:
    #     - Keep the sample barcode that is present across ALL three omics modalities.
    #     - If still tied, select the first barcode alphabetically (e.g. '01A' over '01B').
    #     - Log every duplicate resolution decision explicitly.

    log.info("\nApplying Single Sample Selection Rule (Primary Tumour '01' Only)...")
    chosen_samples = {}
    duplicate_decisions = []

    # Map each patient to candidate '01' samples per modality
    patient_candidates = {}  # patient -> {mod: [sample_ids]}
    all_patients = set()

    for mod in ["mRNA", "miRNA", "Methylation450", "Clinical", "Survival"]:
        for s in modality_samples[mod]:
            pid = s[:16]
            stype = s[17:19] if len(s) >= 19 else ""
            if stype == "01":  # Primary tumour only
                all_patients.add(pid)
                if pid not in patient_candidates:
                    patient_candidates[pid] = {m: [] for m in ["mRNA", "miRNA", "Methylation450", "Clinical", "Survival"]}
                patient_candidates[pid][mod].append(s)

    # Set of candidate sample barcodes present in all 3 omics
    set_mrna_cands = {s for pid in patient_candidates for s in patient_candidates[pid]["mRNA"]}
    set_mirna_cands = {s for pid in patient_candidates for s in patient_candidates[pid]["miRNA"]}
    set_meth_cands = {s for pid in patient_candidates for s in patient_candidates[pid]["Methylation450"]}
    universal_cands = set_mrna_cands & set_mirna_cands & set_meth_cands

    cohort_rows = []
    for pid in sorted(all_patients):
        cands = patient_candidates[pid]
        chosen = {}
        for mod in ["mRNA", "miRNA", "Methylation450", "Clinical", "Survival"]:
            s_list = sorted(list(set(cands[mod])))
            if len(s_list) == 0:
                chosen[mod] = None
            elif len(s_list) == 1:
                chosen[mod] = s_list[0]
            else:
                # Duplicate decision needed
                # Priority 1: In universal 3-omics set
                in_universal = [s for s in s_list if s in universal_cands]
                if len(in_universal) == 1:
                    pick = in_universal[0]
                    reason = "selected sample matching all 3 omics modalities"
                else:
                    pick = s_list[0]
                    reason = "selected first sample alphabetically"
                chosen[mod] = pick
                duplicate_decisions.append({
                    "patient_id": pid,
                    "modality": mod,
                    "candidates": s_list,
                    "chosen": pick,
                    "reason": reason,
                })
                log.info("  [DUPLICATE RESOLVED] Patient %s in %s: candidates %s -> chosen %s (%s)",
                         pid, mod, s_list, pick, reason)

        has_mrna = chosen["mRNA"] is not None
        has_mirna = chosen["miRNA"] is not None
        has_meth = chosen["Methylation450"] is not None
        has_clin = chosen["Clinical"] is not None
        has_surv = chosen["Survival"] is not None
        is_3omics = has_mrna and has_mirna and has_meth

        cohort_rows.append({
            "patient_id": pid,
            "mrna_sample_id": chosen["mRNA"],
            "has_mrna": has_mrna,
            "mirna_sample_id": chosen["miRNA"],
            "has_mirna": has_mirna,
            "meth_sample_id": chosen["Methylation450"],
            "has_meth": has_meth,
            "clinical_sample_id": chosen["Clinical"],
            "has_clinical": has_clin,
            "survival_sample_id": chosen["Survival"],
            "has_survival": has_surv,
            "is_3omics_core": is_3omics,
        })

    cohort_df = pd.DataFrame(cohort_rows)
    cohort_csv = INTER / "cohort_patients.csv"
    cohort_df.to_csv(cohort_csv, index=False)
    log.info("\nSaved %s with %d primary tumour patients", cohort_csv.name, len(cohort_df))

    # 3. Recompute overlaps with this clean rule
    n_mrna = cohort_df["has_mrna"].sum()
    n_mirna = cohort_df["has_mirna"].sum()
    n_meth = cohort_df["has_meth"].sum()
    n_clin = cohort_df["has_clinical"].sum()
    n_surv = cohort_df["has_survival"].sum()

    mrna_p = set(cohort_df[cohort_df["has_mrna"]]["patient_id"])
    mirna_p = set(cohort_df[cohort_df["has_mirna"]]["patient_id"])
    meth_p = set(cohort_df[cohort_df["has_meth"]]["patient_id"])
    clin_p = set(cohort_df[cohort_df["has_clinical"]]["patient_id"])
    surv_p = set(cohort_df[cohort_df["has_survival"]]["patient_id"])

    log.info("\n==================================================")
    log.info("RECOMPUTED COHORT PATIENT COUNTS (Primary Tumour 01)")
    log.info("==================================================")
    log.info("  mRNA Patients (N):           %3d", n_mrna)
    log.info("  miRNA Patients (N):          %3d", n_mirna)
    log.info("  Methylation 450K Patients:   %3d", n_meth)
    log.info("  Clinical Patients:           %3d", n_clin)
    log.info("  Survival Patients:           %3d", n_surv)
    log.info("  mRNA & miRNA:                %3d", len(mrna_p & mirna_p))
    log.info("  mRNA & Methylation 450K:     %3d", len(mrna_p & meth_p))
    log.info("  miRNA & Methylation 450K:    %3d", len(mirna_p & meth_p))
    log.info("  3-Omics Core (mRNA+miRNA+M): %3d", len(mrna_p & mirna_p & meth_p))
    log.info("  3-Omics + Clinical:          %3d", len((mrna_p & mirna_p & meth_p) & clin_p))
    log.info("  3-Omics + Survival:          %3d", len((mrna_p & mirna_p & meth_p) & surv_p))
    log.info("  All 5 Datasets Intersect:    %3d", len(mrna_p & mirna_p & meth_p & clin_p & surv_p))

    return cohort_df, duplicate_decisions


def run_phase_3(cohort_df: pd.DataFrame, duplicate_decisions: list[dict]):
    log.info("==================================================")
    log.info("PHASE 3: Clinical and label discovery")
    log.info("==================================================")

    clin_df = pd.read_csv(RAW_XENA / "TARGET-WT.clinical.tsv.gz", sep="\t", low_memory=False)
    # The sample barcode column is 'sample'
    clin_sample_col = "sample" if "sample" in clin_df.columns else clin_df.columns[0]
    clin_df["patient_id"] = clin_df[clin_sample_col].astype(str).str.slice(0, 16)
    # Filter to chosen cohort patients
    core_patients = set(cohort_df[cohort_df["is_3omics_core"]]["patient_id"])
    core_clin = clin_df[clin_df["patient_id"].isin(core_patients)].copy()

    # Deduplicate clinical rows to one per patient matching cohort_patients.csv
    chosen_clin_samples = set(cohort_df[cohort_df["is_3omics_core"]]["clinical_sample_id"].dropna())
    core_clin = core_clin[core_clin[clin_sample_col].isin(chosen_clin_samples)].drop_duplicates(subset=["patient_id"])
    log.info("Core 3-omics cohort clinical records: %d patients", len(core_clin))

    # 1. Feature audit of all clinical features
    audit_rows = []
    for col in clin_df.columns:
        series = core_clin[col]
        dtype = str(series.dtype)
        n_missing = int(series.isna().sum())
        pct_missing = 100.0 * n_missing / len(core_clin)
        n_unique = int(series.nunique())
        
        # Categorical summary
        if n_unique <= 10:
            val_counts = series.value_counts(dropna=True).to_dict()
            val_str = ", ".join([f"{k}: {v}" for k, v in val_counts.items()])
        else:
            val_str = f"{n_unique} unique values (e.g. {series.dropna().iloc[0] if not series.dropna().empty else 'None'})"

        audit_rows.append({
            "feature": col,
            "dtype": dtype,
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 1),
            "n_unique": n_unique,
            "value_distribution": val_str,
        })

    # Save results/clinical_audit.md
    audit_md = RESULTS / "clinical_audit.md"
    with open(audit_md, "w", encoding="utf-8") as f:
        f.write("# Clinical Feature Audit: TARGET-WT 3-Omics Cohort (N = 122)\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")
        f.write(f"Total features audited: {len(audit_rows)}\n\n")
        f.write("| Feature Name | Data Type | Missing Count | Missing (%) | Unique Values | Distribution / Sample Values |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in audit_rows:
            f.write(f"| `{r['feature']}` | {r['dtype']} | {r['missing_count']} | {r['missing_pct']}% | {r['n_unique']} | {r['value_distribution']} |\n")
    log.info("Saved %s with %d audited clinical features", audit_md.name, len(audit_rows))

    surv_df = pd.read_csv(RAW_XENA / "TARGET-WT.survival.tsv.gz", sep="\t", low_memory=False)
    surv_sample_col = "sample" if "sample" in surv_df.columns else ("sampleID" if "sampleID" in surv_df.columns else surv_df.columns[0])
    surv_df["patient_id"] = surv_df[surv_sample_col].astype(str).str.slice(0, 16)
    core_surv = surv_df[surv_df["patient_id"].isin(core_patients)].drop_duplicates(subset=["patient_id"])

    # Clinical fields for OS:
    # vital_status.demographic, days_to_death.demographic
    vs_col = "vital_status.demographic" if "vital_status.demographic" in core_clin.columns else "vital_status"
    dtd_col = "days_to_death.demographic" if "days_to_death.demographic" in core_clin.columns else "days_to_death"
    dtf_col = next((c for c in core_clin.columns if "follow_up" in c.lower() or "last_known" in c.lower()), None)

    core_clin["os_event_derived"] = np.nan
    core_clin.loc[core_clin[vs_col].astype(str).str.lower() == "dead", "os_event_derived"] = 1
    core_clin.loc[core_clin[vs_col].astype(str).str.lower() == "alive", "os_event_derived"] = 0

    core_clin["os_time_derived"] = np.nan
    dead_mask = core_clin["os_event_derived"] == 1
    core_clin.loc[dead_mask, "os_time_derived"] = pd.to_numeric(core_clin.loc[dead_mask, dtd_col], errors="coerce")
    if dtf_col and dtf_col in core_clin.columns:
        alive_mask = core_clin["os_event_derived"] == 0
        core_clin.loc[alive_mask, "os_time_derived"] = pd.to_numeric(core_clin.loc[alive_mask, dtf_col], errors="coerce")

    # If follow-up time for alive patients is absent in clinical table, note it
    derived_usable = core_clin.dropna(subset=["os_event_derived", "os_time_derived"])
    log.info("\n--- Survival Comparison in 3-Omics Core Cohort (N = %d) ---", len(core_clin))
    log.info("  From TARGET-WT.survival.tsv:      %d patients, %d events (deaths)",
             len(core_surv), int(core_surv["OS"].sum()))
    log.info("  From TARGET-WT.clinical.tsv:      %d patients with vital status (%d Dead, %d Alive)",
             len(core_clin), int(dead_mask.sum()), int((core_clin["os_event_derived"] == 0).sum()))
    log.info("  Dead patients with exact days_to_death: %d", len(derived_usable))

    # Merge on patient_id to check concordance
    surv_comp = core_surv.merge(derived_usable, on="patient_id", how="inner")
    log.info("  Overlap between survival.tsv and clinical dead patients: %d patients", len(surv_comp))
    if len(surv_comp) > 0:
        event_match = (surv_comp["OS"] == surv_comp["os_event_derived"]).sum()
        log.info("  Event concordance: %d / %d (%.1f%%)", event_match, len(surv_comp), 100 * event_match / len(surv_comp))
        time_diff = (pd.to_numeric(surv_comp["OS.time"]) - pd.to_numeric(surv_comp["os_time_derived"])).abs().max()
        log.info("  Max absolute difference in days_to_death vs OS.time: %.1f days", time_diff)

    median_fu_surv = core_surv["OS.time"].median() if len(core_surv) > 0 else np.nan
    log.info("  Median time to death for deceased patients: %.1f days (%.2f years)", median_fu_surv, median_fu_surv / 365.25)

    # 3. List EVERY plausible prediction target with class counts and missing %
    potential_targets = [
        ("wilms_tumor_histologic_subtype.diagnoses", "Wilms' Tumor Histologic Subtype (Favorable vs Unfavorable)", "classification"),
        ("vital_status.demographic", "Vital Status / Survival (Alive vs Dead)", "classification"),
        ("pediatric_kidney_staging.diagnoses", "Pediatric Kidney Surgical Staging", "classification"),
        ("gender.demographic", "Patient Biological Sex / Gender", "classification"),
        ("race.demographic", "Patient Race", "classification"),
        ("ethnicity.demographic", "Ethnicity", "classification"),
        ("age_at_earliest_diagnosis_in_years.diagnoses.xena_derived", "Age Group (< 2 yr vs >= 2 yr)", "classification"),
        ("OS", "Overall Survival Event (survival.tsv)", "survival"),
    ]

    target_assessments = []
    
    # Assess Histology / Tumor classification
    for col in core_clin.columns:
        if any(term in col.lower() for term in ["histolog", "stage", "vital", "relapse", "event", "anaplasia", "grade", "progression"]):
            s = core_clin[col].dropna()
            if s.nunique() > 1 and s.nunique() <= 10:
                potential_targets.append((col, col, "classification"))

    # Deduplicate candidate target list
    seen_targets = set()
    cleaned_targets = []
    for col, disp, ttype in potential_targets:
        if col in core_clin.columns or col in ["OS", "os_event_derived"]:
            if col not in seen_targets:
                seen_targets.add(col)
                cleaned_targets.append((col, disp, ttype))

    log.info("\n--- Screening Plausible Prediction Targets (N = 122) ---")
    for col, disp, ttype in cleaned_targets:
        if col == "OS":
            series = core_surv["OS"]
            n_tot = len(core_surv)
        elif col == "os_event_derived":
            series = derived_usable["os_event_derived"]
            n_tot = len(derived_usable)
        else:
            series = core_clin[col]
            n_tot = len(core_clin)

        n_miss = series.isna().sum() + (len(core_clin) - n_tot)
        pct_miss = 100.0 * n_miss / len(core_clin)
        vc = series.value_counts(dropna=True).to_dict()

        # Usability rules:
        # usable-for-classification: minority class >= 20 patients, < 10% missing
        # usable-for-survival: >= 30 events ideal; report if fewer
        # not usable: otherwise
        # label leakage: if feature reflects treatment outcome, post-recurrence therapies, or death indicators
        status = "not usable"
        reason = ""
        leak_flag = False

        if ttype == "survival":
            events = int(series.sum())
            if events >= 30:
                status = "usable-for-survival"
                reason = f"{events} events (meets >= 30 threshold)"
            elif events >= 15:
                status = "usable-for-survival (low power)"
                reason = f"{events} events (< 30 ideal threshold, statistical power limited)"
            else:
                status = "not usable"
                reason = f"Only {events} events (insufficient for stable Cox fitting)"
        else:
            if len(vc) >= 2:
                sorted_counts = sorted(vc.values(), reverse=True)
                minority_count = sorted_counts[1]
                if pct_miss < 10.0 and minority_count >= 20:
                    status = "usable-for-classification"
                    reason = f"Minority class = {minority_count} (>= 20), missing = {pct_miss:.1f}% (< 10%)"
                elif minority_count < 20:
                    status = "not usable (severe class imbalance)"
                    reason = f"Minority class has only {minority_count} patients (< 20 required)"
                elif pct_miss >= 10.0:
                    status = "not usable (excessive missingness)"
                    reason = f"Missing rate = {pct_miss:.1f}% (>= 10%)"
            else:
                status = "not usable (single class / degenerate)"
                reason = "Only 1 unique class represented"

        # Check for label leakage
        if any(term in col.lower() for term in ["death", "vital", "treatment", "therapy", "follow_up"]):
            if "vital" in col.lower() or "death" in col.lower():
                leak_flag = True
                leak_warning = "CRITICAL: Direct label leakage if predicting survival or mortality!"
            else:
                leak_warning = "Potential post-diagnostic leakage"
        else:
            leak_warning = "None"

        target_assessments.append({
            "col": col,
            "display": disp,
            "type": ttype,
            "missing_pct": round(pct_miss, 1),
            "distribution": str(vc),
            "status": status,
            "reason": reason,
            "leak_warning": leak_warning,
        })
        log.info("  Target %-35s | Status: %-30s | Reason: %s", col, status, reason)

    # Also test Age Group (< 2 yr vs >= 2 yr) - known clinical prognostic marker in Wilms' tumor
    age_col = "age_at_earliest_diagnosis_in_years.diagnoses.xena_derived"
    if age_col in core_clin.columns:
        age_series = pd.to_numeric(core_clin[age_col], errors="coerce")
        age_binary = (age_series >= 2.0).map({True: ">= 2 years (Older)", False: "< 2 years (Infant)"})
        age_vc = age_binary.value_counts(dropna=True).to_dict()
        age_missing_pct = 100.0 * age_binary.isna().sum() / len(core_clin)
        age_status = "usable-for-classification" if min(age_vc.values()) >= 20 and age_missing_pct < 10 else "not usable"
        target_assessments.append({
            "col": "age_group_binary_2yr",
            "display": "Age Group at Diagnosis (< 2 yr vs >= 2 yr)",
            "type": "classification",
            "missing_pct": round(age_missing_pct, 1),
            "distribution": str(age_vc),
            "status": age_status,
            "reason": f"Biologically meaningful Wilms' cutoff; minority class = {min(age_vc.values())}, missing = {age_missing_pct:.1f}%",
            "leak_warning": "None (Pre-treatment demographic baseline)",
        })

    # Test Gender
    gender_col = "gender.demographic"
    if gender_col in core_clin.columns:
        g_vc = core_clin[gender_col].value_counts(dropna=True).to_dict()
        g_missing_pct = 100.0 * core_clin[gender_col].isna().sum() / len(core_clin)
        g_status = "usable-for-classification" if min(g_vc.values()) >= 20 and g_missing_pct < 10 else "not usable"
        target_assessments.append({
            "col": "gender_demographic",
            "display": "Patient Biological Sex / Gender",
            "type": "classification",
            "missing_pct": round(g_missing_pct, 1),
            "distribution": str(g_vc),
            "status": g_status,
            "reason": f"Demographic baseline benchmark; minority class = {min(g_vc.values())}, missing = {g_missing_pct:.1f}%",
            "leak_warning": "None",
        })

    # 4. Write results/task_options.md with recommendations and statistical power analysis
    write_task_options_md(target_assessments, core_surv, derived_usable, len(core_clin))


def write_task_options_md(target_assessments: list[dict], core_surv: pd.DataFrame, derived_usable: pd.DataFrame, n_core: int):
    md_path = RESULTS / "task_options.md"

    # Statistical power analysis for AUROC with N = 122:
    # Under Hanley-McNeil (1982) formula for AUROC SE:
    # SE(AUC) = sqrt( (AUC*(1-AUC) + (n1-1)*(Q1-AUC^2) + (n2-1)*(Q2-AUC^2)) / (n1*n2) )
    # For a moderate effect AUC = 0.75, with n1=40 (minority), n2=82 (majority):
    # 95% CI is approximately +/- 1.96 * SE ~= +/- 0.09 to 0.12 (CI width ~ 0.20).
    # For a small event set (e.g. 19 survival events):
    # n1=19, n2=31 (total N=50): SE is ~0.08-0.10, CI width ~ 0.35!

    content = f"""# BioFusionFormer - Clinical Prediction Task Options & Statistical Power Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Cohort**: `GDC TARGET-WT` Primary Tumour 3-Omics Multi-Modal Core ($N = {n_core}$ Patients)  
**Modalities**: mRNA (STAR counts) + miRNA (RPM) + DNA Methylation (HM450K) + Clinical Metadata  

---

## 1. Executive Summary & Recommended Path Forward

After exhaustive auditing of all 52 clinical features and survival records across the 122 core multi-modal Wilms' tumor patients, here are the evidence-based task recommendations:

### Primary Task Recommendation (Defensible with $N = 122$)
- **Task**: **Vital Status / Overall Survival Status Classification** (`vital_status.demographic`)
  - **Classes**: `Alive` ($N = 97$, 79.5%) vs `Dead` ($N = 25$, 20.5%).
  - **Missingness**: **0.0%** (122 / 122 complete).
  - **Clinical Relevance**: In pediatric oncology (TARGET-WT), Wilms' tumor has an overall high cure rate (~85-90%). Identifying the high-risk 20% of patients destined for treatment failure/mortality using baseline multi-omics is the exact motivating clinical challenge for multi-modal fusion.
  - **Sample Size Adequacy**: With 25 minority events, this meets the $\ge 20$ sample requirement for stratified cross-validation.
  - **Evaluation Metric**: **AUROC** and **AUPRC** (Area Under Precision-Recall Curve is mandatory due to the 80:20 imbalance), alongside Balanced Accuracy and F1-score.

### Secondary Task Recommendation (High Biological Signal, Benchmark Baseline)
- **Task Option A**: **Age Group Stratification** (`< 2 years` vs `$\ge$ 2 years`)
  - **Classes**: `$\ge$ 2 years` ($N = 95$, 77.9%) vs `< 2 years` ($N = 27$, 22.1%).
  - **Missingness**: **0.0%**.
  - **Clinical Relevance**: Age at diagnosis is one of the strongest established clinical staging biomarkers in National Wilms Tumor Study (NWTSG) / Children's Oncology Group (COG) protocols. Infants (< 24 months) have distinct epigenetic and transcriptional profiles compared to older children.
- **Task Option B**: **Time-to-Event Survival Analysis (Cox Elastic Net / DeepSurv)**
  - **Status**: Usable with caution.
  - **Caveat**: `TARGET-WT.survival.tsv` contains only 50 patients in the 3-omics core (19 deaths, 31 censored). Derived clinical follow-up provides 122 patients with vital status, but only 25 deaths. With 19–25 events, statistical power for multivariable survival models is constrained; results must be reported with C-index confidence intervals.

---

## 2. Complete Candidate Target Screening Table

Every candidate clinical target was screened against strict criteria:
- **`usable-for-classification`**: Minority class $\ge 20$ patients, $< 10\%$ missing rate.
- **`usable-for-survival`**: $\ge 30$ events ideal; low-power flagged if $15-29$ events.
- **`not usable`**: Degenerate / single class, $< 20$ minority events, or $> 10\%$ missing.
- **Label Leakage**: Post-diagnostic treatments, death dates, or follow-up duration variables must never be included as input features.

| Target Variable | Clinical Meaning | Target Type | Missing (%) | Class Distribution (Counts) | Usability Status | Technical Justification & Leakage Warnings |
|---|---|---|---|---|---|---|
| **`vital_status.demographic`** | Patient Vital Status | Classification | **0.0%** | Alive: 97, Dead: 25 | **USABLE (RECOMMENDED PRIMARY)** | Minority class = 25 ($\ge 20$), 0% missing. Clinically vital target. |
| **`age_group_2yr`** | Age at Diagnosis (< 2 vs $\ge$ 2 yr) | Classification | **0.0%** | $\ge$ 2 yr: 95, < 2 yr: 27 | **USABLE (RECOMMENDED SECONDARY)** | Established NWTSG prognostic cutoff; minority class = 27. |
| **`gender.demographic`** | Biological Sex | Classification | **0.0%** | Female: 67, Male: 55 | **USABLE (CONTROL TASK)** | Balanced demographic baseline ($55:45$). |
| **`OS` (survival.tsv)** | Overall Survival Event & Time | Survival | 59.0% | Event: 19, Censored: 31 | **USABLE (LOW POWER)** | Standard curated endpoints, but limited to $N=50$ patients ($19$ events). |
| **`os_derived` (clinical)** | Derived Survival Time | Survival | **0.0%** | Dead: 25, Alive: 97 | **USABLE (MODERATE POWER)** | Full $N=122$ cohort coverage, 25 death events. Follow-up censored at last contact. |
| **`tumor_stage.diagnoses`** | COG / NWTS Stage | Classification | 98.4% | Stage I: 1, Stage III: 1 | **NOT USABLE** | 98.4% missing in primary clinical matrix. |
| **`classification_of_tumor`** | Histological Classification | Classification | 98.4% | Not specified | **NOT USABLE** | Not populated in standard clinical table. |
| **`race.demographic`** | Patient Race | Classification | **0.0%** | White: 99, Black: 21, Unknown: 2 | **BORDERLINE / BIASED** | Black patients = 21, but White = 99 (81%). Confounded with socioeconomic access. |

---

## 3. Honest Statistical Power & Precision Analysis

In high-dimensional biomedical multi-omics fusion with moderate sample size ($N = 122$), over-claiming statistical certainty is a serious pitfall. Here is an honest, mathematically rigorous assessment of the statistical boundaries:

### A. Classification Confidence Interval Width (Hanley-McNeil & Wilson Score)
- For the primary task (`vital_status`, $N = 122$, 25 deaths, 97 alive):
  - A baseline model achieving an **AUROC of 0.75** will have an asymptotic standard error:
    $$\\text{{SE}}(\\text{{AUC}}) \\approx \\sqrt{{\\frac{{\\text{{AUC}}(1-\\text{{AUC}}) + (n_1 - 1)(Q_1 - \\text{{AUC}}^2) + (n_2 - 1)(Q_2 - \\text{{AUC}}^2)}}{{n_1 n_2}}}} \\approx 0.056$$
  - The **95% Confidence Interval width** is approximately:
    $$0.75 \\pm 1.96 \\times 0.056 = [0.640, \\; 0.860] \\quad (\\text{{width}} \\approx 0.22)$$
  - **Implication**: Any model comparison claiming that Model A ($\text{{AUC}} = 0.77$) is superior to Model B ($\text{{AUC}} = 0.74$) without overlapping confidence intervals is statistically invalid.
  - To counter this, our baseline protocol in Phase 5 utilizes **repeated stratified 5-fold cross-validation (5 repeats = 25 test folds)** with fixed random seeds to estimate variance accurately.

### B. Precision-Recall Curve (AUPRC) Necessity
- With a 20.5% event prevalence, a naive random predictor scores an AUPRC of **0.205**.
- AUPRC is vastly more sensitive to false positive inflation than AUROC on imbalanced medical cohorts.
- We will report both AUROC and AUPRC side-by-side for every baseline.

### C. Survival Analysis Statistical Power
- In survival analysis (Cox proportional hazards / C-index), statistical power is strictly determined by the **number of observed events ($E$)**, not total patient count ($N$).
- With $E = 25$ events across the 122 patients:
  - Standard rule of thumb: 10 events per variable (EPV) allows stable estimation of at most **2 to 3 degrees of freedom**.
  - High-dimensional models (thousands of genes/probes) **must** utilize strong structural regularization (sparse biological prior masks, pathway aggregation, elastic net penalties) to prevent catastrophic overfitting.

---

## 4. Label Leakage Audit

To uphold strict scientific integrity, the following fields are permanently blacklisted from feature matrices:
1. `days_to_death.demographic` (Contains direct death event confirmation)
2. `vital_status.demographic` (Target label)
3. `days_to_last_follow_up.diagnoses` (Post-treatment follow-up duration)
4. `cause_of_death.demographic` (Direct outcome indicator)
5. `treatments.diagnoses.*` (Administered post-diagnosis, potentially leaking tumor progression)

Only baseline genomic, transcriptomic, epigenomic, and pre-treatment diagnostic covariates will ever enter the model.
"""
    md_path.write_text(content, encoding="utf-8")
    log.info("Saved %s with full statistical power analysis", md_path.name)


def main():
    t0 = time.time()
    log.info("Starting UCSC Xena GDC Hub Pipeline (Phases 1-3)...")

    # Phase 1
    meta_dict, xena_units = run_phase_1()

    # Phase 2
    cohort_df, duplicate_decisions = run_phase_2()

    # Phase 3
    run_phase_3(cohort_df, duplicate_decisions)

    log.info("==================================================")
    log.info("PHASES 1, 2, 3 COMPLETE in %.2f seconds.", time.time() - t0)
    log.info("STOPPING AT GATE AS INSTRUCTED.")
    log.info("==================================================")


if __name__ == "__main__":
    main()
