"""
Phase A4 - Illumina HumanMethylation450 probe annotation
===========================================================
Downloads the official Illumina HM450 manifest CSV from the Illumina website
or the GEO/NCBI platform annotation page, and builds
data/interim/cpg2gene.csv with columns:
  probe_id, chromosome, position, gene_symbol, ucsc_refgene_group, feature_type

Primary source  : Illumina iGenomes / product files
Fallback source : https://webdata.illumina.com/downloads/productfiles/
                  humanmethylation450/
Last resort     : GEO GPL13534 annotation table
"""
import sys
import logging
import csv
import gzip
import zipfile
from io import TextIOWrapper
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "data" / "interim"
LOG  = ROOT / "logs" / "phase_a4.log"

ANNOT_DIR = RAW / "hm450_annotation"
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("A4")

UA = "Mozilla/5.0 (compatible; BioFusionFormer/1.0; academic)"

# ── Download candidates ──────────────────────────────────────────────────────
# Illumina hosts the manifest as a zipped CSV.
# The canonical filename is HumanMethylation450_15017482_v1-2.csv
# (inside a zip). URLs verified manually 2026-10.

ILLUMINA_URLS = [
    # Illumina official direct CSV manifest (verified active)
    "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv",
    # GEO soft GPL platform annotation fallback
    "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/annot/GPL13534.annot.gz",
]
DEST_CSV  = ANNOT_DIR / "humanmethylation450_15017482_v1-2.csv"
DEST_GZ   = ANNOT_DIR / "GPL13534.annot.gz"


def download(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 10000:
        log.info("[SKIP] %s already present (%d bytes)", dest.name, dest.stat().st_size)
        return True
    for attempt in range(1, retries + 1):
        try:
            log.info("[GET attempt %d] %s", attempt, url)
            r = requests.get(url, headers={"User-Agent": UA}, stream=True,
                             timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(65536):
                    if chunk:
                        fh.write(chunk)
            log.info("  Saved: %s (%d bytes)", dest.name, dest.stat().st_size)
            return True
        except Exception as exc:
            log.warning("  Attempt %d failed: %s", attempt, exc)
    return False


def parse_geo_gz(path: Path) -> pd.DataFrame | None:
    """Parse GEO GPL13534 annotation (.annot.gz).
    Format: tab-separated, header starts after comment lines beginning with #.
    Relevant columns: ID, Chromosome_36, Coordinate_36 (or hg19 equivalents),
    UCSC_RefGene_Name, UCSC_RefGene_Group.
    """
    log.info("Parsing GEO annotation: %s", path.name)
    try:
        rows = []
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            # Skip header comment block
            header_line = None
            for line in fh:
                if not line.startswith("#") and line.strip():
                    header_line = line.rstrip("\n").split("\t")
                    break
            if header_line is None:
                log.error("Could not find header in GEO file")
                return None

            log.info("  Columns: %s", header_line[:10])
            reader = csv.DictReader(fh, fieldnames=header_line, delimiter="\t")
            for row in reader:
                rows.append(row)

        df = pd.DataFrame(rows)
        log.info("  GEO rows: %d, cols: %d", len(df), len(df.columns))
        return df
    except Exception as exc:
        log.error("Failed parsing GEO gz: %s", exc)
        return None


def parse_illumina_csv(path: Path) -> pd.DataFrame | None:
    """Parse Illumina HM450 manifest from uncompressed CSV."""
    log.info("Parsing Illumina manifest CSV: %s", path.name)
    try:
        skip = 0
        with open(path, "r", encoding="latin-1") as fh:
            for i, line in enumerate(fh):
                if line.startswith("IlmnID"):
                    skip = i
                    break
        log.info("  Table header starts at line %d", skip)
        df = pd.read_csv(path, skiprows=skip, low_memory=False, encoding="latin-1")
        if "IlmnID" in df.columns:
            ctrl_idx = df[df["IlmnID"].astype(str).str.startswith("[")].index
            if len(ctrl_idx) > 0:
                df = df.iloc[:ctrl_idx[0]]
        log.info("  Illumina manifest rows: %d", len(df))
        return df
    except Exception as exc:
        log.error("Failed parsing Illumina CSV: %s", exc)
        return None


def standardize_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map raw columns to a standard schema."""
    col_lower = {c: c.lower() for c in df.columns}

    def find(candidates: list) -> str | None:
        for cand in candidates:
            for raw, low in col_lower.items():
                if cand in low:
                    return raw
        return None

    probe_col  = find(["ilmnid", "id", "name"])
    chr_col    = find(["chr", "chromosome"])
    pos_col    = find(["mapinfo", "coordinate", "position", "start"])
    gene_col   = find(["refgene_name", "ucsc_refgene_name", "gene_name", "symbol"])
    group_col  = find(["refgene_group", "ucsc_refgene_group", "feature"])

    log.info("Source=%s: probe=%s chr=%s pos=%s gene=%s group=%s",
             source, probe_col, chr_col, pos_col, gene_col, group_col)

    out = pd.DataFrame()
    if probe_col:
        out["probe_id"] = df[probe_col]
    if chr_col:
        out["chromosome"] = df[chr_col]
    if pos_col:
        out["position"] = pd.to_numeric(df[pos_col], errors="coerce")
    if gene_col:
        out["gene_symbol"] = df[gene_col]
    if group_col:
        out["ucsc_refgene_group"] = df[group_col]
    out["source"] = source

    # Keep only rows with a valid probe_id
    if "probe_id" in out.columns:
        out = out[out["probe_id"].notna() & (out["probe_id"] != "")]

    return out


def main():
    result_df = None

    # ── 1. Try Illumina official direct CSV ──────────────────────────────────
    ok_ill = download(ILLUMINA_URLS[0], DEST_CSV)
    if ok_ill:
        raw_df = parse_illumina_csv(DEST_CSV)
        if raw_df is not None and len(raw_df) > 1000:
            result_df = standardize_df(raw_df, "Illumina_official")
            log.info("Using Illumina official source: %d probes", len(result_df))

    # ── 2. Fallback: GEO annotation ──────────────────────────────────────────
    if result_df is None or len(result_df) < 1000:
        ok_geo = download(ILLUMINA_URLS[1], DEST_GZ)
        if ok_geo:
            raw_df2 = parse_geo_gz(DEST_GZ)
            if raw_df2 is not None and len(raw_df2) > 1000:
                result_df = standardize_df(raw_df2, "GEO_GPL13534")
                log.info("Using GEO source: %d probes", len(result_df))

    if result_df is None or len(result_df) < 1000:
        log.error("Could not retrieve HM450 annotation from any source.")
        print("\n=== A4 RESULTS ===")
        print("  FAILED: HM450 annotation not retrievable.")
        print("  Tried:")
        for u in ILLUMINA_URLS:
            print(f"    {u}")
        print("  Continuing without cpg2gene.csv.")
        return

    # ── Filter to cg probes ────────────────────────────────────────────────
    if "probe_id" in result_df.columns:
        result_df = result_df[result_df["probe_id"].str.startswith("cg", na=False)]

    out_path = OUT / "cpg2gene.csv"
    result_df.to_csv(out_path, index=False)
    log.info("Saved cpg2gene.csv: %d rows", len(result_df))

    n_probes = result_df["probe_id"].nunique() if "probe_id" in result_df.columns else 0
    n_genes  = (result_df["gene_symbol"].str.split(";").explode()
                .str.strip().dropna().nunique()
                if "gene_symbol" in result_df.columns else 0)

    print("\n=== A4 RESULTS ===")
    print(f"  CpG probes (cg*)   : {n_probes:,}")
    print(f"  Unique gene symbols: {n_genes:,}")
    print(f"  Source             : {result_df['source'].iloc[0]}")
    print(f"  Output             : {out_path}")


if __name__ == "__main__":
    main()
