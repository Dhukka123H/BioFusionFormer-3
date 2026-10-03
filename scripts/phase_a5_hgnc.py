"""
Phase A5 - HGNC gene symbol <-> Ensembl ID mapping
=====================================================
Downloads the HGNC complete gene set from the HGNC FTP / REST API.
Primary  : https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt
Fallback : https://g-a-huttenhower.sph.harvard.edu/hgnc/ (if primary fails)

Saves to : data/raw/hgnc/hgnc_complete_set.txt
Builds   : data/interim/hgnc_symbol2ensg.parquet
           (columns: hgnc_id, symbol, name, locus_type, ensembl_gene_id)
"""
import sys
import logging
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw" / "hgnc"
OUT  = ROOT / "data" / "interim"
LOG  = ROOT / "logs" / "phase_a5.log"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("A5")

UA = "Mozilla/5.0 (compatible; BioFusionFormer/1.0; academic)"

HGNC_URLS = [
    "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt",
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
]
HGNC_DEST = RAW / "hgnc_complete_set.txt"


def download(url: str, dest: Path, retries: int = 4, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 100_000:
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


def main():
    # ── Download ──────────────────────────────────────────────────────────────
    ok = False
    for url in HGNC_URLS:
        if download(url, HGNC_DEST):
            ok = True
            break

    if not ok:
        log.error("All HGNC URLs failed.")
        print("\n=== A5 RESULTS ===")
        print("  FAILED: HGNC not downloadable from any URL.")
        return

    # ── Parse ─────────────────────────────────────────────────────────────────
    log.info("Parsing HGNC file ...")
    df = pd.read_csv(HGNC_DEST, sep="\t", low_memory=False)
    log.info("  HGNC raw rows: %d  cols: %d", len(df), len(df.columns))

    # Select relevant columns (some may be absent in older dumps)
    keep_cols = ["hgnc_id", "symbol", "name", "locus_type", "ensembl_gene_id",
                 "entrez_id", "refseq_accession"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Filter to rows with ENSG
    has_ensg = df["ensembl_gene_id"].notna() if "ensembl_gene_id" in df.columns else \
        pd.Series([False] * len(df))
    df_ensg = df[has_ensg].copy()

    # Remove version suffix from ENSG if present
    if "ensembl_gene_id" in df_ensg.columns:
        df_ensg["ensembl_gene_id"] = (df_ensg["ensembl_gene_id"]
                                      .astype(str).str.split(".").str[0])

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUT / "hgnc_symbol2ensg.parquet"
    df_ensg.to_parquet(out_path, index=False)
    log.info("Saved: %s (%d rows)", out_path, len(df_ensg))

    print("\n=== A5 RESULTS ===")
    print(f"  Total HGNC entries     : {len(df):,}")
    print(f"  With Ensembl gene ID   : {len(df_ensg):,}")
    print(f"  Unique symbols         : {df_ensg['symbol'].nunique():,}")
    print(f"  Unique ENSG IDs        : {df_ensg['ensembl_gene_id'].nunique():,}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
