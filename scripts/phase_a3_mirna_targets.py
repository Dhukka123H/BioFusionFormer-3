"""
Phase A3 - miRNA target mapping
=================================
Downloads:
  1. miRTarBase validated human miRNA-target interactions
     Primary: https://mirtarbase.cuhk.edu.cn/ (Excel)
     Fallback: https://mirtarbase.bioinfo.tsgh.org.tw/ 
     Last-resort: cached mirror at https://zenodo.org (if available)
  2. miRBase mature miRNA list for hsa- (human)
     From: https://www.mirbase.org/ftp/CURRENT/

Builds data/interim/mirna2pathway.csv with columns:
  mirna_id, target_ensg, pathway_id, pathway_name, support_type
"""
import sys
import logging
import re
import gzip
from pathlib import Path
import requests
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
MIRNA_D = ROOT / "data" / "raw" / "mirna_targets"
INTER   = ROOT / "data" / "interim"
REACT   = ROOT / "data" / "raw" / "reactome"
LOG     = ROOT / "logs" / "phase_a3.log"
MIRNA_D.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("A3")

UA = "Mozilla/5.0 (compatible; BioFusionFormer/1.0; academic)"

# ─── miRTarBase ───────────────────────────────────────────────────────────────
MIRTARBASE_URLS = [
    # Official updated miRTarBase 10.0 human interactions
    "https://awi.cuhk.edu.cn/miRTarBase/downloads/files/10.0/hsa_MTI.csv",
]
MIRTARBASE_DEST = MIRNA_D / "hsa_MTI.csv"

# ─── miRBase ──────────────────────────────────────────────────────────────────
MIRBASE_URL  = "https://mirbase.org/download/mature.fa"
MIRBASE_DEST = MIRNA_D / "mature.fa"

# ─── HGNC symbol→ENSG (built in A5 but we load it if present) ────────────────
HGNC_PATH = ROOT / "data" / "raw" / "hgnc" / "hgnc_complete_set.txt"


def download(url: str, dest: Path, retries: int = 4, timeout: int = 900) -> bool:
    if dest.exists() and dest.stat().st_size > 5000:
        log.info("[SKIP] %s already present (%d bytes)", dest.name, dest.stat().st_size)
        return True
    import subprocess
    import ssl
    import urllib.request
    import shutil
    import gzip

    for attempt in range(1, retries + 1):
        try:
            log.info("[GET attempt %d via curl --compressed] %s", attempt, url)
            res = subprocess.run(["curl.exe", "-s", "-k", "-L", "--compressed", "-o", str(dest), url], timeout=timeout)
            if res.returncode == 0 and dest.exists() and dest.stat().st_size > 5000:
                log.info("  Saved via curl: %s (%d bytes)", dest.name, dest.stat().st_size)
                return True
        except Exception as c_exc:
            log.warning("  Curl attempt %d failed: %s", attempt, c_exc)

        # Fallback to urllib with gzip decompression
        try:
            log.info("  Trying urllib fallback...")
            ctx = ssl.create_default_context()
            try:
                ctx.set_ciphers("DEFAULT@SECLEVEL=1")
            except Exception:
                pass
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
                if response.info().get("Content-Encoding") == "gzip":
                    with gzip.GzipFile(fileobj=response) as gz_in:
                        with open(dest, "wb") as fh:
                            shutil.copyfileobj(gz_in, fh)
                else:
                    with open(dest, "wb") as fh:
                        shutil.copyfileobj(response, fh)
            if dest.exists() and dest.stat().st_size > 5000:
                log.info("  Saved via urllib: %s (%d bytes)", dest.name, dest.stat().st_size)
                return True
        except Exception as exc:
            log.warning("  Urllib fallback failed: %s", exc)
    return False


def fetch_mirtarbase() -> bool:
    for url in MIRTARBASE_URLS:
        if download(url, MIRTARBASE_DEST):
            return True
    log.error("All miRTarBase URLs failed.")
    return False


def load_mirtarbase() -> pd.DataFrame | None:
    if not MIRTARBASE_DEST.exists():
        log.error("miRTarBase file not found: %s", MIRTARBASE_DEST)
        return None
    log.info("Reading miRTarBase CSV ...")
    try:
        df = pd.read_csv(MIRTARBASE_DEST, low_memory=False)
        log.info("  miRTarBase rows: %d  cols: %s", len(df), df.columns.tolist())
        return df
    except Exception as exc:
        log.error("Failed to read CSV: %s", exc)
        return None


def build_symbol2ensg() -> dict[str, str]:
    """Return {HGNC_symbol: ENSG_id} from HGNC file if available."""
    if not HGNC_PATH.exists():
        log.warning("HGNC file not found — symbol->ENSG mapping unavailable; "
                    "will skip symbol conversion. Run A5 first for full coverage.")
        return {}
    log.info("Loading HGNC symbol->ENSG map ...")
    df = pd.read_csv(HGNC_PATH, sep="\t", low_memory=False,
                     usecols=["symbol", "ensembl_gene_id"])
    df = df.dropna(subset=["ensembl_gene_id"])
    mapping = dict(zip(df["symbol"].str.strip(), df["ensembl_gene_id"].str.strip()))
    log.info("  HGNC entries with ENSG: %d", len(mapping))
    return mapping


def load_reactome_gene_pathway() -> pd.DataFrame:
    gp_path = INTER / "reactome" / "gene2pathway.parquet"
    if not gp_path.exists():
        log.error("gene2pathway.parquet missing — run A1 first")
        return pd.DataFrame()
    df = pd.read_parquet(gp_path)
    # keep all-levels for maximum pathway coverage
    return df[df["level"] == "all_levels"][["gene_id", "pathway_id", "pathway_name"]]


def main():
    # ── 1. Download miRTarBase ────────────────────────────────────────────────
    ok_mtb = fetch_mirtarbase()
    if not ok_mtb:
        log.error("CRITICAL: miRTarBase unavailable. Recording failure and continuing.")

    # ── 2. Download miRBase mature sequences ─────────────────────────────────
    ok_mb = download(MIRBASE_URL, MIRBASE_DEST)

    # ── 3. Extract human (hsa-) miRNA names from miRBase ─────────────────────
    hsa_mirnas = set()
    if ok_mb and MIRBASE_DEST.exists():
        log.info("Parsing miRBase mature.fa for hsa- entries ...")
        open_fn = gzip.open if str(MIRBASE_DEST).endswith(".gz") else open
        with open_fn(MIRBASE_DEST, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(">hsa-"):
                    name = line[1:].split()[0]
                    hsa_mirnas.add(name)
        log.info("  Human miRNAs in miRBase: %d", len(hsa_mirnas))
        # Save list
        mirbase_out = MIRNA_D / "hsa_mirbase_names.txt"
        mirbase_out.write_text("\n".join(sorted(hsa_mirnas)), encoding="utf-8")
        log.info("  Saved: %s", mirbase_out)

    # ── 4. Build symbol->ENSG map ─────────────────────────────────────────────
    sym2ensg = build_symbol2ensg()

    # ── 5. Load Reactome gene-pathway ─────────────────────────────────────────
    gp_df = load_reactome_gene_pathway()
    if gp_df.empty:
        log.error("No gene-pathway data available; A3 partial output only.")
        print("\n=== A3 RESULTS (PARTIAL - run A1 first) ===")
        return

    ensg2pathways = gp_df.groupby("gene_id")[["pathway_id", "pathway_name"]].apply(
        lambda x: x.values.tolist()
    ).to_dict()

    # ── 6. Build mirna->pathway via miRTarBase ────────────────────────────────
    records = []
    mirtarbase_ok_flag = False

    if ok_mtb:
        mtb_df = load_mirtarbase()
        if mtb_df is not None:
            mirtarbase_ok_flag = True
            # Normalize column names (differ across miRTarBase versions)
            col_map = {}
            for col in mtb_df.columns:
                lc = col.lower().replace(" ", "_")
                col_map[col] = lc
            mtb_df = mtb_df.rename(columns=col_map)

            # Find relevant columns
            mirna_col    = next((c for c in mtb_df.columns if "mirna" in c), None)
            gene_col     = next((c for c in mtb_df.columns if "target_gene" in c or
                                 "target gene" in c.lower()), None)
            support_col  = next((c for c in mtb_df.columns if "support" in c), None)
            species_col  = next((c for c in mtb_df.columns if "species" in c and
                                 "mirna" not in c), None)

            log.info("Columns used: mirna=%s, gene=%s, support=%s, species=%s",
                     mirna_col, gene_col, support_col, species_col)

            if mirna_col and gene_col:
                # Filter to human targets only (values are 'hsa' or 'Homo sapiens')
                if species_col:
                    human_mtb = mtb_df[mtb_df[species_col].astype(str).str.lower().str.contains("hsa|homo sapiens", na=False)].copy()
                else:
                    human_mtb = mtb_df.copy()

                log.info("  Human miRTarBase raw interactions: %d", len(human_mtb))

                # Standardize columns
                human_mtb["mirna_id"] = human_mtb[mirna_col].astype(str).str.strip()
                human_mtb["target_symbol"] = human_mtb[gene_col].astype(str).str.strip()
                human_mtb["support_type"] = (human_mtb[support_col].astype(str).str.slice(0, 120)
                                             if support_col else "")

                # Exclude explicitly non-functional interactions
                if support_col:
                    human_mtb = human_mtb[~human_mtb["support_type"].str.lower().str.contains("non-functional", na=False)]
                    log.info("  Functional interactions: %d", len(human_mtb))

                # Map gene symbol -> ENSG
                human_mtb["target_ensg"] = human_mtb["target_symbol"].map(sym2ensg)
                # Strip any trailing transcript/version suffix for unmapped
                missing = human_mtb["target_ensg"].isna()
                if missing.any():
                    human_mtb.loc[missing, "target_ensg"] = (
                        human_mtb.loc[missing, "target_symbol"].str.split(".").str[0].map(sym2ensg)
                    )

                valid_targets = human_mtb.dropna(subset=["target_ensg"])[
                    ["mirna_id", "target_symbol", "target_ensg", "support_type"]
                ].drop_duplicates()

                log.info("  Unique (miRNA, target_ensg) pairs: %d", len(valid_targets))

                # Fast vectorized merge with Reactome gene2pathway
                # gp_df columns: gene_id, pathway_id, pathway_name
                merged = valid_targets.merge(gp_df, left_on="target_ensg", right_on="gene_id", how="inner")
                result_df = merged[[
                    "mirna_id", "target_symbol", "target_ensg", "pathway_id", "pathway_name", "support_type"
                ]].drop_duplicates()

                out_path = INTER / "mirna2pathway.csv"
                result_df.to_csv(out_path, index=False)
                log.info("Saved mirna2pathway.csv: %d rows", len(result_df))

                n_mirnas   = result_df["mirna_id"].nunique()
                n_targets  = result_df["target_ensg"].nunique()
                n_pathways = result_df["pathway_id"].nunique()

                print("\n=== A3 RESULTS ===")
                print(f"  miRBase human miRNAs (hsa-)   : {len(hsa_mirnas):,}")
                print(f"  miRTarBase available           : {mirtarbase_ok_flag}")
                print(f"  miRNAs with pathway coverage   : {n_mirnas:,}")
                print(f"  Unique target ENSG IDs         : {n_targets:,}")
                print(f"  Unique pathways covered        : {n_pathways:,}")
                print(f"  Total mirna->pathway rows      : {len(result_df):,}")
                print(f"  Output: {out_path}")
                return

    # Fallback if no records built
    out_path = INTER / "mirna2pathway.csv"
    pd.DataFrame(columns=["mirna_id", "target_symbol", "target_ensg",
                           "pathway_id", "pathway_name", "support_type"]
                ).to_csv(out_path, index=False)
    print("\n=== A3 RESULTS (NO RECORDS) ===")
    print("  Check logs for details.")
    if hsa_mirnas:
        print(f"  miRBase hsa- miRNAs: {len(hsa_mirnas):,}")


if __name__ == "__main__":
    main()
