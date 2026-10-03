"""
Phase A1 - Reactome gene filtering
===================================
Reads Ensembl2Reactome.txt and Ensembl2Reactome_All_Levels.txt,
keeps ONLY human ENSG IDs (prefix ENSG, not ENST/ENSP),
and writes data/interim/reactome/gene2pathway.parquet.

Columns: gene_id, pathway_id, pathway_name, species, level
level: 'direct' | 'all_levels'
"""
import sys
import logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw" / "reactome"
OUT  = ROOT / "data" / "interim" / "reactome"
OUT.mkdir(parents=True, exist_ok=True)
LOG  = ROOT / "logs" / "phase_a1.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("A1")

COLS = ["gene_id", "pathway_id", "url", "pathway_name", "evidence", "species"]

def load_e2r(path: Path, label: str) -> pd.DataFrame:
    log.info("Loading %s ...", path.name)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            rows.append(parts[:6])
    df = pd.DataFrame(rows, columns=COLS)
    df["level"] = label
    log.info("  Raw rows: %d", len(df))
    return df

def main():
    direct = load_e2r(RAW / "Ensembl2Reactome.txt", "direct")
    all_lv = load_e2r(RAW / "Ensembl2Reactome_All_Levels.txt", "all_levels")

    combined = pd.concat([direct, all_lv], ignore_index=True)
    log.info("Combined rows: %d", len(combined))

    # Keep Homo sapiens only
    human = combined[combined["species"] == "Homo sapiens"].copy()
    log.info("Human rows: %d", len(human))

    # Keep only ENSG IDs (gene-level; exclude ENST=transcript, ENSP=protein)
    ensg_mask = human["gene_id"].str.startswith("ENSG")
    genes_only = human[ensg_mask].copy()
    log.info("After ENSG filter: %d rows", len(genes_only))

    # Remove version suffixes (ENSG00000000003.15 -> ENSG00000000003)
    genes_only["gene_id"] = genes_only["gene_id"].str.split(".").str[0]

    # Drop URL column to save space
    genes_only = genes_only.drop(columns=["url", "evidence"])

    # Save parquet
    out_path = OUT / "gene2pathway.parquet"
    genes_only.to_parquet(out_path, index=False)
    log.info("Saved: %s", out_path)

    # Stats
    direct_df  = genes_only[genes_only["level"] == "direct"]
    alllev_df  = genes_only[genes_only["level"] == "all_levels"]

    n_genes_d  = direct_df["gene_id"].nunique()
    n_pairs_d  = len(direct_df)
    n_genes_al = alllev_df["gene_id"].nunique()
    n_pairs_al = len(alllev_df)

    log.info("=" * 60)
    log.info("RESULTS (Homo sapiens, ENSG IDs only)")
    log.info("  Direct:    genes=%d  pairs=%d", n_genes_d, n_pairs_d)
    log.info("  All-levels: genes=%d  pairs=%d", n_genes_al, n_pairs_al)
    log.info("=" * 60)

    print("\n=== A1 RESULTS ===")
    print(f"  Direct mapping:    {n_genes_d:,} genes,  {n_pairs_d:,} gene-pathway pairs")
    print(f"  All-levels mapping: {n_genes_al:,} genes, {n_pairs_al:,} gene-pathway pairs")
    print(f"  Output: {out_path}")

if __name__ == "__main__":
    main()
