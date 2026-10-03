"""
BioFusionFormer - Phase 4: Label-Independent Feature Construction
Constructs:
1. data/processed/mrna_gene.parquet
2. data/processed/mirna.parquet
3. data/processed/meth_gene.parquet & meth_cpg_top.parquet
4. data/interim/priors/mirna_pathway_mask_broad.npz (+ index JSON)
5. data/processed/pathway_features_fold_agnostic.parquet & pathway_hierarchy_mapping.json
"""
import sys
import os
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW_XENA = ROOT / "data" / "raw" / "target_wt_xena"
INTER = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PRIORS = INTER / "priors"
LOG_FILE = ROOT / "logs" / "phase4_features.log"

PROCESSED.mkdir(parents=True, exist_ok=True)
PRIORS.mkdir(parents=True, exist_ok=True)
ROOT.joinpath("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phase4")


def build_mrna_features(cohort_df: pd.DataFrame, reactome_genes: set) -> pd.DataFrame:
    log.info("\n--- 1. Processing mRNA Gene Expression Matrix ---")
    core_cohort = cohort_df[cohort_df["is_3omics_core"]].copy()
    patient_ids = core_cohort["patient_id"].tolist()
    sample_ids = core_cohort["mrna_sample_id"].tolist()
    sample2patient = dict(zip(sample_ids, patient_ids))

    mrna_path = RAW_XENA / "TARGET-WT.star_counts.tsv.gz"
    log.info("Loading mRNA counts from %s ...", mrna_path.name)
    df = pd.read_csv(mrna_path, sep="\t", index_col=0, low_memory=False)

    # Subset to 121 core samples
    df_core = df[sample_ids].copy()
    df_core.columns = [sample2patient[s] for s in df_core.columns]

    # Strip Ensembl version suffixes: ENSG00000223972.5 -> ENSG00000223972
    df_core.index = [idx.split(".")[0] for idx in df_core.index]
    # Deduplicate if any duplicate base ENSGs by taking mean
    if df_core.index.duplicated().any():
        df_core = df_core.groupby(df_core.index).mean()

    initial_genes = len(df_core)
    # Filter to Reactome genes
    reactome_overlap = [g for g in df_core.index if g in reactome_genes]
    df_reactome = df_core.loc[reactome_overlap].copy()
    log.info("Reactome gene coverage: %d / %d (%.1f%% of raw genes, %.1f%% of Reactome)",
             len(df_reactome), initial_genes, 100 * len(df_reactome) / initial_genes,
             100 * len(df_reactome) / len(reactome_genes))

    # Low-variance / near-zero expression filtering:
    # Thresholds: Drop genes where 90% of samples have <= 0.1 log2 counts (or max < 1.0)
    p90 = np.percentile(df_reactome.values, 90, axis=1)
    max_val = np.max(df_reactome.values, axis=1)
    keep_mask = (p90 > 0.1) & (max_val >= 1.0)

    df_filtered = df_reactome[keep_mask].copy()
    log.info("Low-expression filtering: dropped %d near-zero genes; %d genes retained",
             len(df_reactome) - len(df_filtered), len(df_filtered))

    # Transpose to shape (121 patients x N genes)
    mrna_final = df_filtered.T
    mrna_final.index.name = "patient_id"

    out_path = PROCESSED / "mrna_gene.parquet"
    mrna_final.to_parquet(out_path)
    log.info("Saved %s: shape=%s, size=%.2f MB", out_path.name, mrna_final.shape, out_path.stat().st_size / 1024**2)
    return mrna_final


def build_mirna_features(cohort_df: pd.DataFrame, prior_mask_mirnas: set) -> pd.DataFrame:
    log.info("\n--- 2. Processing miRNA Expression Matrix ---")
    core_cohort = cohort_df[cohort_df["is_3omics_core"]].copy()
    patient_ids = core_cohort["patient_id"].tolist()
    sample_ids = core_cohort["mirna_sample_id"].tolist()
    sample2patient = dict(zip(sample_ids, patient_ids))

    mirna_path = RAW_XENA / "TARGET-WT.mirna.tsv.gz"
    log.info("Loading miRNA expression from %s ...", mirna_path.name)
    df = pd.read_csv(mirna_path, sep="\t", index_col=0, low_memory=False)
    df_core = df[sample_ids].copy()
    df_core.columns = [sample2patient[s] for s in df_core.columns]

    # Harmonize stem-loop names to miRBase mature names using hsa.gff3
    gff3_path = ROOT / "data" / "raw" / "mirna_targets" / "hsa.gff3"
    stem2mature = {}
    id2stem = {}
    with open(gff3_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            feature_type = parts[2]
            attrs = dict(item.split("=") for item in parts[8].split(";") if "=" in item)
            if feature_type == "miRNA_primary_transcript":
                mi_id = attrs.get("ID")
                stem_name = attrs.get("Name")
                id2stem[mi_id] = stem_name
                if stem_name not in stem2mature:
                    stem2mature[stem_name] = []
            elif feature_type == "miRNA":
                mature_name = attrs.get("Name")
                derives_from = attrs.get("Derives_from")
                stem_name = id2stem.get(derives_from)
                if stem_name and mature_name:
                    stem2mature[stem_name].append(mature_name)

    # Lowercase lookup map
    stem_lower2mature = {k.lower(): v for k, v in stem2mature.items()}

    # Map each Xena stem-loop row to mature miRNAs
    mature_rows = []
    unmapped_stems = []
    for stem_id, row in df_core.iterrows():
        stem_str = str(stem_id).strip()
        matures = stem2mature.get(stem_str) or stem_lower2mature.get(stem_str.lower())
        if matures:
            for m in matures:
                mature_rows.append((m, row))
        else:
            unmapped_stems.append(stem_str)

    log.info("Harmonization: %d / %d stem-loops mapped to %d mature entries (%d unmapped)",
             len(df_core) - len(unmapped_stems), len(df_core), len(mature_rows), len(unmapped_stems))

    # Build mature DataFrame
    mature_records = {}
    for m, row in mature_rows:
        if m not in mature_records:
            mature_records[m] = []
        mature_records[m].append(row.values)

    # Average if multiple stem-loops produce identical mature miRNA (e.g. let-7a-1, let-7a-2, let-7a-3 -> let-7a-5p)
    avg_data = {m: np.mean(records, axis=0) for m, records in mature_records.items()}
    df_mature = pd.DataFrame(avg_data, index=df_core.columns).T

    # Filter to miRNAs that appear in the prior mask
    prior_overlap = [m for m in df_mature.index if m in prior_mask_mirnas]
    df_prior = df_mature.loc[prior_overlap].copy()
    log.info("Prior mask overlap: %d / %d mature miRNAs found in prior mask", len(df_prior), len(df_mature))

    # Low-expression filter: max log2(RPM+1) >= 0.5 across cohort
    keep_mirna = df_prior.max(axis=1) >= 0.5
    df_final_mirna = df_prior[keep_mirna].copy()
    log.info("Near-zero filter: dropped %d low miRNAs; %d mature miRNAs retained",
             len(df_prior) - len(df_final_mirna), len(df_final_mirna))

    mirna_final = df_final_mirna.T
    mirna_final.index.name = "patient_id"

    out_path = PROCESSED / "mirna.parquet"
    mirna_final.to_parquet(out_path)
    log.info("Saved %s: shape=%s, size=%.2f MB", out_path.name, mirna_final.shape, out_path.stat().st_size / 1024**2)
    return mirna_final


def build_methylation_features(cohort_df: pd.DataFrame, reactome_genes: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("\n--- 3. Processing DNA Methylation 450K Probes & Gene Aggregations ---")
    core_cohort = cohort_df[cohort_df["is_3omics_core"]].copy()
    patient_ids = core_cohort["patient_id"].tolist()
    sample_ids = core_cohort["meth_sample_id"].tolist()
    sample2patient = dict(zip(sample_ids, patient_ids))

    # 1. Load CpG annotations
    cpg_annot = pd.read_csv(INTER / "cpg2gene.csv", low_memory=False)
    log.info("Loaded %d CpG probe annotations", len(cpg_annot))

    # Filter out sex chromosomes
    non_sex = cpg_annot[~cpg_annot["chromosome"].astype(str).str.upper().isin(["X", "Y", "CHRX", "CHRY"])].copy()
    # Filter out probes with no gene
    gene_annotated = non_sex[non_sex["gene_symbol"].notna()].copy()
    log.info("Non-sex & gene-annotated probes: %d (dropped %d sex/intergenic)",
             len(gene_annotated), len(cpg_annot) - len(gene_annotated))

    # Load HGNC symbol to ENSG
    hgnc = pd.read_parquet(INTER / "hgnc_symbol2ensg.parquet")
    sym2ensg = dict(zip(hgnc["symbol"].astype(str).str.strip(), hgnc["ensembl_gene_id"].astype(str).str.strip()))

    # Map promoter probes: TSS1500, TSS200, 5'UTR, 1stExon
    promoter_keywords = ["TSS1500", "TSS200", "5'UTR", "1stExon"]
    is_promoter = gene_annotated["ucsc_refgene_group"].astype(str).apply(
        lambda g: any(k in g for k in promoter_keywords)
    )
    promoter_probes_df = gene_annotated[is_promoter].copy()
    log.info("Promoter-region candidate probes: %d", len(promoter_probes_df))

    # Map each promoter probe to Reactome ENSG IDs
    probe2reactome_genes = {}
    for probe, sym_str in zip(promoter_probes_df["probe_id"], promoter_probes_df["gene_symbol"]):
        syms = [s.strip() for s in str(sym_str).split(";") if s.strip()]
        ensgs = [sym2ensg[s] for s in syms if s in sym2ensg and sym2ensg[s] in reactome_genes]
        if ensgs:
            probe2reactome_genes[probe] = list(set(ensgs))

    target_promoter_probes = set(probe2reactome_genes.keys())
    log.info("Promoter probes mapped to Reactome genes: %d", len(target_promoter_probes))

    # 2. Stream methylation TSV in chunks to avoid high memory
    meth_path = RAW_XENA / "TARGET-WT.methylation450.tsv.gz"
    use_cols = ["Composite Element REF"] + sample_ids

    log.info("Streaming %s across 121 core samples...", meth_path.name)
    chunk_size = 50000
    all_variances = {}
    all_missing_rates = {}
    gene_probe_values = {}  # ensg -> list of probe arrays

    filtered_cpg_rows = {}  # for top variable matrix

    t0 = time.time()
    total_probes_processed = 0

    for chunk in pd.read_csv(meth_path, sep="\t", usecols=use_cols, chunksize=chunk_size, low_memory=False):
        chunk.rename(columns={"Composite Element REF": "probe_id"}, inplace=True)
        chunk.set_index("probe_id", inplace=True)
        # Rename columns to patient_ids
        chunk.columns = [sample2patient[s] for s in chunk.columns]

        # Calculate missing rate per probe
        n_missing = chunk.isna().sum(axis=1)
        miss_rate = n_missing / len(patient_ids)

        # Drop probes with > 20% missing
        valid_probes = miss_rate[miss_rate <= 0.20].index
        chunk_valid = chunk.loc[valid_probes]

        total_probes_processed += len(chunk)

        # Check promoter probes for gene aggregation
        common_promoters = set(chunk_valid.index).intersection(target_promoter_probes)
        if common_promoters:
            sub_prom = chunk_valid.loc[list(common_promoters)]
            for probe_id, vals in sub_prom.iterrows():
                ensg_targets = probe2reactome_genes[probe_id]
                v = vals.values
                for ensg in ensg_targets:
                    if ensg not in gene_probe_values:
                        gene_probe_values[ensg] = []
                    gene_probe_values[ensg].append(v)

        # Unsupervised variance calculation for top 20,000 CpGs
        # Filter out sex chromosomes for top CpGs as well
        non_sex_chunk = chunk_valid[~chunk_valid.index.isin(
            cpg_annot[cpg_annot["chromosome"].astype(str).str.upper().isin(["X", "Y", "CHRX", "CHRY"])]["probe_id"]
        )]
        if not non_sex_chunk.empty:
            vars_chunk = np.nanvar(non_sex_chunk.values, axis=1)
            for p_id, var, m_rate, row_vals in zip(non_sex_chunk.index, vars_chunk, miss_rate.loc[non_sex_chunk.index], non_sex_chunk.values):
                if not np.isnan(var):
                    all_variances[p_id] = var
                    all_missing_rates[p_id] = m_rate
                    filtered_cpg_rows[p_id] = row_vals

    log.info("Processed %d probes in %.1f seconds", total_probes_processed, time.time() - t0)

    # 3. Build gene-level promoter mean methylation matrix
    log.info("Aggregating promoter CpGs to Reactome gene-level mean beta...")
    gene_mean_beta = {}
    for ensg, probe_val_list in gene_probe_values.items():
        # probe_val_list is a list of arrays of length 121
        arr = np.array(probe_val_list)  # shape (n_probes, 121)
        # Mean across promoter probes for each patient (ignoring NaNs)
        with np.errstate(all="ignore"):
            mean_vals = np.nanmean(arr, axis=0)
        # Only keep if not all NaN
        if not np.all(np.isnan(mean_vals)):
            gene_mean_beta[ensg] = mean_vals

    df_meth_gene = pd.DataFrame(gene_mean_beta, index=patient_ids)
    df_meth_gene.index.name = "patient_id"

    out_meth_gene = PROCESSED / "meth_gene.parquet"
    df_meth_gene.to_parquet(out_meth_gene)
    log.info("Saved %s: shape=%s, genes=%d, size=%.2f MB",
             out_meth_gene.name, df_meth_gene.shape, df_meth_gene.shape[1], out_meth_gene.stat().st_size / 1024**2)

    # 4. Build top 20,000 most variable CpGs matrix
    log.info("Selecting top 20,000 most variable non-sex CpGs (unsupervised)...")
    sorted_probes = sorted(all_variances.keys(), key=lambda p: all_variances[p], reverse=True)[:20000]
    top_cpg_data = {p: filtered_cpg_rows[p] for p in sorted_probes}
    df_meth_top = pd.DataFrame(top_cpg_data, index=patient_ids)
    df_meth_top.index.name = "patient_id"

    # Record missing rate statistics
    top_miss_rates = [all_missing_rates[p] for p in sorted_probes]
    log.info("Top 20,000 CpGs: mean missing rate = %.2f%%, max missing rate = %.2f%%",
             100 * np.mean(top_miss_rates), 100 * np.max(top_miss_rates))

    out_meth_top = PROCESSED / "meth_cpg_top.parquet"
    df_meth_top.to_parquet(out_meth_top)
    log.info("Saved %s: shape=%s, size=%.2f MB", out_meth_top.name, df_meth_top.shape, out_meth_top.stat().st_size / 1024**2)

    return df_meth_gene, df_meth_top


def build_broad_mirna_mask(reactome_pathways: list[str]):
    log.info("\n--- 4. Building BROAD miRNA Prior Mask ---")
    df_all = pd.read_parquet(INTER / "mirna2gene_all.parquet")
    log.info("Loaded %d total miRTarBase rows", len(df_all))

    # Filter to functional interactions
    func_df = df_all[df_all["evidence_level"].isin(["Strong", "Weak"])].dropna(subset=["target_ensg"]).copy()
    func_df["is_strong"] = (func_df["evidence_level"] == "Strong").astype(int)

    # Group by (mirna_id, target_ensg)
    pair_counts = func_df.groupby(["mirna_id", "target_ensg"], observed=True).agg(
        n_records=("experiments", "count"),
        is_strong=("is_strong", "max")
    ).reset_index()

    # Broad criteria: strong evidence OR >= 2 independent experiment records
    broad_pairs = pair_counts[(pair_counts["is_strong"] == 1) | (pair_counts["n_records"] >= 2)].copy()
    log.info("Broad target pairs identified: %d (vs strong-only: %d)",
             len(broad_pairs), (pair_counts["is_strong"] == 1).sum())

    # Merge with Reactome gene2pathway
    g2p = pd.read_parquet(INTER / "reactome" / "gene2pathway.parquet")
    g2p_all = g2p[g2p["level"] == "all_levels"][["gene_id", "pathway_id"]].drop_duplicates()

    merged_broad = broad_pairs.merge(g2p_all, left_on="target_ensg", right_on="gene_id", how="inner")
    broad_mp = merged_broad[["mirna_id", "pathway_id"]].drop_duplicates()
    log.info("Broad miRNA-pathway pairs: %d", len(broad_mp))

    # Load miRNA universe and pathway universe from strong mask
    with open(PRIORS / "mirna_pathway_mask_mirnas.json", "r", encoding="utf-8") as f:
        mirna_universe = json.load(f)
    mirna2idx = {m: i for i, m in enumerate(mirna_universe)}
    pathway2idx = {p: i for i, p in enumerate(reactome_pathways)}

    valid_bmp = broad_mp[broad_mp["mirna_id"].isin(mirna2idx) & broad_mp["pathway_id"].isin(pathway2idx)]
    rows = [mirna2idx[m] for m in valid_bmp["mirna_id"]]
    cols = [pathway2idx[p] for p in valid_bmp["pathway_id"]]
    data = np.ones(len(rows), dtype=np.float32)

    broad_mask = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(mirna_universe), len(reactome_pathways)),
        dtype=np.float32
    )

    out_mask_npz = PRIORS / "mirna_pathway_mask_broad.npz"
    sparse.save_npz(out_mask_npz, broad_mask)

    # Save index maps
    with open(PRIORS / "mirna_pathway_mask_broad_mirnas.json", "w", encoding="utf-8") as f:
        json.dump(mirna_universe, f, indent=2)
    with open(PRIORS / "mirna_pathway_mask_broad_pathways.json", "w", encoding="utf-8") as f:
        json.dump(reactome_pathways, f, indent=2)

    # Compare coverage with strong-only mask
    strong_mask = sparse.load_npz(PRIORS / "mirna_pathway_mask.npz")
    log.info("\n=== miRNA Mask Coverage Comparison ===")
    log.info("  Strong-Only Mask: shape=%s, nnz=%d, density=%.4f%%, linked_mirnas=%d / %d (%.1f%%)",
             strong_mask.shape, strong_mask.nnz, 100 * strong_mask.nnz / (strong_mask.shape[0] * strong_mask.shape[1]),
             len(set(strong_mask.tocoo().row)), strong_mask.shape[0],
             100 * len(set(strong_mask.tocoo().row)) / strong_mask.shape[0])
    log.info("  Broad Mask:       shape=%s, nnz=%d, density=%.4f%%, linked_mirnas=%d / %d (%.1f%%)",
             broad_mask.shape, broad_mask.nnz, 100 * broad_mask.nnz / (broad_mask.shape[0] * broad_mask.shape[1]),
             len(set(rows)), broad_mask.shape[0],
             100 * len(set(rows)) / broad_mask.shape[0])


def build_pathway_features(mrna_df: pd.DataFrame, meth_df: pd.DataFrame, mirna_df: pd.DataFrame, inter_dir: Path):
    log.info("\n--- 5. Computing Pathway Features & Hierarchy Mapping ---")
    from src.pathways.scorer import PathwayZScorer, load_pathway_hierarchy_mapping

    # 1. Load gene-pathway mapping
    g2p = pd.read_parquet(inter_dir / "reactome" / "gene2pathway.parquet")
    g2p_all = g2p[g2p["level"] == "all_levels"].drop_duplicates(subset=["gene_id", "pathway_id"])

    # Build pathway -> genes dict
    pathway_genes = g2p_all.groupby("pathway_id")["gene_id"].apply(list).to_dict()

    # Filter pathways: keep pathways with 10 to 300 member genes present in our mRNA data
    avail_mrna_genes = set(mrna_df.columns)
    valid_pathways = {}
    for pid, glist in pathway_genes.items():
        present = [g for g in glist if g in avail_mrna_genes]
        if 10 <= len(present) <= 300:
            valid_pathways[pid] = present

    log.info("Pathways with 10-300 member genes present in mRNA: %d", len(valid_pathways))

    # 2. Score mRNA pathway activity (fold-agnostic baseline)
    mrna_scorer = PathwayZScorer(pathway_to_features=valid_pathways)
    mrna_scorer.fit(mrna_df)
    mrna_pathway_scores = mrna_scorer.transform(mrna_df)
    mrna_pathway_scores.columns = [f"mrna_pway_{c}" for c in mrna_pathway_scores.columns]
    log.info("mRNA pathway score matrix: %s", mrna_pathway_scores.shape)

    # 3. Score Methylation gene-level pathway activity
    avail_meth_genes = set(meth_df.columns)
    valid_meth_pathways = {}
    for pid, glist in pathway_genes.items():
        present = [g for g in glist if g in avail_meth_genes]
        if 10 <= len(present) <= 300:
            valid_meth_pathways[pid] = present

    meth_scorer = PathwayZScorer(pathway_to_features=valid_meth_pathways)
    meth_scorer.fit(meth_df)
    meth_pathway_scores = meth_scorer.transform(meth_df)
    meth_pathway_scores.columns = [f"meth_pway_{c}" for c in meth_pathway_scores.columns]
    log.info("Methylation pathway score matrix: %s", meth_pathway_scores.shape)

    # Combine fold-agnostic pathway features
    combined_pway = pd.concat([mrna_pathway_scores, meth_pathway_scores], axis=1)
    combined_pway.index.name = "patient_id"

    out_pway = PROCESSED / "pathway_features_fold_agnostic.parquet"
    combined_pway.to_parquet(out_pway)
    log.info("Saved %s: shape=%s, size=%.2f MB", out_pway.name, combined_pway.shape, out_pway.stat().st_size / 1024**2)

    # 4. Save pathway hierarchy depth and parent/child mapping
    hierarchy = load_pathway_hierarchy_mapping(inter_dir)
    out_hier = PROCESSED / "pathway_hierarchy_mapping.json"
    with open(out_hier, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2)
    log.info("Saved %s with %d pathways", out_hier.name, len(hierarchy))


def main():
    t0 = time.time()
    log.info("Starting Phase 4: Label-Independent Feature Construction...")

    cohort_df = pd.read_csv(INTER / "cohort_patients.csv")
    g2p = pd.read_parquet(INTER / "reactome" / "gene2pathway.parquet")
    reactome_genes = set(g2p[g2p["level"] == "all_levels"]["gene_id"].unique())

    nodes_df = pd.read_csv(INTER / "reactome" / "pathway_nodes.csv")
    reactome_pathways = sorted(nodes_df["id"].unique().tolist())

    with open(PRIORS / "mirna_pathway_mask_mirnas.json", "r", encoding="utf-8") as f:
        prior_mask_mirnas = set(json.load(f))

    # 1. mRNA
    mrna_path = PROCESSED / "mrna_gene.parquet"
    if mrna_path.exists():
        log.info("Loading existing %s", mrna_path.name)
        mrna_df = pd.read_parquet(mrna_path)
    else:
        mrna_df = build_mrna_features(cohort_df, reactome_genes)

    # 2. miRNA
    mirna_path = PROCESSED / "mirna.parquet"
    if mirna_path.exists():
        log.info("Loading existing %s", mirna_path.name)
        mirna_df = pd.read_parquet(mirna_path)
    else:
        mirna_df = build_mirna_features(cohort_df, prior_mask_mirnas)

    # 3. Methylation
    meth_gene_path = PROCESSED / "meth_gene.parquet"
    meth_cpg_path = PROCESSED / "meth_cpg_top.parquet"
    if meth_gene_path.exists() and meth_cpg_path.exists():
        log.info("Loading existing %s and %s", meth_gene_path.name, meth_cpg_path.name)
        meth_gene_df = pd.read_parquet(meth_gene_path)
    else:
        meth_gene_df, meth_cpg_df = build_methylation_features(cohort_df, reactome_genes)

    # 4. Broad miRNA prior mask
    broad_mask_path = PRIORS / "mirna_pathway_mask_broad.npz"
    if not broad_mask_path.exists():
        build_broad_mirna_mask(reactome_pathways)
    else:
        log.info("Broad miRNA mask already exists: %s", broad_mask_path.name)

    # 5. Pathway features & hierarchy mapping
    build_pathway_features(mrna_df, meth_gene_df, mirna_df, INTER)

    log.info("==================================================")
    log.info("PHASE 4 COMPLETE in %.2f seconds.", time.time() - t0)
    log.info("==================================================")


if __name__ == "__main__":
    main()
