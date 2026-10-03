sex_ensg = {
    'XIST': 'ENSG00000229807',
    'RPS4Y1': 'ENSG00000129824',
    'KDM5D': 'ENSG00000012817',
    'DDX3Y': 'ENSG00000067048',
    'USP9Y': 'ENSG00000114374',
    'EIF1AY': 'ENSG00000198692',
    'TTTY15': 'ENSG00000183878',
}
import gzip
with gzip.open('data/raw/target_wt_xena/TARGET-WT.star_counts.tsv.gz', 'rt') as f:
    f.readline()
    all_ids = [line.split('\t')[0].split('.')[0] for line in f]

id_set = set(all_ids)
print("Sex genes in raw star_counts:")
for gene, ensg in sex_ensg.items():
    found = ensg in id_set
    print(f"  {gene} ({ensg}): {'FOUND' if found else 'NOT FOUND'}")

import pandas as pd
mrna = pd.read_parquet('data/processed/mrna_gene.parquet')
proc_cols = set(mrna.columns)
print("\nSex genes in processed mRNA (Reactome-filtered):")
for gene, ensg in sex_ensg.items():
    found = ensg in proc_cols
    print(f"  {gene} ({ensg}): {'YES' if found else 'NO (filtered out)'}")

# Check miRNA
mirna = pd.read_parquet('data/processed/mirna.parquet')
print(f"\nmiRNA shape: {mirna.shape}")
print("First 5 miRNA cols:", mirna.columns[:5].tolist())

# Check meth raw for sex chromosome probes
print("\nChecking methylation probes for sex chromosomes...")
import gzip as gz2
with gz2.open('data/raw/target_wt_xena/TARGET-WT.methylation450.tsv.gz', 'rt') as f:
    header = f.readline().strip().split('\t')
print(f"Meth probes header: {header[:3]} ... total samples/probe_col: {len(header)}")
