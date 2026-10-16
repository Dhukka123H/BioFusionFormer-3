#!/usr/bin/env python3
"""
scripts/predict.py
==================
Command-line interface for BioFusionFormer inference (Stage 6.3).

Usage:
    python scripts/predict.py --mrna path/to/mrna.csv --mirna path/to/mirna.csv --meth path/to/meth.csv --out preds.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.predict import predict


def parse_args():
    parser = argparse.ArgumentParser(description="BioFusionFormer Inference CLI (Stage 6.3)")
    parser.add_argument("--mrna", required=True, type=str, help="Path to mRNA expression CSV (index=patient_id, cols=Ensembl IDs)")
    parser.add_argument("--mirna", required=True, type=str, help="Path to miRNA expression CSV (index=patient_id, cols=precursors)")
    parser.add_argument("--meth", required=True, type=str, help="Path to methylation beta CSV (index=patient_id, cols=Ensembl IDs)")
    parser.add_argument("--out", default="predictions.csv", type=str, help="Output CSV path for predictions")
    parser.add_argument("--weights", default=None, type=str, help="Optional model checkpoint .pt path")
    return parser.parse_args()


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, index_col=0)
    return df


def main():
    args = parse_args()

    print(f"Loading mRNA from: {args.mrna}")
    mrna_df = load_table(args.mrna)

    print(f"Loading miRNA from: {args.mirna}")
    mirna_df = load_table(args.mirna)

    print(f"Loading methylation from: {args.meth}")
    meth_df = load_table(args.meth)

    print("Running BioFusionFormer prediction pipeline...")
    preds = predict(mrna_df, mirna_df, meth_df, weights_path=args.weights)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_p, index=False)
    print(f"Predictions written to: {out_p}")
    print(preds)


if __name__ == "__main__":
    main()
