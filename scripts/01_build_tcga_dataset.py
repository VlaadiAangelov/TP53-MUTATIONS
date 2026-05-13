#!/usr/bin/env python3
"""
Build processed TCGA dataset for TP53 mutation prediction.

Reads raw files from data/raw/ and writes to data/processed/tcga/:
  - expression_matched.csv.gz  (samples x genes, gene symbols as columns)
  - tp53_labels.csv            (sample_id, tp53_mutant, mutation_type, mutation_type_collapsed)
  - sample_metadata.csv        (sample_id, primary_disease, sample_type)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Maps TCGA MAF effect names to the same collapsed categories used for CCLE
EFFECT_MAP = {
    "Missense_Mutation":        "Missense",
    "Nonsense_Mutation":        "Nonsense",
    "Frame_Shift_Del":          "Frameshift",
    "Frame_Shift_Ins":          "Frameshift",
    "Splice_Site":              "Splice",
    "Silent":                   "Synonymous",
    "In_Frame_Del":             "In-frame indel",
    "In_Frame_Ins":             "In-frame indel",
}

# Severity order used to pick the worst mutation when a sample has multiple TP53 hits
SEVERITY = {
    "Frameshift":      1,
    "Nonsense":        1,
    "Splice":          2,
    "Missense":        3,
    "In-frame indel":  4,
    "Synonymous":      5,
    "Other":           6,
    # WT is never in the mutation table but listed for completeness
    "WT":              7,
}



def collapse_mutations(group: pd.DataFrame) -> pd.Series:
    """For samples with multiple TP53 mutations keep the most severe one."""
    group = group.copy()
    group["_sev"] = group["mutation_type_collapsed"].map(SEVERITY).fillna(6)
    worst = group.nsmallest(1, "_sev").iloc[0]
    return pd.Series({
        "mutation_type":           worst["mutation_type"],
        "mutation_type_collapsed": worst["mutation_type_collapsed"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed TCGA dataset.")
    parser.add_argument("--raw-dir",  default="data/raw",          help="Directory with raw TCGA files")
    parser.add_argument("--out-dir",  default="data/processed/tcga", help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",        help="Use 200 samples for a quick run")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load phenotype — keep Primary Tumor only
    # ------------------------------------------------------------------ #
    print("Loading phenotype …")
    phen = pd.read_csv(raw_dir / "tcga_phenotype.tsv.gz", sep="\t")
    primary = phen[phen["sample_type"] == "Primary Tumor"][["sample", "sample_type", "_primary_disease"]].copy()
    primary.columns = ["sample_id", "sample_type", "primary_disease"]
    primary_ids = set(primary["sample_id"])
    print(f"  Primary tumor samples: {len(primary_ids):,}")

    # ------------------------------------------------------------------ #
    # 2. Load expression — transpose to samples x genes
    # ------------------------------------------------------------------ #
    print("Loading expression matrix (this may take ~30 s) …")
    expr_raw = pd.read_csv(raw_dir / "tcga_expression.tsv.gz", sep="\t", index_col=0)
    # rows=genes (Entrez IDs), cols=samples → transpose
    expr = expr_raw.T
    expr.index.name = "sample_id"

    # Keep primary tumor samples only
    expr = expr.loc[expr.index.isin(primary_ids)]
    print(f"  Expression after primary-tumor filter: {expr.shape}")

    # ------------------------------------------------------------------ #
    # 3. Clean gene names
    # Gene names are already symbols (e.g. A1BG, TP53).
    # A small number (~29) are numeric Entrez-ID-style strings — drop them.
    # ------------------------------------------------------------------ #
    symbol_cols = [c for c in expr.columns if not str(c).lstrip("-").isdigit()]
    dropped = len(expr.columns) - len(symbol_cols)
    expr = expr[symbol_cols]
    # Drop duplicate gene symbols (keep first occurrence)
    expr = expr.loc[:, ~expr.columns.duplicated()]
    # Fill NaN with 0 — undetected genes have TPM=0, so log2(0+1)=0
    nan_count = expr.isnull().sum().sum()
    if nan_count > 0:
        print(f"  Filling {nan_count:,} NaN values with 0 (undetected genes)")
        expr = expr.fillna(0)
    # Drop zero-variance genes (no information content)
    zero_var = (expr.var(axis=0) == 0).sum()
    expr = expr.loc[:, expr.var(axis=0) > 0]
    print(f"  Dropped {dropped} numeric-ID columns and {zero_var} zero-variance genes; kept {expr.shape[1]:,} gene symbols")

    # ------------------------------------------------------------------ #
    # 4. Build TP53 labels
    # ------------------------------------------------------------------ #
    print("Building TP53 labels …")
    mut = pd.read_csv(raw_dir / "tcga_mutations.tsv.gz", sep="\t")
    tp53 = mut[mut["gene"] == "TP53"][["sample", "effect"]].copy()
    tp53 = tp53[tp53["sample"].isin(primary_ids)].copy()
    tp53["mutation_type"] = tp53["effect"]
    tp53["mutation_type_collapsed"] = tp53["effect"].map(EFFECT_MAP).fillna("Other")

    # Collapse multiple mutations per sample to the most severe
    tp53_per_sample = (
        tp53.groupby("sample")
        .apply(collapse_mutations)
        .reset_index()
        .rename(columns={"sample": "sample_id"})
    )

    # All expression samples get a label (WT if not in mutation table)
    all_samples = pd.DataFrame({"sample_id": expr.index})
    labels = all_samples.merge(tp53_per_sample, on="sample_id", how="left")
    labels["tp53_mutant"] = labels["mutation_type"].notna().astype(int)
    labels["mutation_type"] = labels["mutation_type"].fillna("WT")
    labels["mutation_type_collapsed"] = labels["mutation_type_collapsed"].fillna("WT")

    # Collapse rare mutation classes (< 200 mutant samples) into "Other"
    # so "Other" becomes a real predictable class rather than something to discard.
    # WT is excluded from this count since it is not a mutation class.
    MIN_CLASS_SAMPLES = 100
    mutant_mask = labels["mutation_type_collapsed"] != "WT"
    class_counts = labels.loc[mutant_mask, "mutation_type_collapsed"].value_counts()
    rare_classes = set(class_counts[class_counts < MIN_CLASS_SAMPLES].index)
    if rare_classes:
        print(f"  Collapsing into 'Other' (< {MIN_CLASS_SAMPLES} samples): {sorted(rare_classes)}")
        labels.loc[mutant_mask & labels["mutation_type_collapsed"].isin(rare_classes), "mutation_type_collapsed"] = "Other"

    print(f"  TP53 mutant: {labels['tp53_mutant'].sum():,}")
    print(f"  TP53 wild-type: {(labels['tp53_mutant'] == 0).sum():,}")
    print(f"  Mutation type counts:\n{labels['mutation_type_collapsed'].value_counts().to_string()}")

    # ------------------------------------------------------------------ #
    # 5. Align and optionally subsample
    # ------------------------------------------------------------------ #
    common = expr.index.intersection(labels["sample_id"])
    expr   = expr.loc[common]
    labels = labels[labels["sample_id"].isin(common)].reset_index(drop=True)
    meta   = primary[primary["sample_id"].isin(common)].reset_index(drop=True)

    if args.smoke_test:
        expr   = expr.iloc[:200]
        labels = labels.iloc[:200]
        meta   = meta.iloc[:200]
        print("Smoke-test mode: using 200 samples")

    # ------------------------------------------------------------------ #
    # 6. Save
    # ------------------------------------------------------------------ #
    print("Saving processed files …")
    expr.to_csv(out_dir / "expression_matched.csv.gz", compression="gzip")
    labels.to_csv(out_dir / "tp53_labels.csv", index=False)
    meta.to_csv(out_dir / "sample_metadata.csv", index=False)

    print(f"\nDone. Files written to {out_dir}/")
    print(f"  expression_matched.csv.gz : {expr.shape}")
    print(f"  tp53_labels.csv           : {labels.shape}")
    print(f"  sample_metadata.csv       : {meta.shape}")


if __name__ == "__main__":
    main()
