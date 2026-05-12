#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from scipy.stats import ttest_ind

from tp53_ml.config import load_config
from tp53_ml.genes import P53_PATHWAY_GENES, TP53_TARGET_GENES


def main() -> None:
    parser = argparse.ArgumentParser(description="Differential expression and biological interpretation tables.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(cfg["data"]["expression_processed"], index_col=0)
    labels = pd.read_csv(cfg["data"]["labels_processed"]).set_index("sample_id")
    y = labels.loc[X.index, "tp53_mutant"]

    mutant = X.loc[y == 1]
    wt = X.loc[y == 0]
    rows = []
    for gene in X.columns:
        mut_values = mutant[gene]
        wt_values = wt[gene]
        stat = ttest_ind(mut_values, wt_values, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "gene": gene,
                "mean_mutant": mut_values.mean(),
                "mean_wt": wt_values.mean(),
                "difference_mutant_minus_wt": mut_values.mean() - wt_values.mean(),
                "p_value": stat.pvalue,
            }
        )
    de = pd.DataFrame(rows)
    de["rank_p"] = de["p_value"].rank(method="first")
    de["fdr_bh"] = de["p_value"] * len(de) / de["rank_p"]
    de["fdr_bh"] = de["fdr_bh"].clip(upper=1.0)
    de = de.sort_values(["fdr_bh", "p_value"])
    de.to_csv("reports/tables/differential_expression_tp53_mutant_vs_wt.csv", index=False)

    top_de = set(de.head(args.top_n)["gene"])
    target_overlap = sorted(top_de & set(TP53_TARGET_GENES))
    pathway_overlap = sorted(top_de & set(P53_PATHWAY_GENES))

    model_genes_path = Path("reports/tables/binary_best_model_top_genes.csv")
    model_overlap_rows = []
    if model_genes_path.exists():
        model_genes = pd.read_csv(model_genes_path)
        top_model = set(model_genes.head(args.top_n)["gene"])
        model_overlap_rows = [
            {"gene_set": "TP53 direct targets", "overlap_count": len(top_model & set(TP53_TARGET_GENES)), "genes": ";".join(sorted(top_model & set(TP53_TARGET_GENES)))},
            {"gene_set": "p53 pathway", "overlap_count": len(top_model & set(P53_PATHWAY_GENES)), "genes": ";".join(sorted(top_model & set(P53_PATHWAY_GENES)))},
        ]

    summary = pd.DataFrame(
        [
            {"gene_set": "TP53 direct targets in top differential genes", "overlap_count": len(target_overlap), "genes": ";".join(target_overlap)},
            {"gene_set": "p53 pathway genes in top differential genes", "overlap_count": len(pathway_overlap), "genes": ";".join(pathway_overlap)},
            *model_overlap_rows,
        ]
    )
    summary.to_csv("reports/tables/biological_overlap_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
