#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tp53_ml.config import load_config
from tp53_ml.preprocessing import TopVarianceSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA tables and plots.")
    parser.add_argument("--config", default="config/project.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fig_dir = Path("reports/figures")
    table_dir = Path("reports/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(cfg["data"]["expression_processed"], index_col=0)
    labels = pd.read_csv(cfg["data"]["labels_processed"])
    metadata = pd.read_csv(cfg["data"]["metadata_processed"])

    class_counts = labels["tp53_mutant"].map({0: "WT", 1: "Mutant"}).value_counts().rename_axis("class")
    class_table = class_counts.reset_index(name="count")
    class_table["percentage"] = 100 * class_table["count"] / class_table["count"].sum()
    class_table.to_csv(table_dir / "binary_class_balance.csv", index=False)

    mutation_counts = labels["mutation_type_collapsed"].value_counts().rename_axis("mutation_type").reset_index(name="count")
    mutation_counts["percentage"] = 100 * mutation_counts["count"] / mutation_counts["count"].sum()
    mutation_counts.to_csv(table_dir / "mutation_type_balance.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.barplot(class_table, x="class", y="count", hue="class", legend=False, ax=ax, palette=["#4C78A8", "#E45756"])
    ax.set_xlabel("")
    ax.set_ylabel("Samples")
    ax.set_title("TP53 mutation status")
    fig.tight_layout()
    fig.savefig(fig_dir / "binary_class_balance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.barplot(mutation_counts, x="mutation_type", y="count", hue="mutation_type", legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Samples")
    ax.set_title("TP53 mutation type distribution")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(fig_dir / "mutation_type_balance.png", dpi=180)
    plt.close(fig)

    selector = TopVarianceSelector(k=min(3000, X.shape[1]), min_mean_expression=cfg["features"]["min_mean_expression"])
    X_sel = selector.fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_sel)
    pca = PCA(n_components=2, random_state=cfg["project"]["random_state"])
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["sample_id"] = labels["sample_id"].to_numpy()
    pca_df["TP53 status"] = labels["tp53_mutant"].map({0: "WT", 1: "Mutant"}).to_numpy()
    pca_df["mutation_type"] = labels["mutation_type_collapsed"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.scatterplot(pca_df, x="PC1", y="PC2", hue="TP53 status", s=35, linewidth=0, alpha=0.8, ax=ax)
    ax.set_title("PCA of high-variance expression genes")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_tp53_status.png", dpi=180)
    plt.close(fig)

    lineage_col = next((col for col in ["OncotreeLineage", "onCotreeLineage", "lineage", "primary_disease"] if col in metadata.columns), None)
    if lineage_col:
        pca_df = pca_df.merge(metadata[["sample_id", lineage_col]], on="sample_id", how="left")
        top_lineages = pca_df[lineage_col].value_counts().head(12).index
        pca_df["lineage_plot"] = pca_df[lineage_col].where(pca_df[lineage_col].isin(top_lineages), "Other")
        fig, ax = plt.subplots(figsize=(7, 5.2))
        sns.scatterplot(pca_df, x="PC1", y="PC2", hue="lineage_plot", s=30, linewidth=0, alpha=0.8, ax=ax)
        ax.set_title("PCA colored by cancer lineage")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "pca_lineage.png", dpi=180)
        plt.close(fig)

    summary = pd.DataFrame(
        [
            {"metric": "samples", "value": X.shape[0]},
            {"metric": "genes", "value": X.shape[1]},
            {"metric": "missing_values", "value": int(X.isna().sum().sum())},
            {"metric": "mean_expression", "value": float(X.to_numpy().mean())},
            {"metric": "median_expression", "value": float(pd.Series(X.to_numpy().ravel()).median())},
        ]
    )
    summary.to_csv(table_dir / "expression_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
