from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tp53_baseline.config import BaselineConfig
from tp53_baseline.data import (
    TARGET_COLUMN,
    build_labeled_dataset,
    get_numeric_feature_matrix,
    get_tp53_related_columns,
)
from tp53_baseline.model_training import train_top_features_model
from tp53_baseline.pipeline import run_baseline


class BaselinePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        expression_df = pd.DataFrame(
            {
                "Unnamed: 0": ["ACH-1", "ACH-2", "ACH-3", "ACH-4"],
                "TP53 (ENSG00000141510)": [10.0, 11.0, 12.0, 13.0],
                "GENE_A (ENSG000001)": [9.0, 8.5, 1.0, 1.5],
                "GENE_B (ENSG000002)": [5.5, 4.5, 2.0, 1.5],
                "GENE_C (ENSG000003)": [1.0, 1.0, 1.0, 1.0],
            }
        )
        mutation_df = pd.DataFrame(
            {
                "Hugo_Symbol": ["TP53", "KRAS", "TP53"],
                "DepMap_ID": ["ACH-1", "ACH-2", "ACH-3"],
            }
        )

        self.expression_path = self.temp_path / "expression.csv"
        self.mutation_path = self.temp_path / "mutations.csv"
        expression_df.to_csv(self.expression_path, index=False)
        mutation_df.to_csv(self.mutation_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_labeled_dataset_marks_tp53_samples(self) -> None:
        expression_df = pd.read_csv(self.expression_path).rename(columns={"Unnamed: 0": "DepMap_ID"})
        mutation_df = pd.read_csv(self.mutation_path)
        dataset = build_labeled_dataset(expression_df, mutation_df)

        self.assertEqual(dataset[TARGET_COLUMN].tolist(), [1, 0, 1, 0])

    def test_tp53_related_columns_are_excluded(self) -> None:
        expression_df = pd.read_csv(self.expression_path).rename(columns={"Unnamed: 0": "DepMap_ID"})
        mutation_df = pd.read_csv(self.mutation_path)
        dataset = build_labeled_dataset(expression_df, mutation_df)
        numeric_features = get_numeric_feature_matrix(dataset)
        excluded_columns = get_tp53_related_columns(list(numeric_features.columns), "TP53")

        self.assertIn("TP53 (ENSG00000141510)", excluded_columns)
        self.assertEqual(len(excluded_columns), 1)

    def test_run_baseline_writes_expected_outputs(self) -> None:
        output_dir = self.temp_path / "outputs"
        config = BaselineConfig(
            expression_csv=self.expression_path,
            mutation_csv=self.mutation_path,
            output_dir=output_dir,
            top_n=2,
            exclude_prefix="TP53",
        )

        results = run_baseline(config)

        summary = json.loads(results["summary_path"].read_text(encoding="utf-8"))
        ranking_df = pd.read_csv(results["ranking_path"])
        top_features_df = pd.read_csv(results["top_features_path"])

        self.assertEqual(summary["merged_samples"], 4)
        self.assertEqual(summary["tp53_positive_samples"], 2)
        self.assertEqual(summary["tp53_negative_samples"], 2)
        self.assertEqual(summary["excluded_tp53_related_feature_count"], 1)
        self.assertListEqual(top_features_df["feature"].tolist(), ranking_df["feature"].head(2).tolist())
        self.assertNotIn("TP53 (ENSG00000141510)", ranking_df["feature"].tolist())

    def test_train_top_features_model_writes_metrics(self) -> None:
        top_features_path = self.temp_path / "top_features.csv"
        pd.DataFrame({"feature": ["GENE_A (ENSG000001)", "GENE_B (ENSG000002)"]}).to_csv(
            top_features_path,
            index=False,
        )

        results = train_top_features_model(
            expression_csv=self.expression_path,
            mutation_csv=self.mutation_path,
            top_features_csv=top_features_path,
            output_dir=self.temp_path / "model_outputs",
            test_size=0.5,
            random_state=7,
        )

        metrics = json.loads(results["metrics_path"].read_text(encoding="utf-8"))
        predictions_df = pd.read_csv(results["predictions_path"])
        coefficients_df = pd.read_csv(results["coefficients_path"])

        self.assertEqual(metrics["dataset"]["samples"], 4)
        self.assertEqual(metrics["dataset"]["features"], 2)
        self.assertIn("dummy_stratified", metrics)
        self.assertIn("logistic_regression_top500", metrics)
        self.assertEqual(len(predictions_df), 2)
        self.assertEqual(set(coefficients_df["feature"]), {"GENE_A (ENSG000001)", "GENE_B (ENSG000002)"})


if __name__ == "__main__":
    unittest.main()
