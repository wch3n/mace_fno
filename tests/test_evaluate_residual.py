from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mace_fno.cli.evaluate_residual import (
    format_percent,
    improvement_percent,
    load_split_samples,
    metric_improvements,
)


class ResidualEvaluationTests(unittest.TestCase):
    def test_improvement_percent_preserves_deterioration_sign(self) -> None:
        self.assertAlmostEqual(improvement_percent(0.2, 0.15), 25.0)
        self.assertAlmostEqual(improvement_percent(0.2, 0.25), -25.0)
        self.assertIsNone(improvement_percent(0.0, 0.0))

    def test_metric_improvements_reports_energy_and_force_errors(self) -> None:
        baseline = {
            "energy_rmse": 0.010,
            "energy_mae": 0.008,
            "force_rmse": 0.100,
            "force_mae": 0.080,
        }
        corrected = {
            "energy_rmse": 0.008,
            "energy_mae": 0.006,
            "force_rmse": 0.090,
            "force_mae": 0.070,
        }
        result = metric_improvements(baseline, corrected)
        self.assertAlmostEqual(result["energy_rmse_percent"], 20.0)
        self.assertAlmostEqual(result["force_rmse_percent"], 10.0)
        self.assertEqual(format_percent(result["force_rmse_percent"]), "10.00%")
        self.assertEqual(format_percent(None), "--")

    def test_missing_validation_cache_is_reconstructed_from_train_cache(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_cache = root / "train.pt"
            validation_indices = root / "validation.txt"
            samples = [{"index": index} for index in range(5)]
            torch.save({"samples": samples}, train_cache)
            validation_indices.write_text("1 4\n")
            checkpoint = {
                "train_cache": str(train_cache),
                "validation_cache": str(root / "missing-validation.pt"),
                "validation_indices_file": str(validation_indices),
                "validation_fraction": 0.2,
                "seed": 17,
            }

            path, selected, reconstructed = load_split_samples(
                "validation", None, checkpoint
            )

            self.assertEqual(path, train_cache)
            self.assertEqual([sample["index"] for sample in selected], [1, 4])
            self.assertTrue(reconstructed)


if __name__ == "__main__":
    unittest.main()
