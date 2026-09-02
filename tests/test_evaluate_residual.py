from __future__ import annotations

import unittest

from mace_fno.cli.evaluate_residual import (
    format_percent,
    improvement_percent,
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


if __name__ == "__main__":
    unittest.main()
