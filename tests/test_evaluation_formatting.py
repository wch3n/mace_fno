from __future__ import annotations

import contextlib
import io
import unittest

from mace_fno.training.evaluation import print_metrics


class MetricFormattingTests(unittest.TestCase):
    def test_print_metrics_uses_aligned_fixed_width_columns(self) -> None:
        metrics = {
            "energy_mae": 0.0002525,
            "energy_rmse": 0.0002904,
            "energy_bias": -0.0000233,
            "force_mae": 0.0206721,
            "force_rmse": 0.0268975,
            "by_formula": {
                "H128O64": {
                    "energy_rmse": 0.0002904,
                    "force_rmse": 0.0268975,
                }
            },
            "formula_counts": {"H128O64": 30},
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_metrics("frozen MACE validation", metrics)

        rows = output.getvalue().splitlines()
        self.assertEqual(len(rows), 2)
        separators = [
            [index for index, char in enumerate(row) if char == "|"] for row in rows
        ]
        self.assertEqual(separators[0], separators[1])
        self.assertIn("E_MAE=    0.2525", rows[0])
        self.assertIn("E_ME=   -0.0233", rows[0])
        self.assertIn("E_MAE=        --", rows[1])
        self.assertIn("F_RMSE=   26.8975", rows[1])

    def test_empty_metrics_print_nothing(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_metrics("empty", {})
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
