from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def load_prepare_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "llzo_qnep"
        / "prepare_dataset.py"
    )
    specification = importlib.util.spec_from_file_location("llzo_prepare", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


PREPARE = load_prepare_module()


def load_summary_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "llzo_qnep"
        / "summarize_results.py"
    )
    specification = importlib.util.spec_from_file_location("llzo_summary", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


SUMMARY = load_summary_module()


class LLZOBenchmarkTests(unittest.TestCase):
    def test_classify_cell(self) -> None:
        self.assertEqual(PREPARE.classify_cell(np.diag((12.5, 12.5, 12.5))), "cubic")
        self.assertEqual(
            PREPARE.classify_cell(np.diag((12.5, 12.5, 13.0))), "tetragonal"
        )
        self.assertEqual(
            PREPARE.classify_cell(np.diag((12.4, 12.8, 13.0))), "orthorhombic"
        )
        with self.assertRaisesRegex(ValueError, "not aligned"):
            PREPARE.classify_cell(
                np.asarray(((12.5, 0.1, 0.0), (0.0, 12.5, 0.0), (0.0, 0.0, 12.5)))
            )

    def test_stratified_split_is_disjoint_complete_and_deterministic(self) -> None:
        groups = ["orthorhombic"] * 1201 + ["cubic"] * 533 + ["tetragonal"] * 244
        first = PREPARE.stratified_split_indices(groups, 0.10, 0.10, seed=17)
        second = PREPARE.stratified_split_indices(groups, 0.10, 0.10, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(indices) for name, indices in first.items()},
            {"train": 1582, "validation": 198, "test": 198},
        )
        selected = [index for indices in first.values() for index in indices]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(sorted(selected), list(range(len(groups))))

    def test_source_blocked_split_keeps_whole_blocks_together(self) -> None:
        groups = [
            group
            for _ in range(20)
            for group in (
                "cubic",
                "tetragonal",
                "orthorhombic",
                "orthorhombic",
                "orthorhombic",
                "cubic",
                "tetragonal",
                "orthorhombic",
                "orthorhombic",
                "orthorhombic",
            )
        ]
        split = PREPARE.source_blocked_split_indices(
            groups,
            validation_fraction=0.20,
            test_fraction=0.20,
            seed=17,
            block_size=10,
        )
        repeated = PREPARE.source_blocked_split_indices(
            groups,
            validation_fraction=0.20,
            test_fraction=0.20,
            seed=17,
            block_size=10,
        )
        self.assertEqual(split, repeated)
        owner = {
            index: name for name, indices in split.items() for index in indices
        }
        self.assertEqual(sorted(owner), list(range(len(groups))))
        for start in range(0, len(groups), 10):
            assignments = {
                owner[index] for index in range(start, min(start + 10, len(groups)))
            }
            self.assertEqual(len(assignments), 1)
        for indices in split.values():
            self.assertEqual({groups[index] for index in indices}, set(groups))

    def test_source_blocked_split_rejects_missing_cell_class(self) -> None:
        groups = ["cubic"] * 90 + ["tetragonal"] * 10
        with self.assertRaisesRegex(ValueError, "omits cell classes"):
            PREPARE.source_blocked_split_indices(
                groups,
                validation_fraction=0.10,
                test_fraction=0.10,
                seed=1,
                block_size=10,
            )

    def test_summary_combines_metrics_and_audits(self) -> None:
        group = {
            "structures": 10,
            "energy_eV_per_atom": {"rmse": 0.010},
            "forces_eV_per_A": {"rmse": 0.100},
        }
        mace_one = {
            "test": {
                "overall": group,
                "by_benchmark_group": {"cubic": group},
            },
            "test_global_offset": {"overall": group},
        }
        mace_two = {
            "test": {
                "overall": {
                    "energy_eV_per_atom": {"rmse": 0.008},
                    "forces_eV_per_A": {"rmse": 0.080},
                },
                "by_benchmark_group": {
                    "cubic": {
                        "energy_eV_per_atom": {"rmse": 0.008},
                        "forces_eV_per_A": {"rmse": 0.080},
                    }
                },
            }
        }
        residual_group = {"energy_rmse": 0.007, "force_rmse": 0.070}
        fno = {
            "splits": {
                "test": {
                    "frozen_mace": {
                        "energy_rmse": 0.010,
                        "force_rmse": 0.100,
                    },
                    "mace_fno": {
                        "energy_rmse": 0.007,
                        "force_rmse": 0.070,
                        "by_benchmark_group": {"cubic": residual_group},
                    },
                }
            }
        }
        manifest = {
            "split": {
                "seed": 17,
                "counts": {"train": 1582, "validation": 198, "test": 198},
            }
        }
        audit = {
            "promised_exact_checks": {
                "force_additivity": {
                    "observed": 1.0e-8,
                    "threshold": 1.0e-5,
                    "passed": True,
                }
            }
        }
        spectral = {
            "diagnostic_kind": "periodic_3d",
            "low_k_dominant_eigenvalue_fit": {
                "free_power_exponent_p": 2.1,
                "free_log_r2": 0.9,
                "coulomb_p2_log_r2": 0.89,
            },
            "pooled_anisotropic_inverse_quadratic_fit": None,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {
                "manifest": manifest,
                "mace-one": mace_one,
                "mace-two": mace_two,
                "fno": fno,
                "audit": audit,
                "spectral": spectral,
            }
            arguments = ["summarize_results.py"]
            for option, payload in inputs.items():
                filename = root / f"{option}.json"
                filename.write_text(json.dumps(payload))
                arguments.extend((f"--{option}", str(filename)))
            output_json = root / "summary.json"
            output_markdown = root / "summary.md"
            arguments.extend(
                (
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_markdown),
                )
            )
            with patch.object(sys, "argv", arguments):
                SUMMARY.main()
            report = json.loads(output_json.read_text())
            self.assertTrue(report["assessment"]["held_out_energy_improved"])
            self.assertTrue(report["assessment"]["held_out_force_improved"])
            self.assertTrue(report["assessment"]["exact_audit_passed"])
            self.assertIn("MACE, one interaction + FNO", output_markdown.read_text())


if __name__ == "__main__":
    unittest.main()
