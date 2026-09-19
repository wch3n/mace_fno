from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from mace_fno_test_helpers import train_arguments

from mace_fno.training import TrainingConfig


class TrainingArgumentTests(unittest.TestCase):
    def test_training_argument_defaults(self) -> None:
        arguments = train_arguments()
        expected = {
            "spatial_scheme": "auto",
            "cell_mode": "fixed",
            "interlacing_training": "full",
            "metric_hidden_channels": 16,
            "metric_parameterization": "shell_spline",
            "batch_size": 1,
            "mace_training": "frozen",
            "mace_learning_rate": 1.0e-5,
            "early_stopping_patience_steps": 0,
            "spectral_diagnostic_samples": 0,
            "spectral_diagnostic_max_mode": 1,
            "spectral_diagnostic_z_profiles": 3,
            "spectral_diagnostic_depth": "fast",
            "spectral_diagnostic_relative_span_tolerance": 0.05,
        }
        for name, value in expected.items():
            with self.subTest(argument=name):
                self.assertEqual(getattr(arguments, name), value)
        self.assertEqual(
            tuple(arguments.spectral_diagnostic_amplitudes), (0.025, 0.05, 0.1)
        )

    def test_training_arguments_can_be_overridden(self) -> None:
        cases = (
            (
                "deep spectral diagnostics",
                (
                    "--spectral-diagnostic-depth",
                    "deep",
                    "--spectral-diagnostic-amplitudes",
                    "0.01",
                    "0.04",
                ),
                {
                    "spectral_diagnostic_depth": "deep",
                    "spectral_diagnostic_amplitudes": [0.01, 0.04],
                },
            ),
            (
                "anisotropic cell",
                ("--cell-mode", "anisotropic"),
                {"cell_mode": "anisotropic"},
            ),
            (
                "metric EqGINO",
                (
                    "--spectral-symmetry",
                    "metric_eqgino",
                    "--metric-hidden-channels",
                    "12",
                ),
                {"spectral_symmetry": "metric_eqgino", "metric_hidden_channels": 12},
            ),
            (
                "step-based early stopping",
                ("--early-stopping-patience-steps", "500"),
                {"early_stopping_patience_steps": 500},
            ),
        )
        for name, options, expected in cases:
            with self.subTest(case=name):
                arguments = train_arguments(*options)
                for argument, value in expected.items():
                    with self.subTest(argument=argument):
                        self.assertEqual(getattr(arguments, argument), value)

    def test_legacy_eqgino_cli_options_are_rejected(self) -> None:
        for legacy in ("eqgino", "cubic_adaptive"):
            with self.subTest(spectral_symmetry=legacy):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        train_arguments("--spectral-symmetry", legacy)


class TrainingConfigurationTests(unittest.TestCase):
    def _configuration(self, *options: str) -> TrainingConfig:
        return TrainingConfig.from_namespace(train_arguments(*options))

    def test_defaults_resolve_to_planar_configuration(self) -> None:
        configuration = self._configuration()

        self.assertEqual(configuration.model.spatial_scheme, "2d")
        self.assertEqual(configuration.model.resolved_z_modes, 8)
        self.assertEqual(configuration.optimization.evaluation_batch_size, 1)
        self.assertEqual(configuration.optimization.mace_training, "frozen")
        self.assertFalse(configuration.diagnostic.enabled)

    def test_joint_training_has_separate_mace_controls(self) -> None:
        configuration = self._configuration(
            "--steps",
            "20",
            "--mace-training",
            "joint",
            "--mace-learning-rate",
            "2e-5",
            "--mace-warmup-steps",
            "5",
        )

        self.assertEqual(configuration.optimization.mace_training, "joint")
        self.assertEqual(configuration.optimization.mace_learning_rate, 2.0e-5)
        self.assertEqual(configuration.optimization.mace_warmup_steps, 5)

    def test_auto_scheme_resolves_to_slab_when_z_grid_is_present(self) -> None:
        configuration = self._configuration(
            "--z-grid",
            "16",
            "--z-extent",
            "20.0",
        )

        self.assertEqual(configuration.model.spatial_scheme, "2.5d")
        self.assertEqual(configuration.model.z_grid, 16)
        self.assertEqual(configuration.model.z_extent, 20.0)

    def test_periodic_configuration_resolves_z_modes_and_batch_size(self) -> None:
        configuration = self._configuration(
            "--spatial-scheme",
            "3d",
            "--cell-mode",
            "anisotropic",
            "--z-grid",
            "24",
            "--z-modes",
            "6",
            "--spectral-symmetry",
            "metric_eqgino",
            "--batch-size",
            "2",
            "--evaluation-batch-size",
            "5",
        )

        self.assertEqual(configuration.model.resolved_z_modes, 6)
        self.assertEqual(configuration.optimization.evaluation_batch_size, 5)

    def test_diagnostic_output_is_derived_from_checkpoint(self) -> None:
        configuration = self._configuration(
            "--checkpoint",
            "run/model.pt",
            "--spectral-diagnostic-samples",
            "2",
        )

        self.assertEqual(
            configuration.diagnostic.output,
            Path("run/model_spectral_training.json"),
        )

    def test_invalid_cross_section_combinations_are_rejected(self) -> None:
        cases = (
            ("--spatial-scheme", "2d", "--z-modes", "2"),
            ("--spatial-scheme", "3d", "--z-grid", "8", "--z-modes", "5"),
            ("--output-warmup-steps", "1000"),
            ("--evaluation-batch-size", "-1"),
            ("--mace-warmup-steps", "1"),
            ("--mace-training", "joint", "--output-warmup-steps", "1"),
            ("--early-stopping-patience-steps", "-1"),
            (
                "--early-stopping-patience-steps",
                "10",
                "--early-stopping-patience-evals",
                "2",
                "--lr-scheduler",
                "plateau",
            ),
            (
                "--spectral-diagnostic-samples",
                "1",
                "--spectral-diagnostic-max-mode",
                "16",
            ),
        )
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self._configuration(*options)

    def test_validation_sources_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            self._configuration(
                "--validation-file",
                "validation.xyz",
                "--validation-indices-file",
                "validation.txt",
            )


if __name__ == "__main__":
    unittest.main()
