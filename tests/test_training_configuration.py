from __future__ import annotations

import unittest
from pathlib import Path

from mace_fno.cli.config import parse_arguments
from mace_fno.training import TrainingConfig


class TrainingConfigurationTests(unittest.TestCase):
    def _configuration(self, *options: str) -> TrainingConfig:
        arguments = parse_arguments(
            [
                "--mace-model",
                "model.pt",
                "--train-file",
                "train.xyz",
                *options,
            ]
        )
        return TrainingConfig.from_namespace(arguments)

    def test_defaults_resolve_to_planar_configuration(self) -> None:
        configuration = self._configuration()

        self.assertEqual(configuration.model.spatial_scheme, "2d")
        self.assertEqual(configuration.model.resolved_z_modes, 8)
        self.assertEqual(configuration.optimization.evaluation_batch_size, 1)
        self.assertFalse(configuration.diagnostic.enabled)

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
