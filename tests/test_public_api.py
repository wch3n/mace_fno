from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr

import mace_fno
from mace_fno import (
    FNO2D,
    FNO3D,
    FNO2d,
    FNO3d,
    FNOFieldOperator2D,
    FNOFieldOperator3D,
    FNOFieldOperator3d,
    LinearFNO2D,
    LinearFNO2d,
    LinearFNO3D,
    LinearFNO3d,
)
from mace_fno.cli.config import parse_arguments


class PublicAPITests(unittest.TestCase):
    def test_canonical_dimension_aliases_preserve_legacy_api(self) -> None:
        self.assertIs(FNO2D, FNO2d)
        self.assertIs(FNO3D, FNO3d)
        self.assertIs(LinearFNO2D, LinearFNO2d)
        self.assertIs(LinearFNO3D, LinearFNO3d)
        self.assertEqual(FNOFieldOperator2D.__name__, "FNOFieldOperator")
        self.assertIs(FNOFieldOperator3D, FNOFieldOperator3d)

    def test_implementations_are_split_by_geometry(self) -> None:
        self.assertEqual(FNO2D.__module__, "mace_fno.fno_2d")
        self.assertEqual(FNO3D.__module__, "mace_fno.fno_3d")

    def test_train_arguments_can_be_parsed_programmatically(self) -> None:
        args = parse_arguments(
            ["--mace-model", "model.pt", "--train-file", "train.xyz"]
        )
        self.assertEqual(args.spatial_scheme, "auto")
        self.assertEqual(args.cell_mode, "fixed")
        self.assertEqual(args.interlacing_training, "full")
        self.assertEqual(args.metric_hidden_channels, 16)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.spectral_diagnostic_samples, 0)
        self.assertEqual(args.spectral_diagnostic_max_mode, 1)
        self.assertEqual(args.spectral_diagnostic_z_profiles, 3)
        self.assertEqual(args.spectral_diagnostic_depth, "fast")
        self.assertEqual(
            tuple(args.spectral_diagnostic_amplitudes), (0.025, 0.05, 0.1)
        )
        self.assertEqual(args.spectral_diagnostic_relative_span_tolerance, 0.05)

    def test_deep_spectral_diagnostic_arguments_can_be_overridden(self) -> None:
        args = parse_arguments(
            [
                "--mace-model",
                "model.pt",
                "--train-file",
                "train.xyz",
                "--spectral-diagnostic-depth",
                "deep",
                "--spectral-diagnostic-amplitudes",
                "0.01",
                "0.04",
            ]
        )
        self.assertEqual(args.spectral_diagnostic_depth, "deep")
        self.assertEqual(args.spectral_diagnostic_amplitudes, [0.01, 0.04])

    def test_anisotropic_cell_mode_can_be_selected(self) -> None:
        args = parse_arguments(
            [
                "--mace-model",
                "model.pt",
                "--train-file",
                "train.xyz",
                "--cell-mode",
                "anisotropic",
            ]
        )
        self.assertEqual(args.cell_mode, "anisotropic")

    def test_metric_eqgino_arguments_can_be_selected(self) -> None:
        args = parse_arguments(
            [
                "--mace-model",
                "model.pt",
                "--train-file",
                "train.xyz",
                "--spectral-symmetry",
                "metric_eqgino",
                "--metric-hidden-channels",
                "12",
            ]
        )
        self.assertEqual(args.spectral_symmetry, "metric_eqgino")
        self.assertEqual(args.metric_hidden_channels, 12)

    def test_legacy_eqgino_api_and_cli_options_are_removed(self) -> None:
        self.assertFalse(hasattr(mace_fno, "EqGINOSpectralConv3d"))
        self.assertFalse(hasattr(mace_fno, "CubicAdaptiveSpectralConv3d"))
        for legacy in ("eqgino", "cubic_adaptive"):
            with self.subTest(spectral_symmetry=legacy):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_arguments(
                            [
                                "--mace-model",
                                "model.pt",
                                "--train-file",
                                "train.xyz",
                                "--spectral-symmetry",
                                legacy,
                            ]
                        )


if __name__ == "__main__":
    unittest.main()
