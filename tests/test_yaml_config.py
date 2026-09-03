from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import yaml

from mace_fno.cli.config import parse_arguments
from mace_fno.cli.yaml_config import (
    resolved_configuration,
    write_resolved_configuration,
)


class YAMLConfigurationTests(unittest.TestCase):
    def _write_config(self, root: Path, document: object) -> Path:
        path = root / "train.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return path

    def test_nested_yaml_is_parsed_and_paths_are_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._write_config(
                root,
                {
                    "mace_model": "models/local.model",
                    "data": {
                        "train_file": "data/train.xyz",
                        "validation_file": "data/validation.xyz",
                    },
                    "model": {
                        "spatial_scheme": "3d",
                        "cell_mode": "anisotropic",
                        "grid": 24,
                        "spectral_symmetry": "metric_eqgino",
                    },
                    "training": {
                        "steps": 200,
                        "spectral_diagnostic_amplitudes": [0.01, 0.05],
                        "allow_periodic_z": True,
                    },
                    "checkpoint": "run/model.pt",
                },
            )

            args = parse_arguments(["--config", str(config)])

            self.assertEqual(args.config, config.resolve())
            self.assertEqual(args.mace_model, (root / "models/local.model").resolve())
            self.assertEqual(args.train_file, (root / "data/train.xyz").resolve())
            self.assertEqual(args.checkpoint, (root / "run/model.pt").resolve())
            self.assertEqual(args.spatial_scheme, "3d")
            self.assertEqual(args.cell_mode, "anisotropic")
            self.assertEqual(args.grid, 24)
            self.assertEqual(args.steps, 200)
            self.assertEqual(args.spectral_diagnostic_amplitudes, [0.01, 0.05])
            self.assertTrue(args.allow_periodic_z)

    def test_command_line_values_override_yaml_including_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._write_config(
                root,
                {
                    "mace_model": "model.pt",
                    "train_file": "train.xyz",
                    "training": {"steps": 200, "rebuild_cache": True},
                },
            )

            args = parse_arguments(
                [
                    "--config",
                    str(config),
                    "--steps",
                    "350",
                    "--no-rebuild-cache",
                ]
            )

            self.assertEqual(args.steps, 350)
            self.assertFalse(args.rebuild_cache)

    def test_unknown_and_duplicate_yaml_options_are_rejected(self) -> None:
        documents = (
            {"mace_model": "model.pt", "train_file": "train.xyz", "modse": 4},
            {
                "mace_model": "model.pt",
                "data": {"train_file": "train.xyz"},
                "other": {"train_file": "duplicate.xyz"},
            },
        )
        for document in documents:
            with self.subTest(document=document):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    config = self._write_config(Path(temporary_directory), document)
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            parse_arguments(["--config", str(config)])

    def test_yaml_values_retain_argparse_choice_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = self._write_config(
                Path(temporary_directory),
                {
                    "mace_model": "model.pt",
                    "train_file": "train.xyz",
                    "cell_mode": "not-a-cell-mode",
                },
            )
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_arguments(["--config", str(config)])

    def test_resolved_configuration_is_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = parse_arguments(
                [
                    "--mace-model",
                    "model.pt",
                    "--train-file",
                    "train.xyz",
                ]
            )
            resolved = resolved_configuration(args, spatial_scheme="3d")
            output = write_resolved_configuration(root / "model.config.yaml", resolved)
            reloaded = yaml.safe_load(output.read_text(encoding="utf-8"))

            self.assertEqual(reloaded["spatial_scheme"], "3d")
            self.assertTrue(Path(reloaded["mace_model"]).is_absolute())

    def test_tracked_benchmark_configurations_parse(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        cases = (
            ("benchmarks/au_mgo/train_fno_2d.yaml", "2d", "fixed"),
            ("benchmarks/au_mgo/train_fno_2p5d.yaml", "2.5d", "fixed"),
            ("benchmarks/water_scan_qnep/train_fno_3d.yaml", "3d", "isotropic"),
            ("benchmarks/llzo_qnep/train_fno_3d.yaml", "3d", "anisotropic"),
            ("benchmarks/les_water/train_fno_3d.yaml", "3d", "fixed"),
        )
        for relative_path, spatial_scheme, cell_mode in cases:
            with self.subTest(config=relative_path):
                args = parse_arguments(
                    [
                        "--config",
                        str(repository / relative_path),
                        "--mace-model",
                        "/runtime/local.model",
                        "--train-file",
                        "/runtime/train.xyz",
                    ]
                )
                self.assertEqual(args.spatial_scheme, spatial_scheme)
                self.assertEqual(args.cell_mode, cell_mode)


if __name__ == "__main__":
    unittest.main()
