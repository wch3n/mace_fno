"""Near-best checkpoint selection without test-set access or mixed weights."""

from __future__ import annotations

import io
import math
import random
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
import yaml
from mace_fno_test_helpers import train_arguments
from test_trainer import _sample, _ToyJointModel, _ToyResidual

from mace_fno.cli import train_residual
from mace_fno.cli.config import parse_arguments
from mace_fno.training import OptimizationResult, PreparedData, TrainingConfig
from mace_fno.training.selection import EnergyCheckpointSelector


def _metrics(energy, force=1.0, bias=0.0):
    return {"energy_rmse": energy, "force_rmse": force, "energy_bias": bias}


class EnergySelectionTests(unittest.TestCase):
    def test_tightening_ceiling_can_select_an_older_nonwinner(self):
        selector = EnergyCheckpointSelector(0.10)
        for step, (energy, loss) in enumerate(((1.0, 1.09), (1.1, 1.03), (1.5, 1.0))):
            selector.consider(step, _metrics(energy), loss, lambda: {})
        self.assertEqual(selector.selected["candidate"]["step"], 0)
        # The first candidate expires, but step 1 is still better than step 3.
        selector.consider(3, _metrics(2.0), 0.95, lambda: {})
        self.assertEqual(selector.selected["candidate"]["step"], 1)
        self.assertAlmostEqual(selector.selected["metadata"]["constraint_limit"], 1.045)

    def test_pareto_selection_matches_full_history_at_every_check(self):
        rng = random.Random(17)
        for constraint in ("loss", "forces"):
            for tolerance in (0.0, 0.01, 0.25):
                selector = EnergyCheckpointSelector(tolerance, constraint)
                history = []
                for step in range(150):
                    energy, force, loss = [rng.uniform(0.1, 2.0) for _ in range(3)]
                    value = loss if constraint == "loss" else force
                    history.append((energy, value, step))
                    selector.consider(step, _metrics(energy, force), loss, lambda: {})
                    limit = (1 + tolerance) * min(item[1] for item in history)
                    expected = min(item for item in history if item[1] <= limit)
                    self.assertEqual(
                        selector.selected["candidate"]["step"], expected[2]
                    )

    def test_centered_metric_changes_ranking_without_energy_calibration(self):
        for metric, expected in (("raw", 1), ("centered", 0)):
            selector = EnergyCheckpointSelector(0.1, metric=metric)
            selector.consider(0, _metrics(2.0, bias=1.99), 1.0, lambda: {"scale": 7})
            selector.consider(1, _metrics(1.0), 1.0, lambda: {"scale": 8})
            chosen = selector.selected
            self.assertEqual(chosen["candidate"]["step"], expected)
            self.assertFalse(chosen["metadata"]["energy_shift_applied"])
            self.assertEqual(chosen["candidate"]["state"]["scale"], 7 + expected)

    def test_ties_nonfinite_metrics_and_zero_constraint(self):
        selector = EnergyCheckpointSelector(0.0)
        snapshot = mock.Mock(return_value={})
        selector.consider(0, _metrics(1.0), 0.0, snapshot)
        for step, metrics, objective in (
            (1, _metrics(1.0), 0.0),
            (2, _metrics(0.5), 0.1),
            (3, _metrics(math.nan), 0.0),
            (4, _metrics(0.0, force=math.inf), 0.0),
            (5, _metrics(0.0), math.nan),
        ):
            self.assertFalse(selector.consider(step, metrics, objective, snapshot))
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(selector.selected["candidate"]["step"], 0)

    def test_roundtrip_preserves_fallback_candidates(self):
        selector = EnergyCheckpointSelector(0.1)
        selector.consider(
            0, _metrics(1.0), 1.09, lambda: {"weight": torch.tensor([1.0])}
        )
        selector.consider(
            1, _metrics(1.1), 1.03, lambda: {"weight": torch.tensor([2.0])}
        )
        restored = EnergyCheckpointSelector(0.1)
        restored.load_state_dict(selector.state_dict())
        selector.candidates[1]["state"]["weight"].fill_(100)
        restored.consider(2, _metrics(2.0), 0.95, lambda: {})
        self.assertEqual(restored.selected["candidate"]["step"], 1)
        self.assertEqual(restored.selected["candidate"]["state"]["weight"].item(), 2.0)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            EnergyCheckpointSelector(0.2).load_state_dict(selector.state_dict())


class EnergySelectionConfigurationTests(unittest.TestCase):
    def test_yaml_options_and_path(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "mace_model": "mace.pt",
                        "train_file": "train.xyz",
                        "checkpoint": "model.pt",
                        "training": {
                            "energy_checkpoint_tolerance": 0.01,
                            "energy_checkpoint_constraint": "forces",
                            "energy_checkpoint_metric": "centered",
                        },
                    }
                )
            )
            config = TrainingConfig.from_namespace(
                parse_arguments(["--config", str(path)])
            )
            self.assertEqual(
                config.energy_checkpoint, Path(directory) / "model.energy.pt"
            )
            self.assertEqual(config.optimization.energy_checkpoint_tolerance, 0.01)
            self.assertEqual(config.optimization.energy_checkpoint_constraint, "forces")
            self.assertEqual(config.optimization.energy_checkpoint_metric, "centered")

    def test_disabled_by_default_and_invalid_settings(self):
        self.assertIsNone(
            TrainingConfig.from_namespace(train_arguments()).energy_checkpoint
        )
        for options in (
            ("--energy-checkpoint-tolerance", "-1", "--checkpoint", "model.pt"),
            ("--energy-checkpoint-tolerance", "nan", "--checkpoint", "model.pt"),
            ("--energy-checkpoint-tolerance", "inf", "--checkpoint", "model.pt"),
            ("--energy-checkpoint-tolerance", "0.01"),
            (
                "--energy-checkpoint-tolerance",
                "0.01",
                "--checkpoint",
                "model.pt",
                "--last-checkpoint",
                "model.energy.pt",
            ),
            (
                "--energy-checkpoint-tolerance",
                "0.01",
                "--checkpoint",
                "model.pt",
                "--resume",
                "model.energy.pt",
            ),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainingConfig.from_namespace(train_arguments(*options))


class EnergySelectionCLITests(unittest.TestCase):
    def test_secondary_weights_and_metrics_are_paired_for_frozen_and_joint(self):
        for regime, factory in (("frozen", _ToyResidual), ("joint", _ToyJointModel)):
            with self.subTest(regime=regime), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "mace.pt").write_bytes(b"test backbone")
                (root / "train.xyz").write_bytes(b"test data")
                args = train_arguments(
                    "--mace-model",
                    str(root / "mace.pt"),
                    "--train-file",
                    str(root / "train.xyz"),
                    "--checkpoint",
                    str(root / "model.pt"),
                    "--steps",
                    "6",
                    "--eval-interval",
                    "2",
                    "--learning-rate",
                    "0.1",
                    "--mace-training",
                    regime,
                    "--energy-checkpoint-tolerance",
                    "0.1",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float64",
                )
                sample = _sample()
                prepared = PreparedData(
                    samples=[sample],
                    train_samples=[sample],
                    validation_samples=[sample],
                    test_samples=[sample],
                    reference_cell=torch.eye(3, dtype=torch.float64),
                    train_cache_metadata={},
                    validation_cache_metadata=None,
                    test_cache_metadata=None,
                    train_cache_hit=False,
                    validation_cache_hit=False,
                    test_cache_hit=False,
                )
                model = factory()
                saved = []
                original_save = train_residual.save_training_checkpoint

                def save(path, payload):
                    saved.append((Path(path), payload))
                    original_save(path, payload)

                with (
                    redirect_stdout(io.StringIO()),
                    mock.patch.object(
                        train_residual, "parse_arguments", return_value=args
                    ),
                    mock.patch.object(
                        train_residual,
                        "load_mace_calculator",
                        return_value=SimpleNamespace(models=[None]),
                    ),
                    mock.patch.object(
                        train_residual, "prepare_data", return_value=prepared
                    ),
                    mock.patch.object(
                        train_residual, "build_training_model", return_value=model
                    ),
                    mock.patch.object(train_residual, "cache_frozen_targets"),
                    mock.patch.object(
                        train_residual, "save_training_checkpoint", side_effect=save
                    ),
                ):
                    train_residual.main()
                secondary = torch.load(root / "model.energy.pt", weights_only=False)
                last = torch.load(root / "model.last.pt", weights_only=False)
                primary = torch.load(root / "model.pt", weights_only=False)
                metadata = secondary["checkpoint_selection"]
                selected = next(
                    c
                    for c in last["energy_selector"]["candidates"]
                    if c["step"] == secondary["best_step"]
                )
                for key in ("residual_state_dict", "mace_state_dict"):
                    self.assertEqual(secondary[key], selected["state"][key])
                self.assertLessEqual(
                    metadata["selected_constraint"], metadata["constraint_limit"]
                )
                self.assertEqual(
                    secondary["evaluation_metrics"]["validation"],
                    metadata["validation_metrics"],
                )
                self.assertIn("held-out test", secondary["evaluation_metrics"])
                self.assertIsNone(secondary["spectral_diagnostic"])
                self.assertTrue((root / "model.energy.config.yaml").is_file())
                # The secondary file was already written before the final step.
                self.assertTrue(
                    any(
                        path.name == "model.energy.pt"
                        and payload["completed_steps"] < 6
                        for path, payload in saved
                    )
                )
                # Final secondary evaluation restores the primary model in memory.
                for name, value in primary["residual_state_dict"].items():
                    self.assertTrue(torch.equal(model.state_dict()[name], value))

                # An on-the-fly save may select older weights, not the live model.
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(99.0)
                result = OptimizationResult(
                    best_step=100,
                    best_validation_objective=100.0,
                    completed_steps=100,
                    stopped_early=False,
                    warmup_learning_rate=0.1,
                    final_learning_rate=0.1,
                )
                with redirect_stdout(io.StringIO()):
                    train_residual._save_checkpoint(
                        args,
                        TrainingConfig.from_namespace(args),
                        prepared,
                        result,
                        None,
                        model,
                        write_configuration=False,
                        energy_selection={"candidate": selected, "metadata": metadata},
                    )
                saved_old = torch.load(root / "model.energy.pt", weights_only=False)
                self.assertEqual(saved_old["best_step"], selected["step"])
                for key in ("residual_state_dict", "mace_state_dict"):
                    self.assertEqual(saved_old[key], secondary[key])
                self.assertTrue(all(p.item() == 99.0 for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
