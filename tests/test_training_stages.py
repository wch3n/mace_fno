"""Two-stage objectives, isolated selection, and backward-compatible resume."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import test_training_resume as resume_fixtures
import torch
import yaml
from mace_fno_test_helpers import train_arguments
from test_trainer import _sample, _ToyJointModel, _ToyResidual
from test_training_resume import _ProjectionResidual, _StochasticResidual

from mace_fno.cli import train_residual
from mace_fno.cli.config import parse_arguments
from mace_fno.training import TrainingConfig
from mace_fno.training import trainer as trainer_module
from mace_fno.training.evaluation import evaluate, validation_objective
from mace_fno.training.resume import resume_configuration, validate_resume_checkpoint


class TrainingStageTests(unittest.TestCase):
    def setUp(self):
        self.helper = resume_fixtures.TrainingResumeTests()

    def configuration(self, *options):
        return self.helper.configuration(
            "--stage-two-start-step", "5", "--stage-two-energy-scale", "0.1",
            "--lr-scheduler", "plateau", "--lr-patience-evals", "0", *options,
        )

    def test_yaml_and_old_mappings(self):
        legacy = vars(train_arguments())
        for name in list(legacy):
            if name.startswith("stage_two_"):
                legacy.pop(name)
        config = TrainingConfig.from_mapping(legacy)
        self.assertIsNone(config.optimization.stage_two_start_step)
        self.assertFalse(any(
            key.startswith("stage_two_")
            for key in resume_configuration(config)["optimization"]
        ))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(yaml.safe_dump({
                "mace_model": "mace.pt", "train_file": "train.xyz",
                "checkpoint": "model.pt",
                "training": {
                    "steps": 12, "stage_two_start_step": 5,
                    "stage_two_energy_scale": 0.1,
                },
            }))
            config = TrainingConfig.from_namespace(parse_arguments([
                "--config", str(path), "--stage-two-energy-scale", "0.2",
            ]))
            self.assertEqual(config.optimization.loss_settings(2)["energy_scale"], 0.2)
            self.assertEqual(config.optimization.loss_settings(2)["force_weight"], 10.0)
            self.assertEqual(config.stage_one_checkpoint.name, "model.stage1.pt")

    def test_invalid_schedules_and_output_collisions(self):
        cases = (
            ("--stage-two-energy-scale", "0.1"),
            ("--stage-two-start-step", "0"),
            ("--stage-two-start-step", "12"),
            ("--stage-two-start-step", "2", "--output-warmup-steps", "3"),
            ("--stage-two-start-step", "2", "--stage-two-energy-scale", "0"),
            ("--stage-two-start-step", "2", "--stage-two-force-weight", "nan"),
            ("--stage-two-start-step", "2", "--stage-two-force-scale", "inf"),
            ("--stage-two-start-step", "2", "--stage-two-energy-weight", "-1"),
            ("--stage-two-start-step", "2", "--stage-two-energy-weight", "0",
             "--stage-two-force-weight", "0"),
            ("--stage-two-start-step", "2", "--stage-two-learning-rate", "1e-20"),
            ("--stage-two-start-step", "2", "--stage-two-mace-learning-rate", "1e-5"),
            ("--stage-two-start-step", "2", "--checkpoint", "model.pt",
             "--last-checkpoint", "model.stage1.pt"),
            ("--stage-two-start-step", "2", "--checkpoint", "model.pt",
             "--init-from", "model.stage1.pt"),
            ("--stage-two-start-step", "2", "--checkpoint", "model.pt",
             "--energy-checkpoint-tolerance", "0.01",
             "--resume", "model.stage1.energy.pt"),
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.helper.configuration(*options)

    def test_exact_resume_on_both_sides_of_transition(self):
        for factory, options in (
            (_StochasticResidual, ()),
            (_ProjectionResidual, ("--output-warmup-steps", "5")),
            (_ToyJointModel, (
                "--mace-training", "joint", "--mace-warmup-steps", "5",
                "--stage-two-mace-learning-rate", "1e-7",
            )),
        ):
            for constraint in ("loss", "forces"):
                config = self.configuration(
                    "--checkpoint", "unused.pt", "--energy-checkpoint-tolerance", "0.1",
                    "--energy-checkpoint-constraint", constraint,
                    "--stage-two-learning-rate", "0.02", *options,
                )
                full = self.helper.run_training(factory, config)
                self.assertEqual(full[1].optimization_stage, 2)
                self.assertEqual(full[1].stage_start_step, 5)
                self.assertGreaterEqual(full[1].best_step, 5)
                for step, stage in ((2, 1), (5, 1), (5, 2), (7, 2)):
                    with self.subTest(model=factory.__name__, step=step, stage=stage,
                                      constraint=constraint):
                        state = next(s for s in full[2] if s["completed_steps"] == step
                                     and s["optimization_stage"] == stage)
                        resumed = self.helper.run_training(
                            factory, config, resume=self.helper.roundtrip(state),
                        )
                        self.helper.assert_nested_equal(full[0].state_dict(), resumed[0].state_dict())
                        self.helper.assert_nested_equal(full[2][-1], resumed[2][-1])
                        self.helper.assert_nested_equal(full[1].__dict__, resumed[1].__dict__)
                        self.assertEqual(full[4].history, resumed[4].history)

    def test_transition_preserves_weights_and_adam_but_resets_selection(self):
        config = self.configuration(
            "--checkpoint", "unused.pt", "--energy-checkpoint-tolerance", "0.1",
        )
        full = self.helper.run_training(_ToyResidual, config)
        before, after = [s for s in full[2] if s["completed_steps"] == 5]
        for key in ("residual_state_dict", "optimizer_state_dict", "sample_generator_state"):
            self.helper.assert_nested_equal(before[key], after[key])
        self.assertEqual(after["scheduler_state_dict"]["best"], float("inf"))
        self.assertEqual(after["scheduler_state_dict"]["num_bad_epochs"], 0)
        self.assertEqual(after["best_step"], 5)
        self.assertEqual(after["evaluations_at_minimum_lr"], 0)
        self.assertTrue(all(c["step"] == 5 for c in after["energy_selector"]["candidates"]))
        # At identical weights the new objective uses the new energy scale.
        model = _ToyResidual()
        model.load_state_dict(after["residual_state_dict"])
        metrics = evaluate(model, [_sample(x) for x in (-0.3, 0.2, 1.1, 0.4)])
        self.assertAlmostEqual(after["best_objective"], validation_objective(
            metrics, **config.optimization.loss_settings(2),
        ))
        self.assertNotEqual(before["best_objective"], after["best_objective"])
        # The non-aligned stage boundary receives its own validation.
        self.assertIn(5, [entry["step"] for entry in full[4].history])

    def test_minibatch_objective_changes_after_exactly_n_updates(self):
        config = self.configuration(
            "--accumulation-steps", "1", "--lr-scheduler", "none",
            "--stage-two-force-scale", "2", "--stage-two-energy-weight", "3",
            "--stage-two-force-weight", "4", "--stage-two-learning-rate", "0.02",
        )
        with mock.patch.object(
            trainer_module, "energy_force_loss", wraps=trainer_module.energy_force_loss,
        ) as loss:
            full = self.helper.run_training(_ToyResidual, config)
        self.assertEqual(loss.call_count, 12)
        for update, call in enumerate(loss.call_args_list, start=1):
            self.assertEqual(call.kwargs, config.optimization.loss_settings(1 if update <= 5 else 2))
        self.assertEqual(full[1].final_learning_rate, 0.02)

    def test_real_fno_force_gradients_and_random_interlacing_resume(self):
        original = self.helper.configuration
        with mock.patch.object(self.helper, "configuration", side_effect=lambda *options:
            original(*options, "--stage-two-start-step", "2",
                     "--stage-two-energy-scale", "0.2")):
            self.helper.test_metric_eqgino_force_training_with_random_interlacing_resumes()

    def test_patience_does_not_skip_stage_two(self):
        for options in (
            ("--early-stopping-patience-steps", "1"),
            ("--early-stopping-patience-evals", "1", "--minimum-learning-rate", "0.1"),
        ):
            config = self.configuration(*options)
            result = self.helper.run_training(_ToyResidual, config, flat=True)[1]
            self.assertTrue(result.stopped_early)
            self.assertEqual(result.optimization_stage, 2)
            self.assertEqual(result.completed_steps, 6)

    def test_old_single_stage_resume_and_changed_contract_rejection(self):
        config = self.helper.configuration()
        full = self.helper.run_training(_StochasticResidual, config)
        state = self.helper.roundtrip(full[2][4])
        self.assertNotIn("optimization_stage", state)
        validate_resume_checkpoint(state, config)
        resumed = self.helper.run_training(_StochasticResidual, config, resume=state)
        self.helper.assert_nested_equal(full[2][-1], resumed[2][-1])
        with self.assertRaisesRegex(ValueError, "settings differ"):
            validate_resume_checkpoint(state, self.configuration())
        config = self.configuration()
        state = self.helper.run_training(_ToyResidual, config)[2][-1]
        for key in ("optimization_stage", "stage_start_step", "active_loss_settings"):
            broken = deepcopy(state)
            broken.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "resume checkpoint"):
                validate_resume_checkpoint(broken, config)

    def test_cli_saves_separate_stage_files_with_active_objective(self):
        for regime, factory in (("frozen", _ToyResidual), ("joint", _ToyJointModel)):
            with self.subTest(regime=regime), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "mace.pt").write_bytes(b"backbone")
                (root / "train.xyz").write_bytes(b"data")
                args = train_arguments(
                    "--mace-model", str(root / "mace.pt"),
                    "--train-file", str(root / "train.xyz"),
                    "--checkpoint", str(root / "model.pt"),
                    "--steps", "8", "--eval-interval", "2",
                    "--learning-rate", "0.1", "--mace-training", regime,
                    "--stage-two-start-step", "3", "--stage-two-energy-scale", "0.2",
                    "--energy-checkpoint-tolerance", "0.1", "--device", "cpu",
                )
                config = TrainingConfig.from_namespace(args)
                sample = _sample()
                prepared = SimpleNamespace(
                    samples=[sample], train_samples=[sample], validation_samples=[sample],
                    test_samples=[sample], reference_cell=torch.eye(3, dtype=torch.float64),
                )
                model = factory()
                with (
                    redirect_stdout(io.StringIO()),
                    mock.patch.object(train_residual, "parse_arguments", return_value=args),
                    mock.patch.object(train_residual, "load_mace_calculator",
                                      return_value=SimpleNamespace(models=[None])),
                    mock.patch.object(train_residual, "prepare_data", return_value=prepared),
                    mock.patch.object(train_residual, "build_training_model", return_value=model),
                    mock.patch.object(train_residual, "cache_frozen_targets"),
                ):
                    train_residual.main()
                for filename, stage in (
                    ("model.stage1.pt", 1), ("model.stage1.energy.pt", 1),
                    ("model.pt", 2), ("model.energy.pt", 2),
                ):
                    payload = torch.load(root / filename, weights_only=False)
                    self.assertEqual(payload["optimization_stage"], stage)
                    self.assertEqual(payload["energy_scale"], 1.0 if stage == 1 else 0.2)
                    self.assertEqual(payload["training_configuration"]["energy_scale"], 1.0)
                    if stage == 2:
                        self.assertGreaterEqual(payload["best_step"], 3)
                        self.assertIn("held-out test", payload["evaluation_metrics"])
                    model.load_state_dict({
                        **payload["residual_state_dict"],
                        **{f"backbone.mace_model.{name}": value
                           for name, value in (payload["mace_state_dict"] or {}).items()},
                    })
                    metrics = evaluate(model, [sample])
                    self.assertAlmostEqual(payload["best_validation_objective"],
                                           validation_objective(metrics, **config.optimization.loss_settings(stage)))


if __name__ == "__main__":
    unittest.main()
