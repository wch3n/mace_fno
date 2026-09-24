"""Restart parity, checkpoint selection, and compatibility checks."""

from __future__ import annotations

import io
import random
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import yaml
from mace_fno_test_helpers import FakeMACE, train_arguments
from test_trainer import _sample, _ToyJointModel, _ToyResidual

from mace_fno import MACEFNOResidual
from mace_fno.cli import train_residual
from mace_fno.cli.config import parse_arguments
from mace_fno.training import (
    PreparedData,
    TrainingConfig,
    evaluate_frozen_baseline,
    optimize_residual,
    save_training_checkpoint,
)
from mace_fno.training.resume import (
    input_fingerprints,
    load_resume_checkpoint,
    validate_resume_checkpoint,
)


class _Interrupted(Exception):
    pass


class _StochasticResidual(_ToyResidual):
    def forward(self, graph, **kwargs):
        output = super().forward(graph, **kwargs)
        if kwargs.get("training"):
            gain = 1.0 + random.random() + float(np.random.rand()) + torch.rand(())
            output = {name: value * gain for name, value in output.items()}
        return output


class _ProjectionResidual(torch.nn.Module):
    """Exercise the real output-warmup policy with two learned weights."""

    def __init__(self):
        super().__init__()
        self.source = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.long_range = torch.nn.Module()
        operator = self.long_range.field_operator = torch.nn.Module()
        operator.architecture = "nonlinear"
        operator.fno = torch.nn.Module()
        operator.fno.projection_output = torch.nn.Linear(1, 1, bias=False).double()

    def forward(self, graph, **kwargs):
        positions = graph["positions"]
        batch = graph["batch"]
        counts = positions.new_zeros(int(batch.max()) + 1).index_add(
            0,
            batch,
            positions.new_ones(batch.shape[0]),
        )
        scale = (
            self.source
            * self.long_range.field_operator.fno.projection_output.weight[0, 0]
        )
        return {
            "residual_energy": scale * counts,
            "residual_forces": scale * torch.ones_like(positions),
        }


class _Monitor:
    def __init__(self):
        self.history = []

    def evaluate_validation(self, model, **record):
        self.history.append(record)

    def write_history(self):
        pass


class TrainingResumeTests(unittest.TestCase):
    def configuration(self, *options):
        return TrainingConfig.from_namespace(
            train_arguments(
                "--steps",
                "12",
                "--eval-interval",
                "3",
                "--checkpoint-interval",
                "1",
                "--learning-rate",
                "0.1",
                "--energy-weight",
                "1",
                "--force-weight",
                "1",
                "--batch-size",
                "2",
                "--accumulation-steps",
                "2",
                *options,
            )
        )

    def run_training(
        self,
        factory,
        configuration,
        *,
        stop=None,
        resume=None,
        flat=False,
        samples=None,
    ):
        # Construction/setup consume a different random stream after restart.
        seed = 519 if resume is not None else 41
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = factory()
        samples = (
            deepcopy(samples)
            if samples is not None
            else (
                [_sample(0.0)] if flat else [_sample(x) for x in (-0.3, 0.2, 1.1, 0.4)]
            )
        )
        monitor = _Monitor()
        states = []
        best = []

        def latest(state):
            states.append(deepcopy(state))
            if state["completed_steps"] == stop:
                raise _Interrupted()

        def selected(current, result):
            best.append((deepcopy(current.state_dict()), result))

        with redirect_stdout(io.StringIO()):
            baseline = evaluate_frozen_baseline(
                model, samples, samples, [], configuration
            )
            try:
                result = optimize_residual(
                    model,
                    samples,
                    samples,
                    baseline,
                    configuration,
                    device=torch.device("cpu"),
                    spectral_monitor=monitor,
                    resume_state=resume,
                    last_checkpoint_callback=latest,
                    best_checkpoint_callback=selected,
                )
            except _Interrupted:
                result = None
        return model, result, states, best, monitor

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def roundtrip(self, state):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_training_checkpoint(path, state)
            return load_resume_checkpoint(path)

    def test_interrupted_training_matches_uninterrupted_bitwise(self):
        cases = (
            (_StochasticResidual, ()),
            (_ProjectionResidual, ("--output-warmup-steps", "3")),
            (_ToyJointModel, ("--mace-training", "joint", "--mace-warmup-steps", "3")),
        )
        for factory, options in cases:
            configuration = self.configuration(
                "--lr-scheduler",
                "plateau",
                "--lr-patience-evals",
                "0",
                *options,
            )
            full = self.run_training(factory, configuration)
            for stop in (2, 3, 7):  # before, at, and after unfreezing/validation
                with self.subTest(model=factory.__name__, stop=stop):
                    interrupted = self.run_training(factory, configuration, stop=stop)
                    saved = self.roundtrip(interrupted[2][-1])
                    resumed = self.run_training(factory, configuration, resume=saved)
                    self.assertEqual(full[1], resumed[1])
                    self.assert_nested_equal(
                        full[0].state_dict(), resumed[0].state_dict()
                    )
                    self.assert_nested_equal(full[2][-1], resumed[2][-1])
                    self.assertEqual(full[4].history, resumed[4].history)

    def test_metric_eqgino_force_training_with_random_interlacing_resumes(self):
        cell = 8 * torch.eye(3, dtype=torch.float64)
        sample = _sample()
        sample["data"].update(
            {
                "cell": cell.unsqueeze(0),
                "pbc": torch.ones((1, 3), dtype=torch.bool),
                "node_attrs": torch.ones((2, 1), dtype=torch.float64),
            }
        )
        for training in ("frozen", "joint"):
            with self.subTest(training=training):
                configuration = self.configuration(
                    "--steps",
                    "4",
                    "--eval-interval",
                    "2",
                    "--grid",
                    "6",
                    "--z-grid",
                    "6",
                    "--modes",
                    "2",
                    "--spatial-scheme",
                    "3d",
                    "--spectral-symmetry",
                    "metric_eqgino",
                    "--channels",
                    "2",
                    "--source-hidden-channels",
                    "4",
                    "--fno-hidden-channels",
                    "4",
                    "--fno-layers",
                    "1",
                    "--volume-interlacing",
                    "2",
                    "--interlacing-training",
                    "random",
                    "--mace-training",
                    training,
                    "--dtype",
                    "float64",
                    "--batch-size",
                    "1",
                    "--accumulation-steps",
                    "1",
                )

                def factory():
                    return MACEFNOResidual(
                        FakeMACE(),
                        (6, 6),
                        2,
                        (2, 2),
                        invariant_indices=[0, 2],
                        source_hidden_channels=4,
                        fno_hidden_channels=4,
                        fno_layers=1,
                        spatial_scheme="3d",
                        z_grid_size=6,
                        reference_cell=cell,
                        fno_spectral_symmetry="metric_eqgino",
                        mace_training=training,
                        fno_volume_interlacing=2,
                        fno_interlacing_training="random",
                    ).double()

                full = self.run_training(factory, configuration, samples=[sample])
                interrupted = self.run_training(
                    factory, configuration, samples=[sample], stop=1
                )
                resumed = self.run_training(
                    factory,
                    configuration,
                    samples=[sample],
                    resume=self.roundtrip(interrupted[2][-1]),
                )
                self.assertEqual(full[1], resumed[1])
                self.assert_nested_equal(full[0].state_dict(), resumed[0].state_dict())
                self.assert_nested_equal(full[2][-1], resumed[2][-1])

    def test_scheduler_and_patience_continue_without_reset(self):
        for options in (
            ("--early-stopping-patience-steps", "7"),
            (
                "--lr-scheduler",
                "plateau",
                "--lr-patience-evals",
                "0",
                "--lr-decay-factor",
                "0.5",
                "--minimum-learning-rate",
                "0.05",
                "--early-stopping-patience-evals",
                "2",
            ),
        ):
            with self.subTest(options=options):
                configuration = self.configuration(*options)
                full = self.run_training(_ToyResidual, configuration, flat=True)
                interrupted = self.run_training(
                    _ToyResidual, configuration, stop=3, flat=True
                )
                resumed = self.run_training(
                    _ToyResidual,
                    configuration,
                    flat=True,
                    resume=self.roundtrip(interrupted[2][-1]),
                )
                self.assertTrue(full[1].stopped_early)
                self.assertEqual(full[1], resumed[1])
                self.assert_nested_equal(full[2][-1], resumed[2][-1])
                stopped = self.run_training(
                    _ToyResidual,
                    configuration,
                    flat=True,
                    resume=resumed[2][-1],
                )
                self.assertEqual(stopped[1], resumed[1])

    def test_energy_selection_resumes_without_changing_primary_training(self):
        for factory, regime in ((_StochasticResidual, "frozen"), (_ToyJointModel, "joint")):
            for constraint in ("loss", "forces"):
                with self.subTest(regime=regime, constraint=constraint):
                    configuration = self.configuration(
                        "--checkpoint", "model.pt", "--mace-training", regime,
                        "--energy-checkpoint-tolerance", "0.1",
                        "--energy-checkpoint-constraint", constraint,
                    )
                    full = self.run_training(factory, configuration)
                    interrupted = self.run_training(factory, configuration, stop=7)
                    resumed = self.run_training(
                        factory, configuration, resume=self.roundtrip(interrupted[2][-1])
                    )
                    self.assert_nested_equal(full[2][-1], resumed[2][-1])
                    self.assert_nested_equal(full[1].energy_selection, resumed[1].energy_selection)
                    disabled = replace(configuration, optimization=replace(
                        configuration.optimization, energy_checkpoint_tolerance=None,
                    ))
                    original = self.run_training(factory, disabled)
                    self.assertEqual(replace(full[1], energy_selection=None), original[1])
                    self.assert_nested_equal(full[0].state_dict(), original[0].state_dict())
                    changed = replace(configuration, optimization=replace(
                        configuration.optimization, energy_checkpoint_tolerance=0.2,
                    ))
                    with self.assertRaisesRegex(ValueError, "settings differ"):
                        validate_resume_checkpoint(interrupted[2][-1], changed)

    def test_last_weights_are_not_overwritten_by_best_and_budget_can_extend(self):
        configuration = self.configuration("--steps", "6", "--learning-rate", "1")
        trained = self.run_training(_ToyResidual, configuration)
        state = trained[2][-1]
        self.assertLess(state["best_step"], state["completed_steps"])
        self.assertFalse(
            torch.equal(
                state["residual_state_dict"]["scale"],
                state["best_residual_state_dict"]["scale"],
            )
        )
        extended = replace(
            configuration, optimization=replace(configuration.optimization, steps=12)
        )
        continued = self.run_training(
            _ToyResidual, extended, resume=self.roundtrip(state)
        )
        full = self.run_training(_ToyResidual, extended)
        self.assertEqual(continued[1], full[1])
        self.assert_nested_equal(continued[2][-1], full[2][-1])

    def test_weights_only_checkpoints_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"residual_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "full training-state"):
                load_resume_checkpoint(path)

    def test_incompatible_settings_and_shorter_budget_are_rejected(self):
        configuration = self.configuration()
        state = self.run_training(_ToyResidual, configuration, stop=3)[2][-1]
        for option in (("--channels", "8"), ("--batch-size", "3"), ("--seed", "18")):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ValueError, "settings differ"),
            ):
                validate_resume_checkpoint(state, self.configuration(*option))
        with self.assertRaisesRegex(ValueError, "total budget"):
            validate_resume_checkpoint(state, self.configuration("--steps", "2"))

    def test_yaml_paths_defaults_and_collision_checks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "mace_model": "mace.pt",
                        "train_file": "train.xyz",
                        "checkpoint": "best.pt",
                        "resume": "old.last.pt",
                        "checkpoint_interval": 5,
                    }
                )
            )
            configuration = TrainingConfig.from_namespace(
                parse_arguments(["--config", str(path)])
            )
            self.assertEqual(
                configuration.runtime.last_checkpoint, root / "best.last.pt"
            )
            self.assertEqual(configuration.runtime.resume, root / "old.last.pt")
            self.assertEqual(configuration.runtime.checkpoint_interval, 5)
        for options in (
            ("--checkpoint", "same.pt", "--last-checkpoint", "same.pt"),
            ("--checkpoint", "same.pt", "--resume", "same.pt"),
            ("--checkpoint-interval", "-1"),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.configuration(*options)

    def test_source_input_hash_change_is_rejected(self):
        configuration = self.configuration()
        state = self.run_training(_ToyResidual, configuration, stop=3)[2][-1]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"original source")
            data = replace(configuration.data, mace_model=path, train_file=path)
            fingerprints = input_fingerprints(replace(configuration, data=data))
            state["input_fingerprints"] = fingerprints
            validate_resume_checkpoint(state, configuration, fingerprints=fingerprints)
            path.write_bytes(b"modified source")
            changed = input_fingerprints(replace(configuration, data=data))
            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                validate_resume_checkpoint(state, configuration, fingerprints=changed)

    def test_cli_writes_distinct_files_and_resumes_in_a_new_output_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mace_path = root / "mace.pt"
            data_path = root / "train.xyz"
            mace_path.write_bytes(b"test backbone reference")
            data_path.write_bytes(b"test data reference")
            sample = _sample()
            prepared = PreparedData(
                samples=[sample],
                train_samples=[sample],
                validation_samples=[sample],
                test_samples=[],
                reference_cell=torch.eye(3, dtype=torch.float64),
                train_cache_metadata={},
                validation_cache_metadata=None,
                test_cache_metadata=None,
                train_cache_hit=False,
                validation_cache_hit=False,
                test_cache_hit=False,
            )

            def invoke(best, steps, resume=None):
                options = [
                    "--mace-model",
                    str(mace_path),
                    "--train-file",
                    str(data_path),
                    "--checkpoint",
                    str(best),
                    "--steps",
                    str(steps),
                    "--eval-interval",
                    "2",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float64",
                ]
                if resume is not None:
                    options += ["--resume", str(resume)]
                args = train_arguments(*options)
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
                        train_residual,
                        "build_training_model",
                        side_effect=lambda *a, **kw: _ToyResidual(),
                    ),
                    mock.patch.object(train_residual, "cache_frozen_targets"),
                ):
                    train_residual.main()

            best = root / "first" / "best.pt"
            invoke(best, 4)
            latest = best.with_name("best.last.pt")
            initial_state = load_resume_checkpoint(latest)
            self.assertEqual(initial_state["completed_steps"], 4)
            self.assertEqual(
                initial_state["input_fingerprints"].keys(), {"mace_model", "train_file"}
            )
            with self.assertRaisesRegex(ValueError, "full training-state"):
                load_resume_checkpoint(best)
            new_best = root / "continued" / "best.pt"
            invoke(new_best, 8, resume=latest)
            new_state = load_resume_checkpoint(new_best.with_name("best.last.pt"))
            self.assertEqual(new_state["completed_steps"], 8)
            self.assertEqual(load_resume_checkpoint(latest)["completed_steps"], 4)
            self.assertTrue(new_best.with_suffix(".config.yaml").is_file())
            continuous_best = root / "continuous" / "best.pt"
            invoke(continuous_best, 8)
            continuous_state = load_resume_checkpoint(
                continuous_best.with_name("best.last.pt")
            )
            self.assert_nested_equal(
                new_state["optimizer_state_dict"],
                continuous_state["optimizer_state_dict"],
            )
            self.assert_nested_equal(
                new_state["residual_state_dict"],
                continuous_state["residual_state_dict"],
            )


if __name__ == "__main__":
    unittest.main()
