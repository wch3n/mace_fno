from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mace_fno.cli.config import parse_arguments
from mace_fno.cli.train_residual import _save_checkpoint
from mace_fno.training import (
    OptimizationResult,
    PreparedData,
    TrainingConfig,
    evaluate_frozen_baseline,
    evaluate_selected_model,
    optimize_residual,
)


class _ToyResidual(torch.nn.Module):
    """Minimal residual model satisfying the optimizer/evaluation contract."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, graph, **kwargs):
        del kwargs
        batch = graph["batch"]
        positions = graph["positions"]
        num_graphs = int(batch.max()) + 1
        counts = positions.new_zeros(num_graphs).index_add(
            0, batch, positions.new_ones(batch.shape[0])
        )
        return {
            "residual_energy": self.scale * counts,
            "residual_forces": self.scale * torch.ones_like(positions),
        }


def _sample(target: float = 0.5) -> dict[str, object]:
    positions = torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=torch.float64)
    forces = torch.full_like(positions, target)
    energy = torch.tensor((2.0 * target,), dtype=torch.float64)
    return {
        "data": {"positions": positions},
        "energy": energy,
        "forces": forces,
        "base_energy": torch.zeros_like(energy),
        "base_forces": torch.zeros_like(forces),
        "residual_energy": energy.clone(),
        "residual_forces": forces.clone(),
        "num_atoms": 2,
        "formula": "X2",
    }


class ResidualTrainerTests(unittest.TestCase):
    def _arguments(self, *options: str):
        return parse_arguments(
            [
                "--mace-model",
                "model.pt",
                "--train-file",
                "train.xyz",
                *options,
            ]
        )

    def test_optimizer_selects_and_restores_an_improved_residual(self) -> None:
        arguments = self._arguments(
            "--steps",
            "8",
            "--eval-interval",
            "1",
            "--learning-rate",
            "0.1",
            "--energy-weight",
            "1",
            "--force-weight",
            "1",
            "--evaluation-scope",
            "validation-test",
        )
        configuration = TrainingConfig.from_namespace(arguments)
        model = _ToyResidual()
        train_samples = [_sample()]
        validation_samples = [_sample()]

        with redirect_stdout(io.StringIO()):
            baseline = evaluate_frozen_baseline(
                model,
                train_samples,
                validation_samples,
                [],
                configuration,
            )
            result = optimize_residual(
                model,
                train_samples,
                validation_samples,
                baseline,
                configuration,
                device=torch.device("cpu"),
            )
            evaluate_selected_model(
                model,
                train_samples,
                validation_samples,
                [],
                configuration,
            )

        self.assertGreater(result.best_step, 0)
        self.assertEqual(result.completed_steps, 8)
        self.assertFalse(result.stopped_early)
        self.assertLess(result.best_validation_objective, 0.5)
        self.assertGreater(model.scale.item(), 0.0)

    def test_checkpoint_writer_preserves_resolved_model_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "residual.pt"
            arguments = self._arguments(
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
                "--checkpoint",
                str(checkpoint),
            )
            configuration = TrainingConfig.from_namespace(arguments)
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
            result = OptimizationResult(
                best_step=4,
                best_validation_objective=0.2,
                completed_steps=5,
                stopped_early=False,
                warmup_learning_rate=1.0e-3,
                final_learning_rate=5.0e-4,
            )
            with redirect_stdout(io.StringIO()):
                _save_checkpoint(
                    arguments,
                    configuration,
                    prepared,
                    result,
                    None,
                    _ToyResidual(),
                )

            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload["spatial_scheme"], "3d")
            self.assertEqual(payload["cell_mode"], "anisotropic")
            self.assertEqual(payload["n_modes"], (6, 8, 8))
            self.assertEqual(payload["spectral_symmetry"], "metric_eqgino")
            self.assertEqual(payload["best_step"], 4)
            self.assertTrue(checkpoint.with_suffix(".config.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
