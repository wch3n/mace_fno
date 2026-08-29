import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mace_fno.training import (
    batch_graphs,
    configure_output_projection_warmup,
    finish_output_projection_warmup,
    has_reference_labels,
    initialize_scaled_residual_output,
    initialize_zero_residual,
    load_or_create_samples,
    reference_energy,
    reference_forces,
    sample_cache_metadata,
    save_sample_cache,
    split_samples,
)


class _WarmupToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.source_head = torch.nn.Linear(2, 3, bias=False)
        self.long_range = torch.nn.Module()
        self.long_range.field_operator = torch.nn.Module()
        self.long_range.field_operator.architecture = "nonlinear"
        self.long_range.field_operator.fno = torch.nn.Module()
        self.long_range.field_operator.fno.hidden = torch.nn.Linear(
            3, 4, bias=False
        )
        self.long_range.field_operator.fno.projection_output = torch.nn.Linear(
            4, 1, bias=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.source_head(inputs)
        hidden = self.long_range.field_operator.fno.hidden(features)
        return self.long_range.field_operator.fno.projection_output(hidden)


class ReferenceLabelTests(unittest.TestCase):
    def test_small_output_initialization_enables_immediate_upstream_gradients(
        self,
    ) -> None:
        torch.manual_seed(19)
        template = _WarmupToy()
        state = template.state_dict()

        zero_model = _WarmupToy()
        zero_model.load_state_dict(state)
        initialize_zero_residual(zero_model)
        zero_loss = (zero_model(torch.ones((2, 2))) - 1.0).square().mean()
        zero_loss.backward()
        torch.testing.assert_close(
            zero_model.source_head.weight.grad,
            torch.zeros_like(zero_model.source_head.weight.grad),
            atol=0.0,
            rtol=0.0,
        )

        scaled_model = _WarmupToy()
        scaled_model.load_state_dict(state)
        original_projection = (
            scaled_model.long_range.field_operator.fno.projection_output.weight
            .detach()
            .clone()
        )
        initialize_scaled_residual_output(scaled_model, scale=0.1)
        torch.testing.assert_close(
            scaled_model.long_range.field_operator.fno.projection_output.weight,
            0.1 * original_projection,
        )
        scaled_loss = (scaled_model(torch.ones((2, 2))) - 1.0).square().mean()
        scaled_loss.backward()
        self.assertGreater(scaled_model.source_head.weight.grad.abs().max(), 0.0)
        self.assertGreater(
            scaled_model.long_range.field_operator.fno.hidden.weight.grad.abs().max(),
            0.0,
        )

    def test_output_projection_warmup_freezes_then_unfreezes_residual(self) -> None:
        model = _WarmupToy()
        parameters, projection_parameters = configure_output_projection_warmup(
            model, warmup_steps=2
        )
        self.assertEqual(len(projection_parameters), 1)
        self.assertTrue(projection_parameters[0].requires_grad)
        self.assertFalse(model.source_head.weight.requires_grad)
        self.assertFalse(model.long_range.field_operator.fno.hidden.weight.requires_grad)
        self.assertFalse(model.backbone.weight.requires_grad)

        optimizer = torch.optim.Adam(parameters, lr=1.0e-2)
        model(torch.ones((2, 2))).square().mean().backward()
        optimizer.step()
        self.assertIsNotNone(projection_parameters[0].grad)
        self.assertIsNone(model.source_head.weight.grad)

        optimizer.zero_grad(set_to_none=True)
        finish_output_projection_warmup(parameters)
        model(torch.ones((2, 2))).square().mean().backward()
        optimizer.step()
        self.assertIsNotNone(model.source_head.weight.grad)
        self.assertIsNotNone(
            model.long_range.field_operator.fno.hidden.weight.grad
        )
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertEqual(optimizer.state[projection_parameters[0]]["step"], 2)

    def test_zero_warmup_preserves_original_trainability(self) -> None:
        model = _WarmupToy()
        parameters, projection_parameters = configure_output_projection_warmup(
            model, warmup_steps=0
        )
        self.assertEqual(projection_parameters, [])
        self.assertTrue(model.source_head.weight.requires_grad)
        self.assertTrue(model.long_range.field_operator.fno.hidden.weight.requires_grad)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertEqual(
            sum(parameter.numel() for parameter in parameters),
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )

    def test_custom_extended_xyz_fields(self) -> None:
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.info["energy_dft"] = -1.25
        atoms.arrays["forces_dft"] = np.array([[0.1, 0.2, 0.3]])

        self.assertTrue(has_reference_labels(atoms, "energy_dft", "forces_dft"))
        self.assertEqual(reference_energy(atoms, "energy_dft"), -1.25)
        np.testing.assert_allclose(
            reference_forces(atoms, "forces_dft"), [[0.1, 0.2, 0.3]]
        )

    def test_ase_canonical_calculator_results(self) -> None:
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-2.5,
            forces=np.array([[0.4, 0.5, 0.6]]),
        )

        self.assertTrue(has_reference_labels(atoms, "energy", "forces"))
        self.assertEqual(reference_energy(atoms, "energy"), -2.5)
        np.testing.assert_allclose(reference_forces(atoms, "forces"), [[0.4, 0.5, 0.6]])

    def test_explicit_validation_indices_preserve_dataset_order(self) -> None:
        samples = [{"index": index} for index in range(6)]
        train, validation = split_samples(
            samples,
            validation_fraction=0.2,
            seed=17,
            validation_indices=[4, 1],
        )

        self.assertEqual([sample["index"] for sample in train], [0, 2, 3, 5])
        self.assertEqual([sample["index"] for sample in validation], [1, 4])

    def test_batch_graphs_offsets_edges_and_constructs_membership(self) -> None:
        dtype = torch.float64
        first = {
            "positions": torch.zeros((2, 3), dtype=dtype),
            "edge_index": torch.tensor(((0, 1), (1, 0)), dtype=torch.long),
            "cell": torch.eye(3, dtype=dtype),
            "batch": torch.zeros(2, dtype=torch.long),
            "ptr": torch.tensor((0, 2), dtype=torch.long),
        }
        second = {
            "positions": torch.ones((1, 3), dtype=dtype),
            "edge_index": torch.tensor(((0,), (0,)), dtype=torch.long),
            "cell": 2.0 * torch.eye(3, dtype=dtype),
            "batch": torch.zeros(1, dtype=torch.long),
            "ptr": torch.tensor((0, 1), dtype=torch.long),
        }

        combined = batch_graphs([first, second])
        self.assertEqual(combined["positions"].shape, (3, 3))
        self.assertEqual(combined["cell"].shape, (6, 3))
        self.assertEqual(combined["batch"].tolist(), [0, 0, 1])
        self.assertEqual(combined["ptr"].tolist(), [0, 2, 3])
        self.assertEqual(combined["edge_index"][:, -1].tolist(), [2, 2])

    def test_sample_cache_round_trip_avoids_graph_reconstruction(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.xyz"
            model = root / "mace.model"
            cache = root / "samples.pt"
            source.write_text("placeholder")
            model.write_bytes(b"checkpoint")
            metadata = sample_cache_metadata(
                filename=source,
                mace_model=model,
                energy_key="energy",
                forces_key="forces",
                dtype=torch.float64,
                num_atoms=None,
                allow_periodic_z=False,
                skip_cell_mismatch=False,
            )
            samples = [
                {
                    "data": {"positions": torch.zeros((1, 3))},
                    "energy": torch.tensor([-1.0], dtype=torch.float64),
                    "forces": torch.zeros((1, 3), dtype=torch.float64),
                    "num_atoms": 1,
                    "formula": "H",
                }
            ]
            reference_cell = torch.eye(3, dtype=torch.float64)
            save_sample_cache(cache, metadata, samples, reference_cell)

            loaded, loaded_cell, _, cache_hit = load_or_create_samples(
                None,
                source,
                "energy",
                "forces",
                torch.float64,
                None,
                False,
                False,
                model,
                cache,
                False,
            )
            self.assertTrue(cache_hit)
            self.assertEqual(loaded[0]["formula"], "H")
            torch.testing.assert_close(loaded_cell, reference_cell)


if __name__ == "__main__":
    unittest.main()
