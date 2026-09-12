"""Physics and end-to-end checks for in-plane metric-aware slab EqGINO."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from ase import Atoms
from mace_fno_test_helpers import FakeIrreps, FakeMACE, FakeProduct, batch_data
from torch import nn

from mace_fno import (
    LearnedSlabParticleMeshLongRange,
    MACEFNOCalculator,
    MetricEqGINOSpectralConv2D,
    SlabFNOFieldOperator2D,
)
from mace_fno.cli.config import parse_arguments
from mace_fno.training import build_mace_fno_model
from mace_fno.training.checkpoint import training_checkpoint_payload
from mace_fno.training.configuration import TrainingConfig
from mace_fno.training.setup import PreparedData, build_training_model
from mace_fno.training.trainer import OptimizationResult

DTYPE = torch.float64


def transform(field, index):
    if index >= 4:
        field = field.flip(-1)
    return torch.rot90(field, index % 4, (-2, -1))


class SlabEqGINOSpectralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(739)
        self.cell = torch.diag(torch.tensor([8.0, 8.0, 22.0], dtype=DTYPE))

    def test_square_matches_direct_shell_table_convolution(self):
        layer = MetricEqGINOSpectralConv2D(
            4, 6, (3, 3), groups=2, reference_length=8
        ).double()
        field = torch.randn(2, 4, 3, 10, 10, dtype=DTYPE)
        field_k = torch.fft.fft2(field)
        expected_k = field_k.new_zeros(2, 6, 3, 10, 10)
        shells = layer.shell_spline.knots.long().tolist()
        # Independent integer-shell contraction; no physical-k or spline code.
        for nx in range(-2, 3):
            for ny in range(-2, 3):
                weights = layer.radial_weight[..., shells.index(nx * nx + ny * ny)]
                block = field_k[..., nx % 10, ny % 10].reshape(2, 2, 2, 3)
                transformed = torch.einsum(
                    "bgiz,gio->bgoz", block, weights.to(block.dtype)
                )
                expected_k[..., nx % 10, ny % 10] = transformed.reshape(2, 6, 3)
        torch.testing.assert_close(
            layer(field, self.cell),
            torch.fft.ifft2(expected_k).real,
            atol=2e-14,
            rtol=2e-14,
        )

    def test_physical_metric_and_third_vector_independence(self):
        layer = MetricEqGINOSpectralConv2D(1, 1, (3, 3), reference_length=8).double()
        cell = self.cell.clone()
        cell[1] = torch.tensor([2.0, 10.0, 0.0], dtype=DTYPE)
        modes = layer.mode_xy.to(DTYPE)
        # kx=2*pi*nx/8, ky=2*pi*(ny-2*nx/8)/10 for this skew plane.
        expected = (2 * math.pi) ** 2 * (
            (modes[..., 0] / 8) ** 2 + ((modes[..., 1] - modes[..., 0] / 4) / 10) ** 2
        )
        torch.testing.assert_close(
            layer._physical_squared_wavevectors(cell[None])[0], expected
        )
        field = torch.randn(1, 1, 2, 8, 10, dtype=DTYPE)
        rotation = torch.tensor([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]], dtype=DTYPE)
        tilted = cell @ rotation.T
        tilted[2] += tilted[0] + 2 * tilted[1]
        torch.testing.assert_close(
            layer(field, tilted), layer(field, cell), atol=2e-14, rtol=2e-14
        )
        cell.requires_grad_()
        gradient = torch.autograd.grad(layer(field, cell).square().sum(), cell)[0]
        torch.testing.assert_close(
            gradient[2], torch.zeros(3, dtype=DTYPE), atol=0, rtol=0
        )
        self.assertGreater(gradient[:2].abs().max().item(), 0)

    def test_batched_cells_and_rectangular_axis_exchange(self):
        layer = MetricEqGINOSpectralConv2D(2, 2, (2, 2), reference_length=8).double()
        cell = self.cell.clone()
        cell[1, 0], cell[1, 1] = 1.7, 10.0
        field = torch.randn(2, 2, 3, 8, 10, dtype=DTYPE)
        cells = torch.stack([cell, self.cell])
        torch.testing.assert_close(
            layer(field, cells),
            torch.cat([layer(field[i : i + 1], cells[i]) for i in range(2)]),
        )
        swapped_cell = cell[[1, 0, 2]]
        torch.testing.assert_close(
            layer(field[:1].transpose(-1, -2), swapped_cell),
            layer(field[:1], cell).transpose(-1, -2),
            atol=2e-14,
            rtol=2e-14,
        )

    def test_spectral_layer_never_couples_z_and_preserves_real_even_modes(self):
        layer = MetricEqGINOSpectralConv2D(1, 1, (2, 2), reference_length=8).double()
        field = torch.zeros(1, 1, 4, 8, 8, dtype=DTYPE)
        x = torch.arange(8, dtype=DTYPE) * (2 * math.pi / 8)
        field[:, :, 0] = x.cos()[:, None]
        weight = layer.radial_weight[0, 0, 0, 1]
        torch.testing.assert_close(
            layer(field, self.cell), weight * field, atol=1e-14, rtol=1e-14
        )
        self.assertEqual(layer(field, self.cell)[:, :, 1:].abs().max().item(), 0)
        field[:, :, 0] = x.sin()[:, None]
        torch.testing.assert_close(
            layer(field, self.cell), weight * field, atol=1e-14, rtol=1e-14
        )

    def test_field_and_metric_first_and_second_derivatives(self):
        layer = MetricEqGINOSpectralConv2D(1, 1, (2, 2), reference_length=8).double()
        field = torch.randn(1, 1, 1, 4, 4, dtype=DTYPE, requires_grad=True)
        cell = self.cell.clone()
        cell[1, 0], cell[1, 1] = 0.7, 9.3
        cell.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(layer, (field, cell), fast_mode=True))
        self.assertTrue(
            torch.autograd.gradgradcheck(layer, (field, cell), fast_mode=True)
        )

    def test_nonlinear_d4_and_training_evaluation_agree(self):
        for mixing in ("local", "global"):
            for parameterization in ("shell_spline", "radial_mlp"):
                with self.subTest(mixing=mixing, parameterization=parameterization):
                    operator = SlabFNOFieldOperator2D(
                        2,
                        5,
                        (3, 3),
                        hidden_channels=4,
                        n_layers=2,
                        z_mixing=mixing,
                        spectral_symmetry="metric_eqgino",
                        spectral_groups=2,
                        metric_parameterization=parameterization,
                        metric_reference_length=8,
                    ).double()
                    field = torch.randn(2, 2, 5, 10, 10, dtype=DTYPE)
                    expected = operator(field, self.cell)
                    for index in range(8):
                        torch.testing.assert_close(
                            operator(transform(field, index), self.cell),
                            transform(expected, index),
                            atol=2e-14,
                            rtol=2e-14,
                        )
                    operator.eval()
                    torch.testing.assert_close(
                        operator(field, self.cell), expected, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        operator(field[0], self.cell), expected[0]
                    )
                    torch.testing.assert_close(
                        operator(field.roll((1, 2), (-2, -1)), self.cell),
                        expected.roll((1, 2), (-2, -1)),
                        atol=2e-14,
                        rtol=2e-14,
                    )

    def test_invalid_settings_and_cells_fail_explicitly(self):
        for settings in (
            {"architecture": "linear"},
            {"planar_symmetry": "d4"},
            {"spectral_groups": 3},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                SlabFNOFieldOperator2D(
                    2,
                    4,
                    (2, 2),
                    hidden_channels=4,
                    spectral_symmetry="metric_eqgino",
                    **settings,
                )
        layer = MetricEqGINOSpectralConv2D(1, 1, (2, 2)).double()
        field = torch.zeros(2, 1, 3, 8, 8, dtype=DTYPE)
        for cell in (
            torch.zeros_like(self.cell),
            self.cell[None],
            self.cell.float(),
            self.cell * float("nan"),
        ):
            with self.subTest(shape=cell.shape), self.assertRaises(ValueError):
                layer(field, cell)
        with self.assertRaises(TypeError):
            layer(field, self.cell.long())
        with self.assertRaisesRegex(ValueError, "cell"):
            SlabFNOFieldOperator2D(
                1, 3, (2, 2), spectral_symmetry="metric_eqgino"
            ).double()(field)
        with self.assertRaisesRegex(ValueError, "mode count"):
            layer(torch.zeros(1, 1, 2, 3, 8, dtype=DTYPE), self.cell)


class SlabEqGINOEnergyTests(unittest.TestCase):
    def test_force_finite_difference_symmetry_and_backpropagation(self):
        torch.manual_seed(713)
        cell = torch.diag(torch.tensor([8.0, 8.0, 20.0], dtype=DTYPE))
        cell[2, 0] = 3.0
        positions = torch.tensor(
            [[1.13, 2.24, 8.8], [5.47, 3.12, 10.2], [3.75, 6.54, 9.6]],
            dtype=DTYPE,
            requires_grad=True,
        )
        sources = torch.tensor([[1.0, -0.2], [-0.4, 0.7], [-0.6, -0.5]], dtype=DTYPE)
        for interlacing in (1, 2):
            with self.subTest(interlacing=interlacing):
                model = LearnedSlabParticleMeshLongRange(
                    (6, 8, 8),
                    10.0,
                    2,
                    (2, 2),
                    hidden_channels=4,
                    n_layers=2,
                    z_mixing="global",
                    lateral_interlacing=interlacing,
                    spectral_symmetry="metric_eqgino",
                    spectral_groups=2,
                    metric_reference_length=8,
                ).double()
                energy = model(positions, sources, cell)
                forces = -torch.autograd.grad(
                    energy.sum(), positions, create_graph=True
                )[0]
                h = 1e-4
                displaced = torch.zeros_like(positions)
                displaced[0, 0] = h
                finite_force = -(
                    model(positions + displaced, sources, cell)
                    - model(positions - displaced, sources, cell)
                ) / (2 * h)
                torch.testing.assert_close(
                    finite_force.squeeze(), forces[0, 0], atol=1e-9, rtol=1e-6
                )
                torch.testing.assert_close(
                    model(positions + positions.new_tensor([0, 0, 2.7]), sources, cell),
                    energy,
                    atol=2e-13,
                    rtol=2e-13,
                )
                self.assertLess(abs(forces[:, 2].sum().item()), 1e-12)
                rotation = positions.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                rotated = (positions @ rotation.T).detach().requires_grad_()
                rotated_energy = model(rotated, sources, cell)
                rotated_forces = -torch.autograd.grad(rotated_energy.sum(), rotated)[0]
                torch.testing.assert_close(
                    rotated_energy, energy, atol=2e-13, rtol=2e-13
                )
                torch.testing.assert_close(
                    rotated_forces, forces @ rotation.T, atol=2e-13, rtol=2e-13
                )
                forces.square().sum().backward()
                spectral = model.field_operator.fno.blocks[0].spectral
                self.assertIsNotNone(spectral.radial_weight.grad)
                self.assertGreater(spectral.radial_weight.grad.abs().max().item(), 0)


class SlabEqGINOIntegrationTests(unittest.TestCase):
    def _configuration(self, path, training, parameterization="shell_spline"):
        document = {
            "mace_model": "local.model",
            "train_file": "train.xyz",
            "model": {
                "spatial_scheme": "2.5d",
                "grid": 8,
                "modes": 2,
                "z_grid": 6,
                "z_extent": 10.0,
                "z_mixing": "global",
                "channels": 2,
                "source_hidden_channels": 7,
                "fno_hidden_channels": 4,
                "fno_layers": 1,
                "spectral_symmetry": "metric_eqgino",
                "spectral_groups": 2,
                "metric_hidden_channels": 5,
                "metric_parameterization": parameterization,
                "lateral_interlacing": 2,
                "planar_symmetry": "none",
            },
            "training": {
                "mace_training": training,
                "random_residual_initialization": True,
            },
        }
        path.write_text(yaml.safe_dump(document))
        return TrainingConfig.from_namespace(parse_arguments(["--config", str(path)]))

    @staticmethod
    def _backbone():
        mace = FakeMACE()
        mace.products = nn.ModuleList([FakeProduct(FakeIrreps([(4, 0, 1)]))])
        return mace

    def test_yaml_training_checkpoint_roundtrip_and_ase(self):
        torch.manual_seed(741)
        for training in ("frozen", "joint"):
            for parameterization in ("shell_spline", "radial_mlp"):
                with (
                    self.subTest(training=training, parameterization=parameterization),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    configuration = self._configuration(
                        Path(directory) / "train.yaml", training, parameterization
                    )
                    reference = batch_data()["cell"][0]
                    model = build_training_model(
                        self._backbone(),
                        configuration,
                        reference,
                        device=torch.device("cpu"),
                        dtype=DTYPE,
                    )
                    output = model(batch_data(), training=True, compute_force=True)
                    before = model.backbone.mace_model.local_scale.detach().clone()
                    optimizer = torch.optim.Adam(
                        [p for p in model.parameters() if p.requires_grad], lr=1e-3
                    )
                    (
                        output["forces"].square().mean()
                        + output["energy"].square().mean()
                    ).backward()
                    spectral = model.long_range.field_operator.fno.blocks[0].spectral
                    self.assertTrue(
                        all(
                            p.grad is not None and torch.isfinite(p.grad).all()
                            for p in spectral.parameters()
                        )
                    )
                    optimizer.step()
                    after = model.backbone.mace_model.local_scale.detach()
                    self.assertEqual(torch.equal(before, after), training == "frozen")
                    prepared = PreparedData(
                        samples=[],
                        train_samples=[],
                        validation_samples=[],
                        test_samples=[],
                        reference_cell=reference,
                        train_cache_metadata={},
                        validation_cache_metadata=None,
                        test_cache_metadata=None,
                        train_cache_hit=False,
                        validation_cache_hit=False,
                        test_cache_hit=False,
                    )
                    result = OptimizationResult(
                        best_step=1,
                        best_validation_objective=0.0,
                        completed_steps=1,
                        stopped_early=False,
                        warmup_learning_rate=0,
                        final_learning_rate=1e-3,
                    )
                    payload = training_checkpoint_payload(
                        configuration,
                        prepared,
                        result,
                        model,
                        effective_configuration={},
                    )
                    self.assertEqual(payload["spectral_symmetry"], "metric_eqgino")
                    self.assertEqual(payload["spectral_groups"], 2)
                    restored = build_mace_fno_model(
                        payload, self._backbone(), dtype=DTYPE
                    )
                    expected = model(batch_data(), compute_force=True)
                    actual = restored(batch_data(), compute_force=True)
                    for key in ("energy", "forces"):
                        torch.testing.assert_close(
                            actual[key], expected[key], atol=1e-12, rtol=1e-12
                        )
                    if parameterization == "shell_spline":
                        expected_k2 = (2 * math.pi) ** 2 / 90.0
                        self.assertAlmostEqual(
                            spectral.reference_wavenumber_squared.item(),
                            expected_k2,
                            places=14,
                        )

                    def converter(atoms):
                        graph = batch_data()
                        graph["positions"] = torch.tensor(atoms.positions, dtype=DTYPE)
                        graph["cell"] = torch.tensor(atoms.cell.array, dtype=DTYPE)[
                            None
                        ]
                        graph["batch"] = torch.zeros(3, dtype=torch.long)
                        graph["ptr"] = torch.tensor([0, 3])
                        graph["node_attrs"] = graph["node_attrs"][:3]
                        return graph

                    atoms = Atoms(
                        "H3",
                        positions=batch_data()["positions"][:3].numpy(),
                        cell=reference.numpy(),
                        pbc=[True, True, False],
                    )
                    atoms.calc = MACEFNOCalculator(
                        model=restored, graph_converter=converter
                    )
                    direct = restored(converter(atoms), compute_force=True)
                    self.assertAlmostEqual(
                        atoms.get_potential_energy(), direct["energy"].item(), places=12
                    )
                    torch.testing.assert_close(
                        torch.tensor(atoms.get_forces()), direct["forces"]
                    )

    def test_yaml_rejects_linear_or_ensemble_slab_eqgino(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            self._configuration(path, "frozen")
            for option in (
                ["--architecture", "linear"],
                ["--planar-symmetry", "d4"],
                ["--spectral-groups", "3"],
            ):
                with self.subTest(option=option), self.assertRaises(ValueError):
                    TrainingConfig.from_namespace(
                        parse_arguments(["--config", str(path), *option])
                    )


if __name__ == "__main__":
    unittest.main()
