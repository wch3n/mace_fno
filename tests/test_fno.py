from __future__ import annotations

import unittest

import torch

from mace_fno import (
    FNO2D,
    FNOFieldOperator2D,
    LearnedParticleMeshLongRange,
    LinearFNO2D,
)

DTYPE = torch.float64


class FNOTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_batched_shape_and_parameter_gradients(self) -> None:
        model = FNO2D(
            in_channels=2,
            out_channels=3,
            n_modes=(4, 5),
            hidden_channels=8,
            n_layers=2,
        ).to(dtype=DTYPE)
        field = torch.randn((3, 2, 16, 20), dtype=DTYPE, requires_grad=True)
        output = model(field)
        self.assertEqual(output.shape, (3, 3, 16, 20))
        output.square().mean().backward()
        self.assertIsNotNone(field.grad)
        self.assertTrue(torch.isfinite(field.grad).all())
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_zero_field_maps_to_zero(self) -> None:
        model = FNO2D(1, 1, (4, 4), hidden_channels=8, n_layers=2)
        field = torch.zeros((1, 16, 16))
        torch.testing.assert_close(model(field), field, atol=0, rtol=0)

    def test_linear_fno_obeys_superposition(self) -> None:
        model = LinearFNO2D(1, 2, (4, 4)).to(dtype=DTYPE)
        first = torch.randn((3, 1, 16, 16), dtype=DTYPE)
        second = torch.randn((3, 1, 16, 16), dtype=DTYPE)
        scale = -1.7
        combined = model(first + scale * second)
        expected = model(first) + scale * model(second)
        torch.testing.assert_close(combined, expected, atol=3e-12, rtol=3e-12)

    def test_discrete_periodic_translation_equivariance(self) -> None:
        model = FNO2D(1, 1, (4, 4), hidden_channels=8, n_layers=2).to(
            dtype=DTYPE
        )
        field = torch.randn((2, 1, 16, 20), dtype=DTYPE)
        translated = torch.roll(field, shifts=(3, -4), dims=(-2, -1))
        expected = torch.roll(model(field), shifts=(3, -4), dims=(-2, -1))
        actual = model(translated)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    def test_normalization_is_stored_per_channel(self) -> None:
        operator = FNOFieldOperator2D(
            channels=2, n_modes=(3, 3), hidden_channels=4, n_layers=1
        )
        inputs = torch.randn((5, 2, 12, 12))
        targets = 3.0 * torch.randn((5, 2, 12, 12))
        operator.fit_normalization(inputs, targets)
        self.assertEqual(operator.input_scale.shape, (1, 2, 1, 1))
        self.assertEqual(operator.output_scale.shape, (1, 2, 1, 1))
        self.assertTrue((operator.input_scale > 0).all())
        self.assertTrue((operator.output_scale > 0).all())

    def test_learned_energy_has_finite_conservative_forces(self) -> None:
        cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        positions_base = torch.tensor(
            ((1.37, 2.11, 0.0), (7.08, 8.63, 0.0), (4.29, 3.54, 0.0)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedParticleMeshLongRange(
            (24, 24),
            channels=1,
            n_modes=(4, 4),
            hidden_channels=6,
            n_layers=2,
        ).to(dtype=DTYPE)

        positions = positions_base.clone().requires_grad_(True)
        energy = model(positions, charges, cell)
        force = -torch.autograd.grad(energy, positions)[0]
        self.assertTrue(torch.isfinite(force).all())

        step = 1.0e-5
        plus = positions_base.clone()
        minus = positions_base.clone()
        plus[0, 1] += step
        minus[0, 1] -= step
        finite_difference = -(
            model(plus, charges, cell) - model(minus, charges, cell)
        ) / (2.0 * step)
        torch.testing.assert_close(
            force[0, 1], finite_difference, atol=2e-8, rtol=2e-5
        )

    def test_native_density_api_matches_planar_particle_mesh_forward(self) -> None:
        cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        positions = torch.tensor(
            ((1.37, 2.11, 0.0), (7.08, 8.63, 0.0), (4.29, 3.54, 0.0)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedParticleMeshLongRange(
            (12, 12),
            channels=1,
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=1,
        ).to(dtype=DTYPE)

        particle_energy, density, particle_potential = model(
            positions, charges, cell, return_fields=True
        )
        density_energy, density_potential = model.energy_from_density(
            density, cell, return_potential=True
        )
        torch.testing.assert_close(
            density_energy, particle_energy, atol=3e-14, rtol=3e-14
        )
        torch.testing.assert_close(
            density_potential, particle_potential, atol=3e-14, rtol=3e-14
        )


if __name__ == "__main__":
    unittest.main()
