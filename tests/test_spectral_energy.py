from __future__ import annotations

import unittest

import torch

from mace_fno import ParticleMeshLongRange, direct_planar_coulomb_energy
from mace_fno.spectral import PlanarCoulombOperator

DTYPE = torch.float64


class SpectralEnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        self.positions = torch.tensor(
            ((1.37, 2.11, 0.0), (7.08, 8.63, 0.0), (4.29, 3.54, 0.0)),
            dtype=DTYPE,
        )
        self.charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)

    def test_constant_density_has_zero_potential(self) -> None:
        operator = PlanarCoulombOperator(max_modes=(4, 4))
        density = torch.ones((1, 20, 24), dtype=DTYPE)
        potential = operator(density, self.cell)
        torch.testing.assert_close(
            potential, torch.zeros_like(potential), atol=1e-14, rtol=0
        )

    def test_planar_operator_batches_different_cell_areas(self) -> None:
        operator = PlanarCoulombOperator(
            max_modes=(4, 4), deconvolve_assignment=False
        )
        densities = torch.randn((2, 1, 20, 24), dtype=DTYPE)
        cells = torch.stack((self.cell, 1.2 * self.cell))
        batched = operator(densities, cells)
        separate = torch.stack(
            [operator(density, cell) for density, cell in zip(densities, cells)]
        )
        torch.testing.assert_close(batched, separate, atol=2e-13, rtol=2e-13)

    def test_model_rejects_non_neutral_input(self) -> None:
        model = ParticleMeshLongRange((32, 32), max_modes=(4, 4))
        with self.assertRaisesRegex(ValueError, "neutral"):
            model(self.positions, torch.ones(3, dtype=DTYPE), self.cell)

    def test_energy_is_invariant_to_atom_permutation(self) -> None:
        model = ParticleMeshLongRange((40, 48), max_modes=(5, 5))
        permutation = torch.tensor((2, 0, 1))
        original = model(self.positions, self.charges, self.cell)
        permuted = model(
            self.positions[permutation], self.charges[permutation], self.cell
        )
        torch.testing.assert_close(original, permuted, atol=2e-13, rtol=2e-13)

    def test_energy_is_invariant_to_discrete_mesh_translation(self) -> None:
        grid_shape = (40, 48)
        model = ParticleMeshLongRange(grid_shape, max_modes=(5, 5))
        shift = self.cell[0] / grid_shape[0] + 2.0 * self.cell[1] / grid_shape[1]
        original = model(self.positions, self.charges, self.cell)
        translated = model(self.positions + shift, self.charges, self.cell)
        torch.testing.assert_close(original, translated, atol=2e-12, rtol=2e-12)

    def test_autograd_force_matches_finite_difference(self) -> None:
        model = ParticleMeshLongRange((48, 56), max_modes=(5, 5))
        positions = self.positions.clone().requires_grad_(True)
        energy = model(positions, self.charges, self.cell)
        force = -torch.autograd.grad(energy, positions)[0]

        step = 1.0e-5
        plus = self.positions.clone()
        minus = self.positions.clone()
        plus[0, 0] += step
        minus[0, 0] -= step
        finite_difference = -(
            model(plus, self.charges, self.cell) - model(minus, self.charges, self.cell)
        ) / (2.0 * step)
        torch.testing.assert_close(
            force[0, 0], finite_difference, atol=2e-7, rtol=2e-6
        )

    def test_mesh_energy_converges_to_direct_reciprocal_reference(self) -> None:
        max_modes = (4, 4)
        reference = direct_planar_coulomb_energy(
            self.positions, self.charges, self.cell, max_modes
        )
        coarse = ParticleMeshLongRange((16, 16), max_modes=max_modes)(
            self.positions, self.charges, self.cell
        )
        fine = ParticleMeshLongRange((64, 64), max_modes=max_modes)(
            self.positions, self.charges, self.cell
        )
        self.assertLess(
            (fine - reference).abs().item(), (coarse - reference).abs().item()
        )
        self.assertLess(((fine - reference) / reference).abs().item(), 2.0e-3)


if __name__ == "__main__":
    unittest.main()
