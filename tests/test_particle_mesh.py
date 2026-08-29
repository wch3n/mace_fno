from __future__ import annotations

import unittest

import torch

from mace_fno.geometry import mesh_cell_area
from mace_fno.particle_mesh import PeriodicParticleMesh2D


DTYPE = torch.float64


class ParticleMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        self.positions = torch.tensor(
            ((1.17, 2.31, 0.4), (7.23, 8.91, -0.2), (4.44, 3.28, 0.1)),
            dtype=DTYPE,
        )
        self.values = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        self.assignment = PeriodicParticleMesh2D((24, 20))

    def test_integrated_density_conserves_input_values(self) -> None:
        values = torch.stack((self.values, 2.0 * self.values), dim=-1)
        density = self.assignment(self.positions, values, self.cell)
        integrated = density.sum(dim=(-2, -1)) * mesh_cell_area(
            self.cell, self.assignment.grid_shape
        )
        torch.testing.assert_close(integrated, values.sum(dim=0), atol=2e-14, rtol=2e-14)

    def test_atom_permutation_does_not_change_density(self) -> None:
        permutation = torch.tensor((2, 0, 1))
        original = self.assignment(self.positions, self.values, self.cell)
        permuted = self.assignment(
            self.positions[permutation], self.values[permutation], self.cell
        )
        torch.testing.assert_close(original, permuted, atol=1e-14, rtol=1e-14)

    def test_periodic_wrapping_is_continuous(self) -> None:
        left = self.positions.clone()
        right = self.positions.clone()
        left[0, 0] = -1.0e-7
        right[0, 0] = self.cell[0, 0] - 1.0e-7
        density_left = self.assignment(left, self.values, self.cell)
        density_right = self.assignment(right, self.values, self.cell)
        torch.testing.assert_close(density_left, density_right, atol=2e-13, rtol=2e-13)

    def test_translation_by_one_grid_cell_rolls_density(self) -> None:
        shifted_positions = self.positions + self.cell[0] / self.assignment.grid_shape[0]
        original = self.assignment(self.positions, self.values, self.cell)
        shifted = self.assignment(shifted_positions, self.values, self.cell)
        torch.testing.assert_close(
            shifted, torch.roll(original, shifts=1, dims=-2), atol=3e-13, rtol=3e-13
        )

    def test_batched_assignment_matches_independent_graphs(self) -> None:
        second_positions = self.positions + torch.tensor(
            (0.31, -0.27, 0.2), dtype=DTYPE
        )
        positions = torch.cat((self.positions, second_positions))
        values = torch.cat((self.values, 1.7 * self.values))
        cells = torch.stack((self.cell, self.cell))
        batch = torch.tensor((0, 0, 0, 1, 1, 1), dtype=torch.long)

        actual = self.assignment(positions, values, cells, batch=batch)
        expected = torch.stack(
            (
                self.assignment(self.positions, self.values, self.cell),
                self.assignment(second_positions, 1.7 * self.values, self.cell),
            )
        )
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)


if __name__ == "__main__":
    unittest.main()
