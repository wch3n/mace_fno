"""Regression checks for finite-z coordinates and physical planar symmetry."""

from __future__ import annotations

import unittest

import torch

from mace_fno import (
    LearnedSlabParticleMeshLongRange,
    SlabFNOFieldOperator2D,
    SlabParticleMesh,
)

DTYPE = torch.float64


def tilted_plane_rotation() -> torch.Tensor:
    return torch.tensor(
        ((0.8, 0.0, 0.6), (0.0, 1.0, 0.0), (-0.6, 0.0, 0.8)), dtype=DTYPE
    )


class SlabCoordinateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(302)
        self.cell = torch.tensor(
            ((10.0, 0.0, 0.0), (2.0, 12.0, 0.0), (0.0, 0.0, 20.0)),
            dtype=DTYPE,
        )
        self.positions = torch.tensor(
            ((1.37, 2.11, 8.7), (7.08, 8.63, 10.1), (4.29, 3.54, 11.2)),
            dtype=DTYPE,
        )
        self.values = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        self.mesh = SlabParticleMesh((6, 12, 16), z_extent=10.0, z_center="mean")

    def test_third_vector_lateral_component_does_not_change_density(self) -> None:
        for center in ("mean", "cell"):
            mesh = SlabParticleMesh((6, 12, 16), z_extent=10.0, z_center=center)
            for rotation in (torch.eye(3, dtype=DTYPE), tilted_plane_rotation()):
                with self.subTest(center=center, rotation=rotation.tolist()):
                    cell = self.cell @ rotation.T
                    positions = self.positions @ rotation.T
                    tilted = cell.clone()
                    tilted[2] += 0.4 * cell[0] - 0.7 * cell[1]
                    torch.testing.assert_close(
                        mesh(positions, self.values, tilted),
                        mesh(positions, self.values, cell),
                        atol=3e-13,
                        rtol=3e-13,
                    )

    def test_mean_center_normal_translation_with_tilted_third_vector(self) -> None:
        rotation = tilted_plane_rotation()
        cell = self.cell @ rotation.T
        cell[2] += 0.4 * cell[0] - 0.7 * cell[1]
        normal = torch.linalg.cross(cell[0], cell[1])
        normal /= torch.linalg.vector_norm(normal)
        positions = self.positions @ rotation.T
        torch.testing.assert_close(
            self.mesh(positions + 3.7 * normal, self.values, cell),
            self.mesh(positions, self.values, cell),
            atol=3e-13,
            rtol=3e-13,
        )

    def test_skew_lattice_periodicity_with_tilted_third_vector(self) -> None:
        cell = self.cell.clone()
        cell[2] += cell[0] + 0.3 * cell[1]
        reference = self.mesh(self.positions, self.values, cell)
        for axis, size in ((0, 12), (1, 16)):
            with self.subTest(axis=axis):
                torch.testing.assert_close(
                    self.mesh(self.positions + cell[axis] / size, self.values, cell),
                    torch.roll(reference, shifts=1, dims=axis - 2),
                    atol=3e-13,
                    rtol=3e-13,
                )
                torch.testing.assert_close(
                    self.mesh(self.positions + cell[axis], self.values, cell),
                    reference,
                    atol=3e-13,
                    rtol=3e-13,
                )

    def test_batched_tilted_cells_preserve_density_gradients_and_channel_sums(
        self,
    ) -> None:
        rotated_cell = self.cell @ tilted_plane_rotation().T
        rotated_cell[2] += 0.4 * rotated_cell[0]
        cells = torch.stack((self.cell, rotated_cell))
        positions = torch.cat(
            (self.positions, self.positions @ tilted_plane_rotation().T)
        ).requires_grad_(True)
        values = torch.stack((self.values + 0.2, 2 * self.values + 0.5), dim=-1)
        values = torch.cat((values, values * 0.3))
        batch = torch.arange(2).repeat_interleave(3)
        batched = self.mesh(positions, values, cells, batch=batch)
        separate = torch.stack(
            [
                self.mesh(positions[i : i + 3], values[i : i + 3], cells[i // 3])
                for i in (0, 3)
            ]
        )
        torch.testing.assert_close(batched, separate, atol=3e-13, rtol=3e-13)
        batched_grad = torch.autograd.grad(batched.square().sum(), positions)[0]
        separate_grad = torch.autograd.grad(separate.square().sum(), positions)[0]
        torch.testing.assert_close(batched_grad, separate_grad, atol=3e-13, rtol=3e-13)
        areas = torch.linalg.vector_norm(
            torch.linalg.cross(cells[:, 0], cells[:, 1]), dim=-1
        )
        integrated = (
            batched.sum(dim=(-3, -2, -1)) * (areas * 10 / (6 * 12 * 16))[:, None]
        )
        torch.testing.assert_close(
            integrated, values.reshape(2, 3, 2).sum(dim=1), atol=3e-13, rtol=3e-13
        )

    def test_position_and_cell_first_and_second_derivatives(self) -> None:
        mesh = SlabParticleMesh((4, 4, 4), z_extent=10.0)
        positions = self.positions.clone().requires_grad_(True)
        cell = self.cell.clone()
        cell[2] += 0.4 * cell[0]
        cell.requires_grad_(True)

        def energy(pos, lattice):
            return mesh(pos, self.values, lattice).square().sum()

        self.assertTrue(torch.autograd.gradcheck(energy, (positions, cell)))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (positions, cell)))
        cell_gradient = torch.autograd.grad(energy(positions, cell), cell)[0]
        torch.testing.assert_close(
            cell_gradient[2], torch.zeros(3, dtype=DTYPE), atol=0, rtol=0
        )

    def test_invalid_plane_is_rejected_before_coordinate_projection(self) -> None:
        for replacement in ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (float("nan"), 0, 0)):
            with self.subTest(second_vector=replacement):
                cell = self.cell.clone()
                cell[1] = torch.tensor(replacement, dtype=DTYPE)
                with self.assertRaisesRegex(ValueError, "finite, non-zero area"):
                    self.mesh(self.positions, self.values, cell)

    def test_tilted_cell_energy_force_normal_translation_and_backpropagation(
        self,
    ) -> None:
        cell = torch.diag(torch.tensor((12.0, 12.0, 20.0), dtype=DTYPE))
        tilted = cell.clone()
        tilted[2] += 0.4 * cell[0] - 0.7 * cell[1]
        for interlacing in (1, 2):
            with self.subTest(interlacing=interlacing):
                model = (
                    LearnedSlabParticleMeshLongRange(
                        (6, 12, 12),
                        10.0,
                        channels=1,
                        n_modes=(3, 3),
                        hidden_channels=4,
                        n_layers=2,
                        z_mixing="global",
                        planar_symmetry="d4",
                        lateral_interlacing=interlacing,
                    )
                    .to(dtype=DTYPE)
                    .eval()
                )
                positions = self.positions.clone().requires_grad_(True)
                energy = model(positions, self.values, tilted)
                force = -torch.autograd.grad(energy, positions, create_graph=True)[0]
                reference = model(positions, self.values, cell)
                reference_force = -torch.autograd.grad(reference, positions)[0]
                torch.testing.assert_close(energy, reference, atol=3e-13, rtol=3e-13)
                torch.testing.assert_close(
                    force, reference_force, atol=3e-12, rtol=3e-12
                )
                translated = model(
                    positions + positions.new_tensor((0.0, 0.0, 0.37)),
                    self.values,
                    tilted,
                )
                torch.testing.assert_close(energy, translated, atol=3e-13, rtol=3e-13)
                torch.testing.assert_close(
                    force[:, 2].sum(), torch.zeros((), dtype=DTYPE), atol=3e-12, rtol=0
                )
                step = 1e-5
                errors = []
                for axis in range(3):
                    plus, minus = positions.detach().clone(), positions.detach().clone()
                    plus[0, axis] += step
                    minus[0, axis] -= step
                    finite_difference = -(
                        model(plus, self.values, tilted)
                        - model(minus, self.values, tilted)
                    ) / (2 * step)
                    errors.append((force[0, axis] - finite_difference).abs().item())
                self.assertLess(max(errors), 3e-9)
                force.square().sum().backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                )


class SlabPlanarSymmetryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.field = torch.randn((2, 1, 4, 8, 8), dtype=DTYPE)
        self.square = torch.diag(torch.tensor((12.0, 12.0, 20.0), dtype=DTYPE))

    def operator(self, symmetry, architecture="nonlinear"):
        return SlabFNOFieldOperator2D(
            1,
            4,
            (2, 2),
            hidden_channels=3,
            n_layers=1,
            z_mixing="global",
            planar_symmetry=symmetry,
            architecture=architecture,
        ).to(dtype=DTYPE)

    def test_c4_and_d4_reject_rectangles_and_equal_length_skew_cells(self) -> None:
        rectangle = self.square.clone()
        rectangle[1, 1] = 14
        skew = self.square.clone()
        skew[1] = torch.tensor((6.0, 108.0**0.5, 0.0), dtype=DTYPE)
        for symmetry in ("c4", "d4"):
            for architecture in ("linear", "nonlinear"):
                model = self.operator(symmetry, architecture)
                for training in (True, False):
                    model.train(training)
                    for cell in (rectangle, skew):
                        with self.subTest(
                            symmetry=symmetry,
                            architecture=architecture,
                            training=training,
                            cell=cell.tolist(),
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "square in-plane cell"
                            ):
                                model(self.field, cell)

    def test_every_cell_in_a_batch_is_validated(self) -> None:
        cells = torch.stack((self.square, self.square))
        cells[1, 1, 1] = 14
        for symmetry in ("c4", "d4"):
            with self.subTest(symmetry=symmetry):
                with self.assertRaisesRegex(ValueError, "square in-plane cell"):
                    self.operator(symmetry)(self.field, cells)

    def test_rotated_square_plane_and_tilted_third_vector_are_accepted(self) -> None:
        for symmetry in ("c4", "d4"):
            with self.subTest(symmetry=symmetry):
                model = self.operator(symmetry).eval()
                tilted = self.square @ tilted_plane_rotation().T
                tilted[2] += 0.4 * tilted[0]
                shared = model(self.field, self.square)
                batched = model(self.field, torch.stack((self.square, tilted)))
                torch.testing.assert_close(shared, batched, atol=0, rtol=0)
                torch.testing.assert_close(
                    shared[0], model(self.field[0], tilted), atol=3e-13, rtol=3e-13
                )

    def test_physical_cell_is_required_and_batch_shape_must_match(self) -> None:
        for symmetry in ("c4", "d4"):
            with self.subTest(symmetry=symmetry):
                model = self.operator(symmetry)
                with self.assertRaisesRegex(ValueError, "requires a physical cell"):
                    model(self.field)
                with self.assertRaisesRegex(ValueError, "matching the field batch"):
                    model(self.field, self.square.repeat(3, 1, 1))
                with self.assertRaisesRegex(TypeError, "floating-point tensor"):
                    model(self.field, self.square.long())

    def test_nonfinite_or_degenerate_cells_are_rejected(self) -> None:
        for replacement in (0.0, float("nan"), float("inf")):
            with self.subTest(replacement=replacement):
                cell = self.square.clone()
                cell[0, 0] = replacement
                with self.assertRaisesRegex(ValueError, "square in-plane cell"):
                    self.operator("d4")(self.field, cell)

    def test_square_cell_does_not_allow_a_rectangular_grid(self) -> None:
        field = torch.randn((1, 4, 8, 10), dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "square lateral grid"):
            self.operator("d4")(field, self.square)

    def test_no_symmetry_still_accepts_rectangular_and_skew_cells(self) -> None:
        model = self.operator("none").eval()
        expected = model(self.field)
        for cell in (
            torch.diag(torch.tensor((10.0, 14.0, 20.0), dtype=DTYPE)),
            torch.tensor(((12.0, 0, 0), (4.0, 12.0, 0), (0, 0, 20.0)), dtype=DTYPE),
        ):
            with self.subTest(cell=cell.tolist()):
                torch.testing.assert_close(
                    model(self.field, cell), expected, atol=0, rtol=0
                )

    def test_d4_equivariance_and_training_image_average_are_preserved(self) -> None:
        for architecture in ("linear", "nonlinear"):
            with self.subTest(architecture=architecture):
                model = self.operator("d4", architecture).eval()
                expected = model(self.field, self.square)
                for index in range(8):

                    def transform(field):
                        if index >= 4:
                            field = field.flip((-1,))
                        return torch.rot90(field, index % 4, (-2, -1))

                    torch.testing.assert_close(
                        model(transform(self.field), self.square),
                        transform(expected),
                        atol=3e-13,
                        rtol=3e-13,
                    )
                model.train()
                average = torch.stack(
                    [model(self.field, self.square) for _ in range(8)]
                ).mean(0)
                torch.testing.assert_close(average, expected, atol=3e-13, rtol=3e-13)


if __name__ == "__main__":
    unittest.main()
