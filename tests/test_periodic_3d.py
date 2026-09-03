from __future__ import annotations

import unittest

import torch

from mace_fno import (
    FNO3d,
    FNOFieldOperator3d,
    LearnedParticleMeshLongRange3D,
    LinearFNO3d,
    MetricEqGINOSpectralConv3d,
    PeriodicParticleMesh3D,
    cubic_signed_permutation_matrices,
)

DTYPE = torch.float64


def transform_periodic_scalar_grid(
    field: torch.Tensor, transformation: torch.Tensor
) -> torch.Tensor:
    """Apply ``f(x) -> f(R^-1 x)`` about the periodic grid origin."""
    size = field.shape[-1]
    if field.shape[-3:] != (size, size, size):
        raise ValueError("the test transformation requires a cubic grid")
    coordinates = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, device=field.device) for _ in range(3)),
            indexing="ij",
        ),
        dim=-1,
    )
    inverse = transformation.to(device=field.device, dtype=torch.long).T
    source = torch.einsum("ij,zxyj->zxyi", inverse, coordinates).remainder(size)
    return field[..., source[..., 0], source[..., 1], source[..., 2]]


class PeriodicParticleMesh3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = torch.tensor(
            ((9.0, 0.0, 0.0), (1.2, 10.0, 0.0), (0.4, -0.3, 11.0)),
            dtype=DTYPE,
        )
        fractional = torch.tensor(
            ((0.13, 0.27, 0.81), (0.72, 0.63, 0.06), (0.44, 0.18, 0.49)),
            dtype=DTYPE,
        )
        self.positions = fractional @ self.cell
        self.values = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        self.assignment = PeriodicParticleMesh3D((10, 12, 14))

    def test_integrated_density_conserves_every_channel(self) -> None:
        values = torch.stack((self.values, -1.7 * self.values), dim=-1)
        density = self.assignment(self.positions, values, self.cell)
        self.assertEqual(density.shape, (2, 10, 12, 14))
        voxel_volume = torch.linalg.det(self.cell).abs() / (10 * 12 * 14)
        integrated = density.sum(dim=(-3, -2, -1)) * voxel_volume
        torch.testing.assert_close(
            integrated, values.sum(dim=0), atol=3e-14, rtol=3e-14
        )

    def test_each_lattice_grid_translation_rolls_matching_axis(self) -> None:
        original = self.assignment(self.positions, self.values, self.cell)
        shifts = (
            (self.cell[2] / 10, -3),
            (self.cell[0] / 12, -2),
            (self.cell[1] / 14, -1),
        )
        for displacement, axis in shifts:
            with self.subTest(axis=axis):
                shifted = self.assignment(
                    self.positions + displacement, self.values, self.cell
                )
                torch.testing.assert_close(
                    shifted,
                    torch.roll(original, shifts=1, dims=axis),
                    atol=4e-13,
                    rtol=4e-13,
                )

    def test_full_lattice_vectors_wrap_exactly(self) -> None:
        original = self.assignment(self.positions, self.values, self.cell)
        wrapped = self.assignment(
            self.positions + self.cell[0] - 2.0 * self.cell[2],
            self.values,
            self.cell,
        )
        torch.testing.assert_close(original, wrapped, atol=5e-13, rtol=5e-13)

    def test_batched_assignment_matches_independent_graphs(self) -> None:
        second_positions = self.positions + 0.37 * self.cell[1]
        positions = torch.cat((self.positions, second_positions))
        values = torch.cat((self.values, 1.4 * self.values))
        cells = torch.stack((self.cell, self.cell))
        batch = torch.tensor((0, 0, 0, 1, 1, 1), dtype=torch.long)
        actual = self.assignment(positions, values, cells, batch=batch)
        expected = torch.stack(
            (
                self.assignment(self.positions, self.values, self.cell),
                self.assignment(second_positions, 1.4 * self.values, self.cell),
            )
        )
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)


class FullyPeriodicFNOTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)

    def test_nonlinear_shape_and_gradients(self) -> None:
        model = FNO3d(
            2, 3, (3, 3, 4), hidden_channels=5, n_layers=2
        ).to(dtype=DTYPE)
        field = torch.randn((2, 2, 10, 12, 14), dtype=DTYPE, requires_grad=True)
        output = model(field)
        self.assertEqual(output.shape, (2, 3, 10, 12, 14))
        output.square().mean().backward()
        self.assertIsNotNone(field.grad)
        self.assertTrue(torch.isfinite(field.grad).all())
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_zero_field_maps_to_zero(self) -> None:
        model = FNO3d(1, 1, (3, 3, 3), hidden_channels=4, n_layers=2)
        field = torch.zeros((1, 8, 10, 12))
        torch.testing.assert_close(model(field), field, atol=0.0, rtol=0.0)

    def test_discrete_translation_equivariance_in_all_directions(self) -> None:
        model = FNO3d(1, 1, (3, 3, 3), hidden_channels=4, n_layers=2).to(
            dtype=DTYPE
        )
        field = torch.randn((2, 1, 10, 12, 14), dtype=DTYPE)
        translated = torch.roll(field, shifts=(2, -3, 4), dims=(-3, -2, -1))
        expected = torch.roll(
            model(field), shifts=(2, -3, 4), dims=(-3, -2, -1)
        )
        torch.testing.assert_close(
            model(translated), expected, atol=4e-12, rtol=4e-12
        )

    def test_linear_operator_obeys_superposition(self) -> None:
        model = LinearFNO3d(1, 2, (3, 3, 3)).to(dtype=DTYPE)
        first = torch.randn((2, 1, 10, 12, 14), dtype=DTYPE)
        second = torch.randn((2, 1, 10, 12, 14), dtype=DTYPE)
        scale = -0.73
        actual = model(first + scale * second)
        expected = model(first) + scale * model(second)
        torch.testing.assert_close(actual, expected, atol=4e-12, rtol=4e-12)

    def test_isotropic_cell_conditioning_shape_and_gradients(self) -> None:
        model = FNOFieldOperator3d(
            2,
            (2, 2, 2),
            hidden_channels=4,
            n_layers=1,
            cell_conditioning="isotropic",
        ).to(dtype=DTYPE)
        density = torch.randn(
            (2, 2, 8, 8, 8), dtype=DTYPE, requires_grad=True
        )
        cells = torch.stack(
            (8.0 * torch.eye(3, dtype=DTYPE), 10.0 * torch.eye(3, dtype=DTYPE))
        )
        output = model(density, cells)
        self.assertEqual(output.shape, density.shape)
        self.assertEqual(model.fno.in_channels, 3)
        output.square().mean().backward()
        self.assertIsNotNone(density.grad)
        self.assertTrue(torch.isfinite(density.grad).all())

        with self.assertRaisesRegex(ValueError, "requires cell"):
            model(density.detach())
        with self.assertRaisesRegex(ValueError, "requires cubic"):
            model(
                density.detach(),
                torch.stack(
                    (cells[0], torch.diag(torch.tensor((10.0, 10.0, 11.0))))
                ),
            )

        with self.assertRaisesRegex(ValueError, "requires architecture"):
            FNOFieldOperator3d(
                1, (2, 2, 2), architecture="linear", cell_conditioning="isotropic"
            )

    def test_anisotropic_cell_conditioning_uses_rotation_invariant_metric(self) -> None:
        model = FNOFieldOperator3d(
            2,
            (2, 2, 2),
            hidden_channels=4,
            n_layers=1,
            cell_conditioning="anisotropic",
        ).to(dtype=DTYPE)
        density = torch.randn(
            (2, 2, 8, 8, 8), dtype=DTYPE, requires_grad=True
        )
        cells = torch.stack(
            (
                torch.diag(torch.tensor((8.0, 9.0, 10.0), dtype=DTYPE)),
                torch.tensor(
                    ((8.2, 0.1, 0.0), (0.4, 9.1, 0.2), (0.0, -0.3, 10.4)),
                    dtype=DTYPE,
                ),
            )
        )
        output = model(density, cells)
        self.assertEqual(output.shape, density.shape)
        self.assertEqual(model.fno.in_channels, 9)
        output.square().mean().backward()
        self.assertIsNotNone(density.grad)
        self.assertTrue(torch.isfinite(density.grad).all())

        rotation = torch.tensor(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=DTYPE,
        )
        torch.testing.assert_close(
            model(density.detach(), cells @ rotation),
            model(density.detach(), cells),
            atol=2e-12,
            rtol=2e-12,
        )

        singular = cells.clone()
        singular[1, 2] = singular[1, 1]
        with self.assertRaisesRegex(ValueError, "requires finite cells"):
            model(density.detach(), singular)

        with self.assertRaisesRegex(ValueError, "requires architecture"):
            FNOFieldOperator3d(
                1,
                (2, 2, 2),
                architecture="linear",
                cell_conditioning="anisotropic",
            )

    def test_energy_has_conservative_force_along_third_lattice_direction(self) -> None:
        cell = torch.tensor(
            ((9.0, 0.0, 0.0), (0.8, 10.0, 0.0), (0.3, -0.2, 11.0)),
            dtype=DTYPE,
        )
        fractional = torch.tensor(
            ((0.17, 0.21, 0.31), (0.68, 0.73, 0.62), (0.41, 0.36, 0.84)),
            dtype=DTYPE,
        )
        positions_base = fractional @ cell
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedParticleMeshLongRange3D(
            (8, 10, 12),
            channels=1,
            n_modes=(3, 3, 3),
            hidden_channels=4,
            n_layers=1,
        ).to(dtype=DTYPE)

        positions = positions_base.clone().requires_grad_(True)
        energy = model(positions, charges, cell)
        force = -torch.autograd.grad(energy, positions)[0]
        self.assertTrue(torch.isfinite(force).all())

        step = 1.0e-5
        direction = cell[2] / torch.linalg.vector_norm(cell[2])
        plus = positions_base.clone()
        minus = positions_base.clone()
        plus[0] += step * direction
        minus[0] -= step * direction
        finite_difference = -(
            model(plus, charges, cell) - model(minus, charges, cell)
        ) / (2.0 * step)
        projected_force = torch.dot(force[0], direction)
        torch.testing.assert_close(
            projected_force, finite_difference, atol=5e-8, rtol=5e-5
        )

    def test_native_density_api_matches_particle_mesh_forward(self) -> None:
        """The direct-field probe must preserve the native particle-mesh energy."""
        cell = torch.tensor(
            ((9.0, 0.0, 0.0), (0.8, 10.0, 0.0), (0.3, -0.2, 11.0)),
            dtype=DTYPE,
        )
        fractional = torch.tensor(
            ((0.17, 0.21, 0.31), (0.68, 0.73, 0.62), (0.41, 0.36, 0.84)),
            dtype=DTYPE,
        )
        positions = fractional @ cell
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedParticleMeshLongRange3D(
            (8, 10, 12),
            channels=1,
            n_modes=(3, 3, 3),
            hidden_channels=4,
            n_layers=1,
        ).to(dtype=DTYPE)

        particle_energy, density, particle_potential = model(
            positions, charges, cell, return_fields=True
        )
        density_energy, density_potential = model.energy_from_density(
            density, cell, return_potential=True
        )
        torch.testing.assert_close(density_energy, particle_energy, atol=3e-14, rtol=3e-14)
        torch.testing.assert_close(
            density_potential, particle_potential, atol=3e-14, rtol=3e-14
        )

    def test_volume_interlacing_matches_eight_shift_average(self) -> None:
        cell = torch.tensor(
            ((9.0, 0.0, 0.0), (0.8, 10.0, 0.0), (0.3, -0.2, 11.0)),
            dtype=DTYPE,
        )
        fractional = torch.tensor(
            ((0.17, 0.21, 0.31), (0.68, 0.73, 0.62), (0.41, 0.36, 0.84)),
            dtype=DTYPE,
        )
        positions = fractional @ cell
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        common = dict(
            grid_shape=(8, 10, 12),
            channels=1,
            n_modes=(3, 3, 3),
            hidden_channels=4,
            n_layers=1,
        )
        plain = LearnedParticleMeshLongRange3D(
            **common, volume_interlacing=1
        ).to(dtype=DTYPE)
        interlaced = LearnedParticleMeshLongRange3D(
            **common, volume_interlacing=2
        ).to(dtype=DTYPE)
        interlaced.load_state_dict(plain.state_dict())

        offsets = tuple(
            iz * 0.5 * cell[2] / 8
            + ix * 0.5 * cell[0] / 10
            + iy * 0.5 * cell[1] / 12
            for iz in range(2)
            for ix in range(2)
            for iy in range(2)
        )
        expected = torch.stack(
            [plain(positions + offset, charges, cell) for offset in offsets]
        ).mean()
        actual = interlaced(positions, charges, cell)
        torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)

        with self.assertRaisesRegex(ValueError, "different mesh origins"):
            interlaced(positions, charges, cell, return_fields=True)

    def test_random_training_origin_uses_one_interlacing_replica(self) -> None:
        cell = torch.diag(torch.tensor((8.0, 9.0, 10.0), dtype=DTYPE))
        fractional = torch.tensor(
            ((0.173, 0.217, 0.319), (0.683, 0.731, 0.627)), dtype=DTYPE
        )
        positions = fractional @ cell
        charges = torch.tensor((1.0, -1.0), dtype=DTYPE)
        common = dict(
            grid_shape=(8, 8, 8),
            channels=1,
            n_modes=(2, 2, 2),
            hidden_channels=4,
            n_layers=1,
        )
        plain = LearnedParticleMeshLongRange3D(**common).to(dtype=DTYPE)
        random_origin = LearnedParticleMeshLongRange3D(
            **common,
            volume_interlacing=2,
            interlacing_training="random",
        ).to(dtype=DTYPE)
        random_origin.load_state_dict(plain.state_dict())

        offsets = tuple(
            iz * 0.5 * cell[2] / 8
            + ix * 0.5 * cell[0] / 8
            + iy * 0.5 * cell[1] / 8
            for iz in range(2)
            for ix in range(2)
            for iy in range(2)
        )
        replica_energies = torch.stack(
            [plain(positions + offset, charges, cell) for offset in offsets]
        )
        random_origin.train()
        torch.manual_seed(123)
        training_energy = random_origin(positions, charges, cell)
        self.assertTrue(
            bool(torch.isclose(training_energy, replica_energies, atol=2e-14).any())
        )

        random_origin.eval()
        torch.testing.assert_close(
            random_origin(positions, charges, cell),
            replica_energies.mean(),
            atol=3e-14,
            rtol=3e-14,
        )

        density = plain.assignment(positions, charges, cell)
        torch.testing.assert_close(
            random_origin.energy_from_density(density, cell),
            plain.energy_from_density(density, cell),
            atol=3e-14,
            rtol=3e-14,
        )


class MetricEqGINOSpectralConv3DTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(103)
        self.cells = torch.stack(
            (
                torch.diag(torch.tensor((8.0, 9.0, 10.0), dtype=DTYPE)),
                torch.tensor(
                    ((8.2, 0.1, 0.0), (0.4, 9.1, 0.2), (0.0, -0.3, 10.4)),
                    dtype=DTYPE,
                ),
            )
        )

    def test_legacy_eqgino_options_are_rejected(self) -> None:
        for legacy in ("eqgino", "cubic_adaptive"):
            with self.subTest(spectral_symmetry=legacy):
                with self.assertRaisesRegex(
                    ValueError, "must be 'none' or 'metric_eqgino'"
                ):
                    FNO3d(1, 1, (2, 2, 2), spectral_symmetry=legacy)

    def test_heterogeneous_cells_shape_gradients_and_independent_batches(self) -> None:
        layer = MetricEqGINOSpectralConv3d(
            4,
            4,
            (2, 3, 2),
            groups=2,
            radial_hidden_channels=5,
        ).to(dtype=DTYPE)
        field = torch.randn(
            (2, 4, 8, 10, 12), dtype=DTYPE, requires_grad=True
        )
        output = layer(field, self.cells)
        expected = torch.cat(
            [
                layer(field[index : index + 1], self.cells[index])
                for index in range(2)
            ],
            dim=0,
        )
        self.assertEqual(output.shape, field.shape)
        self.assertFalse(output.is_complex())
        torch.testing.assert_close(output, expected, atol=3e-13, rtol=3e-13)

        output.square().mean().backward()
        self.assertIsNotNone(field.grad)
        self.assertTrue(torch.isfinite(field.grad).all())
        parameter_gradients = [
            parameter.grad for parameter in layer.radial_network.parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in parameter_gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
        )

    def test_physical_wavevectors_use_reciprocal_cell_metric(self) -> None:
        layer = MetricEqGINOSpectralConv3d(1, 1, (2, 2, 2)).to(dtype=DTYPE)
        squared = layer._physical_squared_wavevectors(self.cells[:1])
        expected_x = (2.0 * torch.pi / self.cells[0, 0, 0]).square()
        expected_y = (2.0 * torch.pi / self.cells[0, 1, 1]).square()
        expected_z = (2.0 * torch.pi / self.cells[0, 2, 2]).square()
        torch.testing.assert_close(squared[0, 0, 1, 0], expected_x)
        torch.testing.assert_close(squared[0, 0, 0, 1], expected_y)
        torch.testing.assert_close(squared[0, 1, 0, 0], expected_z)
        self.assertNotEqual(squared[0, 0, 1, 0], squared[0, 0, 0, 1])

    def test_rigid_cartesian_cell_rotation_leaves_operator_unchanged(self) -> None:
        layer = MetricEqGINOSpectralConv3d(
            2, 2, (3, 3, 3), groups=2
        ).to(dtype=DTYPE)
        field = torch.randn((2, 2, 8, 8, 8), dtype=DTYPE)
        rotation, _ = torch.linalg.qr(torch.randn((3, 3), dtype=DTYPE))
        rotation = rotation * torch.linalg.det(rotation)
        reference = layer(field, self.cells)
        rotated = layer(field, self.cells @ rotation)
        torch.testing.assert_close(rotated, reference, atol=2e-12, rtol=2e-12)

    def test_cubic_signed_axis_equivariance_is_exact(self) -> None:
        model = FNO3d(
            2,
            2,
            (3, 3, 3),
            hidden_channels=4,
            n_layers=2,
            spectral_symmetry="metric_eqgino",
            spectral_groups=2,
        ).to(dtype=DTYPE)
        field = torch.randn((1, 2, 8, 8, 8), dtype=DTYPE)
        cell = (8.0 * torch.eye(3, dtype=DTYPE)).unsqueeze(0)
        reference = model(field, cell=cell)
        transformations = cubic_signed_permutation_matrices(
            include_reflections=True, dtype=DTYPE
        )
        for index, transformation in enumerate(transformations):
            transformed = transform_periodic_scalar_grid(field, transformation)
            expected = transform_periodic_scalar_grid(reference, transformation)
            with self.subTest(group_element=index):
                torch.testing.assert_close(
                    model(transformed, cell=cell), expected, atol=3e-12, rtol=3e-12
                )

    def test_particle_mesh_energy_and_force_rotate_for_triclinic_cell(self) -> None:
        cell = self.cells[1]
        fractional = torch.tensor(
            ((0.17, 0.21, 0.31), (0.68, 0.73, 0.62), (0.41, 0.36, 0.84)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedParticleMeshLongRange3D(
            (8, 8, 8),
            channels=1,
            n_modes=(3, 3, 3),
            hidden_channels=4,
            n_layers=2,
            spectral_symmetry="metric_eqgino",
            metric_hidden_channels=7,
        ).to(dtype=DTYPE)

        positions = (fractional @ cell).requires_grad_(True)
        energy = model(positions, charges, cell)
        force = -torch.autograd.grad(energy, positions)[0]
        rotation, _ = torch.linalg.qr(torch.randn((3, 3), dtype=DTYPE))
        rotation = rotation * torch.linalg.det(rotation)
        rotated_cell = cell @ rotation
        rotated_positions = (fractional @ rotated_cell).requires_grad_(True)
        rotated_energy = model(rotated_positions, charges, rotated_cell)
        rotated_force = -torch.autograd.grad(rotated_energy, rotated_positions)[0]

        torch.testing.assert_close(rotated_energy, energy, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
            rotated_force, force @ rotation, atol=2e-11, rtol=2e-11
        )

    def test_metric_operator_requires_valid_cells(self) -> None:
        layer = MetricEqGINOSpectralConv3d(1, 1, (2, 2, 2)).to(dtype=DTYPE)
        field = torch.randn((1, 1, 8, 8, 8), dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "cell must have shape"):
            layer(field, self.cells)
        singular = self.cells[0].clone()
        singular[2] = singular[1]
        with self.assertRaisesRegex(ValueError, "finite and nonsingular"):
            layer(field, singular)


if __name__ == "__main__":
    unittest.main()
