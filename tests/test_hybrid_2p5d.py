from __future__ import annotations

import unittest

import torch

from mace_fno import (
    GlobalZMixing,
    LearnedSlabParticleMeshLongRange,
    LinearSlabFNO2D,
    SlabFNO2D,
    SlabParticleMesh,
)

DTYPE = torch.float64


class SlabParticleMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        self.positions = torch.tensor(
            ((1.17, 2.31, 8.4), (7.23, 8.91, 10.2), (4.44, 3.28, 11.1)),
            dtype=DTYPE,
        )
        self.values = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        self.assignment = SlabParticleMesh(
            (8, 16, 20), z_extent=8.0, z_center="mean"
        )

    def test_integrated_density_conserves_all_channels(self) -> None:
        values = torch.stack((self.values, 2.0 * self.values), dim=-1)
        density = self.assignment(self.positions, values, self.cell)
        self.assertEqual(density.shape, (2, 8, 16, 20))
        voxel_volume = 10.0 * 12.0 * 8.0 / (8 * 16 * 20)
        integrated = density.sum(dim=(-3, -2, -1)) * voxel_volume
        torch.testing.assert_close(
            integrated, values.sum(dim=0), atol=3e-14, rtol=3e-14
        )

    def test_mean_center_is_invariant_to_normal_translation(self) -> None:
        original = self.assignment(self.positions, self.values, self.cell)
        translated = self.assignment(
            self.positions + torch.tensor((0.0, 0.0, 3.7), dtype=DTYPE),
            self.values,
            self.cell,
        )
        torch.testing.assert_close(original, translated, atol=3e-13, rtol=3e-13)

    def test_in_plane_grid_translation_rolls_only_in_plane(self) -> None:
        shifted = self.positions + self.cell[0] / 16
        original_density = self.assignment(self.positions, self.values, self.cell)
        shifted_density = self.assignment(shifted, self.values, self.cell)
        torch.testing.assert_close(
            shifted_density,
            torch.roll(original_density, shifts=1, dims=-2),
            atol=3e-13,
            rtol=3e-13,
        )

    def test_z_boundary_is_clamped_not_wrapped(self) -> None:
        assignment = SlabParticleMesh((8, 16, 16), z_extent=16.0, z_center="cell")
        position = torch.tensor(((2.0, 3.0, 2.1),), dtype=DTYPE)
        density = assignment(position, torch.ones(1, dtype=DTYPE), self.cell)
        self.assertGreater(density[:, 0].abs().sum().item(), 0.0)
        torch.testing.assert_close(
            density[:, -1], torch.zeros_like(density[:, -1]), atol=0.0, rtol=0.0
        )

    def test_atoms_outside_finite_window_are_rejected(self) -> None:
        positions = self.positions.clone()
        positions[0, 2] = -20.0
        with self.assertRaisesRegex(ValueError, "outside the finite z window"):
            self.assignment(positions, self.values, self.cell)


class HybridFNOTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_nonlinear_shape_and_gradients(self) -> None:
        model = SlabFNO2D(
            2,
            3,
            6,
            (3, 4),
            hidden_channels=5,
            n_layers=2,
            z_kernel_size=3,
        ).to(dtype=DTYPE)
        field = torch.randn((2, 2, 6, 12, 16), dtype=DTYPE, requires_grad=True)
        output = model(field)
        self.assertEqual(output.shape, (2, 3, 6, 12, 16))
        output.square().mean().backward()
        self.assertIsNotNone(field.grad)
        self.assertTrue(torch.isfinite(field.grad).all())
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_global_z_mixer_reaches_arbitrary_layers_without_wrapping(self) -> None:
        mixer = GlobalZMixing(channels=1, n_z=5).to(dtype=DTYPE)
        with torch.no_grad():
            mixer.weight.zero_()
            mixer.weight[0, 4, 0] = 2.0
        lateral_field = torch.randn((1, 1, 4, 6), dtype=DTYPE)
        field = torch.zeros((1, 1, 5, 4, 6), dtype=DTYPE)
        field[:, :, 0] = lateral_field
        output = mixer(field)
        torch.testing.assert_close(output[:, :, 4], 2.0 * lateral_field)
        torch.testing.assert_close(output[:, :, :4], torch.zeros_like(output[:, :, :4]))

        reversed_field = torch.zeros_like(field)
        reversed_field[:, :, 4] = lateral_field
        torch.testing.assert_close(
            mixer(reversed_field), torch.zeros_like(reversed_field)
        )

    def test_global_nonlinear_shape_and_gradients(self) -> None:
        model = SlabFNO2D(
            2,
            3,
            6,
            (3, 4),
            hidden_channels=5,
            n_layers=2,
            z_mixing="global",
        ).to(dtype=DTYPE)
        field = torch.randn((2, 2, 6, 12, 16), dtype=DTYPE, requires_grad=True)
        output = model(field)
        self.assertEqual(output.shape, (2, 3, 6, 12, 16))
        output.square().mean().backward()
        self.assertIsNotNone(field.grad)
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_zero_field_maps_to_zero(self) -> None:
        field = torch.zeros((1, 6, 12, 12))
        for z_mixing in ("local", "global"):
            with self.subTest(z_mixing=z_mixing):
                model = SlabFNO2D(
                    1,
                    1,
                    6,
                    (3, 3),
                    hidden_channels=4,
                    n_layers=2,
                    z_mixing=z_mixing,
                )
                torch.testing.assert_close(model(field), field, atol=0.0, rtol=0.0)

    def test_translation_equivariance_is_in_plane(self) -> None:
        field = torch.randn((2, 1, 6, 12, 14), dtype=DTYPE)
        translated = torch.roll(field, shifts=(2, -3), dims=(-2, -1))
        for z_mixing in ("local", "global"):
            with self.subTest(z_mixing=z_mixing):
                model = SlabFNO2D(
                    1,
                    1,
                    6,
                    (3, 3),
                    hidden_channels=4,
                    n_layers=2,
                    z_mixing=z_mixing,
                ).to(dtype=DTYPE)
                expected = torch.roll(model(field), shifts=(2, -3), dims=(-2, -1))
                torch.testing.assert_close(
                    model(translated), expected, atol=3e-12, rtol=3e-12
                )

    def test_one_operator_accepts_multiple_lateral_resolutions(self) -> None:
        model = SlabFNO2D(1, 2, 5, (3, 3), hidden_channels=4, n_layers=1).to(dtype=DTYPE)
        first = model(torch.randn((1, 5, 12, 12), dtype=DTYPE))
        second = model(torch.randn((1, 5, 16, 20), dtype=DTYPE))
        self.assertEqual(first.shape, (2, 5, 12, 12))
        self.assertEqual(second.shape, (2, 5, 16, 20))

    def test_dense_linear_operator_obeys_superposition(self) -> None:
        model = LinearSlabFNO2D(1, 2, 5, (3, 3)).to(dtype=DTYPE)
        first = torch.randn((2, 1, 5, 12, 12), dtype=DTYPE)
        second = torch.randn((2, 1, 5, 12, 12), dtype=DTYPE)
        scale = -0.7
        actual = model(first + scale * second)
        expected = model(first) + scale * model(second)
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)

    def test_energy_has_conservative_normal_force(self) -> None:
        cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        positions_base = torch.tensor(
            ((1.37, 2.11, 8.7), (7.08, 8.63, 10.1), (4.29, 3.54, 11.2)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedSlabParticleMeshLongRange(
            (6, 12, 12),
            8.0,
            channels=1,
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=2,
            z_mixing="global",
        ).to(dtype=DTYPE)

        positions = positions_base.clone().requires_grad_(True)
        energy = model(positions, charges, cell)
        force = -torch.autograd.grad(energy, positions)[0]
        self.assertTrue(torch.isfinite(force).all())

        step = 1.0e-5
        plus = positions_base.clone()
        minus = positions_base.clone()
        plus[0, 2] += step
        minus[0, 2] -= step
        finite_difference = -(
            model(plus, charges, cell) - model(minus, charges, cell)
        ) / (2.0 * step)
        torch.testing.assert_close(force[0, 2], finite_difference, atol=3e-8, rtol=3e-5)

    def test_native_density_api_matches_slab_particle_mesh_forward(self) -> None:
        cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        positions = torch.tensor(
            ((1.37, 2.11, 8.7), (7.08, 8.63, 10.1), (4.29, 3.54, 11.2)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedSlabParticleMeshLongRange(
            (6, 12, 12),
            8.0,
            channels=1,
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=1,
            z_mixing="global",
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

    def test_mean_center_energy_is_invariant_to_rigid_normal_translation(self) -> None:
        cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        positions_base = torch.tensor(
            ((1.37, 2.11, 8.7), (7.08, 8.63, 10.1), (4.29, 3.54, 11.2)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedSlabParticleMeshLongRange(
            (8, 12, 12),
            10.0,
            channels=1,
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=2,
            z_center="mean",
            z_mixing="global",
        ).to(dtype=DTYPE)

        positions = positions_base.clone().requires_grad_(True)
        energy = model(positions, charges, cell)
        force = -torch.autograd.grad(energy, positions)[0]
        translated_energy = model(
            positions_base + torch.tensor((0.0, 0.0, 2.37), dtype=DTYPE),
            charges,
            cell,
        )

        torch.testing.assert_close(
            translated_energy, energy.detach(), atol=3e-13, rtol=3e-13
        )
        torch.testing.assert_close(
            force[:, 2].sum(), torch.zeros((), dtype=DTYPE), atol=3e-12, rtol=0.0
        )

    def test_lateral_interlacing_matches_four_shift_average(self) -> None:
        cell = torch.diag(torch.tensor((10.0, 12.0, 20.0), dtype=DTYPE))
        positions = torch.tensor(
            ((1.37, 2.11, 8.7), (7.08, 8.63, 10.1), (4.29, 3.54, 11.2)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        common = dict(
            grid_shape=(8, 12, 16),
            z_extent=10.0,
            channels=1,
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=2,
            z_center="mean",
            z_mixing="global",
        )
        plain = LearnedSlabParticleMeshLongRange(**common, lateral_interlacing=1).to(
            dtype=DTYPE
        )
        interlaced = LearnedSlabParticleMeshLongRange(
            **common, lateral_interlacing=2
        ).to(dtype=DTYPE)
        interlaced.load_state_dict(plain.state_dict())

        offsets = (
            torch.zeros(3, dtype=DTYPE),
            0.5 * cell[1] / 16,
            0.5 * cell[0] / 12,
            0.5 * cell[0] / 12 + 0.5 * cell[1] / 16,
        )
        expected = torch.stack(
            [plain(positions + offset, charges, cell) for offset in offsets]
        ).mean()
        actual = interlaced(positions, charges, cell)
        torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)

        with self.assertRaisesRegex(ValueError, "different mesh origins"):
            interlaced(positions, charges, cell, return_fields=True)

    def test_d4_symmetrized_energy_and_forces_are_equivariant(self) -> None:
        cell = torch.diag(torch.tensor((12.0, 12.0, 20.0), dtype=DTYPE))
        positions_base = torch.tensor(
            ((1.37, 2.11, 8.7), (7.08, 8.63, 10.1), (4.29, 3.54, 11.2)),
            dtype=DTYPE,
        )
        charges = torch.tensor((1.0, -0.4, -0.6), dtype=DTYPE)
        model = LearnedSlabParticleMeshLongRange(
            (8, 12, 12),
            10.0,
            channels=1,
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=2,
            z_center="mean",
            z_mixing="global",
            lateral_interlacing=2,
            planar_symmetry="d4",
        ).to(dtype=DTYPE)
        model.eval()

        positions = positions_base.clone().requires_grad_(True)
        energy = model(positions, charges, cell)
        forces = -torch.autograd.grad(energy, positions)[0]
        rotated_base = positions_base.clone()
        rotated_base[:, 0] = 12.0 - positions_base[:, 1]
        rotated_base[:, 1] = positions_base[:, 0]
        rotated = rotated_base.requires_grad_(True)
        rotated_energy = model(rotated, charges, cell)
        rotated_forces = -torch.autograd.grad(rotated_energy, rotated)[0]
        expected_forces = forces.detach().clone()
        expected_forces[:, 0] = -forces.detach()[:, 1]
        expected_forces[:, 1] = forces.detach()[:, 0]

        torch.testing.assert_close(
            rotated_energy, energy.detach(), atol=3e-13, rtol=3e-13
        )
        torch.testing.assert_close(
            rotated_forces, expected_forces, atol=3e-11, rtol=3e-11
        )

        reflected_base = positions_base.clone()
        reflected_base[:, 0] = 12.0 - positions_base[:, 0]
        reflected = reflected_base.requires_grad_(True)
        reflected_energy = model(reflected, charges, cell)
        reflected_forces = -torch.autograd.grad(reflected_energy, reflected)[0]
        expected_reflected_forces = forces.detach().clone()
        expected_reflected_forces[:, 0] = -forces.detach()[:, 0]
        torch.testing.assert_close(
            reflected_energy, energy.detach(), atol=3e-13, rtol=3e-13
        )
        torch.testing.assert_close(
            reflected_forces,
            expected_reflected_forces,
            atol=3e-11,
            rtol=3e-11,
        )


if __name__ == "__main__":
    unittest.main()
