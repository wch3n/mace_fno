from __future__ import annotations

import math
import unittest

import torch

from mace_fno.spectral_response import (
    fit_anisotropic_inverse_quadratic_response,
    fit_power_law_response,
    fit_reference_power_response,
    quadratic_basis_response,
    quadratic_mode_response,
    slab_coulomb_profile_matrix,
    slab_z_profiles,
    unique_integer_modes,
    unique_integer_modes_2d,
    unit_rms_cosine_mode,
    unit_rms_cosine_mode_2d,
    wavevector_norm,
)


DTYPE = torch.float64


class SpectralResponseTests(unittest.TestCase):
    def test_unique_modes_omit_zero_and_sign_duplicates(self) -> None:
        modes = unique_integer_modes(1)
        self.assertEqual(len(modes), 13)
        self.assertNotIn((0, 0, 0), modes)
        for mode in modes:
            self.assertNotIn(tuple(-component for component in mode), modes)

        planar_modes = unique_integer_modes_2d(1)
        self.assertEqual(len(planar_modes), 4)
        self.assertNotIn((0, 0), planar_modes)

    def test_cosine_mode_is_zero_mean_and_unit_rms(self) -> None:
        mode = unit_rms_cosine_mode(
            (12, 14, 16), (1, -1, 0), device="cpu", dtype=DTYPE
        )
        self.assertAlmostEqual(float(mode.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(mode.square().mean()), 1.0, places=12)
        planar = unit_rms_cosine_mode_2d(
            (14, 16), (1, -1), device="cpu", dtype=DTYPE
        )
        self.assertAlmostEqual(float(planar.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(planar.square().mean()), 1.0, places=12)

    def test_slab_profiles_are_orthonormal_and_kernel_is_symmetric(self) -> None:
        profiles, names = slab_z_profiles(12, 3, device="cpu", dtype=DTYPE)
        self.assertEqual(names, ["monopole", "dipole", "quadrupole"])
        overlap = torch.einsum("mz,nz->mn", profiles, profiles) / 12
        torch.testing.assert_close(
            overlap, torch.eye(3, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12
        )
        template = slab_coulomb_profile_matrix(profiles, 0.4, 8.0)
        torch.testing.assert_close(template, template.T, atol=1.0e-12, rtol=1.0e-12)
        self.assertGreater(float(template[0, 0]), 0.0)

    def test_wavevector_norm_uses_zxy_layout(self) -> None:
        cell = torch.diag(torch.tensor((10.0, 12.0, 15.0), dtype=DTYPE))
        observed = wavevector_norm(cell, (1, 2, 3))
        expected = 2.0 * math.pi * math.sqrt(
            (2.0 / 10.0) ** 2
            + (3.0 / 12.0) ** 2
            + (1.0 / 15.0) ** 2
        )
        self.assertAlmostEqual(float(observed), expected, places=12)

    def test_quadratic_response_recovers_channel_kernel(self) -> None:
        density = torch.zeros((3, 8, 8, 8), dtype=DTYPE)
        mode = unit_rms_cosine_mode((8, 8, 8), (1, 1, 0), device="cpu", dtype=DTYPE)
        kernel = torch.tensor(
            ((3.0, -0.5, 1.0), (-0.5, 2.0, 0.25), (1.0, 0.25, 4.0)),
            dtype=DTYPE,
        )

        def energy(fields: torch.Tensor) -> torch.Tensor:
            amplitudes = (fields * mode).mean(dim=(-3, -2, -1))
            return 0.5 * torch.einsum("bi,ij,bj->b", amplitudes, kernel, amplitudes)

        observed = quadratic_mode_response(density, mode, 0.2, energy)
        torch.testing.assert_close(observed, kernel, atol=1.0e-12, rtol=1.0e-12)

        basis = torch.stack(
            (
                torch.stack((mode, torch.zeros_like(mode), torch.zeros_like(mode))),
                torch.stack((torch.zeros_like(mode), mode, torch.zeros_like(mode))),
                torch.stack((torch.zeros_like(mode), torch.zeros_like(mode), mode)),
            )
        )
        observed_basis = quadratic_basis_response(density, basis, 0.2, energy)
        torch.testing.assert_close(
            observed_basis, kernel, atol=1.0e-12, rtol=1.0e-12
        )

    def test_power_law_fit_recovers_coulomb_shape(self) -> None:
        fit = fit_power_law_response(
            [(0.25, 32.0), (0.5, 8.0), (1.0, 2.0), (2.0, 0.5)]
        )
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertAlmostEqual(fit["free_power_exponent_p"], 2.0, places=12)
        self.assertAlmostEqual(fit["free_log_r2"], 1.0, places=12)
        self.assertAlmostEqual(fit["coulomb_p2_log_r2"], 1.0, places=12)

        planar_fit = fit_reference_power_response(
            [(0.25, 8.0), (0.5, 4.0), (1.0, 2.0), (2.0, 1.0)], 1.0
        )
        self.assertIsNotNone(planar_fit)
        assert planar_fit is not None
        self.assertAlmostEqual(planar_fit["free_power_exponent_p"], 1.0, places=12)
        self.assertAlmostEqual(planar_fit["reference_power_log_r2"], 1.0, places=12)

    def test_anisotropic_inverse_quadratic_fit_recovers_tensor(self) -> None:
        tensor = torch.tensor(
            ((2.0, 0.2, -0.1), (0.2, 3.0, 0.3), (-0.1, 0.3, 4.0)),
            dtype=DTYPE,
        )
        vectors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
        ]
        points = []
        for vector in vectors:
            k = torch.tensor(vector, dtype=DTYPE)
            points.append((vector, float(1.0 / torch.dot(k, tensor @ k))))
        fit = fit_anisotropic_inverse_quadratic_response(points)
        self.assertIsNotNone(fit)
        assert fit is not None
        torch.testing.assert_close(
            torch.tensor(
                fit["dielectric_over_prefactor_matrix"], dtype=DTYPE
            ),
            tensor,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertTrue(fit["positive_definite"])


if __name__ == "__main__":
    unittest.main()
