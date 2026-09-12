"""Phase controls must change probes, not their normalization or wavevectors."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from mace_fno.cli.audit_spectral import parse_arguments
from mace_fno.spectral_response import (
    quadratic_mode_response,
    unit_rms_cosine_mode,
    unit_rms_cosine_mode_2d,
    unit_rms_fourier_mode,
    unit_rms_fourier_mode_2d,
    wavevector_norm,
)
from mace_fno.training.spectral_diagnostic import summarize_amplitude_convergence

DTYPE = torch.float64


class SpectralPhaseTests(unittest.TestCase):
    def test_normalization_orthogonality_and_unchanged_cosine(self) -> None:
        for builder, legacy, shape, mode in (
            (unit_rms_fourier_mode, unit_rms_cosine_mode, (12, 14, 16), (1, -2, 0)),
            (unit_rms_fourier_mode, unit_rms_cosine_mode, (12, 14, 16), (1, -1, 0)),
            (unit_rms_fourier_mode_2d, unit_rms_cosine_mode_2d, (14, 16), (1, -2)),
            (unit_rms_fourier_mode_2d, unit_rms_cosine_mode_2d, (14, 16), (1, -1)),
        ):
            with self.subTest(shape=shape, mode=mode):
                args = dict(device="cpu", dtype=DTYPE)
                cosine = builder(shape, mode, phase="cosine", **args)
                sine = builder(shape, mode, phase="sine", **args)
                torch.testing.assert_close(
                    cosine, legacy(shape, mode, **args), atol=0, rtol=0
                )
                # Independent reconstruction of the original cosine convention.
                axes = torch.meshgrid(
                    *(torch.arange(n, dtype=DTYPE) for n in shape), indexing="ij"
                )
                angle = (
                    2 * math.pi * sum(m * x / n for m, x, n in zip(mode, axes, shape))
                )
                expected = torch.cos(angle)
                expected -= expected.mean()
                expected /= expected.square().mean().sqrt()
                torch.testing.assert_close(cosine, expected, atol=0, rtol=0)
                for field in (cosine, sine):
                    self.assertAlmostEqual(float(field.mean()), 0.0, places=14)
                    self.assertAlmostEqual(float(field.square().mean()), 1.0, places=14)
                self.assertAlmostEqual(float((cosine * sine).mean()), 0.0, places=14)
                negative = builder(shape, tuple(-m for m in mode), phase="sine", **args)
                torch.testing.assert_close(negative, -sine, atol=0, rtol=0)

    def test_reject_unknown_phase_and_vanishing_sine(self) -> None:
        for builder, shape, mode in (
            (unit_rms_fourier_mode, (8, 8, 8), (4, 0, 0)),
            (unit_rms_fourier_mode_2d, (8, 8), (0, 4)),
        ):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "vanishes"):
                    builder(shape, mode, phase="sine", device="cpu", dtype=DTYPE)
                with self.assertRaisesRegex(ValueError, "phase"):
                    builder(shape, mode, phase="unknown", device="cpu", dtype=DTYPE)

    def test_both_phases_recover_analytic_bulk_coulomb_curvature(self) -> None:
        shape = (12, 12, 12)
        cell = torch.diag(torch.tensor((10.0, 12.0, 15.0), dtype=DTYPE))
        axes = [torch.fft.fftfreq(n, dtype=DTYPE) * n for n in shape]
        nz, nx, ny = torch.meshgrid(*axes, indexing="ij")
        k2 = (2 * math.pi) ** 2 * ((nz / 15) ** 2 + (nx / 10) ** 2 + (ny / 12) ** 2)
        green = torch.zeros_like(k2)
        green[k2 > 0] = 1.0 / k2[k2 > 0]
        channel_kernel = torch.tensor(((2.0, 0.3), (0.3, 1.0)), dtype=DTYPE)
        density = (
            torch.randn(
                (2, *shape), generator=torch.Generator().manual_seed(17), dtype=DTYPE
            )
            * 0.02
        )

        def energy(fields: torch.Tensor) -> torch.Tensor:
            spectrum = torch.fft.fftn(fields, dim=(-3, -2, -1), norm="forward")
            return (
                0.5
                * torch.einsum(
                    "bcxyz,cd,bdxyz,xyz->b",
                    spectrum.conj(),
                    channel_kernel.to(spectrum.dtype),
                    spectrum,
                    green.to(spectrum.dtype),
                ).real
            )

        for mode in ((0, 1, 0), (1, 0, 1), (1, 1, -1), (2, 1, 0)):
            expected = channel_kernel / wavevector_norm(cell, mode).square()
            for phase in ("cosine", "sine"):
                with self.subTest(mode=mode, phase=phase):
                    field = unit_rms_fourier_mode(
                        shape, mode, phase=phase, device="cpu", dtype=DTYPE
                    )
                    response = quadratic_mode_response(density, field, 0.02, energy)
                    torch.testing.assert_close(
                        response, expected, atol=1e-10, rtol=1e-10
                    )

    def test_nonlinear_background_can_distinguish_the_phases(self) -> None:
        shape, mode = (12, 12, 12), (0, 1, 0)
        args = dict(device="cpu", dtype=DTYPE)
        background = unit_rms_fourier_mode(shape, (0, 2, 0), **args)
        density = (1.0 + 0.3 * background).unsqueeze(0)

        def energy(fields: torch.Tensor) -> torch.Tensor:
            return 0.25 * fields.pow(4).mean(dim=(1, 2, 3, 4))

        values = []
        for phase in ("cosine", "sine"):
            probe = unit_rms_fourier_mode(shape, mode, phase=phase, **args)
            response = quadratic_mode_response(density, probe, 0.001, energy)
            expected = 3 * (density.square() * probe.square()).mean()
            self.assertAlmostEqual(float(response[0, 0]), float(expected), places=5)
            values.append(float(response[0, 0]))
        self.assertGreater(abs(values[0] - values[1]), 1.0)

    def test_cli_phase_default_and_override(self) -> None:
        for extra, expected in (([], "cosine"), (["--probe-phase", "sine"], "sine")):
            with patch("sys.argv", ["audit", "--checkpoint", "model.pt", *extra]):
                self.assertEqual(parse_arguments().probe_phase, expected)

    def test_amplitude_summary_rejects_mixed_phases(self) -> None:
        base = dict(
            diagnostic_kind="periodic_3d",
            spatial_scheme="3d",
            sample_indices=[0],
            max_mode=2,
        )
        with self.assertRaisesRegex(ValueError, "phase"):
            summarize_amplitude_convergence([base, dict(base, probe_phase="sine")])


if __name__ == "__main__":
    unittest.main()
