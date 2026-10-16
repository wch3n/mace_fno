"""Factorized full-height operators: dense equivalence, forces and persistence."""

import tempfile
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path
from unittest import mock

import test_full_height_slab as full_height
import torch

from mace_fno.fno_slab import SlabFNOBlock2D
from mace_fno.fno_slab_eqgino import MetricEqGINOSpectralConv2D
from mace_fno.training.resume import resume_configuration, validate_resume_checkpoint

DTYPE = torch.float64


class LowRankSpectralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(744)

    def test_factorized_contraction_matches_dense_with_first_and_second_derivatives(
        self,
    ):
        for groups in (1, 2):
            with self.subTest(groups=groups):
                factorized = MetricEqGINOSpectralConv2D(
                    6, 8, (2, 2), groups=groups, rank=2, reference_length=8
                ).double()
                dense = MetricEqGINOSpectralConv2D(
                    6, 8, (2, 2), groups=groups, reference_length=8
                ).double()
                field = torch.randn(2, 6, 3, 8, 8, dtype=DTYPE, requires_grad=True)
                cells = torch.tensor(
                    [
                        [[8, 0, 0], [0, 8, 0], [0, 0, 20]],
                        [[9, 1, 0], [2, 7, 0], [0, 0, 30]],
                    ],
                    dtype=DTYPE,
                    requires_grad=True,
                )
                # Reconstruct at the physical wavevectors, including off-knot values.
                with mock.patch.object(
                    dense, "radial_weights", factorized.radial_weights
                ):
                    expected = dense(field, cells)
                # The production path must never allocate dense mode matrices.
                with mock.patch.object(
                    factorized,
                    "radial_weights",
                    side_effect=AssertionError("dense path"),
                ):
                    actual = factorized(field, cells)
                torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
                variables = (field, cells, *factorized.parameters())
                first = []
                for value in (actual, expected):
                    first.append(
                        torch.autograd.grad(
                            value.square().sum(),
                            variables,
                            create_graph=True,
                        )
                    )
                for actual_grad, expected_grad in zip(*first):
                    torch.testing.assert_close(actual_grad, expected_grad)
                    self.assertTrue(torch.isfinite(actual_grad).all())
                second = [
                    torch.autograd.grad(
                        sum(g.square().sum() for g in gradients),
                        variables,
                    )
                    for gradients in first
                ]
                for actual_grad, expected_grad in zip(*second):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_joint_height_channel_coupling_varies_by_wavelength_without_z_wrapping(
        self,
    ):
        block = SlabFNOBlock2D(
            2,
            3,
            (2, 2),
            z_mixing="full",
            activation="identity",
            spectral_symmetry="metric_eqgino",
            spectral_rank=1,
            metric_reference_length=8,
        ).double()
        with torch.no_grad():
            block.local.weight.zero_()
            spectral = block.spectral.spectral
            spectral.radial_input_factor.zero_()
            spectral.radial_output_factor.zero_()
            spectral.radial_input_factor[0, 0, 0, :] = 1
            spectral.radial_output_factor[0, 0, 5, 1] = 2
            spectral.radial_output_factor[0, 0, 1, 2] = -3
        cell = torch.diag(torch.tensor([8.0, 8.0, 20.0], dtype=DTYPE))
        phase = 2 * torch.pi * torch.arange(8, dtype=DTYPE) / 8
        first = phase[:, None].cos().expand(8, 8)
        second = (phase[:, None] + phase[None, :]).cos()
        field = torch.zeros(1, 2, 3, 8, 8, dtype=DTYPE)
        field[0, 0, 0] = first + second
        expected = torch.zeros_like(field)
        expected[0, 1, 2] = 2 * first
        expected[0, 0, 1] = -3 * second
        torch.testing.assert_close(block(field, cell), expected, atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(
            block(field.roll(1, dims=2), cell), torch.zeros_like(field)
        )

    def test_rank_bound_symmetry_and_vacuum_independence(self):
        layer = MetricEqGINOSpectralConv2D(
            6, 6, (3, 3), rank=2, reference_length=8
        ).double()
        query = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0, 1.7, 5.0], dtype=DTYPE)
        self.assertTrue(
            (torch.linalg.matrix_rank(layer.radial_weights(query)) <= 2).all()
        )
        x = torch.randn(2, 6, 3, 8, 8, dtype=DTYPE)
        cell = torch.diag(torch.tensor([8.0, 8.0, 20.0], dtype=DTYPE))
        expected = layer(x, cell)
        for flip, turns in product((False, True), range(4)):

            def transform(v):
                return torch.rot90(v.flip(-1) if flip else v, turns, (-2, -1))

            torch.testing.assert_close(layer(transform(x), cell), transform(expected))
        torch.testing.assert_close(
            layer(x.roll((2, 3), (-2, -1)), cell), expected.roll((2, 3), (-2, -1))
        )
        cell[2] *= 2
        torch.testing.assert_close(layer(x, cell), expected)

    def test_production_parameter_counts_and_dense_state_compatibility(self):
        for rank, expected in ((0, 5013504), (32, 835584), (64, 1671168)):
            layer = MetricEqGINOSpectralConv2D(384, 384, (8, 8), rank=rank)
            self.assertEqual(sum(p.numel() for p in layer.parameters()), expected)
        torch.manual_seed(745)
        original = MetricEqGINOSpectralConv2D(4, 4, (2, 2))
        torch.manual_seed(745)
        explicit_dense = MetricEqGINOSpectralConv2D(4, 4, (2, 2), rank=0)
        self.assertEqual(set(original.state_dict()), set(explicit_dense.state_dict()))
        explicit_dense.load_state_dict(original.state_dict(), strict=True)

    def test_invalid_rank_is_rejected(self):
        for rank in (-1, 5, 1.5, True):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                MetricEqGINOSpectralConv2D(4, 4, (2, 2), rank=rank)
        with self.assertRaisesRegex(ValueError, "shell_spline"):
            MetricEqGINOSpectralConv2D(
                4, 4, (2, 2), rank=2, parameterization="radial_mlp"
            )
        with self.assertRaises(ValueError):
            SlabFNOBlock2D(2, 3, (2, 2), z_mixing="global", spectral_rank=2)


class LowRankHeightIntegrationTests(full_height.FullHeightIntegrationTests):
    """Repeat force training, checkpoint/ASE and zero-output tests with factors."""

    def configuration(self, *args, **kwargs):
        kwargs.setdefault(
            "spectral_rank", 3 if kwargs.get("z_mixing", "full") == "full" else 0
        )
        return super().configuration(*args, **kwargs)

    def test_rank_configuration_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            cfg = self.configuration(path)
            contract = resume_configuration(cfg)
            self.assertEqual(contract["model"]["spectral_rank"], 3)
            state = {"configuration": contract, "completed_steps": 0}
            validate_resume_checkpoint(state, cfg)
            changed = replace(cfg, model=replace(cfg.model, spectral_rank=2))
            with self.assertRaisesRegex(ValueError, "resume settings differ"):
                validate_resume_checkpoint(state, changed)
            dense = self.configuration(path, spectral_rank=0)
            self.assertNotIn("spectral_rank", resume_configuration(dense)["model"])
            for overrides in (
                {"spectral_rank": -1},
                {"spectral_rank": 13},
                {"spectral_symmetry": "none", "cell_mode": "fixed"},
                {"metric_parameterization": "radial_mlp"},
                {"z_mixing": "global", "spectral_rank": 3},
            ):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    self.configuration(path, **overrides)


if __name__ == "__main__":
    unittest.main()
