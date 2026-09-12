from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from mace_fno.fno_3d import FNOFieldOperator3D, MetricEqGINOSpectralConv3D
from mace_fno.radial import NaturalCubicSpline

DTYPE = torch.float64


class NaturalCubicSplineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(714)
        self.knots = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 8.0], dtype=DTYPE)
        self.spline = NaturalCubicSpline(self.knots).to(dtype=DTYPE)

    def test_cardinal_values_and_parameter_jacobian(self) -> None:
        values = torch.randn(6, 3, dtype=DTYPE, requires_grad=True)
        torch.testing.assert_close(
            self.spline(self.knots, values), values, atol=1e-14, rtol=0
        )
        jac = torch.autograd.functional.jacobian(
            lambda v: self.spline(self.knots, v), values
        )
        torch.testing.assert_close(
            jac.reshape(18, 18), torch.eye(18, dtype=DTYPE), atol=1e-14, rtol=0
        )

    def test_linear_reproduction_and_extrapolation(self) -> None:
        query = torch.linspace(-5, 15, 51, dtype=DTYPE)
        values = torch.stack([2 * self.knots + 3, -self.knots + 1], dim=-1)
        expected = torch.stack([2 * query + 3, -query + 1], dim=-1)
        torch.testing.assert_close(
            self.spline(query, values), expected, atol=1e-13, rtol=0
        )

    def test_single_and_two_knots(self) -> None:
        q = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)
        one = NaturalCubicSpline(torch.tensor([0.0]))
        answer = one(q, torch.tensor([7.0], dtype=DTYPE))
        self.assertEqual(answer.item(), 7.0)
        self.assertEqual(torch.autograd.grad(answer, q)[0].item(), 0.0)
        two = NaturalCubicSpline(torch.tensor([0.0, 2.0]))
        torch.testing.assert_close(
            two(q, torch.tensor([1.0, 5.0], dtype=DTYPE)), 2 * q + 1
        )

    def test_c2_at_all_knots_including_extrapolation_boundaries(self) -> None:
        values = torch.randn(6, dtype=DTYPE)
        for knot in self.knots:
            sides = []
            for delta in [-1e-7, 1e-7]:
                q = (knot + delta).clone().requires_grad_(True)
                y = self.spline(q, values)
                first = torch.autograd.grad(y, q, create_graph=True)[0]
                second = torch.autograd.grad(first, q)[0]
                sides.append(torch.stack([y, first, second]))
            torch.testing.assert_close(sides[0], sides[1], atol=5e-6, rtol=0)

    def test_coordinate_and_value_gradcheck_and_gradgradcheck(self) -> None:
        query = torch.tensor([-1.0, 0.3, 2.7, 9.0], dtype=DTYPE, requires_grad=True)
        values = torch.randn(6, 2, dtype=DTYPE, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(self.spline, (query, values)))
        self.assertTrue(torch.autograd.gradgradcheck(self.spline, (query, values)))

    def test_invalid_knots(self) -> None:
        for knots in [[], [1.0, 1.0], [2.0, 1.0], [0.0, float("nan")]]:
            with self.assertRaises(ValueError):
                NaturalCubicSpline(torch.tensor(knots))


class _IntegerShellReference(nn.Module):
    """Independent full-grid implementation of the original shell multiplier."""

    def __init__(self, layer: MetricEqGINOSpectralConv3D) -> None:
        super().__init__()
        self.radial_weight = nn.Parameter(layer.radial_weight.detach().clone())
        self.groups = layer.groups
        self.n_modes = layer.n_modes
        self.register_buffer("radii", layer.shell_spline.knots.clone())

    def forward(self, field: torch.Tensor, cell=None) -> torch.Tensor:
        axes = [
            torch.fft.fftfreq(s, d=1 / s, dtype=field.dtype, device=field.device)
            for s in field.shape[-3:]
        ]
        modes = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        mask = (modes.abs() < modes.new_tensor(self.n_modes)).all(-1)
        index = torch.searchsorted(self.radii, modes.square().sum(-1).contiguous())
        index = index.clamp(max=self.radii.numel() - 1)
        weights = self.radial_weight[..., index] * mask
        fourier = torch.fft.fftn(field, dim=(-3, -2, -1))
        fourier = fourier.reshape(field.shape[0], self.groups, -1, *field.shape[-3:])
        out = torch.einsum("bgizxy,giozxy->bgozxy", fourier, weights.to(fourier.dtype))
        return torch.fft.ifftn(out.reshape_as(field), dim=(-3, -2, -1)).real


class ShellMetricEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(517)
        self.length = 12.429
        self.cell = torch.eye(3, dtype=DTYPE) * self.length

    def test_original_shell_field_gradients_and_optimizer_updates(self) -> None:
        metric = FNOFieldOperator3D(
            2,
            (4, 4, 4),
            hidden_channels=4,
            n_layers=2,
            spectral_symmetry="metric_eqgino",
            spectral_groups=2,
            metric_reference_length=self.length,
        ).to(dtype=DTYPE)
        original = copy.deepcopy(metric)
        for block in original.fno.blocks:
            block.spectral = _IntegerShellReference(block.spectral)
        params = [dict(m.named_parameters()) for m in [metric, original]]
        self.assertEqual(set(params[0]), set(params[1]))
        optimizers = [
            torch.optim.Adam(m.parameters(), lr=3e-4) for m in [metric, original]
        ]
        field = torch.randn(1, 2, 8, 8, 8, dtype=DTYPE, requires_grad=True)
        for _ in range(3):
            predictions, gradients = [], []
            for model, optimizer in zip([metric, original], optimizers):
                optimizer.zero_grad()
                response = model(field, cell=self.cell)
                energy = (field * response).mean() / 2
                gradient = torch.autograd.grad(energy, field, create_graph=True)[0]
                loss = (response - 0.1).square().mean() + (
                    100 * gradient - 0.02
                ).square().mean()
                loss.backward()
                predictions.append(response.detach())
                gradients.append(gradient.detach())
                optimizer.step()
            torch.testing.assert_close(
                predictions[0], predictions[1], atol=1e-12, rtol=1e-10
            )
            torch.testing.assert_close(
                gradients[0], gradients[1], atol=1e-13, rtol=1e-10
            )
            for key in params[0]:
                torch.testing.assert_close(
                    params[0][key], params[1][key], atol=1e-11, rtol=1e-9
                )

    def test_shell_parameters_are_independent_at_reference_cell(self) -> None:
        layer = MetricEqGINOSpectralConv3D(
            16, 16, (4, 4, 4), groups=4, reference_length=self.length
        ).to(dtype=DTYPE)
        self.assertEqual(layer.radial_weight.shape, (4, 4, 4, 19))
        q = layer.shell_spline.knots * layer.reference_wavenumber_squared
        weights = layer.radial_weights(q)
        torch.testing.assert_close(
            weights, layer.radial_weight.movedim(-1, 0), atol=1e-14, rtol=0
        )
        grad = torch.autograd.grad(weights[3].sum(), layer.radial_weight)[0]
        expected = torch.zeros_like(grad)
        expected[..., 3] = 1
        torch.testing.assert_close(grad, expected, atol=1e-14, rtol=0)

    def test_cell_gradient_and_second_order_parameter_gradient(self) -> None:
        layer = MetricEqGINOSpectralConv3D(
            2, 2, (2, 2, 2), reference_length=self.length
        ).to(dtype=DTYPE)
        field = torch.randn(1, 2, 4, 4, 4, dtype=DTYPE)
        cell = torch.tensor(
            [[10.0, 0.3, 0.1], [0.2, 13.0, -0.4], [0.0, 0.2, 11.0]],
            dtype=DTYPE,
            requires_grad=True,
        )
        self.assertTrue(torch.autograd.gradcheck(lambda c: layer(field, c), (cell,)))
        self.assertTrue(
            torch.autograd.gradgradcheck(lambda c: layer(field, c), (cell,))
        )
        energy = layer(field, cell).square().mean()
        cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]
        parameter_grad = torch.autograd.grad(
            cell_grad.square().sum(), layer.radial_weight
        )[0]
        self.assertTrue(torch.isfinite(parameter_grad).all())
        self.assertGreater(parameter_grad.abs().max().item(), 0.0)

    def test_anchors_do_not_follow_input_cell_and_state_roundtrip(self) -> None:
        layer = MetricEqGINOSpectralConv3D(
            2, 2, (3, 3, 3), reference_length=self.length
        ).to(dtype=DTYPE)
        field = torch.randn(1, 2, 6, 6, 6, dtype=DTYPE)
        original = {k: v.clone() for k, v in layer.state_dict().items()}
        first = layer(field, self.cell)
        second = layer(field, self.cell * 0.9)
        self.assertFalse(torch.allclose(first, second))
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, original[key], atol=0, rtol=0)
        restored = MetricEqGINOSpectralConv3D(2, 2, (3, 3, 3), reference_length=1.0).to(
            dtype=DTYPE
        )
        restored.load_state_dict(original)
        torch.testing.assert_close(
            restored(field, self.cell * 0.9), second, atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
