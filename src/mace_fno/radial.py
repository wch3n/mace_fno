"""Differentiable interpolation of independently learned radial coefficients."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NaturalCubicSpline(nn.Module):
    """Natural cubic interpolation with tangent-linear extrapolation.

    Knots are fixed; values and query coordinates remain differentiable. The
    precomputed map converts knot values into second derivatives. Natural
    endpoint conditions make the linear continuation C2 at both endpoints.
    A single knot represents a constant function, and two knots a line.
    """

    def __init__(self, knots: Tensor) -> None:
        super().__init__()
        knots = torch.as_tensor(knots, dtype=torch.float64).detach().clone()
        if knots.ndim != 1 or not knots.numel():
            raise ValueError("spline knots must be a nonempty vector")
        if not torch.isfinite(knots).all() or not (knots.diff() > 0).all():
            raise ValueError("spline knots must be finite and strictly increasing")
        count = knots.numel()
        system = torch.eye(count, dtype=knots.dtype, device=knots.device)
        rhs = torch.zeros_like(system)
        if count > 2:
            widths = knots.diff()
            row = torch.arange(1, count - 1, device=knots.device)
            system[row, row - 1] = widths[:-1]
            system[row, row] = 2 * (widths[:-1] + widths[1:])
            system[row, row + 1] = widths[1:]
            rhs[row, row - 1] = 6 / widths[:-1]
            rhs[row, row] = -6 * (1 / widths[:-1] + 1 / widths[1:])
            rhs[row, row + 1] = 6 / widths[1:]
        self.register_buffer("knots", knots)
        self.register_buffer("curvature_map", torch.linalg.solve(system, rhs))

    def forward(self, query: Tensor, values: Tensor) -> Tensor:
        """Interpolate ``values[n_knots, ...]`` at ``query[...]``."""
        if values.ndim < 1 or values.shape[0] != self.knots.numel():
            raise ValueError("the first values dimension must match spline knots")
        knots = self.knots.to(device=query.device, dtype=query.dtype)
        flat = values.reshape(values.shape[0], -1)
        points = query.reshape(-1, 1)
        if knots.numel() == 1:
            output = flat[0] + points * 0
        else:
            curvature = self.curvature_map.to(values) @ flat
            # Evaluate cubics only inside the knot interval. Outside, continue
            # the endpoint tangent instead of an unbounded cubic polynomial.
            inside = points.clamp(knots[0], knots[-1])
            index = torch.searchsorted(knots, inside[:, 0].contiguous(), right=True)
            index = (index - 1).clamp(0, knots.numel() - 2)
            width = (knots[index + 1] - knots[index])[:, None]
            a = (knots[index + 1, None] - inside) / width
            b = (inside - knots[index, None]) / width
            output = a * flat[index] + b * flat[index + 1]
            output = output + (width.square() / 6) * (
                (a.pow(3) - a) * curvature[index]
                + (b.pow(3) - b) * curvature[index + 1]
            )
            first_width, last_width = knots[1] - knots[0], knots[-1] - knots[-2]
            first_slope = (flat[1] - flat[0]) / first_width - first_width * (
                2 * curvature[0] + curvature[1]
            ) / 6
            last_slope = (flat[-1] - flat[-2]) / last_width + last_width * (
                curvature[-2] + 2 * curvature[-1]
            ) / 6
            output = torch.where(
                points < knots[0], flat[0] + (points - knots[0]) * first_slope, output
            )
            output = torch.where(
                points > knots[-1], flat[-1] + (points - knots[-1]) * last_slope, output
            )
        return output.reshape(query.shape + values.shape[1:])
