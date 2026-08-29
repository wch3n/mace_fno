"""Direct reciprocal-space references for verification and convergence tests."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .geometry import in_plane_area, reciprocal_vectors_2d


def direct_planar_coulomb_energy(
    positions: Tensor,
    values: Tensor,
    cell: Tensor,
    max_modes: tuple[int, int],
) -> Tensor:
    """Return the truncated point-particle planar reciprocal energy.

    This reference deliberately uses direct structure-factor evaluation rather
    than a mesh. Its computational cost is quadratic in atoms and retained
    modes, which is acceptable for unit tests and small diagnostics.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != positions.shape[0]:
        raise ValueError("values must have shape (n_atoms,) or (n_atoms, channels)")
    if len(max_modes) != 2 or min(max_modes) < 0:
        raise ValueError("max_modes must contain two non-negative integers")

    mx = torch.arange(
        -max_modes[0], max_modes[0] + 1, device=positions.device, dtype=positions.dtype
    )
    my = torch.arange(
        -max_modes[1], max_modes[1] + 1, device=positions.device, dtype=positions.dtype
    )
    mode_x, mode_y = torch.meshgrid(mx, my, indexing="ij")
    keep = (mode_x != 0) | (mode_y != 0)
    mode_x = mode_x[keep]
    mode_y = mode_y[keep]

    b1, b2 = reciprocal_vectors_2d(cell)
    wavevectors = mode_x[:, None] * b1 + mode_y[:, None] * b2
    magnitude = torch.linalg.vector_norm(wavevectors, dim=-1)
    phase = positions @ wavevectors.transpose(0, 1)
    phase_factor = torch.exp(torch.complex(torch.zeros_like(phase), -phase))
    structure_factor = torch.einsum(
        "nc,nk->ck", values.to(dtype=phase_factor.dtype), phase_factor
    )
    kernel = (2.0 * math.pi) / magnitude
    return 0.5 * (kernel[None, :] * structure_factor.abs().square()).sum() / in_plane_area(
        cell
    )
