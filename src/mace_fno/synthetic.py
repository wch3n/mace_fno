"""Synthetic field data for verifying learned long-range operators."""

from __future__ import annotations

import torch
from torch import Tensor

from .particle_mesh import PeriodicParticleMesh2D
from .spectral import PlanarCoulombOperator


@torch.no_grad()
def generate_planar_coulomb_fields(
    n_samples: int,
    n_atoms: int,
    channels: int,
    cell: Tensor,
    grid_shape: tuple[int, int],
    max_modes: tuple[int, int],
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Generate neutral random particle densities and analytic potentials.

    Returns tensors with shape ``(n_samples, channels, nx, ny)``. The function
    is intended for deterministic software verification, not as a physical
    training-set generator for an interatomic potential.
    """
    if n_samples < 1 or n_atoms < 2 or channels < 1:
        raise ValueError(
            "n_samples and channels must be positive, and n_atoms at least 2"
        )

    generator = torch.Generator(device=cell.device)
    generator.manual_seed(seed)
    assignment = PeriodicParticleMesh2D(grid_shape)
    analytic_operator = PlanarCoulombOperator(max_modes=max_modes)
    densities: list[Tensor] = []
    potentials: list[Tensor] = []

    for _ in range(n_samples):
        fractional = torch.rand(
            (n_atoms, 3),
            generator=generator,
            device=cell.device,
            dtype=cell.dtype,
        )
        fractional[:, 2] = 0.0
        positions = fractional @ cell
        values = torch.randn(
            (n_atoms, channels),
            generator=generator,
            device=cell.device,
            dtype=cell.dtype,
        )
        values = values - values.mean(dim=0, keepdim=True)
        density = assignment(positions, values, cell)
        densities.append(density)
        potentials.append(analytic_operator(density, cell))

    return torch.stack(densities), torch.stack(potentials)

