"""Small geometry utilities shared by the particle-mesh modules."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_cell(cell: Tensor) -> None:
    """Validate the row-vector cell convention used throughout the package."""
    if cell.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3); received {tuple(cell.shape)}")
    if not torch.is_floating_point(cell):
        raise TypeError("cell must be a floating-point tensor")


def in_plane_area(cell: Tensor) -> Tensor:
    """Return the area spanned by the first two row vectors of ``cell``."""
    validate_cell(cell)
    normal = torch.linalg.cross(cell[0], cell[1])
    area = torch.linalg.vector_norm(normal)
    if bool((area <= torch.finfo(cell.dtype).eps).detach().cpu()):
        raise ValueError("the first two cell vectors must span a non-zero area")
    return area


def fractional_coordinates(positions: Tensor, cell: Tensor) -> Tensor:
    """Convert Cartesian positions to fractional row-vector coordinates."""
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(
            "positions must have shape (n_atoms, 3); "
            f"received {tuple(positions.shape)}"
        )
    validate_cell(cell)
    return torch.linalg.solve(cell.transpose(0, 1), positions.transpose(0, 1)).transpose(
        0, 1
    )


def reciprocal_vectors_2d(cell: Tensor) -> tuple[Tensor, Tensor]:
    """Return reciprocal vectors dual to the first two cell vectors.

    The result remains valid when the periodic plane is tilted relative to the
    Cartesian axes. The third cell vector is not used to define the planar
    reciprocal basis.
    """
    area = in_plane_area(cell)
    normal = torch.linalg.cross(cell[0], cell[1])
    area_squared = area.square()
    b1 = (2.0 * math.pi) * torch.linalg.cross(cell[1], normal) / area_squared
    b2 = (2.0 * math.pi) * torch.linalg.cross(normal, cell[0]) / area_squared
    return b1, b2


def mesh_cell_area(cell: Tensor, grid_shape: tuple[int, int]) -> Tensor:
    """Return the real-space area represented by one mesh point."""
    nx, ny = grid_shape
    return in_plane_area(cell) / (nx * ny)

