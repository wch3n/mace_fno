"""Analytic spectral operators used to validate the particle-mesh pathway."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .geometry import mesh_cell_area, reciprocal_vectors_2d


class PlanarCoulombOperator(nn.Module):
    """Apply a truncated planar Coulomb kernel to a periodic density mesh.

    The Fourier-space kernel is ``2*pi/|k|`` for non-zero modes. ``max_modes``
    limits the absolute integer mode retained along each periodic direction,
    matching the low-mode role intended for the later FNO.
    """

    def __init__(
        self,
        max_modes: tuple[int, int] | None = None,
        *,
        deconvolve_assignment: bool = True,
    ) -> None:
        super().__init__()
        if max_modes is not None and (len(max_modes) != 2 or min(max_modes) < 0):
            raise ValueError("max_modes must contain two non-negative integers")
        self.max_modes = (
            None if max_modes is None else (int(max_modes[0]), int(max_modes[1]))
        )
        self.deconvolve_assignment = bool(deconvolve_assignment)

    def kernel(self, cell: Tensor, grid_shape: tuple[int, int]) -> Tensor:
        nx, ny = grid_shape
        mx = torch.fft.fftfreq(nx, d=1.0 / nx, device=cell.device, dtype=cell.dtype)
        my = torch.fft.fftfreq(ny, d=1.0 / ny, device=cell.device, dtype=cell.dtype)
        b1, b2 = reciprocal_vectors_2d(cell)
        wavevectors = mx[:, None, None] * b1 + my[None, :, None] * b2
        magnitude = torch.linalg.vector_norm(wavevectors, dim=-1)

        nonzero = magnitude > 0
        if self.max_modes is not None:
            nonzero = (
                nonzero
                & (mx[:, None].abs() <= self.max_modes[0])
                & (my[None, :].abs() <= self.max_modes[1])
            )
        safe_magnitude = torch.where(nonzero, magnitude, torch.ones_like(magnitude))
        kernel = torch.where(
            nonzero,
            (2.0 * math.pi) / safe_magnitude,
            torch.zeros_like(magnitude),
        )
        if self.deconvolve_assignment:
            # Cubic cardinal B-spline assignment has the main-lobe transfer
            # function sinc(m/n)^4 along each grid direction. Density appears
            # twice in the bilinear energy, hence the squared deconvolution.
            assignment_window = (
                torch.sinc(mx[:, None] / nx).pow(4)
                * torch.sinc(my[None, :] / ny).pow(4)
            )
            safe_window = torch.where(
                nonzero, assignment_window, torch.ones_like(assignment_window)
            )
            kernel = torch.where(nonzero, kernel / safe_window.square(), kernel)
        return kernel

    def forward(self, density: Tensor, cell: Tensor) -> Tensor:
        if density.ndim != 3:
            raise ValueError(
                "density must have shape (channels, nx, ny); "
                f"received {tuple(density.shape)}"
            )
        kernel = self.kernel(cell, (density.shape[-2], density.shape[-1]))
        density_k = torch.fft.fft2(density, dim=(-2, -1))
        potential_k = density_k * kernel[None, :, :]
        return torch.fft.ifft2(potential_k, dim=(-2, -1)).real


def mesh_interaction_energy(density: Tensor, potential: Tensor, cell: Tensor) -> Tensor:
    """Evaluate ``1/2 integral rho*phi dA`` for matching mesh fields."""
    if density.shape != potential.shape or density.ndim not in {3, 4}:
        raise ValueError(
            "density and potential must have matching (channels, nx, ny) or "
            "(batch, channels, nx, ny) shapes"
        )
    grid_shape = (density.shape[-2], density.shape[-1])
    if density.ndim == 3:
        point_area = mesh_cell_area(cell, grid_shape)
        return 0.5 * (density * potential).sum() * point_area
    if cell.ndim != 3 or cell.shape != (density.shape[0], 3, 3):
        raise ValueError("batched cells must have shape (batch, 3, 3)")
    areas = torch.linalg.vector_norm(
        torch.linalg.cross(cell[:, 0], cell[:, 1]), dim=1
    )
    point_areas = areas / (grid_shape[0] * grid_shape[1])
    return 0.5 * (density * potential).sum(dim=(1, 2, 3)) * point_areas


def slab_mesh_interaction_energy(
    density: Tensor,
    potential: Tensor,
    cell: Tensor,
    z_extent: float,
) -> Tensor:
    """Evaluate ``1/2 integral rho*phi dA dz`` on a finite-z slab mesh."""
    if z_extent <= 0:
        raise ValueError("z_extent must be positive")
    if density.shape != potential.shape or density.ndim not in {4, 5}:
        raise ValueError(
            "density and potential must have matching (channels, nz, nx, ny) or "
            "(batch, channels, nz, nx, ny) shapes"
        )
    nz, nx, ny = density.shape[-3:]
    if density.ndim == 4:
        point_volume = mesh_cell_area(cell, (nx, ny)) * (float(z_extent) / nz)
        return 0.5 * (density * potential).sum() * point_volume
    if cell.ndim != 3 or cell.shape != (density.shape[0], 3, 3):
        raise ValueError("batched cells must have shape (batch, 3, 3)")
    areas = torch.linalg.vector_norm(
        torch.linalg.cross(cell[:, 0], cell[:, 1]), dim=1
    )
    point_volumes = areas * float(z_extent) / (nz * nx * ny)
    return 0.5 * (density * potential).sum(dim=(1, 2, 3, 4)) * point_volumes


def mesh_interaction_energy_3d(
    density: Tensor,
    potential: Tensor,
    cell: Tensor,
) -> Tensor:
    """Evaluate ``1/2 integral rho*phi dV`` on a fully periodic 3D mesh."""
    if density.shape != potential.shape or density.ndim not in {4, 5}:
        raise ValueError(
            "density and potential must have matching (channels, nz, nx, ny) or "
            "(batch, channels, nz, nx, ny) shapes"
        )
    nz, nx, ny = density.shape[-3:]
    if density.ndim == 4:
        if cell.shape != (3, 3):
            raise ValueError("an unbatched cell must have shape (3, 3)")
        point_volume = torch.linalg.det(cell).abs() / (nz * nx * ny)
        return 0.5 * (density * potential).sum() * point_volume
    if cell.ndim != 3 or cell.shape != (density.shape[0], 3, 3):
        raise ValueError("batched cells must have shape (batch, 3, 3)")
    point_volumes = torch.linalg.det(cell).abs() / (nz * nx * ny)
    return 0.5 * (density * potential).sum(dim=(1, 2, 3, 4)) * point_volumes
