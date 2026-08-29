"""Smooth atom-to-mesh assignment for periodic planes, slabs, and bulk."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .geometry import fractional_coordinates, mesh_cell_area


def _cubic_bspline_weights(t: Tensor) -> Tensor:
    """Cardinal cubic B-spline weights for offsets ``(-1, 0, 1, 2)``."""
    one_minus_t = 1.0 - t
    w0 = one_minus_t.pow(3) / 6.0
    w1 = (3.0 * t.pow(3) - 6.0 * t.square() + 4.0) / 6.0
    w2 = (-3.0 * t.pow(3) + 3.0 * t.square() + 3.0 * t + 1.0) / 6.0
    w3 = t.pow(3) / 6.0
    return torch.stack((w0, w1, w2, w3), dim=-1)


class PeriodicParticleMesh2D(nn.Module):
    """Deposit atom-centred scalar features onto a periodic 2D density mesh.

    Positions use Cartesian coordinates and cells use ASE's row-vector
    convention. The returned tensor has shape ``(channels, nx, ny)`` and is a
    density: integrating it with the mesh-cell area recovers the sum of the
    atom-centred input values.
    """

    def __init__(self, grid_shape: tuple[int, int]) -> None:
        super().__init__()
        if len(grid_shape) != 2 or min(grid_shape) < 4:
            raise ValueError("grid_shape must contain two integers, each at least 4")
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))

    def forward(
        self,
        positions: Tensor,
        values: Tensor,
        cell: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if not torch.is_floating_point(positions):
            raise TypeError("positions must be a floating-point tensor")
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != positions.shape[0]:
            raise ValueError(
                "values must have shape (n_atoms,) or (n_atoms, channels)"
            )
        if not torch.is_floating_point(values):
            raise TypeError("values must be a floating-point tensor")
        if values.device != positions.device or cell.device != positions.device:
            raise ValueError("positions, values, and cell must be on the same device")
        if values.dtype != positions.dtype or cell.dtype != positions.dtype:
            raise ValueError("positions, values, and cell must use the same dtype")

        unbatched = batch is None
        if unbatched:
            if cell.shape != (3, 3):
                raise ValueError("an unbatched cell must have shape (3, 3)")
            batch = torch.zeros(
                positions.shape[0], dtype=torch.long, device=positions.device
            )
            cells = cell.unsqueeze(0)
        else:
            if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
                raise ValueError("batch must have shape (n_atoms,)")
            if batch.dtype != torch.long:
                raise TypeError("batch must use torch.long indices")
            if batch.device != positions.device:
                raise ValueError("batch and positions must be on the same device")
            if cell.ndim != 3 or cell.shape[1:] != (3, 3):
                raise ValueError("batched cells must have shape (n_graphs, 3, 3)")
            cells = cell
        num_graphs = cells.shape[0]
        if batch.numel() == 0 or int(batch.min().detach().cpu()) != 0:
            raise ValueError("batch must contain consecutive non-negative graph indices")
        if int(batch.max().detach().cpu()) + 1 != num_graphs:
            raise ValueError("batch graph count must match the number of cells")

        nx, ny = self.grid_shape
        atom_cells = cells.index_select(0, batch)
        fractional = torch.linalg.solve(
            atom_cells.transpose(1, 2), positions.unsqueeze(-1)
        ).squeeze(-1)[:, :2]
        wrapped = fractional - torch.floor(fractional)
        scale = positions.new_tensor((nx, ny))
        mesh_coordinates = wrapped * scale

        lower = torch.floor(mesh_coordinates)
        local = mesh_coordinates - lower
        lower_index = lower.to(torch.long)

        wx = _cubic_bspline_weights(local[:, 0])
        wy = _cubic_bspline_weights(local[:, 1])
        offsets = torch.tensor((-1, 0, 1, 2), device=positions.device)
        ix = (lower_index[:, 0, None] + offsets[None, :]) % nx
        iy = (lower_index[:, 1, None] + offsets[None, :]) % ny

        graph_offsets = batch[:, None, None] * (nx * ny)
        flat_indices = (
            graph_offsets + ix[:, :, None] * ny + iy[:, None, :]
        ).reshape(-1)
        weights = (wx[:, :, None] * wy[:, None, :]).reshape(values.shape[0], 16)

        normals = torch.linalg.cross(cells[:, 0], cells[:, 1])
        point_areas = torch.linalg.vector_norm(normals, dim=1) / (nx * ny)
        if bool((point_areas <= torch.finfo(cells.dtype).eps).any().detach().cpu()):
            raise ValueError("every in-plane cell must span a non-zero area")
        atom_point_areas = point_areas.index_select(0, batch)
        source = (
            values[:, :, None]
            * weights[:, None, :]
            / atom_point_areas[:, None, None]
        )
        source = source.permute(1, 0, 2).reshape(values.shape[1], -1)

        density_flat = positions.new_zeros(
            (values.shape[1], num_graphs * nx * ny)
        )
        density_flat = density_flat.index_add(1, flat_indices, source)
        density = density_flat.reshape(values.shape[1], num_graphs, nx, ny)
        density = density.permute(1, 0, 2, 3)
        return density[0] if unbatched else density


class PeriodicParticleMesh3D(nn.Module):
    """Deposit atom-centred features on a fully periodic three-dimensional mesh.

    Cartesian positions are converted to fractional coordinates using the full
    triclinic cell and wrapped along all three lattice directions.  The mesh
    layout is ``(channels, nz, nx, ny)`` (or batch first), matching the tensor
    convention used by the 2.5D model while making z periodic.  Integrating the
    returned density with voxel volume ``|det(cell)|/(nz*nx*ny)`` recovers the
    sum of each atom-centred input channel.
    """

    def __init__(self, grid_shape: tuple[int, int, int]) -> None:
        super().__init__()
        if len(grid_shape) != 3 or min(grid_shape) < 4:
            raise ValueError(
                "grid_shape must be (nz, nx, ny), with every dimension at least 4"
            )
        self.grid_shape = tuple(int(size) for size in grid_shape)

    def forward(
        self,
        positions: Tensor,
        values: Tensor,
        cell: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3)")
        if not torch.is_floating_point(positions):
            raise TypeError("positions must be a floating-point tensor")
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != positions.shape[0]:
            raise ValueError(
                "values must have shape (n_atoms,) or (n_atoms, channels)"
            )
        if not torch.is_floating_point(values):
            raise TypeError("values must be a floating-point tensor")
        if values.device != positions.device or cell.device != positions.device:
            raise ValueError("positions, values, and cell must be on the same device")
        if values.dtype != positions.dtype or cell.dtype != positions.dtype:
            raise ValueError("positions, values, and cell must use the same dtype")

        unbatched = batch is None
        if unbatched:
            if cell.shape != (3, 3):
                raise ValueError("an unbatched cell must have shape (3, 3)")
            batch = torch.zeros(
                positions.shape[0], dtype=torch.long, device=positions.device
            )
            cells = cell.unsqueeze(0)
        else:
            if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
                raise ValueError("batch must have shape (n_atoms,)")
            if batch.dtype != torch.long:
                raise TypeError("batch must use torch.long indices")
            if batch.device != positions.device:
                raise ValueError("batch and positions must be on the same device")
            if cell.ndim != 3 or cell.shape[1:] != (3, 3):
                raise ValueError("batched cells must have shape (n_graphs, 3, 3)")
            cells = cell

        num_graphs = cells.shape[0]
        if batch.numel() == 0 or int(batch.min().detach().cpu()) != 0:
            raise ValueError("batch must contain consecutive non-negative graph indices")
        if int(batch.max().detach().cpu()) + 1 != num_graphs:
            raise ValueError("batch graph count must match the number of cells")

        volumes = torch.linalg.det(cells).abs()
        if bool((volumes <= torch.finfo(cells.dtype).eps).any().detach().cpu()):
            raise ValueError("every periodic cell must have non-zero volume")

        nz, nx, ny = self.grid_shape
        atom_cells = cells.index_select(0, batch)
        fractional_xyz = torch.linalg.solve(
            atom_cells.transpose(1, 2), positions.unsqueeze(-1)
        ).squeeze(-1)
        wrapped_xyz = fractional_xyz - torch.floor(fractional_xyz)
        # Tensor fields retain the established (z, x, y) spatial layout.
        wrapped_zxy = wrapped_xyz[:, (2, 0, 1)]
        mesh_coordinates = wrapped_zxy * positions.new_tensor((nz, nx, ny))

        lower = torch.floor(mesh_coordinates)
        local = mesh_coordinates - lower
        lower_index = lower.to(torch.long)
        wz = _cubic_bspline_weights(local[:, 0])
        wx = _cubic_bspline_weights(local[:, 1])
        wy = _cubic_bspline_weights(local[:, 2])
        offsets = torch.tensor((-1, 0, 1, 2), device=positions.device)
        iz = (lower_index[:, 0, None] + offsets[None, :]) % nz
        ix = (lower_index[:, 1, None] + offsets[None, :]) % nx
        iy = (lower_index[:, 2, None] + offsets[None, :]) % ny

        graph_offsets = batch[:, None, None, None] * (nz * nx * ny)
        flat_indices = (
            graph_offsets
            + iz[:, :, None, None] * (nx * ny)
            + ix[:, None, :, None] * ny
            + iy[:, None, None, :]
        ).reshape(-1)
        weights = (
            wz[:, :, None, None]
            * wx[:, None, :, None]
            * wy[:, None, None, :]
        ).reshape(values.shape[0], 64)

        point_volumes = volumes / (nz * nx * ny)
        atom_point_volumes = point_volumes.index_select(0, batch)
        source = (
            values[:, :, None]
            * weights[:, None, :]
            / atom_point_volumes[:, None, None]
        )
        source = source.permute(1, 0, 2).reshape(values.shape[1], -1)

        density_flat = positions.new_zeros(
            (values.shape[1], num_graphs * nz * nx * ny)
        )
        density_flat = density_flat.index_add(1, flat_indices, source)
        density = density_flat.reshape(
            values.shape[1], num_graphs, nz, nx, ny
        ).permute(1, 0, 2, 3, 4)
        return density[0] if unbatched else density


class SlabParticleMesh2p5D(nn.Module):
    """Deposit atom features on a mesh periodic in-plane and finite along z.

    The first two cell vectors define the periodic plane. The surface-normal
    coordinate is the Cartesian projection on their unit normal; it is never
    wrapped. Cubic B-spline support that crosses either end of the finite z
    window is accumulated into the boundary voxel, which conserves deposited
    values without introducing periodic coupling between the two surfaces.

    With ``z_center='mean'`` (the default), each graph is centred on its mean
    atomic height. This makes the field invariant to rigid translations normal
    to the surface. ``z_center='cell'`` instead uses the projected centre of the
    third cell vector and is useful when an externally fixed z origin matters.

    The returned density has shape ``(channels, nz, nx, ny)`` when unbatched or
    ``(batch, channels, nz, nx, ny)`` when batched. Integrating it with voxel
    volume ``area*z_extent/(nz*nx*ny)`` recovers the atom-centred input values.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        z_extent: float,
        z_center: str = "mean",
    ) -> None:
        super().__init__()
        if len(grid_shape) != 3 or min(grid_shape) < 4:
            raise ValueError(
                "grid_shape must be (nz, nx, ny), with every dimension at least 4"
            )
        if z_extent <= 0:
            raise ValueError("z_extent must be positive")
        if z_center not in {"mean", "cell"}:
            raise ValueError("z_center must be 'mean' or 'cell'")
        self.grid_shape = tuple(int(size) for size in grid_shape)
        self.z_extent = float(z_extent)
        self.z_center = z_center

    @property
    def voxel_height(self) -> float:
        return self.z_extent / self.grid_shape[0]

    def forward(
        self,
        positions: Tensor,
        values: Tensor,
        cell: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3)")
        if not torch.is_floating_point(positions):
            raise TypeError("positions must be a floating-point tensor")
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != positions.shape[0]:
            raise ValueError(
                "values must have shape (n_atoms,) or (n_atoms, channels)"
            )
        if not torch.is_floating_point(values):
            raise TypeError("values must be a floating-point tensor")
        if values.device != positions.device or cell.device != positions.device:
            raise ValueError("positions, values, and cell must be on the same device")
        if values.dtype != positions.dtype or cell.dtype != positions.dtype:
            raise ValueError("positions, values, and cell must use the same dtype")

        unbatched = batch is None
        if unbatched:
            if cell.shape != (3, 3):
                raise ValueError("an unbatched cell must have shape (3, 3)")
            batch = torch.zeros(
                positions.shape[0], dtype=torch.long, device=positions.device
            )
            cells = cell.unsqueeze(0)
        else:
            if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
                raise ValueError("batch must have shape (n_atoms,)")
            if batch.dtype != torch.long:
                raise TypeError("batch must use torch.long indices")
            if batch.device != positions.device:
                raise ValueError("batch and positions must be on the same device")
            if cell.ndim != 3 or cell.shape[1:] != (3, 3):
                raise ValueError("batched cells must have shape (n_graphs, 3, 3)")
            cells = cell

        num_graphs = cells.shape[0]
        if batch.numel() == 0 or int(batch.min().detach().cpu()) != 0:
            raise ValueError("batch must contain consecutive non-negative graph indices")
        if int(batch.max().detach().cpu()) + 1 != num_graphs:
            raise ValueError("batch graph count must match the number of cells")

        nz, nx, ny = self.grid_shape
        atom_cells = cells.index_select(0, batch)
        fractional = torch.linalg.solve(
            atom_cells.transpose(1, 2), positions.unsqueeze(-1)
        ).squeeze(-1)[:, :2]
        wrapped_xy = fractional - torch.floor(fractional)
        xy_coordinates = wrapped_xy * positions.new_tensor((nx, ny))

        normals = torch.linalg.cross(cells[:, 0], cells[:, 1])
        areas = torch.linalg.vector_norm(normals, dim=1)
        if bool((areas <= torch.finfo(cells.dtype).eps).any().detach().cpu()):
            raise ValueError("every in-plane cell must span a non-zero area")
        unit_normals = normals / areas[:, None]
        atom_normals = unit_normals.index_select(0, batch)
        heights = (positions * atom_normals).sum(dim=1)
        if self.z_center == "mean":
            height_sums = heights.new_zeros(num_graphs).index_add(0, batch, heights)
            counts = heights.new_zeros(num_graphs).index_add(
                0, batch, heights.new_ones(heights.shape[0])
            )
            centres = height_sums / counts
        else:
            centres = 0.5 * (cells[:, 2] * unit_normals).sum(dim=1)
        relative_heights = heights - centres.index_select(0, batch)
        half_extent = 0.5 * self.z_extent
        tolerance = 16.0 * torch.finfo(positions.dtype).eps * max(
            self.z_extent, 1.0
        )
        if bool(
            (relative_heights.abs() > half_extent + tolerance).any().detach().cpu()
        ):
            maximum = relative_heights.abs().max().detach().cpu().item()
            raise ValueError(
                f"an atom lies outside the finite z window: |z-z_center|={maximum:.6g} "
                f"> z_extent/2={half_extent:.6g}"
            )
        z_coordinates = (
            (relative_heights / self.z_extent + 0.5) * nz - 0.5
        )

        mesh_coordinates = torch.cat((z_coordinates[:, None], xy_coordinates), dim=1)
        lower = torch.floor(mesh_coordinates)
        local = mesh_coordinates - lower
        lower_index = lower.to(torch.long)
        wz = _cubic_bspline_weights(local[:, 0])
        wx = _cubic_bspline_weights(local[:, 1])
        wy = _cubic_bspline_weights(local[:, 2])
        offsets = torch.tensor((-1, 0, 1, 2), device=positions.device)
        # z is deliberately clamped, whereas x and y wrap periodically.
        iz = (lower_index[:, 0, None] + offsets[None, :]).clamp(0, nz - 1)
        ix = (lower_index[:, 1, None] + offsets[None, :]) % nx
        iy = (lower_index[:, 2, None] + offsets[None, :]) % ny

        graph_offsets = batch[:, None, None, None] * (nz * nx * ny)
        flat_indices = (
            graph_offsets
            + iz[:, :, None, None] * (nx * ny)
            + ix[:, None, :, None] * ny
            + iy[:, None, None, :]
        ).reshape(-1)
        weights = (
            wz[:, :, None, None]
            * wx[:, None, :, None]
            * wy[:, None, None, :]
        ).reshape(values.shape[0], 64)

        point_volumes = areas * self.z_extent / (nz * nx * ny)
        atom_point_volumes = point_volumes.index_select(0, batch)
        source = (
            values[:, :, None]
            * weights[:, None, :]
            / atom_point_volumes[:, None, None]
        )
        source = source.permute(1, 0, 2).reshape(values.shape[1], -1)

        density_flat = positions.new_zeros(
            (values.shape[1], num_graphs * nz * nx * ny)
        )
        density_flat = density_flat.index_add(1, flat_indices, source)
        density = density_flat.reshape(
            values.shape[1], num_graphs, nz, nx, ny
        ).permute(1, 0, 2, 3, 4)
        return density[0] if unbatched else density
