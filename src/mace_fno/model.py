"""Energy-conserving composition of particle assignment and field operator."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .fno import FNOFieldOperator, FNOFieldOperator2p5D, FNOFieldOperator3d
from .particle_mesh import (
    PeriodicParticleMesh2D,
    PeriodicParticleMesh3D,
    SlabParticleMesh2p5D,
)
from .spectral import (
    PlanarCoulombOperator,
    mesh_interaction_energy,
    mesh_interaction_energy_3d,
    slab_mesh_interaction_energy,
)


class ParticleMeshEnergy(nn.Module):
    """Compose particle assignment with an interchangeable mesh-field operator."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        field_operator: nn.Module,
        *,
        check_neutrality: bool = True,
        neutrality_tolerance: float = 1.0e-10,
    ) -> None:
        super().__init__()
        self.assignment = PeriodicParticleMesh2D(grid_shape)
        self.field_operator = field_operator
        self.check_neutrality = check_neutrality
        self.neutrality_tolerance = float(neutrality_tolerance)

    def _validate_neutrality(self, values: Tensor) -> None:
        values_2d = values[:, None] if values.ndim == 1 else values
        total = values_2d.sum(dim=0).abs()
        scale = values_2d.abs().sum(dim=0).clamp_min(1.0)
        invalid = total > self.neutrality_tolerance * scale
        if bool(invalid.any().detach().cpu()):
            raise ValueError(
                "every input channel must be neutral when check_neutrality=True; "
                f"channel sums are {values_2d.sum(dim=0).detach().cpu().tolist()}"
            )

    def forward(
        self,
        positions: Tensor,
        values: Tensor,
        cell: Tensor,
        *,
        batch: Tensor | None = None,
        return_fields: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if self.check_neutrality:
            self._validate_neutrality(values)
        density = self.assignment(positions, values, cell, batch=batch)
        potential = self.field_operator(density, cell)
        energy = mesh_interaction_energy(density, potential, cell)
        if return_fields:
            return energy, density, potential
        return energy


class ParticleMeshLongRange(ParticleMeshEnergy):
    """Analytic planar Coulomb specialization of :class:`ParticleMeshEnergy`."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        max_modes: tuple[int, int] | None = None,
        *,
        check_neutrality: bool = True,
        neutrality_tolerance: float = 1.0e-10,
        deconvolve_assignment: bool = True,
    ) -> None:
        operator = PlanarCoulombOperator(
            max_modes=max_modes,
            deconvolve_assignment=deconvolve_assignment,
        )
        super().__init__(
            grid_shape,
            operator,
            check_neutrality=check_neutrality,
            neutrality_tolerance=neutrality_tolerance,
        )


class LearnedParticleMeshLongRange(ParticleMeshEnergy):
    """Particle-mesh long-range energy using a learned fixed-cell 2D FNO."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        channels: int,
        n_modes: tuple[int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        architecture: str = "nonlinear",
        check_neutrality: bool = True,
        neutrality_tolerance: float = 1.0e-10,
    ) -> None:
        operator = FNOFieldOperator(
            channels=channels,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            projection_channels=projection_channels,
            architecture=architecture,
        )
        super().__init__(
            grid_shape,
            operator,
            check_neutrality=check_neutrality,
            neutrality_tolerance=neutrality_tolerance,
        )


class ParticleMeshEnergy2p5D(nn.Module):
    """Conservative particle-mesh energy on periodic x/y and finite z layers."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        z_extent: float,
        field_operator: nn.Module,
        *,
        z_center: str = "mean",
        lateral_interlacing: int = 1,
        check_neutrality: bool = True,
        neutrality_tolerance: float = 1.0e-10,
    ) -> None:
        super().__init__()
        self.assignment = SlabParticleMesh2p5D(
            grid_shape,
            z_extent=z_extent,
            z_center=z_center,
        )
        self.field_operator = field_operator
        if lateral_interlacing not in {1, 2}:
            raise ValueError("lateral_interlacing must be 1 or 2")
        self.lateral_interlacing = int(lateral_interlacing)
        self.check_neutrality = bool(check_neutrality)
        self.neutrality_tolerance = float(neutrality_tolerance)

    def _validate_neutrality(self, values: Tensor) -> None:
        values_2d = values[:, None] if values.ndim == 1 else values
        total = values_2d.sum(dim=0).abs()
        scale = values_2d.abs().sum(dim=0).clamp_min(1.0)
        invalid = total > self.neutrality_tolerance * scale
        if bool(invalid.any().detach().cpu()):
            raise ValueError(
                "every input channel must be neutral when check_neutrality=True; "
                f"channel sums are {values_2d.sum(dim=0).detach().cpu().tolist()}"
            )

    def forward(
        self,
        positions: Tensor,
        values: Tensor,
        cell: Tensor,
        *,
        batch: Tensor | None = None,
        return_fields: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if self.check_neutrality:
            self._validate_neutrality(values)
        if self.lateral_interlacing == 1:
            density = self.assignment(positions, values, cell, batch=batch)
            potential = self.field_operator(density, cell)
            energy = slab_mesh_interaction_energy(
                density,
                potential,
                cell,
                self.assignment.z_extent,
            )
        else:
            if return_fields:
                raise ValueError(
                    "return_fields is unavailable with lateral interlacing because "
                    "the fields use different mesh origins"
                )
            unbatched = batch is None
            if unbatched:
                if cell.shape != (3, 3):
                    raise ValueError("an unbatched cell must have shape (3, 3)")
                cells = cell.unsqueeze(0)
                batch_indices = torch.zeros(
                    positions.shape[0], dtype=torch.long, device=positions.device
                )
            else:
                cells = cell
                batch_indices = batch
                if cells.ndim != 3 or cells.shape[1:] != (3, 3):
                    raise ValueError("batched cells must have shape (n_graphs, 3, 3)")
            num_graphs = cells.shape[0]
            _, nx, ny = self.assignment.grid_shape
            offsets = [
                (ix / (self.lateral_interlacing * nx)) * cells[:, 0]
                + (iy / (self.lateral_interlacing * ny)) * cells[:, 1]
                for ix in range(self.lateral_interlacing)
                for iy in range(self.lateral_interlacing)
            ]
            replica_count = len(offsets)
            replicated_positions = torch.cat(
                [
                    positions + offset.index_select(0, batch_indices)
                    for offset in offsets
                ],
                dim=0,
            )
            replicated_values = torch.cat([values] * replica_count, dim=0)
            replicated_batch = torch.cat(
                [
                    batch_indices + replica * num_graphs
                    for replica in range(replica_count)
                ]
            )
            replicated_cells = cells.repeat(replica_count, 1, 1)
            density = self.assignment(
                replicated_positions,
                replicated_values,
                replicated_cells,
                batch=replicated_batch,
            )
            potential = self.field_operator(density, replicated_cells)
            replica_energy = slab_mesh_interaction_energy(
                density,
                potential,
                replicated_cells,
                self.assignment.z_extent,
            )
            energy = replica_energy.reshape(replica_count, num_graphs).mean(dim=0)
            if unbatched:
                energy = energy.squeeze(0)
        if return_fields:
            return energy, density, potential
        return energy


class LearnedParticleMeshLongRange2p5D(ParticleMeshEnergy2p5D):
    """Learned slab operator with 2D FFTs and explicit nonperiodic z layers."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        z_extent: float,
        channels: int,
        n_modes: tuple[int, int],
        *,
        z_center: str = "mean",
        lateral_interlacing: int = 1,
        z_kernel_size: int = 3,
        z_mixing: str = "local",
        planar_symmetry: str = "none",
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        architecture: str = "nonlinear",
        check_neutrality: bool = True,
        neutrality_tolerance: float = 1.0e-10,
    ) -> None:
        nz, _, _ = grid_shape
        operator = FNOFieldOperator2p5D(
            channels=channels,
            n_z=nz,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            projection_channels=projection_channels,
            z_kernel_size=z_kernel_size,
            z_mixing=z_mixing,
            planar_symmetry=planar_symmetry,
            architecture=architecture,
        )
        super().__init__(
            grid_shape,
            z_extent,
            operator,
            z_center=z_center,
            lateral_interlacing=lateral_interlacing,
            check_neutrality=check_neutrality,
            neutrality_tolerance=neutrality_tolerance,
        )


class ParticleMeshEnergy3D(nn.Module):
    """Conservative particle-mesh energy for a fully periodic 3D cell.

    With ``volume_interlacing=2``, the returned energy is the average over the
    eight combinations of zero and half-grid shifts in the three periodic
    directions.  The replicas are evaluated as one enlarged mesh batch so the
    field operator is called only once.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        field_operator: nn.Module,
        *,
        volume_interlacing: int = 1,
        check_neutrality: bool = True,
        neutrality_tolerance: float = 1.0e-10,
    ) -> None:
        super().__init__()
        self.assignment = PeriodicParticleMesh3D(grid_shape)
        self.field_operator = field_operator
        if volume_interlacing not in {1, 2}:
            raise ValueError("volume_interlacing must be 1 or 2")
        self.volume_interlacing = int(volume_interlacing)
        self.check_neutrality = bool(check_neutrality)
        self.neutrality_tolerance = float(neutrality_tolerance)

    def _validate_neutrality(self, values: Tensor) -> None:
        values_2d = values[:, None] if values.ndim == 1 else values
        total = values_2d.sum(dim=0).abs()
        scale = values_2d.abs().sum(dim=0).clamp_min(1.0)
        invalid = total > self.neutrality_tolerance * scale
        if bool(invalid.any().detach().cpu()):
            raise ValueError(
                "every input channel must be neutral when check_neutrality=True; "
                f"channel sums are {values_2d.sum(dim=0).detach().cpu().tolist()}"
            )

    def forward(
        self,
        positions: Tensor,
        values: Tensor,
        cell: Tensor,
        *,
        batch: Tensor | None = None,
        return_fields: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if self.check_neutrality:
            self._validate_neutrality(values)
        if self.volume_interlacing == 1:
            density = self.assignment(positions, values, cell, batch=batch)
            potential = self.field_operator(density, cell)
            energy = mesh_interaction_energy_3d(density, potential, cell)
        else:
            if return_fields:
                raise ValueError(
                    "return_fields is unavailable with volume interlacing because "
                    "the fields use different mesh origins"
                )
            unbatched = batch is None
            if unbatched:
                if cell.shape != (3, 3):
                    raise ValueError("an unbatched cell must have shape (3, 3)")
                cells = cell.unsqueeze(0)
                batch_indices = torch.zeros(
                    positions.shape[0], dtype=torch.long, device=positions.device
                )
            else:
                cells = cell
                batch_indices = batch
                if cells.ndim != 3 or cells.shape[1:] != (3, 3):
                    raise ValueError("batched cells must have shape (n_graphs, 3, 3)")
            num_graphs = cells.shape[0]
            nz, nx, ny = self.assignment.grid_shape
            offsets = [
                (iz / (self.volume_interlacing * nz)) * cells[:, 2]
                + (ix / (self.volume_interlacing * nx)) * cells[:, 0]
                + (iy / (self.volume_interlacing * ny)) * cells[:, 1]
                for iz in range(self.volume_interlacing)
                for ix in range(self.volume_interlacing)
                for iy in range(self.volume_interlacing)
            ]
            replica_count = len(offsets)
            replicated_positions = torch.cat(
                [
                    positions + offset.index_select(0, batch_indices)
                    for offset in offsets
                ],
                dim=0,
            )
            replicated_values = torch.cat([values] * replica_count, dim=0)
            replicated_batch = torch.cat(
                [
                    batch_indices + replica * num_graphs
                    for replica in range(replica_count)
                ]
            )
            replicated_cells = cells.repeat(replica_count, 1, 1)
            density = self.assignment(
                replicated_positions,
                replicated_values,
                replicated_cells,
                batch=replicated_batch,
            )
            potential = self.field_operator(density, replicated_cells)
            replica_energy = mesh_interaction_energy_3d(
                density,
                potential,
                replicated_cells,
            )
            energy = replica_energy.reshape(replica_count, num_graphs).mean(dim=0)
            if unbatched:
                energy = energy.squeeze(0)
        if return_fields:
            return energy, density, potential
        return energy


class LearnedParticleMeshLongRange3D(ParticleMeshEnergy3D):
    """Learned fixed-cell FNO correction on a fully periodic 3D mesh."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        channels: int,
        n_modes: tuple[int, int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        architecture: str = "nonlinear",
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        volume_interlacing: int = 1,
        check_neutrality: bool = True,
        neutrality_tolerance: float = 1.0e-10,
    ) -> None:
        operator = FNOFieldOperator3d(
            channels=channels,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            projection_channels=projection_channels,
            architecture=architecture,
            spectral_symmetry=spectral_symmetry,
            spectral_groups=spectral_groups,
        )
        super().__init__(
            grid_shape,
            operator,
            volume_interlacing=volume_interlacing,
            check_neutrality=check_neutrality,
            neutrality_tolerance=neutrality_tolerance,
        )
