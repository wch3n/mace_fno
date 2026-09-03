"""Fully periodic 3D Fourier neural operators."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .symmetry import is_cubic_cell


class SpectralConv3d(nn.Module):
    """Learned convolution on truncated modes of a fully periodic 3D field.

    Field tensors use spatial order ``(z, x, y)``.  A real FFT is used along y,
    while independent learned weights cover the positive and negative retained
    modes along z and x.
    """

    _QUADRANTS = ("pp", "pn", "np", "nn")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int, int],
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if len(n_modes) != 3 or min(n_modes) < 1:
            raise ValueError("n_modes must contain three positive integers")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = tuple(int(mode) for mode in n_modes)

        shape = (self.in_channels, self.out_channels, *self.n_modes)
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        for quadrant in self._QUADRANTS:
            self.register_parameter(
                f"weight_{quadrant}_real",
                nn.Parameter(scale * torch.randn(shape)),
            )
            self.register_parameter(
                f"weight_{quadrant}_imag",
                nn.Parameter(scale * torch.randn(shape)),
            )

    @staticmethod
    def _contract(field: Tensor, weight: Tensor) -> Tensor:
        return torch.einsum("bizxy,iozxy->bozxy", field, weight)

    def _weight(self, quadrant: str) -> Tensor:
        return torch.complex(
            getattr(self, f"weight_{quadrant}_real"),
            getattr(self, f"weight_{quadrant}_imag"),
        )

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim != 5 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape (batch, {self.in_channels}, nz, nx, ny); "
                f"received {tuple(field.shape)}"
            )
        nz, nx, ny = field.shape[-3:]
        modes_z, modes_x, modes_y = self.n_modes
        if 2 * modes_z > nz or 2 * modes_x > nx or modes_y > ny // 2 + 1:
            raise ValueError(
                f"n_modes={self.n_modes} is incompatible with grid {(nz, nx, ny)}; "
                "require 2*modes_z <= nz, 2*modes_x <= nx, and "
                "modes_y <= ny//2 + 1"
            )

        field_k = torch.fft.rfftn(field, dim=(-3, -2, -1))
        output_k = field_k.new_zeros(
            (field.shape[0], self.out_channels, nz, nx, ny // 2 + 1)
        )
        z_slices = (slice(0, modes_z), slice(-modes_z, None))
        x_slices = (slice(0, modes_x), slice(-modes_x, None))
        quadrants = ((0, 0, "pp"), (0, 1, "pn"), (1, 0, "np"), (1, 1, "nn"))
        for z_sign, x_sign, name in quadrants:
            selection = (
                slice(None),
                slice(None),
                z_slices[z_sign],
                x_slices[x_sign],
                slice(0, modes_y),
            )
            output_k[selection] = self._contract(field_k[selection], self._weight(name))
        return torch.fft.irfftn(
            output_k,
            s=(nz, nx, ny),
            dim=(-3, -2, -1),
        )


class MetricEqGINOSpectralConv3d(nn.Module):
    """Cell-metric-aware isotropic convolution for periodic scalar fields.

    The layer evaluates a small radial network at the physical wavevector
    magnitude

    ``|k_n|^2 = |2 pi A^{-1} n|^2``.

    Here ``A`` follows ASE's row-vector cell convention and ``n`` is ordered
    consistently with the field's ``(z, x, y)`` mesh.  Dependence on ``|k|^2``
    makes the learned multiplier real and even, preserves Hermitian symmetry,
    and is invariant when the complete cell is rigidly rotated.  The radial
    network is evaluated only on retained modes, so its cost is small relative
    to the full-grid FFT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int, int],
        *,
        groups: int = 1,
        radial_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if len(n_modes) != 3 or min(n_modes) < 1:
            raise ValueError("n_modes must contain three positive integers")
        if groups < 1:
            raise ValueError("groups must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        if radial_hidden_channels < 1:
            raise ValueError("radial_hidden_channels must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = tuple(int(mode) for mode in n_modes)
        self.groups = int(groups)
        self.radial_hidden_channels = int(radial_hidden_channels)
        self.in_channels_per_group = self.in_channels // self.groups
        self.out_channels_per_group = self.out_channels // self.groups

        signed_axes = tuple(
            torch.cat((torch.arange(mode), torch.arange(-mode + 1, 0)))
            for mode in self.n_modes
        )
        mode_z, mode_x, mode_y = torch.meshgrid(*signed_axes, indexing="ij")
        mode_xyz = torch.stack((mode_x, mode_y, mode_z), dim=-1)
        self.register_buffer("mode_xyz", mode_xyz, persistent=True)

        matrix_size = (
            self.groups
            * self.in_channels_per_group
            * self.out_channels_per_group
        )
        self.radial_network = nn.Sequential(
            nn.Linear(1, self.radial_hidden_channels),
            nn.SiLU(),
            nn.Linear(self.radial_hidden_channels, matrix_size),
        )
        scale = 1.0 / math.sqrt(
            self.in_channels_per_group * self.out_channels_per_group
        )
        final = self.radial_network[-1]
        nn.init.normal_(
            final.weight,
            std=scale / math.sqrt(self.radial_hidden_channels),
        )
        nn.init.normal_(final.bias, std=scale)

    def _physical_squared_wavevectors(self, cells: Tensor) -> Tensor:
        """Return retained ``|k|^2`` values with shape ``(batch, z, x, y)``."""
        modes = self.mode_xyz.to(device=cells.device, dtype=cells.dtype)
        flat_modes = modes.reshape(-1, 3).transpose(0, 1)
        right_hand_side = flat_modes.unsqueeze(0).expand(cells.shape[0], -1, -1)
        wavevectors = (
            2.0
            * math.pi
            * torch.linalg.solve(cells, right_hand_side).transpose(1, 2)
        )
        squared = wavevectors.square().sum(dim=-1)
        return squared.reshape(cells.shape[0], *modes.shape[:-1])

    def forward(self, field: Tensor, cell: Tensor) -> Tensor:
        if field.ndim != 5 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape (batch, {self.in_channels}, nz, nx, ny); "
                f"received {tuple(field.shape)}"
            )
        if cell.ndim == 2 and cell.shape == (3, 3) and field.shape[0] == 1:
            cells = cell.unsqueeze(0)
        elif cell.ndim == 3 and cell.shape == (field.shape[0], 3, 3):
            cells = cell
        else:
            raise ValueError("cell must have shape (3, 3) or (batch, 3, 3)")
        if not torch.is_floating_point(cells):
            raise TypeError("cell must be a floating-point tensor")
        if cells.device != field.device or cells.dtype != field.dtype:
            raise ValueError("field and cell must use the same device and dtype")
        determinants = torch.linalg.det(cells)
        valid = torch.isfinite(cells).all(dim=(-2, -1)) & torch.isfinite(determinants)
        valid = valid & (determinants.abs() > torch.finfo(cells.dtype).eps)
        if not bool(valid.all().detach().cpu()):
            raise ValueError("every metric-aware cell must be finite and nonsingular")

        nz, nx, ny = field.shape[-3:]
        modes_z, modes_x, modes_y = self.n_modes
        if 2 * modes_z > nz or 2 * modes_x > nx or 2 * modes_y > ny:
            raise ValueError(
                f"n_modes={self.n_modes} is incompatible with grid {(nz, nx, ny)}; "
                "require twice every retained mode count to fit its grid axis"
            )
        retained_indices = tuple(
            torch.cat(
                (
                    torch.arange(mode, device=field.device),
                    torch.arange(size - mode + 1, size, device=field.device),
                )
            )
            for mode, size in zip(self.n_modes, (nz, nx, ny), strict=True)
        )
        indices_z, indices_x, indices_y = retained_indices

        field_k = torch.fft.fftn(field, dim=(-3, -2, -1))
        retained = field_k[
            :,
            :,
            indices_z[:, None, None],
            indices_x[None, :, None],
            indices_y[None, None, :],
        ]
        retained = retained.reshape(
            field.shape[0],
            self.groups,
            self.in_channels_per_group,
            *retained.shape[-3:],
        )

        squared_wavevectors = self._physical_squared_wavevectors(cells)
        radial_coordinate = torch.log1p(squared_wavevectors).unsqueeze(-1)
        weights = self.radial_network(radial_coordinate)
        weights = weights.reshape(
            field.shape[0],
            *squared_wavevectors.shape[-3:],
            self.groups,
            self.in_channels_per_group,
            self.out_channels_per_group,
        ).permute(0, 4, 5, 6, 1, 2, 3)
        transformed = torch.complex(
            torch.einsum("bgizxy,bgiozxy->bgozxy", retained.real, weights),
            torch.einsum("bgizxy,bgiozxy->bgozxy", retained.imag, weights),
        ).reshape(field.shape[0], self.out_channels, *retained.shape[-3:])

        output_k = field_k.new_zeros(
            (field.shape[0], self.out_channels, nz, nx, ny)
        )
        output_k[
            :,
            :,
            indices_z[:, None, None],
            indices_x[None, :, None],
            indices_y[None, None, :],
        ] = transformed
        return torch.fft.ifftn(output_k, dim=(-3, -2, -1)).real


def _spectral_conv3d(
    in_channels: int,
    out_channels: int,
    n_modes: tuple[int, int, int],
    *,
    spectral_symmetry: str,
    spectral_groups: int,
    metric_hidden_channels: int,
) -> nn.Module:
    if spectral_symmetry == "none":
        if spectral_groups != 1:
            raise ValueError(
                "spectral_groups applies only to metric-aware EqGINO"
            )
        return SpectralConv3d(in_channels, out_channels, n_modes)
    if spectral_symmetry == "metric_eqgino":
        return MetricEqGINOSpectralConv3d(
            in_channels,
            out_channels,
            n_modes,
            groups=spectral_groups,
            radial_hidden_channels=metric_hidden_channels,
        )
    raise ValueError("spectral_symmetry must be 'none' or 'metric_eqgino'")


class FNOBlock3d(nn.Module):
    """One fully periodic 3D spectral convolution plus a pointwise pathway."""

    def __init__(
        self,
        channels: int,
        n_modes: tuple[int, int, int],
        *,
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        metric_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.spectral = _spectral_conv3d(
            channels,
            channels,
            n_modes,
            spectral_symmetry=spectral_symmetry,
            spectral_groups=spectral_groups,
            metric_hidden_channels=metric_hidden_channels,
        )
        self.spectral_symmetry = spectral_symmetry
        self.local = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(
        self,
        field: Tensor,
        cell: Tensor | None = None,
    ) -> Tensor:
        if self.spectral_symmetry == "metric_eqgino":
            if cell is None:
                raise ValueError("metric-aware EqGINO requires cell")
            spectral = self.spectral(field, cell)
        else:
            spectral = self.spectral(field)
        return F.gelu(spectral + self.local(field))


class LinearFNO3d(nn.Module):
    """A single linear Fourier operator on a fully periodic 3D mesh."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int, int],
        *,
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        metric_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spectral = _spectral_conv3d(
            in_channels,
            out_channels,
            n_modes,
            spectral_symmetry=spectral_symmetry,
            spectral_groups=spectral_groups,
            metric_hidden_channels=metric_hidden_channels,
        )
        self.spectral_symmetry = spectral_symmetry

    def forward(
        self,
        field: Tensor,
        cell: Tensor | None = None,
    ) -> Tensor:
        unbatched = field.ndim == 4
        field_batch = field.unsqueeze(0) if unbatched else field
        if field_batch.ndim != 5 or field_batch.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape ({self.in_channels}, nz, nx, ny) or "
                f"(batch, {self.in_channels}, nz, nx, ny); "
                f"received {tuple(field.shape)}"
            )
        if self.spectral_symmetry == "metric_eqgino":
            if cell is None:
                raise ValueError("metric-aware EqGINO requires cell")
            output = self.spectral(field_batch, cell)
        else:
            output = self.spectral(field_batch)
        return output.squeeze(0) if unbatched else output


class FNO3d(nn.Module):
    """Compact fully periodic FNO for fixed-cell three-dimensional fields."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        metric_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, hidden_channels, n_layers) < 1:
            raise ValueError("channel counts and n_layers must be positive")
        if len(n_modes) != 3 or min(n_modes) < 1:
            raise ValueError("n_modes must contain three positive integers")
        projection_channels = projection_channels or hidden_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = tuple(int(mode) for mode in n_modes)
        self.spectral_symmetry = spectral_symmetry
        self.spectral_groups = int(spectral_groups)

        self.lifting = nn.Conv3d(
            self.in_channels, hidden_channels, kernel_size=1, bias=False
        )
        self.blocks = nn.ModuleList(
            FNOBlock3d(
                hidden_channels,
                self.n_modes,
                spectral_symmetry=self.spectral_symmetry,
                spectral_groups=self.spectral_groups,
                metric_hidden_channels=metric_hidden_channels,
            )
            for _ in range(n_layers)
        )
        self.projection_hidden = nn.Conv3d(
            hidden_channels, projection_channels, kernel_size=1, bias=False
        )
        self.projection_output = nn.Conv3d(
            projection_channels, self.out_channels, kernel_size=1, bias=False
        )

    def forward(
        self,
        field: Tensor,
        cell: Tensor | None = None,
    ) -> Tensor:
        unbatched = field.ndim == 4
        if unbatched:
            field = field.unsqueeze(0)
        if field.ndim != 5 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape ({self.in_channels}, nz, nx, ny) or "
                f"(batch, {self.in_channels}, nz, nx, ny); "
                f"received {tuple(field.shape)}"
            )

        hidden = self.lifting(field)
        for block in self.blocks:
            hidden = block(hidden, cell)
        output = self.projection_output(F.gelu(self.projection_hidden(hidden)))
        return output.squeeze(0) if unbatched else output


class FNOFieldOperator3d(nn.Module):
    """Normalized field operator for periodic 3D fields.

    ``cell_conditioning='isotropic'`` appends the logarithmic cubic cell
    length as a spatially constant input channel.  This lets one nonlinear
    operator distinguish structures represented on the same fractional mesh
    but at different physical volumes, while preserving translation and cubic
    signed-axis symmetries.

    ``cell_conditioning='anisotropic'`` instead appends seven constant
    channels: the logarithmic volume length and the six independent entries
    of the volume-normalized lattice metric.  The metric is unchanged by a
    rigid Cartesian rotation of the cell and supplies the operator with both
    cell size and shape without tying it to one laboratory orientation.
    """

    def __init__(
        self,
        channels: int,
        n_modes: tuple[int, int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        architecture: str = "nonlinear",
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        metric_hidden_channels: int = 16,
        cell_conditioning: str = "none",
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "nonlinear"}:
            raise ValueError("architecture must be 'linear' or 'nonlinear'")
        if cell_conditioning not in {"none", "isotropic", "anisotropic"}:
            raise ValueError(
                "cell_conditioning must be 'none', 'isotropic', or 'anisotropic'"
            )
        if cell_conditioning != "none" and architecture != "nonlinear":
            raise ValueError("cell conditioning requires architecture='nonlinear'")
        self.architecture = architecture
        self.channels = int(channels)
        self.spectral_symmetry = spectral_symmetry
        self.spectral_groups = int(spectral_groups)
        self.metric_hidden_channels = int(metric_hidden_channels)
        self.cell_conditioning = cell_conditioning
        conditioning_channels = {
            "none": 0,
            "isotropic": 1,
            "anisotropic": 7,
        }[cell_conditioning]
        input_channels = channels + conditioning_channels
        if architecture == "linear":
            self.fno = LinearFNO3d(
                input_channels,
                channels,
                n_modes,
                spectral_symmetry=spectral_symmetry,
                spectral_groups=spectral_groups,
                metric_hidden_channels=metric_hidden_channels,
            )
        else:
            self.fno = FNO3d(
                input_channels,
                channels,
                n_modes,
                hidden_channels=hidden_channels,
                n_layers=n_layers,
                projection_channels=projection_channels,
                spectral_symmetry=spectral_symmetry,
                spectral_groups=spectral_groups,
                metric_hidden_channels=metric_hidden_channels,
            )
        self.register_buffer("input_scale", torch.ones(1, channels, 1, 1, 1))
        self.register_buffer("output_scale", torch.ones(1, channels, 1, 1, 1))

    @torch.no_grad()
    def fit_normalization(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        minimum_scale: float = 1.0e-12,
    ) -> None:
        if inputs.ndim != 5 or targets.shape != inputs.shape:
            raise ValueError(
                "inputs and targets must have matching "
                "(batch, channels, nz, nx, ny) shapes"
            )
        input_rms = inputs.square().mean(dim=(0, 2, 3, 4), keepdim=True).sqrt()
        output_rms = targets.square().mean(dim=(0, 2, 3, 4), keepdim=True).sqrt()
        self.input_scale.copy_(input_rms.clamp_min(minimum_scale))
        self.output_scale.copy_(output_rms.clamp_min(minimum_scale))

    def forward(self, density: Tensor, cell: Tensor | None = None) -> Tensor:
        cells = None
        if cell is not None:
            if cell.ndim == 2 and cell.shape == (3, 3):
                cells = cell.unsqueeze(0)
            elif cell.ndim == 3 and cell.shape[-2:] == (3, 3):
                cells = cell
            else:
                raise ValueError("cell must have shape (3, 3) or (batch, 3, 3)")
        unbatched = density.ndim == 4
        density_batch = density.unsqueeze(0) if unbatched else density
        if density_batch.ndim != 5 or density_batch.shape[1] != self.channels:
            raise ValueError(
                f"density must have shape ({self.channels}, nz, nx, ny) or "
                f"(batch, {self.channels}, nz, nx, ny)"
            )
        normalized = density_batch / self.input_scale
        if self.cell_conditioning == "isotropic":
            if cells is None:
                raise ValueError("isotropic cell conditioning requires cell")
            if cells.shape[0] != density_batch.shape[0]:
                raise ValueError("cell batch size must match density batch size")
            if any(not is_cubic_cell(value) for value in cells):
                raise ValueError("isotropic cell conditioning requires cubic cells")
            lengths = cells.det().abs().pow(1.0 / 3.0)
            if bool((lengths <= 0).any().detach().cpu()):
                raise ValueError("cell volumes must be positive")
            condition = lengths.log().reshape(-1, 1, 1, 1, 1).expand(
                -1, 1, *density_batch.shape[-3:]
            )
            normalized = torch.cat((normalized, condition), dim=1)
        elif self.cell_conditioning == "anisotropic":
            if cells is None:
                raise ValueError("anisotropic cell conditioning requires cell")
            if cells.shape[0] != density_batch.shape[0]:
                raise ValueError("cell batch size must match density batch size")
            determinants = torch.linalg.det(cells)
            volume_lengths = determinants.abs().pow(1.0 / 3.0)
            valid = torch.isfinite(cells).all(dim=(-2, -1)) & torch.isfinite(
                volume_lengths
            )
            valid = valid & (volume_lengths > 0)
            if not bool(valid.all().detach().cpu()):
                raise ValueError("anisotropic cell conditioning requires finite cells")
            metric = cells @ cells.transpose(-2, -1)
            metric = metric / volume_lengths.square().reshape(-1, 1, 1)
            condition_values = torch.stack(
                (
                    volume_lengths.log(),
                    metric[:, 0, 0],
                    metric[:, 1, 1],
                    metric[:, 2, 2],
                    metric[:, 0, 1],
                    metric[:, 0, 2],
                    metric[:, 1, 2],
                ),
                dim=1,
            )
            condition = condition_values.reshape(-1, 7, 1, 1, 1).expand(
                -1, -1, *density_batch.shape[-3:]
            )
            normalized = torch.cat((normalized, condition), dim=1)
        potential = self.fno(normalized, cell=cells) * self.output_scale
        return potential.squeeze(0) if unbatched else potential
