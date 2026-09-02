"""Fully periodic 3D and EqGINO-style Fourier neural operators."""

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


class EqGINOSpectralConv3d(nn.Module):
    """Isotropic full-FFT convolution for real scalar 3D fields.

    This is an atomistic, real-to-real adaptation of EqGINO's EqFNO layer.  A
    single channel-mixing matrix is shared by all reciprocal-grid modes with
    the same squared radius ``kz**2 + kx**2 + ky**2``.  The resulting operator
    is exactly equivariant to the signed axis permutations of a cubic grid.

    EqGINO carries complex features between spectral blocks.  This project
    instead uses real scalar density and potential fields with real pointwise
    nonlinearities.  The radial weights are consequently real: isotropy gives
    ``W(-k) = W(k)``, while a real output requires
    ``W(-k) = conj(W(k))``.  Together these conditions require real weights and
    preserve Hermitian symmetry without discarding learned parameters.

    ``groups`` implements EqGINO's block-diagonal channel mixing.  A value of
    one is dense; larger divisors reduce the spectral contraction cost while
    limiting cross-group channel communication to the pointwise pathways.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int, int],
        *,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if len(n_modes) != 3 or min(n_modes) < 1:
            raise ValueError("n_modes must contain three positive integers")
        if len(set(int(mode) for mode in n_modes)) != 1:
            raise ValueError("EqGINO isotropy requires equal modes on all axes")
        if groups < 1:
            raise ValueError("groups must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = tuple(int(mode) for mode in n_modes)
        self.groups = int(groups)
        self.in_channels_per_group = self.in_channels // self.groups
        self.out_channels_per_group = self.out_channels // self.groups

        mode = self.n_modes[0]
        signed_modes = torch.cat(
            (torch.arange(mode), torch.arange(-mode + 1, 0))
        )
        kz, kx, ky = torch.meshgrid(
            signed_modes, signed_modes, signed_modes, indexing="ij"
        )
        squared_radius = kz.square() + kx.square() + ky.square()
        radii, radius_indices = torch.unique(
            squared_radius, sorted=True, return_inverse=True
        )
        self.register_buffer(
            "squared_radii", radii, persistent=True
        )
        self.register_buffer(
            "radius_indices",
            radius_indices.reshape(squared_radius.shape),
            persistent=True,
        )

        shape = (
            self.groups,
            self.in_channels_per_group,
            self.out_channels_per_group,
            int(radii.numel()),
        )
        scale = 1.0 / math.sqrt(
            self.in_channels_per_group * self.out_channels_per_group
        )
        self.radial_weight = nn.Parameter(scale * torch.randn(shape))

    @property
    def n_radial_shells(self) -> int:
        """Number of independently learned reciprocal-radius shells."""
        return int(self.squared_radii.numel())

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim != 5 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape (batch, {self.in_channels}, n, n, n); "
                f"received {tuple(field.shape)}"
            )
        nz, nx, ny = field.shape[-3:]
        if not (nz == nx == ny):
            raise ValueError("EqGINO isotropy requires a cubic 3D grid")
        mode = self.n_modes[0]
        if 2 * mode > nz:
            raise ValueError(
                f"n_modes={self.n_modes} is incompatible with grid {(nz, nx, ny)}; "
                "require 2*mode <= grid size"
            )

        positive = torch.arange(mode, device=field.device)
        negative = torch.arange(nz - mode + 1, nz, device=field.device)
        indices = torch.cat((positive, negative))

        field_k = torch.fft.fftn(field, dim=(-3, -2, -1))
        retained = field_k[
            :,
            :,
            indices[:, None, None],
            indices[None, :, None],
            indices[None, None, :],
        ]
        retained = retained.reshape(
            field.shape[0],
            self.groups,
            self.in_channels_per_group,
            *retained.shape[-3:],
        )
        weight = self.radial_weight[..., self.radius_indices]
        transformed = torch.complex(
            torch.einsum("bgizxy,giozxy->bgozxy", retained.real, weight),
            torch.einsum("bgizxy,giozxy->bgozxy", retained.imag, weight),
        ).reshape(
            field.shape[0], self.out_channels, *retained.shape[-3:]
        )

        output_k = field_k.new_zeros(
            (field.shape[0], self.out_channels, nz, nx, ny)
        )
        output_k[
            :,
            :,
            indices[:, None, None],
            indices[None, :, None],
            indices[None, None, :],
        ] = transformed
        return torch.fft.ifftn(output_k, dim=(-3, -2, -1)).real


def _spectral_conv3d(
    in_channels: int,
    out_channels: int,
    n_modes: tuple[int, int, int],
    *,
    spectral_symmetry: str,
    spectral_groups: int,
) -> nn.Module:
    if spectral_symmetry == "none":
        if spectral_groups != 1:
            raise ValueError("spectral_groups applies only to EqGINO symmetry")
        return SpectralConv3d(in_channels, out_channels, n_modes)
    if spectral_symmetry == "eqgino":
        return EqGINOSpectralConv3d(
            in_channels, out_channels, n_modes, groups=spectral_groups
        )
    raise ValueError("spectral_symmetry must be 'none' or 'eqgino'")


class FNOBlock3d(nn.Module):
    """One fully periodic 3D spectral convolution plus a pointwise pathway."""

    def __init__(
        self,
        channels: int,
        n_modes: tuple[int, int, int],
        *,
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
    ) -> None:
        super().__init__()
        self.spectral = _spectral_conv3d(
            channels,
            channels,
            n_modes,
            spectral_symmetry=spectral_symmetry,
            spectral_groups=spectral_groups,
        )
        self.local = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, field: Tensor) -> Tensor:
        return F.gelu(self.spectral(field) + self.local(field))


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
        )

    def forward(self, field: Tensor) -> Tensor:
        unbatched = field.ndim == 4
        field_batch = field.unsqueeze(0) if unbatched else field
        if field_batch.ndim != 5 or field_batch.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape ({self.in_channels}, nz, nx, ny) or "
                f"(batch, {self.in_channels}, nz, nx, ny); "
                f"received {tuple(field.shape)}"
            )
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
            )
            for _ in range(n_layers)
        )
        self.projection_hidden = nn.Conv3d(
            hidden_channels, projection_channels, kernel_size=1, bias=False
        )
        self.projection_output = nn.Conv3d(
            projection_channels, self.out_channels, kernel_size=1, bias=False
        )

    def forward(self, field: Tensor) -> Tensor:
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
            hidden = block(hidden)
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
        if self.spectral_symmetry == "eqgino" and cells is not None:
            if any(not is_cubic_cell(value) for value in cells):
                raise ValueError("EqGINO symmetry requires cubic cells")
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
        potential = self.fno(normalized) * self.output_scale
        return potential.squeeze(0) if unbatched else potential
