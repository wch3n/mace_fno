"""Hybrid 2.5D Fourier neural operators for finite slabs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ._normalization import _fit_channel_normalization
from .fno_2d import SpectralConv2D
from .fno_slab_eqgino import MetricEqGINOSpectralConv2D


def _field_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "identity":
        return nn.Identity()
    raise ValueError("field activation must be 'gelu' or 'identity'")


class SlabSpectralConv2D(nn.Module):
    """Dense ``R(k_parallel, z, z')`` with Fourier transforms only in x/y.

    This layer is intended for controlled linear models with modest channel
    and z dimensions. It learns a separate complex response between every
    input and output z layer at each retained in-plane Fourier mode. There is
    no transform and no periodic wrapping along z.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_z: int,
        n_modes: tuple[int, int],
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, n_z) < 1:
            raise ValueError("channel counts and n_z must be positive")
        if len(n_modes) != 2 or min(n_modes) < 1:
            raise ValueError("n_modes must contain two positive integers")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_z = int(n_z)
        self.n_modes = (int(n_modes[0]), int(n_modes[1]))

        shape = (
            self.in_channels,
            self.out_channels,
            self.n_z,
            self.n_z,
            *self.n_modes,
        )
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels * self.n_z)
        self.weight_positive_real = nn.Parameter(scale * torch.randn(shape))
        self.weight_positive_imag = nn.Parameter(scale * torch.randn(shape))
        self.weight_negative_real = nn.Parameter(scale * torch.randn(shape))
        self.weight_negative_imag = nn.Parameter(scale * torch.randn(shape))

    @staticmethod
    def _contract(field: Tensor, weight: Tensor) -> Tensor:
        # i/o: input/output channels; z/q: input/output finite z layers.
        return torch.einsum("bizxy,iozqxy->boqxy", field, weight)

    def forward(self, field: Tensor) -> Tensor:
        if (
            field.ndim != 5
            or field.shape[1] != self.in_channels
            or field.shape[2] != self.n_z
        ):
            raise ValueError(
                f"field must have shape (batch, {self.in_channels}, {self.n_z}, "
                f"nx, ny); received {tuple(field.shape)}"
            )
        nx, ny = field.shape[-2:]
        modes_x, modes_y = self.n_modes
        if 2 * modes_x > nx or modes_y > ny // 2 + 1:
            raise ValueError(
                f"n_modes={self.n_modes} is incompatible with grid {(nx, ny)}; "
                "require 2*modes_x <= nx and modes_y <= ny//2 + 1"
            )

        field_k = torch.fft.rfft2(field, dim=(-2, -1))
        output_k = field_k.new_zeros(
            (field.shape[0], self.out_channels, self.n_z, nx, ny // 2 + 1)
        )
        positive = torch.complex(self.weight_positive_real, self.weight_positive_imag)
        negative = torch.complex(self.weight_negative_real, self.weight_negative_imag)
        output_k[:, :, :, :modes_x, :modes_y] = self._contract(
            field_k[:, :, :, :modes_x, :modes_y], positive
        )
        output_k[:, :, :, -modes_x:, :modes_y] = self._contract(
            field_k[:, :, :, -modes_x:, :modes_y], negative
        )
        return torch.fft.irfft2(output_k, s=(nx, ny), dim=(-2, -1))


class SlabPlanarSpectralConv2D(nn.Module):
    """Apply one shared 2D spectral convolution independently to each z layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spectral = SpectralConv2D(in_channels, out_channels, n_modes)

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim != 5 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape (batch, {self.in_channels}, nz, nx, ny); "
                f"received {tuple(field.shape)}"
            )
        batch, _, nz, nx, ny = field.shape
        slices = field.permute(0, 2, 1, 3, 4).reshape(
            batch * nz, self.in_channels, nx, ny
        )
        transformed = self.spectral(slices)
        return transformed.reshape(batch, nz, self.out_channels, nx, ny).permute(
            0, 2, 1, 3, 4
        )


class GlobalZMixing(nn.Module):
    """Channel-wise dense mixing between all finite, nonperiodic z layers.

    Each hidden channel learns an independent ``(z_out, z_in)`` response that
    is shared over the lateral grid. Unlike a convolution or a z FFT, the
    matrix does not identify, wrap, or otherwise tie the two slab boundaries.
    """

    def __init__(self, channels: int, n_z: int) -> None:
        super().__init__()
        if min(channels, n_z) < 1:
            raise ValueError("channels and n_z must be positive")
        self.channels = int(channels)
        self.n_z = int(n_z)
        scale = 1.0 / math.sqrt(self.n_z)
        self.weight = nn.Parameter(
            scale * torch.randn(self.channels, self.n_z, self.n_z)
        )

    def forward(self, field: Tensor) -> Tensor:
        if (
            field.ndim != 5
            or field.shape[1] != self.channels
            or field.shape[2] != self.n_z
        ):
            raise ValueError(
                f"field must have shape (batch, {self.channels}, {self.n_z}, "
                f"nx, ny); received {tuple(field.shape)}"
            )
        return torch.einsum("bcixy,coi->bcoxy", field, self.weight)


class SlabFNOBlock2D(nn.Module):
    """Spectral and pointwise mixing with explicit, nonperiodic heights.

    With ``z_mixing='full'``, the spectral matrix couples (channel, height)
    indices jointly at every in-plane wavevector. Otherwise height mixing
    is a separate, wavevector-independent branch, as in earlier checkpoints.
    """

    def __init__(
        self,
        channels: int,
        n_z: int,
        n_modes: tuple[int, int],
        *,
        z_kernel_size: int = 3,
        z_mixing: str = "local",
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        spectral_rank: int = 0,
        metric_hidden_channels: int = 16,
        metric_parameterization: str = "shell_spline",
        metric_reference_length: float = 1.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if spectral_rank and not (
            z_mixing == "full"
            and spectral_symmetry == "metric_eqgino"
            and metric_parameterization == "shell_spline"
        ):
            raise ValueError(
                "spectral_rank requires full z-mixing, metric_eqgino, and shell_spline"
            )
        if z_mixing not in {"local", "global", "full"}:
            raise ValueError("z_mixing must be 'local', 'global', or 'full'")
        self.spectral_symmetry = spectral_symmetry
        if spectral_symmetry == "metric_eqgino":
            if z_mixing == "full":
                self.spectral = LinearMetricSlabFNO2D(
                    channels,
                    n_z,
                    n_modes,
                    spectral_groups=spectral_groups,
                    spectral_rank=spectral_rank,
                    metric_hidden_channels=metric_hidden_channels,
                    metric_parameterization=metric_parameterization,
                    metric_reference_length=metric_reference_length,
                )
            else:
                self.spectral = MetricEqGINOSpectralConv2D(
                    channels,
                    channels,
                    n_modes,
                    groups=spectral_groups,
                    rank=spectral_rank,
                    radial_hidden_channels=metric_hidden_channels,
                    parameterization=metric_parameterization,
                    reference_length=metric_reference_length,
                )
        elif spectral_symmetry == "none":
            if spectral_groups != 1:
                raise ValueError("spectral_groups applies only to metric-aware EqGINO")
            self.spectral = (
                SlabSpectralConv2D(channels, channels, n_z, n_modes)
                if z_mixing == "full"
                else SlabPlanarSpectralConv2D(channels, channels, n_modes)
            )
        else:
            raise ValueError("spectral_symmetry must be 'none' or 'metric_eqgino'")
        if z_mixing == "local":
            if z_kernel_size < 1 or z_kernel_size % 2 == 0:
                raise ValueError("z_kernel_size must be a positive odd integer")
            self.z_mixing = nn.Conv3d(
                channels,
                channels,
                kernel_size=(z_kernel_size, 1, 1),
                padding=(z_kernel_size // 2, 0, 0),
                bias=False,
            )
        elif z_mixing == "global":
            self.z_mixing = GlobalZMixing(channels, n_z)
        else:
            self.z_mixing = None
        self.local = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.activation = _field_activation(activation)

    def forward(self, field: Tensor, cell: Tensor | None = None) -> Tensor:
        if self.spectral_symmetry == "metric_eqgino":
            if cell is None:
                raise ValueError("metric-aware slab EqGINO requires cell")
            spectral = self.spectral(field, cell)
        else:
            spectral = self.spectral(field)
        if self.z_mixing is not None:
            spectral = spectral + self.z_mixing(field)
        return self.activation(spectral + self.local(field))


class LinearSlabFNO2D(nn.Module):
    """A linear, explicit learned ``R(k_parallel, z, z')`` operator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_z: int,
        n_modes: tuple[int, int],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_z = int(n_z)
        self.spectral = SlabSpectralConv2D(in_channels, out_channels, n_z, n_modes)

    def forward(self, field: Tensor) -> Tensor:
        unbatched = field.ndim == 4
        field_batch = field.unsqueeze(0) if unbatched else field
        if (
            field_batch.ndim != 5
            or field_batch.shape[1] != self.in_channels
            or field_batch.shape[2] != self.n_z
        ):
            raise ValueError(
                f"field must have shape ({self.in_channels}, {self.n_z}, nx, ny) "
                f"or (batch, {self.in_channels}, {self.n_z}, nx, ny); "
                f"received {tuple(field.shape)}"
            )
        output = self.spectral(field_batch)
        return output.squeeze(0) if unbatched else output


class LinearMetricSlabFNO2D(nn.Module):
    """Radial ``R(|k_parallel|, z, z')`` with unrestricted finite-height mixing.

    Treat (channel, height) as a compound index for the radial matrix. Only
    x/y are Fourier transformed. The height endpoints remain independent,
    with no wrapping or z translation symmetry imposed. Groups partition
    feature channels, not heights (source channels for the direct map, hidden
    channels when used inside a block). This is linear in the field at fixed cell,
    even when the radial weights depend on the physical cell metric.
    """

    def __init__(
        self,
        channels: int,
        n_z: int,
        n_modes: tuple[int, int],
        *,
        spectral_groups: int = 1,
        spectral_rank: int = 0,
        metric_hidden_channels: int = 16,
        metric_parameterization: str = "shell_spline",
        metric_reference_length: float = 1.0,
    ) -> None:
        super().__init__()
        if min(channels, n_z, spectral_groups) < 1 or channels % spectral_groups:
            raise ValueError("positive spectral_groups must divide feature channels")
        self.channels = int(channels)
        self.n_z = int(n_z)
        self.spectral = MetricEqGINOSpectralConv2D(
            channels * n_z,
            channels * n_z,
            n_modes,
            groups=spectral_groups,
            rank=spectral_rank,
            radial_hidden_channels=metric_hidden_channels,
            parameterization=metric_parameterization,
            reference_length=metric_reference_length,
        )

    def forward(self, field: Tensor, cell: Tensor | None = None) -> Tensor:
        if cell is None:
            raise ValueError("metric-aware linear slab FNO requires cell")
        unbatched = field.ndim == 4
        values = field.unsqueeze(0) if unbatched else field
        if values.ndim != 5 or values.shape[1:3] != (self.channels, self.n_z):
            raise ValueError("field must have matching (channels, nz, nx, ny) axes")
        batch, _, _, nx, ny = values.shape
        flattened = values.reshape(batch, self.channels * self.n_z, 1, nx, ny)
        output = self.spectral(flattened, cell).reshape(values.shape)
        return output.squeeze(0) if unbatched else output


class SlabFNO2D(nn.Module):
    """Hybrid slab FNO with 2D FFTs and finite, nonperiodic z mixing.

    z remains an explicit axis throughout. Consequently the model is periodic
    and discretely translation equivariant only in x/y; the top and bottom z
    boundaries are distinct. Height mixing is local, global, or embedded in
    the full spectral matrix. None uses an FFT or circular padding along z.
    Identity activations give a stacked linear field map with the same
    parameters as the GELU model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_z: int,
        n_modes: tuple[int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        z_kernel_size: int = 3,
        z_mixing: str = "local",
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        spectral_rank: int = 0,
        metric_hidden_channels: int = 16,
        metric_parameterization: str = "shell_spline",
        metric_reference_length: float = 1.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, n_z, hidden_channels, n_layers) < 1:
            raise ValueError("channel counts, n_z, and n_layers must be positive")
        if spectral_rank and not (
            z_mixing == "full"
            and spectral_symmetry == "metric_eqgino"
            and metric_parameterization == "shell_spline"
        ):
            raise ValueError(
                "spectral_rank requires full z-mixing, metric_eqgino, and shell_spline"
            )
        if z_mixing not in {"local", "global", "full"}:
            raise ValueError("z_mixing must be 'local', 'global', or 'full'")
        projection_channels = projection_channels or hidden_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_z = int(n_z)
        self.n_modes = (int(n_modes[0]), int(n_modes[1]))
        self.z_mixing = z_mixing

        self.lifting = nn.Conv3d(
            self.in_channels, hidden_channels, kernel_size=1, bias=False
        )
        self.blocks = nn.ModuleList(
            SlabFNOBlock2D(
                hidden_channels,
                self.n_z,
                self.n_modes,
                z_kernel_size=z_kernel_size,
                z_mixing=z_mixing,
                spectral_symmetry=spectral_symmetry,
                spectral_groups=spectral_groups,
                spectral_rank=spectral_rank,
                metric_hidden_channels=metric_hidden_channels,
                metric_parameterization=metric_parameterization,
                metric_reference_length=metric_reference_length,
                activation=activation,
            )
            for _ in range(n_layers)
        )
        self.projection_hidden = nn.Conv3d(
            hidden_channels, projection_channels, kernel_size=1, bias=False
        )
        self.projection_output = nn.Conv3d(
            projection_channels, self.out_channels, kernel_size=1, bias=False
        )
        self.projection_activation = _field_activation(activation)

    def forward(self, field: Tensor, cell: Tensor | None = None) -> Tensor:
        unbatched = field.ndim == 4
        if unbatched:
            field = field.unsqueeze(0)
        if (
            field.ndim != 5
            or field.shape[1] != self.in_channels
            or field.shape[2] != self.n_z
        ):
            raise ValueError(
                f"field must have shape ({self.in_channels}, {self.n_z}, nx, ny) "
                f"or (batch, {self.in_channels}, {self.n_z}, nx, ny); "
                f"received {tuple(field.shape)}"
            )

        hidden = self.lifting(field)
        for block in self.blocks:
            hidden = block(hidden, cell)
        output = self.projection_output(
            self.projection_activation(self.projection_hidden(hidden))
        )
        return output.squeeze(0) if unbatched else output


class SlabFNOFieldOperator2D(nn.Module):
    """Normalized field-operator adapter for explicit finite z layers.

    C4/D4 ensemble averaging requires a square physical in-plane cell and a
    square lateral mesh. Alternatively, ``spectral_symmetry='metric_eqgino'``
    uses intrinsic radial weights with no ensemble. Pass ``cell`` for either
    path; one cell may be shared by all fields
    or supplied per field in the batch. The metric path accepts nonsquare cells.
    """

    def __init__(
        self,
        channels: int,
        n_z: int,
        n_modes: tuple[int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        z_kernel_size: int = 3,
        z_mixing: str = "local",
        planar_symmetry: str = "none",
        architecture: str = "nonlinear",
        spectral_symmetry: str = "none",
        spectral_groups: int = 1,
        spectral_rank: int = 0,
        metric_hidden_channels: int = 16,
        metric_parameterization: str = "shell_spline",
        metric_reference_length: float = 1.0,
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "nonlinear"}:
            raise ValueError("architecture must be 'linear' or 'nonlinear'")
        if spectral_rank and not (
            z_mixing == "full"
            and spectral_symmetry == "metric_eqgino"
            and metric_parameterization == "shell_spline"
        ):
            raise ValueError(
                "spectral_rank requires full z-mixing, metric_eqgino, and shell_spline"
            )
        if z_mixing not in {"local", "global", "full"}:
            raise ValueError("z_mixing must be 'local', 'global', or 'full'")
        if planar_symmetry not in {"none", "c4", "d4"}:
            raise ValueError("planar_symmetry must be 'none', 'c4', or 'd4'")
        if spectral_symmetry not in {"none", "metric_eqgino"}:
            raise ValueError("spectral_symmetry must be 'none' or 'metric_eqgino'")
        if spectral_symmetry == "metric_eqgino":
            if planar_symmetry != "none":
                raise ValueError(
                    "slab metric_eqgino requires planar_symmetry='none'; no ensemble is needed"
                )
        elif spectral_groups != 1:
            raise ValueError("spectral_groups applies only to metric-aware EqGINO")
        self.spectral_symmetry = spectral_symmetry
        self.spectral_groups = int(spectral_groups)
        self.metric_parameterization = metric_parameterization
        self.architecture = architecture
        self.channels = int(channels)
        self.n_z = int(n_z)
        direct_linear = architecture == "linear" and z_mixing != "full"
        self.z_mixing = "spectral" if direct_linear else z_mixing
        self.planar_symmetry = planar_symmetry
        self._training_symmetry_index = 0
        if direct_linear:
            if spectral_symmetry == "metric_eqgino":
                self.fno = LinearMetricSlabFNO2D(
                    channels,
                    n_z,
                    n_modes,
                    spectral_groups=spectral_groups,
                    spectral_rank=spectral_rank,
                    metric_hidden_channels=metric_hidden_channels,
                    metric_parameterization=metric_parameterization,
                    metric_reference_length=metric_reference_length,
                )
            else:
                self.fno = LinearSlabFNO2D(channels, channels, n_z, n_modes)
        else:
            self.fno = SlabFNO2D(
                channels,
                channels,
                n_z,
                n_modes,
                hidden_channels=hidden_channels,
                n_layers=n_layers,
                projection_channels=projection_channels,
                z_kernel_size=z_kernel_size,
                z_mixing=z_mixing,
                spectral_symmetry=spectral_symmetry,
                spectral_groups=spectral_groups,
                spectral_rank=spectral_rank,
                metric_hidden_channels=metric_hidden_channels,
                metric_parameterization=metric_parameterization,
                metric_reference_length=metric_reference_length,
                activation="identity" if architecture == "linear" else "gelu",
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
        _fit_channel_normalization(
            inputs,
            targets,
            self.input_scale,
            self.output_scale,
            minimum_scale=minimum_scale,
        )

    @staticmethod
    def _validate_symmetry_cell(cell: Tensor | None, batch_size: int) -> None:
        if cell is None:
            raise ValueError("C4/D4 planar symmetry requires a physical cell")
        if cell.shape == (3, 3):
            cells = cell.unsqueeze(0)
        elif cell.shape == (batch_size, 3, 3):
            cells = cell
        else:
            raise ValueError(
                "C4/D4 cell must have shape (3, 3) or (batch, 3, 3), "
                "matching the field batch"
            )
        if not torch.is_floating_point(cells):
            raise TypeError("C4/D4 cell must be a floating-point tensor")
        plane = cells[:, :2]
        lengths = torch.linalg.vector_norm(plane, dim=-1)
        product = lengths[:, 0] * lengths[:, 1]
        dot = (plane[:, 0] * plane[:, 1]).sum(dim=-1)
        tolerance = 1.0e-6
        valid = torch.isfinite(cells).all(dim=(-2, -1))
        valid = valid & torch.isfinite(product)
        valid = valid & (lengths > torch.finfo(cells.dtype).eps).all(dim=-1)
        valid = valid & (
            (lengths[:, 0] - lengths[:, 1]).abs() <= tolerance * lengths.amax(dim=-1)
        )
        valid = valid & (dot.abs() <= tolerance * product)
        if not bool(valid.all().detach().cpu()):
            raise ValueError(
                "C4/D4 planar symmetry requires a finite square in-plane cell "
                "(orthogonal first two vectors of equal length) for every field"
            )

    def forward(self, density: Tensor, cell: Tensor | None = None) -> Tensor:
        unbatched = density.ndim == 4
        density_batch = density.unsqueeze(0) if unbatched else density
        if (
            density_batch.ndim != 5
            or density_batch.shape[1] != self.channels
            or density_batch.shape[2] != self.n_z
        ):
            raise ValueError(
                f"density must have shape ({self.channels}, {self.n_z}, nx, ny) "
                f"or (batch, {self.channels}, {self.n_z}, nx, ny)"
            )
        normalized = density_batch / self.input_scale
        if self.planar_symmetry in {"c4", "d4"}:
            if normalized.shape[-2] != normalized.shape[-1]:
                raise ValueError("C4/D4 planar symmetry requires a square lateral grid")
            self._validate_symmetry_cell(cell, density_batch.shape[0])
            group_size = 4 if self.planar_symmetry == "c4" else 8

            def transform(field: Tensor, index: int) -> Tensor:
                if index >= 4:
                    field = torch.flip(field, dims=(-1,))
                return torch.rot90(field, index % 4, dims=(-2, -1))

            def inverse_transform(field: Tensor, index: int) -> Tensor:
                field = torch.rot90(field, -(index % 4), dims=(-2, -1))
                return torch.flip(field, dims=(-1,)) if index >= 4 else field

            if self.training:
                # Cycle through one group image per optimizer forward. This is
                # balanced symmetry augmentation without an 8x training cost.
                index = self._training_symmetry_index % group_size
                self._training_symmetry_index += 1
                potential = inverse_transform(
                    self.fno(transform(normalized, index)), index
                )
            else:
                transformed_inputs = [
                    transform(normalized, index) for index in range(group_size)
                ]
                transformed = self.fno(torch.cat(transformed_inputs, dim=0))
                chunks = transformed.chunk(group_size, dim=0)
                potential = torch.stack(
                    [
                        inverse_transform(chunk, index)
                        for index, chunk in enumerate(chunks)
                    ],
                    dim=0,
                ).mean(dim=0)
        elif self.spectral_symmetry == "metric_eqgino":
            potential = self.fno(normalized, cell)
        else:
            potential = self.fno(normalized)
        potential = potential * self.output_scale
        return potential.squeeze(0) if unbatched else potential
