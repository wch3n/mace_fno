"""Hybrid 2.5D Fourier neural operators for finite slabs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fno_2d import SpectralConv2D


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
    """In-plane spectral mixing plus local or global nonperiodic z mixing."""

    def __init__(
        self,
        channels: int,
        n_z: int,
        n_modes: tuple[int, int],
        *,
        z_kernel_size: int = 3,
        z_mixing: str = "local",
    ) -> None:
        super().__init__()
        if z_mixing not in {"local", "global"}:
            raise ValueError("z_mixing must be 'local' or 'global'")
        self.spectral = SlabPlanarSpectralConv2D(channels, channels, n_modes)
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
        else:
            self.z_mixing = GlobalZMixing(channels, n_z)
        self.local = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, field: Tensor) -> Tensor:
        return F.gelu(self.spectral(field) + self.z_mixing(field) + self.local(field))


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


class SlabFNO2D(nn.Module):
    """Hybrid slab FNO with 2D FFTs and finite, nonperiodic z mixing.

    z remains an explicit axis throughout. Consequently the model is periodic
    and discretely translation equivariant only in x/y; the top and bottom z
    boundaries are distinct. z mixing can be either a zero-padded local CNN or
    a dense global response; neither uses an FFT or circular padding.
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
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, n_z, hidden_channels, n_layers) < 1:
            raise ValueError("channel counts, n_z, and n_layers must be positive")
        if z_mixing not in {"local", "global"}:
            raise ValueError("z_mixing must be 'local' or 'global'")
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
            hidden = block(hidden)
        output = self.projection_output(F.gelu(self.projection_hidden(hidden)))
        return output.squeeze(0) if unbatched else output


class SlabFNOFieldOperator2D(nn.Module):
    """Normalized field-operator adapter for explicit finite z layers."""

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
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "nonlinear"}:
            raise ValueError("architecture must be 'linear' or 'nonlinear'")
        if z_mixing not in {"local", "global"}:
            raise ValueError("z_mixing must be 'local' or 'global'")
        if planar_symmetry not in {"none", "c4", "d4"}:
            raise ValueError("planar_symmetry must be 'none', 'c4', or 'd4'")
        self.architecture = architecture
        self.channels = int(channels)
        self.n_z = int(n_z)
        self.z_mixing = "spectral" if architecture == "linear" else z_mixing
        self.planar_symmetry = planar_symmetry
        self._training_symmetry_index = 0
        if architecture == "linear":
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
        del cell
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
        else:
            potential = self.fno(normalized)
        potential = potential * self.output_scale
        return potential.squeeze(0) if unbatched else potential
