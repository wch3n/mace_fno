"""Fourier neural operators for planar, finite-slab, and periodic bulk fields."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .symmetry import is_cubic_cell


class SpectralConv2d(nn.Module):
    """Learned global convolution on a truncated set of 2D Fourier modes.

    Real and imaginary weights are stored separately. This makes ``module.double()``
    promote the complete complex-valued kernel safely from complex64 to
    complex128 instead of risking loss of its imaginary component.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int],
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if len(n_modes) != 2 or min(n_modes) < 1:
            raise ValueError("n_modes must contain two positive integers")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = (int(n_modes[0]), int(n_modes[1]))

        shape = (self.in_channels, self.out_channels, *self.n_modes)
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        self.weight_positive_real = nn.Parameter(scale * torch.randn(shape))
        self.weight_positive_imag = nn.Parameter(scale * torch.randn(shape))
        self.weight_negative_real = nn.Parameter(scale * torch.randn(shape))
        self.weight_negative_imag = nn.Parameter(scale * torch.randn(shape))

    @staticmethod
    def _contract(field: Tensor, weight: Tensor) -> Tensor:
        return torch.einsum("bixy,ioxy->boxy", field, weight)

    def _complex_weights(self) -> tuple[Tensor, Tensor]:
        positive = torch.complex(self.weight_positive_real, self.weight_positive_imag)
        negative = torch.complex(self.weight_negative_real, self.weight_negative_imag)
        return positive, negative

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim != 4 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape (batch, {self.in_channels}, nx, ny); "
                f"received {tuple(field.shape)}"
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
            (field.shape[0], self.out_channels, nx, ny // 2 + 1)
        )
        positive, negative = self._complex_weights()
        output_k[:, :, :modes_x, :modes_y] = self._contract(
            field_k[:, :, :modes_x, :modes_y], positive
        )
        output_k[:, :, -modes_x:, :modes_y] = self._contract(
            field_k[:, :, -modes_x:, :modes_y], negative
        )
        return torch.fft.irfft2(output_k, s=(nx, ny), dim=(-2, -1))


class FNOBlock2d(nn.Module):
    """One global spectral convolution plus a local pointwise pathway."""

    def __init__(self, channels: int, n_modes: tuple[int, int]) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, n_modes)
        self.local = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, field: Tensor) -> Tensor:
        return F.gelu(self.spectral(field) + self.local(field))


class LinearFNO2d(nn.Module):
    """A single learned linear Fourier operator.

    This is the appropriate controlled model when the target is a known linear
    Green-function map. It is also a useful baseline against which a nonlinear
    FNO must demonstrate additional value.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spectral = SpectralConv2d(in_channels, out_channels, n_modes)

    def forward(self, field: Tensor) -> Tensor:
        unbatched = field.ndim == 3
        field_batch = field.unsqueeze(0) if unbatched else field
        if field_batch.ndim != 4 or field_batch.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape ({self.in_channels}, nx, ny) or "
                f"(batch, {self.in_channels}, nx, ny); received {tuple(field.shape)}"
            )
        output = self.spectral(field_batch)
        return output.squeeze(0) if unbatched else output


class FNO2d(nn.Module):
    """A compact FNO for fixed-cell, two-dimensional periodic fields.

    No Cartesian positional embedding is used, so the architecture remains
    equivariant to discrete periodic translations. Biases are disabled and
    GELU maps zero to zero, ensuring that a zero input field produces a zero
    output field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_channels < 1 or n_layers < 1:
            raise ValueError("hidden_channels and n_layers must be positive")
        projection_channels = projection_channels or hidden_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = (int(n_modes[0]), int(n_modes[1]))

        self.lifting = nn.Conv2d(
            self.in_channels, hidden_channels, kernel_size=1, bias=False
        )
        self.blocks = nn.ModuleList(
            FNOBlock2d(hidden_channels, self.n_modes) for _ in range(n_layers)
        )
        self.projection_hidden = nn.Conv2d(
            hidden_channels, projection_channels, kernel_size=1, bias=False
        )
        self.projection_output = nn.Conv2d(
            projection_channels, self.out_channels, kernel_size=1, bias=False
        )

    def forward(self, field: Tensor) -> Tensor:
        unbatched = field.ndim == 3
        if unbatched:
            field = field.unsqueeze(0)
        if field.ndim != 4 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape ({self.in_channels}, nx, ny) or "
                f"(batch, {self.in_channels}, nx, ny); received {tuple(field.shape)}"
            )

        hidden = self.lifting(field)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.projection_output(F.gelu(self.projection_hidden(hidden)))
        return output.squeeze(0) if unbatched else output


class FNOFieldOperator(nn.Module):
    """Adapt :class:`FNO2d` to the particle-mesh field-operator interface.

    The current model is intentionally fixed-cell: ``cell`` is accepted for API
    compatibility but is not used by the network. Input and target RMS scales
    are stored as buffers so normalization accompanies checkpoints.
    """

    def __init__(
        self,
        channels: int,
        n_modes: tuple[int, int],
        *,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        architecture: str = "nonlinear",
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "nonlinear"}:
            raise ValueError("architecture must be 'linear' or 'nonlinear'")
        self.architecture = architecture
        if architecture == "linear":
            self.fno = LinearFNO2d(channels, channels, n_modes)
        else:
            self.fno = FNO2d(
                in_channels=channels,
                out_channels=channels,
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                n_layers=n_layers,
                projection_channels=projection_channels,
            )
        self.register_buffer("input_scale", torch.ones(1, channels, 1, 1))
        self.register_buffer("output_scale", torch.ones(1, channels, 1, 1))

    @torch.no_grad()
    def fit_normalization(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        minimum_scale: float = 1.0e-12,
    ) -> None:
        """Store per-channel RMS scales from batched training fields."""
        if inputs.ndim != 4 or targets.shape != inputs.shape:
            raise ValueError(
                "inputs and targets must have matching "
                "(batch, channels, nx, ny) shapes"
            )
        input_rms = inputs.square().mean(dim=(0, 2, 3), keepdim=True).sqrt()
        output_rms = targets.square().mean(dim=(0, 2, 3), keepdim=True).sqrt()
        self.input_scale.copy_(input_rms.clamp_min(minimum_scale))
        self.output_scale.copy_(output_rms.clamp_min(minimum_scale))

    def forward(self, density: Tensor, cell: Tensor | None = None) -> Tensor:
        del cell
        unbatched = density.ndim == 3
        density_batch = density.unsqueeze(0) if unbatched else density
        if density_batch.ndim != 4:
            raise ValueError(
                "density must have shape (channels, nx, ny) or "
                "(batch, channels, nx, ny)"
            )
        normalized = density_batch / self.input_scale
        potential = self.fno(normalized) * self.output_scale
        return potential.squeeze(0) if unbatched else potential


class SpectralConv2p5d(nn.Module):
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


class PlanarSpectralConv2p5d(nn.Module):
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
        self.spectral = SpectralConv2d(in_channels, out_channels, n_modes)

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


class FNOBlock2p5d(nn.Module):
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
        self.spectral = PlanarSpectralConv2p5d(channels, channels, n_modes)
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


class LinearFNO2p5D(nn.Module):
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
        self.spectral = SpectralConv2p5d(in_channels, out_channels, n_z, n_modes)

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


class FNO2p5D(nn.Module):
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
            FNOBlock2p5d(
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


class FNOFieldOperator2p5D(nn.Module):
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
            self.fno = LinearFNO2p5D(channels, channels, n_z, n_modes)
        else:
            self.fno = FNO2p5D(
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
    """Normalized field-operator adapter for fully periodic 3D fields."""

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
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "nonlinear"}:
            raise ValueError("architecture must be 'linear' or 'nonlinear'")
        self.architecture = architecture
        self.channels = int(channels)
        self.spectral_symmetry = spectral_symmetry
        self.spectral_groups = int(spectral_groups)
        if architecture == "linear":
            self.fno = LinearFNO3d(
                channels,
                channels,
                n_modes,
                spectral_symmetry=spectral_symmetry,
                spectral_groups=spectral_groups,
            )
        else:
            self.fno = FNO3d(
                channels,
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
        if self.spectral_symmetry == "eqgino" and cell is not None:
            if cell.ndim == 2 and cell.shape == (3, 3):
                cells = cell.unsqueeze(0)
            elif cell.ndim == 3 and cell.shape[-2:] == (3, 3):
                cells = cell
            else:
                raise ValueError("cell must have shape (3, 3) or (batch, 3, 3)")
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
        potential = self.fno(normalized) * self.output_scale
        return potential.squeeze(0) if unbatched else potential
