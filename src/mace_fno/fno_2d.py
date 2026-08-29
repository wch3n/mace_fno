"""Planar Fourier neural operators."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
