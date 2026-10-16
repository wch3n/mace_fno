"""Metric-aware scalar EqGINO convolution in the periodic plane of a slab."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .radial import NaturalCubicSpline


class MetricEqGINOSpectralConv2D(nn.Module):
    """Apply shared radial channel matrices independently at every finite z layer.

    For ASE row vectors ``a,b``, the in-plane Gram matrix is ``G = A A^T``
    with ``A = cell[:2]``. Thus ``|k_n|^2 = (2 pi)^2 n^T G^-1 n`` uses
    only the physical plane, independently of the third cell vector. Real
    radial multipliers and a balanced signed-mode cutoff preserve Hermitian
    symmetry. There is no z FFT, z coupling, or wrapping in this layer.

    Shell-spline anchors are the integer-radius shells of a reference square
    with side ``reference_length``. They remain fixed when the input cell
    changes. At that square the spline reproduces an independent shell table
    exactly. D4 equivariance requires a square grid and equal x/y mode counts;
    for other cells only symmetries preserving the metric and cutoff apply.
    This is a scalar spectral adaptation, not the full point-cloud EqGINO.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int],
        *,
        groups: int = 1,
        radial_hidden_channels: int = 16,
        parameterization: str = "shell_spline",
        reference_length: float = 1.0,
        rank: int = 0,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels) < 1:
            raise ValueError("channel counts must be positive")
        if len(n_modes) != 2 or min(n_modes) < 1:
            raise ValueError("n_modes must contain two positive integers")
        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError("positive groups must divide both channel counts")
        if radial_hidden_channels < 1:
            raise ValueError("radial_hidden_channels must be positive")
        if parameterization not in {"shell_spline", "radial_mlp"}:
            raise ValueError(
                "metric parameterization must be 'shell_spline' or 'radial_mlp'"
            )
        if not math.isfinite(reference_length) or reference_length <= 0:
            raise ValueError("metric reference_length must be finite and positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = tuple(int(mode) for mode in n_modes)
        self.groups = int(groups)
        self.parameterization = parameterization
        self.in_channels_per_group = self.in_channels // groups
        self.out_channels_per_group = self.out_channels // groups
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer (zero means dense)")
        if rank and parameterization != "shell_spline":
            raise ValueError("low-rank spectral mixing requires shell_spline")
        if rank > min(self.in_channels_per_group, self.out_channels_per_group):
            raise ValueError("rank cannot exceed the channel dimension per group")
        self.rank = rank
        signed_axes = tuple(
            torch.cat((torch.arange(mode), torch.arange(-mode + 1, 0)))
            for mode in self.n_modes
        )
        mode_xy = torch.stack(torch.meshgrid(*signed_axes, indexing="ij"), dim=-1)
        self.register_buffer("mode_xy", mode_xy)
        matrix_shape = (groups, self.in_channels_per_group, self.out_channels_per_group)
        scale = 1.0 / math.sqrt(
            self.in_channels_per_group * self.out_channels_per_group
        )
        if parameterization == "shell_spline":
            radii = torch.unique(mode_xy.square().sum(-1), sorted=True)
            self.shell_spline = NaturalCubicSpline(radii)
            self.register_buffer(
                "reference_wavenumber_squared",
                torch.tensor(
                    (2 * math.pi / reference_length) ** 2, dtype=torch.float64
                ),
            )
            if rank:
                # Match the dense matrix's entry variance at the reference knots:
                # rank * Var(input) * Var(output) = 1 / (in/group * out/group).
                self.radial_input_factor = nn.Parameter(
                    torch.randn(groups, self.in_channels_per_group, rank, radii.numel())
                    / math.sqrt(self.in_channels_per_group)
                )
                self.radial_output_factor = nn.Parameter(
                    torch.randn(
                        groups, rank, self.out_channels_per_group, radii.numel()
                    )
                    / math.sqrt(rank * self.out_channels_per_group)
                )
            else:
                self.radial_weight = nn.Parameter(
                    scale * torch.randn(*matrix_shape, radii.numel())
                )
        else:
            self.radial_network = nn.Sequential(
                nn.Linear(1, radial_hidden_channels),
                nn.SiLU(),
                nn.Linear(radial_hidden_channels, math.prod(matrix_shape)),
            )
            nn.init.normal_(
                self.radial_network[-1].weight,
                std=scale / math.sqrt(radial_hidden_channels),
            )
            nn.init.normal_(self.radial_network[-1].bias, std=scale)

    def radial_weights(self, squared_wavevectors: Tensor) -> Tensor:
        """Return matrices for inspection; low-rank forward never forms these."""
        if self.rank:
            left, right = self.radial_factors(squared_wavevectors)
            return left @ right
        if self.parameterization == "shell_spline":
            return self.shell_spline(
                squared_wavevectors / self.reference_wavenumber_squared,
                self.radial_weight.movedim(-1, 0),
            )
        return self.radial_network(
            torch.log1p(squared_wavevectors).unsqueeze(-1)
        ).reshape(
            *squared_wavevectors.shape,
            self.groups,
            self.in_channels_per_group,
            self.out_channels_per_group,
        )

    def radial_factors(self, squared_wavevectors: Tensor) -> tuple[Tensor, Tensor]:
        """Interpolate factors separately, preserving rank at every wavevector.

        This defines a factorized radial function, not a spline interpolation
        of reconstructed dense matrices (which need not preserve their rank).
        Shapes are ``(*q.shape, groups, in/group, rank)`` and
        ``(*q.shape, groups, rank, out/group)``.
        """
        if not self.rank:
            raise ValueError("radial_factors requires a positive rank")
        query = squared_wavevectors / self.reference_wavenumber_squared
        return (
            self.shell_spline(query, self.radial_input_factor.movedim(-1, 0)),
            self.shell_spline(query, self.radial_output_factor.movedim(-1, 0)),
        )

    def _physical_squared_wavevectors(self, cells: Tensor) -> Tensor:
        """Return retained squared physical wavevectors, shaped (batch, x, y)."""
        plane = cells[:, :2]
        gram = plane @ plane.transpose(-1, -2)
        modes = self.mode_xy.to(device=cells.device, dtype=cells.dtype)
        rhs = modes.reshape(-1, 2).T.unsqueeze(0).expand(cells.shape[0], -1, -1)
        dual_modes = torch.linalg.solve(gram, rhs)
        squared = (2 * math.pi) ** 2 * (rhs * dual_modes).sum(dim=1)
        return squared.reshape(cells.shape[0], *modes.shape[:-1])

    def forward(self, field: Tensor, cell: Tensor) -> Tensor:
        if field.ndim != 5 or field.shape[1] != self.in_channels:
            raise ValueError(
                f"field must have shape (batch, {self.in_channels}, nz, nx, ny)"
            )
        if cell.shape == (3, 3):
            cells = cell.unsqueeze(0).expand(field.shape[0], -1, -1)
        elif cell.shape == (field.shape[0], 3, 3):
            cells = cell
        else:
            raise ValueError("cell must have shape (3, 3) or (batch, 3, 3)")
        if not torch.is_floating_point(cells):
            raise TypeError("cell must be a floating-point tensor")
        if cells.device != field.device or cells.dtype != field.dtype:
            raise ValueError("field and cell must use the same device and dtype")
        area = torch.linalg.vector_norm(
            torch.linalg.cross(cells[:, 0], cells[:, 1]), dim=-1
        )
        valid = torch.isfinite(cells).all(dim=(-2, -1)) & torch.isfinite(area)
        valid = valid & (area > torch.finfo(cells.dtype).eps)
        if not bool(valid.all().detach().cpu()):
            raise ValueError(
                "every metric-aware slab cell must have a finite nondegenerate plane"
            )
        nz, nx, ny = field.shape[-3:]
        if any(
            2 * mode > size for mode, size in zip(self.n_modes, (nx, ny), strict=True)
        ):
            raise ValueError(
                "twice every retained mode count must fit its lateral grid axis"
            )
        ix, iy = (
            torch.cat(
                (
                    torch.arange(mode, device=field.device),
                    torch.arange(size - mode + 1, size, device=field.device),
                )
            )
            for mode, size in zip(self.n_modes, (nx, ny), strict=True)
        )
        field_k = torch.fft.fft2(field, dim=(-2, -1))
        retained = field_k[:, :, :, ix[:, None], iy[None, :]].reshape(
            field.shape[0],
            self.groups,
            self.in_channels_per_group,
            nz,
            ix.numel(),
            iy.numel(),
        )
        # Evaluate the metric once per structure, not once per z layer.
        squared_wavevectors = self._physical_squared_wavevectors(cells)
        if self.rank:
            left, right = (
                factor.permute(0, 3, 4, 5, 1, 2)
                for factor in self.radial_factors(squared_wavevectors)
            )

            def contract(values: Tensor) -> Tensor:
                reduced = torch.einsum("bgizxy,bgirxy->bgrzxy", values, left)
                return torch.einsum("bgrzxy,bgroxy->bgozxy", reduced, right)

            transformed = torch.complex(
                contract(retained.real), contract(retained.imag)
            )
        else:
            weights = self.radial_weights(squared_wavevectors).permute(0, 3, 4, 5, 1, 2)
            transformed = torch.complex(
                torch.einsum("bgizxy,bgioxy->bgozxy", retained.real, weights),
                torch.einsum("bgizxy,bgioxy->bgozxy", retained.imag, weights),
            )
        transformed = transformed.reshape(
            field.shape[0], self.out_channels, nz, ix.numel(), iy.numel()
        )
        output_k = field_k.new_zeros((field.shape[0], self.out_channels, nz, nx, ny))
        output_k[:, :, :, ix[:, None], iy[None, :]] = transformed
        return torch.fft.ifft2(output_k, dim=(-2, -1)).real
