"""Checkpoint diagnostics for the effective Fourier response of latent fields.

The learned FNO is generally nonlinear and has multiple latent input channels,
so an individual layer weight is not an interpretable interaction kernel.  The
functions here instead measure the quadratic curvature of the *end-to-end*
field energy around a deposited density.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from .geometry import reciprocal_vectors_2d


def unique_integer_modes(max_mode: int) -> list[tuple[int, int, int]]:
    """Return nonzero reciprocal modes in ``(z, x, y)`` order without ± duplicates.

    A cosine perturbation is invariant under ``m -> -m``.  Retaining only the
    sign whose first nonzero component is positive avoids double counting the
    same real field while preserving every reciprocal-radius shell.
    """
    if int(max_mode) != max_mode or max_mode < 1:
        raise ValueError("max_mode must be a positive integer")
    result: list[tuple[int, int, int]] = []
    for mode_z in range(-max_mode, max_mode + 1):
        for mode_x in range(-max_mode, max_mode + 1):
            for mode_y in range(-max_mode, max_mode + 1):
                mode = (mode_z, mode_x, mode_y)
                if mode == (0, 0, 0):
                    continue
                for component in mode:
                    if component != 0:
                        if component > 0:
                            result.append(mode)
                        break
    return result


def unique_integer_modes_2d(max_mode: int) -> list[tuple[int, int]]:
    """Return nonzero planar modes in ``(x,y)`` order without ± duplicates."""
    if int(max_mode) != max_mode or max_mode < 1:
        raise ValueError("max_mode must be a positive integer")
    result: list[tuple[int, int]] = []
    for mode_x in range(-max_mode, max_mode + 1):
        for mode_y in range(-max_mode, max_mode + 1):
            if (mode_x, mode_y) == (0, 0):
                continue
            if mode_x > 0 or (mode_x == 0 and mode_y > 0):
                result.append((mode_x, mode_y))
    return result


def unit_rms_cosine_mode(
    grid_shape: tuple[int, int, int],
    mode_zxy: tuple[int, int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Construct a zero-mean, unit-RMS real Fourier mode on a ``(z,x,y)`` grid."""
    if len(grid_shape) != 3 or min(grid_shape) < 2:
        raise ValueError("grid_shape must contain three dimensions of at least two")
    if len(mode_zxy) != 3 or all(component == 0 for component in mode_zxy):
        raise ValueError("mode_zxy must contain one nonzero integer component")
    nz, nx, ny = (int(value) for value in grid_shape)
    mode_z, mode_x, mode_y = (int(value) for value in mode_zxy)
    z = torch.arange(nz, dtype=dtype, device=device).reshape(nz, 1, 1)
    x = torch.arange(nx, dtype=dtype, device=device).reshape(1, nx, 1)
    y = torch.arange(ny, dtype=dtype, device=device).reshape(1, 1, ny)
    phase = 2.0 * math.pi * (
        mode_z * z / nz + mode_x * x / nx + mode_y * y / ny
    )
    field = torch.cos(phase)
    field = field - field.mean()
    return field / field.square().mean().sqrt().clamp_min(torch.finfo(dtype).eps)


def unit_rms_cosine_mode_2d(
    grid_shape: tuple[int, int],
    mode_xy: tuple[int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Construct a zero-mean, unit-RMS cosine on a periodic ``(x,y)`` grid."""
    if len(grid_shape) != 2 or min(grid_shape) < 2:
        raise ValueError("grid_shape must contain two dimensions of at least two")
    if len(mode_xy) != 2 or all(component == 0 for component in mode_xy):
        raise ValueError("mode_xy must contain one nonzero integer component")
    nx, ny = (int(value) for value in grid_shape)
    mode_x, mode_y = (int(value) for value in mode_xy)
    x = torch.arange(nx, dtype=dtype, device=device).reshape(nx, 1)
    y = torch.arange(ny, dtype=dtype, device=device).reshape(1, ny)
    phase = 2.0 * math.pi * (mode_x * x / nx + mode_y * y / ny)
    field = torch.cos(phase)
    field = field - field.mean()
    return field / field.square().mean().sqrt().clamp_min(torch.finfo(dtype).eps)


def wavevector(cell: Tensor, mode_zxy: tuple[int, int, int]) -> Tensor:
    """Return the Cartesian wavevector for a ``(z,x,y)`` integer mesh mode.

    The mesh order is ``(z,x,y)`` whereas fractional cell coordinates are
    ``(x,y,z)``.  For ``r = f @ cell`` and a phase ``2π m·f``, the Cartesian
    reciprocal vector is ``k = 2π cell^{-1} m``.
    """
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    mode_z, mode_x, mode_y = (int(value) for value in mode_zxy)
    mode_xyz = cell.new_tensor((mode_x, mode_y, mode_z))
    return 2.0 * math.pi * torch.linalg.solve(cell, mode_xyz)


def wavevector_norm(cell: Tensor, mode_zxy: tuple[int, int, int]) -> Tensor:
    """Return ``|k|`` in inverse Angstrom for a 3D mesh mode."""
    return torch.linalg.vector_norm(wavevector(cell, mode_zxy))


def planar_wavevector(cell: Tensor, mode_xy: tuple[int, int]) -> Tensor:
    """Return the Cartesian in-plane wavevector for an ``(x,y)`` mode."""
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    mode_x, mode_y = (int(value) for value in mode_xy)
    b1, b2 = reciprocal_vectors_2d(cell)
    return mode_x * b1 + mode_y * b2


def slab_z_profiles(
    n_z: int,
    count: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[Tensor, list[str]]:
    """Return discrete orthonormal monopole, dipole, and quadrupole profiles."""
    if n_z < 3:
        raise ValueError("n_z must be at least three")
    if count not in {1, 2, 3}:
        raise ValueError("count must select one to three z profiles")
    coordinate = (
        torch.arange(n_z, device=device, dtype=dtype) + 0.5
    ) / n_z - 0.5
    candidates = [
        torch.ones_like(coordinate),
        coordinate,
        coordinate.square(),
    ]
    names = ["monopole", "dipole", "quadrupole"]
    profiles: list[Tensor] = []
    for candidate in candidates[:count]:
        profile = candidate
        for previous in profiles:
            profile = profile - (profile * previous).mean() * previous
        profile = profile / profile.square().mean().sqrt().clamp_min(
            torch.finfo(dtype).eps
        )
        profiles.append(profile)
    return torch.stack(profiles), names[:count]


def slab_coulomb_profile_matrix(
    profiles: Tensor,
    k_parallel: float | Tensor,
    z_extent: float,
) -> Tensor:
    r"""Project ``2π exp(-k|z-z'|)/k`` onto discrete z profiles."""
    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (n_profiles, n_z)")
    if z_extent <= 0.0:
        raise ValueError("z_extent must be positive")
    k = torch.as_tensor(k_parallel, dtype=profiles.dtype, device=profiles.device)
    if k.ndim != 0 or not bool((k > 0).detach().cpu()):
        raise ValueError("k_parallel must be a positive scalar")
    n_z = profiles.shape[1]
    z = (
        (torch.arange(n_z, device=profiles.device, dtype=profiles.dtype) + 0.5)
        / n_z
        - 0.5
    ) * z_extent
    green = (2.0 * math.pi / k) * torch.exp(-k * (z[:, None] - z[None, :]).abs())
    dz = z_extent / n_z
    return dz * dz * torch.einsum("mz,zw,nw->mn", profiles, green, profiles)


def quadratic_mode_response(
    density: Tensor,
    mode: Tensor,
    amplitude: float | Tensor,
    evaluate_batch: Callable[[Tensor], Tensor],
) -> Tensor:
    """Measure the channel-space energy curvature for a real Fourier mode.

    ``density`` may be planar ``(channels,nx,ny)`` or volumetric
    ``(channels,nz,nx,ny)``. ``evaluate_batch`` receives a leading batch
    dimension and returns one energy per field. Central differences recover
    the full symmetric channel-response matrix, including cross-channel terms.
    """
    if density.ndim not in {3, 4}:
        raise ValueError("density must contain channels and two or three mesh axes")
    if mode.shape != density.shape[1:]:
        raise ValueError("mode must have the density spatial shape")
    if mode.dtype != density.dtype or mode.device != density.device:
        raise ValueError("mode and density must have matching device and dtype")
    channels = density.shape[0]
    basis = density.new_zeros((channels, *density.shape))
    indices = torch.arange(channels, device=density.device)
    basis[indices, indices] = mode

    return quadratic_basis_response(density, basis, amplitude, evaluate_batch)


def quadratic_basis_response(
    density: Tensor,
    basis: Tensor,
    amplitude: float | Tensor,
    evaluate_batch: Callable[[Tensor], Tensor],
) -> Tensor:
    """Measure energy curvature along arbitrary unit-normalized field probes.

    ``basis`` has shape ``(n_probes, *density.shape)``. The returned symmetric
    matrix contains the second derivatives with respect to the corresponding
    probe amplitudes.
    """
    if density.ndim < 2:
        raise ValueError("density must contain channel and spatial dimensions")
    if basis.ndim != density.ndim + 1 or basis.shape[1:] != density.shape:
        raise ValueError("basis must have shape (n_probes, *density.shape)")
    if basis.shape[0] < 1:
        raise ValueError("at least one probe basis field is required")
    if basis.dtype != density.dtype or basis.device != density.device:
        raise ValueError("basis and density must have matching device and dtype")
    amplitude_tensor = torch.as_tensor(
        amplitude, dtype=density.dtype, device=density.device
    )
    if amplitude_tensor.ndim != 0 or not bool((amplitude_tensor > 0).detach().cpu()):
        raise ValueError("amplitude must be a positive scalar")

    fields: list[Tensor] = []
    diagonal_indices: list[tuple[int, int]] = []
    cross_indices: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    probes = basis.shape[0]
    for probe in range(probes):
        diagonal_indices.append((len(fields), len(fields) + 1))
        fields.extend(
            (
                density + amplitude_tensor * basis[probe],
                density - amplitude_tensor * basis[probe],
            )
        )
    for first in range(probes):
        for second in range(first + 1, probes):
            plus_direction = basis[first] + basis[second]
            minus_direction = basis[first] - basis[second]
            cross_indices[(first, second)] = (
                len(fields),
                len(fields) + 1,
                len(fields) + 2,
                len(fields) + 3,
            )
            fields.extend(
                (
                    density + amplitude_tensor * plus_direction,
                    density - amplitude_tensor * plus_direction,
                    density + amplitude_tensor * minus_direction,
                    density - amplitude_tensor * minus_direction,
                )
            )

    energies = evaluate_batch(torch.stack(fields, dim=0))
    if energies.shape != (len(fields),):
        raise ValueError("evaluate_batch must return one scalar energy per field")
    reference = evaluate_batch(density.unsqueeze(0))
    if reference.shape != (1,):
        raise ValueError("evaluate_batch must return a scalar energy for one field")
    reference_energy = reference[0]
    amplitude_squared = amplitude_tensor.square()
    response = density.new_zeros((probes, probes))
    for probe, (plus, minus) in enumerate(diagonal_indices):
        response[probe, probe] = (
            energies[plus] + energies[minus] - 2.0 * reference_energy
        ) / amplitude_squared
    for (first, second), (pp, pm, mp, mm) in cross_indices.items():
        value = (energies[pp] + energies[pm] - energies[mp] - energies[mm]) / (
            4.0 * amplitude_squared
        )
        response[first, second] = value
        response[second, first] = value
    return response


def fit_reference_power_response(
    points: list[tuple[float, float]],
    reference_exponent: float,
) -> dict[str, float] | None:
    """Fit positive response data to free and fixed-reference power laws.

    The fit is deliberately a post-processing helper: it carries no gradient
    and is intended for validation diagnostics rather than an optimization
    target.  The returned ``R^2`` values are computed in log-response space.
    """
    if not math.isfinite(reference_exponent) or reference_exponent <= 0.0:
        raise ValueError("reference_exponent must be positive and finite")
    if len(points) < 2:
        return None
    values = torch.tensor(points, dtype=torch.float64)
    k = values[:, 0]
    response = values[:, 1]
    valid = torch.isfinite(k) & torch.isfinite(response) & (k > 0.0) & (response > 0.0)
    if int(valid.sum()) < 2:
        return None
    log_k = torch.log(k[valid])
    log_response = torch.log(response[valid])
    design = torch.stack((torch.ones_like(log_k), log_k), dim=1)
    intercept, slope = torch.linalg.lstsq(design, log_response).solution
    predicted = intercept + slope * log_k
    total = torch.square(log_response - log_response.mean()).sum()
    free_r2 = 1.0 if float(total) == 0.0 else 1.0 - float(
        torch.square(log_response - predicted).sum() / total
    )

    reference_intercept = (log_response + reference_exponent * log_k).mean()
    reference_predicted = reference_intercept - reference_exponent * log_k
    reference_r2 = 1.0 if float(total) == 0.0 else 1.0 - float(
        torch.square(log_response - reference_predicted).sum() / total
    )
    return {
        "points": int(valid.sum()),
        "free_power_exponent_p": float(-slope),
        "free_log_r2": free_r2,
        "reference_power_exponent_p": float(reference_exponent),
        "reference_power_log_r2": reference_r2,
        "reference_prefactor_latent_units": float(torch.exp(reference_intercept)),
    }


def fit_power_law_response(
    points: list[tuple[float, float]],
) -> dict[str, float] | None:
    """Fit free and Coulomb ``1/k^2`` laws, preserving the original API."""
    result = fit_reference_power_response(points, 2.0)
    if result is None:
        return None
    result["coulomb_p2_log_r2"] = result["reference_power_log_r2"]
    result["coulomb_prefactor_latent_units"] = result[
        "reference_prefactor_latent_units"
    ]
    return result


def fit_anisotropic_inverse_quadratic_response(
    points: list[tuple[tuple[float, float, float], float]],
) -> dict[str, object] | None:
    r"""Fit ``response(k) = 1 / (k.T @ B @ k)`` for symmetric ``B``.

    ``B`` contains the dielectric tensor divided by an unidentifiable overall
    response prefactor. Its trace-normalized form therefore carries the useful
    directional information. The unconstrained fit reports its eigenvalues so
    a non-positive result is visible rather than silently projected to an SPD
    tensor.
    """
    valid_points = [
        (vector, response)
        for vector, response in points
        if response > 0.0
        and math.isfinite(response)
        and all(math.isfinite(component) for component in vector)
    ]
    if len(valid_points) < 6:
        return None
    vectors = torch.tensor(
        [vector for vector, _ in valid_points], dtype=torch.float64
    )
    response = torch.tensor(
        [value for _, value in valid_points], dtype=torch.float64
    )
    kx, ky, kz = vectors.unbind(dim=1)
    design = torch.stack(
        (kx.square(), ky.square(), kz.square(), 2 * kx * ky, 2 * kx * kz, 2 * ky * kz),
        dim=1,
    )
    rank = int(torch.linalg.matrix_rank(design))
    if rank < 6:
        return None
    coefficients = torch.linalg.lstsq(design, response.reciprocal()).solution
    matrix = torch.stack(
        (
            torch.stack((coefficients[0], coefficients[3], coefficients[4])),
            torch.stack((coefficients[3], coefficients[1], coefficients[5])),
            torch.stack((coefficients[4], coefficients[5], coefficients[2])),
        )
    )
    predicted_inverse = design @ coefficients
    target_inverse = response.reciprocal()
    inverse_total = torch.square(target_inverse - target_inverse.mean()).sum()
    inverse_r2 = 1.0 if float(inverse_total) == 0.0 else 1.0 - float(
        torch.square(target_inverse - predicted_inverse).sum() / inverse_total
    )
    positive_prediction = predicted_inverse > 0.0
    log_r2: float | None = None
    if int(positive_prediction.sum()) >= 2:
        observed_log = torch.log(response[positive_prediction])
        predicted_log = -torch.log(predicted_inverse[positive_prediction])
        log_total = torch.square(observed_log - observed_log.mean()).sum()
        log_r2 = 1.0 if float(log_total) == 0.0 else 1.0 - float(
            torch.square(observed_log - predicted_log).sum() / log_total
        )
    eigenvalues = torch.linalg.eigvalsh(matrix)
    trace = torch.trace(matrix)
    normalized = matrix / trace * 3.0 if float(trace) != 0.0 else None
    return {
        "points": len(valid_points),
        "design_rank": rank,
        "inverse_response_r2": inverse_r2,
        "log_response_r2": log_r2,
        "dielectric_over_prefactor_matrix": matrix.tolist(),
        "trace_normalized_dielectric_tensor": (
            normalized.tolist() if normalized is not None else None
        ),
        "tensor_eigenvalues": eigenvalues.tolist(),
        "positive_definite": bool((eigenvalues > 0.0).all()),
    }
