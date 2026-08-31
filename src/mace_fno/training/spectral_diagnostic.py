"""Validation-only low-wavevector diagnostics for 2D, 2.5D, and 3D residuals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from ..coupling import MACEFNOResidual
from ..spectral_response import (
    fit_anisotropic_inverse_quadratic_response,
    fit_power_law_response,
    fit_reference_power_response,
    planar_wavevector,
    quadratic_basis_response,
    quadratic_mode_response,
    slab_coulomb_profile_matrix,
    slab_z_profiles,
    unique_integer_modes,
    unique_integer_modes_2d,
    unit_rms_cosine_mode,
    unit_rms_cosine_mode_2d,
    wavevector,
)
from .data import collate_samples


def _field_energies(
    model: MACEFNOResidual,
    fields: torch.Tensor,
    cell: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("field_batch_size must be positive")
    energies: list[torch.Tensor] = []
    for start in range(0, fields.shape[0], batch_size):
        field_batch = fields[start : start + batch_size]
        cells = cell.unsqueeze(0).expand(field_batch.shape[0], -1, -1)
        energies.append(model.long_range.energy_from_density(field_batch, cells))
    return torch.cat(energies, dim=0)


def _select_indices(
    samples: list[dict[str, Any]], sample_indices: Sequence[int] | None
) -> list[int]:
    if not samples:
        raise ValueError("at least one validation sample is required")
    if sample_indices is None:
        return list(range(len(samples)))
    selected = [int(index) for index in sample_indices]
    if not selected:
        raise ValueError("sample_indices must not be empty")
    if min(selected) < 0 or max(selected) >= len(samples):
        raise ValueError("a spectral diagnostic sample index is out of range")
    return selected


def _assign_physical_shell_ranks(mode_reports: list[dict[str, Any]]) -> int:
    """Annotate modes using distinct physical |k| values, not integer radii."""
    representatives: list[float] = []
    for report in sorted(mode_reports, key=lambda item: item["k_inverse_angstrom"]):
        magnitude = float(report["k_inverse_angstrom"])
        rank = None
        for index, representative in enumerate(representatives):
            if math.isclose(
                magnitude, representative, rel_tol=2.0e-6, abs_tol=1.0e-10
            ):
                rank = index + 1
                break
        if rank is None:
            representatives.append(magnitude)
            rank = len(representatives)
        report["physical_shell_rank"] = rank
    return len(representatives)


def _shell_average_points(
    mode_reports: list[dict[str, Any]],
    response_key: str,
    *,
    maximum_rank: int | None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    ranks = sorted({int(report["physical_shell_rank"]) for report in mode_reports})
    for rank in ranks:
        if maximum_rank is not None and rank > maximum_rank:
            continue
        selected = [
            report
            for report in mode_reports
            if report["physical_shell_rank"] == rank
            and report.get(response_key) is not None
        ]
        if selected:
            points.append(
                (
                    sum(float(report["k_inverse_angstrom"]) for report in selected)
                    / len(selected),
                    sum(float(report[response_key]) for report in selected)
                    / len(selected),
                )
            )
    return points


def _validate_common(
    samples: list[dict[str, Any]],
    sample_indices: Sequence[int] | None,
    *,
    max_mode: int,
    fit_shells: int,
    relative_amplitude: float,
    field_batch_size: int,
) -> list[int]:
    if min(max_mode, fit_shells, field_batch_size) < 1:
        raise ValueError("max_mode, fit_shells, and field_batch_size must be positive")
    if relative_amplitude <= 0.0:
        raise ValueError("relative_amplitude must be positive")
    return _select_indices(samples, sample_indices)


def periodic_3d_response_diagnostic(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    sample_indices: Sequence[int] | None = None,
    max_mode: int = 1,
    fit_shells: int = 3,
    relative_amplitude: float = 0.05,
    field_batch_size: int = 32,
) -> dict[str, Any]:
    r"""Measure scalar and tensor-aware low-:math:`k` response in periodic 3D."""
    if model.spatial_scheme != "3d":
        raise ValueError("periodic_3d_response_diagnostic requires the 3D scheme")
    if model.long_range.volume_interlacing != 1:
        raise ValueError(
            "the 3D diagnostic requires volume_interlacing=1 because an "
            "interlaced mesh has no unique deposited field"
        )
    selected_indices = _validate_common(
        samples,
        sample_indices,
        max_mode=max_mode,
        fit_shells=fit_shells,
        relative_amplitude=relative_amplitude,
        field_batch_size=field_batch_size,
    )
    grid_shape = tuple(int(value) for value in model.long_range.assignment.grid_shape)
    if len(grid_shape) != 3:
        raise RuntimeError("the periodic 3D mesh must expose three dimensions")
    if 2 * max_mode >= min(grid_shape):
        raise ValueError("max_mode must remain below the mesh Nyquist limit")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    modes = unique_integer_modes(max_mode)
    was_training = model.training
    per_sample: list[dict[str, Any]] = []
    low_k_points: list[tuple[float, float]] = []
    full_range_points: list[tuple[float, float]] = []
    pooled_tensor_points: list[tuple[tuple[float, float, float], float]] = []
    model.eval()
    try:
        with torch.no_grad():
            for sample_index in selected_indices:
                graph, _, _ = collate_samples([samples[sample_index]], device, dtype)
                output = model(
                    graph, training=False, compute_force=False, return_fields=True
                )
                density = output["density"][0]
                if density.ndim != 4:
                    raise RuntimeError("the 3D path must return one unbatched density")
                cell = graph["cell"].reshape(-1, 3, 3)[0]
                volume = torch.linalg.det(cell).abs()
                density_rms = density.square().mean().sqrt().clamp_min(
                    torch.finfo(dtype).eps
                )
                amplitude = density_rms * relative_amplitude
                mode_reports: list[dict[str, Any]] = []
                tensor_points: list[tuple[tuple[float, float, float], float]] = []
                for mode_zxy in modes:
                    perturbation = unit_rms_cosine_mode(
                        grid_shape, mode_zxy, device=device, dtype=dtype
                    )
                    response = quadratic_mode_response(
                        density,
                        perturbation,
                        amplitude,
                        lambda fields: _field_energies(
                            model, fields, cell, batch_size=field_batch_size
                        ),
                    ) / volume
                    eigenvalues = torch.linalg.eigvalsh(response).flip(0)
                    dominant = float(eigenvalues[0])
                    k_vector = wavevector(cell, mode_zxy)
                    k_tuple = tuple(float(component) for component in k_vector)
                    k_norm = float(torch.linalg.vector_norm(k_vector))
                    mode_reports.append(
                        {
                            "mode_zxy": list(mode_zxy),
                            "integer_squared_norm": sum(
                                component * component for component in mode_zxy
                            ),
                            "k_cartesian_inverse_angstrom": list(k_tuple),
                            "k_inverse_angstrom": k_norm,
                            "eigenvalues_per_volume": eigenvalues.cpu().tolist(),
                            "dominant_positive_eigenvalue_per_volume": (
                                dominant if dominant > 0.0 else None
                            ),
                        }
                    )
                    if dominant > 0.0:
                        tensor_points.append((k_tuple, dominant))
                        pooled_tensor_points.append((k_tuple, dominant))
                shell_count = _assign_physical_shell_ranks(mode_reports)
                sample_low_k_points = _shell_average_points(
                    mode_reports,
                    "dominant_positive_eigenvalue_per_volume",
                    maximum_rank=fit_shells,
                )
                sample_full_range_points = _shell_average_points(
                    mode_reports,
                    "dominant_positive_eigenvalue_per_volume",
                    maximum_rank=None,
                )
                low_k_points.extend(sample_low_k_points)
                full_range_points.extend(sample_full_range_points)
                per_sample.append(
                    {
                        "sample_index": sample_index,
                        "cell_volume_angstrom3": float(volume),
                        "density_rms_latent_units_per_angstrom3": float(density_rms),
                        "probe_amplitude_latent_units_per_angstrom3": float(amplitude),
                        "physical_shells": shell_count,
                        "anisotropic_inverse_quadratic_fit": (
                            fit_anisotropic_inverse_quadratic_response(tensor_points)
                        ),
                        "low_k_dominant_eigenvalue_fit": fit_power_law_response(
                            sample_low_k_points
                        ),
                        "full_probed_range_dominant_eigenvalue_fit": (
                            fit_power_law_response(sample_full_range_points)
                        ),
                        "modes": mode_reports,
                    }
                )
    finally:
        model.train(was_training)

    return {
        "diagnostic_kind": "periodic_3d",
        "spatial_scheme": "3d",
        "samples": len(per_sample),
        "sample_indices": selected_indices,
        "grid_shape_zxy": list(grid_shape),
        "max_mode": max_mode,
        "low_k_fit_shells": fit_shells,
        "relative_amplitude": relative_amplitude,
        "field_batch_size": field_batch_size,
        "probes_per_mode": int(model.source_head.channels),
        "field_evaluations_per_mode": int(
            2 * model.source_head.channels**2 + 1
        ),
        "description": (
            "Validation-only curvature of the 3D residual energy under neutral "
            "Fourier perturbations. Scalar fits use distinct physical |k| shells; "
            "individual vectors additionally fit response=1/(k.T B k), where the "
            "trace-normalized B diagnoses anisotropic dielectric shape."
        ),
        "low_k_dominant_eigenvalue_fit": fit_power_law_response(low_k_points),
        "full_probed_range_dominant_eigenvalue_fit": fit_power_law_response(
            full_range_points
        ),
        "pooled_anisotropic_inverse_quadratic_fit": (
            fit_anisotropic_inverse_quadratic_response(pooled_tensor_points)
        ),
        "per_sample_response": per_sample,
    }


def planar_2d_response_diagnostic(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    sample_indices: Sequence[int] | None = None,
    max_mode: int = 1,
    fit_shells: int = 2,
    relative_amplitude: float = 0.05,
    field_batch_size: int = 32,
) -> dict[str, Any]:
    r"""Measure the effective thin-sheet response of a planar 2D FNO."""
    if model.spatial_scheme != "2d":
        raise ValueError("planar_2d_response_diagnostic requires the 2D scheme")
    selected_indices = _validate_common(
        samples,
        sample_indices,
        max_mode=max_mode,
        fit_shells=fit_shells,
        relative_amplitude=relative_amplitude,
        field_batch_size=field_batch_size,
    )
    grid_shape = tuple(int(value) for value in model.long_range.assignment.grid_shape)
    if len(grid_shape) != 2:
        raise RuntimeError("the planar 2D mesh must expose (nx,ny)")
    if 2 * max_mode >= min(grid_shape):
        raise ValueError("max_mode must remain below the planar mesh Nyquist limit")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    modes = unique_integer_modes_2d(max_mode)
    was_training = model.training
    per_sample: list[dict[str, Any]] = []
    low_k_points: list[tuple[float, float]] = []
    full_range_points: list[tuple[float, float]] = []
    model.eval()
    try:
        with torch.no_grad():
            for sample_index in selected_indices:
                graph, _, _ = collate_samples([samples[sample_index]], device, dtype)
                output = model(
                    graph, training=False, compute_force=False, return_fields=True
                )
                density = output["density"][0]
                if density.ndim != 3:
                    raise RuntimeError("the 2D path must return one unbatched density")
                cell = graph["cell"].reshape(-1, 3, 3)[0]
                area = torch.linalg.vector_norm(
                    torch.linalg.cross(cell[0], cell[1])
                )
                density_rms = density.square().mean().sqrt().clamp_min(
                    torch.finfo(dtype).eps
                )
                amplitude = density_rms * relative_amplitude
                mode_reports: list[dict[str, Any]] = []
                for mode_xy in modes:
                    perturbation = unit_rms_cosine_mode_2d(
                        grid_shape, mode_xy, device=device, dtype=dtype
                    )
                    response = quadratic_mode_response(
                        density,
                        perturbation,
                        amplitude,
                        lambda fields: _field_energies(
                            model, fields, cell, batch_size=field_batch_size
                        ),
                    ) / area
                    eigenvalues = torch.linalg.eigvalsh(response).flip(0)
                    dominant = float(eigenvalues[0])
                    k_vector = planar_wavevector(cell, mode_xy)
                    k_norm = float(torch.linalg.vector_norm(k_vector))
                    mode_reports.append(
                        {
                            "mode_xy": list(mode_xy),
                            "k_cartesian_inverse_angstrom": [
                                float(component) for component in k_vector
                            ],
                            "k_parallel_inverse_angstrom": k_norm,
                            "k_inverse_angstrom": k_norm,
                            "channel_eigenvalues_per_area": eigenvalues.cpu().tolist(),
                            "dominant_positive_eigenvalue_per_area": (
                                dominant if dominant > 0.0 else None
                            ),
                        }
                    )
                shell_count = _assign_physical_shell_ranks(mode_reports)
                sample_low_k_points = _shell_average_points(
                    mode_reports,
                    "dominant_positive_eigenvalue_per_area",
                    maximum_rank=fit_shells,
                )
                sample_full_range_points = _shell_average_points(
                    mode_reports,
                    "dominant_positive_eigenvalue_per_area",
                    maximum_rank=None,
                )
                low_k_points.extend(sample_low_k_points)
                full_range_points.extend(sample_full_range_points)
                per_sample.append(
                    {
                        "sample_index": sample_index,
                        "in_plane_area_angstrom2": float(area),
                        "density_rms_latent_units_per_angstrom2": float(density_rms),
                        "probe_amplitude_latent_units_per_angstrom2": float(amplitude),
                        "physical_shells": shell_count,
                        "low_k_planar_response_fit": fit_reference_power_response(
                            sample_low_k_points, 1.0
                        ),
                        "full_probed_range_planar_response_fit": (
                            fit_reference_power_response(
                                sample_full_range_points, 1.0
                            )
                        ),
                        "modes": mode_reports,
                    }
                )
    finally:
        model.train(was_training)

    return {
        "diagnostic_kind": "planar_2d",
        "spatial_scheme": "2d",
        "samples": len(per_sample),
        "sample_indices": selected_indices,
        "grid_shape_xy": list(grid_shape),
        "max_mode": max_mode,
        "low_k_fit_shells": fit_shells,
        "relative_amplitude": relative_amplitude,
        "field_batch_size": field_batch_size,
        "probes_per_mode": int(model.source_head.channels),
        "field_evaluations_per_mode": int(
            2 * model.source_head.channels**2 + 1
        ),
        "description": (
            "Validation-only channel curvature under neutral planar Fourier "
            "probes. The dominant response is compared with the effective "
            "thin-sheet Coulomb kernel 1/k_parallel; no z-profile information "
            "exists in this representation."
        ),
        "low_k_planar_response_fit": fit_reference_power_response(
            low_k_points, 1.0
        ),
        "full_probed_range_planar_response_fit": fit_reference_power_response(
            full_range_points, 1.0
        ),
        "per_sample_response": per_sample,
    }


def _slab_probe_basis(
    density: torch.Tensor,
    planar_mode: torch.Tensor,
    profiles: torch.Tensor,
) -> torch.Tensor:
    channels = density.shape[0]
    profile_count = profiles.shape[0]
    spatial = profiles[:, :, None, None] * planar_mode[None, None, :, :]
    basis = density.new_zeros((channels * profile_count, *density.shape))
    for channel in range(channels):
        basis[
            channel * profile_count : (channel + 1) * profile_count, channel
        ] = spatial
    return basis


def _slab_template_fit(
    response: torch.Tensor,
    template: torch.Tensor,
    channels: int,
) -> dict[str, Any]:
    profile_count = template.shape[0]
    blocks = response.reshape(channels, profile_count, channels, profile_count)
    denominator = template.square().sum().clamp_min(torch.finfo(response.dtype).eps)
    channel_metric = torch.einsum("cmdn,mn->cd", blocks, template) / denominator
    channel_metric = 0.5 * (channel_metric + channel_metric.T)
    reconstruction = torch.einsum("cd,mn->cmdn", channel_metric, template)
    response_norm = torch.linalg.vector_norm(blocks)
    relative_error = None
    if profile_count > 1:
        relative_error = torch.linalg.vector_norm(
            blocks - reconstruction
        ) / response_norm.clamp_min(torch.finfo(response.dtype).eps)
    eigenvalues = torch.linalg.eigvalsh(channel_metric).flip(0)
    return {
        # With one profile the fitted channel metric absorbs the complete
        # response, so a zero reconstruction error would be tautological.
        "relative_frobenius_error": (
            float(relative_error) if relative_error is not None else None
        ),
        "channel_metric_eigenvalues": eigenvalues.cpu().tolist(),
        "channel_metric": channel_metric.cpu().tolist(),
    }


def slab_2p5d_response_diagnostic(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    sample_indices: Sequence[int] | None = None,
    max_mode: int = 1,
    fit_shells: int = 2,
    relative_amplitude: float = 0.05,
    field_batch_size: int = 32,
    z_profiles: int = 3,
) -> dict[str, Any]:
    r"""Measure the channel/z-profile response of a finite-z 2.5D FNO."""
    if model.spatial_scheme != "2.5d":
        raise ValueError("slab_2p5d_response_diagnostic requires the 2.5D scheme")
    if model.long_range.lateral_interlacing != 1:
        raise ValueError(
            "the 2.5D diagnostic requires lateral_interlacing=1 because an "
            "interlaced mesh has no unique deposited field"
        )
    if z_profiles not in {1, 2, 3}:
        raise ValueError("z_profiles must be one, two, or three")
    selected_indices = _validate_common(
        samples,
        sample_indices,
        max_mode=max_mode,
        fit_shells=fit_shells,
        relative_amplitude=relative_amplitude,
        field_batch_size=field_batch_size,
    )
    grid_shape = tuple(int(value) for value in model.long_range.assignment.grid_shape)
    if len(grid_shape) != 3:
        raise RuntimeError("the 2.5D mesh must expose (nz,nx,ny)")
    n_z, n_x, n_y = grid_shape
    if 2 * max_mode >= min(n_x, n_y):
        raise ValueError("max_mode must remain below the lateral mesh Nyquist limit")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    profiles, profile_names = slab_z_profiles(
        n_z, z_profiles, device=device, dtype=dtype
    )
    modes = unique_integer_modes_2d(max_mode)
    z_extent = float(model.long_range.assignment.z_extent)
    was_training = model.training
    per_sample: list[dict[str, Any]] = []
    low_k_points: list[tuple[float, float]] = []
    full_range_points: list[tuple[float, float]] = []
    low_k_template_errors: list[float] = []
    model.eval()
    try:
        with torch.no_grad():
            for sample_index in selected_indices:
                graph, _, _ = collate_samples([samples[sample_index]], device, dtype)
                output = model(
                    graph, training=False, compute_force=False, return_fields=True
                )
                density = output["density"][0]
                if density.ndim != 4:
                    raise RuntimeError(
                        "the 2.5D path must return one unbatched density"
                    )
                cell = graph["cell"].reshape(-1, 3, 3)[0]
                area = torch.linalg.vector_norm(torch.linalg.cross(cell[0], cell[1]))
                effective_volume = area * z_extent
                density_rms = density.square().mean().sqrt().clamp_min(
                    torch.finfo(dtype).eps
                )
                amplitude = density_rms * relative_amplitude
                mode_reports: list[dict[str, Any]] = []
                channels = density.shape[0]
                for mode_xy in modes:
                    planar_mode = unit_rms_cosine_mode_2d(
                        (n_x, n_y), mode_xy, device=device, dtype=dtype
                    )
                    basis = _slab_probe_basis(density, planar_mode, profiles)
                    response = quadratic_basis_response(
                        density,
                        basis,
                        amplitude,
                        lambda fields: _field_energies(
                            model, fields, cell, batch_size=field_batch_size
                        ),
                    ) / effective_volume
                    response_blocks = response.reshape(
                        channels, z_profiles, channels, z_profiles
                    )
                    monopole_response = response_blocks[:, 0, :, 0]
                    monopole_eigenvalues = torch.linalg.eigvalsh(
                        monopole_response
                    ).flip(0)
                    dominant = float(monopole_eigenvalues[0])
                    k_vector = planar_wavevector(cell, mode_xy)
                    k_norm = float(torch.linalg.vector_norm(k_vector))
                    template = slab_coulomb_profile_matrix(
                        profiles, k_norm, z_extent
                    )
                    template_fit = _slab_template_fit(response, template, channels)
                    mode_reports.append(
                        {
                            "mode_xy": list(mode_xy),
                            "k_cartesian_inverse_angstrom": [
                                float(component) for component in k_vector
                            ],
                            "k_parallel_inverse_angstrom": k_norm,
                            "k_inverse_angstrom": k_norm,
                            "monopole_channel_eigenvalues_per_effective_volume": (
                                monopole_eigenvalues.cpu().tolist()
                            ),
                            (
                                "dominant_positive_monopole_eigenvalue_"
                                "per_effective_volume"
                            ): (
                                dominant if dominant > 0.0 else None
                            ),
                            "coulomb_z_profile_template": template.cpu().tolist(),
                            "coulomb_template_fit": template_fit,
                        }
                    )
                shell_count = _assign_physical_shell_ranks(mode_reports)
                sample_low_k_points = _shell_average_points(
                    mode_reports,
                    "dominant_positive_monopole_eigenvalue_per_effective_volume",
                    maximum_rank=fit_shells,
                )
                sample_full_range_points = _shell_average_points(
                    mode_reports,
                    "dominant_positive_monopole_eigenvalue_per_effective_volume",
                    maximum_rank=None,
                )
                low_k_points.extend(sample_low_k_points)
                full_range_points.extend(sample_full_range_points)
                sample_template_errors = [
                    float(report["coulomb_template_fit"]["relative_frobenius_error"])
                    for report in mode_reports
                    if int(report["physical_shell_rank"]) <= fit_shells
                    and report["coulomb_template_fit"]["relative_frobenius_error"]
                    is not None
                ]
                low_k_template_errors.extend(sample_template_errors)
                per_sample.append(
                    {
                        "sample_index": sample_index,
                        "in_plane_area_angstrom2": float(area),
                        "z_extent_angstrom": z_extent,
                        "density_rms_latent_units_per_angstrom3": float(density_rms),
                        "probe_amplitude_latent_units_per_angstrom3": float(amplitude),
                        "physical_shells": shell_count,
                        "low_k_monopole_response_fit": fit_reference_power_response(
                            sample_low_k_points, 1.0
                        ),
                        "full_probed_range_monopole_response_fit": (
                            fit_reference_power_response(
                                sample_full_range_points, 1.0
                            )
                        ),
                        "mean_low_k_coulomb_template_relative_error": (
                            sum(sample_template_errors) / len(sample_template_errors)
                            if sample_template_errors
                            else None
                        ),
                        "modes": mode_reports,
                    }
                )
    finally:
        model.train(was_training)

    return {
        "diagnostic_kind": "slab_2p5d",
        "spatial_scheme": "2.5d",
        "samples": len(per_sample),
        "sample_indices": selected_indices,
        "grid_shape_zxy": list(grid_shape),
        "z_profile_names": profile_names,
        "max_mode": max_mode,
        "low_k_fit_shells": fit_shells,
        "relative_amplitude": relative_amplitude,
        "field_batch_size": field_batch_size,
        "probes_per_mode": int(model.source_head.channels * z_profiles),
        "field_evaluations_per_mode": int(
            2 * (model.source_head.channels * z_profiles) ** 2 + 1
        ),
        "description": (
            "Validation-only channel/z-profile curvature under neutral planar "
            "Fourier probes. The monopole-like branch is compared with 1/k_parallel; "
            "the complete response is projected onto the open-boundary kernel "
            "2*pi*exp(-k|z-z'|)/k using a best-fit latent channel metric."
        ),
        "low_k_monopole_response_fit": fit_reference_power_response(low_k_points, 1.0),
        "full_probed_range_monopole_response_fit": fit_reference_power_response(
            full_range_points, 1.0
        ),
        "mean_low_k_coulomb_template_relative_error": (
            sum(low_k_template_errors) / len(low_k_template_errors)
            if low_k_template_errors
            else None
        ),
        "per_sample_response": per_sample,
    }


def low_k_response_diagnostic(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    sample_indices: Sequence[int] | None = None,
    max_mode: int = 1,
    fit_shells: int = 3,
    relative_amplitude: float = 0.05,
    field_batch_size: int = 32,
    z_profiles: int = 3,
) -> dict[str, Any]:
    """Dispatch to the geometry-appropriate validation-only diagnostic."""
    common = {
        "sample_indices": sample_indices,
        "max_mode": max_mode,
        "fit_shells": fit_shells,
        "relative_amplitude": relative_amplitude,
        "field_batch_size": field_batch_size,
    }
    if model.spatial_scheme == "3d":
        return periodic_3d_response_diagnostic(model, samples, **common)
    if model.spatial_scheme == "2d":
        return planar_2d_response_diagnostic(model, samples, **common)
    if model.spatial_scheme == "2.5d":
        return slab_2p5d_response_diagnostic(
            model, samples, z_profiles=z_profiles, **common
        )
    raise ValueError("unsupported spatial scheme for the low-k diagnostic")


def _response_fit(report: dict[str, Any]) -> dict[str, float] | None:
    kind = report["diagnostic_kind"]
    if kind == "periodic_3d":
        return report["low_k_dominant_eigenvalue_fit"]
    if kind == "planar_2d":
        return report["low_k_planar_response_fit"]
    if kind == "slab_2p5d":
        return report["low_k_monopole_response_fit"]
    raise ValueError(f"unsupported diagnostic kind {kind!r}")


def _mode_key_and_response(
    diagnostic_kind: str,
    sample_index: int,
    mode: dict[str, Any],
) -> tuple[tuple[int, ...], float]:
    if diagnostic_kind == "periodic_3d":
        mode_key = tuple(int(value) for value in mode["mode_zxy"])
        response = float(mode["eigenvalues_per_volume"][0])
    elif diagnostic_kind == "planar_2d":
        mode_key = tuple(int(value) for value in mode["mode_xy"])
        response = float(mode["channel_eigenvalues_per_area"][0])
    elif diagnostic_kind == "slab_2p5d":
        mode_key = tuple(int(value) for value in mode["mode_xy"])
        response = float(
            mode["monopole_channel_eigenvalues_per_effective_volume"][0]
        )
    else:
        raise ValueError(f"unsupported diagnostic kind {diagnostic_kind!r}")
    return (sample_index, *mode_key), response


def summarize_amplitude_convergence(
    reports: Sequence[dict[str, Any]],
    *,
    relative_span_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Summarize finite-amplitude stability of matched spectral curvatures."""
    if len(reports) < 2:
        raise ValueError("at least two amplitude reports are required")
    if relative_span_tolerance <= 0.0:
        raise ValueError("relative_span_tolerance must be positive")
    diagnostic_kind = reports[0]["diagnostic_kind"]
    spatial_scheme = reports[0]["spatial_scheme"]
    sample_indices = reports[0]["sample_indices"]
    max_mode = reports[0]["max_mode"]
    for report in reports[1:]:
        if (
            report["diagnostic_kind"] != diagnostic_kind
            or report["spatial_scheme"] != spatial_scheme
            or report["sample_indices"] != sample_indices
            or report["max_mode"] != max_mode
        ):
            raise ValueError(
                "amplitude reports must probe identical geometry and modes"
            )

    response_maps: list[dict[tuple[int, ...], float]] = []
    for report in reports:
        response_map: dict[tuple[int, ...], float] = {}
        for sample in report["per_sample_response"]:
            sample_index = int(sample["sample_index"])
            for mode in sample["modes"]:
                key, response = _mode_key_and_response(
                    diagnostic_kind, sample_index, mode
                )
                response_map[key] = response
        response_maps.append(response_map)

    union_keys = set().union(*(set(mapping) for mapping in response_maps))
    common_keys = set.intersection(*(set(mapping) for mapping in response_maps))
    relative_spans: list[float] = []
    sign_stable = 0
    mode_summaries: list[dict[str, Any]] = []
    for key in sorted(common_keys):
        values = [mapping[key] for mapping in response_maps]
        maximum_absolute = max(abs(value) for value in values)
        span = max(values) - min(values)
        relative_span = span / maximum_absolute if maximum_absolute > 0.0 else 0.0
        signs = {0 if value == 0.0 else (1 if value > 0.0 else -1) for value in values}
        stable_sign = len(signs - {0}) <= 1
        sign_stable += int(stable_sign)
        relative_spans.append(relative_span)
        mode_summaries.append(
            {
                "sample_and_mode": list(key),
                "leading_curvatures": values,
                "relative_span": relative_span,
                "sign_stable": stable_sign,
            }
        )

    sorted_spans = sorted(relative_spans)
    median_span = None
    if sorted_spans:
        middle = len(sorted_spans) // 2
        median_span = (
            sorted_spans[middle]
            if len(sorted_spans) % 2
            else 0.5 * (sorted_spans[middle - 1] + sorted_spans[middle])
        )
    fits = [_response_fit(report) for report in reports]
    exponents = [
        float(fit["free_power_exponent_p"]) for fit in fits if fit is not None
    ]
    reference_r2 = [
        float(fit["reference_power_log_r2"]) for fit in fits if fit is not None
    ]
    template_errors = [
        report.get("mean_low_k_coulomb_template_relative_error")
        for report in reports
        if report.get("mean_low_k_coulomb_template_relative_error") is not None
    ]
    all_modes_within_tolerance = bool(relative_spans) and all(
        span <= relative_span_tolerance for span in relative_spans
    )
    return {
        "diagnostic_kind": diagnostic_kind,
        "spatial_scheme": spatial_scheme,
        "matched_modes": len(common_keys),
        "union_modes": len(union_keys),
        "all_modes_present_at_every_amplitude": len(common_keys) == len(union_keys),
        "sign_stable_modes": sign_stable,
        "relative_span_tolerance": relative_span_tolerance,
        "median_mode_relative_span": median_span,
        "maximum_mode_relative_span": (
            max(relative_spans) if relative_spans else None
        ),
        "fraction_modes_within_tolerance": (
            sum(span <= relative_span_tolerance for span in relative_spans)
            / len(relative_spans)
            if relative_spans
            else None
        ),
        "curvature_stable_within_tolerance": (
            all_modes_within_tolerance
            and sign_stable == len(common_keys)
            and len(common_keys) == len(union_keys)
        ),
        "free_power_exponents": exponents,
        "free_power_exponent_range": (
            max(exponents) - min(exponents) if len(exponents) >= 2 else None
        ),
        "reference_power_log_r2": reference_r2,
        "slab_template_relative_errors": template_errors,
        "per_mode": mode_summaries,
    }


def amplitude_convergence_diagnostic(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    relative_amplitudes: Sequence[float] = (0.025, 0.05, 0.1),
    relative_span_tolerance: float = 0.05,
    sample_indices: Sequence[int] | None = None,
    max_mode: int = 1,
    fit_shells: int = 3,
    field_batch_size: int = 32,
    z_profiles: int = 3,
) -> dict[str, Any]:
    """Repeat a spectral diagnostic to test finite-amplitude convergence."""
    amplitudes = sorted(float(value) for value in relative_amplitudes)
    if len(amplitudes) < 2 or any(value <= 0.0 for value in amplitudes):
        raise ValueError("relative_amplitudes must contain at least two positives")
    if len(set(amplitudes)) != len(amplitudes):
        raise ValueError("relative_amplitudes must be distinct")
    reports = [
        low_k_response_diagnostic(
            model,
            samples,
            sample_indices=sample_indices,
            max_mode=max_mode,
            fit_shells=fit_shells,
            relative_amplitude=amplitude,
            field_batch_size=field_batch_size,
            z_profiles=z_profiles,
        )
        for amplitude in amplitudes
    ]
    estimated_field_evaluations = sum(
        int(report["field_evaluations_per_mode"])
        * sum(
            len(sample["modes"])
            for sample in report["per_sample_response"]
        )
        for report in reports
    )
    return {
        "diagnostic_kind": "amplitude_convergence",
        "spatial_scheme": model.spatial_scheme,
        "relative_amplitudes": amplitudes,
        "estimated_field_evaluations": estimated_field_evaluations,
        "description": (
            "Post-training finite-amplitude check. A quadratic local response "
            "should give amplitude-independent curvature after division by the "
            "squared perturbation amplitude. This diagnostic is not a loss."
        ),
        "summary": summarize_amplitude_convergence(
            reports, relative_span_tolerance=relative_span_tolerance
        ),
        "runs": reports,
    }
