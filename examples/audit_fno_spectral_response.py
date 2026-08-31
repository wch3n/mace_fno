"""Measure the effective low-wavevector response of a trained 3D FNO.

This is a post-training diagnostic.  It perturbs the deposited latent density
by neutral real Fourier modes and evaluates the curvature of the learned
residual energy.  The latent channels have no fixed physical basis, so the
reported spectra are eigenvalues of the channel-space response matrix rather
than individual-channel ``kernels``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from audit_les_water_3d import build_model, choose_device
from mace_fno.spectral_response import (
    fit_power_law_response,
    quadratic_mode_response,
    unique_integer_modes,
    unit_rms_cosine_mode,
    wavevector_norm,
)
from mace_fno.training import clone_graph


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-mode", type=int, default=2)
    parser.add_argument(
        "--fit-shells",
        type=int,
        default=3,
        help="number of smallest reciprocal-radius shells used for the low-k fit",
    )
    parser.add_argument("--relative-amplitude", type=float, default=0.05)
    parser.add_argument("--field-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _field_energies(
    model: Any,
    fields: torch.Tensor,
    cell: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("field_batch_size must be positive")
    values: list[torch.Tensor] = []
    for start in range(0, fields.shape[0], batch_size):
        field_batch = fields[start : start + batch_size]
        cells = cell.unsqueeze(0).expand(field_batch.shape[0], -1, -1)
        values.append(model.long_range.energy_from_density(field_batch, cells))
    return torch.cat(values, dim=0)


def main() -> None:
    args = parse_arguments()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if min(args.max_mode, args.fit_shells) < 1:
        raise ValueError("max_mode and fit_shells must be positive")
    if args.relative_amplitude <= 0.0:
        raise ValueError("relative_amplitude must be positive")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("spatial_scheme") != "3d":
        raise ValueError("the spectral-response audit requires a 3D checkpoint")
    if int(checkpoint.get("volume_interlacing", 1)) != 1:
        raise ValueError(
            "the spectral-response audit currently requires volume_interlacing=1"
        )
    model = build_model(checkpoint, device)
    model.eval()
    dtype = next(model.parameters()).dtype
    cache_path = args.sample_cache or Path(checkpoint["test_cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    all_samples = cache["samples"]
    selection_generator = torch.Generator().manual_seed(args.seed)
    sample_indices = torch.randperm(len(all_samples), generator=selection_generator)[
        : min(args.samples, len(all_samples))
    ].tolist()

    grid_shape = tuple(int(value) for value in model.long_range.assignment.grid_shape)
    if len(grid_shape) != 3:
        raise RuntimeError("the 3D particle-mesh assignment must expose three grid dimensions")
    if 2 * args.max_mode >= min(grid_shape):
        raise ValueError("max_mode must remain below the mesh Nyquist limit")
    modes = unique_integer_modes(args.max_mode)
    all_sample_reports: list[dict[str, Any]] = []
    fit_points_by_shell: dict[int, list[tuple[float, float]]] = defaultdict(list)

    with torch.no_grad():
        for sample_index in sample_indices:
            sample = all_samples[sample_index]
            graph = clone_graph(sample["data"], device, dtype)
            output = model(graph, training=False, compute_force=False, return_fields=True)
            density = output["density"][0]
            if density.ndim != 4:
                raise RuntimeError("the 3D particle-mesh path must return one unbatched density")
            cell = graph["cell"].reshape(-1, 3, 3)[0]
            volume = torch.linalg.det(cell).abs()
            density_rms = density.square().mean().sqrt().clamp_min(torch.finfo(dtype).eps)
            amplitude = density_rms * args.relative_amplitude
            shell_matrices: dict[int, list[torch.Tensor]] = defaultdict(list)
            shell_k: dict[int, list[float]] = defaultdict(list)

            for mode_zxy in modes:
                perturbation = unit_rms_cosine_mode(
                    grid_shape, mode_zxy, device=device, dtype=dtype
                )
                response = quadratic_mode_response(
                    density,
                    perturbation,
                    amplitude,
                    lambda fields: _field_energies(
                        model, fields, cell, batch_size=args.field_batch_size
                    ),
                )
                squared_norm = sum(component * component for component in mode_zxy)
                shell_matrices[squared_norm].append(response / volume)
                shell_k[squared_norm].append(float(wavevector_norm(cell, mode_zxy)))

            shells: list[dict[str, Any]] = []
            for squared_norm in sorted(shell_matrices):
                response = torch.stack(shell_matrices[squared_norm]).mean(dim=0)
                eigenvalues = torch.linalg.eigvalsh(response).flip(0)
                dominant = float(eigenvalues[0])
                k_values = shell_k[squared_norm]
                k_value = sum(k_values) / len(k_values)
                shell = {
                    "integer_squared_norm": squared_norm,
                    "mode_count": len(shell_matrices[squared_norm]),
                    "k_inverse_angstrom": k_value,
                    "response_matrix_per_volume": response.detach().cpu().tolist(),
                    "eigenvalues_per_volume": eigenvalues.detach().cpu().tolist(),
                    "dominant_positive_eigenvalue_per_volume": (
                        dominant if dominant > 0.0 else None
                    ),
                }
                shells.append(shell)
                if dominant > 0.0:
                    fit_points_by_shell[squared_norm].append((k_value, dominant))
            all_sample_reports.append(
                {
                    "sample_index": sample_index,
                    "cell_volume_angstrom3": float(volume),
                    "density_rms_latent_units_per_angstrom3": float(density_rms),
                    "probe_amplitude_latent_units_per_angstrom3": float(amplitude),
                    "shells": shells,
                }
            )

    shell_order = sorted(fit_points_by_shell)
    low_k_shell_order = shell_order[: args.fit_shells]
    low_k_points = [
        point for squared_norm in low_k_shell_order for point in fit_points_by_shell[squared_norm]
    ]
    full_range_points = [
        point for squared_norm in shell_order for point in fit_points_by_shell[squared_norm]
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "sample_cache": str(cache_path),
        "samples": len(all_sample_reports),
        "sample_indices": sample_indices,
        "grid_shape_zxy": list(grid_shape),
        "max_mode": args.max_mode,
        "low_k_fit_shells": args.fit_shells,
        "low_k_integer_squared_norms": low_k_shell_order,
        "relative_amplitude": args.relative_amplitude,
        "field_batch_size": args.field_batch_size,
        "description": (
            "Local quadratic curvature of the residual FNO energy with respect to "
            "neutral unit-RMS cosine perturbations of its deposited latent fields. "
            "The latent-channel basis is arbitrary; inspect eigenvalues, not a raw channel."
        ),
        "low_k_dominant_eigenvalue_fit": fit_power_law_response(low_k_points),
        "full_probed_range_dominant_eigenvalue_fit": fit_power_law_response(
            full_range_points
        ),
        "per_sample_shell_response": all_sample_reports,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
