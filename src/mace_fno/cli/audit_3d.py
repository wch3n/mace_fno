"""Audit a trained fully periodic 3D MACE-FNO residual checkpoint.

The audit separates invariants that the current implementation promises from
physical symmetries that a generic Cartesian-grid FNO does not enforce. It
checks held-out errors, latent-source neutrality, force additivity, the
acoustic sum rule, arbitrary rigid translations, exact grid and lattice
translations, residual-force finite differences, and cubic signed-axis
transformations of representative held-out periodic configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_fno import MACEFNOResidual
from mace_fno.training import (
    batch_graphs,
    choose_device,
    clone_graph,
    load_mace_fno_model,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--translation", type=float, default=0.1)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--fd-components", type=int, default=12)
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if a promised exact invariant or energy-force consistency "
            "check is violated. Cubic symmetry is promoted from a diagnostic "
            "to a strict check for native EqGINO checkpoints."
        ),
    )
    return parser.parse_args()


def rms(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array * array)))


def cell_directions(cell: torch.Tensor) -> list[torch.Tensor]:
    return [vector / torch.linalg.vector_norm(vector) for vector in cell]


def translated_residual_energies(
    model: MACEFNOResidual,
    sample: dict[str, Any],
    directions: list[torch.Tensor],
    distance: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    graphs = []
    for direction in directions:
        for sign in (-1.0, 1.0):
            graph = clone_graph(sample["data"], device, dtype)
            graph["positions"] += sign * distance * direction
            graphs.append(graph)
    output = model(batch_graphs(graphs), training=False, compute_force=False)
    return output["residual_energy"].detach().cpu().numpy().reshape(-1, 2)


def signed_axis_transform(
    vectors: torch.Tensor,
    cell: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """Apply a signed permutation in the orthonormal cell-axis basis."""
    basis = cell / torch.linalg.vector_norm(cell, dim=1, keepdim=True)
    rotation = basis.T @ transform.to(cell) @ basis
    return torch.einsum("ij,...j->...i", rotation, vectors)


def cubic_transformations(dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    values = {
        "c4_x": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        "c4_y": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        "c4_z": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        "cycle_xyz": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        "swap_xy": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        "inversion": ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
    }
    return {
        name: torch.tensor(matrix, dtype=dtype, device=device)
        for name, matrix in values.items()
    }


def is_cubic(cell: torch.Tensor, tolerance: float = 1.0e-6) -> bool:
    lengths = torch.linalg.vector_norm(cell, dim=1)
    gram = cell @ cell.T
    off_diagonal = gram - torch.diag(torch.diagonal(gram))
    length_scale = lengths.max().clamp_min(1.0)
    return bool(
        (
            (lengths.max() - lengths.min() <= tolerance * length_scale)
            & (off_diagonal.abs().max() <= tolerance * length_scale.square())
        )
        .detach()
        .cpu()
    )


def main() -> None:
    args = parse_arguments()
    if min(args.samples, args.fd_components) < 1:
        raise ValueError("--samples and --fd-components must be positive")
    if min(args.translation, args.fd_step) <= 0.0:
        raise ValueError("translation and finite-difference steps must be positive")

    device = choose_device(args.device)
    model, checkpoint = load_mace_fno_model(args.checkpoint, device=device)
    if model.spatial_scheme != "3d":
        raise ValueError("this audit requires a fully periodic 3D checkpoint")
    dtype = next(model.parameters()).dtype
    cache_path = args.sample_cache or Path(checkpoint["test_cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    all_samples = cache["samples"]
    selection_generator = torch.Generator().manual_seed(args.seed)
    sample_indices = torch.randperm(len(all_samples), generator=selection_generator)[
        : min(args.samples, len(all_samples))
    ].tolist()
    # The strict cubic-equivariance audit must not disappear merely because the
    # first randomly selected configuration belongs to a noncubic cell class.
    cubic_index = next(
        (
            index
            for index, sample in enumerate(all_samples)
            if sample.get("benchmark_group") == "cubic"
        ),
        None,
    )
    if cubic_index is not None:
        sample_indices = [cubic_index] + [
            index for index in sample_indices if index != cubic_index
        ]
        sample_indices = sample_indices[: min(args.samples, len(all_samples))]
    samples = [all_samples[index] for index in sample_indices]

    corrected_energy_errors: list[float] = []
    frozen_energy_errors: list[float] = []
    corrected_force_errors: list[float] = []
    frozen_force_errors: list[float] = []
    corrected_force_errors_by_axis: list[list[float]] = [[], [], []]
    frozen_force_errors_by_axis: list[list[float]] = [[], [], []]
    predicted_residual_energies_per_atom: list[float] = []
    predicted_residual_forces: list[float] = []
    source_sums: list[float] = []
    net_residual_forces: list[list[float]] = []
    force_additivity_errors: list[float] = []
    translation_energy_changes: list[list[float]] = []
    translation_force_mismatch: list[list[float]] = []
    first_output: dict[str, Any] | None = None

    for sample in samples:
        graph = clone_graph(sample["data"], device, dtype)
        output = model(
            graph,
            training=False,
            compute_force=True,
            compute_residual_force=True,
            compute_base_force=True,
        )
        if first_output is None:
            first_output = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in output.items()
                if key
                in {
                    "energy",
                    "forces",
                    "base_energy",
                    "base_forces",
                    "residual_energy",
                    "residual_forces",
                }
            }
        count = int(sample["num_atoms"])
        corrected_energy_errors.append(
            (output["energy"].item() - sample["energy"].item()) / count
        )
        frozen_energy_errors.append(
            (output["base_energy"].item() - sample["energy"].item()) / count
        )
        predicted_residual_energies_per_atom.append(
            output["residual_energy"].item() / count
        )

        corrected_force_error = output["forces"].detach().cpu() - sample["forces"]
        frozen_force_error = output["base_forces"].detach().cpu() - sample["forces"]
        corrected_force_errors.extend(corrected_force_error.reshape(-1).tolist())
        frozen_force_errors.extend(frozen_force_error.reshape(-1).tolist())
        predicted_residual_forces.extend(
            output["residual_forces"].detach().cpu().reshape(-1).tolist()
        )
        for axis in range(3):
            corrected_force_errors_by_axis[axis].extend(
                corrected_force_error[:, axis].tolist()
            )
            frozen_force_errors_by_axis[axis].extend(
                frozen_force_error[:, axis].tolist()
            )
        source_sums.append(output["sources"].sum(dim=0).abs().max().item())
        net_force = output["residual_forces"].sum(dim=0).detach().cpu()
        net_residual_forces.append(net_force.tolist())
        force_additivity_errors.extend(
            (
                output["forces"]
                - output["base_forces"]
                - output["residual_forces"]
            )
            .detach()
            .cpu()
            .abs()
            .reshape(-1)
            .tolist()
        )

        cell = graph["cell"].reshape(-1, 3, 3)[0]
        directions = cell_directions(cell)
        shifted = translated_residual_energies(
            model, sample, directions, args.translation, device, dtype
        )
        derivative = translated_residual_energies(
            model, sample, directions, args.fd_step, device, dtype
        )
        reference_residual = output["residual_energy"].item()
        translation_energy_changes.append(
            np.max(np.abs(shifted - reference_residual), axis=1).tolist()
        )
        finite_translation_force = -(
            derivative[:, 1] - derivative[:, 0]
        ) / (2.0 * args.fd_step)
        projected_net_force = np.asarray(
            [
                torch.dot(net_force, direction.detach().cpu()).item()
                for direction in directions
            ]
        )
        translation_force_mismatch.append(
            (finite_translation_force - projected_net_force).tolist()
        )

    assert first_output is not None
    first_sample = samples[0]
    first_graph = clone_graph(first_sample["data"], device, dtype)
    first_cell = first_graph["cell"].reshape(-1, 3, 3)[0]
    directions = cell_directions(first_cell)
    reference_residual = first_output["residual_energy"].item()
    reference_residual_force = first_output["residual_forces"]

    natural_grid_shape = (
        int(checkpoint["grid_shape"][0]),
        int(checkpoint["grid_shape"][1]),
        int(checkpoint["z_grid_size"]),
    )
    one_grid_translation_changes = []
    full_lattice_translation_changes = []
    for axis, vector in enumerate(first_cell):
        grid_graph = clone_graph(first_sample["data"], device, dtype)
        grid_graph["positions"] += vector / natural_grid_shape[axis]
        grid_energy = model(grid_graph, training=False, compute_force=False)[
            "residual_energy"
        ].item()
        one_grid_translation_changes.append(abs(grid_energy - reference_residual))

        lattice_graph = clone_graph(first_sample["data"], device, dtype)
        lattice_graph["positions"] += vector
        lattice_energy = model(lattice_graph, training=False, compute_force=False)[
            "residual_energy"
        ].item()
        full_lattice_translation_changes.append(
            abs(lattice_energy - reference_residual)
        )

    translated_graphs = []
    for direction in directions:
        for sign in (-1.0, 1.0):
            graph = clone_graph(first_sample["data"], device, dtype)
            graph["positions"] += sign * args.translation * direction
            translated_graphs.append(graph)
    translated_output = model(
        batch_graphs(translated_graphs),
        training=False,
        compute_force=False,
        compute_residual_force=True,
    )
    translated_forces = translated_output["residual_forces"].detach().cpu().reshape(
        3, 2, int(first_sample["num_atoms"]), 3
    )
    continuous_force_equivariance_rms = [
        rms((translated_forces[axis] - reference_residual_force).numpy())
        for axis in range(3)
    ]
    continuous_force_equivariance_max = [
        float((translated_forces[axis] - reference_residual_force).abs().max())
        for axis in range(3)
    ]

    generator = torch.Generator().manual_seed(args.seed)
    n_atoms = int(first_sample["num_atoms"])
    flat_indices = torch.randperm(3 * n_atoms, generator=generator)[
        : min(args.fd_components, 3 * n_atoms)
    ]
    fd_graphs = []
    fd_pairs = []
    for flat_index in flat_indices.tolist():
        atom, axis = divmod(flat_index, 3)
        pair = []
        for sign in (-1.0, 1.0):
            graph = clone_graph(first_sample["data"], device, dtype)
            graph["positions"][atom, axis] += sign * args.fd_step
            pair.append(len(fd_graphs))
            fd_graphs.append(graph)
        fd_pairs.append((atom, axis, pair))
    fd_output = model(batch_graphs(fd_graphs), training=False, compute_force=False)
    fd_energies = fd_output["residual_energy"].detach().cpu()
    fd_errors = []
    for atom, axis, (minus_index, plus_index) in fd_pairs:
        finite_force = -(fd_energies[plus_index] - fd_energies[minus_index]) / (
            2.0 * args.fd_step
        )
        fd_errors.append(
            abs(finite_force.item() - reference_residual_force[atom, axis].item())
        )

    cubic_report = None
    if is_cubic(first_cell):
        centre = 0.5 * first_cell.sum(dim=0)
        transformations = cubic_transformations(dtype, device)
        transformed_graphs = []
        for transform in transformations.values():
            graph = clone_graph(first_sample["data"], device, dtype)
            graph["positions"] = centre + signed_axis_transform(
                graph["positions"] - centre, first_cell, transform
            )
            graph["shifts"] = signed_axis_transform(
                graph["shifts"], first_cell, transform
            )
            transformed_graphs.append(graph)
        transformed_output = model(
            batch_graphs(transformed_graphs),
            training=False,
            compute_force=True,
            compute_residual_force=True,
            compute_base_force=True,
        )
        transformed_base_forces = transformed_output["base_forces"].detach().cpu()
        transformed_residual_forces = (
            transformed_output["residual_forces"].detach().cpu()
        )
        transformed_total_forces = transformed_output["forces"].detach().cpu()
        cubic_report = {}
        for index, (name, transform) in enumerate(transformations.items()):
            atom_slice = slice(index * n_atoms, (index + 1) * n_atoms)
            expected_base_force = signed_axis_transform(
                first_output["base_forces"].to(device), first_cell, transform
            ).cpu()
            expected_residual_force = signed_axis_transform(
                reference_residual_force.to(device), first_cell, transform
            ).cpu()
            expected_total_force = signed_axis_transform(
                first_output["forces"].to(device), first_cell, transform
            ).cpu()
            cubic_report[name] = {
                "determinant": float(torch.linalg.det(transform).item()),
                "base_energy_change_mev": 1000.0
                * abs(
                    transformed_output["base_energy"][index].item()
                    - first_output["base_energy"].item()
                ),
                "residual_energy_change_mev": 1000.0
                * abs(
                    transformed_output["residual_energy"][index].item()
                    - reference_residual
                ),
                "total_energy_change_mev": 1000.0
                * abs(
                    transformed_output["energy"][index].item()
                    - first_output["energy"].item()
                ),
                "base_force_equivariance_rmse_mev_per_angstrom": 1000.0
                * rms(
                    (
                        transformed_base_forces[atom_slice] - expected_base_force
                    ).numpy()
                ),
                "residual_force_equivariance_rmse_mev_per_angstrom": 1000.0
                * rms(
                    (
                        transformed_residual_forces[atom_slice]
                        - expected_residual_force
                    ).numpy()
                ),
                "total_force_equivariance_rmse_mev_per_angstrom": 1000.0
                * rms(
                    (
                        transformed_total_forces[atom_slice] - expected_total_force
                    ).numpy()
                ),
            }

    net = np.asarray(net_residual_forces)
    translation_changes = np.asarray(translation_energy_changes)
    translation_mismatch = np.asarray(translation_force_mismatch)
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "spectral_symmetry": checkpoint.get("spectral_symmetry", "none"),
        "spectral_groups": checkpoint.get("spectral_groups", 1),
        "volume_interlacing": checkpoint.get("volume_interlacing", 1),
        "interlacing_training": checkpoint.get("interlacing_training", "full"),
        "sample_cache": str(cache_path),
        "samples": len(samples),
        "sample_indices": sample_indices,
        "grid_shape_zxy": [
            int(checkpoint["z_grid_size"]),
            int(checkpoint["grid_shape"][0]),
            int(checkpoint["grid_shape"][1]),
        ],
        "n_modes_zxy": [int(value) for value in checkpoint["n_modes"]],
        "subset_frozen_energy_rmse_mev_per_atom": 1000.0
        * rms(frozen_energy_errors),
        "subset_corrected_energy_rmse_mev_per_atom": 1000.0
        * rms(corrected_energy_errors),
        "subset_frozen_force_rmse_mev_per_angstrom": 1000.0
        * rms(frozen_force_errors),
        "subset_corrected_force_rmse_mev_per_angstrom": 1000.0
        * rms(corrected_force_errors),
        "subset_frozen_force_rmse_by_axis_mev_per_angstrom": [
            1000.0 * rms(axis_errors) for axis_errors in frozen_force_errors_by_axis
        ],
        "subset_corrected_force_rmse_by_axis_mev_per_angstrom": [
            1000.0 * rms(axis_errors)
            for axis_errors in corrected_force_errors_by_axis
        ],
        "predicted_residual_energy_rms_mev_per_atom": 1000.0
        * rms(predicted_residual_energies_per_atom),
        "predicted_residual_force_rms_mev_per_angstrom": 1000.0
        * rms(predicted_residual_forces),
        "max_source_sum": max(source_sums),
        "force_additivity_max_mev_per_angstrom": 1000.0
        * max(force_additivity_errors),
        "net_residual_force_rms_mev_per_angstrom": (
            1000.0 * np.sqrt(np.mean(net * net, axis=0))
        ).tolist(),
        "rigid_translation_distance_angstrom": args.translation,
        "rigid_translation_energy_max_mev_by_cell_axis": (
            1000.0 * translation_changes.max(axis=0)
        ).tolist(),
        "rigid_translation_residual_force_rmse_mev_per_angstrom_by_cell_axis": [
            1000.0 * value for value in continuous_force_equivariance_rms
        ],
        "rigid_translation_residual_force_max_mev_per_angstrom_by_cell_axis": [
            1000.0 * value for value in continuous_force_equivariance_max
        ],
        "translation_derivative_mismatch_max_mev_per_angstrom_by_cell_axis": (
            1000.0 * np.abs(translation_mismatch).max(axis=0)
        ).tolist(),
        "one_grid_translation_energy_change_mev_by_cell_axis": [
            1000.0 * value for value in one_grid_translation_changes
        ],
        "full_lattice_translation_energy_change_mev_by_cell_axis": [
            1000.0 * value for value in full_lattice_translation_changes
        ],
        "residual_force_fd_rms_mev_per_angstrom": 1000.0 * rms(fd_errors),
        "residual_force_fd_max_mev_per_angstrom": 1000.0 * max(fd_errors),
        "cubic_signed_axis_transformations": cubic_report,
    }

    exact_checks = {
        "latent_source_neutrality": {
            "observed": report["max_source_sum"],
            "threshold": 1.0e-10,
        },
        "force_additivity": {
            "observed": report["force_additivity_max_mev_per_angstrom"],
            "threshold": 1.0e-5,
        },
        "one_grid_translation": {
            "observed": max(
                report["one_grid_translation_energy_change_mev_by_cell_axis"]
            ),
            "threshold": 1.0e-7,
        },
        "full_lattice_translation": {
            "observed": max(
                report["full_lattice_translation_energy_change_mev_by_cell_axis"]
            ),
            "threshold": 1.0e-7,
        },
        "translation_energy_force_consistency": {
            "observed": max(
                report[
                    "translation_derivative_mismatch_max_mev_per_angstrom_by_cell_axis"
                ]
            ),
            "threshold": 0.05,
        },
        "residual_force_finite_difference": {
            "observed": report["residual_force_fd_max_mev_per_angstrom"],
            "threshold": 0.05,
        },
    }
    if cubic_report is not None:
        maximum_residual_energy_change = max(
            item["residual_energy_change_mev"] for item in cubic_report.values()
        )
        maximum_residual_force_rmse = max(
            item["residual_force_equivariance_rmse_mev_per_angstrom"]
            for item in cubic_report.values()
        )
        report["diagnostic_status"] = {
            "continuous_translation_exact_to_0.01_mev": bool(
                max(report["rigid_translation_energy_max_mev_by_cell_axis"])
                <= 0.01
            ),
            "cubic_residual_energy_exact_to_1e-5_mev": bool(
                maximum_residual_energy_change <= 1.0e-5
            ),
            "cubic_residual_force_equivariant_to_1e-4_mev_per_angstrom": bool(
                maximum_residual_force_rmse <= 1.0e-4
            ),
        }
        if checkpoint.get("spectral_symmetry", "none") in {
            "eqgino",
            "cubic_adaptive",
        }:
            float32 = checkpoint.get("dtype") == "float32"
            exact_checks["cubic_residual_energy_invariance"] = {
                "observed": maximum_residual_energy_change,
                "threshold": 2.0e-2 if float32 else 1.0e-5,
            }
            exact_checks["cubic_residual_force_covariance"] = {
                "observed": maximum_residual_force_rmse,
                "threshold": 1.0e-1 if float32 else 1.0e-4,
            }
    else:
        report["diagnostic_status"] = {
            "continuous_translation_exact_to_0.01_mev": bool(
                max(report["rigid_translation_energy_max_mev_by_cell_axis"])
                <= 0.01
            ),
            "cubic_geometry_available": False,
        }

    for check in exact_checks.values():
        check["passed"] = bool(check["observed"] <= check["threshold"])
    report["promised_exact_checks"] = exact_checks

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.strict:
        failures = [
            name for name, check in exact_checks.items() if not check["passed"]
        ]
        if failures:
            raise RuntimeError("failed promised invariant checks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
