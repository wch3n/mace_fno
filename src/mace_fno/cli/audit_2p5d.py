"""Audit a trained 2.5D MACE-FNO residual checkpoint.

The training RMSE alone does not test whether the learned correction respects
the symmetries of an atomistic energy.  This script checks held-out errors,
per-graph source neutrality, the acoustic sum rule, rigid translations, exact
in-plane one-grid translations, and residual-force finite differences.
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
    infer_checkpoint_z_mixing,
    load_mace_fno_model,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--translation", type=float, default=0.1)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--fd-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on violated exact invariants or inconsistent force derivatives",
    )
    return parser.parse_args()


def rms(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array * array)))


def rotate_c4(vectors: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Rotate Cartesian vectors by +90 degrees in an orthogonal square plane."""
    first = cell[0] / torch.linalg.vector_norm(cell[0])
    second = cell[1] / torch.linalg.vector_norm(cell[1])
    normal = torch.linalg.cross(first, second)
    return (
        -torch.einsum("...i,i->...", vectors, second)[..., None] * first
        + torch.einsum("...i,i->...", vectors, first)[..., None] * second
        + torch.einsum("...i,i->...", vectors, normal)[..., None] * normal
    )


def reflect_first_axis(vectors: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Reflect Cartesian vectors across the plane normal to the first cell axis."""
    first = cell[0] / torch.linalg.vector_norm(cell[0])
    return (
        vectors - 2.0 * torch.einsum("...i,i->...", vectors, first)[..., None] * first
    )


def translated_energies(
    model: MACEFNOResidual,
    sample: dict[str, Any],
    directions: list[torch.Tensor],
    distance: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    graphs = []
    for direction in directions:
        for sign in (-1.0, 1.0):
            graph = clone_graph(sample["data"], device, dtype)
            graph["positions"] += sign * distance * direction
            graphs.append(graph)
    output = model(batch_graphs(graphs), training=False, compute_force=False)
    return (
        output["energy"].detach().cpu().numpy().reshape(len(directions), 2),
        output["residual_energy"].detach().cpu().numpy().reshape(len(directions), 2),
    )


def main() -> None:
    args = parse_arguments()
    if min(args.samples, args.fd_components) < 1:
        raise ValueError("--samples and --fd-components must be positive")
    if min(args.translation, args.fd_step) <= 0.0:
        raise ValueError("translation and finite-difference steps must be positive")

    device = choose_device(args.device)
    model, checkpoint = load_mace_fno_model(args.checkpoint, device=device)
    if model.spatial_scheme != "2.5d":
        raise ValueError("this audit requires a 2.5D residual checkpoint")
    dtype = next(model.parameters()).dtype
    cache_path = args.sample_cache or Path(checkpoint["test_cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    all_samples = cache["samples"]
    selection_generator = torch.Generator().manual_seed(args.seed)
    sample_indices = torch.randperm(len(all_samples), generator=selection_generator)[
        : min(args.samples, len(all_samples))
    ].tolist()
    samples = [all_samples[index] for index in sample_indices]

    energy_errors: list[float] = []
    force_errors: list[float] = []
    force_errors_by_axis: list[list[float]] = [[], [], []]
    source_sums: list[float] = []
    net_residual_forces: list[list[float]] = []
    translation_energy_changes: list[list[float]] = []
    translation_force_mismatch: list[list[float]] = []
    first_reference_residual: float | None = None
    first_reference_base: float | None = None
    first_reference_force: torch.Tensor | None = None
    first_reference_total_force: torch.Tensor | None = None

    for sample in samples:
        graph = clone_graph(sample["data"], device, dtype)
        output = model(
            graph,
            training=False,
            compute_force=True,
            compute_residual_force=True,
        )
        if first_reference_residual is None:
            first_reference_residual = output["residual_energy"].item()
            first_reference_base = output["base_energy"].item()
            first_reference_force = output["residual_forces"].detach().cpu()
            first_reference_total_force = output["forces"].detach().cpu()
        count = int(sample["num_atoms"])
        energy_errors.append(
            (output["energy"].item() - sample["energy"].item()) / count
        )
        force_error = output["forces"].detach().cpu() - sample["forces"]
        force_errors.extend(force_error.reshape(-1).tolist())
        for axis in range(3):
            force_errors_by_axis[axis].extend(force_error[:, axis].tolist())
        source_sums.append(output["sources"].sum(dim=0).abs().max().item())
        net_force = output["residual_forces"].sum(dim=0).detach().cpu()
        net_residual_forces.append(net_force.tolist())

        cell = graph["cell"].reshape(-1, 3, 3)[0]
        normal = torch.linalg.cross(cell[0], cell[1])
        directions = [
            cell[0] / torch.linalg.vector_norm(cell[0]),
            cell[1] / torch.linalg.vector_norm(cell[1]),
            normal / torch.linalg.vector_norm(normal),
        ]
        _, shifted_residual = translated_energies(
            model, sample, directions, args.translation, device, dtype
        )
        _, derivative_residual = translated_energies(
            model, sample, directions, args.fd_step, device, dtype
        )
        reference_residual = output["residual_energy"].item()
        changes = np.max(np.abs(shifted_residual - reference_residual), axis=1)
        finite_translation_force = -(
            derivative_residual[:, 1] - derivative_residual[:, 0]
        ) / (2.0 * args.fd_step)
        translation_energy_changes.append(changes.tolist())
        projected_net_force = np.asarray(
            [
                torch.dot(net_force, direction.detach().cpu()).item()
                for direction in directions
            ]
        )
        translation_force_mismatch.append(
            (finite_translation_force - projected_net_force).tolist()
        )

    # A complete lateral grid-cell translation must be exact even though a
    # fractional grid translation exposes the finite-mesh egg-box error.
    first_graph = clone_graph(samples[0]["data"], device, dtype)
    first_cell = first_graph["cell"].reshape(-1, 3, 3)[0]
    grid_directions = [first_cell[0], first_cell[1]]
    grid_distances = [
        1.0 / checkpoint["grid_shape"][0],
        1.0 / checkpoint["grid_shape"][1],
    ]
    assert first_reference_residual is not None
    reference_residual = first_reference_residual
    grid_translation_changes = []
    for vector, fraction in zip(grid_directions, grid_distances):
        graph = clone_graph(samples[0]["data"], device, dtype)
        graph["positions"] += fraction * vector
        shifted = model(graph, training=False, compute_force=False)[
            "residual_energy"
        ].item()
        grid_translation_changes.append(abs(shifted - reference_residual))

    generator = torch.Generator().manual_seed(args.seed)
    n_atoms = int(samples[0]["num_atoms"])
    flat_indices = torch.randperm(3 * n_atoms, generator=generator)[
        : args.fd_components
    ]
    fd_graphs = []
    fd_pairs = []
    for flat_index in flat_indices.tolist():
        atom, axis = divmod(flat_index, 3)
        pair = []
        for sign in (-1.0, 1.0):
            graph = clone_graph(samples[0]["data"], device, dtype)
            graph["positions"][atom, axis] += sign * args.fd_step
            pair.append(len(fd_graphs))
            fd_graphs.append(graph)
        fd_pairs.append((atom, axis, pair))
    fd_output = model(batch_graphs(fd_graphs), training=False, compute_force=False)
    fd_energies = fd_output["residual_energy"].detach().cpu()
    assert first_reference_force is not None
    reference_force = first_reference_force
    fd_errors = []
    for atom, axis, (minus_index, plus_index) in fd_pairs:
        finite_force = -(fd_energies[plus_index] - fd_energies[minus_index]) / (
            2.0 * args.fd_step
        )
        fd_errors.append(abs(finite_force.item() - reference_force[atom, axis].item()))

    c4_report = None
    reflection_report = None
    c4_graph = clone_graph(samples[0]["data"], device, dtype)
    c4_cell = c4_graph["cell"].reshape(-1, 3, 3)[0]
    first_length = torch.linalg.vector_norm(c4_cell[0])
    second_length = torch.linalg.vector_norm(c4_cell[1])
    orthogonality = torch.dot(c4_cell[0], c4_cell[1]).abs()
    if bool(
        ((first_length - second_length).abs() <= 1.0e-6 * first_length)
        & (orthogonality <= 1.0e-6 * first_length * second_length)
    ):
        centre = 0.5 * (c4_cell[0] + c4_cell[1])
        c4_graph["positions"] = centre + rotate_c4(
            c4_graph["positions"] - centre, c4_cell
        )
        c4_graph["shifts"] = rotate_c4(c4_graph["shifts"], c4_cell)
        c4_output = model(
            c4_graph,
            training=False,
            compute_force=True,
            compute_residual_force=True,
        )
        assert first_reference_base is not None
        assert first_reference_total_force is not None
        expected_residual_force = rotate_c4(first_reference_force, c4_cell.cpu())
        expected_total_force = rotate_c4(first_reference_total_force, c4_cell.cpu())
        c4_report = {
            "base_energy_change_mev": 1000.0
            * abs(c4_output["base_energy"].item() - first_reference_base),
            "residual_energy_change_mev": 1000.0
            * abs(c4_output["residual_energy"].item() - first_reference_residual),
            "residual_force_equivariance_rmse_mev_per_angstrom": 1000.0
            * rms(
                (c4_output["residual_forces"].detach().cpu() - expected_residual_force)
                .reshape(-1)
                .tolist()
            ),
            "total_force_equivariance_rmse_mev_per_angstrom": 1000.0
            * rms(
                (c4_output["forces"].detach().cpu() - expected_total_force)
                .reshape(-1)
                .tolist()
            ),
        }
        reflection_graph = clone_graph(samples[0]["data"], device, dtype)
        reflection_graph["positions"] = centre + reflect_first_axis(
            reflection_graph["positions"] - centre, c4_cell
        )
        reflection_graph["shifts"] = reflect_first_axis(
            reflection_graph["shifts"], c4_cell
        )
        reflection_output = model(
            reflection_graph,
            training=False,
            compute_force=True,
            compute_residual_force=True,
        )
        expected_reflected_force = reflect_first_axis(
            first_reference_force, c4_cell.cpu()
        )
        reflection_report = {
            "base_energy_change_mev": 1000.0
            * abs(reflection_output["base_energy"].item() - first_reference_base),
            "residual_energy_change_mev": 1000.0
            * abs(
                reflection_output["residual_energy"].item() - first_reference_residual
            ),
            "residual_force_equivariance_rmse_mev_per_angstrom": 1000.0
            * rms(
                (
                    reflection_output["residual_forces"].detach().cpu()
                    - expected_reflected_force
                )
                .reshape(-1)
                .tolist()
            ),
        }

    net = np.asarray(net_residual_forces)
    changes = np.asarray(translation_energy_changes)
    translation_mismatch = np.asarray(translation_force_mismatch)
    report = {
        "checkpoint": str(args.checkpoint),
        "sample_cache": str(cache_path),
        "samples": len(samples),
        "sample_indices": sample_indices,
        "z_center": checkpoint["z_center"],
        "z_mixing": infer_checkpoint_z_mixing(checkpoint),
        "lateral_interlacing": checkpoint.get("lateral_interlacing", 1),
        "planar_symmetry": checkpoint.get("planar_symmetry", "none"),
        "subset_energy_rmse_mev_per_atom": 1000.0 * rms(energy_errors),
        "subset_force_rmse_mev_per_angstrom": 1000.0 * rms(force_errors),
        "subset_force_rmse_by_axis_mev_per_angstrom": [
            1000.0 * rms(axis_errors) for axis_errors in force_errors_by_axis
        ],
        "max_source_sum": max(source_sums),
        "net_residual_force_rms_mev_per_angstrom": (
            1000.0 * np.sqrt(np.mean(net * net, axis=0))
        ).tolist(),
        "rigid_translation_distance_angstrom": args.translation,
        "rigid_translation_energy_max_mev": (1000.0 * changes.max(axis=0)).tolist(),
        "translation_derivative_mismatch_max_mev_per_angstrom": (
            1000.0 * np.abs(translation_mismatch).max(axis=0)
        ).tolist(),
        "one_grid_translation_energy_max_mev": 1000.0 * max(grid_translation_changes),
        "residual_force_fd_rms_mev_per_angstrom": 1000.0 * rms(fd_errors),
        "residual_force_fd_max_mev_per_angstrom": 1000.0 * max(fd_errors),
        "c4_rotation": c4_report,
        "in_plane_reflection": reflection_report,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.strict:
        failures = []
        if report["max_source_sum"] > 1.0e-10:
            failures.append("latent-source neutrality")
        if report["one_grid_translation_energy_max_mev"] > 1.0e-7:
            failures.append("one-grid lateral translation")
        if report["residual_force_fd_max_mev_per_angstrom"] > 0.05:
            failures.append("residual force finite difference")
        if checkpoint["z_center"] == "mean":
            if report["rigid_translation_energy_max_mev"][2] > 1.0e-7:
                failures.append("rigid normal translation")
            if report["net_residual_force_rms_mev_per_angstrom"][2] > 1.0e-5:
                failures.append("normal acoustic sum rule")
        if checkpoint.get("planar_symmetry", "none") == "c4":
            if c4_report is None:
                failures.append("C4 audit geometry")
            elif (
                c4_report["residual_energy_change_mev"] > 1.0e-7
                or c4_report["residual_force_equivariance_rmse_mev_per_angstrom"]
                > 1.0e-5
            ):
                failures.append("C4 residual equivariance")
        if checkpoint.get("planar_symmetry", "none") == "d4":
            if c4_report is None or reflection_report is None:
                failures.append("D4 audit geometry")
            elif (
                c4_report["residual_energy_change_mev"] > 1.0e-7
                or c4_report["residual_force_equivariance_rmse_mev_per_angstrom"]
                > 1.0e-5
                or reflection_report["residual_energy_change_mev"] > 1.0e-7
                or reflection_report[
                    "residual_force_equivariance_rmse_mev_per_angstrom"
                ]
                > 1.0e-5
            ):
                failures.append("D4 residual equivariance")
        if failures:
            raise RuntimeError("failed audit checks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
