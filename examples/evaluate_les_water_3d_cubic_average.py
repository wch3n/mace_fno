"""Evaluate inference-only cubic averaging of a trained water 3D FNO.

The frozen MACE backbone is evaluated once per structure. Its invariant scalar
descriptors generate one set of latent sources, while the particle-mesh/FNO
branch is averaged over either the 24 proper cubic rotations (O) or all 48
signed-axis operations (O_h). Differentiating the averaged scalar energy gives
the corresponding conservative residual forces, including derivatives of the
MACE-derived latent sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch

from mace_fno import (
    cubic_signed_permutation_matrices,
    is_cubic_cell,
    transform_in_cell_axis_basis,
)
from audit_les_water_3d import build_model, choose_device
from train_mace_residual import clone_graph


LABELS = ("raw", "o24", "oh48")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1151)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if O_h averaging is not invariant/equivariant or conservative",
    )
    return parser.parse_args()


def selected_group_energies(
    energies: torch.Tensor,
    transformations: torch.Tensor,
    labels: Sequence[str],
) -> dict[str, torch.Tensor]:
    determinants = torch.linalg.det(transformations)
    identity_error = (
        transformations
        - torch.eye(3, dtype=transformations.dtype, device=transformations.device)
    ).abs().amax(dim=(1, 2))
    identity_indices = torch.nonzero(identity_error == 0.0).reshape(-1)
    if identity_indices.numel() != 1:
        raise RuntimeError("the cubic group must contain identity exactly once")
    result = {}
    for label in labels:
        if label == "raw":
            result[label] = energies[identity_indices[0]]
        elif label == "o24":
            result[label] = energies[determinants > 0].mean()
        elif label == "oh48":
            result[label] = energies.mean()
        else:
            raise ValueError(f"unknown cubic-average label {label!r}")
    return result


def cubic_residual_predictions(
    model: torch.nn.Module,
    graph: dict[str, Any],
    transformations: torch.Tensor,
    *,
    labels: Sequence[str] = LABELS,
    compute_forces: bool = True,
) -> dict[str, Any]:
    """Evaluate raw/O/O_h residuals using one frozen-MACE descriptor pass."""
    if graph["positions"].shape[0] < 1:
        raise ValueError("at least one atom is required")
    batch = graph["batch"].to(torch.long)
    if int(batch.min().detach().cpu()) != 0 or int(batch.max().detach().cpu()) != 0:
        raise ValueError("cubic averaging currently accepts one graph at a time")
    positions = graph["positions"]
    positions.requires_grad_(True)
    cell = graph["cell"].reshape(-1, 3, 3)[0]
    if not is_cubic_cell(cell):
        raise ValueError("O/O_h averaging requires an orthogonal cubic cell")

    _, invariants, _ = model.backbone(graph)
    sources = model.source_head(invariants, batch)
    centre = 0.5 * cell.sum(dim=0)
    transformed = centre + transform_in_cell_axis_basis(
        positions - centre, cell, transformations
    )
    group_size, atom_count = transformed.shape[:2]
    group_positions = transformed.reshape(group_size * atom_count, 3)
    group_sources = sources.unsqueeze(0).expand(group_size, -1, -1).reshape(
        group_size * atom_count, -1
    )
    group_cells = cell.unsqueeze(0).expand(group_size, -1, -1)
    group_batch = torch.arange(
        group_size, dtype=torch.long, device=positions.device
    ).repeat_interleave(atom_count)
    orientation_energies = model.long_range(
        group_positions,
        group_sources,
        group_cells,
        batch=group_batch,
    )
    energies = selected_group_energies(orientation_energies, transformations, labels)

    forces: dict[str, torch.Tensor] = {}
    if compute_forces:
        for index, label in enumerate(labels):
            gradient = torch.autograd.grad(
                energies[label],
                positions,
                retain_graph=index + 1 < len(labels),
                create_graph=False,
            )[0]
            forces[label] = -gradient.detach().cpu()
    return {
        "energies": {label: value.detach().cpu() for label, value in energies.items()},
        "forces": forces,
        "orientation_energies": orientation_energies.detach().cpu(),
        "max_source_sum": sources.sum(dim=0).abs().max().detach().cpu().item(),
    }


def new_accumulator() -> dict[str, Any]:
    return {
        "energy_errors": [],
        "force_errors": [],
        "force_errors_by_axis": [[], [], []],
        "residual_energies_per_atom": [],
        "residual_forces": [],
        "net_residual_forces": [],
    }


def update_accumulator(
    accumulator: dict[str, Any],
    sample: dict[str, Any],
    residual_energy: torch.Tensor,
    residual_forces: torch.Tensor,
) -> None:
    base_energy = sample["base_energy"].to(torch.float64).reshape(-1)[0]
    base_forces = sample["base_forces"].to(torch.float64)
    target_energy = sample["energy"].to(torch.float64).reshape(-1)[0]
    target_forces = sample["forces"].to(torch.float64)
    atom_count = int(sample["num_atoms"])
    residual_energy = residual_energy.to(torch.float64).reshape(-1)[0]
    residual_forces = residual_forces.to(torch.float64)
    predicted_energy = base_energy + residual_energy
    predicted_forces = base_forces + residual_forces
    energy_error = (predicted_energy - target_energy) / atom_count
    force_error = predicted_forces - target_forces
    accumulator["energy_errors"].append(energy_error.item())
    accumulator["force_errors"].extend(force_error.reshape(-1).tolist())
    for axis in range(3):
        accumulator["force_errors_by_axis"][axis].extend(
            force_error[:, axis].tolist()
        )
    accumulator["residual_energies_per_atom"].append(
        (residual_energy / atom_count).item()
    )
    accumulator["residual_forces"].extend(residual_forces.reshape(-1).tolist())
    accumulator["net_residual_forces"].append(
        residual_forces.sum(dim=0).tolist()
    )


def finalize_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    energy = np.asarray(accumulator["energy_errors"], dtype=float)
    force = np.asarray(accumulator["force_errors"], dtype=float)
    residual_energy = np.asarray(
        accumulator["residual_energies_per_atom"], dtype=float
    )
    residual_force = np.asarray(accumulator["residual_forces"], dtype=float)
    net_residual_force = np.asarray(
        accumulator["net_residual_forces"], dtype=float
    )
    return {
        "energy_mae_mev_per_atom": 1000.0 * float(np.mean(np.abs(energy))),
        "energy_rmse_mev_per_atom": 1000.0
        * float(np.sqrt(np.mean(energy * energy))),
        "energy_bias_mev_per_atom": 1000.0 * float(np.mean(energy)),
        "force_mae_mev_per_angstrom": 1000.0 * float(np.mean(np.abs(force))),
        "force_rmse_mev_per_angstrom": 1000.0
        * float(np.sqrt(np.mean(force * force))),
        "force_rmse_by_axis_mev_per_angstrom": [
            1000.0 * float(np.sqrt(np.mean(np.square(axis_values))))
            for axis_values in accumulator["force_errors_by_axis"]
        ],
        "residual_energy_rms_mev_per_atom": 1000.0
        * float(np.sqrt(np.mean(residual_energy * residual_energy))),
        "residual_force_rms_mev_per_angstrom": 1000.0
        * float(np.sqrt(np.mean(residual_force * residual_force))),
        "net_residual_force_rms_mev_per_angstrom": (
            1000.0
            * np.sqrt(np.mean(net_residual_force * net_residual_force, axis=0))
        ).tolist(),
    }


def frozen_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    accumulator = new_accumulator()
    for sample in samples:
        zero_energy = torch.zeros((), dtype=torch.float64)
        zero_forces = torch.zeros_like(sample["forces"], dtype=torch.float64)
        update_accumulator(accumulator, sample, zero_energy, zero_forces)
    return finalize_accumulator(accumulator)


def named_audit_transformations(dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
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


def audit_oh48(
    model: torch.nn.Module,
    sample: dict[str, Any],
    group: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    fd_step: float,
    seed: int,
) -> dict[str, Any]:
    reference_graph = clone_graph(sample["data"], device, dtype)
    reference = cubic_residual_predictions(
        model, reference_graph, group, labels=("oh48",), compute_forces=True
    )
    reference_energy = reference["energies"]["oh48"].item()
    reference_force = reference["forces"]["oh48"]
    cell = reference_graph["cell"].reshape(-1, 3, 3)[0]
    centre = 0.5 * cell.sum(dim=0)
    transformations = named_audit_transformations(dtype, device)
    operations = {}
    for name, transformation in transformations.items():
        transformed_graph = clone_graph(sample["data"], device, dtype)
        transformed_graph["positions"] = centre + transform_in_cell_axis_basis(
            transformed_graph["positions"] - centre, cell, transformation
        )
        transformed_graph["shifts"] = transform_in_cell_axis_basis(
            transformed_graph["shifts"], cell, transformation
        )
        transformed = cubic_residual_predictions(
            model,
            transformed_graph,
            group,
            labels=("oh48",),
            compute_forces=True,
        )
        back_rotated_force = transform_in_cell_axis_basis(
            transformed["forces"]["oh48"].to(device),
            cell,
            transformation.T,
        ).cpu()
        force_difference = back_rotated_force - reference_force
        operations[name] = {
            "residual_energy_change_mev": 1000.0
            * abs(transformed["energies"]["oh48"].item() - reference_energy),
            "back_rotated_residual_force_rmse_mev_per_angstrom": 1000.0
            * float(force_difference.square().mean().sqrt()),
            "back_rotated_residual_force_max_mev_per_angstrom": 1000.0
            * float(force_difference.abs().max()),
        }

    generator = torch.Generator().manual_seed(seed)
    flat_index = int(
        torch.randint(3 * int(sample["num_atoms"]), (), generator=generator)
    )
    atom, axis = divmod(flat_index, 3)
    energies = []
    for sign in (-1.0, 1.0):
        graph = clone_graph(sample["data"], device, dtype)
        graph["positions"][atom, axis] += sign * fd_step
        prediction = cubic_residual_predictions(
            model, graph, group, labels=("oh48",), compute_forces=False
        )
        energies.append(prediction["energies"]["oh48"].item())
    finite_force = -(energies[1] - energies[0]) / (2.0 * fd_step)
    fd_error = abs(finite_force - reference_force[atom, axis].item())
    return {
        "sample_index": 0,
        "operations": operations,
        "max_residual_energy_change_mev": max(
            item["residual_energy_change_mev"] for item in operations.values()
        ),
        "max_back_rotated_residual_force_rmse_mev_per_angstrom": max(
            item["back_rotated_residual_force_rmse_mev_per_angstrom"]
            for item in operations.values()
        ),
        "finite_difference": {
            "atom": atom,
            "axis": axis,
            "step_angstrom": fd_step,
            "autograd_force_mev_per_angstrom": 1000.0
            * reference_force[atom, axis].item(),
            "finite_difference_force_mev_per_angstrom": 1000.0 * finite_force,
            "absolute_error_mev_per_angstrom": 1000.0 * fd_error,
        },
    }


def main() -> None:
    args = parse_arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.fd_step <= 0.0:
        raise ValueError("--fd-step must be positive")
    start = perf_counter()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint, device)
    dtype = next(model.parameters()).dtype
    cache_path = args.sample_cache or Path(checkpoint["test_cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    samples = cache["samples"]
    if args.limit is not None:
        samples = samples[: args.limit]
    group = cubic_signed_permutation_matrices(dtype=dtype, device=device)
    accumulators = {label: new_accumulator() for label in LABELS}
    orientation_energy_ranges = []
    source_sums = []

    for index, sample in enumerate(samples, start=1):
        graph = clone_graph(sample["data"], device, dtype)
        prediction = cubic_residual_predictions(model, graph, group)
        for label in LABELS:
            update_accumulator(
                accumulators[label],
                sample,
                prediction["energies"][label],
                prediction["forces"][label],
            )
        orientation_energies = prediction["orientation_energies"]
        orientation_energy_ranges.append(
            (orientation_energies.max() - orientation_energies.min()).item()
        )
        source_sums.append(prediction["max_source_sum"])
        if index == 1 or index % 5 == 0 or index == len(samples):
            print(
                f"evaluated {index}/{len(samples)} structures "
                f"({perf_counter() - start:.1f} s)",
                flush=True,
            )

    symmetry_audit = audit_oh48(
        model,
        samples[0],
        group,
        device,
        dtype,
        args.fd_step,
        args.seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "checkpoint": str(args.checkpoint),
        "sample_cache": str(cache_path),
        "structures": len(samples),
        "atoms_per_structure": sorted(
            {int(sample["num_atoms"]) for sample in samples}
        ),
        "group_sizes": {"o24": 24, "oh48": 48},
        "elapsed_seconds": perf_counter() - start,
        "max_source_sum": max(source_sums),
        "raw_orientation_energy_range": {
            "mean_mev": 1000.0 * float(np.mean(orientation_energy_ranges)),
            "max_mev": 1000.0 * float(np.max(orientation_energy_ranges)),
        },
        "metrics": {
            "frozen_mace": frozen_metrics(samples),
            **{
                label: finalize_accumulator(accumulators[label])
                for label in LABELS
            },
        },
        "oh48_symmetry_and_force_audit": symmetry_audit,
    }
    print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.strict:
        failures = []
        if symmetry_audit["max_residual_energy_change_mev"] > 1.0e-4:
            failures.append("O_h residual-energy invariance")
        if (
            symmetry_audit[
                "max_back_rotated_residual_force_rmse_mev_per_angstrom"
            ]
            > 1.0e-4
        ):
            failures.append("O_h residual-force equivariance")
        if (
            symmetry_audit["finite_difference"]["absolute_error_mev_per_angstrom"]
            > 0.05
        ):
            failures.append("O_h residual-force finite difference")
        if failures:
            raise RuntimeError("failed cubic-average checks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
