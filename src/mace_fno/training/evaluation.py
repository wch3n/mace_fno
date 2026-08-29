"""Frozen-target preparation and metrics for MACE-FNO residuals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch

from ..coupling import MACEFNOResidual
from .data import collate_samples


def ensure_frozen_residual_targets(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> bool:
    """Cache frozen-MACE predictions and reference-minus-MACE labels."""
    required = {
        "base_energy",
        "base_forces",
        "residual_energy",
        "residual_forces",
    }
    if all(required <= set(sample) for sample in samples):
        return False
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model.backbone.mace_model.eval()
    model_dtype = next(model.parameters()).dtype
    for start in range(0, len(samples), batch_size):
        sample_batch = samples[start : start + batch_size]
        graph, _, _ = collate_samples(sample_batch, device, model_dtype)
        output = model.backbone.mace_model(
            graph,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
        )
        base_energy = (
            model.backbone.corrected_base_energy(graph, output)
            .detach()
            .cpu()
            .to(torch.float64)
        )
        base_forces = output["forces"].detach().cpu().to(torch.float64)
        node_start = 0
        for graph_index, sample in enumerate(sample_batch):
            node_stop = node_start + sample["num_atoms"]
            sample_base_energy = base_energy[graph_index : graph_index + 1]
            sample_base_forces = base_forces[node_start:node_stop]
            sample["base_energy"] = sample_base_energy
            sample["base_forces"] = sample_base_forces
            sample["residual_energy"] = sample["energy"] - sample_base_energy
            sample["residual_forces"] = sample["forces"] - sample_base_forces
            node_start = node_stop
        completed = start + len(sample_batch)
        if completed % 500 == 0 or completed == len(samples):
            print(f"cached frozen targets: {completed}/{len(samples)}", flush=True)
    return True


def evaluate(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    baseline: bool = False,
    energy_shift_per_atom: float | Mapping[str, float] = 0.0,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Evaluate energy-per-atom and force errors, including formula groups."""
    if not samples:
        return {}
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    energy_errors = []
    force_errors = []
    group_energy_errors: dict[str, list[torch.Tensor]] = defaultdict(list)
    group_force_errors: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, len(samples), batch_size):
        sample_batch = samples[start : start + batch_size]
        graph, target_energy, target_forces = collate_samples(
            sample_batch, device, model_dtype
        )
        cached_base_energy = torch.cat(
            [sample["base_energy"] for sample in sample_batch]
        ).to(device=device)
        cached_base_forces = torch.cat(
            [sample["base_forces"] for sample in sample_batch]
        ).to(device=device)
        if baseline:
            predicted_energy = cached_base_energy
            predicted_forces = cached_base_forces
        else:
            output = model(
                graph,
                training=False,
                compute_force=False,
                compute_residual_force=True,
            )
            predicted_energy = cached_base_energy + output["residual_energy"]
            predicted_forces = cached_base_forces + output["residual_forces"]
        atoms_per_graph = torch.bincount(graph["batch"]).to(predicted_energy)
        if isinstance(energy_shift_per_atom, Mapping):
            shifts = predicted_energy.new_tensor(
                [
                    float(energy_shift_per_atom.get(sample["formula"], 0.0))
                    for sample in sample_batch
                ]
            )
        else:
            shifts = predicted_energy.new_full(
                predicted_energy.shape, float(energy_shift_per_atom)
            )
        predicted_energy = predicted_energy + shifts * atoms_per_graph
        energy_error = ((predicted_energy - target_energy) / atoms_per_graph).detach()
        force_error = (predicted_forces - target_forces).detach()
        energy_errors.append(energy_error)
        force_errors.append(force_error.reshape(-1))
        for graph_index, sample in enumerate(sample_batch):
            atom_mask = graph["batch"] == graph_index
            group_energy_errors[sample["formula"]].append(
                energy_error[graph_index : graph_index + 1]
            )
            group_force_errors[sample["formula"]].append(
                force_error[atom_mask].reshape(-1)
            )

    def metrics(
        energy_error_list: list[torch.Tensor],
        force_error_list: list[torch.Tensor],
    ) -> dict[str, float]:
        energy_error = torch.cat(energy_error_list)
        force_error = torch.cat(force_error_list)
        return {
            "energy_mae": energy_error.abs().mean().item(),
            "energy_rmse": energy_error.square().mean().sqrt().item(),
            "energy_bias": energy_error.mean().item(),
            "force_mae": force_error.abs().mean().item(),
            "force_rmse": force_error.square().mean().sqrt().item(),
        }

    result: dict[str, Any] = metrics(energy_errors, force_errors)
    result["by_formula"] = {
        formula: metrics(group_energy_errors[formula], group_force_errors[formula])
        for formula in sorted(group_energy_errors)
    }
    result["formula_counts"] = {
        formula: len(group_energy_errors[formula])
        for formula in sorted(group_energy_errors)
    }
    model.train(was_training)
    return result


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    """Print the standard residual benchmark metrics."""
    if not metrics:
        return
    print(
        f"{label}: E_MAE={1000.0 * metrics['energy_mae']:.4f} meV/atom, "
        f"E_RMSE={1000.0 * metrics['energy_rmse']:.4f} meV/atom, "
        f"E_bias={1000.0 * metrics['energy_bias']:.4f} meV/atom, "
        f"F_MAE={1000.0 * metrics['force_mae']:.4f} meV/A, "
        f"F_RMSE={1000.0 * metrics['force_rmse']:.4f} meV/A",
        flush=True,
    )
    for formula, group in metrics.get("by_formula", {}).items():
        print(
            f"  {formula} (n={metrics['formula_counts'][formula]}): "
            f"E_RMSE={1000.0 * group['energy_rmse']:.4f} meV/atom, "
            f"F_RMSE={1000.0 * group['force_rmse']:.4f} meV/A",
            flush=True,
        )


def validation_objective(
    metrics: Mapping[str, Any],
    *,
    energy_weight: float,
    force_weight: float,
    energy_scale: float,
    force_scale: float,
) -> float:
    """Match the normalized energy/force objective used during fitting."""
    if not metrics:
        return float("inf")
    return (
        float(energy_weight)
        * (float(metrics["energy_rmse"]) / float(energy_scale)) ** 2
        + float(force_weight)
        * (float(metrics["force_rmse"]) / float(force_scale)) ** 2
    )
