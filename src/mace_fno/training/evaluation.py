"""Reference-target preparation and metrics for MACE-FNO models."""

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
    benchmark_energy_errors: dict[str, list[torch.Tensor]] = defaultdict(list)
    benchmark_force_errors: dict[str, list[torch.Tensor]] = defaultdict(list)
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
        joint_training = getattr(model, "mace_training", "frozen") == "joint"
        if baseline:
            predicted_energy = cached_base_energy
            predicted_forces = cached_base_forces
        elif joint_training:
            output = model(
                graph,
                training=False,
                compute_force=True,
                compute_residual_force=False,
            )
            predicted_energy = output["energy"]
            predicted_forces = output["forces"]
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
            if "benchmark_group" in sample:
                benchmark_group = str(sample["benchmark_group"])
                benchmark_energy_errors[benchmark_group].append(
                    energy_error[graph_index : graph_index + 1]
                )
                benchmark_force_errors[benchmark_group].append(
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
    if benchmark_energy_errors:
        result["by_benchmark_group"] = {
            group: metrics(
                benchmark_energy_errors[group], benchmark_force_errors[group]
            )
            for group in sorted(benchmark_energy_errors)
        }
        result["benchmark_group_counts"] = {
            group: len(benchmark_energy_errors[group])
            for group in sorted(benchmark_energy_errors)
        }
    model.train(was_training)
    return result


_METRIC_LABEL_WIDTH = 32
_METRIC_VALUE_WIDTH = 10


def _metric_value(value: float | None) -> str:
    if value is None:
        return f"{'--':>{_METRIC_VALUE_WIDTH}}"
    return f"{1000.0 * value:>{_METRIC_VALUE_WIDTH}.4f}"


def _metric_row(
    label: str,
    *,
    energy_mae: float | None,
    energy_rmse: float | None,
    energy_bias: float | None,
    force_mae: float | None,
    force_rmse: float | None,
) -> str:
    return (
        f"{label:<{_METRIC_LABEL_WIDTH}} | "
        f"E_MAE={_metric_value(energy_mae)} | "
        f"E_RMSE={_metric_value(energy_rmse)} | "
        f"E_ME={_metric_value(energy_bias)} | "
        f"F_MAE={_metric_value(force_mae)} | "
        f"F_RMSE={_metric_value(force_rmse)} | "
        "units: E=meV/atom, F=meV/A"
    )


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    """Print standard residual metrics in stable, fixed-width columns."""
    if not metrics:
        return
    print(
        _metric_row(
            label,
            energy_mae=metrics["energy_mae"],
            energy_rmse=metrics["energy_rmse"],
            energy_bias=metrics["energy_bias"],
            force_mae=metrics["force_mae"],
            force_rmse=metrics["force_rmse"],
        ),
        flush=True,
    )
    for formula, group in metrics.get("by_formula", {}).items():
        print(
            _metric_row(
                f"  {formula} (n={metrics['formula_counts'][formula]})",
                energy_mae=None,
                energy_rmse=group["energy_rmse"],
                energy_bias=None,
                force_mae=None,
                force_rmse=group["force_rmse"],
            ),
            flush=True,
        )
    for name, group in metrics.get("by_benchmark_group", {}).items():
        print(
            _metric_row(
                f"  group={name} (n={metrics['benchmark_group_counts'][name]})",
                energy_mae=None,
                energy_rmse=group["energy_rmse"],
                energy_bias=None,
                force_mae=None,
                force_rmse=group["force_rmse"],
            ),
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
