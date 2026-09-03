"""Optimization and model selection for frozen or joint MACE-FNO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..coupling import MACEFNOResidual, energy_force_loss
from .checkpoint import (
    load_mace_state_dict,
    load_residual_state_dict,
    mace_state_dict,
    residual_state_dict,
)
from .configuration import TrainingConfig
from .data import collate_samples
from .evaluation import evaluate, print_metrics, validation_objective
from .initialization import (
    configure_output_projection_warmup,
    finish_output_projection_warmup,
)
from .monitor import SpectralMonitor

Sample = dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    """State selected by validation and needed for reporting/checkpointing."""

    best_step: int
    best_validation_objective: float
    completed_steps: int
    stopped_early: bool
    warmup_learning_rate: float
    final_learning_rate: float
    final_mace_learning_rate: float | None = None


def evaluate_frozen_baseline(
    model: MACEFNOResidual,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    test_samples: list[Sample],
    configuration: TrainingConfig,
) -> dict[str, Any]:
    """Report initial-MACE errors and non-learned energy-offset controls."""
    optimization = configuration.optimization
    batch_size = optimization.evaluation_batch_size
    baseline_train = (
        evaluate(model, train_samples, baseline=True, batch_size=batch_size)
        if optimization.evaluation_scope == "all"
        else {}
    )
    baseline_validation = evaluate(
        model, validation_samples, baseline=True, batch_size=batch_size
    )
    baseline_test = evaluate(model, test_samples, baseline=True, batch_size=batch_size)
    baseline_label = (
        "frozen MACE" if optimization.mace_training == "frozen" else "initial MACE"
    )
    print_metrics(f"{baseline_label} train", baseline_train)
    print_metrics(f"{baseline_label} validation", baseline_validation)
    print_metrics(f"{baseline_label} held-out test", baseline_test)

    if baseline_train:
        energy_shift = -baseline_train["energy_bias"]
        formula_shifts = {
            formula: -metrics["energy_bias"]
            for formula, metrics in baseline_train["by_formula"].items()
        }
        for label, samples, shift in (
            ("constant-offset validation", validation_samples, energy_shift),
            ("formula-offset validation", validation_samples, formula_shifts),
            ("constant-offset held-out test", test_samples, energy_shift),
            ("formula-offset held-out test", test_samples, formula_shifts),
        ):
            print_metrics(
                label,
                evaluate(
                    model,
                    samples,
                    baseline=True,
                    energy_shift_per_atom=shift,
                    batch_size=batch_size,
                ),
            )
    return baseline_validation


def optimize_residual(
    model: MACEFNOResidual,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    baseline_validation: dict[str, Any],
    configuration: TrainingConfig,
    *,
    device: torch.device,
    spectral_monitor: SpectralMonitor | None = None,
) -> OptimizationResult:
    """Optimize the configured parameters and restore the best validation state."""
    optimization = configuration.optimization
    joint_training = optimization.mace_training == "joint"
    model.train()
    model_dtype = next(model.parameters()).dtype
    warmup_learning_rate = (
        optimization.output_warmup_learning_rate or optimization.learning_rate
    )
    mace_parameters: list[torch.nn.Parameter] = []
    if joint_training:
        mace_parameters = list(model.backbone.mace_model.parameters())
        if not mace_parameters:
            raise ValueError("joint training requires a MACE model with parameters")
        mace_parameter_ids = {id(parameter) for parameter in mace_parameters}
        parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in mace_parameter_ids
        ]
        model.backbone.set_trainable(optimization.mace_warmup_steps == 0)
        warmup_parameters: list[torch.nn.Parameter] = []
        optimizer = torch.optim.Adam(
            [
                {"params": parameters, "lr": optimization.learning_rate},
                {"params": mace_parameters, "lr": optimization.mace_learning_rate},
            ]
        )
        print(
            "joint MACE-FNO optimization: "
            f"fno_learning_rate={optimization.learning_rate:.6e}, "
            f"mace_learning_rate={optimization.mace_learning_rate:.6e}, "
            f"mace_warmup_steps={optimization.mace_warmup_steps}",
            flush=True,
        )
    else:
        parameters, warmup_parameters = configure_output_projection_warmup(
            model,
            optimization.output_warmup_steps,
        )
        optimizer = torch.optim.Adam(
            parameters,
            lr=(
                warmup_learning_rate
                if optimization.output_warmup_steps
                else optimization.learning_rate
            ),
        )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=optimization.lr_decay_factor,
            patience=optimization.lr_patience_evals,
            min_lr=(
                [
                    optimization.minimum_learning_rate,
                    min(
                        optimization.minimum_learning_rate,
                        optimization.mace_learning_rate,
                    ),
                ]
                if joint_training
                else optimization.minimum_learning_rate
            ),
        )
        if optimization.lr_scheduler == "plateau"
        else None
    )
    generator = torch.Generator().manual_seed(configuration.runtime.seed + 1)

    if optimization.output_warmup_steps:
        print(
            "output-projection warm-up: "
            f"steps={optimization.output_warmup_steps}, "
            f"learning_rate={warmup_learning_rate:.6e}, "
            f"active_parameters={sum(p.numel() for p in warmup_parameters)}/"
            f"{sum(p.numel() for p in parameters)}",
            flush=True,
        )

    baseline_objective = validation_objective(
        baseline_validation,
        energy_weight=optimization.energy_weight,
        force_weight=optimization.force_weight,
        energy_scale=optimization.energy_scale,
        force_scale=optimization.force_scale,
    )
    initial_validation = evaluate(
        model,
        validation_samples,
        batch_size=optimization.evaluation_batch_size,
    )
    best_step = 0
    best_objective = validation_objective(
        initial_validation,
        energy_weight=optimization.energy_weight,
        force_weight=optimization.force_weight,
        energy_scale=optimization.energy_scale,
        force_scale=optimization.force_scale,
    )
    best_residual_state = residual_state_dict(model)
    best_mace_state = mace_state_dict(model) if joint_training else None
    print(
        f"validation objective at initial MACE baseline: {baseline_objective:.6e}",
        flush=True,
    )
    print(
        f"validation objective at initialized combined model: {best_objective:.6e}",
        flush=True,
    )

    completed_steps = 0
    stopped_early = False
    evaluations_at_minimum_lr = 0
    for step in range(optimization.steps):
        if step == optimization.mace_warmup_steps and optimization.mace_warmup_steps:
            model.backbone.set_trainable(True)
            model.train()
            print(
                f"MACE warm-up complete at step {step}: "
                f"unfroze {sum(p.numel() for p in mace_parameters)} parameters",
                flush=True,
            )
        if (
            step == optimization.output_warmup_steps
            and optimization.output_warmup_steps
        ):
            finish_output_projection_warmup(parameters)
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = optimization.learning_rate
            print(
                f"output-projection warm-up complete at step {step}: "
                f"unfroze {sum(p.numel() for p in parameters)} parameters, "
                f"learning_rate={previous_learning_rate:.6e} -> "
                f"{optimization.learning_rate:.6e}",
                flush=True,
            )

        optimizer.zero_grad(set_to_none=True)
        accumulated = {"loss": 0.0, "energy": 0.0, "forces": 0.0}
        for _ in range(optimization.accumulation_steps):
            sample_indices = torch.randint(
                len(train_samples),
                (optimization.batch_size,),
                generator=generator,
            ).tolist()
            sample_batch = [train_samples[index] for index in sample_indices]
            graph, target_energy, target_forces = collate_samples(
                sample_batch, device, model_dtype
            )
            if joint_training:
                output = model(
                    graph,
                    training=True,
                    compute_force=True,
                    compute_residual_force=False,
                )
                predicted_energy = output["energy"]
                predicted_forces = output["forces"]
            else:
                target_energy = torch.cat(
                    [sample["residual_energy"] for sample in sample_batch]
                ).to(device=device)
                target_forces = torch.cat(
                    [sample["residual_forces"] for sample in sample_batch]
                ).to(device=device)
                output = model(
                    graph,
                    training=True,
                    compute_force=False,
                    compute_residual_force=True,
                )
                predicted_energy = output["residual_energy"]
                predicted_forces = output["residual_forces"]
            target_energy = target_energy.to(dtype=predicted_energy.dtype)
            target_forces = target_forces.to(dtype=predicted_forces.dtype)
            terms = energy_force_loss(
                predicted_energy,
                predicted_forces,
                target_energy,
                target_forces,
                graph["batch"],
                energy_weight=optimization.energy_weight,
                force_weight=optimization.force_weight,
                energy_scale=optimization.energy_scale,
                force_scale=optimization.force_scale,
            )
            (terms["loss"] / optimization.accumulation_steps).backward()
            for name in accumulated:
                accumulated[name] += (
                    terms[name].item() / optimization.accumulation_steps
                )
        optimizer.step()
        completed_steps = step + 1

        if step == 0 or completed_steps % max(1, optimization.steps // 10) == 0:
            print(
                f"step {completed_steps:5d}/{optimization.steps}: "
                f"loss={accumulated['loss']:.6e}, "
                f"energy={accumulated['energy']:.6e}, "
                f"forces={accumulated['forces']:.6e}",
                flush=True,
            )
        if (
            completed_steps % optimization.eval_interval != 0
            and completed_steps != optimization.steps
        ):
            continue

        validation_metrics = evaluate(
            model,
            validation_samples,
            batch_size=optimization.evaluation_batch_size,
        )
        print_metrics(f"validation step {completed_steps}", validation_metrics)
        score = validation_objective(
            validation_metrics,
            energy_weight=optimization.energy_weight,
            force_weight=optimization.force_weight,
            energy_scale=optimization.energy_scale,
            force_scale=optimization.force_scale,
        )
        print(f"validation objective step {completed_steps}: {score:.6e}", flush=True)
        if spectral_monitor is not None:
            spectral_monitor.evaluate_validation(
                model,
                step=completed_steps,
                validation_objective=score,
            )

        improved = score < best_objective
        if improved:
            best_step = completed_steps
            best_objective = score
            best_residual_state = residual_state_dict(model)
            if joint_training:
                best_mace_state = mace_state_dict(model)
            print(f"new best validation step: {best_step}", flush=True)

        scheduler_start_step = max(
            optimization.output_warmup_steps,
            optimization.mace_warmup_steps,
        )
        if scheduler is not None and completed_steps > scheduler_start_step:
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            previous_mace_learning_rate = (
                optimizer.param_groups[1]["lr"] if joint_training else None
            )
            scheduler.step(score)
            current_learning_rate = optimizer.param_groups[0]["lr"]
            if current_learning_rate != previous_learning_rate:
                message = (
                    f"learning rate step {completed_steps}: "
                    f"FNO {previous_learning_rate:.6e} -> "
                    f"{current_learning_rate:.6e}"
                )
                if joint_training:
                    message += (
                        f", MACE {previous_mace_learning_rate:.6e} -> "
                        f"{optimizer.param_groups[1]['lr']:.6e}"
                    )
                print(message, flush=True)
            at_minimum_learning_rate = current_learning_rate <= (
                optimization.minimum_learning_rate
                * (1.0 + 16.0 * np.finfo(np.float64).eps)
            )
            if at_minimum_learning_rate:
                evaluations_at_minimum_lr = (
                    0 if improved else evaluations_at_minimum_lr + 1
                )
            else:
                evaluations_at_minimum_lr = 0
            if (
                optimization.early_stopping_patience_evals
                and evaluations_at_minimum_lr
                >= optimization.early_stopping_patience_evals
            ):
                stopped_early = True
                print(
                    f"early stopping at step {completed_steps}: no validation "
                    f"improvement in {evaluations_at_minimum_lr} checks at "
                    f"minimum learning rate {current_learning_rate:.6e}",
                    flush=True,
                )
                break

    load_residual_state_dict(model, best_residual_state)
    if best_mace_state is not None:
        load_mace_state_dict(model, best_mace_state)
    print(
        f"restored best validation step {best_step} (objective={best_objective:.6e})",
        flush=True,
    )
    return OptimizationResult(
        best_step=best_step,
        best_validation_objective=best_objective,
        completed_steps=completed_steps,
        stopped_early=stopped_early,
        warmup_learning_rate=warmup_learning_rate,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
        final_mace_learning_rate=(
            float(optimizer.param_groups[1]["lr"]) if joint_training else None
        ),
    )


def evaluate_selected_model(
    model: MACEFNOResidual,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    test_samples: list[Sample],
    configuration: TrainingConfig,
) -> None:
    """Report the errors of the restored best residual checkpoint."""
    optimization = configuration.optimization
    batch_size = optimization.evaluation_batch_size
    if optimization.evaluation_scope == "all":
        print_metrics(
            "selected train",
            evaluate(model, train_samples, batch_size=batch_size),
        )
    print_metrics(
        "selected validation",
        evaluate(model, validation_samples, batch_size=batch_size),
    )
    print_metrics(
        "selected held-out test",
        evaluate(model, test_samples, batch_size=batch_size),
    )
