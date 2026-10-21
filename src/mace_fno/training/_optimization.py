"""Optimizer setup, update steps, and learning-rate policies for training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..coupling import MACEFNOResidual, energy_force_loss
from .configuration import OptimizationConfig
from .data import collate_samples
from .initialization import (
    configure_output_projection_warmup,
    finish_output_projection_warmup,
)


@dataclass
class OptimizerSetup:
    """Optimizer resources shared by warm-up, stage changes, and validation."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None
    initial_scheduler_state: dict[str, Any] | None
    parameters: list[torch.nn.Parameter]
    mace_parameters: list[torch.nn.Parameter]
    warmup_learning_rate: float


def build_optimizer(
    model: MACEFNOResidual, optimization: OptimizationConfig
) -> OptimizerSetup:
    """Create the parameter groups and scheduler without advancing RNG state."""
    joint_training = optimization.mace_training == "joint"
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
    initial_scheduler_state = deepcopy(scheduler.state_dict()) if scheduler else None
    if optimization.output_warmup_steps:
        print(
            "output-projection warm-up: "
            f"steps={optimization.output_warmup_steps}, "
            f"learning_rate={warmup_learning_rate:.6e}, "
            f"active_parameters={sum(p.numel() for p in warmup_parameters)}/"
            f"{sum(p.numel() for p in parameters)}",
            flush=True,
        )

    return OptimizerSetup(
        optimizer=optimizer,
        scheduler=scheduler,
        initial_scheduler_state=initial_scheduler_state,
        parameters=parameters,
        mace_parameters=mace_parameters,
        warmup_learning_rate=warmup_learning_rate,
    )


def finish_warmup(
    model: MACEFNOResidual,
    optimization: OptimizationConfig,
    setup: OptimizerSetup,
    *,
    step: int,
    optimization_stage: int,
) -> None:
    """Unfreeze the configured parameters at the start of the matching step."""
    optimizer = setup.optimizer
    parameters = setup.parameters
    mace_parameters = setup.mace_parameters
    if (
        step == optimization.mace_warmup_steps
        and optimization.mace_warmup_steps
        and optimization_stage == 1
    ):
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
        and optimization_stage == 1
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


def start_second_stage(optimization: OptimizationConfig, setup: OptimizerSetup) -> None:
    """Set second-stage learning rates and reset only scheduler history."""
    optimizer = setup.optimizer
    scheduler = setup.scheduler
    initial_scheduler_state = setup.initial_scheduler_state
    for index, new_rate in enumerate(
        (
            optimization.stage_two_learning_rate,
            optimization.stage_two_mace_learning_rate,
        )
    ):
        if new_rate is not None:
            optimizer.param_groups[index]["lr"] = new_rate
    if scheduler is not None:
        scheduler.load_state_dict(deepcopy(initial_scheduler_state))
        # A smaller second-stage MACE rate must not be raised by its floor.
        scheduler.min_lrs = [
            min(floor, group["lr"])
            for floor, group in zip(scheduler.min_lrs, optimizer.param_groups)
        ]
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def train_step(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    optimization: OptimizationConfig,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    loss_settings: dict[str, float],
) -> dict[str, float]:
    """Accumulate one optimizer update, preserving sampling and gradient order."""
    joint_training = optimization.mace_training == "joint"
    optimizer.zero_grad(set_to_none=True)
    accumulated = {"loss": 0.0, "energy": 0.0, "forces": 0.0}
    for _ in range(optimization.accumulation_steps):
        sample_indices = torch.randint(
            len(samples),
            (optimization.batch_size,),
            generator=generator,
        ).tolist()
        sample_batch = [samples[index] for index in sample_indices]
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
            **loss_settings,
        )
        (terms["loss"] / optimization.accumulation_steps).backward()
        for name in accumulated:
            accumulated[name] += terms[name].item() / optimization.accumulation_steps
    optimizer.step()
    return accumulated


def update_scheduler(
    optimization: OptimizationConfig,
    setup: OptimizerSetup,
    *,
    completed_steps: int,
    optimization_stage: int,
    score: float,
    improved: bool,
    evaluations_at_minimum_lr: int,
) -> tuple[int, bool]:
    """Advance the plateau scheduler and check patience at its learning-rate floor."""
    optimizer = setup.optimizer
    scheduler = setup.scheduler
    joint_training = optimization.mace_training == "joint"
    staged = optimization.stage_two_start_step is not None
    stopped_early = False
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
            optimization.minimum_learning_rate * (1.0 + 16.0 * np.finfo(np.float64).eps)
        )
        if at_minimum_learning_rate:
            evaluations_at_minimum_lr = 0 if improved else evaluations_at_minimum_lr + 1
        else:
            evaluations_at_minimum_lr = 0
        if (
            optimization.early_stopping_patience_evals
            and (not staged or optimization_stage == 2)
            and evaluations_at_minimum_lr >= optimization.early_stopping_patience_evals
        ):
            stopped_early = True
            print(
                f"early stopping at step {completed_steps}: no validation "
                f"improvement in {evaluations_at_minimum_lr} checks at "
                f"minimum learning rate {current_learning_rate:.6e}",
                flush=True,
            )

    return evaluations_at_minimum_lr, stopped_early
