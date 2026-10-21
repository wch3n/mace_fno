"""Optimization and model selection for frozen or joint MACE-FNO training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ..coupling import MACEFNOResidual
from ._optimization import (
    build_optimizer,
    finish_warmup,
    start_second_stage,
    train_step,
    update_scheduler,
)
from ._training_state import TrainingState
from .checkpoint import (
    load_mace_state_dict,
    load_residual_state_dict,
    mace_state_dict,
    residual_state_dict,
)
from .configuration import TrainingConfig
from .evaluation import evaluate, print_metrics, validation_objective
from .monitor import SpectralMonitor
from .resume import (
    restore_rng_state,
    validate_resume_checkpoint,
)
from .selection import EnergyCheckpointSelector

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
    energy_selection: dict[str, Any] | None = None
    optimization_stage: int = 1
    stage_start_step: int = 0


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
    best_checkpoint_callback: (
        Callable[[MACEFNOResidual, OptimizationResult], None] | None
    ) = None,
    resume_state: Mapping[str, Any] | None = None,
    last_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    energy_checkpoint_callback: Callable[[OptimizationResult], None] | None = None,
) -> OptimizationResult:
    """Optimize the configured parameters and restore the best validation state."""
    optimization = configuration.optimization
    if resume_state is not None:
        validate_resume_checkpoint(resume_state, configuration)
        if resume_state["device_type"] != device.type:
            raise ValueError("resume requires the same device type (CPU or CUDA)")
    staged = optimization.stage_two_start_step is not None
    state = TrainingState()
    state.optimization_stage = (
        resume_state.get("optimization_stage", 1) if resume_state else 1
    )
    state.stage_start_step = (
        resume_state.get("stage_start_step", 0) if resume_state else 0
    )
    loss_settings = optimization.loss_settings(state.optimization_stage)
    joint_training = optimization.mace_training == "joint"
    model.train()
    model_dtype = next(model.parameters()).dtype
    optimizer_setup = build_optimizer(model, optimization)
    optimizer = optimizer_setup.optimizer
    warmup_learning_rate = optimizer_setup.warmup_learning_rate
    generator = torch.Generator().manual_seed(configuration.runtime.seed + 1)

    baseline_objective = validation_objective(
        baseline_validation,
        **loss_settings,
    )
    print(
        f"validation objective at initial MACE baseline: {baseline_objective:.6e}",
        flush=True,
    )
    if resume_state is None:
        initial_validation = evaluate(
            model,
            validation_samples,
            batch_size=optimization.evaluation_batch_size,
        )
        state.best_step = 0
        state.best_objective = validation_objective(
            initial_validation,
            **loss_settings,
        )
        state.best_residual_state = residual_state_dict(model)
        state.best_mace_state = mace_state_dict(model) if joint_training else None
        state.completed_steps = 0
        state.stopped_early = False
        state.evaluations_at_minimum_lr = 0
        print(
            f"validation objective at initialized combined model: {state.best_objective:.6e}",
            flush=True,
        )
    else:
        state.restore(
            model,
            resume_state,
            optimizer_setup,
            generator,
            spectral_monitor,
            joint_training=joint_training,
        )
        print(
            f"resuming after step {state.completed_steps} toward {optimization.steps} total "
            f"steps (best step {state.best_step}, objective={state.best_objective:.6e})",
            flush=True,
        )
        if state.stopped_early:
            print("saved early-stopping condition is already satisfied", flush=True)

    energy_selector = None
    if optimization.energy_checkpoint_tolerance is not None:
        energy_selector = EnergyCheckpointSelector(
            optimization.energy_checkpoint_tolerance,
            optimization.energy_checkpoint_constraint,
            optimization.energy_checkpoint_metric,
        )
        if resume_state is not None:
            if resume_state.get("energy_selector") is None:
                raise ValueError(
                    "resume checkpoint is missing energy-selection candidates"
                )
            energy_selector.load_state_dict(resume_state["energy_selector"])
        else:
            energy_selector.consider(
                0,
                initial_validation,
                state.best_objective,
                lambda: {
                    "residual_state_dict": state.best_residual_state,
                    "mace_state_dict": state.best_mace_state,
                },
            )

    def current_result() -> OptimizationResult:
        return OptimizationResult(
            best_step=state.best_step,
            best_validation_objective=state.best_objective,
            completed_steps=state.completed_steps,
            stopped_early=state.stopped_early,
            warmup_learning_rate=warmup_learning_rate,
            final_learning_rate=float(optimizer.param_groups[0]["lr"]),
            final_mace_learning_rate=(
                float(optimizer.param_groups[1]["lr"]) if joint_training else None
            ),
            energy_selection=energy_selector.selected if energy_selector else None,
            optimization_stage=state.optimization_stage,
            stage_start_step=state.stage_start_step,
        )

    def save_latest() -> None:
        if (
            last_checkpoint_callback is None
            or state.last_saved_step == state.completed_steps
        ):
            return
        payload = state.checkpoint_payload(
            model,
            configuration,
            optimizer_setup,
            generator,
            device=device,
            spectral_monitor=spectral_monitor,
            energy_selector=energy_selector,
            loss_settings=loss_settings,
        )
        last_checkpoint_callback(payload)
        state.last_saved_step = state.completed_steps

    if best_checkpoint_callback is not None:
        if resume_state is not None:
            load_residual_state_dict(model, state.best_residual_state)
            if state.best_mace_state is not None:
                load_mace_state_dict(model, state.best_mace_state)
        best_checkpoint_callback(model, current_result())
        if resume_state is not None:
            load_residual_state_dict(model, resume_state["residual_state_dict"])
            if joint_training:
                load_mace_state_dict(model, resume_state["mace_state_dict"])

    if energy_selector is not None and energy_checkpoint_callback is not None:
        energy_checkpoint_callback(current_result())

    if resume_state is not None:
        # Setup, cache preparation, and model construction must not advance the
        # resumed RNG stream. Restore last, immediately before optimization.
        restore_rng_state(resume_state["rng_state"], device)
    else:
        save_latest()

    if staged:
        print(
            f"two-stage optimization: switch after {optimization.stage_two_start_step} "
            "updates; early stopping applies only in stage two; "
            "preserve Adam moments and reset selection/scheduler history",
            flush=True,
        )

    for step in range(state.completed_steps, optimization.steps):
        if state.stopped_early:
            break
        finish_warmup(
            model,
            optimization,
            optimizer_setup,
            step=step,
            optimization_stage=state.optimization_stage,
        )

        if (
            staged
            and state.optimization_stage == 1
            and step == optimization.stage_two_start_step
        ):
            state.optimization_stage = 2
            state.stage_start_step = state.completed_steps
            loss_settings = optimization.loss_settings(2)
            start_second_stage(optimization, optimizer_setup)
            state.evaluations_at_minimum_lr = 0
            transition_metrics = evaluate(
                model,
                validation_samples,
                batch_size=optimization.evaluation_batch_size,
            )
            state.best_objective = validation_objective(
                transition_metrics, **loss_settings
            )
            state.best_step = state.completed_steps
            state.best_residual_state = residual_state_dict(model)
            state.best_mace_state = mace_state_dict(model) if joint_training else None
            if energy_selector is not None:
                energy_selector = EnergyCheckpointSelector(
                    optimization.energy_checkpoint_tolerance,
                    optimization.energy_checkpoint_constraint,
                    optimization.energy_checkpoint_metric,
                )
                energy_selector.consider(
                    state.completed_steps,
                    transition_metrics,
                    state.best_objective,
                    lambda: {
                        "residual_state_dict": state.best_residual_state,
                        "mace_state_dict": state.best_mace_state,
                    },
                )
            print(
                f"stage two starts after step {state.completed_steps}: "
                + ", ".join(
                    f"{key}={value:.6e}" for key, value in loss_settings.items()
                )
                + f", learning_rates={[group['lr'] for group in optimizer.param_groups]}"
                + f", initial validation objective={state.best_objective:.6e}",
                flush=True,
            )
            if best_checkpoint_callback is not None:
                best_checkpoint_callback(model, current_result())
            if energy_selector is not None and energy_checkpoint_callback is not None:
                energy_checkpoint_callback(current_result())
            # Persist the reset state even though no new update has occurred.
            state.last_saved_step = -1
            save_latest()

        accumulated = train_step(
            model,
            train_samples,
            optimization,
            optimizer,
            generator,
            device=device,
            model_dtype=model_dtype,
            loss_settings=loss_settings,
        )
        state.completed_steps = step + 1

        if step == 0 or state.completed_steps % max(1, optimization.steps // 10) == 0:
            print(
                f"step {state.completed_steps:5d}/{optimization.steps}: "
                f"loss={accumulated['loss']:.6e}, "
                f"energy={accumulated['energy']:.6e}, "
                f"forces={accumulated['forces']:.6e}",
                flush=True,
            )
        if (
            state.completed_steps % optimization.eval_interval != 0
            and state.completed_steps != optimization.steps
            and state.completed_steps != optimization.stage_two_start_step
        ):
            interval = configuration.runtime.checkpoint_interval
            if interval and state.completed_steps % interval == 0:
                save_latest()
            continue

        validation_metrics = evaluate(
            model,
            validation_samples,
            batch_size=optimization.evaluation_batch_size,
        )
        print_metrics(f"validation step {state.completed_steps}", validation_metrics)
        score = validation_objective(
            validation_metrics,
            **loss_settings,
        )
        stage_label = f" (stage {state.optimization_stage})" if staged else ""
        print(
            f"validation objective step {state.completed_steps}: {score:.6e}{stage_label}",
            flush=True,
        )
        if energy_selector is not None:
            selection_changed = energy_selector.consider(
                state.completed_steps,
                validation_metrics,
                score,
                lambda: {
                    "residual_state_dict": residual_state_dict(model),
                    "mace_state_dict": mace_state_dict(model)
                    if joint_training
                    else None,
                },
            )
            if selection_changed and energy_checkpoint_callback is not None:
                energy_checkpoint_callback(current_result())
        if spectral_monitor is not None:
            spectral_monitor.evaluate_validation(
                model,
                step=state.completed_steps,
                validation_objective=score,
                **({"optimization_stage": state.optimization_stage} if staged else {}),
            )

        improved = score < state.best_objective
        if improved:
            state.best_step = state.completed_steps
            state.best_objective = score
            state.best_residual_state = residual_state_dict(model)
            if joint_training:
                state.best_mace_state = mace_state_dict(model)
            print(f"new best validation step: {state.best_step}", flush=True)
            if best_checkpoint_callback is not None:
                best_checkpoint_callback(model, current_result())

        state.evaluations_at_minimum_lr, state.stopped_early = update_scheduler(
            optimization,
            optimizer_setup,
            completed_steps=state.completed_steps,
            optimization_stage=state.optimization_stage,
            score=score,
            improved=improved,
            evaluations_at_minimum_lr=state.evaluations_at_minimum_lr,
        )

        if (
            optimization.early_stopping_patience_steps
            and (not staged or state.optimization_stage == 2)
            and state.completed_steps < optimization.steps
            and state.completed_steps - state.best_step
            >= optimization.early_stopping_patience_steps
        ):
            state.stopped_early = True
            print(
                f"early stopping at step {state.completed_steps}: validation objective "
                f"has not improved for {state.completed_steps - state.best_step} optimizer steps",
                flush=True,
            )
        save_latest()
        if state.stopped_early:
            break

    save_latest()
    load_residual_state_dict(model, state.best_residual_state)
    if state.best_mace_state is not None:
        load_mace_state_dict(model, state.best_mace_state)
    print(
        f"restored best validation step {state.best_step} (objective={state.best_objective:.6e})",
        flush=True,
    )
    return current_result()


def evaluate_selected_model(
    model: MACEFNOResidual,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    test_samples: list[Sample],
    configuration: TrainingConfig,
    *,
    label: str = "selected",
) -> dict[str, Any]:
    """Report the errors of the restored best residual checkpoint."""
    optimization = configuration.optimization
    batch_size = optimization.evaluation_batch_size
    splits = {"validation": validation_samples, "held-out test": test_samples}
    if optimization.evaluation_scope == "all":
        splits = {"train": train_samples, **splits}
    metrics = {}
    for split, samples in splits.items():
        metrics[split] = evaluate(model, samples, batch_size=batch_size)
        print_metrics(f"{label} {split}", metrics[split])
    return metrics
