"""Train a frozen or jointly optimized MACE-FNO model.

The input should be an extended XYZ file containing reference total energies
and forces. Validation data are never used for gradients, and a separate test
file is evaluated only before and after optimization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from time import perf_counter

import torch

from mace_fno import MACEFNOResidual
from mace_fno.cli.config import parse_arguments
from mace_fno.cli.yaml_config import (
    resolved_configuration,
    write_resolved_configuration,
)
from mace_fno.training import (
    OptimizationResult,
    PreparedData,
    SpectralMonitor,
    TrainingConfig,
    build_training_model,
    cache_frozen_targets,
    choose_device,
    elapsed_since,
    evaluate_frozen_baseline,
    evaluate_selected_model,
    load_mace_calculator,
    optimize_residual,
    prepare_data,
    save_training_checkpoint,
    training_checkpoint_payload,
)
from mace_fno.training.checkpoint import (
    load_mace_state_dict,
    load_residual_state_dict,
    mace_state_dict,
    residual_state_dict,
)
from mace_fno.training.finetune import (
    initialize_finetune_model,
    load_finetune_checkpoint,
)
from mace_fno.training.resume import (
    input_fingerprints,
    load_resume_checkpoint,
    validate_resume_checkpoint,
)


def _save_checkpoint(
    args: argparse.Namespace,
    configuration: TrainingConfig,
    prepared: PreparedData,
    result: OptimizationResult,
    spectral_monitor: SpectralMonitor | None,
    model: MACEFNOResidual,
    *,
    write_configuration: bool = True,
    energy_selection: dict | None = None,
    evaluation_metrics: dict | None = None,
    fingerprints: dict | None = None,
    fine_tuning: dict | None = None,
) -> None:
    """Save one self-describing checkpoint and its resolved YAML sidecar."""
    checkpoint = configuration.runtime.checkpoint
    if energy_selection is not None:
        checkpoint = configuration.energy_checkpoint
        candidate = energy_selection["candidate"]
        result = replace(
            result, best_step=candidate["step"],
            best_validation_objective=candidate["validation_objective"],
        )
        # The monitor describes the primary best-loss model, not this selection.
        spectral_monitor = None
    if checkpoint is None:
        return

    model_config = configuration.model
    optimization = configuration.optimization
    effective_configuration = resolved_configuration(
        args,
        spatial_scheme=model_config.spatial_scheme,
        z_modes=(
            model_config.resolved_z_modes if model_config.spatial_scheme == "3d" else 0
        ),
        evaluation_batch_size=optimization.evaluation_batch_size,
        output_warmup_learning_rate=result.warmup_learning_rate,
        last_checkpoint=configuration.runtime.last_checkpoint,
    )
    spectral_record = None
    if spectral_monitor is not None:
        diagnostic_output = configuration.diagnostic.output
        spectral_record = {
            "configuration": spectral_monitor.report_configuration,
            "history": spectral_monitor.history,
            "output": str(diagnostic_output) if diagnostic_output is not None else None,
        }

    payload = training_checkpoint_payload(
        configuration,
        prepared,
        result,
        model,
        effective_configuration=effective_configuration,
        spectral_diagnostic=spectral_record,
    )
    if energy_selection is not None:
        payload.update(energy_selection["candidate"]["state"])
        payload["checkpoint_selection"] = energy_selection["metadata"]
    if evaluation_metrics is not None:
        payload["evaluation_metrics"] = evaluation_metrics
    if fingerprints is not None:
        payload["input_fingerprints"] = fingerprints
    if fine_tuning is not None:
        payload["fine_tuning"] = fine_tuning
    save_training_checkpoint(checkpoint, payload)
    label = "energy-selected" if energy_selection is not None else "best"
    print(
        f"{label} checkpoint: {checkpoint} "
        f"(step={result.best_step}, "
        f"validation_objective={result.best_validation_objective:.6e})",
        flush=True,
    )
    if energy_selection is not None:
        metadata = energy_selection["metadata"]
        print(
            f"  validation {metadata['energy_metric']} E_RMSE="
            f"{metadata['selected_energy_score']:.6e} eV/atom, "
            f"{metadata['constraint']}={metadata['selected_constraint']:.6e} "
            f"<= {metadata['constraint_limit']:.6e} "
            f"(best={metadata['best_constraint']:.6e}, "
            f"tolerance={metadata['relative_tolerance']:.2%})",
            flush=True,
        )
    if not write_configuration:
        return
    configuration_path = checkpoint.with_suffix(".config.yaml")
    write_resolved_configuration(configuration_path, effective_configuration)
    print(f"resolved configuration: {configuration_path}")


def main() -> None:
    """Execute one configured MACE-FNO training run."""
    total_start = perf_counter()
    args = parse_arguments()
    configuration = TrainingConfig.from_namespace(args)
    resume_state = (
        load_resume_checkpoint(configuration.runtime.resume)
        if configuration.runtime.resume is not None else None
    )
    fingerprints = (
        input_fingerprints(configuration)
        if configuration.runtime.last_checkpoint is not None else None
    )
    if resume_state is not None:
        validate_resume_checkpoint(resume_state, configuration, fingerprints=fingerprints)
    initialization = (
        load_finetune_checkpoint(configuration.runtime.init_from, configuration)
        if configuration.runtime.init_from is not None
        else None
    )
    fine_tuning = resume_state.get("fine_tuning") if resume_state is not None else None
    if initialization is not None:
        fine_tuning = initialization.provenance
    torch.manual_seed(configuration.runtime.seed)
    device = choose_device(configuration.runtime.device)
    dtype = torch.float32 if configuration.runtime.dtype == "float32" else torch.float64

    setup_start = perf_counter()
    calculator = load_mace_calculator(configuration, device)
    prepared = prepare_data(calculator, configuration, dtype)
    spectral_monitor = SpectralMonitor.create(
        configuration, prepared.validation_samples
    )
    model = build_training_model(
        calculator.models[0],
        configuration,
        (
            initialization.reference_cell
            if initialization is not None
            else prepared.reference_cell
        ),
        device=device,
        dtype=dtype,
    )
    setup_seconds = elapsed_since(setup_start, device)

    target_cache_start = perf_counter()
    cache_frozen_targets(model, prepared, configuration, device=device)
    target_cache_seconds = elapsed_since(target_cache_start, device)
    if initialization is not None:
        # Keep baseline/cache targets tied to the original MACE file. Joint
        # prediction uses the restored MACE weights through the live model.
        initialize_finetune_model(model, initialization)
        model._validate_cells(
            prepared.reference_cell.unsqueeze(0).to(device=device, dtype=dtype)
        )
        print(
            f"initialized from {fine_tuning['parent_checkpoint']} "
            f"(parent step {fine_tuning['parent_step']}); fresh optimizer, "
            "scheduler, selection history and step count; no energy shift",
            flush=True,
        )

    print(
        f"selected structures: {len(prepared.samples)} "
        f"({len(prepared.train_samples)} train, "
        f"{len(prepared.validation_samples)} validation, "
        f"{len(prepared.test_samples)} held-out test)",
        flush=True,
    )
    initial_evaluation_start = perf_counter()
    baseline_validation = evaluate_frozen_baseline(
        model,
        prepared.train_samples,
        prepared.validation_samples,
        prepared.test_samples,
        configuration,
    )
    initial_evaluation_seconds = elapsed_since(initial_evaluation_start, device)

    def save_best_checkpoint(
        current_model: MACEFNOResidual,
        current_result: OptimizationResult,
    ) -> None:
        _save_checkpoint(
            args,
            configuration,
            prepared,
            current_result,
            spectral_monitor,
            current_model,
            write_configuration=False,
            fingerprints=fingerprints,
            fine_tuning=fine_tuning,
        )

    best_checkpoint_callback = (
        save_best_checkpoint if configuration.runtime.checkpoint is not None else None
    )

    def save_last_checkpoint(state: dict) -> None:
        state["input_fingerprints"] = fingerprints
        state["reference_cell"] = (
            getattr(model, "reference_cell", prepared.reference_cell).detach().cpu()
        )
        if fine_tuning is not None:
            state["fine_tuning"] = fine_tuning
        state["training_configuration"] = resolved_configuration(
            args, last_checkpoint=configuration.runtime.last_checkpoint,
        )
        save_training_checkpoint(configuration.runtime.last_checkpoint, state)
        print(
            f"latest training state: {configuration.runtime.last_checkpoint} "
            f"(step={state['completed_steps']})",
            flush=True,
        )

    def save_energy_checkpoint(current_result: OptimizationResult) -> None:
        if current_result.energy_selection is not None:
            _save_checkpoint(
                args, configuration, prepared, current_result, None, model,
                write_configuration=False,
                energy_selection=current_result.energy_selection,
                fingerprints=fingerprints,
                fine_tuning=fine_tuning,
            )

    optimization_start = perf_counter()
    result = optimize_residual(
        model,
        prepared.train_samples,
        prepared.validation_samples,
        baseline_validation,
        configuration,
        device=device,
        spectral_monitor=spectral_monitor,
        best_checkpoint_callback=best_checkpoint_callback,
        resume_state=resume_state,
        last_checkpoint_callback=(
            save_last_checkpoint if configuration.runtime.last_checkpoint is not None else None
        ),
        energy_checkpoint_callback=save_energy_checkpoint,
    )
    optimization_seconds = elapsed_since(optimization_start, device)

    final_evaluation_start = perf_counter()
    selected_metrics = evaluate_selected_model(
        model,
        prepared.train_samples,
        prepared.validation_samples,
        prepared.test_samples,
        configuration,
    )
    if spectral_monitor is not None:
        spectral_monitor.evaluate_selected(
            model,
            step=result.best_step,
            validation_objective=result.best_validation_objective,
        )
    _save_checkpoint(
        args,
        configuration,
        prepared,
        result,
        spectral_monitor,
        model,
        evaluation_metrics=selected_metrics,
        fingerprints=fingerprints,
        fine_tuning=fine_tuning,
    )

    if result.energy_selection is not None:
        primary_residual = residual_state_dict(model)
        primary_mace = (
            mace_state_dict(model)
            if configuration.optimization.mace_training == "joint" else None
        )
        state = result.energy_selection["candidate"]["state"]
        try:
            load_residual_state_dict(model, state["residual_state_dict"])
            if state["mace_state_dict"] is not None:
                load_mace_state_dict(model, state["mace_state_dict"])
            energy_metrics = evaluate_selected_model(
                model, prepared.train_samples, prepared.validation_samples,
                prepared.test_samples, configuration, label="energy-selected",
            )
            _save_checkpoint(
                args, configuration, prepared, result, None, model,
                energy_selection=result.energy_selection,
                evaluation_metrics=energy_metrics,
                fingerprints=fingerprints,
                fine_tuning=fine_tuning,
            )
        finally:
            load_residual_state_dict(model, primary_residual)
            if primary_mace is not None:
                load_mace_state_dict(model, primary_mace)
    final_evaluation_seconds = elapsed_since(final_evaluation_start, device)

    total_seconds = elapsed_since(total_start, device)
    print(
        "timings: "
        f"setup={setup_seconds:.2f}s, "
        f"baseline-target-cache={target_cache_seconds:.2f}s, "
        f"initial-evaluation={initial_evaluation_seconds:.2f}s, "
        f"optimization+validation={optimization_seconds:.2f}s, "
        f"final-evaluation={final_evaluation_seconds:.2f}s, "
        f"total={total_seconds:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
