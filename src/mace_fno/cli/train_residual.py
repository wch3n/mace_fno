"""Train a frozen or jointly optimized MACE-FNO model.

The input should be an extended XYZ file containing reference total energies
and forces. Validation data are never used for gradients, and a separate test
file is evaluated only before and after optimization.
"""

from __future__ import annotations

import argparse
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
) -> None:
    """Save one self-describing checkpoint and its resolved YAML sidecar."""
    checkpoint = configuration.runtime.checkpoint
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
    save_training_checkpoint(checkpoint, payload)
    print(
        f"best checkpoint: {checkpoint} "
        f"(step={result.best_step}, "
        f"validation_objective={result.best_validation_objective:.6e})",
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
        prepared.reference_cell,
        device=device,
        dtype=dtype,
    )
    setup_seconds = elapsed_since(setup_start, device)

    target_cache_start = perf_counter()
    cache_frozen_targets(model, prepared, configuration, device=device)
    target_cache_seconds = elapsed_since(target_cache_start, device)

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
        )

    best_checkpoint_callback = (
        save_best_checkpoint if configuration.runtime.checkpoint is not None else None
    )

    def save_last_checkpoint(state: dict) -> None:
        state["input_fingerprints"] = fingerprints
        state["training_configuration"] = resolved_configuration(
            args, last_checkpoint=configuration.runtime.last_checkpoint,
        )
        save_training_checkpoint(configuration.runtime.last_checkpoint, state)
        print(
            f"latest training state: {configuration.runtime.last_checkpoint} "
            f"(step={state['completed_steps']})",
            flush=True,
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
    )
    optimization_seconds = elapsed_since(optimization_start, device)

    final_evaluation_start = perf_counter()
    evaluate_selected_model(
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
    final_evaluation_seconds = elapsed_since(final_evaluation_start, device)

    _save_checkpoint(
        args,
        configuration,
        prepared,
        result,
        spectral_monitor,
        model,
    )

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
