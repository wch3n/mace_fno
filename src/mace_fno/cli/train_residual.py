"""Train a frozen-MACE plus 2D, slab, or periodic-3D FNO residual.

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


def _save_checkpoint(
    args: argparse.Namespace,
    configuration: TrainingConfig,
    prepared: PreparedData,
    result: OptimizationResult,
    spectral_monitor: SpectralMonitor | None,
    model: MACEFNOResidual,
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
    print(f"checkpoint: {checkpoint}")
    configuration_path = checkpoint.with_suffix(".config.yaml")
    write_resolved_configuration(configuration_path, effective_configuration)
    print(f"resolved configuration: {configuration_path}")


def main() -> None:
    """Execute one configured frozen-MACE residual training run."""
    total_start = perf_counter()
    args = parse_arguments()
    configuration = TrainingConfig.from_namespace(args)
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

    optimization_start = perf_counter()
    result = optimize_residual(
        model,
        prepared.train_samples,
        prepared.validation_samples,
        baseline_validation,
        configuration,
        device=device,
        spectral_monitor=spectral_monitor,
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
        f"frozen-target-cache={target_cache_seconds:.2f}s, "
        f"initial-evaluation={initial_evaluation_seconds:.2f}s, "
        f"optimization+validation={optimization_seconds:.2f}s, "
        f"final-evaluation={final_evaluation_seconds:.2f}s, "
        f"total={total_seconds:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
