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
    CHECKPOINT_FORMAT_VERSION,
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
    residual_state_dict,
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

    data = configuration.data
    model_config = configuration.model
    optimization = configuration.optimization
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
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

    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "training_configuration": effective_configuration,
            "residual_state_dict": residual_state_dict(model),
            "mace_model": str(data.mace_model),
            "mace_head": data.head,
            "train_file": str(data.train_file),
            "validation_file": (
                str(data.validation_file) if data.validation_file is not None else None
            ),
            "validation_indices_file": (
                str(data.validation_indices_file)
                if data.validation_indices_file is not None
                else None
            ),
            "test_file": str(data.test_file) if data.test_file is not None else None,
            "grid_shape": (model_config.grid, model_config.grid),
            "n_modes": (
                (
                    model_config.resolved_z_modes,
                    model_config.modes,
                    model_config.modes,
                )
                if model_config.spatial_scheme == "3d"
                else (model_config.modes, model_config.modes)
            ),
            "spatial_scheme": model_config.spatial_scheme,
            "cell_mode": model_config.cell_mode,
            "z_grid_size": model_config.z_grid or None,
            "z_extent": (
                model_config.z_extent if model_config.spatial_scheme == "2.5d" else None
            ),
            "z_center": (
                model_config.z_center if model_config.spatial_scheme == "2.5d" else None
            ),
            "lateral_interlacing": (
                model_config.lateral_interlacing
                if model_config.spatial_scheme == "2.5d"
                else 1
            ),
            "volume_interlacing": (
                model_config.volume_interlacing
                if model_config.spatial_scheme == "3d"
                else 1
            ),
            "interlacing_training": (
                model_config.interlacing_training
                if model_config.spatial_scheme == "3d"
                else "full"
            ),
            "planar_symmetry": (
                model_config.planar_symmetry
                if model_config.spatial_scheme == "2.5d"
                else "none"
            ),
            "spectral_symmetry": (
                model_config.spectral_symmetry
                if model_config.spatial_scheme == "3d"
                else "none"
            ),
            "spectral_groups": (
                model_config.spectral_groups
                if model_config.spatial_scheme == "3d"
                else 1
            ),
            "metric_hidden_channels": (
                model_config.metric_hidden_channels
                if model_config.spatial_scheme == "3d"
                else 16
            ),
            "z_kernel_size": (
                model_config.z_kernel_size
                if model_config.spatial_scheme == "2.5d"
                else None
            ),
            "z_mixing": (
                "spectral"
                if model_config.spatial_scheme == "2.5d"
                and model_config.architecture == "linear"
                else model_config.z_mixing
                if model_config.spatial_scheme == "2.5d"
                else None
            ),
            "channels": model_config.channels,
            "source_hidden_channels": model_config.source_hidden_channels,
            "fno_hidden_channels": model_config.fno_hidden_channels,
            "fno_layers": model_config.fno_layers,
            "architecture": model_config.architecture,
            "reference_cell": prepared.reference_cell.detach().cpu(),
            "num_atoms": data.num_atoms,
            "validation_fraction": data.validation_fraction,
            "skip_cell_mismatch": data.skip_cell_mismatch,
            "accumulation_steps": optimization.accumulation_steps,
            "batch_size": optimization.batch_size,
            "evaluation_batch_size": optimization.evaluation_batch_size,
            "evaluation_scope": optimization.evaluation_scope,
            "steps": optimization.steps,
            "completed_steps": result.completed_steps,
            "stopped_early": result.stopped_early,
            "eval_interval": optimization.eval_interval,
            "learning_rate": optimization.learning_rate,
            "output_initialization_scale": optimization.output_initialization_scale,
            "output_warmup_steps": optimization.output_warmup_steps,
            "output_warmup_learning_rate": result.warmup_learning_rate,
            "final_learning_rate": result.final_learning_rate,
            "lr_scheduler": optimization.lr_scheduler,
            "lr_decay_factor": optimization.lr_decay_factor,
            "lr_patience_evals": optimization.lr_patience_evals,
            "minimum_learning_rate": optimization.minimum_learning_rate,
            "early_stopping_patience_evals": (
                optimization.early_stopping_patience_evals
            ),
            "energy_weight": optimization.energy_weight,
            "force_weight": optimization.force_weight,
            "energy_scale": optimization.energy_scale,
            "force_scale": optimization.force_scale,
            "dtype": configuration.runtime.dtype,
            "train_cache": str(data.train_cache) if data.train_cache else None,
            "validation_cache": (
                str(data.validation_cache) if data.validation_cache else None
            ),
            "test_cache": str(data.test_cache) if data.test_cache else None,
            "best_step": result.best_step,
            "best_validation_objective": result.best_validation_objective,
            "spectral_diagnostic": spectral_record,
            "seed": configuration.runtime.seed,
        },
        checkpoint,
    )
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
