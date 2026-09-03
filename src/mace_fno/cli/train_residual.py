"""Train frozen MACE plus a 2D, hybrid 2.5D, or periodic 3D FNO residual.

The input should be an extended XYZ file containing reference total energies
and forces. Validation data are never used for gradients, and a separate test
file is evaluated only before and after optimization. Training and evaluation
use true multi-configuration batches for MACE and the particle-mesh residual.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from mace_fno import MACEFNOResidual, energy_force_loss
from mace_fno.cli.config import parse_arguments
from mace_fno.cli.yaml_config import (
    resolved_configuration,
    write_resolved_configuration,
)
from mace_fno.training import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingConfig,
    amplitude_convergence_diagnostic,
    choose_device,
    collate_samples,
    configure_output_projection_warmup,
    elapsed_since,
    ensure_frozen_residual_targets,
    evaluate,
    finish_output_projection_warmup,
    initialize_scaled_residual_output,
    initialize_zero_residual,
    load_or_create_samples,
    load_residual_state_dict,
    low_k_response_diagnostic,
    print_metrics,
    residual_state_dict,
    save_sample_cache,
    split_samples,
    validation_objective,
)


def _print_low_k_diagnostic(label: str, report: dict[str, Any]) -> None:
    """Print the compact, validation-only infrared response summary."""
    diagnostic_kind = report.get("diagnostic_kind")
    planar = diagnostic_kind == "planar_2d"
    slab = diagnostic_kind == "slab_2p5d"
    surface = planar or slab
    if planar:
        fit = report["low_k_planar_response_fit"]
    elif slab:
        fit = report["low_k_monopole_response_fit"]
    else:
        fit = report["low_k_dominant_eigenvalue_fit"]
    if fit is None:
        summary = "no positive leading response on at least two low-k shells"
    else:
        reference_label = "R2_1/k" if surface else "R2_1/k2"
        reference_r2 = (
            fit["reference_power_log_r2"]
            if surface
            else fit["coulomb_p2_log_r2"]
        )
        summary = (
            f"p={fit['free_power_exponent_p']:.3f} | "
            f"R2_free={fit['free_log_r2']:.3f} | "
            f"{reference_label}={reference_r2:.3f} | "
            f"points={fit['points']}"
        )
        if slab and report["mean_low_k_coulomb_template_relative_error"] is not None:
            summary += (
                " | z_template_relerr="
                f"{report['mean_low_k_coulomb_template_relative_error']:.3f}"
            )
        if not surface:
            tensor_fit = report.get("pooled_anisotropic_inverse_quadratic_fit")
            if tensor_fit is not None and tensor_fit["log_response_r2"] is not None:
                summary += f" | tensor_R2={tensor_fit['log_response_r2']:.3f}"
    print(f"{label:<32} | {summary}", flush=True)


def _print_amplitude_diagnostic(label: str, report: dict[str, Any]) -> None:
    """Print the compact finite-amplitude convergence result."""
    summary = report["summary"]
    stable = "yes" if summary["curvature_stable_within_tolerance"] else "no"
    median_span = summary["median_mode_relative_span"]
    maximum_span = summary["maximum_mode_relative_span"]
    exponent_range = summary["free_power_exponent_range"]
    fields = [
        f"stable={stable}",
        f"matched_modes={summary['matched_modes']}",
    ]
    if median_span is not None:
        fields.append(f"median_span={median_span:.3e}")
    if maximum_span is not None:
        fields.append(f"max_span={maximum_span:.3e}")
    if exponent_range is not None:
        fields.append(f"p_range={exponent_range:.3e}")
    print(f"{label:<32} | {' | '.join(fields)}", flush=True)


def _compact_amplitude_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep convergence evidence without duplicating every full spectral run."""
    return {key: value for key, value in report.items() if key != "runs"}


def _write_low_k_history(
    output: Path | None,
    *,
    configuration: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    """Persist the diagnostic trace independently of the model checkpoint."""
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"configuration": configuration, "history": history}, indent=2
        )
        + "\n"
    )


def main() -> None:
    total_start = perf_counter()
    args = parse_arguments()
    configuration = TrainingConfig.from_namespace(args)
    spatial_scheme = configuration.model.spatial_scheme
    resolved_z_modes = configuration.model.resolved_z_modes
    evaluation_batch_size = configuration.optimization.evaluation_batch_size
    spectral_diagnostic_enabled = configuration.diagnostic.enabled
    spectral_diagnostic_output = configuration.diagnostic.output

    # Importing here keeps the base package usable without the optional MACE extra.
    from mace.calculators import MACECalculator

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    setup_start = perf_counter()
    calculator_kwargs = dict(
        model_paths=str(args.mace_model),
        device=str(device),
        # Load the checkpoint in float64 first so the exact atomic reference
        # energies remain available to the hybrid float32 correction. Graphs
        # are cast to the requested model dtype when each batch is collated.
        default_dtype="float64",
    )
    if args.head is not None:
        calculator_kwargs["head"] = args.head
    calculator = MACECalculator(**calculator_kwargs)
    if len(calculator.models) != 1:
        raise ValueError("the first residual experiment requires one MACE model")
    (
        samples,
        reference_cell,
        train_cache_metadata,
        train_cache_hit,
    ) = load_or_create_samples(
        calculator,
        args.train_file,
        args.energy_key,
        args.forces_key,
        dtype,
        args.num_atoms,
        args.allow_periodic_z,
        args.skip_cell_mismatch,
        args.mace_model,
        args.train_cache,
        args.rebuild_cache,
        spatial_scheme=spatial_scheme,
        cell_mode=args.cell_mode,
    )
    if args.validation_file is None:
        validation_indices = None
        if args.validation_indices_file is not None:
            validation_indices = [
                int(value) for value in args.validation_indices_file.read_text().split()
            ]
        train_samples, validation_samples = split_samples(
            samples,
            args.validation_fraction,
            args.seed + 2,
            validation_indices=validation_indices,
        )
    else:
        train_samples = samples
        (
            validation_samples,
            _,
            validation_cache_metadata,
            validation_cache_hit,
        ) = load_or_create_samples(
            calculator,
            args.validation_file,
            args.energy_key,
            args.forces_key,
            dtype,
            args.num_atoms,
            args.allow_periodic_z,
            args.skip_cell_mismatch,
            args.mace_model,
            args.validation_cache,
            args.rebuild_cache,
            reference_cell=reference_cell,
            spatial_scheme=spatial_scheme,
            cell_mode=args.cell_mode,
        )
    if args.validation_file is None:
        validation_cache_metadata = None
        validation_cache_hit = False
    test_samples: list[dict[str, Any]] = []
    if args.test_file is not None:
        (
            test_samples,
            _,
            test_cache_metadata,
            test_cache_hit,
        ) = load_or_create_samples(
            calculator,
            args.test_file,
            args.energy_key,
            args.forces_key,
            dtype,
            args.num_atoms,
            args.allow_periodic_z,
            args.skip_cell_mismatch,
            args.mace_model,
            args.test_cache,
            args.rebuild_cache,
            reference_cell=reference_cell,
            spatial_scheme=spatial_scheme,
            cell_mode=args.cell_mode,
        )
    else:
        test_cache_metadata = None
        test_cache_hit = False

    spectral_sample_indices: list[int] = []
    spectral_history: list[dict[str, Any]] = []
    spectral_configuration: dict[str, Any] | None = None
    spectral_validation_z_profiles = (
        1 if spatial_scheme == "2.5d" else args.spectral_diagnostic_z_profiles
    )
    spectral_final_amplitudes: list[float] = []
    if spectral_diagnostic_enabled:
        if not validation_samples:
            raise ValueError("the spectral diagnostic requires validation samples")
        selection_generator = torch.Generator().manual_seed(args.seed + 947)
        spectral_sample_indices = torch.randperm(
            len(validation_samples), generator=selection_generator
        )[: min(args.spectral_diagnostic_samples, len(validation_samples))].tolist()
        if args.spectral_diagnostic_depth == "deep":
            spectral_final_amplitudes = sorted(
                set(map(float, args.spectral_diagnostic_amplitudes))
                | {float(args.spectral_diagnostic_relative_amplitude)}
            )
        spectral_configuration = {
            "spatial_scheme": spatial_scheme,
            "sample_indices": spectral_sample_indices,
            "max_mode": args.spectral_diagnostic_max_mode,
            "fit_shells": args.spectral_diagnostic_fit_shells,
            "relative_amplitude": args.spectral_diagnostic_relative_amplitude,
            "field_batch_size": args.spectral_diagnostic_field_batch_size,
            "z_profiles": args.spectral_diagnostic_z_profiles,
            "validation_z_profiles": spectral_validation_z_profiles,
            "selected_z_profiles": args.spectral_diagnostic_z_profiles,
            "depth": args.spectral_diagnostic_depth,
            "selected_amplitudes": spectral_final_amplitudes,
            "relative_span_tolerance": (
                args.spectral_diagnostic_relative_span_tolerance
            ),
            "description": (
                "Geometry-aware latent-field curvature diagnostic. Routine "
                "validation uses the inexpensive probe; deeper settings apply "
                "only after restoring the selected checkpoint. It is not a loss."
            ),
        }

    model = MACEFNOResidual(
        calculator.models[0],
        (args.grid, args.grid),
        args.channels,
        (args.modes, args.modes),
        source_hidden_channels=args.source_hidden_channels,
        fno_hidden_channels=args.fno_hidden_channels,
        fno_layers=args.fno_layers,
        fno_architecture=args.architecture,
        spatial_scheme=spatial_scheme,
        z_grid_size=args.z_grid or None,
        fno_z_modes=resolved_z_modes if spatial_scheme == "3d" else None,
        z_extent=args.z_extent,
        z_center=args.z_center,
        fno_lateral_interlacing=args.lateral_interlacing,
        fno_volume_interlacing=args.volume_interlacing,
        fno_interlacing_training=args.interlacing_training,
        fno_z_kernel_size=args.z_kernel_size,
        fno_z_mixing=args.z_mixing,
        fno_planar_symmetry=args.planar_symmetry,
        fno_spectral_symmetry=args.spectral_symmetry,
        fno_spectral_groups=args.spectral_groups,
        fno_metric_hidden_channels=args.metric_hidden_channels,
        reference_cell=reference_cell,
        cell_mode=args.cell_mode,
    ).to(device=device, dtype=dtype)
    if args.output_initialization_scale:
        initialize_scaled_residual_output(model, args.output_initialization_scale)
    elif not args.random_residual_initialization:
        initialize_zero_residual(model)

    setup_seconds = elapsed_since(setup_start, device)
    target_cache_start = perf_counter()

    train_cache_changed = ensure_frozen_residual_targets(
        model,
        samples,
        device=device,
        batch_size=evaluation_batch_size,
    )
    if train_cache_changed or not train_cache_hit:
        save_sample_cache(
            args.train_cache,
            train_cache_metadata,
            samples,
            reference_cell,
        )
    if args.validation_file is not None:
        validation_cache_changed = ensure_frozen_residual_targets(
            model,
            validation_samples,
            device=device,
            batch_size=evaluation_batch_size,
        )
        if validation_cache_changed or not validation_cache_hit:
            save_sample_cache(
                args.validation_cache,
                validation_cache_metadata,
                validation_samples,
                reference_cell,
            )
    if args.test_file is not None:
        test_cache_changed = ensure_frozen_residual_targets(
            model,
            test_samples,
            device=device,
            batch_size=evaluation_batch_size,
        )
        if test_cache_changed or not test_cache_hit:
            save_sample_cache(
                args.test_cache,
                test_cache_metadata,
                test_samples,
                reference_cell,
            )
    target_cache_seconds = elapsed_since(target_cache_start, device)

    model.train()
    model_dtype = next(model.parameters()).dtype
    parameters, warmup_parameters = configure_output_projection_warmup(
        model,
        args.output_warmup_steps,
    )
    warmup_learning_rate = args.output_warmup_learning_rate or args.learning_rate
    optimizer = torch.optim.Adam(
        parameters,
        lr=(warmup_learning_rate if args.output_warmup_steps else args.learning_rate),
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_patience_evals,
            min_lr=args.minimum_learning_rate,
        )
        if args.lr_scheduler == "plateau"
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + 1)

    if args.output_warmup_steps:
        print(
            "output-projection warm-up: "
            f"steps={args.output_warmup_steps}, "
            f"learning_rate={warmup_learning_rate:.6e}, "
            f"active_parameters={sum(p.numel() for p in warmup_parameters)}/"
            f"{sum(p.numel() for p in parameters)}",
            flush=True,
        )

    print(
        f"selected structures: {len(samples)} "
        f"({len(train_samples)} train, {len(validation_samples)} validation, "
        f"{len(test_samples)} held-out test)",
        flush=True,
    )
    initial_evaluation_start = perf_counter()
    baseline_train = (
        evaluate(
            model,
            train_samples,
            baseline=True,
            batch_size=evaluation_batch_size,
        )
        if args.evaluation_scope == "all"
        else {}
    )
    baseline_validation = evaluate(
        model, validation_samples, baseline=True, batch_size=evaluation_batch_size
    )
    baseline_test = evaluate(
        model, test_samples, baseline=True, batch_size=evaluation_batch_size
    )
    print_metrics("frozen MACE train", baseline_train)
    print_metrics("frozen MACE validation", baseline_validation)
    print_metrics("frozen MACE held-out test", baseline_test)
    if baseline_train:
        energy_shift = -baseline_train["energy_bias"]
        print_metrics(
            "constant-offset validation",
            evaluate(
                model,
                validation_samples,
                baseline=True,
                energy_shift_per_atom=energy_shift,
                batch_size=evaluation_batch_size,
            ),
        )
        formula_shifts = {
            formula: -metrics["energy_bias"]
            for formula, metrics in baseline_train["by_formula"].items()
        }
        print_metrics(
            "formula-offset validation",
            evaluate(
                model,
                validation_samples,
                baseline=True,
                energy_shift_per_atom=formula_shifts,
                batch_size=evaluation_batch_size,
            ),
        )
        print_metrics(
            "constant-offset held-out test",
            evaluate(
                model,
                test_samples,
                baseline=True,
                energy_shift_per_atom=energy_shift,
                batch_size=evaluation_batch_size,
            ),
        )
        print_metrics(
            "formula-offset held-out test",
            evaluate(
                model,
                test_samples,
                baseline=True,
                energy_shift_per_atom=formula_shifts,
                batch_size=evaluation_batch_size,
            ),
        )
    initial_evaluation_seconds = elapsed_since(initial_evaluation_start, device)

    best_step = 0
    best_validation_objective = validation_objective(
        baseline_validation,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        energy_scale=args.energy_scale,
        force_scale=args.force_scale,
    )
    best_residual_state = residual_state_dict(model)
    print(
        f"validation objective at frozen baseline: " f"{best_validation_objective:.6e}",
        flush=True,
    )

    completed_steps = 0
    stopped_early = False
    evaluations_at_minimum_lr = 0
    optimization_start = perf_counter()
    for step in range(args.steps):
        if step == args.output_warmup_steps and args.output_warmup_steps:
            finish_output_projection_warmup(parameters)
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
            print(
                f"output-projection warm-up complete at step {step}: "
                f"unfroze {sum(p.numel() for p in parameters)} parameters, "
                f"learning_rate={previous_learning_rate:.6e} -> "
                f"{args.learning_rate:.6e}",
                flush=True,
            )
        optimizer.zero_grad(set_to_none=True)
        accumulated = {"loss": 0.0, "energy": 0.0, "forces": 0.0}
        for _ in range(args.accumulation_steps):
            sample_indices = torch.randint(
                len(train_samples),
                (args.batch_size,),
                generator=generator,
            ).tolist()
            sample_batch = [train_samples[index] for index in sample_indices]
            graph, target_energy, target_forces = collate_samples(
                sample_batch, device, model_dtype
            )
            del target_energy, target_forces
            target_residual_energy = torch.cat(
                [sample["residual_energy"] for sample in sample_batch]
            ).to(device=device)
            target_residual_forces = torch.cat(
                [sample["residual_forces"] for sample in sample_batch]
            ).to(device=device)
            output = model(
                graph,
                training=True,
                compute_force=False,
                compute_residual_force=True,
            )
            target_residual_energy = target_residual_energy.to(
                dtype=output["residual_energy"].dtype
            )
            target_residual_forces = target_residual_forces.to(
                dtype=output["residual_forces"].dtype
            )
            terms = energy_force_loss(
                output["residual_energy"],
                output["residual_forces"],
                target_residual_energy,
                target_residual_forces,
                graph["batch"],
                energy_weight=args.energy_weight,
                force_weight=args.force_weight,
                energy_scale=args.energy_scale,
                force_scale=args.force_scale,
            )
            (terms["loss"] / args.accumulation_steps).backward()
            for name in accumulated:
                accumulated[name] += terms[name].item() / args.accumulation_steps
        optimizer.step()
        completed_steps = step + 1

        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"step {step + 1:5d}/{args.steps}: "
                f"loss={accumulated['loss']:.6e}, "
                f"energy={accumulated['energy']:.6e}, "
                f"forces={accumulated['forces']:.6e}",
                flush=True,
            )
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.steps:
            validation_metrics = evaluate(
                model, validation_samples, batch_size=evaluation_batch_size
            )
            print_metrics(f"validation step {step + 1}", validation_metrics)
            score = validation_objective(
                validation_metrics,
                energy_weight=args.energy_weight,
                force_weight=args.force_weight,
                energy_scale=args.energy_scale,
                force_scale=args.force_scale,
            )
            print(
                f"validation objective step {step + 1}: {score:.6e}",
                flush=True,
            )
            if spectral_diagnostic_enabled:
                spectral_report = low_k_response_diagnostic(
                    model,
                    validation_samples,
                    sample_indices=spectral_sample_indices,
                    max_mode=args.spectral_diagnostic_max_mode,
                    fit_shells=args.spectral_diagnostic_fit_shells,
                    relative_amplitude=args.spectral_diagnostic_relative_amplitude,
                    field_batch_size=args.spectral_diagnostic_field_batch_size,
                    z_profiles=spectral_validation_z_profiles,
                )
                spectral_report.update(
                    {
                        "step": step + 1,
                        "stage": "validation",
                        "validation_objective": score,
                    }
                )
                spectral_history.append(spectral_report)
                _print_low_k_diagnostic(
                    f"low-k response step {step + 1}", spectral_report
                )
                assert spectral_configuration is not None
                _write_low_k_history(
                    spectral_diagnostic_output,
                    configuration=spectral_configuration,
                    history=spectral_history,
                )
            improved = score < best_validation_objective
            if improved:
                best_step = step + 1
                best_validation_objective = score
                best_residual_state = residual_state_dict(model)
                print(f"new best validation step: {best_step}", flush=True)
            if scheduler is not None and step + 1 > args.output_warmup_steps:
                previous_learning_rate = optimizer.param_groups[0]["lr"]
                scheduler.step(score)
                current_learning_rate = optimizer.param_groups[0]["lr"]
                if current_learning_rate != previous_learning_rate:
                    print(
                        f"learning rate step {step + 1}: "
                        f"{previous_learning_rate:.6e} -> "
                        f"{current_learning_rate:.6e}",
                        flush=True,
                    )
                at_minimum_learning_rate = current_learning_rate <= (
                    args.minimum_learning_rate * (1.0 + 16.0 * np.finfo(np.float64).eps)
                )
                if at_minimum_learning_rate:
                    evaluations_at_minimum_lr = (
                        0 if improved else evaluations_at_minimum_lr + 1
                    )
                else:
                    evaluations_at_minimum_lr = 0
                if (
                    args.early_stopping_patience_evals
                    and evaluations_at_minimum_lr >= args.early_stopping_patience_evals
                ):
                    stopped_early = True
                    print(
                        f"early stopping at step {step + 1}: no validation "
                        f"improvement in {evaluations_at_minimum_lr} checks at "
                        f"minimum learning rate {current_learning_rate:.6e}",
                        flush=True,
                    )
                    break

    optimization_seconds = elapsed_since(optimization_start, device)

    load_residual_state_dict(model, best_residual_state)
    print(
        f"restored best validation step {best_step} "
        f"(objective={best_validation_objective:.6e})",
        flush=True,
    )
    final_evaluation_start = perf_counter()
    if args.evaluation_scope == "all":
        print_metrics(
            "selected train",
            evaluate(model, train_samples, batch_size=evaluation_batch_size),
        )
    print_metrics(
        "selected validation",
        evaluate(model, validation_samples, batch_size=evaluation_batch_size),
    )
    print_metrics(
        "selected held-out test",
        evaluate(model, test_samples, batch_size=evaluation_batch_size),
    )
    if spectral_diagnostic_enabled:
        amplitude_report: dict[str, Any] | None = None
        if args.spectral_diagnostic_depth == "deep":
            amplitude_report = amplitude_convergence_diagnostic(
                model,
                validation_samples,
                sample_indices=spectral_sample_indices,
                relative_amplitudes=spectral_final_amplitudes,
                relative_span_tolerance=(
                    args.spectral_diagnostic_relative_span_tolerance
                ),
                max_mode=args.spectral_diagnostic_max_mode,
                fit_shells=args.spectral_diagnostic_fit_shells,
                field_batch_size=args.spectral_diagnostic_field_batch_size,
                z_profiles=args.spectral_diagnostic_z_profiles,
            )
            selected_spectral_report = next(
                report
                for report in amplitude_report["runs"]
                if report["relative_amplitude"]
                == float(args.spectral_diagnostic_relative_amplitude)
            )
        else:
            selected_spectral_report = low_k_response_diagnostic(
                model,
                validation_samples,
                sample_indices=spectral_sample_indices,
                max_mode=args.spectral_diagnostic_max_mode,
                fit_shells=args.spectral_diagnostic_fit_shells,
                relative_amplitude=args.spectral_diagnostic_relative_amplitude,
                field_batch_size=args.spectral_diagnostic_field_batch_size,
                z_profiles=args.spectral_diagnostic_z_profiles,
            )
        selected_spectral_report.update(
            {
                "step": best_step,
                "stage": "selected",
                "validation_objective": best_validation_objective,
            }
        )
        spectral_history.append(selected_spectral_report)
        _print_low_k_diagnostic("low-k response selected", selected_spectral_report)
        if amplitude_report is not None:
            compact_amplitude_report = _compact_amplitude_report(amplitude_report)
            compact_amplitude_report.update(
                {
                    "step": best_step,
                    "stage": "selected_amplitude_convergence",
                    "validation_objective": best_validation_objective,
                }
            )
            spectral_history.append(compact_amplitude_report)
            _print_amplitude_diagnostic(
                "amplitude convergence selected", amplitude_report
            )
        assert spectral_configuration is not None
        _write_low_k_history(
            spectral_diagnostic_output,
            configuration=spectral_configuration,
            history=spectral_history,
        )
    final_evaluation_seconds = elapsed_since(final_evaluation_start, device)

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        effective_configuration = resolved_configuration(
            args,
            spatial_scheme=spatial_scheme,
            z_modes=resolved_z_modes if spatial_scheme == "3d" else 0,
            evaluation_batch_size=evaluation_batch_size,
            output_warmup_learning_rate=warmup_learning_rate,
        )
        torch.save(
            {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "training_configuration": effective_configuration,
                "residual_state_dict": residual_state_dict(model),
                "mace_model": str(args.mace_model),
                "mace_head": args.head,
                "train_file": str(args.train_file),
                "validation_file": (
                    str(args.validation_file)
                    if args.validation_file is not None
                    else None
                ),
                "validation_indices_file": (
                    str(args.validation_indices_file)
                    if args.validation_indices_file is not None
                    else None
                ),
                "test_file": (
                    str(args.test_file) if args.test_file is not None else None
                ),
                "grid_shape": (args.grid, args.grid),
                "n_modes": (
                    (resolved_z_modes, args.modes, args.modes)
                    if spatial_scheme == "3d"
                    else (args.modes, args.modes)
                ),
                "spatial_scheme": spatial_scheme,
                "cell_mode": args.cell_mode,
                "z_grid_size": args.z_grid or None,
                "z_extent": args.z_extent if spatial_scheme == "2.5d" else None,
                "z_center": args.z_center if spatial_scheme == "2.5d" else None,
                "lateral_interlacing": (
                    args.lateral_interlacing if spatial_scheme == "2.5d" else 1
                ),
                "volume_interlacing": (
                    args.volume_interlacing if spatial_scheme == "3d" else 1
                ),
                "interlacing_training": (
                    args.interlacing_training if spatial_scheme == "3d" else "full"
                ),
                "planar_symmetry": (
                    args.planar_symmetry if spatial_scheme == "2.5d" else "none"
                ),
                "spectral_symmetry": (
                    args.spectral_symmetry if spatial_scheme == "3d" else "none"
                ),
                "spectral_groups": (
                    args.spectral_groups if spatial_scheme == "3d" else 1
                ),
                "metric_hidden_channels": (
                    args.metric_hidden_channels if spatial_scheme == "3d" else 16
                ),
                "z_kernel_size": (
                    args.z_kernel_size if spatial_scheme == "2.5d" else None
                ),
                "z_mixing": (
                    "spectral"
                    if spatial_scheme == "2.5d" and args.architecture == "linear"
                    else args.z_mixing if spatial_scheme == "2.5d" else None
                ),
                "channels": args.channels,
                "source_hidden_channels": args.source_hidden_channels,
                "fno_hidden_channels": args.fno_hidden_channels,
                "fno_layers": args.fno_layers,
                "architecture": args.architecture,
                "reference_cell": reference_cell.detach().cpu(),
                "num_atoms": args.num_atoms,
                "validation_fraction": args.validation_fraction,
                "skip_cell_mismatch": args.skip_cell_mismatch,
                "accumulation_steps": args.accumulation_steps,
                "batch_size": args.batch_size,
                "evaluation_batch_size": evaluation_batch_size,
                "evaluation_scope": args.evaluation_scope,
                "steps": args.steps,
                "completed_steps": completed_steps,
                "stopped_early": stopped_early,
                "eval_interval": args.eval_interval,
                "learning_rate": args.learning_rate,
                "output_initialization_scale": args.output_initialization_scale,
                "output_warmup_steps": args.output_warmup_steps,
                "output_warmup_learning_rate": warmup_learning_rate,
                "final_learning_rate": optimizer.param_groups[0]["lr"],
                "lr_scheduler": args.lr_scheduler,
                "lr_decay_factor": args.lr_decay_factor,
                "lr_patience_evals": args.lr_patience_evals,
                "minimum_learning_rate": args.minimum_learning_rate,
                "early_stopping_patience_evals": (args.early_stopping_patience_evals),
                "energy_weight": args.energy_weight,
                "force_weight": args.force_weight,
                "energy_scale": args.energy_scale,
                "force_scale": args.force_scale,
                "dtype": args.dtype,
                "train_cache": (
                    str(args.train_cache) if args.train_cache is not None else None
                ),
                "validation_cache": (
                    str(args.validation_cache)
                    if args.validation_cache is not None
                    else None
                ),
                "test_cache": (
                    str(args.test_cache) if args.test_cache is not None else None
                ),
                "best_step": best_step,
                "best_validation_objective": best_validation_objective,
                "spectral_diagnostic": (
                    {
                        "configuration": spectral_configuration,
                        "history": spectral_history,
                        "output": (
                            str(spectral_diagnostic_output)
                            if spectral_diagnostic_output is not None
                            else None
                        ),
                    }
                    if spectral_diagnostic_enabled
                    else None
                ),
                "seed": args.seed,
            },
            args.checkpoint,
        )
        print(f"checkpoint: {args.checkpoint}")
        configuration_path = args.checkpoint.with_suffix(".config.yaml")
        write_resolved_configuration(configuration_path, effective_configuration)
        print(f"resolved configuration: {configuration_path}")

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
