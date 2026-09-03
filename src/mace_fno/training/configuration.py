"""Typed, validated configuration for frozen-MACE residual training."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Input data, labels, and preprocessing-cache settings."""

    mace_model: Path
    train_file: Path
    validation_file: Path | None
    validation_indices_file: Path | None
    test_file: Path | None
    train_cache: Path | None
    validation_cache: Path | None
    test_cache: Path | None
    rebuild_cache: bool
    energy_key: str
    forces_key: str
    head: str | None
    num_atoms: int | None
    allow_periodic_z: bool
    skip_cell_mismatch: bool
    validation_fraction: float

    def validate(self) -> None:
        if (
            self.validation_file is not None
            and self.validation_indices_file is not None
        ):
            raise ValueError(
                "--validation-file and --validation-indices-file are mutually exclusive"
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must satisfy 0 <= fraction < 1")


@dataclass(frozen=True)
class ModelConfig:
    """Resolved mesh, operator, symmetry, and channel settings."""

    grid: int
    modes: int
    spatial_scheme: str
    cell_mode: str
    z_grid: int
    z_modes: int
    z_extent: float | None
    z_center: str
    lateral_interlacing: int
    volume_interlacing: int
    interlacing_training: str
    planar_symmetry: str
    spectral_symmetry: str
    spectral_groups: int
    metric_hidden_channels: int
    z_kernel_size: int
    z_mixing: str
    channels: int
    source_hidden_channels: int
    fno_hidden_channels: int
    fno_layers: int
    architecture: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ModelConfig:
        scheme = str(values["spatial_scheme"])
        if scheme == "auto":
            scheme = "2.5d" if int(values["z_grid"]) else "2d"
        return cls(
            grid=int(values["grid"]),
            modes=int(values["modes"]),
            spatial_scheme=scheme,
            cell_mode=str(values["cell_mode"]),
            z_grid=int(values["z_grid"]),
            z_modes=int(values["z_modes"]),
            z_extent=values["z_extent"],
            z_center=str(values["z_center"]),
            lateral_interlacing=int(values["lateral_interlacing"]),
            volume_interlacing=int(values["volume_interlacing"]),
            interlacing_training=str(values["interlacing_training"]),
            planar_symmetry=str(values["planar_symmetry"]),
            spectral_symmetry=str(values["spectral_symmetry"]),
            spectral_groups=int(values["spectral_groups"]),
            metric_hidden_channels=int(values["metric_hidden_channels"]),
            z_kernel_size=int(values["z_kernel_size"]),
            z_mixing=str(values["z_mixing"]),
            channels=int(values["channels"]),
            source_hidden_channels=int(values["source_hidden_channels"]),
            fno_hidden_channels=int(values["fno_hidden_channels"]),
            fno_layers=int(values["fno_layers"]),
            architecture=str(values["architecture"]),
        )

    def validate(self) -> None:
        if min(self.grid, self.modes, self.channels) < 1:
            raise ValueError("grid, modes, and channels must be positive")
        if 2 * self.modes > self.grid:
            raise ValueError("require 2*modes <= grid")
        if self.z_grid < 0 or self.z_modes < 0:
            raise ValueError("z_grid and z_modes must be non-negative")
        if self.spectral_groups < 1:
            raise ValueError("spectral_groups must be positive")
        if self.metric_hidden_channels < 1:
            raise ValueError("metric_hidden_channels must be positive")
        if self.spectral_symmetry == "none" and self.spectral_groups != 1:
            raise ValueError("--spectral-groups applies only with metric-aware EqGINO")
        if self.interlacing_training == "random" and self.volume_interlacing != 2:
            raise ValueError(
                "--interlacing-training random requires --volume-interlacing 2"
            )
        if self.cell_mode != "fixed" and self.spatial_scheme != "3d":
            raise ValueError("variable --cell-mode requires --spatial-scheme 3d")
        if self.cell_mode != "fixed" and self.architecture != "nonlinear":
            raise ValueError("variable --cell-mode requires --architecture nonlinear")

        if self.spatial_scheme == "2d":
            self._validate_planar()
        elif self.spatial_scheme == "2.5d":
            self._validate_slab()
        elif self.spatial_scheme == "3d":
            self._validate_periodic_3d()
        else:
            raise ValueError(f"unsupported spatial scheme {self.spatial_scheme!r}")

    def _validate_planar(self) -> None:
        if self.z_grid:
            raise ValueError("--z-grid is incompatible with --spatial-scheme 2d")
        if self.z_extent is not None:
            raise ValueError("--z-extent requires the 2.5D scheme")
        if self.z_modes:
            raise ValueError("--z-modes applies only to the 3D scheme")
        if self.z_mixing != "local":
            raise ValueError("--z-mixing global requires the 2.5D scheme")
        if self.lateral_interlacing != 1:
            raise ValueError("--lateral-interlacing applies only to the 2.5D scheme")
        if self.volume_interlacing != 1:
            raise ValueError("--volume-interlacing applies only to the 3D scheme")
        if self.planar_symmetry != "none":
            raise ValueError("--planar-symmetry applies only to the 2.5D scheme")
        if self.spectral_symmetry != "none":
            raise ValueError("--spectral-symmetry applies only to the 3D scheme")

    def _validate_slab(self) -> None:
        if self.z_grid < 4:
            raise ValueError("the 2.5D scheme requires z_grid >= 4")
        if self.z_extent is None or self.z_extent <= 0:
            raise ValueError("the 2.5D scheme requires a positive --z-extent")
        if self.z_modes:
            raise ValueError("--z-modes applies only to the 3D scheme")
        if self.spectral_symmetry != "none":
            raise ValueError("--spectral-symmetry applies only to the 3D scheme")
        if self.volume_interlacing != 1:
            raise ValueError("--volume-interlacing applies only to the 3D scheme")
        if self.z_mixing == "local" and (
            self.z_kernel_size < 1 or self.z_kernel_size % 2 == 0
        ):
            raise ValueError("z_kernel_size must be a positive odd integer")

    def _validate_periodic_3d(self) -> None:
        if self.z_extent is not None:
            raise ValueError("--z-extent is incompatible with the periodic 3D scheme")
        if self.z_grid < 4:
            raise ValueError("the 3D scheme requires z_grid >= 4")
        if 2 * self.resolved_z_modes > self.z_grid:
            raise ValueError("the 3D scheme requires 2*z_modes <= z_grid")
        if self.z_mixing != "local":
            raise ValueError("--z-mixing applies only to the 2.5D scheme")
        if self.lateral_interlacing != 1:
            raise ValueError("--lateral-interlacing applies only to the 2.5D scheme")
        if self.planar_symmetry != "none":
            raise ValueError("--planar-symmetry applies only to the 2.5D scheme")
        if self.spectral_symmetry == "metric_eqgino":
            grouped_channels = (
                self.channels
                if self.architecture == "linear"
                else self.fno_hidden_channels
            )
            if grouped_channels % self.spectral_groups:
                raise ValueError(
                    "metric-aware EqGINO channels must be divisible by spectral_groups"
                )

    @property
    def resolved_z_modes(self) -> int:
        """Return the explicit z-mode count or the planar mode count by default."""
        return self.z_modes or self.modes


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer, objective, batching, and model-selection settings."""

    steps: int
    learning_rate: float
    output_initialization_scale: float
    output_warmup_steps: int
    output_warmup_learning_rate: float
    lr_scheduler: str
    lr_decay_factor: float
    lr_patience_evals: int
    minimum_learning_rate: float
    early_stopping_patience_evals: int
    energy_weight: float
    force_weight: float
    energy_scale: float
    force_scale: float
    eval_interval: int
    evaluation_scope: str
    accumulation_steps: int
    batch_size: int
    evaluation_batch_size: int
    random_residual_initialization: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OptimizationConfig:
        evaluation_batch_size = int(
            values["evaluation_batch_size"] or values["batch_size"]
        )
        return cls(
            steps=int(values["steps"]),
            learning_rate=float(values["learning_rate"]),
            output_initialization_scale=float(values["output_initialization_scale"]),
            output_warmup_steps=int(values["output_warmup_steps"]),
            output_warmup_learning_rate=float(values["output_warmup_learning_rate"]),
            lr_scheduler=str(values["lr_scheduler"]),
            lr_decay_factor=float(values["lr_decay_factor"]),
            lr_patience_evals=int(values["lr_patience_evals"]),
            minimum_learning_rate=float(values["minimum_learning_rate"]),
            early_stopping_patience_evals=int(values["early_stopping_patience_evals"]),
            energy_weight=float(values["energy_weight"]),
            force_weight=float(values["force_weight"]),
            energy_scale=float(values["energy_scale"]),
            force_scale=float(values["force_scale"]),
            eval_interval=int(values["eval_interval"]),
            evaluation_scope=str(values["evaluation_scope"]),
            accumulation_steps=int(values["accumulation_steps"]),
            batch_size=int(values["batch_size"]),
            evaluation_batch_size=evaluation_batch_size,
            random_residual_initialization=bool(
                values["random_residual_initialization"]
            ),
        )

    def validate(self, architecture: str) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.output_warmup_steps < 0 or self.output_warmup_steps >= self.steps:
            raise ValueError("output_warmup_steps must satisfy 0 <= warm-up < steps")
        if self.output_warmup_learning_rate < 0.0:
            raise ValueError("output_warmup_learning_rate must be non-negative")
        if self.output_initialization_scale < 0.0:
            raise ValueError("output_initialization_scale must be non-negative")
        if self.output_warmup_steps and architecture != "nonlinear":
            raise ValueError(
                "output-projection warm-up requires --architecture nonlinear"
            )
        if self.output_initialization_scale and architecture != "nonlinear":
            raise ValueError(
                "scaled output initialization requires --architecture nonlinear"
            )
        if self.output_initialization_scale and self.random_residual_initialization:
            raise ValueError(
                "--output-initialization-scale and --random-residual-initialization "
                "are mutually exclusive"
            )
        if self.output_initialization_scale and self.output_warmup_steps:
            raise ValueError(
                "scaled output initialization and output-projection warm-up are "
                "mutually exclusive"
            )
        if min(self.eval_interval, self.accumulation_steps, self.batch_size) < 1:
            raise ValueError(
                "eval_interval, accumulation_steps, and batch_size must be positive"
            )
        if self.evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive")
        if not 0.0 < self.lr_decay_factor < 1.0:
            raise ValueError("lr_decay_factor must be between zero and one")
        if self.lr_patience_evals < 0 or self.early_stopping_patience_evals < 0:
            raise ValueError(
                "learning-rate and early-stopping patience must be non-negative"
            )
        if not 0.0 < self.minimum_learning_rate <= self.learning_rate:
            raise ValueError(
                "minimum_learning_rate must be positive and no larger than learning_rate"
            )
        if self.early_stopping_patience_evals and self.lr_scheduler != "plateau":
            raise ValueError("early stopping requires --lr-scheduler plateau")


@dataclass(frozen=True)
class DiagnosticConfig:
    """Geometry-aware spectral diagnostic settings."""

    samples: int
    max_mode: int
    fit_shells: int
    relative_amplitude: float
    field_batch_size: int
    z_profiles: int
    depth: str
    amplitudes: tuple[float, ...]
    relative_span_tolerance: float
    output: Path | None

    @property
    def enabled(self) -> bool:
        return self.samples > 0

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        checkpoint: Path | None,
    ) -> DiagnosticConfig:
        samples = int(values["spectral_diagnostic_samples"])
        output = values["spectral_diagnostic_output"]
        if samples > 0 and output is None and checkpoint is not None:
            output = checkpoint.with_name(f"{checkpoint.stem}_spectral_training.json")
        return cls(
            samples=samples,
            max_mode=int(values["spectral_diagnostic_max_mode"]),
            fit_shells=int(values["spectral_diagnostic_fit_shells"]),
            relative_amplitude=float(values["spectral_diagnostic_relative_amplitude"]),
            field_batch_size=int(values["spectral_diagnostic_field_batch_size"]),
            z_profiles=int(values["spectral_diagnostic_z_profiles"]),
            depth=str(values["spectral_diagnostic_depth"]),
            amplitudes=tuple(
                float(value) for value in values["spectral_diagnostic_amplitudes"]
            ),
            relative_span_tolerance=float(
                values["spectral_diagnostic_relative_span_tolerance"]
            ),
            output=output,
        )

    def validate(self, model: ModelConfig) -> None:
        if self.samples < 0:
            raise ValueError("spectral_diagnostic_samples must be non-negative")
        if self.output is not None and not self.enabled:
            raise ValueError(
                "--spectral-diagnostic-output requires --spectral-diagnostic-samples"
            )
        if not self.enabled:
            return
        if model.spatial_scheme == "2.5d" and model.lateral_interlacing != 1:
            raise ValueError(
                "the 2.5D diagnostic requires --lateral-interlacing 1 because "
                "the interlaced mesh has no unique deposited field"
            )
        if min(self.max_mode, self.fit_shells, self.field_batch_size) < 1:
            raise ValueError(
                "spectral diagnostic mode, fit-shell, and field-batch settings "
                "must be positive"
            )
        if not math.isfinite(self.relative_amplitude) or self.relative_amplitude <= 0:
            raise ValueError("spectral_diagnostic_relative_amplitude must be positive")
        if (
            not math.isfinite(self.relative_span_tolerance)
            or self.relative_span_tolerance < 0
        ):
            raise ValueError(
                "spectral_diagnostic_relative_span_tolerance must be non-negative"
            )
        if self.depth == "deep":
            if len(self.amplitudes) < 2 or any(
                not math.isfinite(value) or value <= 0 for value in self.amplitudes
            ):
                raise ValueError(
                    "deep spectral diagnostics require at least two positive amplitudes"
                )
            if len(set(self.amplitudes)) != len(self.amplitudes):
                raise ValueError("spectral diagnostic amplitudes must be distinct")
        grid_minimum = (
            min(model.z_grid, model.grid)
            if model.spatial_scheme == "3d"
            else model.grid
        )
        if 2 * self.max_mode >= grid_minimum:
            raise ValueError(
                "spectral diagnostic max mode must remain below the mesh Nyquist limit"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution environment and output paths."""

    seed: int
    device: str
    dtype: str
    checkpoint: Path | None
    source_config: Path | None


@dataclass(frozen=True)
class TrainingConfig:
    """Complete validated configuration for one residual-training run."""

    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    diagnostic: DiagnosticConfig
    runtime: RuntimeConfig

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> TrainingConfig:
        return cls.from_mapping(vars(namespace))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        data = DataConfig(
            mace_model=values["mace_model"],
            train_file=values["train_file"],
            validation_file=values["validation_file"],
            validation_indices_file=values["validation_indices_file"],
            test_file=values["test_file"],
            train_cache=values["train_cache"],
            validation_cache=values["validation_cache"],
            test_cache=values["test_cache"],
            rebuild_cache=bool(values["rebuild_cache"]),
            energy_key=str(values["energy_key"]),
            forces_key=str(values["forces_key"]),
            head=values["head"],
            num_atoms=values["num_atoms"],
            allow_periodic_z=bool(values["allow_periodic_z"]),
            skip_cell_mismatch=bool(values["skip_cell_mismatch"]),
            validation_fraction=float(values["validation_fraction"]),
        )
        model = ModelConfig.from_mapping(values)
        optimization = OptimizationConfig.from_mapping(values)
        checkpoint = values["checkpoint"]
        runtime = RuntimeConfig(
            seed=int(values["seed"]),
            device=str(values["device"]),
            dtype=str(values["dtype"]),
            checkpoint=checkpoint,
            source_config=values.get("config"),
        )
        diagnostic = DiagnosticConfig.from_mapping(values, checkpoint=checkpoint)
        configuration = cls(
            data=data,
            model=model,
            optimization=optimization,
            diagnostic=diagnostic,
            runtime=runtime,
        )
        configuration.validate()
        return configuration

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.optimization.validate(self.model.architecture)
        self.diagnostic.validate(self.model)
