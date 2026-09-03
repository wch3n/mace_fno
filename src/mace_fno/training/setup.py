"""Preparation of MACE, datasets, and frozen residual targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..coupling import MACEFNOResidual
from .configuration import TrainingConfig
from .data import load_or_create_samples, save_sample_cache, split_samples
from .evaluation import ensure_frozen_residual_targets
from .initialization import initialize_scaled_residual_output, initialize_zero_residual

Sample = dict[str, Any]


@dataclass
class PreparedData:
    """Datasets and cache bookkeeping needed by one training run."""

    samples: list[Sample]
    train_samples: list[Sample]
    validation_samples: list[Sample]
    test_samples: list[Sample]
    reference_cell: Tensor
    train_cache_metadata: dict[str, Any]
    validation_cache_metadata: dict[str, Any] | None
    test_cache_metadata: dict[str, Any] | None
    train_cache_hit: bool
    validation_cache_hit: bool
    test_cache_hit: bool


def load_mace_calculator(configuration: TrainingConfig, device: torch.device) -> Any:
    """Load exactly one frozen MACE model without making MACE a base dependency."""
    from mace.calculators import MACECalculator

    data = configuration.data
    calculator_kwargs: dict[str, Any] = {
        "model_paths": str(data.mace_model),
        "device": str(device),
        # Preserve exact atomic reference energies when the residual uses float32.
        "default_dtype": "float64",
    }
    if data.head is not None:
        calculator_kwargs["head"] = data.head
    calculator = MACECalculator(**calculator_kwargs)
    if len(calculator.models) != 1:
        raise ValueError("residual training requires exactly one MACE model")
    return calculator


def prepare_data(
    calculator: Any,
    configuration: TrainingConfig,
    dtype: torch.dtype,
) -> PreparedData:
    """Load training, validation, and test samples with consistent cell handling."""
    data = configuration.data
    model = configuration.model
    runtime = configuration.runtime
    samples, reference_cell, train_metadata, train_hit = load_or_create_samples(
        calculator,
        data.train_file,
        data.energy_key,
        data.forces_key,
        dtype,
        data.num_atoms,
        data.allow_periodic_z,
        data.skip_cell_mismatch,
        data.mace_model,
        data.train_cache,
        data.rebuild_cache,
        spatial_scheme=model.spatial_scheme,
        cell_mode=model.cell_mode,
    )

    validation_metadata = None
    validation_hit = False
    if data.validation_file is None:
        validation_indices = None
        if data.validation_indices_file is not None:
            validation_indices = [
                int(value) for value in data.validation_indices_file.read_text().split()
            ]
        train_samples, validation_samples = split_samples(
            samples,
            data.validation_fraction,
            runtime.seed + 2,
            validation_indices=validation_indices,
        )
    else:
        train_samples = samples
        validation_samples, _, validation_metadata, validation_hit = (
            load_or_create_samples(
                calculator,
                data.validation_file,
                data.energy_key,
                data.forces_key,
                dtype,
                data.num_atoms,
                data.allow_periodic_z,
                data.skip_cell_mismatch,
                data.mace_model,
                data.validation_cache,
                data.rebuild_cache,
                reference_cell=reference_cell,
                spatial_scheme=model.spatial_scheme,
                cell_mode=model.cell_mode,
            )
        )

    test_samples: list[Sample] = []
    test_metadata = None
    test_hit = False
    if data.test_file is not None:
        test_samples, _, test_metadata, test_hit = load_or_create_samples(
            calculator,
            data.test_file,
            data.energy_key,
            data.forces_key,
            dtype,
            data.num_atoms,
            data.allow_periodic_z,
            data.skip_cell_mismatch,
            data.mace_model,
            data.test_cache,
            data.rebuild_cache,
            reference_cell=reference_cell,
            spatial_scheme=model.spatial_scheme,
            cell_mode=model.cell_mode,
        )

    return PreparedData(
        samples=samples,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        reference_cell=reference_cell,
        train_cache_metadata=train_metadata,
        validation_cache_metadata=validation_metadata,
        test_cache_metadata=test_metadata,
        train_cache_hit=train_hit,
        validation_cache_hit=validation_hit,
        test_cache_hit=test_hit,
    )


def build_training_model(
    mace_model: torch.nn.Module,
    configuration: TrainingConfig,
    reference_cell: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> MACEFNOResidual:
    """Construct and initialize the trainable residual around frozen MACE."""
    model_config = configuration.model
    optimization = configuration.optimization
    model = MACEFNOResidual(
        mace_model,
        (model_config.grid, model_config.grid),
        model_config.channels,
        (model_config.modes, model_config.modes),
        source_hidden_channels=model_config.source_hidden_channels,
        fno_hidden_channels=model_config.fno_hidden_channels,
        fno_layers=model_config.fno_layers,
        fno_architecture=model_config.architecture,
        spatial_scheme=model_config.spatial_scheme,
        z_grid_size=model_config.z_grid or None,
        fno_z_modes=(
            model_config.resolved_z_modes
            if model_config.spatial_scheme == "3d"
            else None
        ),
        z_extent=model_config.z_extent,
        z_center=model_config.z_center,
        fno_lateral_interlacing=model_config.lateral_interlacing,
        fno_volume_interlacing=model_config.volume_interlacing,
        fno_interlacing_training=model_config.interlacing_training,
        fno_z_kernel_size=model_config.z_kernel_size,
        fno_z_mixing=model_config.z_mixing,
        fno_planar_symmetry=model_config.planar_symmetry,
        fno_spectral_symmetry=model_config.spectral_symmetry,
        fno_spectral_groups=model_config.spectral_groups,
        fno_metric_hidden_channels=model_config.metric_hidden_channels,
        reference_cell=reference_cell,
        cell_mode=model_config.cell_mode,
    ).to(device=device, dtype=dtype)
    if optimization.output_initialization_scale:
        initialize_scaled_residual_output(
            model, optimization.output_initialization_scale
        )
    elif not optimization.random_residual_initialization:
        initialize_zero_residual(model)
    return model


def cache_frozen_targets(
    model: MACEFNOResidual,
    prepared: PreparedData,
    configuration: TrainingConfig,
    *,
    device: torch.device,
) -> None:
    """Compute missing DFT-minus-MACE targets and persist configured caches."""
    data = configuration.data
    batch_size = configuration.optimization.evaluation_batch_size

    train_changed = ensure_frozen_residual_targets(
        model,
        prepared.samples,
        device=device,
        batch_size=batch_size,
    )
    if train_changed or not prepared.train_cache_hit:
        save_sample_cache(
            data.train_cache,
            prepared.train_cache_metadata,
            prepared.samples,
            prepared.reference_cell,
        )

    if data.validation_file is not None:
        validation_changed = ensure_frozen_residual_targets(
            model,
            prepared.validation_samples,
            device=device,
            batch_size=batch_size,
        )
        if validation_changed or not prepared.validation_cache_hit:
            save_sample_cache(
                data.validation_cache,
                prepared.validation_cache_metadata,
                prepared.validation_samples,
                prepared.reference_cell,
            )

    if data.test_file is not None:
        test_changed = ensure_frozen_residual_targets(
            model,
            prepared.test_samples,
            device=device,
            batch_size=batch_size,
        )
        if test_changed or not prepared.test_cache_hit:
            save_sample_cache(
                data.test_cache,
                prepared.test_cache_metadata,
                prepared.test_samples,
                prepared.reference_cell,
            )
