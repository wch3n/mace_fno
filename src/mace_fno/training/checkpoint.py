"""Save and reconstruct frozen-MACE residual models.

The training checkpoint deliberately stores only the learned residual weights;
the much larger frozen MACE model remains in its original file.  This module is
the single compatibility boundary for reconstructing the combined model from
both files.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..coupling import MACEFNOResidual

if TYPE_CHECKING:
    from .configuration import TrainingConfig
    from .setup import PreparedData
    from .trainer import OptimizationResult

CHECKPOINT_FORMAT_VERSION = 1


def training_checkpoint_payload(
    configuration: TrainingConfig,
    prepared: PreparedData,
    result: OptimizationResult,
    model: torch.nn.Module,
    *,
    effective_configuration: Mapping[str, Any],
    spectral_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete, versioned payload for one training run.

    Keeping this schema beside the reconstruction code makes additions and
    compatibility changes auditable in one module. The flat version-1 layout
    is retained so existing inference and benchmark tooling remains valid.
    """
    data = configuration.data
    model_config = configuration.model
    optimization = configuration.optimization
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "training_configuration": dict(effective_configuration),
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
        "early_stopping_patience_evals": optimization.early_stopping_patience_evals,
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
        "spectral_diagnostic": (
            dict(spectral_diagnostic) if spectral_diagnostic is not None else None
        ),
        "seed": configuration.runtime.seed,
    }


def save_training_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Serialize a training payload, creating its parent directory if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), destination)
    return destination


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a MACE-FNO training checkpoint.

    Checkpoints written before explicit versioning are treated as version 0.
    Newer, unknown formats fail early instead of being reconstructed with
    potentially incorrect defaults.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("a MACE-FNO checkpoint must contain a mapping")
    checkpoint = dict(payload)
    version = int(checkpoint.get("checkpoint_format_version", 0))
    if version < 0 or version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported MACE-FNO checkpoint format version {version}; "
            f"this installation supports versions 0-{CHECKPOINT_FORMAT_VERSION}"
        )
    if not isinstance(checkpoint.get("residual_state_dict"), Mapping):
        raise KeyError("checkpoint does not contain a residual_state_dict mapping")
    return checkpoint


def checkpoint_dtype(
    checkpoint: Mapping[str, Any],
    override: torch.dtype | str | None = None,
) -> torch.dtype:
    """Resolve the inference dtype from an override or checkpoint metadata."""
    value: torch.dtype | str = override or checkpoint.get("dtype", "float32")
    if isinstance(value, torch.dtype):
        if value not in {torch.float32, torch.float64}:
            raise ValueError("MACE-FNO reconstruction supports float32 or float64")
        return value
    normalized = str(value).lower().removeprefix("torch.")
    if normalized == "float32":
        return torch.float32
    if normalized == "float64":
        return torch.float64
    raise ValueError(f"unsupported checkpoint dtype {value!r}")


def infer_checkpoint_z_mixing(checkpoint: Mapping[str, Any]) -> str:
    """Recover 2.5D z mixing for checkpoints predating that metadata."""
    configured = checkpoint.get("z_mixing")
    if configured in {"local", "global"}:
        return str(configured)
    if configured == "spectral" or checkpoint.get("architecture") == "linear":
        return "local"
    state = checkpoint.get("residual_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("cannot infer z_mixing without residual_state_dict")
    key = "long_range.field_operator.fno.blocks.0.z_mixing.weight"
    weight = state.get(key)
    if weight is None:
        raise KeyError("cannot infer the missing z_mixing checkpoint metadata")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"checkpoint value {key!r} must be a tensor")
    if weight.ndim == 5:
        return "local"
    if weight.ndim == 3:
        return "global"
    raise ValueError(f"unrecognized z-mixing weight shape {tuple(weight.shape)}")


def _required(checkpoint: Mapping[str, Any], key: str) -> Any:
    if key not in checkpoint or checkpoint[key] is None:
        raise KeyError(f"checkpoint is missing required model metadata {key!r}")
    return checkpoint[key]


def checkpoint_model_parameters(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Translate checkpoint metadata into ``MACEFNOResidual`` arguments."""
    scheme = str(checkpoint.get("spatial_scheme", "2d")).lower()
    if scheme not in {"2d", "2.5d", "3d"}:
        raise ValueError(f"unsupported checkpoint spatial_scheme {scheme!r}")
    grid_shape = tuple(int(value) for value in _required(checkpoint, "grid_shape"))
    if len(grid_shape) != 2:
        raise ValueError("checkpoint grid_shape must contain two planar dimensions")
    modes = tuple(int(value) for value in _required(checkpoint, "n_modes"))
    expected_modes = 3 if scheme == "3d" else 2
    if len(modes) != expected_modes:
        raise ValueError(
            f"a {scheme} checkpoint must store {expected_modes} Fourier-mode counts"
        )

    parameters: dict[str, Any] = {
        "grid_shape": grid_shape,
        "channels": int(_required(checkpoint, "channels")),
        "n_modes": modes[1:] if scheme == "3d" else modes,
        "source_hidden_channels": int(
            _required(checkpoint, "source_hidden_channels")
        ),
        "fno_hidden_channels": int(_required(checkpoint, "fno_hidden_channels")),
        "fno_layers": int(_required(checkpoint, "fno_layers")),
        "fno_architecture": str(_required(checkpoint, "architecture")),
        "spatial_scheme": scheme,
        "reference_cell": _required(checkpoint, "reference_cell"),
        "cell_mode": str(checkpoint.get("cell_mode") or "fixed"),
    }
    if scheme == "2.5d":
        parameters.update(
            {
                "z_grid_size": int(_required(checkpoint, "z_grid_size")),
                "z_extent": float(_required(checkpoint, "z_extent")),
                "z_center": str(checkpoint.get("z_center") or "mean"),
                "fno_lateral_interlacing": int(
                    checkpoint.get("lateral_interlacing", 1)
                ),
                "fno_z_kernel_size": int(checkpoint.get("z_kernel_size") or 3),
                "fno_z_mixing": infer_checkpoint_z_mixing(checkpoint),
                "fno_planar_symmetry": str(
                    checkpoint.get("planar_symmetry") or "none"
                ),
            }
        )
    elif scheme == "3d":
        parameters.update(
            {
                "z_grid_size": int(_required(checkpoint, "z_grid_size")),
                "fno_z_modes": modes[0],
                "fno_volume_interlacing": int(
                    checkpoint.get("volume_interlacing", 1)
                ),
                "fno_interlacing_training": str(
                    checkpoint.get("interlacing_training") or "full"
                ),
                "fno_spectral_symmetry": str(
                    checkpoint.get("spectral_symmetry") or "none"
                ),
                "fno_spectral_groups": int(checkpoint.get("spectral_groups", 1)),
                "fno_metric_hidden_channels": int(
                    checkpoint.get("metric_hidden_channels", 16)
                ),
            }
        )
    return parameters


def build_mace_fno_model(
    checkpoint: Mapping[str, Any],
    mace_model: torch.nn.Module,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | str | None = None,
) -> MACEFNOResidual:
    """Reconstruct a combined model from metadata and a loaded MACE module."""
    resolved_device = torch.device(device)
    resolved_dtype = checkpoint_dtype(checkpoint, dtype)
    model = MACEFNOResidual(
        mace_model,
        **checkpoint_model_parameters(checkpoint),
    ).to(device=resolved_device, dtype=resolved_dtype)
    load_residual_state_dict(model, _required(checkpoint, "residual_state_dict"))
    model.eval()
    return model


def resolve_checkpoint_model_path(
    value: str | Path, checkpoint_path: str | Path
) -> Path:
    """Resolve a frozen-MACE path, including a relocated legacy run tree."""
    stored = Path(value).expanduser()
    if stored.is_absolute() and stored.exists():
        return stored
    checkpoint_location = Path(checkpoint_path).expanduser().resolve()
    if not stored.is_absolute():
        candidates = (Path.cwd() / stored, checkpoint_location.parent / stored)
        existing = next(
            (candidate for candidate in candidates if candidate.exists()), None
        )
        if existing is not None:
            return existing

    # Version-0 checkpoints often contain an absolute path below the former
    # repository-local artifacts/ directory. If the complete run tree was
    # moved to scratch, recover the same suffix relative to a checkpoint
    # ancestor instead of requiring binary checkpoint rewriting.
    artifact_indices = [
        index for index, part in enumerate(stored.parts) if part == "artifacts"
    ]
    if artifact_indices:
        suffix = stored.parts[artifact_indices[-1] + 1 :]
        for ancestor in (checkpoint_location.parent, *checkpoint_location.parents):
            relocated = ancestor.joinpath(*suffix)
            if relocated.exists():
                return relocated
    return stored


def load_mace_fno_components(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | str | None = None,
    mace_model_path: str | Path | None = None,
    mace_head: str | None = None,
) -> tuple[MACEFNOResidual, dict[str, Any], Any]:
    """Load the combined model plus MACE's atom-to-graph converter.

    The returned MACE calculator owns the species table, cutoff, and graph
    construction settings used by its checkpoint.  It is exposed so inference
    adapters such as the ASE calculator can build exactly the same graph as
    standalone MACE without loading the backbone twice.
    """
    path = Path(checkpoint_path).expanduser()
    checkpoint = load_checkpoint_payload(path)
    stored_model = mace_model_path or _required(checkpoint, "mace_model")
    resolved_model_path = resolve_checkpoint_model_path(stored_model, path)
    try:
        from mace.calculators import MACECalculator
    except ImportError as error:
        raise ImportError(
            "loading a frozen-MACE checkpoint requires the optional "
            "'mace-torch' dependency"
        ) from error

    resolved_device = torch.device(device)
    calculator_kwargs: dict[str, Any] = {
        "model_paths": str(resolved_model_path),
        "device": str(resolved_device),
        "default_dtype": "float64",
    }
    resolved_head = mace_head if mace_head is not None else checkpoint.get("mace_head")
    if resolved_head is not None:
        calculator_kwargs["head"] = resolved_head
    calculator = MACECalculator(**calculator_kwargs)
    model = build_mace_fno_model(
        checkpoint,
        calculator.models[0],
        device=resolved_device,
        dtype=dtype,
    )
    return model, checkpoint, calculator


def load_mace_fno_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | str | None = None,
    mace_model_path: str | Path | None = None,
    mace_head: str | None = None,
) -> tuple[MACEFNOResidual, dict[str, Any]]:
    """Load the frozen MACE file and residual checkpoint as one model."""
    model, checkpoint, _ = load_mace_fno_components(
        checkpoint_path,
        device=device,
        dtype=dtype,
        mace_model_path=mace_model_path,
        mace_head=mace_head,
    )
    return model, checkpoint


def residual_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return learned state without duplicating the frozen MACE checkpoint."""
    prefix = "backbone.mace_model."
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    }


def load_residual_state_dict(
    model: MACEFNOResidual,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Restore residual-only state while retaining the frozen backbone."""
    complete_state = model.state_dict()
    unexpected = set(state) - set(complete_state)
    if unexpected:
        raise KeyError(
            f"residual checkpoint contains unexpected keys: {sorted(unexpected)}"
        )
    complete_state.update(
        {
            key: value.to(
                device=complete_state[key].device,
                dtype=complete_state[key].dtype,
            )
            for key, value in state.items()
        }
    )
    model.load_state_dict(complete_state)
