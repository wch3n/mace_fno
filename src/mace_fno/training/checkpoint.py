"""Save and reconstruct frozen-MACE residual models.

The training checkpoint deliberately stores only the learned residual weights;
the much larger frozen MACE model remains in its original file.  This module is
the single compatibility boundary for reconstructing the combined model from
both files.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..coupling import MACEFNOResidual

CHECKPOINT_FORMAT_VERSION = 1


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


def residual_state_dict(model: MACEFNOResidual) -> dict[str, torch.Tensor]:
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
