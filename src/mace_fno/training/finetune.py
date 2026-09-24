"""Weights-only initialization, deliberately separate from exact training resume."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import torch

from ..coupling import MACEFNOResidual
from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_mace_state_dict,
    load_residual_state_dict,
    resolve_checkpoint_model_path,
)
from .configuration import ModelConfig, TrainingConfig
from .resume import RESUME_FORMAT_VERSION


def _digest(handle: BinaryIO) -> str:
    result = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return _digest(handle)


def _model_settings(model: ModelConfig) -> dict[str, Any]:
    settings = asdict(model)
    # A saved resolved YAML spells out this default, while a user YAML may not.
    settings["z_modes"] = model.resolved_z_modes if model.spatial_scheme == "3d" else 0
    return settings


@dataclass(frozen=True)
class FineTuneCheckpoint:
    """Only transferable model state and provenance, never optimizer history."""

    residual_state: Mapping[str, torch.Tensor]
    mace_state: Mapping[str, torch.Tensor] | None
    reference_cell: torch.Tensor
    provenance: dict[str, Any]


def load_finetune_checkpoint(
    path: str | Path,
    configuration: TrainingConfig,
) -> FineTuneCheckpoint:
    """Validate a trusted best/energy/latest checkpoint before model construction.

    The architecture, precision, training mode, head and original MACE reference
    must match. Data and optimization settings may change. For an older selected
    checkpoint without an input hash, its original MACE file must be accessible.
    """
    source = Path(path).expanduser().resolve()
    # Hash and load the same open inode even if a live run atomically replaces
    # its latest checkpoint while we are reading it.
    with source.open("rb") as handle:
        digest = _digest(handle)
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("--init-from requires a MACE-FNO checkpoint mapping")
    latest = "resume_format_version" in payload
    if latest:
        if payload["resume_format_version"] != RESUME_FORMAT_VERSION:
            raise ValueError("unsupported init-from training-state checkpoint version")
        saved = payload.get("configuration")
        if not isinstance(saved, Mapping):
            raise ValueError("init-from checkpoint is missing training configuration")
        model = ModelConfig.from_mapping(saved["model"])
        optimization = saved["optimization"]
        data = saved["data"]
        dtype = saved["dtype"]
        step = payload.get("completed_steps")
    else:
        version = int(payload.get("checkpoint_format_version", 0))
        if not 0 <= version <= CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported init-from model checkpoint version")
        saved = payload.get("training_configuration")
        if not isinstance(saved, Mapping):
            raise ValueError(
                "init-from checkpoint is missing saved training configuration"
            )
        model = ModelConfig.from_mapping(saved)
        optimization = saved
        data = saved
        dtype = saved["dtype"]
        step = payload.get("best_step")
    expected = _model_settings(configuration.model)
    actual = _model_settings(model)
    differences = [key for key in expected if actual[key] != expected[key]]
    if differences:
        raise ValueError("init-from model settings differ: " + ", ".join(differences))
    if dtype != configuration.runtime.dtype:
        raise ValueError("init-from requires the same compute dtype")
    regime = optimization["mace_training"]
    if regime != configuration.optimization.mace_training:
        raise ValueError("init-from requires the same frozen/joint MACE training mode")
    if data.get("head") != configuration.data.head:
        raise ValueError("init-from requires the same MACE head")

    expected_hash = payload.get("input_fingerprints", {}).get("mace_model")
    verification = "checkpoint_sha256"
    if expected_hash is None:
        original = resolve_checkpoint_model_path(data["mace_model"], source)
        if not original.is_file():
            raise ValueError(
                "cannot verify init-from MACE reference: checkpoint has no hash "
                "and its original MACE file is unavailable"
            )
        expected_hash = _file_digest(original)
        verification = "original_reference_file_sha256"
    if _file_digest(configuration.data.mace_model) != expected_hash:
        raise ValueError("init-from MACE reference fingerprint differs")

    residual = payload.get("residual_state_dict")
    mace = payload.get("mace_state_dict")
    if not isinstance(residual, Mapping) or not residual:
        raise ValueError("init-from checkpoint has no residual weights")
    if regime == "joint" and not isinstance(mace, Mapping):
        raise ValueError("joint init-from checkpoint has no MACE weights")
    if regime == "frozen" and mace is not None:
        raise ValueError(
            "frozen init-from checkpoint unexpectedly contains MACE weights"
        )
    reference = residual.get("reference_cell", payload.get("reference_cell"))
    if (
        not isinstance(reference, torch.Tensor)
        or reference.shape != (3, 3)
        or not torch.isfinite(reference).all()
    ):
        raise ValueError("init-from checkpoint has no valid (3, 3) reference cell")
    return FineTuneCheckpoint(
        residual_state=dict(residual),
        mace_state=dict(mace) if mace is not None else None,
        reference_cell=reference.detach().clone(),
        provenance={
            "parent_checkpoint": str(source),
            "parent_sha256": digest,
            "parent_step": step,
            "parent_kind": "latest_training_state" if latest else "selected_model",
            "mace_reference_sha256": expected_hash,
            "mace_reference_verification": verification,
            "parent_energy_scale": optimization["energy_scale"],
            "parent_force_scale": optimization["force_scale"],
            "optimizer_state": "reset",
            "scheduler_and_selection_history": "reset",
            "step_numbering": "new_optimization_stage",
            "energy_shift_applied": False,
        },
    )


def _validate_weights(
    state: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(
            f"init-from {label} keys differ: missing={missing}, unexpected={extra}"
        )
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"init-from {label} tensor shape differs: {key}")


def initialize_finetune_model(
    model: MACEFNOResidual,
    checkpoint: FineTuneCheckpoint,
) -> None:
    """Restore complete compatible weights without resetting learned projections."""
    state = model.state_dict()
    residual = {
        k: v for k, v in state.items() if not k.startswith("backbone.mace_model.")
    }
    _validate_weights(checkpoint.residual_state, residual, "residual")
    if checkpoint.mace_state is not None:
        _validate_weights(
            checkpoint.mace_state, model.backbone.mace_model.state_dict(), "MACE"
        )
    # Validate both branches before changing either one.
    load_residual_state_dict(model, checkpoint.residual_state)
    if checkpoint.mace_state is not None:
        load_mace_state_dict(model, checkpoint.mace_state)
