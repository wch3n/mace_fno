"""Training-state persistence, distinct from selected inference checkpoints."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .configuration import TrainingConfig

RESUME_FORMAT_VERSION = 1


def cpu_snapshot(value: Any) -> Any:
    """Copy nested optimizer state without duplicating moments on the GPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return deepcopy(value)


def resume_configuration(configuration: TrainingConfig) -> dict[str, Any]:
    """Settings that must agree for continuation, excluding output/cache paths."""
    data = asdict(configuration.data)
    for name in ("train_cache", "validation_cache", "test_cache", "rebuild_cache"):
        data.pop(name)
    for name, value in data.items():
        if isinstance(value, Path):
            data[name] = str(value.expanduser().resolve())
    optimization = asdict(configuration.optimization)
    optimization.pop("steps")  # --steps is the total budget, which may be extended.
    diagnostic = asdict(configuration.diagnostic)
    diagnostic.pop("output")
    return {
        "data": data,
        "model": asdict(configuration.model),
        "optimization": optimization,
        "diagnostic": diagnostic,
        "seed": configuration.runtime.seed,
        "dtype": configuration.runtime.dtype,
    }


def input_fingerprints(configuration: TrainingConfig) -> dict[str, str]:
    """Hash source inputs once per invocation, not once per checkpoint."""
    fingerprints = {}
    for name in (
        "mace_model",
        "train_file",
        "validation_file",
        "validation_indices_file",
        "test_file",
    ):
        path = getattr(configuration.data, name)
        if path is None:
            continue
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        fingerprints[name] = digest.hexdigest()
    return fingerprints


def load_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted, full-state checkpoint and reject weights-only files."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "resume_format_version" not in state:
        raise ValueError(
            "--resume requires a full training-state (.last.pt) checkpoint. "
            "Best-model and older weights-only checkpoints cannot resume optimization."
        )
    if state["resume_format_version"] != RESUME_FORMAT_VERSION:
        raise ValueError("unsupported resume checkpoint version")
    required = {
        "configuration",
        "completed_steps",
        "best_step",
        "best_objective",
        "stopped_early",
        "evaluations_at_minimum_lr",
        "residual_state_dict",
        "mace_state_dict",
        "best_residual_state_dict",
        "best_mace_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "sample_generator_state",
        "requires_grad",
        "spectral_history",
        "device_type",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(f"incomplete resume checkpoint: missing {sorted(missing)}")
    return state


def validate_resume_checkpoint(
    state: Mapping[str, Any],
    configuration: TrainingConfig,
    *,
    fingerprints: Mapping[str, str] | None = None,
) -> None:
    """Reject changed training contracts before model state or outputs are touched."""
    expected = resume_configuration(configuration)
    saved = state["configuration"]
    differences = [key for key in expected if expected[key] != saved.get(key)]
    if differences:
        raise ValueError(
            "resume settings differ in "
            + ", ".join(differences)
            + ". Only the total step budget, device index, output paths, checkpoint interval, "
            "and cache locations may change."
        )
    completed = state["completed_steps"]
    if configuration.optimization.steps < completed:
        raise ValueError(
            f"steps is a total budget and must be >= completed_steps ({completed})"
        )
    if fingerprints is not None and dict(fingerprints) != state.get(
        "input_fingerprints"
    ):
        raise ValueError(
            "resume input fingerprints differ: source data or MACE model changed"
        )


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Include global RNGs used by stochastic model operations/interlacing."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    cuda = state["cuda"]
    if device.type == "cuda" and cuda is not None:
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all(cuda)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
