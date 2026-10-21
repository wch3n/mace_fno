"""Mutable training progress and its unchanged restart-checkpoint representation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from ..coupling import MACEFNOResidual
from ._optimization import OptimizerSetup
from .checkpoint import (
    load_mace_state_dict,
    load_residual_state_dict,
    mace_state_dict,
    residual_state_dict,
)
from .configuration import TrainingConfig
from .monitor import SpectralMonitor
from .resume import (
    RESUME_FORMAT_VERSION,
    capture_rng_state,
    cpu_snapshot,
    resume_configuration,
)
from .selection import EnergyCheckpointSelector


@dataclass
class TrainingState:
    """Keep validation selection and progress separate from optimizer resources."""

    best_step: int = 0
    best_objective: float = float("inf")
    best_residual_state: dict[str, Any] = field(default_factory=dict)
    best_mace_state: dict[str, Any] | None = None
    completed_steps: int = 0
    stopped_early: bool = False
    evaluations_at_minimum_lr: int = 0
    optimization_stage: int = 1
    stage_start_step: int = 0
    last_saved_step: int = -1

    def restore(
        self,
        model: MACEFNOResidual,
        resume_state: Mapping[str, Any],
        setup: OptimizerSetup,
        generator: torch.Generator,
        spectral_monitor: SpectralMonitor | None,
        *,
        joint_training: bool,
    ) -> None:
        """Restore weights, optimizer and progress; defer global RNG restoration."""
        optimizer = setup.optimizer
        scheduler = setup.scheduler
        load_residual_state_dict(model, resume_state["residual_state_dict"])
        if joint_training:
            load_mace_state_dict(model, resume_state["mace_state_dict"])
            model.backbone.set_trainable(
                any(
                    enabled
                    for name, enabled in resume_state["requires_grad"].items()
                    if name.startswith("backbone.mace_model.")
                )
            )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(resume_state["requires_grad"][name])
        model.train()
        optimizer.load_state_dict(deepcopy(resume_state["optimizer_state_dict"]))
        if scheduler is not None:
            scheduler.load_state_dict(deepcopy(resume_state["scheduler_state_dict"]))
        generator.set_state(resume_state["sample_generator_state"])
        self.best_step = resume_state["best_step"]
        self.best_objective = resume_state["best_objective"]
        self.best_residual_state = resume_state["best_residual_state_dict"]
        self.best_mace_state = resume_state["best_mace_state_dict"]
        self.completed_steps = resume_state["completed_steps"]
        self.stopped_early = resume_state["stopped_early"]
        self.evaluations_at_minimum_lr = resume_state["evaluations_at_minimum_lr"]
        if spectral_monitor is not None:
            spectral_monitor.history = deepcopy(resume_state["spectral_history"])
            spectral_monitor.write_history()

    def checkpoint_payload(
        self,
        model: MACEFNOResidual,
        configuration: TrainingConfig,
        setup: OptimizerSetup,
        generator: torch.Generator,
        *,
        device: torch.device,
        spectral_monitor: SpectralMonitor | None,
        energy_selector: EnergyCheckpointSelector | None,
        loss_settings: dict[str, float],
    ) -> dict[str, Any]:
        """Snapshot current and selected weights without changing serialized keys."""
        optimizer = setup.optimizer
        scheduler = setup.scheduler
        joint_training = configuration.optimization.mace_training == "joint"
        staged = configuration.optimization.stage_two_start_step is not None
        # Save before restoring the best weights: Adam's moments belong to the
        # current iterate, which need not be the validation-selected model.
        payload = {
            "resume_format_version": RESUME_FORMAT_VERSION,
            "configuration": resume_configuration(configuration),
            "device_type": device.type,
            "completed_steps": self.completed_steps,
            "best_step": self.best_step,
            "best_objective": self.best_objective,
            "stopped_early": self.stopped_early,
            "evaluations_at_minimum_lr": self.evaluations_at_minimum_lr,
            "residual_state_dict": residual_state_dict(model),
            "mace_state_dict": mace_state_dict(model) if joint_training else None,
            "best_residual_state_dict": self.best_residual_state,
            "best_mace_state_dict": self.best_mace_state,
            "optimizer_state_dict": cpu_snapshot(optimizer.state_dict()),
            "scheduler_state_dict": deepcopy(scheduler.state_dict())
            if scheduler
            else None,
            "sample_generator_state": generator.get_state(),
            "rng_state": capture_rng_state(device),
            "requires_grad": {
                name: parameter.requires_grad
                for name, parameter in model.named_parameters()
            },
            "spectral_history": deepcopy(spectral_monitor.history)
            if spectral_monitor
            else [],
            "energy_selector": energy_selector.state_dict()
            if energy_selector
            else None,
        }
        if staged:
            payload.update(
                optimization_stage=self.optimization_stage,
                stage_start_step=self.stage_start_step,
                active_loss_settings=dict(loss_settings),
            )
        return payload
