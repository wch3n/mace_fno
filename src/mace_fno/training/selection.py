"""Validation-only energy selection within a near-optimal loss/force window."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any


class EnergyCheckpointSelector:
    """Keep the eligible energy/constraint Pareto frontier, with CPU snapshots.

    The constraint ceiling can only decrease. A currently nonwinning candidate
    may win later when a better energy candidate falls outside that ceiling.
    Discarding dominated or already ineligible candidates is safe, but keeping
    only the current winner would not implement the final selection rule.
    """

    def __init__(self, tolerance: float, constraint: str = "loss", metric: str = "raw"):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "energy checkpoint tolerance must be finite and non-negative"
            )
        if constraint not in {"loss", "forces"} or metric not in {"raw", "centered"}:
            raise ValueError("invalid energy checkpoint constraint or metric")
        self.tolerance = tolerance
        self.constraint = constraint
        self.metric = metric
        self.best_constraint = math.inf
        self.candidates: list[dict[str, Any]] = []

    @property
    def selected(self) -> dict[str, Any] | None:
        if not self.candidates:
            return None
        candidate = min(
            self.candidates,
            key=lambda item: (
                item["energy_score"],
                item["constraint_value"],
                item["step"],
            ),
        )
        return {
            "candidate": candidate,
            "metadata": {
                "rule": "minimum_validation_energy_within_relative_tolerance",
                "constraint": self.constraint,
                "relative_tolerance": self.tolerance,
                "energy_metric": self.metric,
                "best_constraint": self.best_constraint,
                "constraint_limit": (1.0 + self.tolerance) * self.best_constraint,
                "selected_step": candidate["step"],
                "selected_constraint": candidate["constraint_value"],
                "selected_energy_score": candidate["energy_score"],
                "validation_metrics": candidate["validation_metrics"],
                "energy_shift_applied": False,
            },
        }

    def consider(
        self,
        step: int,
        metrics: Mapping[str, Any],
        objective: float,
        snapshot: Callable[[], dict[str, Any]],
    ) -> bool:
        """Record one validation check. Return whether the saved selection changed.

        ``snapshot`` must produce detached, independent CPU state dictionaries.
        It is called only for a candidate that joins the Pareto frontier.
        """
        previous = self.selected
        energy = float(metrics.get("energy_rmse", math.nan))
        force = float(metrics.get("force_rmse", math.nan))
        bias = float(metrics.get("energy_bias", math.nan))
        if not all(math.isfinite(x) and x >= 0 for x in (energy, force, objective)):
            return False
        if self.metric == "centered" and not math.isfinite(bias):
            return False
        score = (
            energy if self.metric == "raw" else math.sqrt(max(0.0, energy**2 - bias**2))
        )
        constraint = objective if self.constraint == "loss" else force
        self.best_constraint = min(self.best_constraint, constraint)
        limit = (1.0 + self.tolerance) * self.best_constraint
        self.candidates = [c for c in self.candidates if c["constraint_value"] <= limit]
        dominated = any(
            c["energy_score"] <= score and c["constraint_value"] <= constraint
            for c in self.candidates
        )
        if constraint <= limit and not dominated:
            self.candidates = [
                c
                for c in self.candidates
                if not (
                    score <= c["energy_score"] and constraint <= c["constraint_value"]
                )
            ]
            self.candidates.append(
                {
                    "step": step,
                    "energy_score": score,
                    "constraint_value": constraint,
                    "validation_objective": objective,
                    "validation_metrics": deepcopy(dict(metrics)),
                    "state": snapshot(),
                }
            )
        current = self.selected
        return (previous is None) != (current is None) or (
            previous is not None
            and current is not None
            and previous["metadata"] != current["metadata"]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "constraint": self.constraint,
            "metric": self.metric,
            "best_constraint": self.best_constraint,
            "candidates": self.candidates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in ("tolerance", "constraint", "metric"):
            if state[name] != getattr(self, name):
                raise ValueError(f"cannot change energy checkpoint {name} on resume")
        self.best_constraint = float(state["best_constraint"])
        self.candidates = deepcopy(state["candidates"])
