"""ASE inference adapter for a frozen or jointly trained MACE-FNO checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from .coupling import MACEFNOResidual
from .training.checkpoint import load_mace_fno_components
from .training.runtime import choose_device


class MACEFNOCalculator(Calculator):
    """Evaluate combined MACE-FNO energies and conservative forces in ASE.

    Stress is intentionally unavailable: the current FNO checkpoints may be
    conditioned on fixed, isotropic, or anisotropic 3D cells, but no virial
    derivative has yet been validated.  ``results`` additionally exposes
    ``mace_energy`` and ``residual_energy`` for diagnostics.

    ``model`` and ``graph_converter`` form an in-memory construction path used
    by tests and advanced workflows.  Normal use should pass ``checkpoint``.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | str | None = None,
        mace_model_path: str | Path | None = None,
        mace_head: str | None = None,
        model: MACEFNOResidual | None = None,
        graph_converter: Callable[[Atoms], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        resolved_device = (
            choose_device(device) if isinstance(device, str) else torch.device(device)
        )
        checkpoint_mode = checkpoint is not None
        component_mode = model is not None or graph_converter is not None
        if checkpoint_mode and component_mode:
            raise ValueError(
                "pass either checkpoint or model plus graph_converter, not both"
            )
        if not checkpoint_mode and (model is None or graph_converter is None):
            raise ValueError(
                "checkpoint is required unless both model and graph_converter "
                "are supplied"
            )

        self.checkpoint_metadata: dict[str, Any] | None = None
        self._mace_calculator: Any | None = None
        if checkpoint is not None:
            loaded_model, metadata, mace_calculator = load_mace_fno_components(
                checkpoint,
                device=resolved_device,
                dtype=dtype,
                mace_model_path=mace_model_path,
                mace_head=mace_head,
            )
            model = loaded_model
            graph_converter = mace_calculator._atoms_to_batch
            self.checkpoint_metadata = metadata
            self._mace_calculator = mace_calculator
        else:
            assert model is not None
            if dtype is None:
                resolved_dtype = next(model.parameters()).dtype
            elif isinstance(dtype, torch.dtype):
                resolved_dtype = dtype
            else:
                normalized = str(dtype).lower().removeprefix("torch.")
                if normalized not in {"float32", "float64"}:
                    raise ValueError("dtype must be float32 or float64")
                resolved_dtype = getattr(torch, normalized)
            model = model.to(device=resolved_device, dtype=resolved_dtype)

        assert model is not None and graph_converter is not None
        model.eval()
        self.model = model
        self._graph_converter = graph_converter
        self.device = resolved_device
        self.dtype = next(model.parameters()).dtype

    def _validate_periodicity(self, atoms: Atoms) -> None:
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if self.model.spatial_scheme == "3d":
            if not bool(pbc.all()):
                raise ValueError("a 3D MACE-FNO checkpoint requires periodicity on xyz")
        elif not bool(pbc[:2].all()):
            raise ValueError("a 2D/2.5D MACE-FNO checkpoint requires periodicity on xy")

    def _atoms_to_graph(self, atoms: Atoms) -> dict[str, Any]:
        converted = self._graph_converter(atoms)
        if isinstance(converted, Mapping):
            raw = dict(converted)
        else:
            if hasattr(converted, "to"):
                converted = converted.to(self.device)
            if not hasattr(converted, "to_dict"):
                raise TypeError(
                    "graph_converter must return a mapping or expose to_dict()"
                )
            raw = converted.to_dict()

        graph: dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(value, torch.Tensor):
                graph[key] = value
                continue
            target_dtype = self.dtype if value.is_floating_point() else value.dtype
            graph[key] = value.detach().clone().to(
                device=self.device,
                dtype=target_dtype,
            )
        return graph

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        requested = set(properties or ["energy"])
        unsupported = requested - set(self.implemented_properties)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise PropertyNotImplementedError(
                f"MACEFNOCalculator does not implement: {names}"
            )
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("ASE did not provide an Atoms object")
        self._validate_periodicity(self.atoms)
        graph = self._atoms_to_graph(self.atoms)
        compute_forces = "forces" in requested
        with torch.enable_grad():
            output = self.model(
                graph,
                training=False,
                compute_force=compute_forces,
            )
        if output["energy"].numel() != 1:
            raise ValueError("ASE inference requires exactly one graph")
        energy = float(output["energy"].detach().cpu().reshape(-1)[0])
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "mace_energy": float(
                output["base_energy"].detach().cpu().reshape(-1)[0]
            ),
            "residual_energy": float(
                output["residual_energy"].detach().cpu().reshape(-1)[0]
            ),
        }
        if compute_forces:
            forces = output["forces"]
            if forces is None or forces.shape != (len(self.atoms), 3):
                shape = None if forces is None else tuple(forces.shape)
                raise ValueError(
                    "MACE-FNO returned an unexpected force shape: "
                    f"{shape}, expected {(len(self.atoms), 3)}"
                )
            self.results["forces"] = forces.detach().cpu().numpy()
