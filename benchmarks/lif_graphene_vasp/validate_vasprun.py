#!/usr/bin/env python3
"""Validate a completed VASP static run without third-party Python packages.

The scheduler-side check intentionally uses only the Python standard library.
This keeps restart and dependency decisions independent of user-site NumPy,
SciPy, and pymatgen binary compatibility under the VASP module environment.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


class VasprunValidationError(RuntimeError):
    """A vasprun.xml file is missing, incomplete, or unsuccessful."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class VasprunSummary:
    """Minimal pymatgen-compatible data used by the validator and collector."""

    parameters: dict[str, int]
    ionic_steps: list[dict[str, Any]]
    final_structure: tuple[None, ...]
    final_energy: float
    vasp_version: str
    converged_electronic: bool
    converged_ionic: bool


def _has_closing_modeling_tag(path: Path) -> bool:
    """Check for the final XML tag without loading a potentially large file."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8192))
        tail = handle.read().rstrip()
    return tail.endswith(b"</modeling>")


def _parse_float_row(text: str | None, *, label: str) -> list[float]:
    if text is None:
        raise ValueError(f"empty {label} row")
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"{label} row has {len(values)} rather than 3 values")
    return values


def _parse_vasprun(path: Path) -> VasprunSummary:
    """Stream the small completion summary needed from a potentially large XML."""
    parameters: dict[str, int] = {}
    vasp_version = "unknown"
    atom_count: int | None = None
    ionic_steps: list[dict[str, Any]] = []
    tag_stack: list[str] = []
    current_step: dict[str, Any] | None = None
    direct_energy: dict[str, float] | None = None
    direct_forces: list[list[float]] | None = None

    try:
        events = ElementTree.iterparse(path, events=("start", "end"))
        for event, element in events:
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "start":
                parent = tag_stack[-1] if tag_stack else None
                if tag == "calculation":
                    current_step = {"electronic_steps": [], "forces": []}
                elif tag == "energy" and parent == "calculation":
                    direct_energy = {}
                elif (
                    tag == "varray"
                    and parent == "calculation"
                    and element.get("name") == "forces"
                ):
                    direct_forces = []
                tag_stack.append(tag)
                continue

            parent = tag_stack[-2] if len(tag_stack) > 1 else None
            text = element.text.strip() if element.text else ""
            if tag == "i":
                name = element.get("name")
                if parent == "generator" and name == "version" and text:
                    vasp_version = text
                elif parent == "incar" and name in {"NELM", "NSW", "IBRION"}:
                    parameters[name] = int(text)
                elif direct_energy is not None and parent == "energy" and name:
                    direct_energy[name] = float(text)
            elif tag == "atoms" and parent == "atominfo":
                atom_count = int(text)
            elif tag == "v" and direct_forces is not None and parent == "varray":
                direct_forces.append(_parse_float_row(text, label="force"))
            elif tag == "scstep" and parent == "calculation":
                if current_step is None:
                    raise ValueError("electronic step found outside a calculation")
                current_step["electronic_steps"].append(None)
            elif tag == "energy" and parent == "calculation":
                if current_step is None or direct_energy is None:
                    raise ValueError("calculation energy was not initialized")
                current_step["energy"] = direct_energy
                direct_energy = None
            elif (
                tag == "varray"
                and parent == "calculation"
                and element.get("name") == "forces"
            ):
                if current_step is None or direct_forces is None:
                    raise ValueError("calculation forces were not initialized")
                current_step["forces"] = direct_forces
                direct_forces = None
            elif tag == "calculation":
                if current_step is None:
                    raise ValueError("calculation was not initialized")
                ionic_steps.append(current_step)
                current_step = None

            tag_stack.pop()
            element.clear()
    except (ElementTree.ParseError, OSError, TypeError, ValueError) as error:
        raise VasprunValidationError(
            "parse_error", f"could not parse {path}: {error}"
        ) from error

    if atom_count is None:
        raise VasprunValidationError("no_results", f"{path} has no atom count")
    if not ionic_steps:
        raise VasprunValidationError("no_results", f"{path} has no ionic step")
    missing_parameters = {"NELM", "NSW", "IBRION"} - set(parameters)
    if missing_parameters:
        raise VasprunValidationError(
            "no_results",
            f"{path} is missing parameters {sorted(missing_parameters)}",
        )

    last_step = ionic_steps[-1]
    energy_values = last_step.get("energy", {})
    energy: float | None = None
    for key in ("e_0_energy", "e_fr_energy", "e_wo_entrp"):
        if key in energy_values:
            energy = float(energy_values[key])
            break
    if energy is None:
        raise VasprunValidationError("no_results", f"{path} has no final energy")

    electronic_steps = len(last_step["electronic_steps"])
    converged_electronic = electronic_steps < parameters["NELM"]
    nsw = parameters["NSW"]
    converged_ionic = nsw == 0 or len(ionic_steps) < nsw
    return VasprunSummary(
        parameters=parameters,
        ionic_steps=ionic_steps,
        final_structure=(None,) * atom_count,
        final_energy=energy,
        vasp_version=vasp_version,
        converged_electronic=converged_electronic,
        converged_ionic=converged_ionic,
    )


def load_successful_vasprun(
    path: Path,
    *,
    expected_atoms: int | None = None,
    require_static: bool = False,
) -> VasprunSummary:
    """Return a validated lightweight summary or raise an explicit error."""
    path = Path(path)
    if not path.is_file():
        raise VasprunValidationError("missing", f"missing {path}")
    if path.stat().st_size == 0:
        raise VasprunValidationError("empty", f"empty {path}")
    if not _has_closing_modeling_tag(path):
        raise VasprunValidationError(
            "incomplete_xml", f"{path} does not end with </modeling>"
        )

    run = _parse_vasprun(path)
    if not run.converged_electronic:
        nelm = run.parameters["NELM"]
        electronic_steps = len(run.ionic_steps[-1]["electronic_steps"])
        raise VasprunValidationError(
            "unconverged_electronic",
            f"electronic SCF did not converge ({electronic_steps}/{nelm} steps)",
        )
    if not run.converged_ionic:
        raise VasprunValidationError("unconverged_ionic", "ionic run did not converge")

    nsw = run.parameters["NSW"]
    ibrion = run.parameters["IBRION"]
    if require_static and (nsw != 0 or ibrion != -1):
        raise VasprunValidationError(
            "wrong_run_type",
            f"expected static NSW=0, IBRION=-1; found NSW={nsw}, IBRION={ibrion}",
        )

    atom_count = len(run.final_structure)
    if expected_atoms is not None and atom_count != expected_atoms:
        raise VasprunValidationError(
            "wrong_atom_count",
            f"expected {expected_atoms} atoms; vasprun contains {atom_count}",
        )
    if not math.isfinite(run.final_energy):
        raise VasprunValidationError("nonfinite_energy", "final energy is not finite")

    forces = run.ionic_steps[-1].get("forces", [])
    if len(forces) != atom_count or any(len(row) != 3 for row in forces):
        shape = (len(forces), len(forces[0]) if forces else 0)
        raise VasprunValidationError(
            "invalid_forces",
            f"expected force shape ({atom_count}, 3); found {shape}",
        )
    if not all(math.isfinite(value) for row in forces for value in row):
        raise VasprunValidationError("nonfinite_forces", "forces contain NaN or Inf")
    return run


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vasprun", type=Path, help="Path to vasprun.xml")
    parser.add_argument("--expected-atoms", type=int)
    parser.add_argument("--require-static", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        run = load_successful_vasprun(
            args.vasprun,
            expected_atoms=args.expected_atoms,
            require_static=args.require_static,
        )
    except VasprunValidationError as error:
        if not args.quiet:
            print(f"INVALID [{error.status}]: {error}")
        return 1

    if not args.quiet:
        electronic_steps = len(run.ionic_steps[-1]["electronic_steps"])
        max_force = max(
            math.sqrt(sum(component * component for component in force))
            for force in run.ionic_steps[-1]["forces"]
        )
        print(
            "SUCCESS: "
            f"{args.vasprun} | VASP {run.vasp_version} | "
            f"E={run.final_energy:.10f} eV | "
            f"SCF steps={electronic_steps} | max|F|={max_force:.6f} eV/A"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
