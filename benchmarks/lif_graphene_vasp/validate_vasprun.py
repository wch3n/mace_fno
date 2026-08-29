#!/usr/bin/env python3
"""Validate that a VASP static calculation completed with usable results.

The command exits with status zero only when ``vasprun.xml`` is complete,
parseable, electronically and ionically converged, and contains finite energy
and force data.  It is shared by the Slurm restart/skip logic and the result
collector so they use exactly the same definition of a successful run.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from pymatgen.io.vasp.outputs import Vasprun


class VasprunValidationError(RuntimeError):
    """A ``vasprun.xml`` file is missing, incomplete, or unsuccessful."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _has_closing_modeling_tag(path: Path) -> bool:
    """Check for the final XML tag without loading a potentially large file."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8192))
        tail = handle.read().rstrip()
    return tail.endswith(b"</modeling>")


def load_successful_vasprun(
    path: Path,
    *,
    expected_atoms: int | None = None,
    require_static: bool = False,
) -> Vasprun:
    """Return a validated ``Vasprun`` or raise ``VasprunValidationError``."""
    path = Path(path)
    if not path.is_file():
        raise VasprunValidationError("missing", f"missing {path}")
    if path.stat().st_size == 0:
        raise VasprunValidationError("empty", f"empty {path}")
    if not _has_closing_modeling_tag(path):
        raise VasprunValidationError(
            "incomplete_xml", f"{path} does not end with </modeling>"
        )

    try:
        run = Vasprun(
            path,
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=True,
        )
    except Exception as error:
        raise VasprunValidationError(
            "parse_error", f"could not parse {path}: {error}"
        ) from error

    if not run.ionic_steps:
        raise VasprunValidationError("no_results", f"{path} has no ionic step")
    if not run.converged_electronic:
        nelm = run.parameters.get("NELM", "unknown")
        electronic_steps = len(run.ionic_steps[-1].get("electronic_steps", []))
        raise VasprunValidationError(
            "unconverged_electronic",
            f"electronic SCF did not converge ({electronic_steps}/{nelm} steps)",
        )
    if not run.converged_ionic:
        raise VasprunValidationError("unconverged_ionic", "ionic run did not converge")

    nsw = int(run.parameters.get("NSW", 0))
    ibrion = int(run.parameters.get("IBRION", -1))
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

    energy = float(run.final_energy)
    if not math.isfinite(energy):
        raise VasprunValidationError("nonfinite_energy", "final energy is not finite")

    forces = np.asarray(run.ionic_steps[-1].get("forces"), dtype=float)
    if forces.shape != (atom_count, 3):
        raise VasprunValidationError(
            "invalid_forces",
            f"expected force shape ({atom_count}, 3); found {forces.shape}",
        )
    if not np.isfinite(forces).all():
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
        max_force = float(np.linalg.norm(run.ionic_steps[-1]["forces"], axis=1).max())
        print(
            "SUCCESS: "
            f"{args.vasprun} | VASP {run.vasp_version} | "
            f"E={float(run.final_energy):.10f} eV | "
            f"SCF steps={electronic_steps} | max|F|={max_force:.6f} eV/A"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
