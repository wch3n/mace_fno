"""Relax fixed-substrate adsorption endpoints and test an energy sign switch.

The command supports conventional MACE and frozen-MACE plus FNO checkpoints.
It relaxes only atoms of the requested element, writes every generated
artifact to an explicit output directory, and reports

    delta E = E(wetting) - E(non-wetting)

for the undoped and doped structures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import FIRE


STATE_DEFINITIONS = (
    ("1-undoped", "undoped", "non-wetting"),
    ("3-undoped", "undoped", "wetting"),
    ("1-doped", "doped", "non-wetting"),
    ("3-doped", "doped", "wetting"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structures-dir",
        type=Path,
        required=True,
        help="Directory containing 1/3-doped.xyz and 1/3-undoped.xyz",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-kind", choices=("mace-fno", "mace"), required=True
    )
    parser.add_argument("--mace-head")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--movable-element", default="Au")
    parser.add_argument("--energy-key", default="energy")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def get_reference_energy(atoms: Any, key: str) -> float:
    """Read an input energy without evaluating the attached calculator."""
    if key in atoms.info:
        return float(atoms.info[key])
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if key in results:
        return float(results[key])
    raise KeyError(f"reference energy key {key!r} is absent")


def movable_indices(atoms: Any, element: str) -> list[int]:
    indices = [atom.index for atom in atoms if atom.symbol == element]
    if not indices:
        raise ValueError(f"structure contains no movable {element} atoms")
    return indices


def force_max(forces: np.ndarray, indices: list[int]) -> float:
    selected = np.asarray(forces, dtype=float)[indices]
    return float(np.linalg.norm(selected, axis=1).max())


def energy_components(calculator: Any) -> dict[str, float]:
    return {
        key: float(calculator.results[key])
        for key in ("mace_energy", "residual_energy")
        if key in calculator.results
    }


def make_calculator(args: argparse.Namespace) -> Any:
    if args.model_kind == "mace-fno":
        from mace_fno import MACEFNOCalculator

        return MACEFNOCalculator(
            args.model,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            mace_head=args.mace_head,
        )

    from mace.calculators import MACECalculator

    kwargs: dict[str, Any] = {
        "model_paths": str(args.model),
        "device": args.device,
        "default_dtype": args.dtype,
    }
    if args.mace_head is not None:
        kwargs["head"] = args.mace_head
    return MACECalculator(**kwargs)


def prepare_output_directory(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"refusing to write into non-empty {path}; pass --force to overwrite"
        )
    path.mkdir(parents=True, exist_ok=True)


def relax_state(
    source_path: Path,
    state: str,
    composition: str,
    geometry: str,
    calculator: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source = read(source_path)
    reference = get_reference_energy(source, args.energy_key)
    moving = movable_indices(source, args.movable_element)
    moving_set = set(moving)
    fixed = [index for index in range(len(source)) if index not in moving_set]

    atoms = source.copy()
    atoms.set_constraint(FixAtoms(indices=fixed))
    atoms.calc = calculator

    initial_energy = float(atoms.get_potential_energy())
    initial_forces = np.asarray(atoms.get_forces(), dtype=float)
    initial_components = energy_components(calculator)
    initial_positions = np.asarray(atoms.positions[moving], dtype=float).copy()

    optimizer = FIRE(
        atoms,
        logfile=str(args.output_dir / f"{state}.log"),
        trajectory=str(args.output_dir / f"{state}.traj"),
    )
    converged = bool(optimizer.run(fmax=args.fmax, steps=args.steps))

    final_energy = float(atoms.get_potential_energy())
    final_forces = np.asarray(atoms.get_forces(), dtype=float)
    final_components = energy_components(calculator)
    displacement = np.asarray(atoms.positions[moving], dtype=float) - initial_positions

    saved = atoms.copy()
    saved.set_constraint()
    saved.calc = SinglePointCalculator(
        saved, energy=final_energy, forces=final_forces
    )
    write(args.output_dir / f"{state}-relaxed.xyz", saved, format="extxyz")

    record: dict[str, Any] = {
        "state": state,
        "composition": composition,
        "geometry": geometry,
        "source": str(source_path.resolve()),
        "num_atoms": len(atoms),
        "movable_indices": moving,
        "fixed_indices": fixed,
        "reference_energy_eV": reference,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "initial_fmax_eV_per_A": force_max(initial_forces, moving),
        "final_fmax_eV_per_A": force_max(final_forces, moving),
        "optimizer_steps": int(optimizer.get_number_of_steps()),
        "converged": converged,
        "movable_rms_displacement_A": float(
            np.sqrt(np.mean(np.square(displacement)))
        ),
        "initial_components_eV": initial_components,
        "final_components_eV": final_components,
    }
    print_state(record)
    return record


def sign_name(value: float, tolerance: float = 1.0e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def summarize_switch(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for composition in ("undoped", "doped"):
        non_wetting = records[f"1-{composition}"]
        wetting = records[f"3-{composition}"]
        reference_delta = 1000.0 * (
            wetting["reference_energy_eV"] - non_wetting["reference_energy_eV"]
        )
        predicted_delta = 1000.0 * (
            wetting["final_energy_eV"] - non_wetting["final_energy_eV"]
        )
        summary[composition] = {
            "definition": "E(wetting) - E(non-wetting)",
            "reference_delta_meV": reference_delta,
            "predicted_delta_meV": predicted_delta,
            "error_meV": predicted_delta - reference_delta,
            "reference_sign": sign_name(reference_delta),
            "predicted_sign": sign_name(predicted_delta),
            "sign_match": sign_name(predicted_delta) == sign_name(reference_delta),
        }
    summary["sign_reversal_reproduced"] = (
        summary["undoped"]["predicted_sign"] == "positive"
        and summary["doped"]["predicted_sign"] == "negative"
    )
    return summary


def print_state(record: dict[str, Any]) -> None:
    print(
        f"{record['state']:<12} | "
        f"E_DFT={record['reference_energy_eV']:>18.6f} eV | "
        f"E_final={record['final_energy_eV']:>18.6f} eV | "
        f"steps={record['optimizer_steps']:>4d} | "
        f"Fmax={record['final_fmax_eV_per_A']:>9.5f} eV/A | "
        f"converged={str(record['converged']):<5}",
        flush=True,
    )


def print_switch(summary: dict[str, Any]) -> None:
    print("\nDelta E = E(wetting) - E(non-wetting)", flush=True)
    for composition in ("undoped", "doped"):
        result = summary[composition]
        print(
            f"{composition:<9} | "
            f"DFT={result['reference_delta_meV']:>10.3f} meV | "
            f"model={result['predicted_delta_meV']:>10.3f} meV | "
            f"error={result['error_meV']:>10.3f} meV | "
            f"sign_match={str(result['sign_match']):<5}",
            flush=True,
        )
    print(
        f"sign_reversal_reproduced={summary['sign_reversal_reproduced']}",
        flush=True,
    )


def main() -> None:
    args = parse_arguments()
    prepare_output_directory(args.output_dir, args.force)
    calculator = make_calculator(args)

    records = {}
    for state, composition, geometry in STATE_DEFINITIONS:
        source_path = args.structures_dir / f"{state}.xyz"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        records[state] = relax_state(
            source_path,
            state,
            composition,
            geometry,
            calculator,
            args,
        )

    summary = summarize_switch(records)
    results = {
        "model": str(args.model.resolve()),
        "model_kind": args.model_kind,
        "mace_head": args.mace_head,
        "device": args.device,
        "dtype": args.dtype,
        "optimizer": "ASE FIRE",
        "fmax_eV_per_A": args.fmax,
        "maximum_steps": args.steps,
        "movable_element": args.movable_element,
        "records": records,
        "summary": summary,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print_switch(summary)
    print(f"wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
