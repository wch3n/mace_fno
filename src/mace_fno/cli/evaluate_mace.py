"""Evaluate one frozen MACE checkpoint on independent XYZ data.

Besides aggregate energy and force errors, this reports each chemical formula
separately and tests global and formula-specific constant energy offsets fitted
only on the training set. These controls distinguish a spatially learned
correction from a trivial composition-dependent energy calibration.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--energy-key", default="REF_energy")
    parser.add_argument("--forces-key", default="REF_forces")
    parser.add_argument(
        "--group-key",
        help="Optional Atoms.info field used for additional grouped metrics",
    )
    parser.add_argument("--head")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def scalar_metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
    }


def reference_energy(atoms: Any, key: str) -> Any:
    """Read an energy without triggering a new calculator evaluation."""
    if key in atoms.info:
        return atoms.info[key]
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if key in results:
        return results[key]
    raise KeyError(key)


def reference_forces(atoms: Any, key: str) -> Any:
    """Read forces without triggering a new calculator evaluation."""
    if key in atoms.arrays:
        return atoms.arrays[key]
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if key in results:
        return results[key]
    raise KeyError(key)


def has_reference_labels(atoms: Any, energy_key: str, forces_key: str) -> bool:
    try:
        reference_energy(atoms, energy_key)
        reference_forces(atoms, forces_key)
    except KeyError:
        return False
    return True


def load_labeled(
    filename: Path, energy_key: str, forces_key: str
) -> tuple[list[Any], int]:
    structures = read(filename, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    labeled = [
        atoms
        for atoms in structures
        if has_reference_labels(atoms, energy_key, forces_key)
    ]
    skipped = len(structures) - len(labeled)
    if not labeled:
        raise ValueError(f"no fully labeled structures found in {filename}")
    return labeled, skipped


def predict(
    calculator: Any,
    structures: list[Any],
    energy_key: str,
    forces_key: str,
    label: str,
    group_key: str | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index, source in enumerate(structures):
        atoms = source.copy()
        atoms.calc = calculator
        target_energy = float(reference_energy(source, energy_key))
        target_forces = np.asarray(reference_forces(source, forces_key), dtype=float)
        predicted_energy = float(atoms.get_potential_energy())
        predicted_forces = np.asarray(atoms.get_forces(), dtype=float)
        record = {
            "formula": source.get_chemical_formula(),
            "num_atoms": len(source),
            "energy_error_per_atom": (predicted_energy - target_energy)
            / len(source),
            "force_errors": (predicted_forces - target_forces).reshape(-1),
        }
        if group_key is not None:
            if group_key not in source.info:
                raise KeyError(
                    f"{label} structure {index} lacks group key {group_key!r}"
                )
            record["benchmark_group"] = str(source.info[group_key])
        records.append(record)
        if (index + 1) % 100 == 0 or index + 1 == len(structures):
            print(f"{label}: predicted {index + 1}/{len(structures)}", flush=True)
    return records


def summarize(
    records: list[dict[str, Any]], shifts: dict[str, float] | None = None
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["formula"]].append(record)

    def one_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        energy_errors = np.asarray(
            [
                record["energy_error_per_atom"]
                + (shifts or {}).get(record["formula"], 0.0)
                for record in group
            ]
        )
        force_errors = np.concatenate([record["force_errors"] for record in group])
        return {
            "structures": len(group),
            "energy_eV_per_atom": scalar_metrics(energy_errors),
            "forces_eV_per_A": scalar_metrics(force_errors),
        }

    result = {
        "overall": one_group(records),
        "by_formula": {
            formula: one_group(group) for formula, group in sorted(grouped.items())
        },
    }
    if all("benchmark_group" in record for record in records):
        benchmark_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            benchmark_groups[record["benchmark_group"]].append(record)
        result["by_benchmark_group"] = {
            name: one_group(group)
            for name, group in sorted(benchmark_groups.items())
        }
    return result


def fitted_shifts(
    records: list[dict[str, Any]], by_formula: bool
) -> dict[str, float]:
    if not by_formula:
        correction = -float(
            np.mean([record["energy_error_per_atom"] for record in records])
        )
        return {record["formula"]: correction for record in records}

    errors: dict[str, list[float]] = defaultdict(list)
    for record in records:
        errors[record["formula"]].append(record["energy_error_per_atom"])
    return {formula: -float(np.mean(values)) for formula, values in errors.items()}


def print_summary(label: str, summary: dict[str, Any]) -> None:
    metrics = summary["overall"]
    energy = metrics["energy_eV_per_atom"]
    forces = metrics["forces_eV_per_A"]
    print(
        f"{label}: E_MAE={1000 * energy['mae']:.3f} meV/atom, "
        f"E_RMSE={1000 * energy['rmse']:.3f} meV/atom, "
        f"E_bias={1000 * energy['bias']:.3f} meV/atom, "
        f"F_MAE={1000 * forces['mae']:.3f} meV/A, "
        f"F_RMSE={1000 * forces['rmse']:.3f} meV/A",
        flush=True,
    )
    for formula, group in summary["by_formula"].items():
        energy = group["energy_eV_per_atom"]
        forces = group["forces_eV_per_A"]
        print(
            f"  {formula} (n={group['structures']}): "
            f"E_RMSE={1000 * energy['rmse']:.3f} meV/atom, "
            f"F_RMSE={1000 * forces['rmse']:.3f} meV/A",
            flush=True,
        )
    for name, group in summary.get("by_benchmark_group", {}).items():
        energy = group["energy_eV_per_atom"]
        forces = group["forces_eV_per_A"]
        print(
            f"  group={name} (n={group['structures']}): "
            f"E_RMSE={1000 * energy['rmse']:.3f} meV/atom, "
            f"F_RMSE={1000 * forces['rmse']:.3f} meV/A",
            flush=True,
        )


def main() -> None:
    args = parse_arguments()
    from mace.calculators import MACECalculator

    calculator_kwargs = {
        "model_paths": str(args.model),
        "device": args.device,
        "default_dtype": args.dtype,
    }
    if args.head is not None:
        calculator_kwargs["head"] = args.head
    calculator = MACECalculator(**calculator_kwargs)

    train, skipped_train = load_labeled(
        args.train_file, args.energy_key, args.forces_key
    )
    test, skipped_test = load_labeled(args.test_file, args.energy_key, args.forces_key)
    print(
        f"model={args.model}; head={args.head}; "
        f"train={len(train)} (skipped {skipped_train}); "
        f"test={len(test)} (skipped {skipped_test})",
        flush=True,
    )
    train_records = predict(
        calculator,
        train,
        args.energy_key,
        args.forces_key,
        "train",
        args.group_key,
    )
    test_records = predict(
        calculator,
        test,
        args.energy_key,
        args.forces_key,
        "test",
        args.group_key,
    )

    global_shifts = fitted_shifts(train_records, by_formula=False)
    formula_shifts = fitted_shifts(train_records, by_formula=True)
    results = {
        "model": str(args.model),
        "head": args.head,
        "group_key": args.group_key,
        "train_file": str(args.train_file),
        "test_file": str(args.test_file),
        "skipped_train": skipped_train,
        "skipped_test": skipped_test,
        "train": summarize(train_records),
        "test": summarize(test_records),
        "test_global_offset": summarize(test_records, global_shifts),
        "test_formula_offsets": summarize(test_records, formula_shifts),
        "global_shift_eV_per_atom": next(iter(global_shifts.values())),
        "formula_shifts_eV_per_atom": formula_shifts,
    }
    print_summary("train", results["train"])
    print_summary("test", results["test"])
    print_summary("test + train-fitted global offset", results["test_global_offset"])
    print_summary(
        "test + train-fitted formula offsets", results["test_formula_offsets"]
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
