#!/usr/bin/env python3
"""Evaluate published LLZO NEP/QNEP files on the deterministic benchmark test set.

These models were fitted to the full 1,978-structure source dataset. Their
numbers on our test partition are therefore contextual in-sample references,
not independent held-out estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from mace_fno.cli.evaluate_mace import (
    print_summary,
    reference_energy,
    reference_forces,
    summarize,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Published model label and parameter file; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_model(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"model must use NAME=PATH syntax: {specification!r}")
    name, raw_path = specification.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_file():
        raise ValueError(f"invalid model specification: {specification!r}")
    return name, path


def evaluate_model(
    name: str,
    model_path: Path,
    structures: list[Any],
) -> dict[str, Any]:
    try:
        from calorine.calculators import CPUNEP
    except ImportError as error:
        raise RuntimeError(
            "calorine is required for the optional published NEP/QNEP comparison"
        ) from error

    calculator = CPUNEP(str(model_path))
    records = []
    for index, source in enumerate(structures):
        atoms = source.copy()
        atoms.calc = calculator
        energy_error = float(atoms.get_potential_energy()) - float(
            reference_energy(source, "energy")
        )
        force_errors = np.asarray(atoms.get_forces(), dtype=float) - np.asarray(
            reference_forces(source, "forces"), dtype=float
        )
        records.append(
            {
                "formula": source.get_chemical_formula(),
                "num_atoms": len(source),
                "energy_error_per_atom": energy_error / len(source),
                "force_errors": force_errors.reshape(-1),
                "benchmark_group": str(source.info["benchmark_group"]),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(structures):
            print(f"{name}: predicted {index + 1}/{len(structures)}", flush=True)
    summary = summarize(records)
    print_summary(name, summary)
    return {
        "model": str(model_path),
        "test": summary,
    }


def main() -> None:
    args = parse_arguments()
    structures = read(args.test_file, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    if not structures:
        raise ValueError(f"no structures found in {args.test_file}")
    models = dict(parse_model(item) for item in args.model)
    if len(models) != len(args.model):
        raise ValueError("published model labels must be unique")
    report = {
        "test_file": str(args.test_file.resolve()),
        "structures": len(structures),
        "comparison_scope": (
            "contextual in-sample reference: published models used the full source set"
        ),
        "models": {
            name: evaluate_model(name, path, structures)
            for name, path in models.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
