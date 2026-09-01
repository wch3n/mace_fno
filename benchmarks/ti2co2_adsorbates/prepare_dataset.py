#!/usr/bin/env python3
"""Audit, deduplicate, and split the Ti2CO2 adsorbate data.

The two published local split labels, ``1331`` and ``2332``, contain the same
underlying structures.  This preparation step verifies that equivalence,
removes exact configuration duplicates before splitting, and writes one
formula/temperature-stratified benchmark partition.  The current slab FNO is
fixed-cell, so every model in this benchmark uses the dominant in-plane cell;
the uniformly contracted-cell structures are counted in the manifest but are
excluded from all three model comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write


SPLIT_NAMES = ("train", "validation", "test")
SOURCE_FILES = (
    "pbe0-rvv10_1331_train.xyz",
    "pbe0-rvv10_1331_test.xyz",
    "pbe0-rvv10_2332_train.xyz",
    "pbe0-rvv10_2332_test.xyz",
)
EXPECTED_FORMULAS = {
    "C12O24Ti24",
    "C12O25Ti24",
    "C12HO25Ti24",
    "C12HO26Ti24",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256sum(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_key(atoms: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(np.round(atoms.positions, 8).astype(np.float64).tobytes())
    digest.update(np.round(atoms.cell.array, 8).astype(np.float64).tobytes())
    return digest.hexdigest()


def cell_key(atoms: Any) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(atoms.cell.array).reshape(-1), 8))


def reference_energy(atoms: Any) -> float:
    if "energy_dft" not in atoms.info:
        raise KeyError("energy_dft")
    return float(atoms.info["energy_dft"])


def reference_forces(atoms: Any) -> np.ndarray:
    if "forces_dft" not in atoms.arrays:
        raise KeyError("forces_dft")
    return np.asarray(atoms.arrays["forces_dft"], dtype=float)


def is_isolated_atom(atoms: Any) -> bool:
    return len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"


def audit_labeled(atoms: Any, source: str, index: int) -> None:
    formula = atoms.get_chemical_formula()
    if formula not in EXPECTED_FORMULAS:
        raise ValueError(f"{source}: frame {index} has unexpected formula {formula}")
    if not bool(np.all(atoms.pbc)):
        raise ValueError(f"{source}: frame {index} is not fully periodic")
    cell = np.asarray(atoms.cell.array, dtype=float)
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{source}: frame {index} has a non-aligned cell")
    if not np.isclose(cell[2, 2], 25.0, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{source}: frame {index} has c={cell[2, 2]:.8f} A")
    energy = reference_energy(atoms)
    forces = reference_forces(atoms)
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"{source}: frame {index} force shape is {forces.shape}")
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError(f"{source}: frame {index} contains non-finite labels")


def read_sources(source_root: Path) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    sources: dict[str, list[Any]] = {}
    provenance: dict[str, Any] = {}
    for name in SOURCE_FILES:
        filename = source_root / name
        if not filename.is_file():
            raise FileNotFoundError(filename)
        structures = read(filename, index=":")
        if not isinstance(structures, list):
            structures = [structures]
        sources[name] = structures
        provenance[name] = {
            "path": str(filename.resolve()),
            "size": filename.stat().st_size,
            "sha256": sha256sum(filename),
            "frames": len(structures),
        }
    return sources, provenance


def labeled_and_isolated(
    structures: list[Any], source_name: str
) -> tuple[list[Any], list[Any]]:
    labeled = []
    isolated = []
    for index, atoms in enumerate(structures):
        if is_isolated_atom(atoms):
            reference_energy(atoms)
            isolated.append(atoms)
            continue
        audit_labeled(atoms, source_name, index)
        labeled.append(atoms)
    return labeled, isolated


def canonical_map(structures: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for atoms in structures:
        grouped[configuration_key(atoms)].append(atoms)

    maximum_energy_difference = 0.0
    maximum_force_difference = 0.0
    metadata_conflicts = 0
    canonical: dict[str, Any] = {}
    for key, group in grouped.items():
        first = group[0]
        canonical[key] = first
        temperatures = {str(atoms.info.get("temperature")) for atoms in group}
        metadata_conflicts += int(len(temperatures) > 1)
        for duplicate in group[1:]:
            maximum_energy_difference = max(
                maximum_energy_difference,
                abs(reference_energy(duplicate) - reference_energy(first)),
            )
            maximum_force_difference = max(
                maximum_force_difference,
                float(
                    np.max(
                        np.abs(reference_forces(duplicate) - reference_forces(first))
                    )
                ),
            )
    if maximum_energy_difference > 1.0e-4 or maximum_force_difference > 1.0e-4:
        raise ValueError(
            "duplicate geometries have inconsistent labels: "
            f"max dE={maximum_energy_difference:.3e} eV, "
            f"max dF={maximum_force_difference:.3e} eV/A"
        )
    summary = {
        "input_frames": len(structures),
        "unique_configurations": len(canonical),
        "duplicate_groups": sum(len(group) > 1 for group in grouped.values()),
        "duplicate_extra_frames": sum(len(group) - 1 for group in grouped.values()),
        "temperature_metadata_conflicts": metadata_conflicts,
        "maximum_duplicate_energy_difference_eV": maximum_energy_difference,
        "maximum_duplicate_force_difference_eV_per_A": maximum_force_difference,
    }
    return canonical, summary


def verify_alternative_splits(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    if set(first) != set(second):
        raise ValueError(
            "the 1331 and 2332 sources do not contain the same unique geometries"
        )
    maximum_energy_difference = 0.0
    maximum_force_difference = 0.0
    for key in first:
        maximum_energy_difference = max(
            maximum_energy_difference,
            abs(reference_energy(first[key]) - reference_energy(second[key])),
        )
        maximum_force_difference = max(
            maximum_force_difference,
            float(
                np.max(
                    np.abs(
                        reference_forces(first[key]) - reference_forces(second[key])
                    )
                )
            ),
        )
    if maximum_energy_difference > 1.0e-4 or maximum_force_difference > 1.0e-4:
        raise ValueError(
            "the 1331 and 2332 labels disagree beyond the audit tolerance"
        )
    return {
        "unique_configurations": len(first),
        "maximum_energy_difference_eV": maximum_energy_difference,
        "maximum_force_difference_eV_per_A": maximum_force_difference,
    }


def select_dominant_cell(
    canonical: dict[str, Any],
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    counts = Counter(cell_key(atoms) for atoms in canonical.values())
    dominant = max(counts, key=lambda key: (counts[key], key))
    selected = sorted(
        ((key, atoms) for key, atoms in canonical.items() if cell_key(atoms) == dominant),
        key=lambda item: item[0],
    )
    cell = np.asarray(dominant, dtype=float).reshape(3, 3)
    summary = {
        "policy": "dominant fixed cell shared by every model",
        "selected_cell_A": cell.tolist(),
        "selected_configurations": len(selected),
        "excluded_cell_configurations": len(canonical) - len(selected),
        "observed_cells": [
            {
                "cell_A": np.asarray(key, dtype=float).reshape(3, 3).tolist(),
                "configurations": count,
            }
            for key, count in sorted(counts.items(), key=lambda item: -item[1])
        ],
    }
    return selected, summary


def split_counts(size: int, validation_fraction: float, test_fraction: float) -> tuple[int, int]:
    test_count = int(round(test_fraction * size))
    validation_count = int(round(validation_fraction * size))
    if size >= 10:
        test_count = max(test_count, 1)
    if size >= 20:
        validation_count = max(validation_count, 1)
    while test_count + validation_count >= size:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    return validation_count, test_count


def stratified_split(
    selected: list[tuple[str, Any]],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[tuple[str, Any]]]:
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one")

    strata: dict[tuple[str, str], list[tuple[str, Any]]] = defaultdict(list)
    for item in selected:
        atoms = item[1]
        stratum = (atoms.get_chemical_formula(), str(atoms.info.get("temperature")))
        strata[stratum].append(item)

    generator = np.random.default_rng(seed)
    result = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(strata):
        group = sorted(strata[stratum], key=lambda item: item[0])
        order = generator.permutation(len(group))
        shuffled = [group[int(index)] for index in order]
        validation_count, test_count = split_counts(
            len(group), validation_fraction, test_fraction
        )
        result["test"].extend(shuffled[:test_count])
        result["validation"].extend(
            shuffled[test_count : test_count + validation_count]
        )
        result["train"].extend(shuffled[test_count + validation_count :])

    for name in SPLIT_NAMES:
        result[name].sort(key=lambda item: item[0])
    return result


def copy_and_label(items: list[tuple[str, Any]], split: str) -> list[Any]:
    structures = []
    for key, source in items:
        atoms = source.copy()
        atoms.info["benchmark_id"] = f"ti2co2-{key[:16]}"
        atoms.info["benchmark_split"] = split
        atoms.info["benchmark_stratum"] = (
            f"{atoms.get_chemical_formula()}|{atoms.info.get('temperature')}"
        )
        structures.append(atoms)
    return structures


def counts_by(values: list[Any], field: str) -> dict[str, int]:
    if field == "formula":
        counter = Counter(atoms.get_chemical_formula() for atoms in values)
    else:
        counter = Counter(str(atoms.info.get(field)) for atoms in values)
    return dict(sorted(counter.items()))


def geometry_summary(values: list[Any]) -> dict[str, Any]:
    maximum_forces = []
    minimum_distances = []
    z_spans = []
    for atoms in values:
        maximum_forces.append(float(np.linalg.norm(reference_forces(atoms), axis=1).max()))
        distances = atoms.get_all_distances(mic=True)
        distances[distances < 1.0e-8] = np.inf
        minimum_distances.append(float(distances.min()))
        z_spans.append(float(np.ptp(atoms.positions[:, 2])))
    return {
        "maximum_reference_force_eV_per_A": {
            "median": float(np.median(maximum_forces)),
            "maximum": float(np.max(maximum_forces)),
            "configurations_above_20": sum(value > 20.0 for value in maximum_forces),
        },
        "minimum_pair_distance_A": {
            "minimum": float(np.min(minimum_distances)),
            "configurations_below_0p8": sum(value < 0.8 for value in minimum_distances),
        },
        "z_span_A": {
            "minimum": float(np.min(z_spans)),
            "maximum": float(np.max(z_spans)),
        },
    }


def atomic_references(structures: list[Any]) -> list[Any]:
    references: dict[str, Any] = {}
    for atoms in structures:
        if is_isolated_atom(atoms):
            references[atoms.get_chemical_formula()] = atoms
    expected = {"H", "C", "O", "Ti"}
    if set(references) != expected:
        raise ValueError(
            f"isolated references are {sorted(references)}, expected {sorted(expected)}"
        )
    return [references[element].copy() for element in ("H", "C", "O", "Ti")]


def write_atomic(filename: Path, structures: list[Any]) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(f".{filename.name}.tmp-{os.getpid()}")
    try:
        write(temporary, structures, format="extxyz")
        temporary.replace(filename)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_arguments()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sources, provenance = read_sources(source_root)

    labeled: dict[str, list[Any]] = {}
    isolated: dict[str, list[Any]] = {}
    for name, structures in sources.items():
        labeled[name], isolated[name] = labeled_and_isolated(structures, name)

    split_1331 = labeled["pbe0-rvv10_1331_train.xyz"] + labeled[
        "pbe0-rvv10_1331_test.xyz"
    ]
    split_2332 = labeled["pbe0-rvv10_2332_train.xyz"] + labeled[
        "pbe0-rvv10_2332_test.xyz"
    ]
    canonical_1331, duplicate_1331 = canonical_map(split_1331)
    canonical_2332, duplicate_2332 = canonical_map(split_2332)
    equivalence = verify_alternative_splits(canonical_1331, canonical_2332)
    selected, cell_summary = select_dominant_cell(canonical_1331)
    partitions = stratified_split(
        selected,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    prepared = {
        name: copy_and_label(partitions[name], name) for name in SPLIT_NAMES
    }
    references = atomic_references(
        isolated["pbe0-rvv10_1331_train.xyz"]
        + isolated["pbe0-rvv10_2332_train.xyz"]
    )
    for atoms in references:
        atoms.info["benchmark_id"] = f"isolated-{atoms.get_chemical_formula()}"
        atoms.info["benchmark_split"] = "train"
    prepared["train"].extend(references)

    output_files = {name: output_root / f"{name}.xyz" for name in SPLIT_NAMES}
    manifest_file = output_root / "split_manifest.json"
    request = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "source_sha256": {name: item["sha256"] for name, item in provenance.items()},
    }
    if not args.force and manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("request") != request:
            raise FileExistsError(
                f"prepared data in {output_root} use another request; pass --force"
            )
        for name, filename in output_files.items():
            expected = manifest["outputs"][name]["sha256"]
            if not filename.is_file() or sha256sum(filename) != expected:
                raise ValueError(f"prepared output failed integrity check: {filename}")
        print(f"verified existing prepared data in {output_root}", flush=True)
        return
    if not args.force and any(path.exists() for path in output_files.values()):
        raise FileExistsError(
            f"prepared outputs exist without a matching manifest in {output_root}"
        )

    for name, filename in output_files.items():
        write_atomic(filename, prepared[name])

    labeled_prepared = {
        name: [atoms for atoms in values if not is_isolated_atom(atoms)]
        for name, values in prepared.items()
    }
    manifest = {
        "request": request,
        "source": provenance,
        "audit": {
            "duplicates_1331": duplicate_1331,
            "duplicates_2332": duplicate_2332,
            "alternative_split_equivalence": equivalence,
            "cell_selection": cell_summary,
            "geometry": geometry_summary(
                [atoms for _, atoms in selected]
            ),
            "isolated_atomic_energies_eV": {
                atoms.get_chemical_formula(): reference_energy(atoms)
                for atoms in references
            },
        },
        "split": {
            "method": "deduplicated formula-and-temperature-stratified sampling",
            "counts": {
                name: {
                    "labeled_configurations": len(labeled_prepared[name]),
                    "isolated_atomic_references": sum(
                        is_isolated_atom(atoms) for atoms in prepared[name]
                    ),
                    "by_formula": counts_by(labeled_prepared[name], "formula"),
                    "by_temperature": counts_by(labeled_prepared[name], "temperature"),
                }
                for name in SPLIT_NAMES
            },
        },
        "outputs": {
            name: {
                "path": str(filename.resolve()),
                "size": filename.stat().st_size,
                "sha256": sha256sum(filename),
            }
            for name, filename in output_files.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    counts = manifest["split"]["counts"]
    print(
        "wrote labeled train/validation/test = "
        f"{counts['train']['labeled_configurations']}/"
        f"{counts['validation']['labeled_configurations']}/"
        f"{counts['test']['labeled_configurations']} to {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
