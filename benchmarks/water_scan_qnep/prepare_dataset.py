#!/usr/bin/env python3
"""Prepare exact-geometry-disjoint Water-SCAN energy/force benchmarks.

Deduplicate the source training pool before its fit/validation split. Keep only
unique source-validation geometries absent from that pool for the final test.
Raw files are never rewritten; the manifest records every source occurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write

RECORD_ID = 18335947
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
SPLIT_PROTOCOL = "exact-geometry-disjoint-source-roles-v1"
ENERGY_LABEL_TOLERANCE = 1.0e-6  # eV per structure
FORCE_LABEL_TOLERANCE = 1.0e-5  # eV/angstrom per component
FILES = {
    "reference-structures-training-water-SCAN.xyz": {
        "md5": "1844300168cb9f2b6431595f12aee910",
        "size": 70_774_288,
        "required": True,
    },
    "reference-structures-validation-water-SCAN.xyz": {
        "md5": "656650c11254088a82fe5a1ee3428aec",
        "size": 23_785_963,
        "required": True,
    },
    "models/nep-water-SCAN.txt": {
        "md5": "f109fcb708709de6bd369227fa7d4111",
        "size": 37_052,
        "required": False,
    },
    "models/qnep-mode1-water-SCAN.txt": {
        "md5": "7d6f8c2e8b00b73868872674f7d45628",
        "size": 38_036,
        "required": False,
    },
    "models/qnep-mode2-water-SCAN.txt": {
        "md5": "3f072bd0bfd862a1a6ddc9bd46866418",
        "size": 38_036,
        "required": False,
    },
}


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "water_scan_qnep",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Prepared output directory; defaults to DATA_ROOT/prepared",
    )
    parser.add_argument(
        "--download-published-models",
        action="store_true",
        help="Also fetch the three published NEP/QNEP parameter files.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def md5sum(filename: Path) -> str:
    digest = hashlib.md5()
    with filename.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(relative_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://zenodo.org/records/{RECORD_ID}/files/"
        f"{Path(relative_name).name}?download=1"
    )
    temporary = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    print(f"downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_files(
    data_root: Path,
    *,
    download: bool,
    include_models: bool,
) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for relative_name, expected in FILES.items():
        if not expected["required"] and not include_models:
            continue
        filename = data_root / relative_name
        if not filename.is_file() and download:
            download_file(relative_name, filename)
        if not filename.is_file():
            raise FileNotFoundError(f"missing {filename}; rerun with --download")
        actual_size = filename.stat().st_size
        actual_md5 = md5sum(filename)
        if actual_size != expected["size"] or actual_md5 != expected["md5"]:
            raise ValueError(
                f"integrity check failed for {filename}: "
                f"size={actual_size}, md5={actual_md5}"
            )
        provenance[relative_name] = {
            "size": actual_size,
            "md5": actual_md5,
        }
        print(
            f"verified {relative_name}: {actual_size} bytes, md5={actual_md5}",
            flush=True,
        )
    return provenance


def reference_energy(atoms: Any) -> float:
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if "energy" not in results:
        raise KeyError("energy")
    return float(results["energy"])


def reference_forces(atoms: Any) -> np.ndarray:
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=float)
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if "forces" not in results:
        raise KeyError("forces")
    return np.asarray(results["forces"], dtype=float)


def check_structures(filename: Path) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    structures = read(filename, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    if not structures:
        raise ValueError(f"no structures found in {filename}")

    side_lengths = []
    energies = []
    for index, atoms in enumerate(structures):
        if len(atoms) != 384:
            raise ValueError(f"{filename}: frame {index} has {len(atoms)} atoms")
        composition = Counter(atoms.get_chemical_symbols())
        if composition != Counter({"H": 256, "O": 128}):
            raise ValueError(
                f"{filename}: frame {index} has composition {dict(composition)}"
            )
        if not bool(np.all(atoms.pbc)):
            raise ValueError(f"{filename}: frame {index} is not fully periodic")
        cell = np.asarray(atoms.cell.array, dtype=float)
        length = float(abs(np.linalg.det(cell)) ** (1.0 / 3.0))
        if length <= 0.0 or not np.allclose(
            cell, length * np.eye(3), atol=2.0e-6, rtol=2.0e-6
        ):
            raise ValueError(f"{filename}: frame {index} is not an aligned cubic cell")
        energy = reference_energy(atoms)
        forces = reference_forces(atoms)
        if forces.shape != (384, 3):
            raise ValueError(f"{filename}: frame {index} force shape is {forces.shape}")
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError(f"{filename}: frame {index} has non-finite labels")
        side_lengths.append(length)
        energies.append(energy)

    lengths = np.asarray(side_lengths)
    summary = {
        "frames": len(structures),
        "atoms_per_frame": 384,
        "formula": "H256O128",
        "cell_length_A": {
            "minimum": float(lengths.min()),
            "median": float(np.median(lengths)),
            "maximum": float(lengths.max()),
        },
        "energy_eV": {
            "minimum": float(np.min(energies)),
            "maximum": float(np.max(energies)),
        },
    }
    print(
        f"checked {filename.name}: {len(structures)} frames, "
        f"L={lengths.min():.6f}--{lengths.max():.6f} A",
        flush=True,
    )
    return structures, lengths, summary


def stratified_validation_indices(
    side_lengths: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> list[int]:
    if len(side_lengths) < 2:
        raise ValueError("at least two unique training geometries are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    validation_size = round(validation_fraction * len(side_lengths))
    validation_size = min(max(validation_size, 1), len(side_lengths) - 1)
    ordered = np.argsort(side_lengths, kind="stable")
    bins = np.array_split(ordered, validation_size)
    generator = np.random.default_rng(seed)
    selected = [int(group[generator.integers(len(group))]) for group in bins]
    return sorted(selected)


def canonical_atom_order(atoms: Any) -> np.ndarray:
    """Sort species and Cartesian coordinates without changing the structure."""
    positions = np.asarray(atoms.positions)
    return np.lexsort(
        (positions[:, 2], positions[:, 1], positions[:, 0], atoms.numbers)
    )


def geometry_fingerprint(atoms: Any) -> str:
    """Exact, atom-permutation-invariant species/position/cell/PBC identity.

    No coordinate rounding, structural relaxation, rotation alignment, or
    periodic-image matching is performed. This is not a near-duplicate test.
    """
    order = canonical_atom_order(atoms)
    rows = np.column_stack((atoms.numbers, atoms.positions))[order]
    values = np.concatenate((np.asarray(atoms.cell).ravel(), rows.ravel()))
    if not np.isfinite(values).all():
        raise ValueError("non-finite geometry cannot be fingerprinted")
    values = np.asarray(values, dtype="<f8")
    values[values == 0.0] = 0.0  # Normalize the sign bit of numerical zero.
    digest = hashlib.sha256()
    digest.update(np.asarray([len(atoms)], dtype="<i8").tobytes())
    digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def geometry_disjoint_split(
    official_train: list[Any],
    official_test: list[Any],
    validation_fraction: float,
    seed: int,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Keep first representatives, check labels, and preserve source roles."""
    sources = {"training": official_train, "validation": official_test}
    groups: dict[str, dict[str, list[int]]] = {}
    for source, structures in sources.items():
        for index, atoms in enumerate(structures):
            key = geometry_fingerprint(atoms)
            groups.setdefault(key, {"training": [], "validation": []})[source].append(
                index
            )

    max_energy_span = 0.0
    max_force_span = 0.0
    for key, occurrences in groups.items():
        members = [
            sources[source][index]
            for source, indices in occurrences.items()
            for index in indices
        ]
        energies = np.asarray([reference_energy(atoms) for atoms in members])
        forces = np.stack(
            [reference_forces(atoms)[canonical_atom_order(atoms)] for atoms in members]
        )
        if not np.isfinite(energies).all() or not np.isfinite(forces).all():
            raise ValueError(f"non-finite energy/force labels in geometry {key}")
        energy_span = float(np.ptp(energies))
        force_span = float(np.max(np.ptp(forces, axis=0)))
        max_energy_span = max(max_energy_span, energy_span)
        max_force_span = max(max_force_span, force_span)
        if energy_span > ENERGY_LABEL_TOLERANCE or force_span > FORCE_LABEL_TOLERANCE:
            raise ValueError(
                f"conflicting duplicate labels in geometry {key}: "
                f"energy span={energy_span} eV, force span={force_span} eV/A"
            )

    train_keys = [key for key, indices in groups.items() if indices["training"]]
    test_keys = [
        key
        for key, indices in groups.items()
        if indices["validation"] and not indices["training"]
    ]
    if not test_keys:
        raise ValueError("no source-validation geometries remain for an unseen test")
    lengths = np.asarray(
        [
            abs(np.linalg.det(official_train[groups[key]["training"][0]].cell))
            ** (1 / 3)
            for key in train_keys
        ]
    )
    val_local = set(stratified_validation_indices(lengths, validation_fraction, seed))
    split_keys = {
        "train": [
            key for index, key in enumerate(train_keys) if index not in val_local
        ],
        "validation": [
            key for index, key in enumerate(train_keys) if index in val_local
        ],
        "test": test_keys,
    }
    splits: dict[str, list[Any]] = {}
    source_indices: dict[str, list[int]] = {}
    assignments = {}
    for split, keys in split_keys.items():
        source = "validation" if split == "test" else "training"
        indices = [groups[key][source][0] for key in keys]
        # Copy labels too: Atoms.copy() alone does not retain its calculator.
        from ase.calculators.singlepoint import SinglePointCalculator

        splits[split] = []
        source_indices[split] = indices
        for output_index, (key, index) in enumerate(zip(keys, indices)):
            original = sources[source][index]
            atoms = original.copy()
            atoms.calc = (
                SinglePointCalculator(atoms, **original.calc.results)
                if original.calc is not None
                else None
            )
            atoms.info.update(
                benchmark_split=split,
                source_index=index,
                source_file=source,
                geometry_sha256=key,
            )
            splits[split].append(atoms)
            assignments[key] = {
                "split": split,
                "output_index": output_index,
                "representative_source": source,
                "representative_index": index,
            }

    shared = [
        key
        for key, indices in groups.items()
        if indices["training"] and indices["validation"]
    ]
    unique_source_validation = sum(
        bool(value["validation"]) for value in groups.values()
    )
    report = {
        "protocol": SPLIT_PROTOCOL,
        "identity": "SHA256 of exact float64 cell, PBC and sorted species/Cartesian positions",
        "scope": "exact geometry and atom permutations; not trajectory or near-duplicate independence",
        "representative_policy": "first occurrence; source training takes precedence; no label averaging",
        "source_counts": {name: len(value) for name, value in sources.items()},
        "unique_source_training": len(train_keys),
        "unique_source_validation": unique_source_validation,
        "unique_union": len(groups),
        "shared_unique_geometries": len(shared),
        "source_validation_entries_seen_in_training": sum(
            len(groups[key]["validation"]) for key in shared
        ),
        "extra_repeated_source_training_entries": len(official_train) - len(train_keys),
        "extra_repeated_source_validation_entries": len(official_test)
        - unique_source_validation,
        "label_consistency": {
            "max_energy_span_eV": max_energy_span,
            "max_force_component_span_eV_per_A": max_force_span,
            "energy_tolerance_eV": ENERGY_LABEL_TOLERANCE,
            "force_tolerance_eV_per_A": FORCE_LABEL_TOLERANCE,
        },
        "source_indices": source_indices,
        "counts": {name: len(value) for name, value in splits.items()},
        "pairwise_geometry_overlap": {
            f"{first}-{second}": len(set(split_keys[first]) & set(split_keys[second]))
            for first, second in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        },
        "geometry_groups": [
            {"sha256": key, "source_occurrences": value, **assignments[key]}
            for key, value in groups.items()
        ],
    }
    if any(report["pairwise_geometry_overlap"].values()):
        raise AssertionError("geometry leakage in prepared splits")
    return splits, report


def label_split(structures: list[Any], name: str, source_indices: list[int]) -> None:
    for atoms, source_index in zip(structures, source_indices):
        atoms.info["benchmark_split"] = name
        atoms.info["source_index"] = int(source_index)


def main() -> None:
    args = parse_arguments()
    data_root = args.data_root.resolve()
    provenance = ensure_files(
        data_root,
        download=args.download,
        include_models=args.download_published_models,
    )
    source_train = data_root / "reference-structures-training-water-SCAN.xyz"
    source_test = data_root / "reference-structures-validation-water-SCAN.xyz"
    official_train, _, train_summary = check_structures(source_train)
    official_test, _, test_summary = check_structures(source_test)

    splits, deduplication = geometry_disjoint_split(
        official_train, official_test, args.validation_fraction, args.seed
    )
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    validation_indices = deduplication["source_indices"]["validation"]
    prepared = (args.output_dir or data_root / "prepared").resolve()
    outputs = {
        "train": prepared / "train.xyz",
        "validation": prepared / "validation.xyz",
        "test": prepared / "test.xyz",
    }
    manifest_file = prepared / "split_manifest.json"
    if not args.force and any(path.exists() for path in outputs.values()):
        if not all(path.is_file() and path.stat().st_size for path in outputs.values()):
            raise FileExistsError(
                f"incomplete prepared data in {prepared}; rerun with --force"
            )
        if not manifest_file.is_file():
            raise FileExistsError(
                f"prepared files exist without {manifest_file}; rerun with --force"
            )
        manifest = json.loads(manifest_file.read_text())
        expected = (
            args.seed,
            args.validation_fraction,
            validation_indices,
            SPLIT_PROTOCOL,
        )
        observed = (
            manifest["split"]["seed"],
            manifest["split"]["validation_fraction"],
            manifest["split"]["validation_indices"],
            manifest.get("deduplication", {}).get("protocol"),
        )
        if observed != expected:
            raise FileExistsError(
                f"prepared split differs in {prepared}; use a new --output-dir "
                "to preserve prior benchmark evidence"
            )
        if manifest["source"]["files"] != provenance:
            raise FileExistsError("source provenance differs from the prepared split")
        for name, path in outputs.items():
            if md5sum(path) != manifest["prepared"][name]["md5"]:
                raise ValueError(f"prepared checksum mismatch: {path}")
        print(f"prepared split already exists: {prepared}", flush=True)
        return

    prepared.mkdir(parents=True, exist_ok=True)
    write(outputs["train"], train, format="extxyz")
    write(outputs["validation"], validation, format="extxyz")
    write(outputs["test"], test, format="extxyz")
    indices_file = prepared / f"validation_indices_seed{args.seed}.txt"
    indices_file.write_text("\n".join(map(str, validation_indices)) + "\n")

    manifest = {
        "source": {
            "zenodo_record": RECORD_ID,
            "url": RECORD_URL,
            "files": provenance,
            "official_training": train_summary,
            "official_validation": test_summary,
        },
        "split": {
            "method": "deduplicate source training, stratify its unique geometries by cell length; exclude source-training geometries from unique test",
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "validation_indices": validation_indices,
            "counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
        },
        "deduplication": deduplication,
        "prepared": {
            name: {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "md5": md5sum(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote train/validation/test = {len(train)}/"
        f"{len(validation)}/{len(test)} to {prepared}",
        flush=True,
    )


if __name__ == "__main__":
    main()
