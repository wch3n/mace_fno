#!/usr/bin/env python3
"""Download, audit, and split the published LLZO NEP/QNEP data.

The Zenodo record supplies one 1,978-configuration file and no official
train/test partition or trajectory identifiers.  The default reproduces the
original deterministic 80:10:10 frame-level split stratified by cell shape.
An optional source-order blocked split keeps contiguous groups of published
frames together as a conservative surrogate for a trajectory-level holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write

RECORD_ID = 18335947
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
SOURCE_FILE = "reference-structures-LiLaZrO-PBEsol.xyz"
FILES = {
    SOURCE_FILE: {
        "md5": "b3d29b0074577773f0bacfd043393d58",
        "size": 40_566_150,
        "required": True,
    },
    "models/nep-LiLaZrO-PBEsol.txt": {
        "md5": "e583aeb372b5fc83c1d97d6a7c8fad78",
        "size": 94_750,
        "required": False,
    },
    "models/qnep-mode1-LiLaZrO-PBEsol.txt": {
        "md5": "f7d57a15c8f09f34471521a3768765c6",
        "size": 96_694,
        "required": False,
    },
    "models/qnep-mode2-LiLaZrO-PBEsol.txt": {
        "md5": "9d3407840d0e37de20559c20abad7d37",
        "size": 96_694,
        "required": False,
    },
}
EXPECTED_COMPOSITION = Counter({"Li": 56, "La": 24, "Zr": 16, "O": 96})


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "llzo_qnep",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-published-models",
        action="store_true",
        help="Also fetch the published NEP and two QNEP parameter files.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--split-method",
        choices=("frame-stratified", "source-blocked"),
        default="frame-stratified",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
        help=(
            "Number of contiguous source frames kept together by the "
            "source-blocked split"
        ),
    )
    parser.add_argument(
        "--prepared-name",
        default="prepared",
        help="Output directory name below data-root",
    )
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
        provenance[relative_name] = {"size": actual_size, "md5": actual_md5}
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


def classify_cell(cell: np.ndarray, tolerance: float = 2.0e-6) -> str:
    """Classify an aligned orthogonal cell by its pattern of side lengths."""
    diagonal = np.diag(cell)
    if not np.allclose(cell, np.diag(diagonal), atol=tolerance, rtol=tolerance):
        raise ValueError("cell is not aligned orthorhombic")
    lengths = np.abs(diagonal)
    equal = np.isclose(
        lengths.reshape(3, 1),
        lengths.reshape(1, 3),
        atol=tolerance,
        rtol=tolerance,
    )
    if bool(equal.all()):
        return "cubic"
    if bool(equal[np.triu_indices(3, k=1)].any()):
        return "tetragonal"
    return "orthorhombic"


def audit_structures(filename: Path) -> tuple[list[Any], list[str], dict[str, Any]]:
    structures = read(filename, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    if not structures:
        raise ValueError(f"no structures found in {filename}")

    groups: list[str] = []
    side_lengths = []
    volumes = []
    energies = []
    maximum_force = 0.0
    for index, atoms in enumerate(structures):
        if len(atoms) != 192:
            raise ValueError(f"{filename}: frame {index} has {len(atoms)} atoms")
        composition = Counter(atoms.get_chemical_symbols())
        if composition != EXPECTED_COMPOSITION:
            raise ValueError(
                f"{filename}: frame {index} has composition {dict(composition)}"
            )
        if not bool(np.all(atoms.pbc)):
            raise ValueError(f"{filename}: frame {index} is not fully periodic")
        cell = np.asarray(atoms.cell.array, dtype=float)
        determinant = float(np.linalg.det(cell))
        if not np.isfinite(cell).all() or determinant <= 0.0:
            raise ValueError(f"{filename}: frame {index} has an invalid cell")
        try:
            cell_group = classify_cell(cell)
        except ValueError as error:
            raise ValueError(f"{filename}: frame {index}: {error}") from error
        energy = reference_energy(atoms)
        forces = reference_forces(atoms)
        virial = np.asarray(atoms.info.get("virial"), dtype=float)
        if forces.shape != (192, 3):
            raise ValueError(f"{filename}: frame {index} force shape is {forces.shape}")
        if virial.shape != (3, 3):
            raise ValueError(f"{filename}: frame {index} virial shape is {virial.shape}")
        if not (
            np.isfinite(energy)
            and np.isfinite(forces).all()
            and np.isfinite(virial).all()
        ):
            raise ValueError(f"{filename}: frame {index} has non-finite labels")
        groups.append(cell_group)
        side_lengths.append(np.linalg.norm(cell, axis=1))
        volumes.append(determinant)
        energies.append(energy)
        maximum_force = max(maximum_force, float(np.abs(forces).max()))

    lengths = np.asarray(side_lengths)
    group_counts = Counter(groups)
    summary = {
        "frames": len(structures),
        "atoms_per_frame": 192,
        "formula": "Li56La24Zr16O96",
        "cell_class_counts": dict(sorted(group_counts.items())),
        "cell_lengths_A": {
            axis: {
                "minimum": float(lengths[:, index].min()),
                "median": float(np.median(lengths[:, index])),
                "maximum": float(lengths[:, index].max()),
            }
            for index, axis in enumerate(("a", "b", "c"))
        },
        "volume_A3": {
            "minimum": float(np.min(volumes)),
            "median": float(np.median(volumes)),
            "maximum": float(np.max(volumes)),
        },
        "energy_eV": {
            "minimum": float(np.min(energies)),
            "maximum": float(np.max(energies)),
        },
        "maximum_absolute_force_eV_per_A": maximum_force,
        "virial_labels": len(structures),
    }
    print(
        f"audited {filename.name}: {len(structures)} frames; "
        + ", ".join(f"{name}={count}" for name, count in sorted(group_counts.items())),
        flush=True,
    )
    return structures, groups, summary


def allocated_group_counts(groups: list[str], fraction: float) -> dict[str, int]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("split fractions must be between zero and one")
    counts = Counter(groups)
    target = round(fraction * len(groups))
    exact = {name: fraction * count for name, count in counts.items()}
    allocation = {name: int(np.floor(value)) for name, value in exact.items()}
    remaining = target - sum(allocation.values())
    order = sorted(counts, key=lambda name: (-(exact[name] % 1.0), name))
    for name in order[:remaining]:
        allocation[name] += 1
    return allocation


def stratified_split_indices(
    groups: list[str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one")
    validation_counts = allocated_group_counts(groups, validation_fraction)
    test_counts = allocated_group_counts(groups, test_fraction)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)

    generator = np.random.default_rng(seed)
    validation: list[int] = []
    test: list[int] = []
    train: list[int] = []
    for group in sorted(by_group):
        indices = np.asarray(by_group[group], dtype=int)
        generator.shuffle(indices)
        n_validation = validation_counts[group]
        n_test = test_counts[group]
        if n_validation + n_test >= len(indices):
            raise ValueError(f"split leaves no training structures for {group}")
        validation.extend(indices[:n_validation].tolist())
        test.extend(indices[n_validation : n_validation + n_test].tolist())
        train.extend(indices[n_validation + n_test :].tolist())
    return {
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
    }


def source_blocked_split_indices(
    groups: list[str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    block_size: int,
) -> dict[str, list[int]]:
    """Split whole contiguous source-order blocks without frame leakage.

    The source file does not identify trajectories.  Keeping fixed contiguous
    blocks intact therefore provides a transparent, reproducible surrogate
    without claiming knowledge that is absent from the published metadata.
    """
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one")
    if min(validation_fraction, test_fraction) <= 0.0:
        raise ValueError("split fractions must be between zero and one")
    if block_size < 2:
        raise ValueError("block_size must be at least two")

    blocks = [
        list(range(start, min(start + block_size, len(groups))))
        for start in range(0, len(groups), block_size)
    ]
    if len(blocks) < 3:
        raise ValueError("source-blocked split requires at least three blocks")
    n_validation = max(1, round(validation_fraction * len(blocks)))
    n_test = max(1, round(test_fraction * len(blocks)))
    if n_validation + n_test >= len(blocks):
        raise ValueError("split leaves no training blocks")

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(blocks)).tolist()
    block_assignments = {
        "validation": order[:n_validation],
        "test": order[n_validation : n_validation + n_test],
        "train": order[n_validation + n_test :],
    }
    split_indices = {
        split: sorted(
            index
            for block_index in selected_blocks
            for index in blocks[block_index]
        )
        for split, selected_blocks in block_assignments.items()
    }

    available_groups = set(groups)
    for split, indices in split_indices.items():
        represented = {groups[index] for index in indices}
        missing = sorted(available_groups - represented)
        if missing:
            raise ValueError(
                f"source-blocked {split} split omits cell classes {missing}; "
                "choose another seed or a smaller block size"
            )
    return split_indices


def label_split(
    structures: list[Any],
    groups: list[str],
    indices: list[int],
    split: str,
) -> list[Any]:
    selected = []
    for source_index in indices:
        atoms = structures[source_index]
        atoms.info["benchmark_split"] = split
        atoms.info["benchmark_group"] = groups[source_index]
        atoms.info["source_index"] = int(source_index)
        selected.append(atoms)
    return selected


def main() -> None:
    args = parse_arguments()
    if not args.prepared_name or Path(args.prepared_name).name != args.prepared_name:
        raise ValueError("prepared-name must be one directory name")
    data_root = args.data_root.expanduser().resolve()
    provenance = ensure_files(
        data_root,
        download=args.download,
        include_models=args.download_published_models,
    )
    structures, groups, source_summary = audit_structures(data_root / SOURCE_FILE)
    if args.split_method == "frame-stratified":
        split_indices = stratified_split_indices(
            groups,
            args.validation_fraction,
            args.test_fraction,
            args.seed,
        )
    else:
        split_indices = source_blocked_split_indices(
            groups,
            args.validation_fraction,
            args.test_fraction,
            args.seed,
            args.block_size,
        )
    prepared_structures = {
        split: label_split(structures, groups, indices, split)
        for split, indices in split_indices.items()
    }

    prepared = data_root / args.prepared_name
    outputs = {
        split: prepared / f"{split}.xyz"
        for split in ("train", "validation", "test")
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
            args.split_method,
            args.block_size if args.split_method == "source-blocked" else None,
            args.seed,
            args.validation_fraction,
            args.test_fraction,
            split_indices,
        )
        observed = (
            manifest["split"].get("method_key", "frame-stratified"),
            manifest["split"].get("block_size"),
            manifest["split"]["seed"],
            manifest["split"]["validation_fraction"],
            manifest["split"]["test_fraction"],
            manifest["split"]["indices"],
        )
        if observed != expected:
            raise FileExistsError(
                f"prepared split differs in {prepared}; rerun with --force"
            )
        print(f"prepared split already exists: {prepared}", flush=True)
        return

    prepared.mkdir(parents=True, exist_ok=True)
    for split, filename in outputs.items():
        write(filename, prepared_structures[split], format="extxyz")

    split_group_counts = {
        split: dict(
            sorted(Counter(groups[index] for index in indices).items())
        )
        for split, indices in split_indices.items()
    }
    manifest = {
        "source": {
            "zenodo_record": RECORD_ID,
            "url": RECORD_URL,
            "files": provenance,
            "audit": source_summary,
        },
        "split": {
            "method": (
                "cell-class-stratified deterministic frame split"
                if args.split_method == "frame-stratified"
                else "deterministic contiguous source-order block split"
            ),
            "method_key": args.split_method,
            "block_size": (
                args.block_size if args.split_method == "source-blocked" else None
            ),
            "trajectory_identifiers_available": False,
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "counts": {
                split: len(indices) for split, indices in split_indices.items()
            },
            "cell_class_counts": split_group_counts,
            "indices": split_indices,
        },
        "prepared": {
            split: {
                "path": str(filename.resolve()),
                "size": filename.stat().st_size,
                "md5": md5sum(filename),
            }
            for split, filename in outputs.items()
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        "wrote train/validation/test = "
        + "/".join(str(len(prepared_structures[name])) for name in outputs)
        + f" to {prepared}",
        flush=True,
    )


if __name__ == "__main__":
    main()
