#!/usr/bin/env python3
"""Download, audit, and split the published Water-SCAN/QNEP data.

The official training file is split into fit and model-selection subsets.
The official validation file remains untouched by model selection and is
renamed test.xyz in the prepared benchmark.
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


def audit_structures(filename: Path) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
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
        f"audited {filename.name}: {len(structures)} frames, "
        f"L={lengths.min():.6f}--{lengths.max():.6f} A",
        flush=True,
    )
    return structures, lengths, summary


def stratified_validation_indices(
    side_lengths: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> list[int]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    validation_size = round(validation_fraction * len(side_lengths))
    validation_size = min(max(validation_size, 1), len(side_lengths) - 1)
    ordered = np.argsort(side_lengths, kind="stable")
    bins = np.array_split(ordered, validation_size)
    generator = np.random.default_rng(seed)
    selected = [int(group[generator.integers(len(group))]) for group in bins]
    return sorted(selected)


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
    official_train, train_lengths, train_summary = audit_structures(source_train)
    official_test, _, test_summary = audit_structures(source_test)

    validation_indices = stratified_validation_indices(
        train_lengths, args.validation_fraction, args.seed
    )
    selected = set(validation_indices)
    train_indices = [
        index for index in range(len(official_train)) if index not in selected
    ]
    train = [official_train[index] for index in train_indices]
    validation = [official_train[index] for index in validation_indices]
    test_indices = list(range(len(official_test)))
    label_split(train, "train", train_indices)
    label_split(validation, "validation", validation_indices)
    label_split(official_test, "test", test_indices)

    prepared = data_root / "prepared"
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
        expected = (args.seed, args.validation_fraction, validation_indices)
        observed = (
            manifest["split"]["seed"],
            manifest["split"]["validation_fraction"],
            manifest["split"]["validation_indices"],
        )
        if observed != expected:
            raise FileExistsError(
                f"prepared split differs in {prepared}; rerun with --force"
            )
        print(f"prepared split already exists: {prepared}", flush=True)
        return

    prepared.mkdir(parents=True, exist_ok=True)
    write(outputs["train"], train, format="extxyz")
    write(outputs["validation"], validation, format="extxyz")
    write(outputs["test"], official_test, format="extxyz")
    indices_file = prepared / f"validation_indices_seed{args.seed}.txt"
    indices_file.write_text("\n".join(map(str, validation_indices)) + "\n")

    manifest = {
        "source": {
            "zenodo_record": RECORD_ID,
            "url": RECORD_URL,
            "files": provenance,
            "official_training": train_summary,
            "official_validation_used_as_test": test_summary,
        },
        "split": {
            "method": "cell-length-stratified deterministic sampling",
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "validation_indices": validation_indices,
            "counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(official_test),
            },
        },
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
        f"{len(validation)}/{len(official_test)} to {prepared}",
        flush=True,
    )


if __name__ == "__main__":
    main()
