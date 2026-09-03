"""Dataset, graph-batching, and cache utilities for residual training."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

CACHE_FORMAT_VERSION = 2


def _is_positive_isotropic_scaling(
    cell: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> bool:
    """Return whether the cell is a positive uniform scaling of the reference."""
    reference_norm = reference.square().sum()
    if bool(reference_norm <= 0):
        return False
    scale = (cell * reference).sum() / reference_norm
    if bool(scale <= 0):
        return False
    return bool(torch.allclose(cell, scale * reference, atol=atol, rtol=rtol))


def _is_finite_nonsingular_cell(cell: torch.Tensor) -> bool:
    """Return whether a cell is finite and has nonzero volume."""
    if cell.shape != (3, 3) or not bool(torch.isfinite(cell).all()):
        return False
    determinant = torch.linalg.det(cell)
    return bool(
        torch.isfinite(determinant)
        & (determinant.abs() > torch.finfo(cell.dtype).eps)
    )


def clone_graph(
    data: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Return device-local leaves because MACE marks positions for gradients."""
    cloned: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            cloned[key] = value
            continue
        target_dtype = (
            dtype if dtype is not None and value.is_floating_point() else None
        )
        cloned[key] = value.detach().clone().to(device=device, dtype=target_dtype)
    return cloned


def batch_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine single-graph MACE dictionaries without rebuilding neighbors."""
    if not graphs:
        raise ValueError("at least one graph is required")
    keys = set(graphs[0])
    if any(set(graph) != keys for graph in graphs[1:]):
        raise ValueError("all graph dictionaries must contain the same keys")
    counts = [int(graph["positions"].shape[0]) for graph in graphs]
    offsets = []
    running = 0
    for count in counts:
        offsets.append(running)
        running += count

    combined: dict[str, Any] = {}
    for key in sorted(keys - {"batch", "ptr"}):
        values = [graph[key] for graph in graphs]
        if not all(isinstance(value, torch.Tensor) for value in values):
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"non-tensor graph field {key!r} differs")
            combined[key] = values[0]
        elif key == "edge_index":
            combined[key] = torch.cat(
                [value + offset for value, offset in zip(values, offsets)], dim=1
            )
        elif values[0].ndim == 0:
            combined[key] = torch.stack(values)
        else:
            combined[key] = torch.cat(values, dim=0)

    device = graphs[0]["positions"].device
    combined["batch"] = torch.repeat_interleave(
        torch.arange(len(graphs), dtype=torch.long, device=device),
        torch.tensor(counts, dtype=torch.long, device=device),
    )
    combined["ptr"] = torch.tensor(
        [0, *[offset + count for offset, count in zip(offsets, counts)]],
        dtype=torch.long,
        device=device,
    )
    return combined


def collate_samples(
    samples: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Batch cached samples and transfer their graph and labels to a device."""
    graph = clone_graph(
        batch_graphs([sample["data"] for sample in samples]), device, dtype
    )
    energy = torch.cat([sample["energy"] for sample in samples]).to(device=device)
    forces = torch.cat([sample["forces"] for sample in samples]).to(device=device)
    return graph, energy, forces


def reference_energy(atoms: Any, key: str) -> Any:
    """Read an energy from either an XYZ info field or ASE calculator results."""
    if key in atoms.info:
        return atoms.info[key]
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if key in results:
        return results[key]
    raise KeyError(key)


def reference_forces(atoms: Any, key: str) -> Any:
    """Read forces from either an XYZ array or ASE calculator results."""
    if key in atoms.arrays:
        return atoms.arrays[key]
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if key in results:
        return results[key]
    raise KeyError(key)


def has_reference_labels(atoms: Any, energy_key: str, forces_key: str) -> bool:
    """Return whether both requested labels are available without recalculation."""
    try:
        reference_energy(atoms, energy_key)
        reference_forces(atoms, forces_key)
    except KeyError:
        return False
    return True


def sample_cache_metadata(
    *,
    filename: Path,
    mace_model: Path,
    energy_key: str,
    forces_key: str,
    dtype: torch.dtype,
    num_atoms: int | None,
    allow_periodic_z: bool,
    skip_cell_mismatch: bool,
    spatial_scheme: str = "2d",
    cell_mode: str = "fixed",
) -> dict[str, Any]:
    """Build the compatibility fingerprint stored beside preprocessed samples."""
    source = filename.resolve()
    model = mace_model.resolve()
    source_stat = source.stat()
    model_stat = model.stat()
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "source": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "mace_model": str(model),
        "mace_model_size": model_stat.st_size,
        "mace_model_mtime_ns": model_stat.st_mtime_ns,
        "energy_key": energy_key,
        "forces_key": forces_key,
        "dtype": str(dtype),
        "num_atoms": num_atoms,
        "allow_periodic_z": allow_periodic_z,
        "skip_cell_mismatch": skip_cell_mismatch,
        "spatial_scheme": spatial_scheme,
        "cell_mode": cell_mode,
    }
    if dtype == torch.float32:
        metadata["mixed_precision_atomic_energy"] = "float64-reference-v1"
    return metadata


def save_sample_cache(
    cache_file: Path | None,
    metadata: Mapping[str, Any],
    samples: list[dict[str, Any]],
    reference_cell: torch.Tensor,
) -> None:
    """Atomically write a preprocessed sample cache when a path is supplied."""
    if cache_file is None:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(f".{cache_file.name}.tmp-{os.getpid()}")
    torch.save(
        {
            "metadata": dict(metadata),
            "samples": samples,
            "reference_cell": reference_cell.detach().cpu(),
        },
        temporary,
    )
    temporary.replace(cache_file)
    print(f"wrote sample cache: {cache_file}", flush=True)


def load_or_create_samples(
    calculator: Any,
    filename: Path,
    energy_key: str,
    forces_key: str,
    dtype: torch.dtype,
    num_atoms: int | None,
    allow_periodic_z: bool,
    skip_cell_mismatch: bool,
    mace_model: Path,
    cache_file: Path | None,
    rebuild_cache: bool,
    reference_cell: torch.Tensor | None = None,
    spatial_scheme: str = "2d",
    cell_mode: str = "fixed",
) -> tuple[list[dict[str, Any]], torch.Tensor, dict[str, Any], bool]:
    """Load a compatible cache or construct samples from an extended XYZ file."""
    metadata = sample_cache_metadata(
        filename=filename,
        mace_model=mace_model,
        energy_key=energy_key,
        forces_key=forces_key,
        dtype=dtype,
        num_atoms=num_atoms,
        allow_periodic_z=allow_periodic_z,
        skip_cell_mismatch=skip_cell_mismatch,
        spatial_scheme=spatial_scheme,
        cell_mode=cell_mode,
    )
    if cache_file is not None and cache_file.is_file() and not rebuild_cache:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        if payload.get("metadata") == metadata:
            cached_cell = payload["reference_cell"].to(dtype=dtype)
            compared_cached = cached_cell if spatial_scheme == "3d" else cached_cell[:2]
            compared_reference = (
                (reference_cell if spatial_scheme == "3d" else reference_cell[:2])
                if reference_cell is not None
                else None
            )
            if compared_reference is not None:
                if cell_mode == "isotropic":
                    cells_match = _is_positive_isotropic_scaling(
                        compared_cached, compared_reference
                    )
                elif cell_mode == "anisotropic":
                    cells_match = _is_finite_nonsingular_cell(cached_cell)
                else:
                    cells_match = torch.allclose(
                        compared_cached, compared_reference, atol=1.0e-6, rtol=1.0e-6
                    )
                if not cells_match:
                    raise ValueError(f"cached cell mismatch in {cache_file}")
            print(f"loaded sample cache: {cache_file}", flush=True)
            return payload["samples"], cached_cell, metadata, True
        print(f"warning: rebuilding incompatible cache {cache_file}", flush=True)

    samples, loaded_cell = load_samples(
        calculator,
        filename,
        energy_key,
        forces_key,
        dtype,
        num_atoms,
        allow_periodic_z,
        skip_cell_mismatch,
        reference_cell=reference_cell,
        spatial_scheme=spatial_scheme,
        cell_mode=cell_mode,
    )
    return samples, loaded_cell, metadata, False


def load_samples(
    calculator: Any,
    filename: Path,
    energy_key: str,
    forces_key: str,
    dtype: torch.dtype,
    num_atoms: int | None,
    allow_periodic_z: bool,
    skip_cell_mismatch: bool,
    reference_cell: torch.Tensor | None = None,
    spatial_scheme: str = "2d",
    cell_mode: str = "fixed",
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Read labeled structures and build the corresponding MACE graph samples."""
    if cell_mode not in {"fixed", "isotropic", "anisotropic"}:
        raise ValueError(
            "cell_mode must be 'fixed', 'isotropic', or 'anisotropic'"
        )
    if cell_mode != "fixed" and spatial_scheme != "3d":
        raise ValueError("variable-cell modes apply only to the 3D scheme")
    configurations = read(filename, index=":")
    if not isinstance(configurations, list):
        configurations = [configurations]
    if not configurations:
        raise ValueError(f"no structures were read from {filename}")

    if num_atoms is not None:
        configurations = [atoms for atoms in configurations if len(atoms) == num_atoms]
        if not configurations:
            raise ValueError(f"no structures with {num_atoms} atoms were found")

    indexed_configurations = [
        (index, atoms)
        for index, atoms in enumerate(configurations)
        if has_reference_labels(atoms, energy_key, forces_key)
    ]
    skipped = len(configurations) - len(indexed_configurations)
    if skipped:
        print(
            f"warning: skipped {skipped} structures without both {energy_key!r} "
            f"and {forces_key!r} labels in {filename}",
            flush=True,
        )
    if not indexed_configurations:
        raise ValueError(
            f"no structures in {filename} have both {energy_key!r} and "
            f"{forces_key!r} labels"
        )

    if reference_cell is None:
        reference_cell = torch.as_tensor(
            indexed_configurations[0][1].cell.array, dtype=dtype
        )
    else:
        reference_cell = reference_cell.detach().to(device="cpu", dtype=dtype)

    samples: list[dict[str, Any]] = []
    periodic_z_count = 0
    cell_mismatch_count = 0
    for index, atoms in indexed_configurations:
        if not bool(atoms.pbc[0] and atoms.pbc[1]):
            raise ValueError(f"selected configuration {index} must be periodic in-plane")
        if spatial_scheme == "3d" and not bool(atoms.pbc[2]):
            raise ValueError(
                f"selected configuration {index} must be periodic in all three "
                "directions for the 3D FNO"
            )
        if spatial_scheme != "3d" and bool(atoms.pbc[2]):
            periodic_z_count += 1
            if not allow_periodic_z:
                raise ValueError(
                    "the input is periodic along z, whereas the residual is not; "
                    "pass --allow-periodic-z only when the vacuum separation makes "
                    "inter-slab long-range coupling negligible"
                )
        cell = torch.as_tensor(atoms.cell.array, dtype=dtype)
        compared_cell = cell if spatial_scheme == "3d" else cell[:2]
        compared_reference = (
            reference_cell if spatial_scheme == "3d" else reference_cell[:2]
        )
        if cell_mode == "isotropic":
            cells_match = _is_positive_isotropic_scaling(
                compared_cell, compared_reference
            )
        elif cell_mode == "anisotropic":
            cells_match = _is_finite_nonsingular_cell(cell)
        else:
            cells_match = torch.allclose(
                compared_cell, compared_reference, atol=1.0e-6, rtol=1.0e-6
            )
        if not cells_match:
            if skip_cell_mismatch:
                cell_mismatch_count += 1
                continue
            raise ValueError(
                f"cell mismatch at index {index}: cell_mode={cell_mode!r} "
                "requires the same relevant vectors (fixed) or a positive "
                "uniform scaling of the reference cubic cell (isotropic), or "
                "a finite nonsingular 3D cell (anisotropic)"
            )
        batch = calculator._atoms_to_batch(atoms)
        batch_dict = batch.to("cpu").to_dict()
        graph_keys = {
            "positions",
            "cell",
            "batch",
            "ptr",
            "node_attrs",
            "edge_index",
            "shifts",
            "head",
        }
        graph_keys.update(getattr(calculator.models[0], "embedding_specs", {}))
        missing_keys = graph_keys - set(batch_dict)
        if missing_keys:
            raise KeyError(f"MACE graph is missing required keys: {sorted(missing_keys)}")
        # Keep reference labels in float64 until the frozen baseline is subtracted.
        target_energy = torch.as_tensor(
            [reference_energy(atoms, energy_key)], dtype=torch.float64
        )
        target_forces = torch.as_tensor(
            np.asarray(reference_forces(atoms, forces_key)), dtype=torch.float64
        )
        sample = {
            "data": {key: batch_dict[key] for key in graph_keys},
            "energy": target_energy,
            "forces": target_forces,
            "num_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
        }
        if "benchmark_group" in atoms.info:
            sample["benchmark_group"] = str(atoms.info["benchmark_group"])
        samples.append(sample)
    if periodic_z_count and spatial_scheme != "3d":
        print(
            f"warning: {periodic_z_count} selected structures are periodic along z; "
            "the MACE baseline remains 3D-periodic but the FNO residual is periodic "
            "only along the first two lattice vectors",
            flush=True,
        )
    if cell_mismatch_count:
        print(
            f"warning: skipped {cell_mismatch_count} structures whose in-plane "
            f"cell differs from the reference cell in {filename}",
            flush=True,
        )
    if not samples:
        raise ValueError(f"no structures in {filename} match the selected cell")
    return samples, reference_cell


def split_samples(
    samples: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
    validation_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples deterministically or according to explicit validation indices."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= value < 1")
    if validation_indices is not None:
        if not validation_indices:
            raise ValueError("validation_indices must not be empty")
        if len(set(validation_indices)) != len(validation_indices):
            raise ValueError("validation_indices must not contain duplicates")
        if min(validation_indices) < 0 or max(validation_indices) >= len(samples):
            raise ValueError("a validation index is outside the sample range")
        if len(validation_indices) == len(samples):
            raise ValueError("validation_indices must leave at least one training sample")
        selected = set(validation_indices)
        train = [
            sample for index, sample in enumerate(samples) if index not in selected
        ]
        validation = [
            sample for index, sample in enumerate(samples) if index in selected
        ]
        return train, validation
    if validation_fraction == 0.0 or len(samples) == 1:
        return samples, []
    validation_size = max(1, round(validation_fraction * len(samples)))
    validation_size = min(validation_size, len(samples) - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(samples), generator=generator).tolist()
    selected = set(permutation[:validation_size])
    train = [sample for index, sample in enumerate(samples) if index not in selected]
    validation = [sample for index, sample in enumerate(samples) if index in selected]
    return train, validation
