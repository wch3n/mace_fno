"""Train frozen MACE plus a 2D, hybrid 2.5D, or periodic 3D FNO residual.

The input should be an extended XYZ file containing reference total energies
and forces. Validation data are never used for gradients, and a separate test
file is evaluated only before and after optimization. Training and evaluation
use true multi-configuration batches for MACE and the particle-mesh residual.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from ase.io import read

from mace_fno import MACEFNOResidual, energy_force_loss


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument(
        "--validation-file",
        type=Path,
        help="Optional validation XYZ; otherwise split the training file",
    )
    parser.add_argument(
        "--validation-indices-file",
        type=Path,
        help=(
            "Optional zero-based training-set indices to use for validation; "
            "mutually exclusive with --validation-file"
        ),
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Held-out XYZ evaluated only before and after optimization",
    )
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore and replace compatible preprocessed sample caches",
    )
    parser.add_argument("--energy-key", default="REF_energy")
    parser.add_argument("--forces-key", default="REF_forces")
    parser.add_argument(
        "--head",
        help="MACE head name (needed only when a multi-head checkpoint is ambiguous)",
    )
    parser.add_argument(
        "--num-atoms",
        type=int,
        help="Keep one fixed-cell subset selected by its atom count",
    )
    parser.add_argument(
        "--allow-periodic-z",
        action="store_true",
        help=(
            "Accept 3D-periodic input with a 2D/2.5D residual that remains "
            "nonperiodic in z"
        ),
    )
    parser.add_argument(
        "--skip-cell-mismatch",
        action="store_true",
        help=(
            "Skip structures outside the reference in-plane cell (2D/2.5D) "
            "or complete cell (3D)"
        ),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument(
        "--spatial-scheme",
        choices=("auto", "2d", "2.5d", "3d"),
        default="auto",
        help=(
            "Mesh periodicity; auto preserves legacy behavior (2D without "
            "--z-grid, otherwise 2.5D)"
        ),
    )
    parser.add_argument(
        "--z-grid",
        type=int,
        default=0,
        help="Number of explicit z layers for the 2.5D or 3D scheme",
    )
    parser.add_argument(
        "--z-modes",
        type=int,
        default=0,
        help="Retained z Fourier modes for 3D; zero uses --modes",
    )
    parser.add_argument(
        "--z-extent",
        type=float,
        help="Physical width in angstrom of the nonperiodic z window",
    )
    parser.add_argument(
        "--z-center",
        choices=("mean", "cell"),
        default="mean",
        help="Centre the finite z window on each graph's mean height or cell centre",
    )
    parser.add_argument(
        "--lateral-interlacing",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "For 2.5D, average one mesh origin or a 2x2 set of half-grid "
            "origins to suppress lateral particle-mesh egg-box errors"
        ),
    )
    parser.add_argument(
        "--volume-interlacing",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "For 3D, average one mesh origin or a 2x2x2 set of half-grid "
            "origins to suppress particle-mesh egg-box errors"
        ),
    )
    parser.add_argument(
        "--planar-symmetry",
        choices=("none", "c4", "d4"),
        default="none",
        help=(
            "For square 2.5D cells, group-average the field operator to enforce "
            "fourfold rotations (c4) or rotations and reflections (d4)"
        ),
    )
    parser.add_argument(
        "--spectral-symmetry",
        choices=("none", "eqgino"),
        default="none",
        help=(
            "For a cubic 3D mesh and cell, use EqGINO-style full-FFT radial "
            "weight sharing to enforce signed-axis equivariance"
        ),
    )
    parser.add_argument(
        "--spectral-groups",
        type=int,
        default=1,
        help=(
            "Block-diagonal channel groups in the EqGINO spectral contraction; "
            "one retains dense channel mixing"
        ),
    )
    parser.add_argument(
        "--z-kernel-size",
        type=int,
        default=3,
        help="Odd nonperiodic z-CNN kernel size for the nonlinear 2.5D FNO",
    )
    parser.add_argument(
        "--z-mixing",
        choices=("local", "global"),
        default="local",
        help=(
            "Use a zero-padded local z CNN or channel-wise dense global "
            "nonperiodic z mixing"
        ),
    )
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--source-hidden-channels", type=int, default=64)
    parser.add_argument("--fno-hidden-channels", type=int, default=32)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument(
        "--architecture", choices=("linear", "nonlinear"), default="nonlinear"
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--output-initialization-scale",
        type=float,
        default=0.0,
        help=(
            "Scale the nonlinear FNO's random final-projection initialization; "
            "zero preserves the exact frozen-MACE start, while a small positive "
            "value enables upstream gradients from the first step"
        ),
    )
    parser.add_argument(
        "--output-warmup-steps",
        type=int,
        default=0,
        help=(
            "For a nonlinear FNO, train only the final output projection for "
            "this many initial optimizer steps before unfreezing the complete "
            "residual branch"
        ),
    )
    parser.add_argument(
        "--output-warmup-learning-rate",
        type=float,
        default=0.0,
        help=(
            "Learning rate during output-projection warm-up; zero uses the "
            "main --learning-rate"
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Optionally reduce the learning rate when validation stalls",
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience-evals", type=int, default=4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument(
        "--early-stopping-patience-evals",
        type=int,
        default=0,
        help=(
            "Stop after this many non-improving validation checks at the minimum "
            "learning rate; zero disables early stopping"
        ),
    )
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--force-scale", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument(
        "--evaluation-scope",
        choices=("all", "validation-test"),
        default="all",
        help="Skip full-train and null-control diagnostics in routine repeats",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="True minibatches accumulated before each optimizer update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Configurations evaluated together in one MACE and FNO forward pass",
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=0,
        help="Evaluation batch size; zero uses --batch-size",
    )
    parser.add_argument(
        "--random-residual-initialization",
        action="store_true",
        help="Do not initialize the combined model to the frozen-MACE baseline",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help=(
            "Model compute dtype; float32 retains and accumulates MACE atomic "
            "reference energies in float64"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def elapsed_since(start: float, device: torch.device) -> float:
    """Return a stage wall time after completing queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter() - start


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
        cloned[key] = (
            value.detach()
            .clone()
            .to(
                device=device,
                dtype=target_dtype,
            )
        )
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
    graph = clone_graph(
        batch_graphs([sample["data"] for sample in samples]),
        device,
        dtype,
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


CACHE_FORMAT_VERSION = 1


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
) -> dict[str, Any]:
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
) -> tuple[list[dict[str, Any]], torch.Tensor, dict[str, Any], bool]:
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
            if compared_reference is not None and not torch.allclose(
                compared_cached, compared_reference, atol=1.0e-6, rtol=1.0e-6
            ):
                raise ValueError(f"cached fixed-cell mismatch in {cache_file}")
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
) -> tuple[list[dict[str, Any]], torch.Tensor]:
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
            raise ValueError(
                f"selected configuration {index} must be periodic in-plane"
            )
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
        if not torch.allclose(
            compared_cell, compared_reference, atol=1.0e-6, rtol=1.0e-6
        ):
            if skip_cell_mismatch:
                cell_mismatch_count += 1
                continue
            raise ValueError(
                "this fixed-cell FNO experiment requires the same relevant "
                f"cell vectors in every configuration; mismatch at index {index}"
            )
        batch = calculator._atoms_to_batch(atoms)  # MACE's graph-construction path
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
        embedding_specs = getattr(calculator.models[0], "embedding_specs", {})
        graph_keys.update(embedding_specs)
        missing_keys = graph_keys - set(batch_dict)
        if missing_keys:
            raise KeyError(
                f"MACE graph is missing required keys: {sorted(missing_keys)}"
            )
        # Preserve labels in float64 even when the model runs in float32. Total
        # LES energies are large enough that float32 labels lose the sub-meV/atom
        # signal. Residual labels are cast only after the large frozen baseline
        # has been subtracted in float64.
        target_energy = torch.as_tensor(
            [reference_energy(atoms, energy_key)], dtype=torch.float64
        )
        target_forces = torch.as_tensor(
            np.asarray(reference_forces(atoms, forces_key)), dtype=torch.float64
        )
        samples.append(
            {
                "data": {key: batch_dict[key] for key in graph_keys},
                "energy": target_energy,
                "forces": target_forces,
                "num_atoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
            }
        )
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
            raise ValueError(
                "validation_indices must leave at least one training sample"
            )
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(samples), generator=generator).tolist()
    validation_indices = set(permutation[:validation_size])
    train = [
        sample
        for index, sample in enumerate(samples)
        if index not in validation_indices
    ]
    validation = [
        sample for index, sample in enumerate(samples) if index in validation_indices
    ]
    return train, validation


def initialize_zero_residual(model: MACEFNOResidual) -> None:
    """Make the initial combined prediction exactly equal to frozen MACE."""
    operator = model.long_range.field_operator
    if operator.architecture == "linear":
        for parameter in operator.parameters():
            parameter.data.zero_()
    else:
        operator.fno.projection_output.weight.data.zero_()


def initialize_scaled_residual_output(
    model: MACEFNOResidual,
    scale: float,
) -> None:
    """Scale the nonlinear output projection without freezing upstream layers."""
    if scale < 0.0:
        raise ValueError("output initialization scale must be non-negative")
    operator = model.long_range.field_operator
    if operator.architecture != "nonlinear":
        raise ValueError("scaled output initialization requires a nonlinear FNO")
    operator.fno.projection_output.weight.data.mul_(float(scale))


def configure_output_projection_warmup(
    model: MACEFNOResidual,
    warmup_steps: int,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Freeze the residual branch except its final projection during warm-up.

    The first returned list contains every parameter that was trainable before
    warm-up. Passing it to the optimizer allows later unfreezing without
    rebuilding the optimizer or discarding its state. The second list contains
    the output-projection parameters that remain active during warm-up.
    """
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if warmup_steps == 0:
        return parameters, []
    operator = model.long_range.field_operator
    if operator.architecture != "nonlinear":
        raise ValueError("output-projection warm-up requires a nonlinear FNO")
    projection_parameters = list(operator.fno.projection_output.parameters())
    if not projection_parameters:
        raise ValueError("the nonlinear FNO has no output-projection parameters")
    projection_ids = {id(parameter) for parameter in projection_parameters}
    for parameter in parameters:
        parameter.requires_grad_(id(parameter) in projection_ids)
    return parameters, projection_parameters


def finish_output_projection_warmup(
    parameters: list[torch.nn.Parameter],
) -> None:
    """Unfreeze every residual parameter captured before warm-up."""
    for parameter in parameters:
        parameter.requires_grad_(True)


def ensure_frozen_residual_targets(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> bool:
    """Cache frozen-MACE predictions and reference-minus-MACE labels."""
    required = {
        "base_energy",
        "base_forces",
        "residual_energy",
        "residual_forces",
    }
    if all(required <= set(sample) for sample in samples):
        return False
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model.backbone.mace_model.eval()
    model_dtype = next(model.parameters()).dtype
    for start in range(0, len(samples), batch_size):
        sample_batch = samples[start : start + batch_size]
        graph, _, _ = collate_samples(sample_batch, device, model_dtype)
        output = model.backbone.mace_model(
            graph,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
        )
        base_energy = (
            model.backbone.corrected_base_energy(graph, output)
            .detach()
            .cpu()
            .to(torch.float64)
        )
        base_forces = output["forces"].detach().cpu().to(torch.float64)
        node_start = 0
        for graph_index, sample in enumerate(sample_batch):
            node_stop = node_start + sample["num_atoms"]
            sample_base_energy = base_energy[graph_index : graph_index + 1]
            sample_base_forces = base_forces[node_start:node_stop]
            sample["base_energy"] = sample_base_energy
            sample["base_forces"] = sample_base_forces
            sample["residual_energy"] = sample["energy"] - sample_base_energy
            sample["residual_forces"] = sample["forces"] - sample_base_forces
            node_start = node_stop
        if (start + len(sample_batch)) % 500 == 0 or start + len(sample_batch) == len(
            samples
        ):
            print(
                f"cached frozen targets: {start + len(sample_batch)}/{len(samples)}",
                flush=True,
            )
    return True


def evaluate(
    model: MACEFNOResidual,
    samples: list[dict[str, Any]],
    *,
    baseline: bool = False,
    energy_shift_per_atom: float | Mapping[str, float] = 0.0,
    batch_size: int = 1,
) -> dict[str, Any]:
    if not samples:
        return {}
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    energy_errors = []
    force_errors = []
    group_energy_errors: dict[str, list[torch.Tensor]] = defaultdict(list)
    group_force_errors: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, len(samples), batch_size):
        sample_batch = samples[start : start + batch_size]
        graph, target_energy, target_forces = collate_samples(
            sample_batch, device, model_dtype
        )
        cached_base_energy = torch.cat(
            [sample["base_energy"] for sample in sample_batch]
        ).to(device=device)
        cached_base_forces = torch.cat(
            [sample["base_forces"] for sample in sample_batch]
        ).to(device=device)
        if baseline:
            predicted_energy = cached_base_energy
            predicted_forces = cached_base_forces
        else:
            output = model(
                graph,
                training=False,
                compute_force=False,
                compute_residual_force=True,
            )
            predicted_energy = cached_base_energy + output["residual_energy"]
            predicted_forces = cached_base_forces + output["residual_forces"]
        atoms_per_graph = torch.bincount(graph["batch"]).to(predicted_energy)
        if isinstance(energy_shift_per_atom, Mapping):
            shifts = predicted_energy.new_tensor(
                [
                    float(energy_shift_per_atom.get(sample["formula"], 0.0))
                    for sample in sample_batch
                ]
            )
        else:
            shifts = predicted_energy.new_full(
                predicted_energy.shape, float(energy_shift_per_atom)
            )
        predicted_energy = predicted_energy + shifts * atoms_per_graph
        energy_error = ((predicted_energy - target_energy) / atoms_per_graph).detach()
        force_error = (predicted_forces - target_forces).detach()
        energy_errors.append(energy_error)
        force_errors.append(force_error.reshape(-1))
        for graph_index, sample in enumerate(sample_batch):
            atom_mask = graph["batch"] == graph_index
            group_energy_errors[sample["formula"]].append(
                energy_error[graph_index : graph_index + 1]
            )
            group_force_errors[sample["formula"]].append(
                force_error[atom_mask].reshape(-1)
            )

    def metrics(
        energy_error_list: list[torch.Tensor],
        force_error_list: list[torch.Tensor],
    ) -> dict[str, float]:
        energy_error = torch.cat(energy_error_list)
        force_error = torch.cat(force_error_list)
        return {
            "energy_mae": energy_error.abs().mean().item(),
            "energy_rmse": energy_error.square().mean().sqrt().item(),
            "energy_bias": energy_error.mean().item(),
            "force_mae": force_error.abs().mean().item(),
            "force_rmse": force_error.square().mean().sqrt().item(),
        }

    result: dict[str, Any] = metrics(energy_errors, force_errors)
    result["by_formula"] = {
        formula: metrics(group_energy_errors[formula], group_force_errors[formula])
        for formula in sorted(group_energy_errors)
    }
    result["formula_counts"] = {
        formula: len(group_energy_errors[formula])
        for formula in sorted(group_energy_errors)
    }
    model.train(was_training)
    return result


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    print(
        f"{label}: E_MAE={1000.0 * metrics['energy_mae']:.4f} meV/atom, "
        f"E_RMSE={1000.0 * metrics['energy_rmse']:.4f} meV/atom, "
        f"E_bias={1000.0 * metrics['energy_bias']:.4f} meV/atom, "
        f"F_MAE={1000.0 * metrics['force_mae']:.4f} meV/A, "
        f"F_RMSE={1000.0 * metrics['force_rmse']:.4f} meV/A",
        flush=True,
    )
    for formula, group in metrics.get("by_formula", {}).items():
        print(
            f"  {formula} (n={metrics['formula_counts'][formula]}): "
            f"E_RMSE={1000.0 * group['energy_rmse']:.4f} meV/atom, "
            f"F_RMSE={1000.0 * group['force_rmse']:.4f} meV/A",
            flush=True,
        )


def residual_state_dict(model: MACEFNOResidual) -> dict[str, torch.Tensor]:
    """Save the learned branch without duplicating the frozen MACE checkpoint."""
    prefix = "backbone.mace_model."
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    }


def load_residual_state_dict(
    model: MACEFNOResidual, state: Mapping[str, torch.Tensor]
) -> None:
    """Restore a residual-only state while retaining the frozen backbone."""
    complete_state = model.state_dict()
    complete_state.update(
        {
            key: value.to(
                device=complete_state[key].device,
                dtype=complete_state[key].dtype,
            )
            for key, value in state.items()
        }
    )
    model.load_state_dict(complete_state)


def validation_objective(
    metrics: Mapping[str, Any],
    *,
    energy_weight: float,
    force_weight: float,
    energy_scale: float,
    force_scale: float,
) -> float:
    """Match the normalized energy/force objective used during fitting."""
    if not metrics:
        return float("inf")
    return (
        float(energy_weight)
        * (float(metrics["energy_rmse"]) / float(energy_scale)) ** 2
        + float(force_weight) * (float(metrics["force_rmse"]) / float(force_scale)) ** 2
    )


def main() -> None:
    total_start = perf_counter()
    args = parse_arguments()
    if args.steps < 1 or min(args.grid, args.modes, args.channels) < 1:
        raise ValueError("steps, grid, modes, and channels must be positive")
    if args.output_warmup_steps < 0 or args.output_warmup_steps >= args.steps:
        raise ValueError("output_warmup_steps must satisfy 0 <= warm-up < steps")
    if args.output_warmup_learning_rate < 0.0:
        raise ValueError("output_warmup_learning_rate must be non-negative")
    if args.output_initialization_scale < 0.0:
        raise ValueError("output_initialization_scale must be non-negative")
    if args.output_warmup_steps and args.architecture != "nonlinear":
        raise ValueError("output-projection warm-up requires --architecture nonlinear")
    if args.output_initialization_scale and args.architecture != "nonlinear":
        raise ValueError("scaled output initialization requires --architecture nonlinear")
    if args.output_initialization_scale and args.random_residual_initialization:
        raise ValueError(
            "--output-initialization-scale and --random-residual-initialization "
            "are mutually exclusive"
        )
    if args.output_initialization_scale and args.output_warmup_steps:
        raise ValueError(
            "scaled output initialization and output-projection warm-up are "
            "mutually exclusive"
        )
    if 2 * args.modes > args.grid:
        raise ValueError("require 2*modes <= grid")
    if args.z_grid < 0 or args.z_modes < 0:
        raise ValueError("z_grid and z_modes must be non-negative")
    spatial_scheme = args.spatial_scheme
    if spatial_scheme == "auto":
        spatial_scheme = "2.5d" if args.z_grid else "2d"
    resolved_z_modes = args.z_modes or args.modes
    if args.spectral_groups < 1:
        raise ValueError("spectral_groups must be positive")
    if args.spectral_symmetry == "none" and args.spectral_groups != 1:
        raise ValueError("--spectral-groups applies only with EqGINO symmetry")
    if spatial_scheme == "2d":
        if args.z_grid:
            raise ValueError("--z-grid is incompatible with --spatial-scheme 2d")
        if args.z_extent is not None:
            raise ValueError("--z-extent requires the 2.5D scheme")
        if args.z_modes:
            raise ValueError("--z-modes applies only to the 3D scheme")
        if args.z_mixing != "local":
            raise ValueError("--z-mixing global requires the 2.5D scheme")
        if args.lateral_interlacing != 1:
            raise ValueError("--lateral-interlacing applies only to the 2.5D scheme")
        if args.volume_interlacing != 1:
            raise ValueError("--volume-interlacing applies only to the 3D scheme")
        if args.planar_symmetry != "none":
            raise ValueError("--planar-symmetry applies only to the 2.5D scheme")
        if args.spectral_symmetry != "none":
            raise ValueError("--spectral-symmetry applies only to the 3D scheme")
    elif spatial_scheme == "2.5d":
        if args.z_grid < 4:
            raise ValueError("the 2.5D scheme requires z_grid >= 4")
        if args.z_extent is None or args.z_extent <= 0:
            raise ValueError("the 2.5D scheme requires a positive --z-extent")
        if args.z_modes:
            raise ValueError("--z-modes applies only to the 3D scheme")
        if args.spectral_symmetry != "none":
            raise ValueError("--spectral-symmetry applies only to the 3D scheme")
        if args.volume_interlacing != 1:
            raise ValueError("--volume-interlacing applies only to the 3D scheme")
        if args.z_mixing == "local" and (
            args.z_kernel_size < 1 or args.z_kernel_size % 2 == 0
        ):
            raise ValueError("z_kernel_size must be a positive odd integer")
    else:
        if args.z_extent is not None:
            raise ValueError("--z-extent is incompatible with the periodic 3D scheme")
        if args.z_grid < 4:
            raise ValueError("the 3D scheme requires z_grid >= 4")
        if 2 * resolved_z_modes > args.z_grid:
            raise ValueError("the 3D scheme requires 2*z_modes <= z_grid")
        if args.z_mixing != "local":
            raise ValueError("--z-mixing applies only to the 2.5D scheme")
        if args.lateral_interlacing != 1:
            raise ValueError("--lateral-interlacing applies only to the 2.5D scheme")
        if args.planar_symmetry != "none":
            raise ValueError("--planar-symmetry applies only to the 2.5D scheme")
        if args.spectral_symmetry == "eqgino":
            if args.z_grid != args.grid:
                raise ValueError("EqGINO symmetry requires z_grid == grid")
            if resolved_z_modes != args.modes:
                raise ValueError("EqGINO symmetry requires z_modes == modes")
            grouped_channels = (
                args.channels
                if args.architecture == "linear"
                else args.fno_hidden_channels
            )
            if grouped_channels % args.spectral_groups:
                raise ValueError(
                    "EqGINO spectral channels must be divisible by spectral_groups"
                )
    if min(args.eval_interval, args.accumulation_steps, args.batch_size) < 1:
        raise ValueError(
            "eval_interval, accumulation_steps, and batch_size must be positive"
        )
    if not 0.0 < args.lr_decay_factor < 1.0:
        raise ValueError("lr_decay_factor must be between zero and one")
    if args.lr_patience_evals < 0 or args.early_stopping_patience_evals < 0:
        raise ValueError(
            "learning-rate and early-stopping patience must be non-negative"
        )
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError(
            "minimum_learning_rate must be positive and no larger than learning_rate"
        )
    if args.early_stopping_patience_evals and args.lr_scheduler != "plateau":
        raise ValueError("early stopping requires --lr-scheduler plateau")
    evaluation_batch_size = args.evaluation_batch_size or args.batch_size
    if evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be non-negative")
    if args.validation_file is not None and args.validation_indices_file is not None:
        raise ValueError(
            "--validation-file and --validation-indices-file are mutually exclusive"
        )

    # Importing here keeps the base package usable without the optional MACE extra.
    from mace.calculators import MACECalculator

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    setup_start = perf_counter()
    calculator_kwargs = dict(
        model_paths=str(args.mace_model),
        device=str(device),
        # Load the checkpoint in float64 first so the exact atomic reference
        # energies remain available to the hybrid float32 correction. Graphs
        # are cast to the requested model dtype when each batch is collated.
        default_dtype="float64",
    )
    if args.head is not None:
        calculator_kwargs["head"] = args.head
    calculator = MACECalculator(**calculator_kwargs)
    if len(calculator.models) != 1:
        raise ValueError("the first residual experiment requires one MACE model")
    (
        samples,
        reference_cell,
        train_cache_metadata,
        train_cache_hit,
    ) = load_or_create_samples(
        calculator,
        args.train_file,
        args.energy_key,
        args.forces_key,
        dtype,
        args.num_atoms,
        args.allow_periodic_z,
        args.skip_cell_mismatch,
        args.mace_model,
        args.train_cache,
        args.rebuild_cache,
        spatial_scheme=spatial_scheme,
    )
    if args.validation_file is None:
        validation_indices = None
        if args.validation_indices_file is not None:
            validation_indices = [
                int(value) for value in args.validation_indices_file.read_text().split()
            ]
        train_samples, validation_samples = split_samples(
            samples,
            args.validation_fraction,
            args.seed + 2,
            validation_indices=validation_indices,
        )
    else:
        train_samples = samples
        (
            validation_samples,
            _,
            validation_cache_metadata,
            validation_cache_hit,
        ) = load_or_create_samples(
            calculator,
            args.validation_file,
            args.energy_key,
            args.forces_key,
            dtype,
            args.num_atoms,
            args.allow_periodic_z,
            args.skip_cell_mismatch,
            args.mace_model,
            args.validation_cache,
            args.rebuild_cache,
            reference_cell=reference_cell,
            spatial_scheme=spatial_scheme,
        )
    if args.validation_file is None:
        validation_cache_metadata = None
        validation_cache_hit = False
    test_samples: list[dict[str, Any]] = []
    if args.test_file is not None:
        (
            test_samples,
            _,
            test_cache_metadata,
            test_cache_hit,
        ) = load_or_create_samples(
            calculator,
            args.test_file,
            args.energy_key,
            args.forces_key,
            dtype,
            args.num_atoms,
            args.allow_periodic_z,
            args.skip_cell_mismatch,
            args.mace_model,
            args.test_cache,
            args.rebuild_cache,
            reference_cell=reference_cell,
            spatial_scheme=spatial_scheme,
        )
    else:
        test_cache_metadata = None
        test_cache_hit = False

    model = MACEFNOResidual(
        calculator.models[0],
        (args.grid, args.grid),
        args.channels,
        (args.modes, args.modes),
        source_hidden_channels=args.source_hidden_channels,
        fno_hidden_channels=args.fno_hidden_channels,
        fno_layers=args.fno_layers,
        fno_architecture=args.architecture,
        spatial_scheme=spatial_scheme,
        z_grid_size=args.z_grid or None,
        fno_z_modes=resolved_z_modes if spatial_scheme == "3d" else None,
        z_extent=args.z_extent,
        z_center=args.z_center,
        fno_lateral_interlacing=args.lateral_interlacing,
        fno_volume_interlacing=args.volume_interlacing,
        fno_z_kernel_size=args.z_kernel_size,
        fno_z_mixing=args.z_mixing,
        fno_planar_symmetry=args.planar_symmetry,
        fno_spectral_symmetry=args.spectral_symmetry,
        fno_spectral_groups=args.spectral_groups,
        reference_cell=reference_cell,
    ).to(device=device, dtype=dtype)
    if args.output_initialization_scale:
        initialize_scaled_residual_output(model, args.output_initialization_scale)
    elif not args.random_residual_initialization:
        initialize_zero_residual(model)

    setup_seconds = elapsed_since(setup_start, device)
    target_cache_start = perf_counter()

    train_cache_changed = ensure_frozen_residual_targets(
        model,
        samples,
        device=device,
        batch_size=evaluation_batch_size,
    )
    if train_cache_changed or not train_cache_hit:
        save_sample_cache(
            args.train_cache,
            train_cache_metadata,
            samples,
            reference_cell,
        )
    if args.validation_file is not None:
        validation_cache_changed = ensure_frozen_residual_targets(
            model,
            validation_samples,
            device=device,
            batch_size=evaluation_batch_size,
        )
        if validation_cache_changed or not validation_cache_hit:
            save_sample_cache(
                args.validation_cache,
                validation_cache_metadata,
                validation_samples,
                reference_cell,
            )
    if args.test_file is not None:
        test_cache_changed = ensure_frozen_residual_targets(
            model,
            test_samples,
            device=device,
            batch_size=evaluation_batch_size,
        )
        if test_cache_changed or not test_cache_hit:
            save_sample_cache(
                args.test_cache,
                test_cache_metadata,
                test_samples,
                reference_cell,
            )
    target_cache_seconds = elapsed_since(target_cache_start, device)

    model.train()
    model_dtype = next(model.parameters()).dtype
    parameters, warmup_parameters = configure_output_projection_warmup(
        model,
        args.output_warmup_steps,
    )
    warmup_learning_rate = args.output_warmup_learning_rate or args.learning_rate
    optimizer = torch.optim.Adam(
        parameters,
        lr=(warmup_learning_rate if args.output_warmup_steps else args.learning_rate),
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_patience_evals,
            min_lr=args.minimum_learning_rate,
        )
        if args.lr_scheduler == "plateau"
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + 1)

    if args.output_warmup_steps:
        print(
            "output-projection warm-up: "
            f"steps={args.output_warmup_steps}, "
            f"learning_rate={warmup_learning_rate:.6e}, "
            f"active_parameters={sum(p.numel() for p in warmup_parameters)}/"
            f"{sum(p.numel() for p in parameters)}",
            flush=True,
        )

    print(
        f"selected structures: {len(samples)} "
        f"({len(train_samples)} train, {len(validation_samples)} validation, "
        f"{len(test_samples)} held-out test)",
        flush=True,
    )
    initial_evaluation_start = perf_counter()
    baseline_train = (
        evaluate(
            model,
            train_samples,
            baseline=True,
            batch_size=evaluation_batch_size,
        )
        if args.evaluation_scope == "all"
        else {}
    )
    baseline_validation = evaluate(
        model, validation_samples, baseline=True, batch_size=evaluation_batch_size
    )
    baseline_test = evaluate(
        model, test_samples, baseline=True, batch_size=evaluation_batch_size
    )
    print_metrics("frozen MACE train", baseline_train)
    print_metrics("frozen MACE validation", baseline_validation)
    print_metrics("frozen MACE held-out test", baseline_test)
    if baseline_train:
        energy_shift = -baseline_train["energy_bias"]
        print_metrics(
            "constant-offset validation",
            evaluate(
                model,
                validation_samples,
                baseline=True,
                energy_shift_per_atom=energy_shift,
                batch_size=evaluation_batch_size,
            ),
        )
        formula_shifts = {
            formula: -metrics["energy_bias"]
            for formula, metrics in baseline_train["by_formula"].items()
        }
        print_metrics(
            "formula-offset validation",
            evaluate(
                model,
                validation_samples,
                baseline=True,
                energy_shift_per_atom=formula_shifts,
                batch_size=evaluation_batch_size,
            ),
        )
        print_metrics(
            "constant-offset held-out test",
            evaluate(
                model,
                test_samples,
                baseline=True,
                energy_shift_per_atom=energy_shift,
                batch_size=evaluation_batch_size,
            ),
        )
        print_metrics(
            "formula-offset held-out test",
            evaluate(
                model,
                test_samples,
                baseline=True,
                energy_shift_per_atom=formula_shifts,
                batch_size=evaluation_batch_size,
            ),
        )
    initial_evaluation_seconds = elapsed_since(initial_evaluation_start, device)

    best_step = 0
    best_validation_objective = validation_objective(
        baseline_validation,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        energy_scale=args.energy_scale,
        force_scale=args.force_scale,
    )
    best_residual_state = residual_state_dict(model)
    print(
        f"validation objective at frozen baseline: " f"{best_validation_objective:.6e}",
        flush=True,
    )

    completed_steps = 0
    stopped_early = False
    evaluations_at_minimum_lr = 0
    optimization_start = perf_counter()
    for step in range(args.steps):
        if step == args.output_warmup_steps and args.output_warmup_steps:
            finish_output_projection_warmup(parameters)
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
            print(
                f"output-projection warm-up complete at step {step}: "
                f"unfroze {sum(p.numel() for p in parameters)} parameters, "
                f"learning_rate={previous_learning_rate:.6e} -> "
                f"{args.learning_rate:.6e}",
                flush=True,
            )
        optimizer.zero_grad(set_to_none=True)
        accumulated = {"loss": 0.0, "energy": 0.0, "forces": 0.0}
        for _ in range(args.accumulation_steps):
            sample_indices = torch.randint(
                len(train_samples),
                (args.batch_size,),
                generator=generator,
            ).tolist()
            sample_batch = [train_samples[index] for index in sample_indices]
            graph, target_energy, target_forces = collate_samples(
                sample_batch, device, model_dtype
            )
            del target_energy, target_forces
            target_residual_energy = torch.cat(
                [sample["residual_energy"] for sample in sample_batch]
            ).to(device=device)
            target_residual_forces = torch.cat(
                [sample["residual_forces"] for sample in sample_batch]
            ).to(device=device)
            output = model(
                graph,
                training=True,
                compute_force=False,
                compute_residual_force=True,
            )
            target_residual_energy = target_residual_energy.to(
                dtype=output["residual_energy"].dtype
            )
            target_residual_forces = target_residual_forces.to(
                dtype=output["residual_forces"].dtype
            )
            terms = energy_force_loss(
                output["residual_energy"],
                output["residual_forces"],
                target_residual_energy,
                target_residual_forces,
                graph["batch"],
                energy_weight=args.energy_weight,
                force_weight=args.force_weight,
                energy_scale=args.energy_scale,
                force_scale=args.force_scale,
            )
            (terms["loss"] / args.accumulation_steps).backward()
            for name in accumulated:
                accumulated[name] += terms[name].item() / args.accumulation_steps
        optimizer.step()
        completed_steps = step + 1

        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"step {step + 1:5d}/{args.steps}: "
                f"loss={accumulated['loss']:.6e}, "
                f"energy={accumulated['energy']:.6e}, "
                f"forces={accumulated['forces']:.6e}",
                flush=True,
            )
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.steps:
            validation_metrics = evaluate(
                model, validation_samples, batch_size=evaluation_batch_size
            )
            print_metrics(f"validation step {step + 1}", validation_metrics)
            score = validation_objective(
                validation_metrics,
                energy_weight=args.energy_weight,
                force_weight=args.force_weight,
                energy_scale=args.energy_scale,
                force_scale=args.force_scale,
            )
            print(
                f"validation objective step {step + 1}: {score:.6e}",
                flush=True,
            )
            improved = score < best_validation_objective
            if improved:
                best_step = step + 1
                best_validation_objective = score
                best_residual_state = residual_state_dict(model)
                print(f"new best validation step: {best_step}", flush=True)
            if scheduler is not None and step + 1 > args.output_warmup_steps:
                previous_learning_rate = optimizer.param_groups[0]["lr"]
                scheduler.step(score)
                current_learning_rate = optimizer.param_groups[0]["lr"]
                if current_learning_rate != previous_learning_rate:
                    print(
                        f"learning rate step {step + 1}: "
                        f"{previous_learning_rate:.6e} -> "
                        f"{current_learning_rate:.6e}",
                        flush=True,
                    )
                at_minimum_learning_rate = current_learning_rate <= (
                    args.minimum_learning_rate * (1.0 + 16.0 * np.finfo(np.float64).eps)
                )
                if at_minimum_learning_rate:
                    evaluations_at_minimum_lr = (
                        0 if improved else evaluations_at_minimum_lr + 1
                    )
                else:
                    evaluations_at_minimum_lr = 0
                if (
                    args.early_stopping_patience_evals
                    and evaluations_at_minimum_lr >= args.early_stopping_patience_evals
                ):
                    stopped_early = True
                    print(
                        f"early stopping at step {step + 1}: no validation "
                        f"improvement in {evaluations_at_minimum_lr} checks at "
                        f"minimum learning rate {current_learning_rate:.6e}",
                        flush=True,
                    )
                    break

    optimization_seconds = elapsed_since(optimization_start, device)

    load_residual_state_dict(model, best_residual_state)
    print(
        f"restored best validation step {best_step} "
        f"(objective={best_validation_objective:.6e})",
        flush=True,
    )
    final_evaluation_start = perf_counter()
    if args.evaluation_scope == "all":
        print_metrics(
            "selected train",
            evaluate(model, train_samples, batch_size=evaluation_batch_size),
        )
    print_metrics(
        "selected validation",
        evaluate(model, validation_samples, batch_size=evaluation_batch_size),
    )
    print_metrics(
        "selected held-out test",
        evaluate(model, test_samples, batch_size=evaluation_batch_size),
    )
    final_evaluation_seconds = elapsed_since(final_evaluation_start, device)

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "residual_state_dict": residual_state_dict(model),
                "mace_model": str(args.mace_model),
                "mace_head": args.head,
                "train_file": str(args.train_file),
                "validation_file": (
                    str(args.validation_file)
                    if args.validation_file is not None
                    else None
                ),
                "validation_indices_file": (
                    str(args.validation_indices_file)
                    if args.validation_indices_file is not None
                    else None
                ),
                "test_file": (
                    str(args.test_file) if args.test_file is not None else None
                ),
                "grid_shape": (args.grid, args.grid),
                "n_modes": (
                    (resolved_z_modes, args.modes, args.modes)
                    if spatial_scheme == "3d"
                    else (args.modes, args.modes)
                ),
                "spatial_scheme": spatial_scheme,
                "z_grid_size": args.z_grid or None,
                "z_extent": args.z_extent if spatial_scheme == "2.5d" else None,
                "z_center": args.z_center if spatial_scheme == "2.5d" else None,
                "lateral_interlacing": (
                    args.lateral_interlacing if spatial_scheme == "2.5d" else 1
                ),
                "volume_interlacing": (
                    args.volume_interlacing if spatial_scheme == "3d" else 1
                ),
                "planar_symmetry": (
                    args.planar_symmetry if spatial_scheme == "2.5d" else "none"
                ),
                "spectral_symmetry": (
                    args.spectral_symmetry if spatial_scheme == "3d" else "none"
                ),
                "spectral_groups": (
                    args.spectral_groups if spatial_scheme == "3d" else 1
                ),
                "z_kernel_size": (
                    args.z_kernel_size if spatial_scheme == "2.5d" else None
                ),
                "z_mixing": (
                    "spectral"
                    if spatial_scheme == "2.5d" and args.architecture == "linear"
                    else args.z_mixing if spatial_scheme == "2.5d" else None
                ),
                "channels": args.channels,
                "source_hidden_channels": args.source_hidden_channels,
                "fno_hidden_channels": args.fno_hidden_channels,
                "fno_layers": args.fno_layers,
                "architecture": args.architecture,
                "reference_cell": reference_cell.detach().cpu(),
                "num_atoms": args.num_atoms,
                "validation_fraction": args.validation_fraction,
                "skip_cell_mismatch": args.skip_cell_mismatch,
                "accumulation_steps": args.accumulation_steps,
                "batch_size": args.batch_size,
                "evaluation_batch_size": evaluation_batch_size,
                "evaluation_scope": args.evaluation_scope,
                "steps": args.steps,
                "completed_steps": completed_steps,
                "stopped_early": stopped_early,
                "eval_interval": args.eval_interval,
                "learning_rate": args.learning_rate,
                "output_initialization_scale": args.output_initialization_scale,
                "output_warmup_steps": args.output_warmup_steps,
                "output_warmup_learning_rate": warmup_learning_rate,
                "final_learning_rate": optimizer.param_groups[0]["lr"],
                "lr_scheduler": args.lr_scheduler,
                "lr_decay_factor": args.lr_decay_factor,
                "lr_patience_evals": args.lr_patience_evals,
                "minimum_learning_rate": args.minimum_learning_rate,
                "early_stopping_patience_evals": (args.early_stopping_patience_evals),
                "energy_weight": args.energy_weight,
                "force_weight": args.force_weight,
                "energy_scale": args.energy_scale,
                "force_scale": args.force_scale,
                "dtype": args.dtype,
                "train_cache": (
                    str(args.train_cache) if args.train_cache is not None else None
                ),
                "validation_cache": (
                    str(args.validation_cache)
                    if args.validation_cache is not None
                    else None
                ),
                "test_cache": (
                    str(args.test_cache) if args.test_cache is not None else None
                ),
                "best_step": best_step,
                "best_validation_objective": best_validation_objective,
                "seed": args.seed,
            },
            args.checkpoint,
        )
        print(f"checkpoint: {args.checkpoint}")

    total_seconds = elapsed_since(total_start, device)
    print(
        "timings: "
        f"setup={setup_seconds:.2f}s, "
        f"frozen-target-cache={target_cache_seconds:.2f}s, "
        f"initial-evaluation={initial_evaluation_seconds:.2f}s, "
        f"optimization+validation={optimization_seconds:.2f}s, "
        f"final-evaluation={final_evaluation_seconds:.2f}s, "
        f"total={total_seconds:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
