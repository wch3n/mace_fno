#!/usr/bin/env python3
"""Aggregate matched LLZO MACE+FNO evaluations across training seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mace-one", type=Path, required=True)
    parser.add_argument("--mace-two", type=Path, required=True)
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        required=True,
        metavar=("SEED", "FNO_JSON", "AUDIT_JSON", "SPECTRAL_JSON"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_json(filename: Path | str) -> dict[str, Any]:
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def mace_metrics(report: dict[str, Any]) -> dict[str, float]:
    values = report["test"]["overall"]
    return {
        "energy_rmse": float(values["energy_eV_per_atom"]["rmse"]),
        "force_rmse": float(values["forces_eV_per_A"]["rmse"]),
    }


def reduction(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def maximum_cubic_metric(audit: dict[str, Any], key: str) -> float | None:
    transformations = audit.get("cubic_signed_axis_transformations")
    if not transformations:
        return None
    return max(float(values[key]) for values in transformations.values())


def parse_run(
    fields: list[str], baseline: dict[str, float]
) -> dict[str, Any]:
    seed_text, fno_path, audit_path, spectral_path = fields
    seed = int(seed_text)
    evaluation = load_json(fno_path)
    audit = load_json(audit_path)
    spectral = load_json(spectral_path)
    test = evaluation["splits"]["test"]["mace_fno"]
    energy_rmse = float(test["energy_rmse"])
    force_rmse = float(test["force_rmse"])
    exact_checks = audit["promised_exact_checks"]
    power = spectral.get("low_k_dominant_eigenvalue_fit") or {}
    return {
        "seed": seed,
        "evaluation": fno_path,
        "audit": audit_path,
        "spectral": spectral_path,
        "best_step": evaluation.get("best_step"),
        "energy_rmse": energy_rmse,
        "force_rmse": force_rmse,
        "energy_reduction_percent": reduction(
            baseline["energy_rmse"], energy_rmse
        ),
        "force_reduction_percent": reduction(
            baseline["force_rmse"], force_rmse
        ),
        "by_benchmark_group": test.get("by_benchmark_group", {}),
        "all_promised_exact_checks_passed": all(
            bool(check["passed"]) for check in exact_checks.values()
        ),
        "diagnostic_status": audit.get("diagnostic_status", {}),
        "maximum_translation_energy_change_mev": max(
            float(value)
            for value in audit["rigid_translation_energy_max_mev_by_cell_axis"]
        ),
        "maximum_cubic_energy_change_mev": maximum_cubic_metric(
            audit, "residual_energy_change_mev"
        ),
        "maximum_cubic_force_rmse_mev_per_angstrom": maximum_cubic_metric(
            audit, "residual_force_equivariance_rmse_mev_per_angstrom"
        ),
        "low_k_power_exponent": power.get("free_power_exponent_p"),
        "low_k_free_log_r2": power.get("free_log_r2"),
        "low_k_coulomb_log_r2": power.get("coulomb_p2_log_r2"),
    }


def aggregate_groups(runs: list[dict[str, Any]]) -> dict[str, Any]:
    group_names = set.intersection(
        *(set(run["by_benchmark_group"]) for run in runs)
    )
    return {
        group: {
            "energy_rmse": distribution(
                [
                    float(run["by_benchmark_group"][group]["energy_rmse"])
                    for run in runs
                ]
            ),
            "force_rmse": distribution(
                [
                    float(run["by_benchmark_group"][group]["force_rmse"])
                    for run in runs
                ]
            ),
        }
        for group in sorted(group_names)
    }


def pm(value: dict[str, float | int], scale: float = 1.0) -> str:
    return (
        f"{scale * float(value['mean']):.3f} +/- "
        f"{scale * float(value['sample_std']):.3f}"
    )


def main() -> None:
    args = parse_arguments()
    manifest = load_json(args.manifest)
    mace_one = mace_metrics(load_json(args.mace_one))
    mace_two = mace_metrics(load_json(args.mace_two))
    runs = [parse_run(fields, mace_one) for fields in args.run]
    seeds = [run["seed"] for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("run seeds must be unique")
    runs.sort(key=lambda run: run["seed"])

    aggregate = {
        "energy_rmse": distribution([run["energy_rmse"] for run in runs]),
        "force_rmse": distribution([run["force_rmse"] for run in runs]),
        "energy_reduction_percent": distribution(
            [run["energy_reduction_percent"] for run in runs]
        ),
        "force_reduction_percent": distribution(
            [run["force_reduction_percent"] for run in runs]
        ),
        "maximum_translation_energy_change_mev": distribution(
            [run["maximum_translation_energy_change_mev"] for run in runs]
        ),
    }
    report = {
        "split": manifest["split"],
        "mace_one": mace_one,
        "mace_two": mace_two,
        "runs": runs,
        "aggregate": aggregate,
        "group_aggregate": aggregate_groups(runs),
        "assessment": {
            "all_seeds_improve_energy": all(
                run["energy_rmse"] < mace_one["energy_rmse"] for run in runs
            ),
            "all_seeds_improve_forces": all(
                run["force_rmse"] < mace_one["force_rmse"] for run in runs
            ),
            "all_promised_exact_checks_passed": all(
                run["all_promised_exact_checks_passed"] for run in runs
            ),
        },
    }

    split = manifest["split"]
    lines = [
        "# LLZO blocked-split, symmetry-controlled multi-seed summary",
        "",
        (
            f"Split: {split['method']} (seed {split['seed']}, "
            f"block size {split.get('block_size')}); "
            f"{split['counts']['train']}/{split['counts']['validation']}/"
            f"{split['counts']['test']} train/validation/test structures."
        ),
        "",
        "| Model or seed | E RMSE (meV/atom) | F RMSE (meV/A) | E reduction | F reduction |",
        "|---|---:|---:|---:|---:|",
        (
            f"| MACE, one interaction | {1000*mace_one['energy_rmse']:.3f} | "
            f"{1000*mace_one['force_rmse']:.3f} | -- | -- |"
        ),
        (
            f"| MACE, two interactions | {1000*mace_two['energy_rmse']:.3f} | "
            f"{1000*mace_two['force_rmse']:.3f} | "
            f"{reduction(mace_one['energy_rmse'], mace_two['energy_rmse']):+.1f}% | "
            f"{reduction(mace_one['force_rmse'], mace_two['force_rmse']):+.1f}% |"
        ),
    ]
    for run in runs:
        lines.append(
            f"| MACE+FNO, seed {run['seed']} | {1000*run['energy_rmse']:.3f} | "
            f"{1000*run['force_rmse']:.3f} | "
            f"{run['energy_reduction_percent']:+.1f}% | "
            f"{run['force_reduction_percent']:+.1f}% |"
        )
    lines.append(
        f"| MACE+FNO, mean +/- s.d. | {pm(aggregate['energy_rmse'], 1000)} | "
        f"{pm(aggregate['force_rmse'], 1000)} | "
        f"{pm(aggregate['energy_reduction_percent'])}% | "
        f"{pm(aggregate['force_reduction_percent'])}% |"
    )
    lines.extend(
        [
            "",
            "## Symmetry and spectral diagnostics",
            "",
            "| Seed | Max arbitrary-translation dE (meV/cell) | Max cubic dE (meV/cell) | Max cubic force RMSE (meV/A) | Exact checks | low-k p | fixed 1/k2 R2 |",
            "|---:|---:|---:|---:|:---:|---:|---:|",
        ]
    )
    for run in runs:
        cubic_energy = run["maximum_cubic_energy_change_mev"]
        cubic_force = run["maximum_cubic_force_rmse_mev_per_angstrom"]
        exponent = run["low_k_power_exponent"]
        coulomb_r2 = run["low_k_coulomb_log_r2"]
        lines.append(
            f"| {run['seed']} | {run['maximum_translation_energy_change_mev']:.6g} | "
            f"{'--' if cubic_energy is None else f'{cubic_energy:.6g}'} | "
            f"{'--' if cubic_force is None else f'{cubic_force:.6g}'} | "
            f"{'yes' if run['all_promised_exact_checks_passed'] else 'no'} | "
            f"{'--' if exponent is None else f'{float(exponent):.3f}'} | "
            f"{'--' if coulomb_r2 is None else f'{float(coulomb_r2):.3f}'} |"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_json}", flush=True)
    print(f"wrote {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
