#!/usr/bin/env python3
"""Summarize a liquid-water EqGINO optimization pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "EVALUATION_JSON"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path | str) -> dict[str, Any]:
    filename = Path(path)
    if not filename.is_file():
        raise FileNotFoundError(filename)
    return json.loads(filename.read_text())


def reduction(reference: float, corrected: float) -> float:
    return 100.0 * (reference - corrected) / reference


def main() -> None:
    args = parse_arguments()
    baseline_report = load_json(args.baseline)
    baseline_values = baseline_report["test"]["overall"]
    baseline = {
        "energy_rmse": float(baseline_values["energy_eV_per_atom"]["rmse"]),
        "force_rmse": float(baseline_values["forces_eV_per_A"]["rmse"]),
    }
    runs = []
    validation_baselines = []
    seen = set()
    for name, filename in args.run:
        if name in seen:
            raise ValueError(f"duplicate run name: {name}")
        seen.add(name)
        report = load_json(filename)
        validation_split = report["splits"]["validation"]
        test_values = report["splits"]["test"]["mace_fno"]
        validation_values = validation_split["mace_fno"]
        validation_baselines.append(validation_split["frozen_mace"])
        energy_rmse = float(test_values["energy_rmse"])
        force_rmse = float(test_values["force_rmse"])
        runs.append(
            {
                "name": name,
                "evaluation": filename,
                "best_step": report.get("best_step"),
                "validation_energy_rmse": float(
                    validation_values["energy_rmse"]
                ),
                "validation_force_rmse": float(
                    validation_values["force_rmse"]
                ),
                "energy_rmse": energy_rmse,
                "force_rmse": force_rmse,
                "energy_reduction_percent": reduction(
                    baseline["energy_rmse"], energy_rmse
                ),
                "force_reduction_percent": reduction(
                    baseline["force_rmse"], force_rmse
                ),
            }
        )
    validation_reference = validation_baselines[0]
    for other in validation_baselines[1:]:
        for key in ("energy_rmse", "force_rmse"):
            if abs(float(other[key]) - float(validation_reference[key])) > 1.0e-12:
                raise ValueError("frozen validation metrics differ between runs")
    baseline["validation_energy_rmse"] = float(
        validation_reference["energy_rmse"]
    )
    baseline["validation_force_rmse"] = float(
        validation_reference["force_rmse"]
    )
    ranked = sorted(
        runs,
        key=lambda run: (
            run["validation_force_rmse"],
            run["validation_energy_rmse"],
        ),
    )
    report = {
        "baseline": baseline,
        "runs": runs,
        "ranking_criterion": (
            "validation force RMSE, then validation energy RMSE"
        ),
        "recommended_full_run": ranked[0]["name"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# LES liquid-water EqGINO pilot",
        "",
        "The 30-structure validation split selects the optimization setting. "
        "The published 50-structure test set is reported only for continuity "
        "with the LES benchmark; confirm the selected setting across seeds.",
        "",
        "| Configuration | Best step | Validation E | Validation F | Test E | Test F | Test E reduction | Test F reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Frozen MACE | 0 | "
            f"{1000 * baseline['validation_energy_rmse']:.4f} | "
            f"{1000 * baseline['validation_force_rmse']:.3f} | "
            f"{1000 * baseline['energy_rmse']:.4f} | "
            f"{1000 * baseline['force_rmse']:.3f} | -- | -- |"
        ),
    ]
    for run in runs:
        lines.append(
            f"| {run['name']} | {run['best_step']} | "
            f"{1000 * run['validation_energy_rmse']:.4f} | "
            f"{1000 * run['validation_force_rmse']:.3f} | "
            f"{1000 * run['energy_rmse']:.4f} | "
            f"{1000 * run['force_rmse']:.3f} | "
            f"{run['energy_reduction_percent']:+.1f}% | "
            f"{run['force_reduction_percent']:+.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Lowest validation force RMSE: `{ranked[0]['name']}`.",
            "",
        ]
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
