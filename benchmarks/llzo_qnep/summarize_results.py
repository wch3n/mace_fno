#!/usr/bin/env python3
"""Combine LLZO baseline, residual, and physics-audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mace-one", type=Path, required=True)
    parser.add_argument("--mace-two", type=Path, required=True)
    parser.add_argument("--fno", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--published", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_json(filename: Path) -> dict[str, Any]:
    if not filename.is_file():
        raise FileNotFoundError(filename)
    return json.loads(filename.read_text())


def mace_metrics(report: dict[str, Any], split: str = "test") -> dict[str, float]:
    metrics = report[split]["overall"]
    return {
        "energy_rmse": float(metrics["energy_eV_per_atom"]["rmse"]),
        "force_rmse": float(metrics["forces_eV_per_A"]["rmse"]),
    }


def residual_metrics(
    report: dict[str, Any], model: str, split: str = "test"
) -> dict[str, float]:
    metrics = report["splits"][split][model]
    return {
        "energy_rmse": float(metrics["energy_rmse"]),
        "force_rmse": float(metrics["force_rmse"]),
    }


def improvement_percent(reference: float, value: float) -> float | None:
    if reference == 0.0:
        return None
    return 100.0 * (reference - value) / reference


def format_number(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def format_percent(value: float | None) -> str:
    return "--" if value is None else f"{value:+.1f}%"


def baseline_group_metrics(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        group: {
            "energy_rmse": float(values["energy_eV_per_atom"]["rmse"]),
            "force_rmse": float(values["forces_eV_per_A"]["rmse"]),
        }
        for group, values in report["test"]["by_benchmark_group"].items()
    }


def residual_group_metrics(
    report: dict[str, Any], model: str
) -> dict[str, dict[str, float]]:
    return {
        group: {
            "energy_rmse": float(values["energy_rmse"]),
            "force_rmse": float(values["force_rmse"]),
        }
        for group, values in report["splits"]["test"][model][
            "by_benchmark_group"
        ].items()
    }


def published_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if report is None:
        return []
    rows = []
    for name, model in report["models"].items():
        metrics = mace_metrics(model)
        rows.append({"model": name, **metrics, "scope": "published-full-data"})
    return rows


def main() -> None:
    args = parse_arguments()
    manifest = load_json(args.manifest)
    mace_one_report = load_json(args.mace_one)
    mace_two_report = load_json(args.mace_two)
    fno_report = load_json(args.fno)
    audit_report = load_json(args.audit)
    spectral_report = load_json(args.spectral)
    published_report = (
        load_json(args.published)
        if args.published is not None and args.published.is_file()
        else None
    )

    mace_one = mace_metrics(mace_one_report)
    mace_two = mace_metrics(mace_two_report)
    frozen_in_fno = residual_metrics(fno_report, "frozen_mace")
    mace_fno = residual_metrics(fno_report, "mace_fno")
    rows = [
        {"model": "MACE, one interaction", **mace_one, "scope": "held-out"},
        {"model": "MACE, two interactions", **mace_two, "scope": "held-out"},
        {"model": "MACE, one interaction + FNO", **mace_fno, "scope": "held-out"},
        *published_rows(published_report),
    ]
    improvements = {
        "energy_rmse_percent": improvement_percent(
            mace_one["energy_rmse"], mace_fno["energy_rmse"]
        ),
        "force_rmse_percent": improvement_percent(
            mace_one["force_rmse"], mace_fno["force_rmse"]
        ),
    }
    frozen_cross_check = {
        "energy_rmse_absolute_difference": abs(
            mace_one["energy_rmse"] - frozen_in_fno["energy_rmse"]
        ),
        "force_rmse_absolute_difference": abs(
            mace_one["force_rmse"] - frozen_in_fno["force_rmse"]
        ),
    }
    exact_checks = audit_report["promised_exact_checks"]
    exact_audit_passed = all(bool(check["passed"]) for check in exact_checks.values())

    one_groups = baseline_group_metrics(mace_one_report)
    two_groups = baseline_group_metrics(mace_two_report)
    fno_groups = residual_group_metrics(fno_report, "mace_fno")
    group_comparison = {
        group: {
            "mace_one": one_groups[group],
            "mace_two": two_groups[group],
            "mace_fno": fno_groups[group],
            "energy_rmse_improvement_percent": improvement_percent(
                one_groups[group]["energy_rmse"], fno_groups[group]["energy_rmse"]
            ),
            "force_rmse_improvement_percent": improvement_percent(
                one_groups[group]["force_rmse"], fno_groups[group]["force_rmse"]
            ),
        }
        for group in sorted(set(one_groups) & set(two_groups) & set(fno_groups))
    }

    power_fit = spectral_report.get("low_k_dominant_eigenvalue_fit")
    tensor_fit = spectral_report.get("pooled_anisotropic_inverse_quadratic_fit")
    report = {
        "split": manifest["split"],
        "models": rows,
        "mace_fno_improvement_over_one_interaction_mace": improvements,
        "frozen_mace_cross_check": frozen_cross_check,
        "group_comparison": group_comparison,
        "constant_offset_control": {
            "energy_rmse": float(
                mace_one_report["test_global_offset"]["overall"][
                    "energy_eV_per_atom"
                ]["rmse"]
            )
        },
        "audit": {
            "all_promised_exact_checks_passed": exact_audit_passed,
            "checks": exact_checks,
        },
        "spectral": {
            "diagnostic_kind": spectral_report.get("diagnostic_kind"),
            "low_k_power_fit": power_fit,
            "anisotropic_inverse_quadratic_fit": tensor_fit,
        },
        "assessment": {
            "held_out_energy_improved": mace_fno["energy_rmse"] < mace_one["energy_rmse"],
            "held_out_force_improved": mace_fno["force_rmse"] < mace_one["force_rmse"],
            "beats_train_fitted_constant_energy_offset": (
                mace_fno["energy_rmse"]
                < float(
                    mace_one_report["test_global_offset"]["overall"][
                        "energy_eV_per_atom"
                    ]["rmse"]
                )
            ),
            "exact_audit_passed": exact_audit_passed,
        },
    }

    lines = [
        "# LLZO MACE+FNO benchmark summary",
        "",
        (
            f"Deterministic split (seed {manifest['split']['seed']}): "
            f"{manifest['split']['counts']['train']}/"
            f"{manifest['split']['counts']['validation']}/"
            f"{manifest['split']['counts']['test']} train/validation/test structures."
        ),
        "",
        "| Model | Scope | E RMSE (meV/atom) | F RMSE (meV/A) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['scope']} | "
            f"{1000.0 * row['energy_rmse']:.3f} | "
            f"{1000.0 * row['force_rmse']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Relative to one-interaction MACE, the FNO changes the held-out "
            f"energy RMSE by {format_percent(improvements['energy_rmse_percent'])} "
            "and the force RMSE by "
            f"{format_percent(improvements['force_rmse_percent'])}.",
            "",
            "## Cell-class breakdown",
            "",
            "| Cell class | nl0 E | nl1 E | FNO E | E gain | nl0 F | nl1 F | FNO F | F gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, values in group_comparison.items():
        lines.append(
            f"| {group} | {1000 * values['mace_one']['energy_rmse']:.3f} | "
            f"{1000 * values['mace_two']['energy_rmse']:.3f} | "
            f"{1000 * values['mace_fno']['energy_rmse']:.3f} | "
            f"{format_percent(values['energy_rmse_improvement_percent'])} | "
            f"{1000 * values['mace_one']['force_rmse']:.3f} | "
            f"{1000 * values['mace_two']['force_rmse']:.3f} | "
            f"{1000 * values['mace_fno']['force_rmse']:.3f} | "
            f"{format_percent(values['force_rmse_improvement_percent'])} |"
        )
    lines.extend(
        [
            "",
            "## Physics audit",
            "",
            "| Check | Observed | Threshold | Passed |",
            "|---|---:|---:|:---:|",
        ]
    )
    for name, check in exact_checks.items():
        lines.append(
            f"| {name.replace('_', ' ')} | {check['observed']:.6g} | "
            f"{check['threshold']:.6g} | {'yes' if check['passed'] else 'no'} |"
        )
    if power_fit is not None:
        lines.extend(
            [
                "",
                "## Low-wavevector diagnostic",
                "",
                f"- Free fitted power: p = {power_fit['free_power_exponent_p']:.3f}",
                f"- Free-power log-space R2: {power_fit['free_log_r2']:.3f}",
                f"- Fixed 1/k2 log-space R2: {power_fit['coulomb_p2_log_r2']:.3f}",
            ]
        )
    if tensor_fit is not None:
        lines.append(
            "- Anisotropic inverse-quadratic fit R2: "
            + format_number(tensor_fit.get("log_response_r2"))
        )
        lines.append(
            "- Fitted tensor positive definite: "
            + ("yes" if tensor_fit["positive_definite"] else "no")
        )
    if published_report is not None:
        lines.extend(
            [
                "",
                "The published NEP/QNEP rows are contextual only: those parameter "
                "files were fitted to the complete source dataset, which includes "
                "our deterministic test subset.",
            ]
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_json}", flush=True)
    print(f"wrote {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
