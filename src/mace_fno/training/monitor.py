"""Spectral response monitoring during and after residual optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from ..coupling import MACEFNOResidual
from .configuration import TrainingConfig
from .spectral_diagnostic import (
    amplitude_convergence_diagnostic,
    low_k_response_diagnostic,
)

Sample = dict[str, Any]


def print_low_k_diagnostic(label: str, report: dict[str, Any]) -> None:
    """Print a compact geometry-aware infrared-response summary."""
    diagnostic_kind = report.get("diagnostic_kind")
    planar = diagnostic_kind == "planar_2d"
    slab = diagnostic_kind == "slab_2p5d"
    surface = planar or slab
    if planar:
        fit = report["low_k_planar_response_fit"]
    elif slab:
        fit = report["low_k_monopole_response_fit"]
    else:
        fit = report["low_k_dominant_eigenvalue_fit"]
    if fit is None:
        summary = "no positive leading response on at least two low-k shells"
    else:
        reference_label = "R2_1/k" if surface else "R2_1/k2"
        reference_r2 = (
            fit["reference_power_log_r2"] if surface else fit["coulomb_p2_log_r2"]
        )
        summary = (
            f"p={fit['free_power_exponent_p']:.3f} | "
            f"R2_free={fit['free_log_r2']:.3f} | "
            f"{reference_label}={reference_r2:.3f} | "
            f"points={fit['points']}"
        )
        if slab and report["mean_low_k_coulomb_template_relative_error"] is not None:
            summary += (
                " | z_template_relerr="
                f"{report['mean_low_k_coulomb_template_relative_error']:.3f}"
            )
        if not surface:
            tensor_fit = report.get("pooled_anisotropic_inverse_quadratic_fit")
            if tensor_fit is not None and tensor_fit["log_response_r2"] is not None:
                summary += f" | tensor_R2={tensor_fit['log_response_r2']:.3f}"
    print(f"{label:<32} | {summary}", flush=True)


def print_amplitude_diagnostic(label: str, report: dict[str, Any]) -> None:
    """Print a compact finite-amplitude convergence result."""
    summary = report["summary"]
    stable = "yes" if summary["curvature_stable_within_tolerance"] else "no"
    median_span = summary["median_mode_relative_span"]
    maximum_span = summary["maximum_mode_relative_span"]
    exponent_range = summary["free_power_exponent_range"]
    fields = [
        f"stable={stable}",
        f"matched_modes={summary['matched_modes']}",
    ]
    if median_span is not None:
        fields.append(f"median_span={median_span:.3e}")
    if maximum_span is not None:
        fields.append(f"max_span={maximum_span:.3e}")
    if exponent_range is not None:
        fields.append(f"p_range={exponent_range:.3e}")
    print(f"{label:<32} | {' | '.join(fields)}", flush=True)


@dataclass
class SpectralMonitor:
    """Own the fixed spectral probes and their persistent training history."""

    configuration: TrainingConfig
    validation_samples: list[Sample]
    sample_indices: list[int]
    validation_z_profiles: int
    final_amplitudes: list[float]
    report_configuration: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        configuration: TrainingConfig,
        validation_samples: list[Sample],
    ) -> SpectralMonitor | None:
        diagnostic = configuration.diagnostic
        if not diagnostic.enabled:
            return None
        if not validation_samples:
            raise ValueError("the spectral diagnostic requires validation samples")

        generator = torch.Generator().manual_seed(configuration.runtime.seed + 947)
        sample_indices = torch.randperm(len(validation_samples), generator=generator)[
            : min(diagnostic.samples, len(validation_samples))
        ].tolist()
        validation_z_profiles = (
            1 if configuration.model.spatial_scheme == "2.5d" else diagnostic.z_profiles
        )
        final_amplitudes = []
        if diagnostic.depth == "deep":
            final_amplitudes = sorted(
                set(diagnostic.amplitudes) | {diagnostic.relative_amplitude}
            )
        report_configuration = {
            "spatial_scheme": configuration.model.spatial_scheme,
            "sample_indices": sample_indices,
            "max_mode": diagnostic.max_mode,
            "fit_shells": diagnostic.fit_shells,
            "relative_amplitude": diagnostic.relative_amplitude,
            "field_batch_size": diagnostic.field_batch_size,
            "z_profiles": diagnostic.z_profiles,
            "validation_z_profiles": validation_z_profiles,
            "selected_z_profiles": diagnostic.z_profiles,
            "depth": diagnostic.depth,
            "selected_amplitudes": final_amplitudes,
            "relative_span_tolerance": diagnostic.relative_span_tolerance,
            "description": (
                "Geometry-aware latent-field curvature diagnostic. Routine "
                "validation uses the inexpensive probe; deeper settings apply "
                "only after restoring the selected checkpoint. It is not a loss."
            ),
        }
        return cls(
            configuration=configuration,
            validation_samples=validation_samples,
            sample_indices=sample_indices,
            validation_z_profiles=validation_z_profiles,
            final_amplitudes=final_amplitudes,
            report_configuration=report_configuration,
        )

    def evaluate_validation(
        self,
        model: MACEFNOResidual,
        *,
        step: int,
        validation_objective: float,
    ) -> None:
        """Evaluate and persist the inexpensive probe at a validation step."""
        diagnostic = self.configuration.diagnostic
        report = low_k_response_diagnostic(
            model,
            self.validation_samples,
            sample_indices=self.sample_indices,
            max_mode=diagnostic.max_mode,
            fit_shells=diagnostic.fit_shells,
            relative_amplitude=diagnostic.relative_amplitude,
            field_batch_size=diagnostic.field_batch_size,
            z_profiles=self.validation_z_profiles,
        )
        report.update(
            {
                "step": step,
                "stage": "validation",
                "validation_objective": validation_objective,
            }
        )
        self.history.append(report)
        print_low_k_diagnostic(f"low-k response step {step}", report)
        self.write_history()

    def evaluate_selected(
        self,
        model: MACEFNOResidual,
        *,
        step: int,
        validation_objective: float,
    ) -> None:
        """Evaluate the restored best checkpoint, including an optional deep audit."""
        diagnostic = self.configuration.diagnostic
        amplitude_report: dict[str, Any] | None = None
        if diagnostic.depth == "deep":
            amplitude_report = amplitude_convergence_diagnostic(
                model,
                self.validation_samples,
                sample_indices=self.sample_indices,
                relative_amplitudes=self.final_amplitudes,
                relative_span_tolerance=diagnostic.relative_span_tolerance,
                max_mode=diagnostic.max_mode,
                fit_shells=diagnostic.fit_shells,
                field_batch_size=diagnostic.field_batch_size,
                z_profiles=diagnostic.z_profiles,
            )
            selected_report = next(
                report
                for report in amplitude_report["runs"]
                if report["relative_amplitude"] == diagnostic.relative_amplitude
            )
        else:
            selected_report = low_k_response_diagnostic(
                model,
                self.validation_samples,
                sample_indices=self.sample_indices,
                max_mode=diagnostic.max_mode,
                fit_shells=diagnostic.fit_shells,
                relative_amplitude=diagnostic.relative_amplitude,
                field_batch_size=diagnostic.field_batch_size,
                z_profiles=diagnostic.z_profiles,
            )
        selected_report.update(
            {
                "step": step,
                "stage": "selected",
                "validation_objective": validation_objective,
            }
        )
        self.history.append(selected_report)
        print_low_k_diagnostic("low-k response selected", selected_report)

        if amplitude_report is not None:
            compact_report = {
                key: value for key, value in amplitude_report.items() if key != "runs"
            }
            compact_report.update(
                {
                    "step": step,
                    "stage": "selected_amplitude_convergence",
                    "validation_objective": validation_objective,
                }
            )
            self.history.append(compact_report)
            print_amplitude_diagnostic(
                "amplitude convergence selected", amplitude_report
            )
        self.write_history()

    def write_history(self) -> None:
        """Persist the diagnostic trace independently of the model checkpoint."""
        output = self.configuration.diagnostic.output
        if output is None:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "configuration": self.report_configuration,
                    "history": self.history,
                },
                indent=2,
            )
            + "\n"
        )
