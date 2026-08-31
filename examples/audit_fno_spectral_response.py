"""Audit the low-wavevector response of a trained MACE-FNO checkpoint.

The geometry-aware diagnostic dispatches to the planar 2D, open-boundary
2.5D slab, or fully periodic 3D probe.  An optional amplitude sweep checks
that the reported quadratic response is insensitive to probe magnitude.
These quantities are diagnostics only and do not enter the training loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mace_fno.training import (
    amplitude_convergence_diagnostic,
    choose_device,
    load_mace_fno_model,
    low_k_response_diagnostic,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-mode", type=int, default=2)
    parser.add_argument(
        "--fit-shells",
        type=int,
        default=3,
        help="number of smallest physical reciprocal shells used for the fit",
    )
    parser.add_argument("--relative-amplitude", type=float, default=0.05)
    parser.add_argument(
        "--relative-amplitudes",
        type=float,
        nargs="+",
        help=(
            "run a post-training amplitude-convergence audit instead of one "
            "probe; for example: 0.025 0.05 0.1"
        ),
    )
    parser.add_argument("--relative-span-tolerance", type=float, default=0.05)
    parser.add_argument("--field-batch-size", type=int, default=32)
    parser.add_argument(
        "--z-profiles",
        type=int,
        default=3,
        help="number of z profiles for the 2.5D slab diagnostic",
    )
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if min(args.max_mode, args.fit_shells, args.field_batch_size) < 1:
        raise ValueError("max_mode, fit_shells, and field_batch_size must be positive")
    if args.z_profiles < 1:
        raise ValueError("z_profiles must be positive")
    if args.relative_span_tolerance < 0.0:
        raise ValueError("relative_span_tolerance must be non-negative")

    device = choose_device(args.device)
    model, checkpoint = load_mace_fno_model(args.checkpoint, device=device)
    cache_path = args.sample_cache or Path(checkpoint["test_cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    all_samples = cache["samples"]
    generator = torch.Generator().manual_seed(args.seed)
    sample_indices = torch.randperm(len(all_samples), generator=generator)[
        : min(args.samples, len(all_samples))
    ].tolist()

    common = {
        "sample_indices": sample_indices,
        "max_mode": args.max_mode,
        "fit_shells": args.fit_shells,
        "field_batch_size": args.field_batch_size,
        "z_profiles": args.z_profiles,
    }
    if args.relative_amplitudes is None:
        report = low_k_response_diagnostic(
            model,
            all_samples,
            relative_amplitude=args.relative_amplitude,
            **common,
        )
    else:
        report = amplitude_convergence_diagnostic(
            model,
            all_samples,
            relative_amplitudes=args.relative_amplitudes,
            relative_span_tolerance=args.relative_span_tolerance,
            **common,
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "sample_cache": str(cache_path),
        "sample_indices": sample_indices,
        **report,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
