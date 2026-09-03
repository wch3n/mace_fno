"""Evaluate a frozen or jointly trained MACE-FNO checkpoint from caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..training.checkpoint import load_mace_fno_model
from ..training.data import split_samples
from ..training.evaluation import (
    ensure_frozen_residual_targets,
    evaluate,
    print_metrics,
)
from ..training.runtime import choose_device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def improvement_percent(baseline: float, corrected: float) -> float | None:
    """Return the signed error reduction, or None for a zero baseline."""
    if baseline == 0.0:
        return None
    return 100.0 * (baseline - corrected) / baseline


def metric_improvements(
    baseline: dict[str, Any], corrected: dict[str, Any]
) -> dict[str, float | None]:
    return {
        "energy_rmse_percent": improvement_percent(
            float(baseline["energy_rmse"]), float(corrected["energy_rmse"])
        ),
        "energy_mae_percent": improvement_percent(
            float(baseline["energy_mae"]), float(corrected["energy_mae"])
        ),
        "force_rmse_percent": improvement_percent(
            float(baseline["force_rmse"]), float(corrected["force_rmse"])
        ),
        "force_mae_percent": improvement_percent(
            float(baseline["force_mae"]), float(corrected["force_mae"])
        ),
    }


def format_percent(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}%"


def resolve_cache_path(
    split: str,
    override: Path | None,
    checkpoint: dict[str, Any],
) -> Path:
    value = override or checkpoint.get(f"{split}_cache")
    if value is None:
        raise ValueError(
            f"no {split} cache is recorded; pass --{split}-cache explicitly"
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _samples_from_cache(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"no samples found in {path}")
    return samples


def load_split_samples(
    split: str,
    override: Path | None,
    checkpoint: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], bool]:
    """Load one cache, reconstructing an absent internal validation split."""
    try:
        path = resolve_cache_path(split, override, checkpoint)
    except (FileNotFoundError, ValueError):
        if split != "validation" or override is not None:
            raise
        train_path = resolve_cache_path("train", None, checkpoint)
        all_samples = _samples_from_cache(train_path)
        indices_file = checkpoint.get("validation_indices_file")
        validation_indices = None
        if indices_file is not None:
            indices_path = Path(indices_file).expanduser()
            if not indices_path.is_file():
                raise FileNotFoundError(indices_path)
            validation_indices = [
                int(value) for value in indices_path.read_text().split()
            ]
        _, samples = split_samples(
            all_samples,
            float(checkpoint.get("validation_fraction", 0.2)),
            int(checkpoint.get("seed", 17)) + 2,
            validation_indices=validation_indices,
        )
        if not samples:
            raise ValueError("the reconstructed validation split is empty")
        print(
            "validation cache is absent; reconstructed the recorded split "
            f"from {train_path}",
            flush=True,
        )
        return train_path, samples, True
    return path, _samples_from_cache(path), False


def main() -> None:
    args = parse_arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = choose_device(args.device)
    model, checkpoint = load_mace_fno_model(
        args.checkpoint,
        device=device,
        dtype=args.dtype,
    )
    model.eval()

    overrides = {
        "train": args.train_cache,
        "validation": args.validation_cache,
        "test": args.test_cache,
    }
    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "mace_model": str(checkpoint["mace_model"]),
        "best_step": checkpoint.get("best_step"),
        "mace_training": checkpoint.get("mace_training", "frozen"),
        "cell_mode": checkpoint.get("cell_mode", "fixed"),
        "spatial_scheme": checkpoint.get("spatial_scheme", "2d"),
        "splits": {},
    }
    for split in dict.fromkeys(args.splits):
        cache_path, samples, reconstructed = load_split_samples(
            split, overrides[split], checkpoint
        )
        if checkpoint.get("mace_training", "frozen") == "joint":
            required_baseline = {"base_energy", "base_forces"}
            if any(not required_baseline <= set(sample) for sample in samples):
                raise ValueError(
                    "joint-checkpoint baseline comparison requires caches written "
                    "before fine-tuning; the selected cache lacks initial-MACE "
                    "predictions"
                )
        ensure_frozen_residual_targets(
            model,
            samples,
            device=device,
            batch_size=args.batch_size,
        )
        baseline = evaluate(
            model,
            samples,
            baseline=True,
            batch_size=args.batch_size,
        )
        corrected = evaluate(model, samples, batch_size=args.batch_size)
        baseline_label = (
            "frozen MACE"
            if checkpoint.get("mace_training", "frozen") == "frozen"
            else "initial MACE"
        )
        corrected_label = (
            "MACE+FNO"
            if checkpoint.get("mace_training", "frozen") == "frozen"
            else "joint MACE+FNO"
        )
        print_metrics(f"{baseline_label} {split}", baseline)
        print_metrics(f"{corrected_label} {split}", corrected)
        improvements = metric_improvements(baseline, corrected)
        print(
            f"{split} improvement: "
            f"E_RMSE={format_percent(improvements['energy_rmse_percent'])}, "
            f"F_RMSE={format_percent(improvements['force_rmse_percent'])}",
            flush=True,
        )
        results["splits"][split] = {
            "cache": str(cache_path),
            "reconstructed_from_train_cache": reconstructed,
            "samples": len(samples),
            "frozen_mace": baseline,
            "mace_fno": corrected,
            "improvement": improvements,
        }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
