"""Run numerical G=1 checks or matched GPU measurements after training.

Water spectral checks were discontinued at the user's request on 2026-10-11.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(args, log):
    print("COMMAND:", *args, flush=True)
    with log.open("w") as stream:
        subprocess.run(
            [str(a) for a in args], stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def checked_result(task):
    run = Path(task["checkpoint"]).parent
    assert read(run / "status.json")["state"] == "complete", task["name"]
    result = read(run / "results.json")
    assert (
        result["test_evaluated"] and result["completed_steps"] == task["expected_steps"]
    )
    best = result["selections"]["best_loss"]
    assert best["checkpoint"] == task["checkpoint"]
    assert digest(task["checkpoint"]) == best["checkpoint_sha256"]
    assert result["primary_test"]["samples"] == 134
    return result


def audits(root, manifest, index):
    task = manifest["tasks"][index]
    assert task["role"] == "accuracy" and task["spectral_groups"] == 1
    result = checked_result(task)
    system = manifest["systems"][task["system"]]
    cache = system["test_cache"]
    assert digest(cache) == system["input_checksums"][cache]
    output = root / "audits" / task["name"]
    output.mkdir(parents=True, exist_ok=False)
    common = [
        "--checkpoint",
        task["checkpoint"],
        "--sample-cache",
        cache,
        "--samples",
        "32",
        "--seed",
        "947",
        "--device",
        "cuda",
    ]
    command(
        [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "mace_fno.cli.audit_3d",
            *common,
            "--fd-step",
            "1e-4",
            "--fd-components",
            "12",
            "--strict",
            "--output",
            output / "numerical.json",
        ],
        output / "numerical.log",
    )
    assert checked_result(task) == result
    save(
        output / "status.json",
        dict(
            state="complete",
            task=task,
            requested_checks=["numerical"],
            spectral_checks="not_requested",
            checkpoint_sha256=result["selections"]["best_loss"]["checkpoint_sha256"],
            finished=datetime.now(timezone.utc).isoformat(),
            job_id=os.getenv("SLURM_JOB_ID"),
        ),
    )


def measurements(root, manifest, scripts):
    import numpy as np
    import yaml

    output = root / "measurements"
    output.mkdir(exist_ok=False)
    for name in ("time_inference.py", "measure_memory.py", "compare_interlacing.py"):
        shutil.copy2(scripts / name, output / name)
    controls = read(scripts / "inference_controls.json")
    models = controls["models"]
    for spec in models.values():
        assert digest(spec["path"]) == spec["expected_sha256"]
        if "reference_predictions" in spec:
            assert (
                digest(spec["reference_predictions"])
                == spec["reference_predictions_sha256"]
            )
    info = manifest["systems"]["water_scan_nl1"]
    assert controls["test_file"] == info["test_file"]
    assert digest(controls["test_file"]) == controls["test_sha256"]
    timing_models = dict(models)
    for regime in ("frozen", "joint"):
        task = next(
            t
            for t in manifest["tasks"]
            if t["role"] == "accuracy"
            and (t["mace_layers"], t["regime"], t["seed"]) == (1, regime, 17)
        )
        result = checked_result(task)
        cfg = yaml.safe_load(Path(task["config"]).read_text())
        assert cfg["spectral_groups"] == 1
        assert cfg["mace_model"] == models["MACE_1"]["path"]
        spec = dict(
            kind="fno",
            path=task["checkpoint"],
            mace_model=cfg["mace_model"],
            expected_sha256=result["selections"]["best_loss"]["checkpoint_sha256"],
            mace_sha256=models["MACE_1"]["expected_sha256"],
            reference_test=result["primary_test"],
        )
        models[f"MACE_1-FNO_{regime}"] = spec
        for level in (1, 2):
            name = f"MACE_1-FNO_{regime}_level{level}"
            timing_models[name] = dict(spec, volume_interlacing_override=level)
            models[name] = timing_models[name]
    shared = dict(test_file=controls["test_file"], test_sha256=controls["test_sha256"])
    protocol = dict(
        cpu_threads=4,
        structures=3,
        warmup=5,
        rounds=3,
        repeats=10,
        expected_gpu="NVIDIA A100-SXM4-40GB",
        paired_timing_directory="timing",
    )
    save(
        output / "manifest.json",
        dict(protocol, systems={"water_scan": dict(shared, models=models)}),
    )
    save(
        output / "timing_manifest.json",
        dict(protocol, systems={"water_scan": dict(shared, models=timing_models)}),
    )
    command(
        [
            sys.executable,
            "-B",
            "-u",
            output / "time_inference.py",
            "--manifest",
            output / "timing_manifest.json",
            "--output",
            output / "timing",
        ],
        output / "timing.log",
    )
    timing = read(output / "timing/summary.json")["water_scan"]
    for name, meta in timing["models"].items():
        if "FNO" in name:
            assert meta["spectral_groups"] == 1
    shutil.copy2(output / "timing/summary.json", output / "timing_reference.json")
    (output / "structures").mkdir()
    (output / "results").mkdir()
    shutil.copy2(output / "timing/water_scan_structures.xyz", output / "structures")
    rows = []
    for name in timing_models:
        command(
            [
                sys.executable,
                "-B",
                "-u",
                output / "measure_memory.py",
                "water_scan",
                name,
            ],
            output / (name + "_memory.log"),
        )
        data = read(output / "results" / ("water_scan__" + name + ".json"))
        assert len(data["calls"]) == 9
        time_data = timing["timing"][name]["end_to_end"]
        assert time_data["calls"] == 90
        rows.append(
            dict(
                model=name,
                parameters=timing["models"][name]["parameters"],
                median_time_ms=time_data["median_ms"],
                peak_allocated_MiB=max(c["peak_allocated_bytes"] for c in data["calls"])
                / 2**20,
            )
        )
    for regime in ("frozen", "joint"):
        for level in (1, 2):
            command(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    output / "compare_interlacing.py",
                    "water_scan",
                    regime,
                    str(level),
                ],
                output / f"interlacing_{regime}_{level}.log",
            )
            folder = output / "results" / f"water_scan__{regime}__seed17__level{level}"
            data = np.load(folder / "test_predictions.npz")
            errors = [
                1000
                * np.sqrt(
                    np.mean(
                        ((data["energy_eV"] - data["target_energy_eV"]) / data["atoms"])
                        ** 2
                    )
                ),
                1000
                * np.sqrt(
                    np.mean((data["forces_eV_A"] - data["target_forces_eV_A"]) ** 2)
                ),
            ]
            report = read(folder / "summary.json")
            np.testing.assert_allclose(
                errors,
                [
                    report["test"]["energy_rmse_meV_atom"],
                    report["test"]["force_rmse_meV_A"],
                ],
                rtol=1e-12,
            )
    with (output / "costs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(
        output / "status.json",
        dict(
            state="complete",
            spectral_groups=1,
            seed=17,
            finished=datetime.now(timezone.utc).isoformat(),
            job_id=os.getenv("SLURM_JOB_ID"),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("mode", choices=("audits", "measurements"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.campaign.resolve()
    scripts = Path(__file__).resolve().parent
    for relative, expected in read(scripts / "sha256.json").items():
        assert digest(scripts / relative) == expected, relative
    manifest = read(root / "manifest.json")
    for relative, expected in manifest["source_checksums"].items():
        assert digest(root / relative) == expected, relative
    if args.mode == "audits":
        assert args.index is not None and 0 <= args.index < 20
        audits(root, manifest, args.index)
    else:
        measurements(root, manifest, scripts)


if __name__ == "__main__":
    main()
