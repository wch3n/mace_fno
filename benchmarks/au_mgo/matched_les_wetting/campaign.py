"""Re-evaluate Table VII with the five completed scalar-vector MACE1-LES fits."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

SEEDS = (17, 29, 41, 53, 67)
STATES = ("1-undoped", "3-undoped", "1-doped", "3-doped")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare(args):
    root, training, previous = (
        args.root.resolve(),
        args.training.resolve(),
        args.previous.resolve(),
    )
    tasks = []
    for seed in SEEDS:
        folder = training / f"runs/les_nl1_seed{seed}"
        assert load(folder / "status.json")["state"] == "complete"
        metrics = load(folder / "evaluation/metrics.json")
        assert metrics["architecture"]["product_irreps"] == ["128x0e+128x1o"]
        assert metrics["architecture"]["parameters"] == 463104
        model = Path(metrics["model"])
        assert digest(model) == metrics["model_sha256"]
        tasks.append(
            dict(
                index=len(tasks),
                name=f"au2_mgo_MACE1_LES_seed{seed}",
                seed=seed,
                kind="mace",
                model=str(model),
                model_sha256=metrics["model_sha256"],
                training_metrics=str(folder / "evaluation/metrics.json"),
            )
        )
    root.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "logs", "results", "provenance"):
        (root / name).mkdir()
    old = load(previous / "manifest.json")
    for state in STATES:
        relative = f"inputs/{state}.xyz"
        assert digest(previous / relative) == old["source_checksums"][relative]
        shutil.copy2(previous / relative, root / relative)
    shutil.copy2(
        previous / "evaluate_adsorption_switch.py",
        root / "evaluate_adsorption_switch.py",
    )
    for name in ("campaign.py", "job.slurm"):
        shutil.copy2(Path(__file__).with_name(name), root / name)
    shutil.copytree(
        training / "source",
        root / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for task in tasks:
        shutil.copy2(
            task["training_metrics"],
            root / "provenance" / f"seed{task['seed']}_metrics.json",
        )
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,
        training_campaign=str(training),
        previous_wetting_campaign=str(previous),
        product_irreps=["128x0e+128x1o"],
        fmax=0.01,
        maximum_steps=500,
        selection="same native stage-two validation-loss models as Table VI",
        source_checksums={
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    )
    dump(root / "manifest.json", manifest)
    print(root)


def run(root):
    import les
    import mace
    import numpy as np
    import torch
    from ase.io import read

    root = root.resolve()
    manifest = load(root / "manifest.json")
    for name, expected in manifest["source_checksums"].items():
        assert digest(root / name) == expected, name
    for package in (mace, les):
        assert Path(package.__file__).resolve().is_relative_to(root / "source"), (
            package.__file__
        )
    assert torch.cuda.is_available()
    dump(
        root / "environment.json",
        dict(
            python=sys.version,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            mace=mace.__file__,
            les=les.__file__,
            job=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    for task in manifest["tasks"]:
        model = Path(task["model"])
        assert digest(model) == task["model_sha256"]
        saved = torch.load(model, map_location="cpu", weights_only=False)
        irreps = [str(p.linear.irreps_out) for p in saved.products]
        assert irreps == manifest["product_irreps"], irreps
        assert sum(p.numel() for p in saved.parameters()) == 463104
        del saved
        output = root / "results" / task["name"]
        output.mkdir(exist_ok=False)
        status = dict(
            state="running",
            task=task,
            model=str(model),
            model_sha256=task["model_sha256"],
            product_irreps=irreps,
            job_id=os.environ.get("SLURM_JOB_ID"),
            started=datetime.now(timezone.utc).isoformat(),
        )
        dump(output / "status.json", status)
        try:
            command = [
                sys.executable,
                "-B",
                "-u",
                str(root / "evaluate_adsorption_switch.py"),
                "--model",
                str(model),
                "--model-kind",
                "mace",
                "--device",
                "cuda",
                "--dtype",
                "float64",
                "--structures-dir",
                str(root / "inputs"),
                "--output-dir",
                str(output / "endpoints"),
                "--fmax",
                "0.01",
                "--steps",
                "500",
            ]
            with (output / "evaluation.log").open("x") as handle:
                subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, check=True
                )
            result = load(output / "endpoints/results.json")
            checks = {}
            assert set(result["records"]) == set(STATES)
            for name, record in result["records"].items():
                initial = read(root / "inputs" / f"{name}.xyz")
                final = read(output / "endpoints" / f"{name}-relaxed.xyz")
                moving, fixed = record["movable_indices"], record["fixed_indices"]
                assert len(moving) == 2 and all(
                    initial[i].symbol == "Au" for i in moving
                )
                assert len(fixed) == 108
                assert np.array_equal(initial.numbers, final.numbers)
                assert np.array_equal(initial.pbc, final.pbc)
                assert np.array_equal(initial.cell.array, final.cell.array)
                assert np.allclose(
                    initial.positions[fixed], final.positions[fixed], atol=5e-8, rtol=0
                )
                normal = np.cross(final.cell[0], final.cell[1])
                normal /= np.linalg.norm(normal)
                d = final.get_distance(*moving, mic=True, vector=True)
                angle = float(
                    np.degrees(
                        np.arcsin(np.clip(abs(d @ normal) / np.linalg.norm(d), 0, 1))
                    )
                )
                basin = angle < 45 if record["geometry"] == "wetting" else angle > 45
                checks[name] = dict(
                    converged=record["converged"]
                    and record["final_fmax_eV_per_A"] <= 0.01,
                    orientation_angle_from_plane_deg=angle,
                    expected_orientation=bool(basin),
                    substrate_fixed=True,
                    cell_fixed=True,
                    initial_signed_energy_error_meV=1000
                    * (record["initial_energy_eV"] - record["reference_energy_eV"]),
                )
            clean = all(
                c["converged"] and c["expected_orientation"] for c in checks.values()
            )
            assert digest(model) == task["model_sha256"]
            dump(
                output / "checks.json",
                dict(all_converged_expected_basins=clean, states=checks),
            )
            status.update(
                state="complete" if clean else "needs_review", checks_passed=clean
            )
            assert clean, f"Unconverged or changed orientation: {task['name']}"
            print(task["name"], result["summary"], flush=True)
        except BaseException as exc:
            status.update(
                state="failed", error=repr(exc), traceback=traceback.format_exc()
            )
            raise
        finally:
            status["finished"] = datetime.now(timezone.utc).isoformat()
            dump(output / "status.json", status)
    dump(
        root / "complete.json",
        dict(
            state="complete",
            models=5,
            endpoints=20,
            finished=datetime.now(timezone.utc).isoformat(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else run(args.root)
