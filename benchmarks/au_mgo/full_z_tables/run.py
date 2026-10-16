"""Evaluate one immutable full-z model per GPU, then collect all models."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(args, logfile):
    with logfile.open("x") as stream:
        subprocess.run(
            [str(x) for x in args], stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def worker(index):
    import numpy as np
    from ase.io import read

    manifest = json.loads((ROOT / "manifest.json").read_text())
    task = manifest["tasks"][index]
    assert digest(Path(task["checkpoint"])) == task["checkpoint_sha256"]
    output = ROOT / "results" / task["name"]
    output.mkdir(exist_ok=False)
    status = dict(
        state="running",
        task=task,
        job_id=os.environ.get("SLURM_JOB_ID"),
        started=datetime.now(timezone.utc).isoformat(),
    )
    save(output / "status.json", status)
    try:
        if task.get("completed_test"):
            assert task["training_complete"] and not task["preliminary"]
            archived_test = Path(task["completed_test"])
            assert digest(archived_test) == task["completed_test_sha256"]
            test = json.loads(archived_test.read_text())
            assert test["checkpoint"] == task["source_checkpoint"]
            test["checkpoint"] = task["checkpoint"]
            test["reused_completed_test"] = dict(
                source=task["completed_test_source"],
                sha256=task["completed_test_sha256"],
                checkpoint_sha256=task["checkpoint_sha256"],
            )
            save(output / "test.json", test)
        else:
            command(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    ROOT / "evaluate_test.py",
                    task["config"],
                    ROOT / "data/test.xyz",
                    output / "test.json",
                ],
                output / "test.log",
            )
        test = json.loads((output / "test.json").read_text())
        assert test["samples"] == 500 and not test["energy_shift_applied"]
        assert (
            test["checkpoint"] == task["checkpoint"]
            and test["best_step"] == task["selected_step"]
        )
        command(
            [
                sys.executable,
                "-B",
                "-u",
                ROOT / "evaluate_adsorption_switch.py",
                "--model",
                task["checkpoint"],
                "--model-kind",
                "mace-fno",
                "--device",
                "cuda",
                "--dtype",
                "float64",
                "--structures-dir",
                ROOT / "inputs",
                "--output-dir",
                output / "endpoints",
                "--fmax",
                "0.01",
                "--steps",
                "500",
            ],
            output / "wetting.log",
        )
        result = json.loads((output / "endpoints/results.json").read_text())
        checks = {}
        for name, record in result["records"].items():
            initial = read(ROOT / "inputs" / f"{name}.xyz")
            final = read(output / "endpoints" / f"{name}-relaxed.xyz")
            moving, fixed = record["movable_indices"], record["fixed_indices"]
            assert len(moving) == 2 and len(fixed) == 108
            assert np.array_equal(initial.cell.array, final.cell.array)
            assert np.allclose(
                initial.positions[fixed], final.positions[fixed], atol=5e-8, rtol=0
            )
            normal = np.cross(final.cell[0], final.cell[1])
            normal /= np.linalg.norm(normal)
            delta = final.get_distance(*moving, mic=True, vector=True)
            angle = float(
                np.degrees(
                    np.arcsin(
                        np.clip(abs(delta @ normal) / np.linalg.norm(delta), 0, 1)
                    )
                )
            )
            basin = angle < 45 if record["geometry"] == "wetting" else angle > 45
            checks[name] = dict(
                converged=record["converged"],
                expected_orientation=bool(basin),
                orientation_angle_from_plane_deg=angle,
                substrate_fixed=True,
                cell_fixed=True,
            )
        clean = all(
            c["converged"] and c["expected_orientation"] for c in checks.values()
        )
        save(
            output / "checks.json",
            dict(all_converged_expected_basins=clean, states=checks),
        )
        assert digest(Path(task["checkpoint"])) == task["checkpoint_sha256"]
        status.update(
            state="complete" if clean else "needs_review", checks_passed=clean
        )
        print(task["name"], status["state"], flush=True)
    except Exception as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        save(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.index is not None:
        worker(args.index)
        return
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["checksums"].items():
        assert digest(ROOT / name) == expected, name
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert devices and all(devices), devices

    def queue(device, indices):
        for index in indices:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            subprocess.run(
                [sys.executable, "-B", "-u", __file__, "--index", str(index)],
                env=env,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        jobs = [
            executor.submit(
                queue, device, list(range(i, len(manifest["tasks"]), len(devices)))
            )
            for i, device in enumerate(devices)
        ]
        for job in jobs:
            job.result()
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "collect.py"), str(ROOT)], check=True
    )


if __name__ == "__main__":
    main()
