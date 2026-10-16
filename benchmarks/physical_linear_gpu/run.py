"""Run one immutable linear fit, verify selection, and evaluate held-out errors."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    mode, index = sys.argv[1], int(sys.argv[2])
    assert mode in ["smoke", "train"]
    task = manifest["smoke_tasks" if mode == "smoke" else "tasks"][index]
    run = Path(task["checkpoint"]).parent
    run.mkdir(exist_ok=False)
    status = dict(
        state="running",
        stage="preflight",
        task=task,
        job_id=os.getenv("SLURM_JOB_ID"),
        started=time.time(),
    )

    def save():
        (run / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    def command(args, name):
        with (run / name).open("x") as log:
            result = subprocess.run(
                list(map(str, args)), stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print((run / name).read_text()[-10000:], flush=True)
            raise RuntimeError(f"{name}: exit {result.returncode}")

    save()
    try:
        for name, checksum in manifest["source_hashes"].items():
            assert digest(root / name) == checksum, name
        for name, record in manifest["inputs"].items():
            assert digest(root / name) == record["sha256"], name
        if mode == "train":
            assert (
                json.loads((root / "smoke_passed.json").read_text())["state"]
                == "passed"
            )
        status["stage"] = "training"
        save()
        start = time.monotonic()
        command(
            [
                sys.executable,
                "-B",
                "-u",
                root / "measured.py",
                "train",
                "--config",
                task["config"],
            ],
            "training.log",
        )
        status["training_seconds"] = time.monotonic() - start
        import torch

        cp = torch.load(task["checkpoint"], map_location="cpu", weights_only=False)
        assert cp["architecture"] == "linear" and cp["completed_steps"] == task["steps"]
        assert cp["spectral_groups"] == 1 and cp["seed"] == task["seed"]
        assert cp["mace_training"] == task["opt_level"] and cp["test_cache"] is None
        assert bool(cp.get("mace_state_dict")) == (task["opt_level"] == "joint")
        if task["system"] == "au2_mgo":
            assert cp["z_mixing"] == "spectral"
        history = [
            json.loads(line)
            for line in (run / "validation_history.jsonl").read_text().splitlines()
            if line.strip()
        ]
        best = min(r["validation_objective"] for r in history)
        assert math.isclose(
            cp["best_validation_objective"], best, rel_tol=2e-6, abs_tol=1e-10
        )
        status.update(
            selected_step=cp["best_step"],
            checkpoint_sha256=digest(task["checkpoint"]),
            stage="test",
        )
        del cp
        save()
        command(
            [
                sys.executable,
                "-B",
                "-u",
                root / "measured.py",
                "evaluate",
                task["config"],
                task["test_file"],
                run / "test.json",
            ],
            "test.log",
        )
        result = json.loads((run / "test.json").read_text())
        assert (
            not result["energy_shift_applied"]
            and result["checkpoint"] == task["checkpoint"]
        )
        assert result["samples"] == (
            4 if mode == "smoke" else 134 if task["system"] == "water" else 500
        )
        assert all(
            math.isfinite(v)
            for k, v in result["splits"]["test"]["mace_fno"].items()
            if isinstance(v, (float, int))
        )
        status.update(state="complete", stage="complete", finished=time.time())
        save()
        print("COMPLETE", task["name"], flush=True)
    except Exception as error:
        status.update(
            state="failed",
            error=repr(error),
            traceback=traceback.format_exc(),
            finished=time.time(),
        )
        save()
        raise


if __name__ == "__main__":
    main()
