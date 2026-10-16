"""Run immutable full-height configurations, verify selected weights, evaluate test."""

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
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    mode, index = sys.argv[1], int(sys.argv[2])
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

    def command(args, log):
        with (run / log).open("x") as handle:
            result = subprocess.run(
                list(map(str, args)), stdout=handle, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print((run / log).read_text()[-16000:], flush=True)
            raise RuntimeError(f"{log}: exit {result.returncode}")

    save()
    try:
        for rel, checksum in manifest["source_hashes"].items():
            assert digest(root / rel) == checksum, f"Changed source/config: {rel}"
        for rel, record in manifest["inputs"].items():
            assert digest(root / rel) == record["sha256"], (
                f"Changed data/backbone: {rel}"
            )
        if mode != "smoke":
            assert all(
                json.loads((Path(t["checkpoint"]).parent / "status.json").read_text())[
                    "state"
                ]
                == "complete"
                for t in manifest["smoke_tasks"]
            )
        status["stage"] = "training"
        save()
        start = time.monotonic()
        command(
            [
                sys.executable,
                "-B",
                "-u",
                root / "measure_train.py",
                "--config",
                task["config"],
            ],
            "training.log",
        )
        status["training_seconds"] = time.monotonic() - start
        memory = json.loads((run / "memory.json").read_text())
        status["peak_allocated_GiB"] = memory["peak_allocated_bytes"] / 1024**3
        if mode == "smoke":
            assert (
                memory["peak_allocated_bytes"] < 0.95 * memory["total_memory_bytes"]
            ), "Insufficient GPU memory headroom"
        import torch

        cp = torch.load(task["checkpoint"], map_location="cpu", weights_only=False)
        assert (
            cp["architecture"] == "nonlinear"
            and cp["z_mixing"] == "full"
            and cp["fno_layers"] == 4
        )
        assert cp["completed_steps"] == task["steps"]
        assert cp["mace_training"] == task["opt_level"] and cp["test_cache"] is None
        assert cp.get("spectral_rank", 0) == 0
        assert bool(cp.get("mace_state_dict")) == (task["opt_level"] == "joint")
        history = [
            json.loads(line)
            for line in (run / "validation_history.jsonl").read_text().splitlines()
        ]
        best = min(row["validation_objective"] for row in history)
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
                root / "evaluate.py",
                task["config"],
                task["test_file"],
                run / "test.json",
            ],
            "test.log",
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
