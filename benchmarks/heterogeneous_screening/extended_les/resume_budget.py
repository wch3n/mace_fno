"""Verify an archived budget-only revision, resume, and run existing evaluations."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from run_response import convergence

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def verify():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    revision = json.loads((ROOT / "budget_resume.json").read_text())
    for name, digest in manifest["sources"].items():
        assert sha(ROOT / name) == digest, name
    for name, record in manifest["inputs"].items():
        assert sha(ROOT / name) == record["sha256"], name
    for name, digest in manifest["installed_sources"].items():
        assert sha(name) == digest, name
    for name, version in manifest["environment"]["packages"].items():
        assert importlib.metadata.version(name) == version, name
    archive = ROOT / revision["archive"]
    old = json.loads((archive / "manifest.json").read_text())
    certificate = json.loads((archive / "smoke_passed.json").read_text())
    assert sha(archive / "manifest.json") == revision["previous_manifest_sha256"]
    assert certificate["manifest_sha256"] == revision["previous_manifest_sha256"]
    assert (
        manifest["training"]["epochs"] == 1200
        and manifest["training"]["stage_two_start"] == 450
    )
    assert (
        manifest["inputs"] == old["inputs"]
        and manifest["installed_sources"] == old["installed_sources"]
    )
    for name, digest in old["sources"].items():
        if name in revision["changed_sources"]:
            assert sha(archive / name) == digest, name
        else:
            assert sha(ROOT / name) == digest, name
    for task in manifest["tasks"]:
        path = Path(task["train"]["config"])
        relative = path.relative_to(ROOT)
        before = yaml.safe_load((archive / relative).read_text())
        after = yaml.safe_load(path.read_text())
        diff = {
            k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
        }
        assert diff == {"max_num_epochs", "restart_latest"}, diff
        assert before["max_num_epochs"] == 2400 and not before["restart_latest"]
        assert after["max_num_epochs"] == 1200 and after["restart_latest"]
        record = revision["runs"][task["name"]]
        assert sha(record["checkpoint"]) == record["checkpoint_sha256"]
    return manifest, revision


def main():
    manifest, revision = verify()
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        print("Verified budget-only revision and preserved checkpoints", flush=True)
        return
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task = manifest["tasks"][index]
    run = task["train"]
    directory = Path(run["directory"])
    old_status = json.loads((directory / "status.json").read_text())
    assert old_status.get("status", old_status.get("state")) == "ready_to_resume_1200"
    yukawa = "md" in manifest
    key = "state" if yukawa else "status"
    record = {
        key: "running",
        "seed": task["seed"],
        "mode": "train",
        "task": task["name"],
        "job": os.environ.get("SLURM_JOB_ID"),
        "start": time.time(),
        "training_backend": "cuequivariance",
        "total_epoch_budget": 1200,
        "resume_epoch": revision["runs"][task["name"]]["epoch"] + 1,
    }
    dump(directory / "status.json", record)
    try:
        with (directory / "train.log").open("x") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    str(ROOT / "resume_train.py"),
                    "--config",
                    run["config"],
                ],
                cwd=directory,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        assert Path(run["model"]).is_file()
        assert (
            json.loads((directory / "resume_verified.json").read_text())["status"]
            == "passed"
        )
        assert json.loads((directory / "optimizer_coverage.json").read_text())[
            "all_trainable_parameters_covered"
        ]
        updates = json.loads((directory / "first_response_updates.json").read_text())
        assert updates and all(v["max_abs_update"] > 0 for v in updates.values())
        audit = convergence(directory, 450)
        assert audit["last"]["epoch"] < 1200
        dump(directory / "convergence.json", audit)
        record["convergence"] = audit
        with (directory / "evaluation.log").open("x") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    str(ROOT / "evaluate.py"),
                    "train",
                    str(index),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record.update(
            {
                key: "complete" if yukawa else "completed",
                "model_sha256": sha(run["model"]),
            }
        )
    except BaseException as exc:
        record.update({key: "failed", "error": repr(exc)})
        raise
    finally:
        record["end"] = time.time()
        dump(directory / "status.json", record)


if __name__ == "__main__":
    main()
