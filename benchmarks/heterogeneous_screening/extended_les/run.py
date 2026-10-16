"""Run a checked smoke test or one training/evaluation task."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify():
    import les
    import mace

    manifest = json.loads((ROOT / "manifest.json").read_text())
    for module in (mace, les):
        assert str(ROOT / "venv") in module.__file__, module.__file__
    for key, value in manifest["inputs"].items():
        assert sha(ROOT / key) == value["sha256"], key
    for key, value in manifest["sources"].items():
        assert sha(ROOT / key) == value, key
    for key, value in manifest["installed_sources"].items():
        assert sha(key) == value, key
    return manifest


def command(argv, log, cwd=ROOT):
    print("RUN", argv, flush=True)
    with Path(log).open("w") as f:
        subprocess.run(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)


def train(task, mode):
    run = task[mode]
    cueq = mode == "train" and task.get("training_backend") == "cuequivariance"
    if cueq:
        check = json.loads(
            (ROOT / "cueq_validation" / task["name"] / "passed.json").read_text()
        )
        assert check["status"] == "passed" and check["task"] == task["name"]
        assert check["les_config_sha256"] == sha(task["les_arguments"])
        for name, digest in check["source_hashes"].items():
            assert sha(ROOT / name) == digest, name
    directory = Path(run["directory"])
    state = directory / "status.json"
    assert not state.exists(), f"Existing task state: {state}; inspect before resuming"
    record = dict(
        status="running",
        mode=mode,
        start=time.time(),
        job=os.environ.get("SLURM_JOB_ID"),
        task=task["name"],
        config_sha256=sha(run["config"]),
        training_backend="cuequivariance" if cueq else "e3nn",
    )
    state.write_text(json.dumps(record, indent=2) + "\n")
    try:
        launcher = (
            [str(ROOT / "train_cueq.py")] if cueq else ["-m", "mace.cli.run_train"]
        )
        command(
            [sys.executable, "-u", *launcher, "--config", run["config"]],
            directory / "train.log",
            directory,
        )
        assert Path(run["model"]).is_file(), run["model"]
        command(
            [sys.executable, "-u", str(ROOT / "evaluate.py"), mode, str(task["index"])],
            directory / "evaluation.log",
        )
        record.update(
            status="completed", model_sha256=sha(run["model"]), end=time.time()
        )
    except Exception as exc:
        record.update(status="failed", error=repr(exc), end=time.time())
        raise
    finally:
        state.write_text(json.dumps(record, indent=2) + "\n")


def main():
    manifest = verify()
    mode = sys.argv[1]
    if mode == "smoke":
        for task in manifest["tasks"]:
            train(task, mode)
        (ROOT / "smoke_passed.json").write_text(
            json.dumps(
                dict(
                    time=time.time(),
                    tasks=3,
                    checks="4-epoch force training including stage two; reload; flags; finite forces; finite-difference force; GPU memory",
                ),
                indent=2,
            )
            + "\n"
        )
    elif mode == "train":
        assert (ROOT / "smoke_passed.json").is_file(), "Smoke test must pass first"
        train(manifest["tasks"][int(sys.argv[2])], mode)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
