"""Full-state continuation for the four author-requested Au2/MgO runs."""

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_task(root, index):
    manifest = json.loads((root / "manifest.json").read_text())
    task = manifest["tasks"][index]
    allowed = {
        "au2_mgo_low_rank_full_z_seed17_20261009": (2, 3),
        "au2_mgo_full_zmix_nl1_5seed_20261006T070300Z": (8, 9),
    }
    assert index in allowed[root.name]
    assert task["opt_level"] in ("frozen", "joint") and task["steps"] == 240000
    for rel, checksum in manifest["source_hashes"].items():
        assert digest(root / rel) == checksum, f"Changed source/config: {rel}"
    for rel, record in manifest["inputs"].items():
        assert digest(root / rel) == record["sha256"], f"Changed input: {rel}"
    assert all(
        json.loads((Path(t["checkpoint"]).parent / "status.json").read_text())["state"]
        == "complete"
        for t in manifest["smoke_tasks"]
    )
    return task


def checked_state(task):
    import torch

    from mace_fno.cli.config import parse_arguments
    from mace_fno.training.configuration import TrainingConfig
    from mace_fno.training.resume import input_fingerprints, validate_resume_checkpoint

    last = Path(task["checkpoint"]).with_name("model.last.pt")
    configuration = TrainingConfig.from_namespace(
        parse_arguments(["--config", task["config"], "--resume", str(last)])
    )
    state = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
    assert state["resume_format_version"] == 1
    validate_resume_checkpoint(
        state, configuration, fingerprints=input_fingerprints(configuration)
    )
    assert state["device_type"] == "cuda" and not state["stopped_early"]
    assert state["optimizer_state_dict"]["state"] and state["scheduler_state_dict"]
    assert state["rng_state"]["cuda"] and state["energy_selector"] is not None
    assert configuration.model.z_mixing == "full"
    assert getattr(configuration.model, "spectral_rank", 0) == task.get("rank", 0)
    assert configuration.runtime.seed == task["seed"]
    assert configuration.optimization.mace_training == task["opt_level"]
    assert bool(state.get("mace_state_dict")) == (task["opt_level"] == "joint")
    return state


def checked_history(task, state):
    history = [
        json.loads(line)
        for line in (Path(task["checkpoint"]).parent / "validation_history.jsonl")
        .read_text()
        .splitlines()
    ]
    committed = [row for row in history if row["step"] <= state["completed_steps"]]
    assert committed and committed[-1]["step"] == state["completed_steps"]
    assert all(a["step"] < b["step"] for a, b in zip(committed, committed[1:]))
    assert math.isclose(
        min(row["validation_objective"] for row in committed),
        state["best_objective"],
        rel_tol=2e-6,
        abs_tol=1e-10,
    )
    assert 0 < state["completed_steps"] <= task["steps"]
    return history, committed


def training(task, start_step):
    import yaml

    import mace_fno.training.trainer as trainer
    from mace_fno.cli.train_residual import main as native_train

    cfg = yaml.safe_load(Path(task["config"]).read_text())
    history = Path(task["checkpoint"]).parent / "validation_history.jsonl"
    original = trainer.print_metrics

    def print_metrics(label, metrics):
        original(label, metrics)
        match = re.fullmatch(r"validation step (\d+)", label)
        if match:
            step = int(match[1])
            assert step > start_step
            objective = sum(
                cfg[f"{key}_weight"]
                * (metrics[f"{key}_rmse"] / cfg[f"{key}_scale"]) ** 2
                for key in ("energy", "force")
            )
            with history.open("a") as handle:
                handle.write(
                    json.dumps(
                        dict(
                            step=step,
                            label=label,
                            metrics=metrics,
                            validation_objective=objective,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )

    trainer.print_metrics = print_metrics
    sys.argv = [
        sys.argv[0],
        "--config",
        task["config"],
        "--resume",
        str(Path(task["checkpoint"]).with_name("model.last.pt")),
    ]
    native_train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--train", type=int, metavar="START_STEP")
    args = parser.parse_args()
    root = args.root.resolve()
    task = load_task(root, args.index)
    if args.train is not None:
        training(task, args.train)
        return
    run = Path(task["checkpoint"]).parent
    if not args.check:
        lock = (run / "continuation.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = checked_state(task)
    history, committed = checked_history(task, state)
    start_step = state["completed_steps"]
    report = dict(
        task=task["name"],
        seed=task["seed"],
        rank=task.get("rank", 0),
        completed_steps=start_step,
        target_steps=task["steps"],
        best_step=state["best_step"],
        learning_rates=state["scheduler_state_dict"]["_last_lr"],
        compatible=True,
        committed_history_rows=len(committed),
        uncommitted_history_rows=len(history) - len(committed),
        optimizer_present=bool(state["optimizer_state_dict"]["state"]),
        rng_present=bool(state["rng_state"]["cuda"]),
        energy_selection_present=state["energy_selector"] is not None,
    )
    print(json.dumps(report), flush=True)
    if args.report:
        report["last_checkpoint_sha256"] = digest(
            Path(task["checkpoint"]).with_name("model.last.pt")
        )
        write_json(args.report, report)
    if args.check:
        return
    prior_status = json.loads((run / "status.json").read_text())
    if prior_status["state"] == "complete":
        print("Already complete; no continuation needed.", flush=True)
        return
    attempt = run / ("resume_" + os.environ["SLURM_JOB_ID"])
    attempt.mkdir(exist_ok=False)
    for name in ("status.json", "validation_history.jsonl"):
        shutil.copy2(run / name, attempt / name)
    # Native checkpoint saves use atomic replacement, so these links preserve
    # the exact input files without copying several GB per continuation.
    for name in ("model.pt", "model.last.pt", "model.energy.pt"):
        if (run / name).exists():
            os.link(run / name, attempt / name)
    if len(committed) != len(history):
        temporary = run / "validation_history.resume.tmp"
        temporary.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in committed)
        )
        temporary.replace(run / "validation_history.jsonl")
    del state
    gc.collect()
    status = dict(
        state="running",
        stage="resume_training",
        task=task,
        job_id=os.environ["SLURM_JOB_ID"],
        parent_job_id=prior_status.get("job_id"),
        started=time.time(),
        resumed_from_step=start_step,
        resume_directory=str(attempt),
    )

    def save():
        write_json(run / "status.json", status)
        write_json(attempt / "resume.status.json", status)

    def command(arguments, filename):
        with (attempt / filename).open("x") as handle:
            result = subprocess.run(
                list(map(str, arguments)), stdout=handle, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print((attempt / filename).read_text()[-16000:], flush=True)
            raise RuntimeError(f"{filename}: exit {result.returncode}")

    save()
    try:
        command(
            [
                sys.executable,
                "-B",
                "-u",
                Path(__file__).resolve(),
                root,
                args.index,
                "--train",
                start_step,
            ],
            "training.log",
        )
        import torch

        cp = torch.load(
            task["checkpoint"], map_location="cpu", weights_only=False, mmap=True
        )
        assert cp["completed_steps"] == task["steps"]
        assert (
            cp["architecture"] == "nonlinear"
            and cp["z_mixing"] == "full"
            and cp["fno_layers"] == 4
        )
        assert cp["mace_training"] == task["opt_level"]
        assert cp.get("spectral_rank", 0) == task.get("rank", 0)
        assert bool(cp.get("mace_state_dict")) == (task["opt_level"] == "joint")
        history = [
            json.loads(line)
            for line in (run / "validation_history.jsonl").read_text().splitlines()
        ]
        assert math.isclose(
            cp["best_validation_objective"],
            min(row["validation_objective"] for row in history),
            rel_tol=2e-6,
            abs_tol=1e-10,
        )
        status.update(
            stage="test",
            selected_step=cp["best_step"],
            checkpoint_sha256=digest(task["checkpoint"]),
        )
        del cp
        save()
        assert not (run / "test.json").exists()
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
