"""Finalize agreed full-z budgets using native full-state resume and validation selection."""

import argparse
import fcntl
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import resume_helpers as resume
import table_helpers as tables


def load_task(root, index):
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["checksums"].items():
        assert resume.digest(root / relative) == expected, relative
    item = manifest["tasks"][index]
    parent = Path(item["parent"])
    original = json.loads((parent / "manifest.json").read_text())
    assert resume.digest(parent / "manifest.json") == item["parent_manifest_sha256"]
    for relative, expected in original["source_hashes"].items():
        assert resume.digest(parent / relative) == expected, relative
    for relative, record in original["inputs"].items():
        assert resume.digest(parent / relative) == record["sha256"], relative
    task = original["tasks"][item["parent_index"]].copy()
    assert task["name"] == item["name"] and task["steps"] == 240000
    return item, task


def inspect(item, task):
    import torch

    torch.set_num_threads(1)
    if item["mode"] == "evaluate":
        run = Path(task["checkpoint"]).parent
        status = json.loads((run / "status.json").read_text())
        cp = torch.load(
            task["checkpoint"], map_location="cpu", weights_only=False, mmap=True
        )
        assert status["state"] == status["stage"] == "complete"
        assert cp["completed_steps"] == 240000
        assert resume.digest(task["checkpoint"]) == status["checkpoint_sha256"]
        return dict(
            completed_steps=240000, best_step=cp["best_step"], target_steps=240000
        )
    state = resume.checked_state(task)
    _, committed = resume.checked_history(task, state)
    target = item["target_steps"] or state["completed_steps"]
    assert state["completed_steps"] <= target <= task["steps"]
    return dict(
        completed_steps=state["completed_steps"],
        best_step=state["best_step"],
        target_steps=target,
        committed_history_rows=len(committed),
        compatible=True,
        optimizer_present=True,
        rng_present=True,
    )


def command(arguments, logfile):
    with logfile.open("x") as handle:
        subprocess.run(
            list(map(str, arguments)),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def finish(root, index, item, task):
    import torch
    import yaml

    run = Path(task["checkpoint"]).parent
    lock = (run / "continuation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Dependencies must have ended before any output checkpoint can be replaced.
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%T"], text=True
    ).splitlines()
    live = [row for row in queue if row.split("|")[0] == item["previous_job"]]
    assert not live, f"Previous training allocation is still active: {live}"
    report = inspect(item, task)
    output = root / "results" / task["name"]
    output.mkdir(exist_ok=False)
    prior = json.loads((run / "status.json").read_text())
    status = dict(
        state="running",
        stage=item["mode"],
        job_id=os.environ["SLURM_JOB_ID"],
        task=task,
        original_budget_steps=240000,
        target_steps=report["target_steps"],
        resumed_from_step=report["completed_steps"],
        parent_job_id=prior.get("job_id"),
        reason=item["reason"],
        started=time.time(),
        budget_revision=str(root),
    )

    def save():
        resume.write_json(output / "status.json", status)
        if item["mode"] != "evaluate":
            resume.write_json(run / "status.json", status)

    save()
    try:
        for name in ("status.json", "validation_history.jsonl"):
            # Preserve the status from before this revision, not the running status.
            if name == "status.json":
                resume.write_json(output / "prior.status.json", prior)
            else:
                shutil.copy2(run / name, output / name)
        if item["mode"] != "evaluate":
            for name in ("model.pt", "model.last.pt", "model.energy.pt"):
                if (run / name).exists():
                    os.link(run / name, output / ("prior." + name))
            state = resume.checked_state(task)
            history, committed = resume.checked_history(task, state)
            if len(history) != len(committed):
                temporary = run / "validation_history.budget.tmp"
                temporary.write_text(
                    "".join(json.dumps(row) + "\n" for row in committed)
                )
                temporary.replace(run / "validation_history.jsonl")
            del state
            gc.collect()
            cfg = yaml.safe_load(Path(task["config"]).read_text())
            cfg["steps"] = report["target_steps"]
            revised = output / "training.yaml"
            revised.write_text(yaml.safe_dump(cfg, sort_keys=False))
            command(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    __file__,
                    root,
                    index,
                    "--train",
                    report["completed_steps"],
                ],
                output / "training.log",
            )
        cp = torch.load(
            task["checkpoint"], map_location="cpu", weights_only=False, mmap=True
        )
        assert cp["completed_steps"] == report["target_steps"]
        assert (cp["architecture"], cp["z_mixing"], cp["fno_layers"]) == (
            "nonlinear",
            "full",
            4,
        )
        assert (cp["seed"], cp["mace_training"]) == (task["seed"], task["opt_level"])
        history = [
            json.loads(line)
            for line in (run / "validation_history.jsonl").read_text().splitlines()
        ]
        assert math.isclose(
            cp["best_validation_objective"],
            min(r["validation_objective"] for r in history),
            rel_tol=2e-6,
        )
        if item["mode"] == "finalize":
            assert cp["best_step"] == report["best_step"]
        status.update(
            stage="test",
            selected_step=cp["best_step"],
            completed_steps=cp["completed_steps"],
            checkpoint_sha256=resume.digest(task["checkpoint"]),
        )
        del cp
        gc.collect()
        save()
        if item["mode"] != "evaluate":
            assert not (run / "test.json").exists()
            command(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    Path(item["parent"]) / "evaluate.py",
                    output / "training.yaml",
                    task["test_file"],
                    run / "test.json",
                ],
                output / "test.log",
            )
        test = json.loads((run / "test.json").read_text())
        assert test["samples"] == 500 and not test["energy_shift_applied"]
        assert (
            test["checkpoint"] == task["checkpoint"]
            and test["best_step"] == status["selected_step"]
        )
        assert test["checkpoint_selection"] == "best_validation_loss"
        status.update(state="complete", stage="complete", finished=time.time())
        save()
        # Wetting uses an immutable snapshot of the exact test-evaluated checkpoint.
        evaluation = output / "evaluation"
        evaluation.mkdir()
        (evaluation / "results").mkdir()
        (evaluation / "inputs").symlink_to(root / "inputs", target_is_directory=True)
        shutil.copy2(
            root / "evaluate_adsorption_switch.py",
            evaluation / "evaluate_adsorption_switch.py",
        )
        snapshot = evaluation / "model.pt"
        os.link(task["checkpoint"], snapshot)
        archived_test = evaluation / "completed_test.json"
        shutil.copy2(run / "test.json", archived_test)
        evaluation_task = dict(
            name=task["name"],
            checkpoint=str(snapshot),
            checkpoint_sha256=status["checkpoint_sha256"],
            source_checkpoint=task["checkpoint"],
            selected_step=status["selected_step"],
            completed_test=str(archived_test),
            completed_test_sha256=resume.digest(archived_test),
            completed_test_source=str(run / "test.json"),
            training_complete=True,
            preliminary=False,
            regime=task["opt_level"],
            seed=task["seed"],
            depth=task["depth"],
            budget=report["target_steps"],
        )
        resume.write_json(evaluation / "manifest.json", dict(tasks=[evaluation_task]))
        tables.ROOT = evaluation
        tables.worker(0)
        wet_status = json.loads(
            (evaluation / "results" / task["name"] / "status.json").read_text()
        )
        status.update(
            wetting_state=wet_status["state"],
            wetting_results=str(evaluation / "results" / task["name"]),
            evaluation_finished=time.time(),
        )
        save()
        print(json.dumps(status), flush=True)
    except Exception as error:
        # Do not invalidate completed training/test if only the wetting stage fails.
        if status["state"] == "complete":
            status.update(wetting_state="failed", wetting_error=repr(error))
        else:
            status.update(
                state="failed", error=repr(error), traceback=traceback.format_exc()
            )
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--train", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    item, task = load_task(root, args.index)
    if args.train is not None:
        task["config"] = str(root / "results" / task["name"] / "training.yaml")
        resume.training(task, args.train)
    elif args.check:
        print(json.dumps(dict(name=task["name"], **inspect(item, task))), flush=True)
    else:
        finish(root, args.index, item, task)


if __name__ == "__main__":
    main()
