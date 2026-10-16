"""Train a G=1 model, verify validation selection, and evaluate accuracy runs."""

import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

from common import command, digest, dump, load, metric_row, now, score, verify_inputs


def main():
    root = Path(__file__).resolve().parent
    manifest = load(root / "manifest.json")
    task = manifest["tasks"][int(sys.argv[1])]
    assert load(root / "preflight.json")["state"] == "passed"
    assert load(root / "smoke.json")["state"] == "passed"
    verify_inputs(root, task["system"])
    assert digest(task["config"]) == task["config_sha256"]
    run = Path(task["checkpoint"]).parent
    run.mkdir(exist_ok=False)
    status = dict(
        state="running",
        stage="training",
        started=now(),
        task=task,
        job_id=os.getenv("SLURM_JOB_ID"),
        array_task=os.getenv("SLURM_ARRAY_TASK_ID"),
    )
    dump(run / "status.json", status)
    try:
        if "reuse_run" in task:
            original = Path(task["reuse_run"])
            assert digest(original / "model.pt") == task["reuse_checkpoint_sha256"]
            for name in ("model.pt", "model.config.yaml", "validation_history.jsonl"):
                shutil.copy2(original / name, run / name)
        else:
            command(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    root / "train.py",
                    "--config",
                    task["config"],
                ],
                run / "training.log",
            )
        import torch
        import yaml

        cfg = yaml.safe_load(Path(task["config"]).read_text())
        history = [
            json.loads(line)
            for line in (run / "validation_history.jsonl").read_text().splitlines()
        ]
        history = [row for row in history if row["step"] is not None]
        assert [row["step"] for row in history] == list(
            range(0, cfg["steps"] + 1, cfg["eval_interval"])
        )
        state = torch.load(task["checkpoint"], map_location="cpu", weights_only=False)
        assert state["completed_steps"] == cfg["steps"]
        assert state["spectral_groups"] == 1 and state["seed"] == task["seed"]
        assert state["mace_training"] == task["regime"]
        assert bool(state.get("mace_state_dict")) == (task["regime"] == "joint")
        assert not state["evaluation_metrics"].get("held-out test")
        metrics = state["evaluation_metrics"]["validation"]
        value = score(metrics, cfg)
        assert math.isclose(
            value,
            min(row["validation_objective"] for row in history),
            rel_tol=2e-6,
            abs_tol=1e-10,
        )
        best = dict(
            checkpoint=task["checkpoint"],
            checkpoint_sha256=digest(task["checkpoint"]),
            selected_step=state["best_step"],
            validation_objective=value,
            metrics=metrics,
            energy_shift_applied=False,
            **metric_row(metrics),
        )
        del state
        result = dict(
            task=task,
            completed_steps=cfg["steps"],
            selections={"best_loss": best},
            selection_recomputed_from_history=True,
            test_evaluated=False,
        )
        dump(run / "results.json", result)
        if task["role"] == "accuracy":
            status["stage"] = "held-out-test"
            dump(run / "status.json", status)
            info = manifest["systems"][task["system"]]
            command(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "mace_fno.cli.evaluate_residual",
                    "--checkpoint",
                    task["checkpoint"],
                    "--test-cache",
                    info["test_cache"],
                    "--splits",
                    "test",
                    "--batch-size",
                    "4",
                    "--device",
                    "cuda",
                    "--dtype",
                    "float64",
                    "--output",
                    run / "test.json",
                ],
                run / "test.log",
            )
            test = load(run / "test.json")
            assert test["splits"]["test"]["samples"] == 134
            assert test["checkpoint"] == task["checkpoint"]
            assert test["best_step"] == best["selected_step"]
            result.update(test_evaluated=True, primary_test=test["splits"]["test"])
        verify_inputs(root, task["system"])
        result["finished"] = now()
        dump(run / "results.json", result)
        status.update(state="complete", stage="complete", finished=now())
        dump(run / "status.json", status)
        print("COMPLETE", task["name"], flush=True)
    except Exception as error:
        status.update(
            state="failed",
            finished=now(),
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        dump(run / "status.json", status)
        raise


if __name__ == "__main__":
    main()
