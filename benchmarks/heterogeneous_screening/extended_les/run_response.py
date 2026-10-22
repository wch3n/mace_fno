"""Verify, train and evaluate the five-seed induced-response campaign."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify():
    import les
    import mace

    manifest = json.loads((ROOT / "manifest.json").read_text())
    for module in (mace, les):
        assert (
            Path(module.__file__).resolve()
            == Path(manifest["environment"][module.__name__ + "_path"]).resolve()
        )
    for key, record in manifest["inputs"].items():
        assert sha(ROOT / key) == record["sha256"], key
    for key, digest in manifest["sources"].items():
        assert sha(ROOT / key) == digest, key
    for key, digest in manifest["installed_sources"].items():
        assert sha(key) == digest, key
    if manifest.get("interaction_first"):
        import yaml

        first = manifest["interaction_first"]
        assert first == "RealAgnosticInteractionBlock"
        allowed = {
            "name",
            "interaction_first",
            "work_dir",
            "log_dir",
            "model_dir",
            "checkpoints_dir",
            "results_dir",
            "train_file",
            "valid_file",
            "les_arguments",
            "max_num_epochs",
            "restart_latest",
        }
        for task in manifest["tasks"]:
            for mode in ("train", "smoke"):
                if mode not in task:
                    continue
                config = yaml.safe_load(Path(task[mode]["config"]).read_text())
                original = yaml.safe_load(
                    (
                        ROOT / "provenance" / f"seed{task['seed']}_{mode}.yaml"
                    ).read_text()
                )
                differences = {
                    k
                    for k in original.keys() | config.keys()
                    if original.get(k) != config.get(k)
                }
                assert differences <= allowed, differences
                assert (
                    config["interaction_first"] == first
                    and not config["restart_latest"]
                )
                assert config["enable_cueq"] and config["seed"] == task["seed"]
                assert sha(config["les_arguments"]) == sha(
                    ROOT / "provenance/les_response.yaml"
                )
        return manifest
    certificate = manifest["conversion_certificate"]
    assert sha(ROOT / certificate["path"]) == certificate["sha256"]
    check = json.loads((ROOT / certificate["path"]).read_text())
    for name, digest in check["source_hashes"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(manifest["tasks"][0]["les_arguments"]) == check["les_config_sha256"]
    return manifest


def command(argv, log, cwd):
    print("RUN", argv, flush=True)
    with Path(log).open("w") as handle:
        subprocess.run(
            argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True
        )


def convergence(directory, stage_two):
    lines = (directory / "train.log").read_text().splitlines()
    stage_two_seen = False
    stopped = False
    rows = []
    for line in lines:
        if "Changing loss based on Stage Two Weights" in line:
            stage_two_seen = True
        if (
            stage_two_seen
            and "Stopping optimization after" in line
            and "starting Stage Two" not in line
        ):
            stopped = True
        match = re.search(
            r"Epoch (\d+):.*loss=([\d.eE+-]+).*RMSE_E_per_atom=\s*([\d.eE+-]+).*RMSE_F=\s*([\d.eE+-]+)",
            line,
        )
        if match:
            rows.append(
                dict(
                    epoch=int(match[1]),
                    loss=float(match[2]),
                    energy_rmse=float(match[3]),
                    force_rmse=float(match[4]),
                )
            )
    assert stage_two_seen and rows
    stage_rows = [r for r in rows if r["epoch"] >= stage_two]
    assert stage_rows
    return dict(
        converged=stopped,
        criterion="50 validation epochs without improved stage-two selection loss",
        stop_reason="validation_plateau" if stopped else "epoch_budget_exhausted",
        evaluations=len(rows),
        last=rows[-1],
        best=min(stage_rows, key=lambda x: x["loss"]),
    )


def train(manifest, index, mode):
    task = manifest["tasks"][index]
    run = task[mode]
    directory = Path(run["directory"])
    status = directory / "status.json"
    assert not status.exists(), (
        f"Existing state requires explicit resume handling: {status}"
    )
    if mode == "train":
        check = json.loads((ROOT / "smoke_passed.json").read_text())
        assert check["manifest_sha256"] == sha(ROOT / "manifest.json")
    record = dict(
        status="running",
        mode=mode,
        seed=task["seed"],
        task=task["name"],
        training_backend="cuequivariance",
        all_readouts_optimized=True,
        job=os.environ.get("SLURM_JOB_ID"),
        start=time.time(),
        config_sha256=sha(run["config"]),
    )
    status.write_text(json.dumps(record, indent=2) + "\n")
    try:
        command(
            [
                sys.executable,
                "-u",
                str(ROOT / "train_response.py"),
                "--config",
                run["config"],
            ],
            directory / "train.log",
            directory,
        )
        assert Path(run["model"]).is_file()
        coverage = json.loads((directory / "optimizer_coverage.json").read_text())
        assert coverage["all_trainable_parameters_covered"]
        updates = json.loads((directory / "first_response_updates.json").read_text())
        assert updates and all(r["max_abs_update"] > 0 for r in updates.values()), (
            updates
        )
        check = convergence(
            directory, manifest["training"]["stage_two_start"] if mode == "train" else 2
        )
        (directory / "convergence.json").write_text(json.dumps(check, indent=2) + "\n")
        record["convergence"] = check
        command(
            [sys.executable, "-u", str(ROOT / "evaluate.py"), mode, str(index)],
            directory / "evaluation.log",
            ROOT,
        )
        record.update(
            status="completed", end=time.time(), model_sha256=sha(run["model"])
        )
    except Exception as exc:
        record.update(status="failed", end=time.time(), error=repr(exc))
        raise
    finally:
        status.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    manifest = verify()
    mode = sys.argv[1]
    if mode == "check":
        if manifest.get("interaction_first"):
            command(
                [sys.executable, "-u", str(ROOT / "check_first_block.py")],
                ROOT / "backend_parity.log",
                ROOT,
            )
        result = train(manifest, 0, "smoke")
        (ROOT / "smoke_passed.json").write_text(
            json.dumps(
                dict(
                    status="passed",
                    time=time.time(),
                    manifest_sha256=sha(ROOT / "manifest.json"),
                    task=result["task"],
                    checks="All optimizer parameters covered; every extra readout updated; four epochs; stage two; export/reload; finite-difference forces",
                ),
                indent=2,
            )
            + "\n"
        )
    elif mode == "train":
        train(manifest, int(sys.argv[2]), mode)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
