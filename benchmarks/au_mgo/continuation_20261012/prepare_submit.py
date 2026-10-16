#!/usr/bin/env python3
"""Freeze and submit the four checked Au2/MgO continuations."""

import argparse
import getpass
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGNS = Path.home() / "mace_fno_runs/_campaigns"
SPECS = [
    (
        "rank64_frozen",
        "au2_mgo_low_rank_full_z_seed17_20261009",
        2,
        "gpu",
        "12:00:00",
        "au-r64-frozen-s17-resume",
    ),
    (
        "rank64_joint",
        "au2_mgo_low_rank_full_z_seed17_20261009",
        3,
        "gpu",
        "12:00:00",
        "au-r64-joint-s17-resume",
    ),
    (
        "dense_joint53",
        "au2_mgo_full_zmix_nl1_5seed_20261006T070300Z",
        8,
        "ia",
        "48:00:00",
        "au-full-z-joint-s53",
    ),
    (
        "dense_joint67",
        "au2_mgo_full_zmix_nl1_5seed_20261006T070300Z",
        9,
        "ia",
        "48:00:00",
        "au-full-z-joint-s67",
    ),
]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare():
    plan_path = HERE / "submission_plan.json"
    assert not plan_path.exists(), "Preparation already exists."
    destinations = {CAMPAIGNS / s[1] / "resume_20261012" for s in SPECS}
    assert all(not p.exists() for p in destinations), "A destination already exists."
    tasks = []
    # Validate every stopped input before creating campaign-side files.
    for key, name, index, partition, walltime, jobname in SPECS:
        root = CAMPAIGNS / name
        task = json.loads((root / "manifest.json").read_text())["tasks"][index]
        report = json.loads((HERE / f"preflight_{key}.json").read_text())
        assert report["compatible"] and report["task"] == task["name"]
        assert report["target_steps"] == task["steps"] == 240000
        assert report["completed_steps"] < task["steps"]
        assert (
            digest(Path(task["checkpoint"]).with_name("model.last.pt"))
            == report["last_checkpoint_sha256"]
        )
        dest = root / "resume_20261012"
        cmd = [
            "sbatch",
            "--parsable",
            f"--partition={partition}",
            f"--qos={partition}",
            f"--time={walltime}",
            f"--job-name={jobname}",
            f"--chdir={root}",
            f"--output={dest}/{key}-%j.out",
            f"--error={dest}/{key}-%j.err",
            str(dest / "resume.slurm"),
            str(root),
            str(index),
            str(dest / "resume_run.py"),
        ]
        tasks.append(
            dict(
                key=key,
                task=task["name"],
                root=str(root),
                index=index,
                partition=partition,
                walltime=walltime,
                job_name=jobname,
                start_step=report["completed_steps"],
                target_steps=task["steps"],
                checkpoint_sha256=report["last_checkpoint_sha256"],
                command=cmd,
                timeout_followon_hours=12 if partition == "ia" else None,
            )
        )
    for dest in destinations:
        dest.mkdir()
        for filename in ["resume_run.py", "resume.slurm", "README.md"]:
            shutil.copy2(HERE / filename, dest / filename)
    for spec, record in zip(SPECS, tasks):
        shutil.copy2(
            HERE / f"preflight_{spec[0]}.json",
            Path(record["root"]) / "resume_20261012" / f"preflight_{spec[0]}.json",
        )
    plan = dict(
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,
        driver_sha256=digest(HERE / "resume_run.py"),
        slurm_sha256=digest(HERE / "resume.slurm"),
    )
    dump(plan_path, plan)
    for dest in destinations:
        dump(dest / "submission_plan.json", plan)
    print(json.dumps(plan, indent=2))


def submit():
    plan = json.loads((HERE / "submission_plan.json").read_text())
    record_path = HERE / "submission.json"
    record = (
        json.loads(record_path.read_text()) if record_path.exists() else dict(jobs=[])
    )
    active_names = set(
        subprocess.check_output(
            ["squeue", "-h", "-u", getpass.getuser(), "-o", "%j"], text=True
        ).splitlines()
    )
    done = {j["key"] for j in record["jobs"]}
    for t in plan["tasks"]:
        dest = Path(t["root"]) / "resume_20261012"
        assert digest(dest / "resume_run.py") == plan["driver_sha256"]
        assert digest(dest / "resume.slurm") == plan["slurm_sha256"]
        if t["key"] not in done:
            assert t["job_name"] not in active_names, f"Already queued: {t['job_name']}"

    def save():
        dump(record_path, record)
        for root in {t["root"] for t in plan["tasks"]}:
            dump(Path(root) / "resume_20261012/submission.json", record)

    def launch(key, cmd, **extra):
        found = next((j for j in record["jobs"] if j["key"] == key), None)
        if found:
            return found["job_id"]
        job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        assert job.isdecimal(), job
        record["jobs"].append(
            dict(
                key=key,
                job_id=job,
                command=cmd,
                submitted_utc=datetime.now(timezone.utc).isoformat(),
                **extra,
            )
        )
        save()
        print(key, job, flush=True)
        return job

    for t in plan["tasks"]:
        launch(
            t["key"],
            t["command"],
            start_step=t["start_step"],
            target_steps=t["target_steps"],
        )
    for t in plan["tasks"]:
        if t["timeout_followon_hours"] is None:
            continue
        previous = next(j["job_id"] for j in record["jobs"] if j["key"] == t["key"])
        cmd = list(t["command"])
        cmd = [
            "--time=12:00:00"
            if arg.startswith("--time=")
            else f"--job-name={t['job_name']}-finish"
            if arg.startswith("--job-name=")
            else arg
            for arg in cmd
        ]
        cmd[2:2] = [f"--dependency=afterany:{previous}", "--kill-on-invalid-dep=yes"]
        cmd.append(previous)
        launch(
            t["key"] + "_timeout_followon",
            cmd,
            parent_job_id=previous,
            condition="Train only if parent Slurm state is TIMEOUT; skip on COMPLETED; refuse other terminal states",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "submit"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else submit()
