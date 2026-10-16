"""Replace obsolete continuations only after all revised jobs are safely held in queue."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path.home() / "mace_fno_runs/_campaigns/au2_mgo_full_z_finalize_20261016"
    manifest = json.loads((root / "manifest.json").read_text())
    preflight = json.loads((root / "preflight/complete.json").read_text())
    assert preflight == dict(all_passed=True, tasks=15)
    record_path = root / "submission.json"
    assert not record_path.exists(), (
        "Submission already recorded; inspect it before retrying."
    )
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=[])

    def save():
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(record_path)

    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", "wchen", "-o", "%i|%T|%j"], text=True
    )
    active = {row.split("|")[0]: row.split("|")[1:] for row in queue.splitlines()}
    for job in manifest["obsolete_pending_jobs"]:
        assert active[job][0] == "PENDING" and active[job][1].startswith("au-")
    for job in manifest["stopping_jobs"]:
        assert active[job][0] == "RUNNING" and active[job][1].startswith("au-")
    save()
    common = ["sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes"]
    first = manifest["tasks"][0]
    arguments = common + [
        "--partition=debug-gpu",
        "--qos=debug-gpu",
        "--time=02:00:00",
        "--array=0-4%2",
        "--job-name=au-fullz-nl1-final-eval",
        "--dependency=afterany:" + ":".join(manifest["stopping_jobs"]),
        f"--output={root}/logs/mace1-%A_%a.out",
        f"--error={root}/logs/mace1-%A_%a.err",
        str(root / "job.slurm"),
        str(root),
        first["parent"],
    ]
    job = subprocess.check_output(arguments, text=True).strip().split(";")[0]
    record["jobs"].append(
        dict(
            job_id=job,
            indices=list(range(5)),
            kind="mace1_final_evaluation",
            command=arguments,
        )
    )
    save()
    for task in manifest["tasks"][5:]:
        arguments = common + [
            f"--job-name=au-nl2-{task['regime']}-s{task['seed']}-100k",
            f"--dependency=afterany:{task['previous_job']}",
            f"--output={root}/logs/{task['name']}-%j.out",
            f"--error={root}/logs/{task['name']}-%j.err",
            str(root / "job.slurm"),
            str(root),
            task["parent"],
            str(task["index"]),
        ]
        job = subprocess.check_output(arguments, text=True).strip().split(";")[0]
        record["jobs"].append(
            dict(
                job_id=job,
                indices=[task["index"]],
                kind="mace2_100k_and_evaluation",
                command=arguments,
            )
        )
        save()
    subprocess.run(["scancel", *manifest["obsolete_pending_jobs"]], check=True)
    record["cancelled_obsolete_continuations"] = manifest["obsolete_pending_jobs"]
    save()
    subprocess.run(["scancel", *manifest["stopping_jobs"]], check=True)
    record["stopped_mace1_training"] = manifest["stopping_jobs"]
    save()
    for entry in record["jobs"]:
        subprocess.run(["scontrol", "release", entry["job_id"]], check=True)
        entry["released"] = True
        save()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
