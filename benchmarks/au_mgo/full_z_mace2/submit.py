"""Queue checked fresh fits and conditional full-state checkpoint continuations."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    preflight = json.loads((ROOT / "preflight.json").read_text())
    assert preflight["state"] == "passed" and preflight["backbone_interactions"] == 2
    manifest = json.loads((ROOT / "manifest.json").read_text())
    record_path = ROOT / "submission.json"
    record = (
        json.loads(record_path.read_text()) if record_path.exists() else dict(jobs=[])
    )

    def launch(key, options, arguments):
        existing = next((j for j in record["jobs"] if j["key"] == key), None)
        if existing:
            return existing["job_id"]
        command = [
            "sbatch",
            "--parsable",
            f"--chdir={ROOT}",
            *options,
            str(ROOT / "job.slurm"),
            str(ROOT),
            *arguments,
        ]
        job = (
            subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
        )
        assert job.isdecimal(), job
        record["jobs"].append(
            dict(
                key=key,
                job_id=job,
                command=command,
                submitted_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        save(record_path, record)
        print(key, job, flush=True)
        return job

    smoke = launch(
        "smoke",
        [
            "--job-name=au-nl2-fullz-check",
            "--time=02:00:00",
            "--array=0-1",
            "--output=logs/smoke-%A_%a.out",
            "--error=logs/smoke-%A_%a.err",
        ],
        ["smoke"],
    )
    production = launch(
        "production",
        [
            "--job-name=au-nl2-full-z",
            "--array=0-9",
            f"--dependency=afterok:{smoke}",
            "--kill-on-invalid-dep=yes",
            "--output=logs/train-%A_%a.out",
            "--error=logs/train-%A_%a.err",
        ],
        ["train"],
    )
    for task in manifest["tasks"]:
        index = task["index"]
        parent = f"{production}_{index}"
        for attempt in (1, 2):
            key = f"resume{attempt}_task{index}"
            parent = launch(
                key,
                [
                    f"--job-name=au-nl2-{task['opt_level']}-s{task['seed']}-r{attempt}",
                    f"--dependency=afterany:{parent}",
                    "--kill-on-invalid-dep=yes",
                    f"--output=logs/{key}-%j.out",
                    f"--error=logs/{key}-%j.err",
                ],
                ["resume", str(index), parent],
            )


if __name__ == "__main__":
    main()
