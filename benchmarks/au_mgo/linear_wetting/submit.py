"""Submit ready linear-FNO wetting fits and individually dependent evaluations."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from campaign import save, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify(root)
    receipt = root / "submission.json"
    result = json.loads(receipt.read_text()) if receipt.exists() else {"jobs": []}
    known = {r["key"]: r["job_id"] for r in result["jobs"]}

    def submit(key, options, script, trailing):
        if key in known:
            return known[key]
        cmd = [
            "sbatch",
            "--parsable",
            f"--chdir={root}",
            f"--job-name=linear-wet-{key}",
            f"--output={root}/logs/{key}-%j.out",
            f"--error={root}/logs/{key}-%j.err",
            *options,
            str(root / script),
            str(root),
            *trailing,
        ]
        if args.dry_run:
            print(json.dumps({"key": key, "command": cmd}))
            return f"<{key}>"
        value = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        assert value.isdigit(), value
        result["jobs"].append(
            dict(
                key=key,
                job_id=value,
                command=cmd,
                submitted_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        save(receipt, result)
        known[key] = value
        print(key, value, flush=True)
        return value

    ready = submit(
        "ready",
        ["--gres=gpu:4", "--cpus-per-task=16", "--mem=128G"],
        "job.slurm",
        ["ready"],
    )
    jobs = [ready]
    for t in manifest["tasks"]:
        if not t["ready_at_preparation"]:
            jobs.append(
                submit(
                    f"task{t['index']}",
                    [
                        f"--dependency=afterok:{t['training_dependency']}",
                        "--kill-on-invalid-dep=yes",
                    ],
                    "job.slurm",
                    ["worker", str(t["index"])],
                )
            )
    submit(
        "collect",
        ["--dependency=afterok:" + ":".join(jobs), "--kill-on-invalid-dep=yes"],
        "collect.slurm",
        [],
    )


if __name__ == "__main__":
    main()
