"""Submit one GPU trial with an after-success dependency on the smoke check."""

import argparse
import subprocess
from pathlib import Path

from common import dump, load, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "production"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    assert load(root / "preflight.json")["state"] == "passed"
    marker = root / f"submission_{args.mode}.json"
    with marker.open("x") as stream:
        stream.write('{"state":"submitting"}\n')
    command = ["sbatch", "--parsable", "--job-name=water-g1-" + args.mode]
    if args.mode == "smoke":
        command += ["--partition=debug-gpu", "--qos=debug-gpu", "--time=00:30:00"]
        mode = "smoke"
    else:
        smoke = load(root / "submission_smoke.json")["job_id"]
        command += [
            "--partition=gpu",
            "--qos=gpu",
            f"--dependency=afterok:{smoke}",
            "--kill-on-invalid-dep=yes",
        ]
        mode = "train"
    command += [
        f"--output={root}/logs/{args.mode}-%j.out",
        f"--error={root}/logs/{args.mode}-%j.err",
        str(root / "job.slurm"),
        str(root),
        mode,
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    record = dict(
        state="submitted" if result.returncode == 0 else "failed",
        time=now(),
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )
    if result.returncode == 0:
        record["job_id"] = result.stdout.strip().split(";")[0]
    dump(marker, record)
    print(record, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
