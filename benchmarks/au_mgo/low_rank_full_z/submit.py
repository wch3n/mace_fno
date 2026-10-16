"""Submit the GPU smoke array and gate the production trials on its success."""

import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("smoke", "production"))
args = parser.parse_args()
root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
assert json.loads((root / "preflight.json").read_text())["state"] == "passed"
record = root / f"submission_{args.mode}.json"
if record.exists():
    raise FileExistsError(record)
cmd = ["sbatch", "--parsable"]
if args.mode == "smoke":
    cmd += [
        "--partition=debug-gpu",
        "--qos=debug-gpu",
        "--time=02:00:00",
        "--job-name=au-lowrank-smoke",
        f"--array=0-{len(manifest['smoke_tasks']) - 1}%2",
    ]
    mode = "smoke"
else:
    smoke = json.loads((root / "submission_smoke.json").read_text())["job_id"]
    cmd += [
        "--job-name=au-lowrank-full-z",
        f"--array=0-{len(manifest['tasks']) - 1}%2",
        f"--dependency=afterok:{smoke}",
        "--kill-on-invalid-dep=yes",
    ]
    mode = "train"
cmd += [
    f"--output=logs/{args.mode}-%A_%a.out",
    f"--error=logs/{args.mode}-%A_%a.err",
    "job.slurm",
    str(root),
    mode,
]
job = subprocess.check_output(cmd, cwd=root, text=True).strip().split(";")[0]
record.write_text(
    json.dumps(dict(job_id=job, command=cmd, cwd=str(root)), indent=2) + "\n"
)
print(job, flush=True)
