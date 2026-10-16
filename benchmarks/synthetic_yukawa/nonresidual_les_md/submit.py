"""Submit two GPU checks, two gated training/MD jobs and the Fig. 4 update."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(root):
    root = root.resolve()
    record_path = root / "submission.json"
    assert not record_path.exists(), (
        "Submission already recorded; inspect existing jobs before resubmitting."
    )
    record = dict(
        submitted=datetime.now(timezone.utc).isoformat(), root=str(root), jobs={}
    )

    def submit(label, options, script, *args):
        cmd = ["sbatch", "--parsable", *options, str(root / script), str(root), *args]
        proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True)
        job = proc.stdout.strip().split(";")[0]
        assert job.isdigit(), proc.stdout
        record["jobs"][label] = dict(id=job, command=cmd, stderr=proc.stderr)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(label, job, flush=True)
        return job

    check = submit(
        "checks",
        [
            "--job-name=les-nonres-md-check",
            "--array=0-1",
            "--time=00:30:00",
            "--output=logs/check-%A_%a.out",
            "--error=logs/check-%A_%a.err",
        ],
        "job.slurm",
        "check",
    )
    production = submit(
        "training_and_md",
        [
            "--job-name=les-nonres-md",
            "--array=0-1",
            f"--dependency=afterok:{check}",
            "--kill-on-invalid-dep=yes",
            "--output=logs/production-%A_%a.out",
            "--error=logs/production-%A_%a.err",
        ],
        "job.slurm",
        "run",
    )
    submit(
        "figure_update",
        [
            f"--dependency=afterok:{production}",
            "--kill-on-invalid-dep=yes",
            "--output=logs/figure-%j.out",
            "--error=logs/figure-%j.err",
        ],
        "update_figure.slurm",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
