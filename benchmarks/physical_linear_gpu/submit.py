"""Queue the two production arrays and retire their held ia predecessors."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--retire-parent", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    assert json.loads((root / "preflight.json").read_text())["state"] == "passed"
    assert json.loads((root / "smoke_passed.json").read_text())["state"] == "passed"
    smoke = json.loads((root / "submission_smoke.json").read_text())["job_id"]
    if not args.retire_parent:
        for system, indices, limit, name in [
            ("water", "0-19", 4, "water-linear-g1"),
            ("au2_mgo", "20-28", 2, "au-linear-gpu"),
        ]:
            path = root / f"submission_{system}.json"
            if path.exists():
                print(
                    "Already submitted", system, json.loads(path.read_text())["job_id"]
                )
                continue
            cmd = [
                "sbatch",
                "--parsable",
                f"--job-name={name}",
                f"--array={indices}%{limit}",
                f"--dependency=afterok:{smoke}",
                "--kill-on-invalid-dep=yes",
                "--output=" + str(root / "logs/train-%A_%a.out"),
                "--error=" + str(root / "logs/train-%A_%a.err"),
                str(root / "job.slurm"),
                str(root),
                "train",
            ]
            result = subprocess.run(cmd, text=True, capture_output=True, check=True)
            record = dict(
                job_id=result.stdout.strip().split(";")[0],
                command=cmd,
                submitted_utc=datetime.now(timezone.utc).isoformat(),
            )
            path.write_text(json.dumps(record, indent=2) + "\n")
            print(system, record["job_id"])
        return
    replacement = json.loads((root / "submission_au2_mgo.json").read_text())["job_id"]
    oldroot = Path(manifest["parent"])
    receipt_path = oldroot / "gpu_migration_20261011.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["state"] == "held_for_preparation"
    mapping = []
    for task in manifest["tasks"]:
        if not task["replaces_job"]:
            continue
        newjob = f"{replacement}_{task['index']}"
        oldjob = task["replaces_job"]
        fresh = subprocess.check_output(
            ["scontrol", "show", "job", "-o", newjob], text=True
        )
        assert "Partition=gpu " in fresh and (
            "JobState=PENDING" in fresh or "JobState=RUNNING" in fresh
        ), fresh
        previous = subprocess.check_output(
            ["scontrol", "show", "job", "-o", oldjob], text=True
        )
        assert "JobState=PENDING" in previous and "Reason=JobHeldUser" in previous, (
            previous
        )
        subprocess.run(["scancel", oldjob], check=True)
        mapping.append(
            dict(previous_job=oldjob, replacement_job=newjob, name=task["name"])
        )
        receipt.update(replacements=mapping, replacement_campaign=str(root))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    assert len(mapping) == 9
    receipt.update(
        state="moved_to_gpu", finished_utc=datetime.now(timezone.utc).isoformat()
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    (root / "migration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Retired all nine held ia jobs; replacements are queued/running on gpu.")


if __name__ == "__main__":
    main()
