"""Complete requested numerical checks without any spectral field probes."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("index", type=int, choices=range(20))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    scripts = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "followup"))
    from followup import checked_result, command, digest, read, save

    for relative, expected in read(scripts / "sha256.json").items():
        assert digest(scripts / relative) == expected, relative
    manifest = read(root / "manifest.json")
    for relative, expected in manifest["source_checksums"].items():
        assert digest(root / relative) == expected, relative
    for relative, expected in read(root / "followup/sha256.json").items():
        assert digest(root / "followup" / relative) == expected, relative
    task = manifest["tasks"][args.index]
    assert task["role"] == "accuracy" and task["spectral_groups"] == 1
    result = checked_result(task)
    system = manifest["systems"][task["system"]]
    cache = system["test_cache"]
    assert digest(cache) == system["input_checksums"][cache]
    # Preserve the report location in this archived campaign.
    output = root / "audits" / task["name"]
    report = output / "numerical.json"
    if report.exists():
        numerical = read(report)
        assert numerical["checkpoint"] == task["checkpoint"]
        assert numerical["sample_cache"] == cache
        assert numerical["spectral_groups"] == 1 and numerical["samples"] == 32
        assert all(c["passed"] for c in numerical["promised_exact_checks"].values())
    if args.check_only:
        print(
            "NUMERICAL PREFLIGHT PASSED",
            task["name"],
            "existing" if report.exists() else "ready",
        )
        return
    output.mkdir(exist_ok=True, parents=True)
    previous = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    if not report.exists():
        job = os.environ["SLURM_JOB_ID"]
        # Execute the hashed script snapshot stored in the campaign.
        command(
            [
                sys.executable,
                "-B",
                "-u",
                scripts / "audit_3d.py",
                "--checkpoint",
                task["checkpoint"],
                "--sample-cache",
                cache,
                "--samples",
                "32",
                "--seed",
                "947",
                "--device",
                "cuda",
                "--fd-step",
                "1e-4",
                "--fd-components",
                "12",
                "--fd-batch-size",
                "2",
                "--strict",
                "--output",
                report,
            ],
            output / f"numerical-only-{job}.log",
        )
    numerical = read(report)
    assert (
        numerical["checkpoint"] == task["checkpoint"]
        and numerical["sample_cache"] == cache
    )
    assert numerical["spectral_groups"] == 1 and numerical["samples"] == 32
    assert len(numerical["sample_indices"]) == 32
    assert all(c["passed"] for c in numerical["promised_exact_checks"].values())
    assert checked_result(task) == result
    for name, expected in previous.items():
        assert digest(output / name) == expected, name
    save(
        output / "numerical_status.json",
        dict(
            state="complete",
            requested_checks=["numerical"],
            spectral_checks="not_requested",
            task=task,
            checkpoint_sha256=result["selections"]["best_loss"]["checkpoint_sha256"],
            report_sha256=digest(report),
            script_sha256=digest(__file__),
            finished=datetime.now(timezone.utc).isoformat(),
            job_id=os.getenv("SLURM_JOB_ID"),
        ),
    )
    print("NUMERICAL CHECKS COMPLETE", task["name"])


if __name__ == "__main__":
    main()
