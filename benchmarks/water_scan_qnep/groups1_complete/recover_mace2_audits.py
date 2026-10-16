"""Recover MACE2 audits using smaller finite-difference energy batches."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("index", type=int, choices=range(10, 20))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    scripts = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "followup"))
    from followup import checked_result, command, digest, read, save

    for relative, checksum in read(scripts / "sha256.json").items():
        assert digest(scripts / relative) == checksum, relative
    manifest = read(root / "manifest.json")
    for relative, checksum in manifest["source_checksums"].items():
        assert digest(root / relative) == checksum, relative
    for relative, checksum in read(root / "followup/sha256.json").items():
        assert digest(root / "followup" / relative) == checksum, relative
    task = manifest["tasks"][args.index]
    assert task["role"] == "accuracy" and task["mace_layers"] == 2
    assert task["spectral_groups"] == 1
    result = checked_result(task)
    system = manifest["systems"][task["system"]]
    cache = system["test_cache"]
    assert digest(cache) == system["input_checksums"][cache]
    output = root / "audits" / task["name"]
    for name in (
        "status.json",
        "numerical.json",
        "spectral_cosine.json",
        "spectral_sine.json",
    ):
        assert not (output / name).exists(), (
            f"Existing report requires inspection: {output / name}"
        )
    if args.check_only:
        print("RECOVERY PREFLIGHT PASSED", task["name"])
        return

    output.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_JOB_ID"]
    provenance = dict(
        task=task["name"],
        job_id=job,
        fd_batch_size=2,
        checkpoint_sha256=result["selections"]["best_loss"]["checkpoint_sha256"],
        started=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(__file__),
        audit_sha256=digest(scripts / "audit_3d.py"),
        original_logs={p.name: digest(p) for p in output.glob("*.log")},
    )
    save(output / f"memory-recovery-{job}.json", provenance)
    common = [
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
    ]
    command(
        [
            sys.executable,
            "-B",
            "-u",
            scripts / "audit_3d.py",
            *common,
            "--fd-step",
            "1e-4",
            "--fd-components",
            "12",
            "--fd-batch-size",
            "2",
            "--strict",
            "--output",
            output / "numerical.json",
        ],
        output / f"numerical.memory-recovery-{job}.log",
    )
    numerical = read(output / "numerical.json")
    assert numerical["spectral_groups"] == 1 and numerical["fd_batch_size"] == 2
    assert all(c["passed"] for c in numerical["promised_exact_checks"].values())
    for phase in ("cosine", "sine"):
        command(
            [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "mace_fno.cli.audit_spectral",
                *common,
                "--max-mode",
                "2",
                "--fit-shells",
                "3",
                "--probe-phase",
                phase,
                "--relative-amplitudes",
                "0.0125",
                "0.025",
                "0.05",
                "0.1",
                "--field-batch-size",
                "32",
                "--relative-span-tolerance",
                "0.05",
                "--output",
                output / f"spectral_{phase}.json",
            ],
            output / f"spectral_{phase}.memory-recovery-{job}.log",
        )
        spectral = read(output / f"spectral_{phase}.json")
        assert spectral["probe_phase"] == phase
        assert spectral["relative_amplitudes"] == [0.0125, 0.025, 0.05, 0.1]
        assert len(spectral["runs"]) == 4
        for entry in spectral["runs"]:
            assert entry["samples"] == 32 and len(entry["per_sample_response"]) == 32
        assert spectral["sample_indices"] == numerical["sample_indices"]
        for report in (numerical, spectral):
            assert report["checkpoint"] == task["checkpoint"]
            assert report["sample_cache"] == cache
            assert len(report["sample_indices"]) == 32
    for name, checksum in provenance["original_logs"].items():
        assert digest(output / name) == checksum, name
    assert checked_result(task) == result
    save(
        output / "status.json",
        dict(
            state="complete",
            task=task,
            recovery=provenance,
            job_id=job,
            checkpoint_sha256=provenance["checkpoint_sha256"],
            finished=datetime.now(timezone.utc).isoformat(),
        ),
    )


if __name__ == "__main__":
    main()
