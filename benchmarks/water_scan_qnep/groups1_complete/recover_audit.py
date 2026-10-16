"""Finish a timed-out sine sweep, retaining verified numerical/cosine reports."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("index", type=int, choices=range(5))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    sys.path.insert(0, str(root / "followup"))
    from followup import checked_result, command, digest, read, save

    manifest = read(root / "manifest.json")
    for relative, checksum in manifest["source_checksums"].items():
        assert digest(root / relative) == checksum, relative
    for relative, checksum in read(root / "followup/sha256.json").items():
        assert digest(root / "followup" / relative) == checksum, relative
    task = manifest["tasks"][args.index]
    assert task["role"] == "accuracy" and task["spectral_groups"] == 1
    result = checked_result(task)
    system = manifest["systems"][task["system"]]
    cache = system["test_cache"]
    assert digest(cache) == system["input_checksums"][cache]
    output = root / "audits" / task["name"]
    numerical = read(output / "numerical.json")
    cosine = read(output / "spectral_cosine.json")
    for report in (numerical, cosine):
        assert report["checkpoint"] == task["checkpoint"]
        assert report["sample_cache"] == cache
        assert len(report["sample_indices"]) == 32
    assert numerical["spectral_groups"] == 1
    assert all(v["passed"] for v in numerical["promised_exact_checks"].values())
    assert cosine["probe_phase"] == "cosine"
    assert cosine["relative_amplitudes"] == [0.0125, 0.025, 0.05, 0.1]
    assert len(cosine["runs"]) == 4
    for report, amplitude in zip(cosine["runs"], cosine["relative_amplitudes"]):
        assert report["relative_amplitude"] == amplitude
        assert report["samples"] == 32 and report["max_mode"] == 2
        assert report["sample_indices"] == cosine["sample_indices"]
        assert len(report["per_sample_response"]) == 32
    original = {
        name: digest(output / name)
        for name in ("numerical.json", "spectral_cosine.json")
    }
    assert not (output / "spectral_sine.json").exists(), (
        "Sine report already exists; inspect before retry"
    )
    assert not (output / "status.json").exists(), "Audit already has final status"
    if args.check_only:
        print("RECOVERY PREFLIGHT PASSED", task["name"])
        return
    job = os.environ["SLURM_JOB_ID"]
    provenance = dict(
        task=task["name"],
        job_id=job,
        reused_reports=original,
        checkpoint_sha256=result["selections"]["best_loss"]["checkpoint_sha256"],
        started=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(__file__),
    )
    save(output / f"recovery-{job}.json", provenance)
    command(
        [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "mace_fno.cli.audit_spectral",
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
            "--max-mode",
            "2",
            "--fit-shells",
            "3",
            "--probe-phase",
            "sine",
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
            output / "spectral_sine.json",
        ],
        output / f"spectral_sine.recovery-{job}.log",
    )
    sine = read(output / "spectral_sine.json")
    assert sine["probe_phase"] == "sine"
    for key in ("checkpoint", "sample_cache", "sample_indices", "relative_amplitudes"):
        assert sine[key] == cosine[key], key
    for name, checksum in original.items():
        assert digest(output / name) == checksum
    assert checked_result(task) == result
    save(
        output / "status.json",
        dict(
            state="complete",
            task=task,
            checkpoint_sha256=provenance["checkpoint_sha256"],
            recovery=provenance,
            finished=datetime.now(timezone.utc).isoformat(),
            job_id=job,
        ),
    )


if __name__ == "__main__":
    main()
