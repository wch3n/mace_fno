"""Prepare all manuscript water FNO fits with one spectral group."""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--manuscript-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    campaigns = args.runs_root.resolve()
    root = args.output.resolve()
    pilot = campaigns / "water_scan_groups1_seed29_20261009"
    convergence = campaigns / "physical_convergence_matched_20260930"
    base = campaigns / "water_scan_clean_e400_20260924T103613Z"
    pilot_manifest = json.loads((pilot / "manifest.json").read_text())
    for name, checksum in pilot_manifest["source_checksums"].items():
        assert digest(pilot / name) == checksum, name
    rows = list(csv.DictReader((args.manuscript_data / "test_per_seed.csv").open()))
    primary = [r for r in rows if r["opt_level"] in ("frozen", "joint")]
    assert len(primary) == 20
    conv = json.loads((convergence / "manifest.json").read_text())
    variants = [t for t in conv["tasks"] if t["system"] == "water_scan"]
    assert len(variants) == 15
    root.mkdir(parents=True, exist_ok=False)
    for directory in (
        "configs",
        "inputs",
        "references",
        "runs",
        "logs",
        "smoke",
        "analysis",
    ):
        (root / directory).mkdir()
    shutil.copytree(
        pilot / "source",
        root / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("common.py", "train.py"):
        shutil.copy2(pilot / name, root / name)
    for name in (
        "prepare.py",
        "run.py",
        "smoke.py",
        "collect.py",
        "job.slurm",
        "README.md",
    ):
        shutil.copy2(Path(__file__).parent / name, root / name)
    tasks, systems = [], {}
    for layer in (1, 2):
        ready = json.loads((base / "ready" / f"water_scan_nl{layer}.json").read_text())
        assert ready["state"] == "passed"
        systems[f"water_scan_nl{layer}"] = {
            "split_counts": [471, 52, 134],
            "input_checksums": {},
            "test_file": ready["test_file"],
            "test_cache": ready["test_cache"],
        }
        for field in ("test_file", "test_cache"):
            path = ready[field]
            assert digest(path) == ready[field + "_sha256"]
            systems[f"water_scan_nl{layer}"]["input_checksums"][path] = digest(path)

    def add(config, name, layer, role, variant="production", reuse=None):
        old = yaml.safe_load(config.read_text())
        assert old["spectral_groups"] == 4 and old["architecture"] == "nonlinear"
        assert old["test_file"] is None and old["test_cache"] is None
        assert old["seed"] in (17, 29, 41, 53, 67)
        assert old["steps"] == (
            200000 if layer == 1 and old["mace_training"] == "joint" else 120000
        )
        checkpoint = root / "runs" / name / "model.pt"
        new = dict(
            old,
            spectral_groups=1,
            checkpoint=str(checkpoint),
            last_checkpoint=str(checkpoint.with_name("model.last.pt")),
        )
        # G=4 optimizer states cannot initialize a fresh G=1 fit.
        new.pop("resume", None)
        allowed = {"spectral_groups", "checkpoint", "last_checkpoint", "resume"}
        changes = {
            key: {"before": old.get(key), "after": new.get(key)}
            for key in old.keys() | new.keys()
            if old.get(key) != new.get(key)
        }
        assert set(changes) <= allowed
        path = root / "configs" / (name + ".yaml")
        path.write_text(yaml.safe_dump(new, sort_keys=False))
        system = f"water_scan_nl{layer}"
        for field in (
            "mace_model",
            "train_file",
            "validation_file",
            "train_cache",
            "validation_cache",
        ):
            source = old[field]
            checks = systems[system]["input_checksums"]
            if source not in checks:
                checks[source] = digest(source)
        task = dict(
            index=len(tasks),
            name=name,
            system=system,
            role=role,
            variant=variant,
            seed=new["seed"],
            regime=new["mace_training"],
            mace_layers=layer,
            expected_steps=new["steps"],
            config=str(path),
            config_sha256=digest(path),
            checkpoint=str(checkpoint),
            reference_config=str(config),
            reference_config_sha256=digest(config),
            changes=changes,
            **{
                k: new[k]
                for k in ("grid", "modes", "channels", "fno_layers", "spectral_groups")
            },
        )
        shutil.copy2(config, root / "references" / (name + ".yaml"))
        if reuse:
            reuse_cfg = yaml.safe_load(
                (reuse.parents[1] / "configs" / (reuse.name + ".yaml")).read_text()
            )
            assert reuse_cfg["spectral_groups"] == 1
            for key in set(new) | set(reuse_cfg):
                if key not in allowed:
                    assert new.get(key) == reuse_cfg.get(key), (
                        key,
                        new.get(key),
                        reuse_cfg.get(key),
                    )
            assert (
                json.loads((reuse / "status.json").read_text())["state"] == "complete"
            )
            task["reuse_run"] = str(reuse)
            task["reuse_checkpoint_sha256"] = digest(reuse / "model.pt")
        tasks.append(task)

    for row in primary:
        layer, seed, regime = (
            int(row["mace_layers"]),
            int(row["seed"]),
            row["opt_level"],
        )
        reuse = (
            pilot / "runs" / "water_scan_nl1_frozen_groups1_seed29"
            if (layer, regime, seed) == (1, "frozen", 29)
            else None
        )
        add(
            campaigns / row["config_source"],
            f"water_scan_nl{layer}_{regime}_groups1_seed{seed}",
            layer,
            "accuracy",
            reuse=reuse,
        )
    for task in variants:
        add(
            Path(task["config"]),
            task["name"].replace("_seed", "_groups1_seed"),
            1,
            "convergence",
            task["variant"],
        )
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        source_campaign=str(pilot),
        spectral_groups=1,
        seeds=[17, 29, 41, 53, 67],
        tasks=tasks,
        systems=systems,
        package_versions=pilot_manifest["package_versions"],
        selection="minimum combined validation loss; raw energy; no fitted offset",
        scope="20 accuracy models (one reused) and 15 convergence variants",
        manuscript_updates="Export results first; replace manuscript values only after verification.",
    )
    manifest["source_checksums"] = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "preflight.json").write_text(
        json.dumps(
            {
                "state": "passed",
                "tasks": len(tasks),
                "new_fits": 34,
                "reused_models": 1,
                "inputs_verified": True,
                "configuration_changes_checked": True,
                "gpu_smoke_required": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "campaign": str(root),
                "tasks": len(tasks),
                "new_fits": 34,
                "reused_index": next(t["index"] for t in tasks if "reuse_run" in t),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
