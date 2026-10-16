"""Prepare five fresh full-response fits from the immutable pilot campaign."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 41, 53, 67)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--stage-two", type=int, default=450)
    args = parser.parse_args()
    root, pilot = args.root.resolve(), args.pilot.resolve()
    assert 0 < args.stage_two < args.epochs
    assert not root.exists(), root
    old = json.loads((pilot / "manifest.json").read_text())
    response = old["tasks"][2]
    baseline = yaml.safe_load(Path(response["train"]["config"]).read_text())
    assert (
        json.loads((Path(response["train"]["directory"]) / "status.json").read_text())[
            "status"
        ]
        == "completed"
    )
    for key, digest in old["sources"].items():
        assert sha(pilot / key) == digest, key
    root.mkdir()
    for name in ("configs", "runs", "smoke", "logs", "results", "provenance"):
        (root / name).mkdir()
    (root / "data").symlink_to(pilot / "data", target_is_directory=True)
    (root / "venv").symlink_to(pilot / "venv", target_is_directory=True)
    shutil.copy2(pilot / "manifest.json", root / "provenance/pilot_manifest.json")
    shutil.copy2(pilot / "results/summary.json", root / "provenance/pilot_summary.json")
    certificate = pilot / "cueq_validation/les_response_seed17/passed.json"
    shutil.copy2(certificate, root / "provenance/cueq_conversion_passed.json")
    les_path = root / "configs/les_response.yaml"
    shutil.copy2(response["les_arguments"], les_path)
    tasks = []
    for index, seed in enumerate(SEEDS):
        name = f"les_response_seed{seed}"
        task = dict(
            index=index,
            seed=seed,
            name=name,
            variant="response",
            les_arguments=str(les_path),
            flags=copy.deepcopy(response["flags"]),
            training_backend="cuequivariance",
        )
        for mode in ("train", "smoke") if index == 0 else ("train",):
            directory = root / ("runs" if mode == "train" else "smoke") / name
            directory.mkdir()
            c = copy.deepcopy(baseline)
            c.update(
                name=name,
                seed=seed,
                max_num_epochs=args.epochs,
                start_swa=args.stage_two,
                work_dir=str(directory),
                log_dir=str(directory / "logs"),
                model_dir=str(directory / "models"),
                checkpoints_dir=str(directory / "checkpoints"),
                results_dir=str(directory / "results"),
                les_arguments=str(les_path),
                train_file=str(root / "data/train.xyz"),
                valid_file=str(root / "data/valid.xyz"),
                restart_latest=False,
                enable_cueq=True,
            )
            if mode == "smoke":
                c.update(
                    max_num_epochs=4,
                    start_swa=2,
                    train_file=str(root / "data/smoke_train.xyz"),
                    valid_file=str(root / "data/smoke_valid.xyz"),
                )
            config = root / "configs" / f"{name}_{mode}.yaml"
            config.write_text(yaml.safe_dump(c, sort_keys=False))
            task[mode] = dict(
                directory=str(directory),
                config=str(config),
                model=str(directory / "models" / f"{name}_stagetwo.model"),
            )
        tasks.append(task)
    for filename in (
        "cueq_compat.py",
        "train_cueq.py",
        "check_cueq.py",
        "optimizer_compat.py",
        "train_response.py",
        "prepare_response.py",
        "run_response.py",
        "response.slurm",
        "collect.py",
        "collect.slurm",
        "evaluate.py",
        "README_response.md",
    ):
        shutil.copy2(HERE / filename, root / filename)
    for filename in ("eval_probe.py", "eval_probe2.py", "analyze_probe2.py"):
        shutil.copy2(pilot / filename, root / filename)
    certified = json.loads(certificate.read_text())
    for name, digest in certified["source_hashes"].items():
        assert sha(root / name) == digest, name
    assert sha(les_path) == certified["les_config_sha256"]
    sources = {
        str(p.relative_to(root)): sha(p)
        for p in [
            *root.glob("*.py"),
            *root.glob("*.slurm"),
            *(root / "configs").glob("*.yaml"),
        ]
    }
    manifest = dict(
        root=str(root),
        pilot=str(pilot),
        tasks=tasks,
        seeds=list(SEEDS),
        inputs=copy.deepcopy(old["inputs"]),
        sources=sources,
        installed_sources=copy.deepcopy(old["installed_sources"]),
        environment=copy.deepcopy(old["environment"]),
        upstream_commits=copy.deepcopy(old["upstream_commits"]),
        probe=copy.deepcopy(old["probe"]),
        paper=old["paper"],
        training=dict(
            n_train=660,
            n_valid=240,
            epochs=args.epochs,
            stage_two_start=args.stage_two,
            validation_patience=baseline["patience"],
            initialization="fresh for every seed; no pilot warm start",
        ),
        selection=old["selection"],
        fit_scope="full training; validation-loss plateau required before claiming convergence",
        optimizer_correction="All trainable parameters included exactly once; original optimizer omitted 98944 response/multipole readout parameters",
        conversion_certificate=dict(
            path="provenance/cueq_conversion_passed.json", sha256=sha(certificate)
        ),
    )
    for key, record in manifest["inputs"].items():
        assert sha(root / key) == record["sha256"], key
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(root=str(root), seeds=list(SEEDS), training=manifest["training"]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
