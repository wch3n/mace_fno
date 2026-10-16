"""Matched synthetic LES reruns: change the first block, retain the protocol."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from verify import FIRST, ROOT, digest, dump, verify

SEEDS = (17, 29, 41, 53, 67)
DATASETS = ("kappa0p05", "kappa0p2", "kappa0p5", "hetscreen")


def prepare(root, runs_root, repo):
    import yaml
    from ase.io import read, write

    root, runs_root, repo = root.resolve(), runs_root.resolve(), repo.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "source",
        "data",
        "configs",
        "provenance",
        "logs",
        "runs",
        "smoke",
        "validation",
    ):
        (root / name).mkdir()
    for package in ("mace", "les"):
        source = Path(importlib.util.find_spec(package).origin).parent
        shutil.copytree(
            source,
            root / "source" / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for p in Path(__file__).parent.iterdir():
        if p.suffix in (".py", ".slurm", ".md"):
            shutil.copy2(p, root / p.name)
    for name in ("eval_probe.py", "eval_probe2.py", "analyze_probe2.py"):
        shutil.copy2(repo / "benchmarks/heterogeneous_screening" / name, root / name)
    shutil.copy2(
        repo / "benchmarks/les_backbone_audit_20261011.json",
        root / "provenance/backbone_audit.json",
    )
    shutil.copy2(
        runs_root / "yukawa_L32_kappa_production/analysis/maceles1200_metrics.json",
        root / "provenance/homogeneous_metrics.json",
    )
    shutil.copy2(
        runs_root / "hetscreen_v2/final_eval_20261006/test_metrics_mace.json",
        root / "provenance/heterogeneous_metrics.json",
    )
    shutil.copy2(
        runs_root
        / "hetscreen_v2/final_eval_20261009_probe20/probeiii_test_models_A.json",
        root / "provenance/probe_residual.json",
    )
    for source, dest in (
        ("probeiii_test_configs.xyz", "probe.xyz"),
        ("probeiii_test_reference.json", "probe_reference.json"),
    ):
        shutil.copy2(
            runs_root / "hetscreen_v2/probe_test20" / source, root / "data" / dest
        )
    # These are the existing charge-only LES defaults, now explicit for conversion.
    les = dict(
        use_atomwise=False,
        sigma=1.0,
        dl=2.0,
        remove_self_interaction=True,
        compute_bec=False,
    )
    (root / "configs/les.yaml").write_text(yaml.safe_dump(les, sort_keys=False))
    tasks, counts, origins = [], {}, {}
    for dataset in DATASETS:
        hetero = dataset == "hetscreen"
        base = runs_root / ("hetscreen_v2" if hetero else "yukawa_L32_kappa_production")
        directory = root / "data" / dataset
        directory.mkdir()
        counts[dataset] = {}
        for split in ("train", "valid", "test"):
            original = base / "data" / f"{dataset}_{split}.xyz"
            target = directory / f"{split}.xyz"
            shutil.copy2(original, target)
            assert digest(original) == digest(target)
            origins[str(target.relative_to(root))] = str(original)
            frames = read(target, ":")
            isolated = [
                a for a in frames if a.info.get("config_type") == "IsolatedAtom"
            ]
            normal = [a for a in frames if a.info.get("config_type") != "IsolatedAtom"]
            counts[dataset][split] = len(normal)
            if split in ("train", "valid"):
                write(
                    directory / f"smoke_{split}.xyz",
                    isolated + normal[: 8 if split == "train" else 4],
                )
        for seed in SEEDS:
            old = (
                base
                / ("" if hetero else dataset.replace("kappa", "kappa_"))
                / f"maceles1200_seed{seed}"
            )
            name = f"les_nonres_{dataset}_seed{seed}"
            config = yaml.safe_load((old / "config.yaml").read_text())
            assert config["seed"] == seed and config["model"] == "MACELES"
            shutil.copy2(old / "config.yaml", root / "provenance" / f"{name}.yaml")
            models = [
                p
                for p in (old / "models").glob("*_stagetwo.model")
                if "compiled" not in p.name
            ]
            (model,) = models
            control = root / "provenance" / f"{name}_residual.model"
            shutil.copy2(model, control)
            assert digest(model) == digest(control)
            run_dir = root / "runs" / name
            config.update(
                name=name,
                work_dir=str(run_dir),
                interaction_first=FIRST,
                enable_cueq=True,
                restart_latest=False,
                les_arguments=str(root / "configs/les.yaml"),
            )
            for key, child in (
                ("log_dir", "logs"),
                ("model_dir", "models"),
                ("checkpoints_dir", "checkpoints"),
                ("results_dir", "results"),
            ):
                config[key] = str(run_dir / child)
            for key, split in (
                ("train_file", "train"),
                ("valid_file", "valid"),
                ("test_file", "test"),
            ):
                if key in config:
                    config[key] = str(directory / f"{split}.xyz")
            config_path = root / "configs" / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            task = dict(
                index=len(tasks),
                name=name,
                dataset=dataset,
                seed=seed,
                config=str(config_path.relative_to(root)),
                run_dir=str(run_dir.relative_to(root)),
                old_directory=str(old),
                control_model=str(control.relative_to(root)),
                original_model=str(model),
                original_model_sha256=digest(model),
                expected_parameters=347776 if hetero else 278528,
                expected_control_parameters=200320 if hetero else 180224,
            )
            if seed == 17 and dataset in ("kappa0p05", "hetscreen"):
                task["smoke_configs"] = {}
                for backend in ("native", "cueq"):
                    smoke = dict(config)
                    smoke_dir = root / "smoke" / dataset / backend
                    smoke.update(
                        name=f"smoke_{dataset}_{backend}",
                        work_dir=str(smoke_dir),
                        train_file=str(directory / "smoke_train.xyz"),
                        valid_file=str(directory / "smoke_valid.xyz"),
                        max_num_epochs=2,
                        start_swa=1,
                        enable_cueq=backend == "cueq",
                    )
                    smoke.pop("test_file", None)
                    for key, child in (
                        ("log_dir", "logs"),
                        ("model_dir", "models"),
                        ("checkpoints_dir", "checkpoints"),
                        ("results_dir", "results"),
                    ):
                        smoke[key] = str(smoke_dir / child)
                    path = root / "configs" / f"smoke_{dataset}_{backend}.yaml"
                    path.write_text(yaml.safe_dump(smoke, sort_keys=False))
                    task["smoke_configs"][backend] = str(path.relative_to(root))
            tasks.append(task)
    packages = (
        "mace-torch",
        "torch",
        "e3nn",
        "numpy",
        "ase",
        "PyYAML",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
    )
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,
        purpose="Paired-seed residual versus nonresidual first-block comparison",
        counts=counts,
        original_data=origins,
        source="Unmodified installed MACE and LES snapshots",
        package_versions={p: importlib.metadata.version(p) for p in packages},
        checkpoint_selection="Native validation-selected stage-two export; never selected on test",
        protocol=dict(max_num_epochs=1200, start_swa=450, patience=50, batch_size=4),
        inputs_and_source_checksums={
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    )
    dump(root / "manifest.json", manifest)
    print(json.dumps(dict(root=str(root), tasks=len(tasks), counts=counts), indent=2))


def run(index, validation=False):
    manifest = verify()
    task = manifest["tasks"][index]
    output = ROOT / ("validation/" + task["dataset"] if validation else task["run_dir"])
    output.mkdir(parents=True, exist_ok=False)
    status = dict(
        state="running",
        stage="preflight",
        job_id=os.environ.get("SLURM_JOB_ID"),
        started=datetime.now(timezone.utc).isoformat(),
        task=task["name"],
        commands=[],
    )

    def command(stage, argv):
        status["stage"] = stage
        record = dict(stage=stage, argv=argv)
        status["commands"].append(record)
        dump(output / "status.json", status)
        with (output / f"{stage}.log").open("x") as stream:
            proc = subprocess.run(
                [sys.executable, "-B", "-u", *argv],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        record["exit_code"] = proc.returncode
        if proc.returncode:
            print(
                "\n".join((output / f"{stage}.log").read_text().splitlines()[-60:]),
                flush=True,
            )
            raise subprocess.CalledProcessError(proc.returncode, argv)
        print(stage, "passed", flush=True)

    try:
        import torch

        assert torch.cuda.is_available()
        status["gpu"] = torch.cuda.get_device_name()
        if validation:
            command(
                "native_smoke",
                [
                    "-m",
                    "mace.cli.run_train",
                    "--config",
                    task["smoke_configs"]["native"],
                ],
            )
            command("backend_parity", ["check_cueq.py", str(index)])
            command(
                "cueq_smoke",
                ["train_cueq.py", "--config", task["smoke_configs"]["cueq"]],
            )
            command("smoke_verify", ["verify.py", "--smoke", str(index)])
        else:
            for dataset in ("kappa0p05", "hetscreen"):
                checked = json.loads(
                    (ROOT / "validation" / dataset / "status.json").read_text()
                )
                assert checked["state"] == "complete"
            command("training", ["train_cueq.py", "--config", task["config"]])
            command("evaluation", ["evaluate.py", str(index)])
        status.update(state="complete", stage="complete")
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        dump(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    for mode in ("run", "validate"):
        p = sub.add_parser(mode)
        p.add_argument("index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.runs_root, args.repo)
    else:
        run(args.index, args.mode == "validate")
