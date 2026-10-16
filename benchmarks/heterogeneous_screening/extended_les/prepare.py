"""Prepare the seed-17 extended-LES comparison without modifying existing runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import ase.io
import les
import mace
import torch
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--stage-two", type=int, default=23)
parser.add_argument(
    "--cueq",
    action="store_true",
    help="Accelerate the multipole and response fits after check_cueq.py passes",
)
ARGS = parser.parse_args()
ROOT = ARGS.root.resolve()
HERE = Path(__file__).resolve().parent
OLD = Path("/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs/hetscreen_v2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    assert 0 < ARGS.stage_two < ARGS.epochs
    assert not (ROOT / "manifest.json").exists(), "Campaign already prepared"
    for module in (mace, les):
        assert str(ROOT / "venv") in module.__file__, module.__file__
    for name in ("data", "configs", "runs", "smoke", "logs", "results", "provenance"):
        (ROOT / name).mkdir(exist_ok=True)
    inputs = {}
    paths = {
        "train.xyz": OLD / "data/hetscreen_train.xyz",
        "valid.xyz": OLD / "data/hetscreen_valid.xyz",
        "test.xyz": OLD / "data/hetscreen_test.xyz",
        "probe.xyz": OLD / "probe_test20/probeiii_test_configs.xyz",
        "probe_reference.json": OLD / "probe_test20/probeiii_test_reference.json",
    }
    for dest, source in paths.items():
        target = ROOT / "data" / dest
        shutil.copy2(source, target)
        inputs[str(target.relative_to(ROOT))] = {
            "source": str(source),
            "sha256": sha(target),
        }
        assert sha(source) == sha(target)

    train = ase.io.read(ROOT / "data/train.xyz", ":")
    valid = ase.io.read(ROOT / "data/valid.xyz", ":")
    isolated = [a for a in train if len(a) == 1]
    train_bulk = [a for a in train if len(a) > 1]
    assert len(train_bulk) == 660 and len(valid) == 240 and len(isolated) == 3
    ase.io.write(ROOT / "data/smoke_train.xyz", isolated + train_bulk[:8])
    ase.io.write(ROOT / "data/smoke_valid.xyz", valid[:4])
    for filename in ("smoke_train.xyz", "smoke_valid.xyz"):
        inputs[f"data/{filename}"] = {
            "source": "training/validation subset only",
            "sha256": sha(ROOT / "data" / filename),
        }

    baseline_path = OLD / "maceles1200_seed17/config.yaml"
    shutil.copy2(baseline_path, ROOT / "provenance/original_les_config.yaml")
    baseline = yaml.safe_load(baseline_path.read_text())
    example = (
        ROOT
        / "upstream/extended_les_fit/MLIPs/MACE-LES/water/maceles-uQiqiu-r-4.5-nl-1/les.yaml"
    )
    shutil.copy2(example, ROOT / "provenance/authors_full_response.yaml")
    full = yaml.safe_load(example.read_text())
    common = dict(
        sigma=1.0,
        dl=2.0,
        remove_self_interaction=True,
        use_atomwise=False,
        is_periodic=None,
        output_scale=0.1,
        kappa_scale=0.01,
        alpha_scale=0.01,
        make_kappa_positive=False,
        compute_bec=False,
    )
    variants = {
        "monopole": dict(
            use_dipole=False,
            use_quad=False,
            use_induced_charge=False,
            use_induced_dipole=False,
            use_anisotropic_polarizability=False,
        ),
        "multipoles": dict(
            use_dipole=True,
            use_quad=True,
            use_induced_charge=False,
            use_induced_dipole=False,
            use_anisotropic_polarizability=False,
        ),
        "response": full,
    }
    tasks = []
    for index, (variant, flags) in enumerate(variants.items()):
        name = f"les_{variant}_seed17"
        les_config = ROOT / "configs" / f"{name}_les.yaml"
        les_config.write_text(yaml.safe_dump(common | flags, sort_keys=False))
        task = dict(
            index=index,
            name=name,
            variant=variant,
            seed=17,
            les_arguments=str(les_config),
            flags=flags,
            training_backend="cuequivariance" if ARGS.cueq and index > 0 else "e3nn",
        )
        for mode in ("train", "smoke"):
            directory = ROOT / ("runs" if mode == "train" else "smoke") / name
            directory.mkdir(exist_ok=True)
            c = dict(baseline)
            c.update(
                name=name,
                work_dir=str(directory),
                log_dir=str(directory / "logs"),
                model_dir=str(directory / "models"),
                checkpoints_dir=str(directory / "checkpoints"),
                results_dir=str(directory / "results"),
                train_file=str(
                    ROOT
                    / "data"
                    / ("train.xyz" if mode == "train" else "smoke_train.xyz")
                ),
                valid_file=str(
                    ROOT
                    / "data"
                    / ("valid.xyz" if mode == "train" else "smoke_valid.xyz")
                ),
                les_arguments=str(les_config),
                restart_latest=False,
                save_cpu=True,
                max_num_epochs=ARGS.epochs,
                start_swa=ARGS.stage_two,
            )
            c.pop("test_file", None)
            if mode == "train" and task["training_backend"] == "cuequivariance":
                c["enable_cueq"] = True
            if mode == "smoke":
                c.update(max_num_epochs=4, start_swa=2, patience=50)
            config = ROOT / "configs" / f"{name}_{mode}.yaml"
            config.write_text(yaml.safe_dump(c, sort_keys=False))
            task[mode] = dict(
                config=str(config),
                directory=str(directory),
                model=str(directory / "models" / f"{name}_stagetwo.model"),
            )
        tasks.append(task)

    for filename in (
        "run.py",
        "evaluate.py",
        "job.slurm",
        "collect.py",
        "collect.slurm",
        "README.md",
        "prepare.py",
        "cueq_compat.py",
        "train_cueq.py",
        "check_cueq.py",
        "check_cueq.slurm",
    ):
        shutil.copy2(HERE / filename, ROOT / filename)
    for filename in ("eval_probe.py", "eval_probe2.py", "analyze_probe2.py"):
        shutil.copy2(HERE.parent / filename, ROOT / filename)
    packages = {
        name: importlib.metadata.version(name)
        for name in ("mace-torch", "les", "torch", "numpy", "e3nn", "ase")
    }
    env = dict(
        python=sys.executable,
        packages=packages,
        mace_path=mace.__file__,
        les_path=les.__file__,
        torch_path=torch.__file__,
    )
    (ROOT / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    with (ROOT / "requirements.freeze.txt").open("w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
    commits = {
        name: subprocess.check_output(
            ["git", "-C", str(ROOT / "upstream" / name), "rev-parse", "HEAD"], text=True
        ).strip()
        for name in ("mace", "les", "extended_les_fit")
    }
    sources = {
        str(p.relative_to(ROOT)): sha(p)
        for p in list(ROOT.glob("*.py"))
        + list(ROOT.glob("*.slurm"))
        + list((ROOT / "configs").glob("*.yaml"))
    }
    installed = {
        str(p): sha(p)
        for module in (mace, les)
        for p in Path(module.__file__).parent.rglob("*.py")
    }
    manifest = dict(
        root=str(ROOT),
        paper="https://arxiv.org/abs/2605.05746v2",
        upstream_commits=commits,
        environment=env,
        inputs=inputs,
        sources=sources,
        installed_sources=installed,
        tasks=tasks,
        selection="Validation-selected stage-two EMA model; no test or probe tuning",
        training=dict(
            n_train=660,
            n_valid=240,
            epochs=ARGS.epochs,
            stage_two_start=ARGS.stage_two,
            seed=17,
            baseline_config_sha256=sha(baseline_path),
        ),
        fit_scope="limited-epoch pilot; not a converged accuracy comparison",
        probe=dict(
            n_sets=40,
            n_trajectories=20,
            n_variants=8,
            displacement_A=0.05,
            slope="within-set centred, pooled through-origin least squares",
            null="two remote ionic swaps at fixed S per set",
        ),
    )
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(dict(tasks=len(tasks), commits=commits, environment=env), indent=2)
    )


if __name__ == "__main__":
    main()
