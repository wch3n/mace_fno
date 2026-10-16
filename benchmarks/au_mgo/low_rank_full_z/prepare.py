"""Snapshot matched rank-32/64 slab trials from an existing dense campaign."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    args = parser.parse_args()
    parent = args.dense_campaign.resolve()
    root = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[3]
    previous = json.loads((parent / "manifest.json").read_text())
    templates = {}
    for opt in ("frozen", "joint"):
        task = next(
            t for t in previous["tasks"] if t["opt_level"] == opt and t["seed"] == 17
        )
        path = Path(task["config"])
        assert digest(path) == previous["source_hashes"][str(path.relative_to(parent))]
        cfg = yaml.safe_load(path.read_text())
        assert cfg["architecture"] == "nonlinear" and cfg["z_mixing"] == "full"
        assert cfg["spectral_symmetry"] == "metric_eqgino"
        assert cfg["metric_parameterization"] == "shell_spline"
        assert cfg["steps"] == 240000 and cfg["spectral_groups"] == 1
        assert (cfg["fno_hidden_channels"], cfg["z_grid"], cfg["modes"]) == (16, 24, 8)
        templates[opt] = (cfg, path)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("configs", "references", "models", "data", "logs", "runs", "smoke"):
        (root / folder).mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for folder in ("src", "tests"):
        shutil.copytree(repo / folder, root / "source" / folder, ignore=ignore)
    shutil.copytree(parent / "mace_source", root / "mace_source", ignore=ignore)
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(repo / name, root / "source" / name)
    for name in ("train.py", "evaluate.py", "measure_train.py", "run.py"):
        source = parent / name
        assert digest(source) == previous["source_hashes"][name]
        shutil.copy2(source, root / name)
    runner = root / "run.py"
    text = runner.read_text()
    before = "        assert cp['completed_steps']==task['steps']\n"
    assert text.count(before) == 1
    runner.write_text(
        text.replace(
            before, before + "        assert cp['spectral_rank']==task['rank']\n"
        )
    )
    for source in Path(__file__).parent.iterdir():
        if source.is_file():
            shutil.copy2(source, root / source.name)
    patch = subprocess.check_output(["git", "diff"], cwd=repo, text=True)
    (root / "references" / "working_tree.patch").write_text(patch)

    inputs = {}
    for relative, info in previous["inputs"].items():
        source, target = parent / relative, root / relative
        assert digest(source) == info["sha256"], relative
        shutil.copy2(source, target)
        inputs[relative] = dict(info, source=str(source))
    for split in ("train", "validation"):
        relative = f"data/smoke_{split}.xyz"
        shutil.copy2(parent / relative, root / relative)
        inputs[relative] = dict(
            source=str(parent / relative), sha256=digest(root / relative)
        )

    tasks, smoke = [], []
    for rank in (32, 64):
        for opt in ("frozen", "joint"):
            original, source_config = templates[opt]
            reference = root / "references" / source_config.name
            if not reference.exists():
                shutil.copy2(source_config, reference)
            for seed in args.seeds:
                name = f"au2_mgo_nl1_{opt}_full_z_rank{rank}_seed{seed}"
                cfg = dict(original, spectral_rank=rank, seed=seed)
                for key in ("mace_model", "train_file", "validation_file"):
                    cfg[key] = str(root / Path(original[key]).relative_to(parent))
                checkpoint = root / "runs" / name / "model.pt"
                cfg.update(
                    checkpoint=str(checkpoint),
                    last_checkpoint=str(checkpoint.with_name("model.last.pt")),
                )
                path = root / "configs" / f"{name}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                changes = {
                    key: dict(before=original.get(key), after=value)
                    for key, value in cfg.items()
                    if original.get(key) != value
                }
                assert set(changes) <= {
                    "spectral_rank",
                    "seed",
                    "mace_model",
                    "train_file",
                    "validation_file",
                    "checkpoint",
                    "last_checkpoint",
                }
                tasks.append(
                    dict(
                        index=len(tasks),
                        name=name,
                        rank=rank,
                        opt_level=opt,
                        seed=seed,
                        steps=cfg["steps"],
                        config=str(path),
                        checkpoint=str(checkpoint),
                        test_file=str(root / "data/test.xyz"),
                        source_config=str(source_config),
                        changes=changes,
                    )
                )
            name = f"au2_mgo_nl1_{opt}_full_z_rank{rank}_smoke"
            checkpoint = root / "smoke" / name / "model.pt"
            smoke_cfg = dict(
                cfg,
                seed=17,
                steps=12,
                eval_interval=6,
                checkpoint_interval=6,
                mace_warmup_steps=2 if opt == "joint" else 0,
                train_file=str(root / "data/smoke_train.xyz"),
                validation_file=str(root / "data/smoke_validation.xyz"),
                checkpoint=str(checkpoint),
                last_checkpoint=str(checkpoint.with_name("model.last.pt")),
            )
            path = root / "configs" / f"{name}.yaml"
            path.write_text(yaml.safe_dump(smoke_cfg, sort_keys=False))
            smoke.append(
                dict(
                    index=len(smoke),
                    name=name,
                    rank=rank,
                    opt_level=opt,
                    seed=17,
                    steps=12,
                    config=str(path),
                    checkpoint=str(checkpoint),
                    test_file=str(root / "data/smoke_validation.xyz"),
                )
            )
    manifest = dict(
        root=str(root),
        parent=str(parent),
        tasks=tasks,
        smoke_tasks=smoke,
        inputs=inputs,
        independent_variable="Wavelength-dependent factor rank (32 or 64) of full channel-height mixing.",
        interpolation="Natural cubic splines of each factor separately; product is rank-limited at every query.",
        initialization="Random factors match dense matrix entry variance at shell knots; final projection remains zero.",
        comparison="First-stage seed-17 frozen/joint trials; same data, backbone, mesh, optimizer and step budget as dense campaign.",
        selection="Minimum validation loss; held-out test evaluated only after training, no energy shifts.",
        spectral_parameters_per_layer={
            "dense": 5013504,
            "rank32": 835584,
            "rank64": 1671168,
        },
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
    )
    manifest["source_hashes"] = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".yaml", ".slurm", ".md", ".toml", ".patch")
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root)
    print(f"Prepared {len(tasks)} production trials and {len(smoke)} smoke tests.")


if __name__ == "__main__":
    main()
