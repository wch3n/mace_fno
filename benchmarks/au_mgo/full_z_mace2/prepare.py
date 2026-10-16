"""Create the matched MACE2 dense full-z campaign from frozen reference inputs."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from ase.io import read, write

REPO = Path("/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mace_fno")
PARENT = Path(
    "/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs/_campaigns/physical_linear_5seed_20261005"
)
DENSE = Path(
    "/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs/_campaigns/au2_mgo_full_zmix_nl1_5seed_20261006T070300Z"
)
SEEDS = (17, 29, 41, 53, 67)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(exist_ok=False)
    for folder in ("configs", "references", "models", "data", "logs", "runs", "smoke"):
        (root / folder).mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    dense_manifest = json.loads((DENSE / "manifest.json").read_text())
    for relative, checksum in dense_manifest["source_hashes"].items():
        assert digest(DENSE / relative) == checksum, relative
    shutil.copytree(DENSE / "source", root / "source", ignore=ignore)
    shutil.copytree(DENSE / "mace_source", root / "mace_source", ignore=ignore)
    for name in ("train.py", "evaluate.py", "measure_train.py"):
        shutil.copy2(DENSE / name, root / name)
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in (".py", ".slurm", ".md"):
            shutil.copy2(path, root / path.name)
    shutil.copy2(DENSE / "manifest.json", root / "references/mace1_dense_manifest.json")
    parent = json.loads((PARENT / "manifest.json").read_text())
    inputs = {}
    for split, count in (("train", 4275), ("validation", 225), ("test", 500)):
        src = PARENT / f"data/au2_mgo/{split}.xyz"
        checksum = digest(src)
        assert checksum == parent["inputs"][str(src.relative_to(PARENT))]["sha256"]
        dst = root / f"data/{split}.xyz"
        shutil.copy2(src, dst)
        frames = read(dst, index=":")
        assert len(frames) == count
        assert {len(frame) for frame in frames} == {110}
        inputs[str(dst.relative_to(root))] = dict(
            source=str(src), sha256=checksum, structures=count, atoms=110
        )
        if split != "test":
            write(root / f"data/smoke_{split}.xyz", frames[:8], format="extxyz")
            relative = f"data/smoke_{split}.xyz"
            inputs[relative] = dict(
                source=f"First eight {split} structures",
                sha256=digest(root / relative),
                structures=8,
            )
    src = PARENT / "models/au2_mgo_nl2.model"
    checksum = digest(src)
    assert checksum == parent["inputs"][str(src.relative_to(PARENT))]["sha256"]
    model = root / "models/mace_nl2.model"
    shutil.copy2(src, model)
    inputs[str(model.relative_to(root))] = dict(source=str(src), sha256=checksum)
    tasks, smoke = [], []
    for opt in ("frozen", "joint"):
        ref = PARENT / f"references/au2_mgo_nl2_{opt}.yaml"
        assert digest(ref) == parent["source_hashes"][str(ref.relative_to(PARENT))]
        original = yaml.safe_load(ref.read_text())
        shutil.copy2(ref, root / "references" / ref.name)
        assert original["architecture"] == "nonlinear" and original["fno_layers"] == 4
        assert original["steps"] == 240000 and original["spectral_groups"] == 1
        cfg = dict(original)
        cfg.update(
            mace_model=str(model),
            train_file=str(root / "data/train.xyz"),
            validation_file=str(root / "data/validation.xyz"),
            validation_indices_file=None,
            test_file=None,
            train_cache=None,
            validation_cache=None,
            test_cache=None,
            z_mixing="full",
            batch_size=1,
            accumulation_steps=4,
            evaluation_batch_size=1,
            resume=None,
            init_from=None,
            checkpoint_interval=500,
            early_stopping_patience_steps=0,
        )
        assert (
            original["batch_size"] * original["accumulation_steps"]
            == cfg["batch_size"] * cfg["accumulation_steps"]
            == 4
        )
        for seed in SEEDS:
            name = f"au2_mgo_nl2_{opt}_full_z_seed{seed}"
            checkpoint = root / "runs" / name / "model.pt"
            config = dict(
                cfg,
                seed=seed,
                checkpoint=str(checkpoint),
                last_checkpoint=str(checkpoint.with_name("model.last.pt")),
            )
            path = root / "configs" / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            tasks.append(
                dict(
                    index=len(tasks),
                    name=name,
                    system="au2_mgo",
                    depth=2,
                    opt_level=opt,
                    seed=seed,
                    steps=cfg["steps"],
                    config=str(path),
                    checkpoint=str(checkpoint),
                    test_file=str(root / "data/test.xyz"),
                    source_config=str(ref),
                    source_sha256=digest(ref),
                    changes={
                        key: dict(before=original.get(key), after=value)
                        for key, value in config.items()
                        if original.get(key) != value
                    },
                )
            )
        name = f"au2_mgo_nl2_{opt}_full_z_smoke"
        checkpoint = root / "smoke" / name / "model.pt"
        config = dict(
            cfg,
            seed=17,
            steps=12,
            eval_interval=6,
            checkpoint_interval=6,
            mace_warmup_steps=2 if opt == "joint" else 0,
            train_file=str(root / "data/train.xyz"),
            validation_file=str(root / "data/validation.xyz"),
            checkpoint=str(checkpoint),
            last_checkpoint=str(checkpoint.with_name("model.last.pt")),
        )
        path = root / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        smoke.append(
            dict(
                index=len(smoke),
                name=name,
                system="au2_mgo",
                depth=2,
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
        parent=str(PARENT),
        software_parent=str(DENSE),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        tasks=tasks,
        smoke_tasks=smoke,
        inputs=inputs,
        independent_variable="Nonlinear slab height coupling: global -> full. Other architectural and optimization settings unchanged.",
        batching="Four one-structure microbatches per optimizer step, preserving batch size four. Evaluation batch one.",
        selection="model.pt: minimum validation loss. Raw held-out energy/force RMSE. No test labels used in training or selection.",
        cache_policy="In-memory targets and graphs only. No persistent graph caches.",
        smoke_scope="Both frozen and joint: complete production train/validation inputs, physical batch sizes, 12 steps, joint warmup shortened to two steps to exercise backbone gradients.",
        linear_comparison="Existing linear slab uses one direct source-height matrix, not an activation-only matched stack.",
    )
    manifest["source_hashes"] = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".yaml", ".slurm", ".sh", ".md", ".toml", ".patch")
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root, len(tasks), "production tasks;", len(smoke), "smoke tasks")


if __name__ == "__main__":
    main()
