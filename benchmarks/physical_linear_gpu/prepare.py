"""Prepare G=1 linear-water reruns and the pending linear-slab GPU jobs."""

import argparse
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
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent, root = args.parent.resolve(), args.output.resolve()
    previous = json.loads((parent / "manifest.json").read_text())
    # Configurations and the exact original trainer/model code are immutable.
    for name, expected in previous["source_hashes"].items():
        assert digest(parent / name) == expected, name
    for name, record in previous["inputs"].items():
        assert digest(parent / name) == record["sha256"], name
    root.mkdir(exist_ok=False)
    for name in [
        "configs",
        "references",
        "models",
        "data",
        "runs",
        "smoke",
        "logs",
        "analysis",
    ]:
        (root / name).mkdir()
    for name in ["source", "mace_source"]:
        shutil.copytree(
            parent / name,
            root / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ["train.py", "evaluate.py"]:
        shutil.copy2(parent / name, root / name)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, root / path.name)
    inputs = {}
    for system in ["water", "au2_mgo"]:
        shutil.copytree(parent / "data" / system, root / "data" / system)
        for path in (root / "data" / system).iterdir():
            assert path.is_file()
            relative = str(path.relative_to(root))
            assert digest(path) == digest(parent / relative)
            inputs[relative] = dict(source=str(parent / relative), sha256=digest(path))
    for name in ["water_nl1.model", "water_nl2.model", "au2_mgo_nl2.model"]:
        shutil.copy2(parent / "models" / name, root / "models" / name)
        inputs[f"models/{name}"] = dict(
            source=str(parent / "models" / name), sha256=digest(root / "models" / name)
        )
    tasks, smoke = [], []
    for oldtask in previous["tasks"]:
        is_water = oldtask["system"] == "water"
        if not is_water and oldtask["index"] not in range(51, 60):
            continue
        assert is_water or not Path(oldtask["checkpoint"]).parent.exists()
        original = yaml.safe_load(Path(oldtask["config"]).read_text())
        name = oldtask["name"].replace(
            "_linear_", "_linear_g1_" if is_water else "_linear_gpu_"
        )
        cp = root / "runs" / name / "model.pt"
        cfg = dict(original)
        for key in ["mace_model", "train_file", "validation_file"]:
            cfg[key] = str(root / Path(original[key]).relative_to(parent))
        cfg.update(
            checkpoint=str(cp), last_checkpoint=str(cp.with_name("model.last.pt"))
        )
        if is_water:
            assert original["spectral_groups"] == 4
            cfg["spectral_groups"] = 1
        else:
            assert (
                original["spectral_groups"] == 1
                and original["evaluation_batch_size"] == 8
            )
            cfg["evaluation_batch_size"] = 1
        allowed = {
            "mace_model",
            "train_file",
            "validation_file",
            "checkpoint",
            "last_checkpoint",
            "spectral_groups" if is_water else "evaluation_batch_size",
        }
        changes = {
            k: dict(before=original.get(k), after=cfg.get(k))
            for k in original.keys() | cfg.keys()
            if original.get(k) != cfg.get(k)
        }
        assert set(changes) <= allowed
        assert cfg["resume"] is None and cfg["init_from"] is None
        assert (
            cfg["architecture"] == "linear" and cfg["spectral_diagnostic_samples"] == 0
        )
        config = root / "configs" / f"{name}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        shutil.copy2(oldtask["config"], root / "references" / f"{name}.yaml")
        tasks.append(
            dict(
                index=len(tasks),
                name=name,
                system=oldtask["system"],
                depth=oldtask["depth"],
                opt_level=oldtask["opt_level"],
                seed=cfg["seed"],
                steps=cfg["steps"],
                config=str(config),
                checkpoint=str(cp),
                test_file=str(root / "data" / oldtask["system"] / "test.xyz"),
                parent_index=oldtask["index"],
                parent_config=oldtask["config"],
                changes=changes,
                replaces_job=None if is_water else f"15848117_{oldtask['index']}",
            )
        )
        key = (oldtask["system"], oldtask["depth"], oldtask["opt_level"])
        if key not in {(t["system"], t["depth"], t["opt_level"]) for t in smoke}:
            smoke_name = f"{key[0]}_nl{key[1]}_{key[2]}_smoke"
            smoke_cp = root / "smoke" / smoke_name / "model.pt"
            sc = dict(
                cfg,
                seed=17,
                steps=6,
                eval_interval=3,
                checkpoint_interval=3,
                mace_warmup_steps=0,
                train_file=str(root / "data" / key[0] / "smoke_train.xyz"),
                validation_file=str(root / "data" / key[0] / "smoke_validation.xyz"),
                checkpoint=str(smoke_cp),
                last_checkpoint=str(smoke_cp.with_name("model.last.pt")),
            )
            config = root / "configs" / f"{smoke_name}.yaml"
            config.write_text(yaml.safe_dump(sc, sort_keys=False))
            smoke.append(
                dict(
                    index=len(smoke),
                    name=smoke_name,
                    system=key[0],
                    depth=key[1],
                    opt_level=key[2],
                    seed=17,
                    steps=6,
                    config=str(config),
                    checkpoint=str(smoke_cp),
                    test_file=sc["validation_file"],
                )
            )
    assert len(tasks) == 29 and len(smoke) == 6
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(parent),
        tasks=tasks,
        smoke_tasks=smoke,
        inputs=inputs,
        seeds=[17, 29, 41, 53, 67],
        scope="20 fresh G=1 linear-water fits; 9 pending Au2/MgO MACE2 fits on gpu",
        selection=previous["selection"],
        linear_slab=previous["linear_slab"],
        training_changes="Water: spectral_groups 4 to 1. Au2/MgO: evaluation_batch_size 8 to 1 only.",
        spectral_checks=False,
        source_policy="Exact original model and trainer snapshot; new orchestration and memory reporting only.",
    )
    manifest["source_hashes"] = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix in [".py", ".yaml", ".slurm", ".md"]
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root, len(tasks), "fits;", len(smoke), "GPU smoke cases")


if __name__ == "__main__":
    main()
