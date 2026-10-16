"""Snapshot current validation-selected dense or low-rank full-z models."""

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stable_copy(source, target):
    for attempt in range(3):
        before = source.stat()
        shutil.copy2(source, target)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            expected = digest(target)
            if digest(source) == expected:
                return expected
    raise RuntimeError(f"Checkpoint changed repeatedly during snapshot: {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_campaign", type=Path)
    parser.add_argument("wetting_campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--regime",
        choices=["frozen", "joint"],
        help="Evaluate only the selected training regime.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require completed 240000-step fits and reuse their saved test evaluations.",
    )
    args = parser.parse_args()
    parent, wetting, root = (
        p.resolve() for p in (args.full_campaign, args.wetting_campaign, args.output)
    )
    torch.set_num_threads(1)
    root.mkdir(exist_ok=False)
    manifest = json.loads((parent / "manifest.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        assert digest(parent / name) == expected, name
    for name, record in manifest["inputs"].items():
        assert digest(parent / name) == record["sha256"], name
    for folder in [
        "models",
        "configs",
        "histories",
        "data",
        "inputs",
        "logs",
        "results",
        "completed_tests",
    ]:
        (root / folder).mkdir()
    for folder in ["source", "mace_source"]:
        shutil.copytree(
            parent / folder,
            root / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(parent / "evaluate.py", root / "evaluate_test.py")
    shutil.copy2(
        wetting / "evaluate_adsorption_switch.py",
        root / "evaluate_adsorption_switch.py",
    )
    for name in ["prepare.py", "run.py", "job.slurm", "collect.py"]:
        shutil.copy2(Path(__file__).parent / name, root / name)
    shutil.copy2(parent / "data/test.xyz", root / "data/test.xyz")
    shutil.copy2(parent / "models/mace_nl1.model", root / "models/mace_nl1.model")
    wet_manifest = json.loads((wetting / "manifest.json").read_text())
    for name in ["1-undoped", "3-undoped", "1-doped", "3-doped"]:
        relative = f"inputs/{name}.xyz"
        assert digest(wetting / relative) == wet_manifest["source_checksums"][relative]
        shutil.copy2(wetting / relative, root / relative)
    backbone_hash = digest(root / "models/mace_nl1.model")
    tasks = []
    for task in manifest["tasks"]:
        if args.regime and task["opt_level"] != args.regime:
            continue
        name = task["name"]
        source = Path(task["checkpoint"])
        cp_path = root / "models" / f"{name}.pt"
        checksum = stable_copy(source, cp_path)
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        rank = task.get("rank", 0)
        assert (cp["z_mixing"], cp.get("spectral_rank", 0), cp["architecture"]) == (
            "full",
            rank,
            "nonlinear",
        )
        assert (cp["seed"], cp["mace_training"]) == (task["seed"], task["opt_level"])
        assert (
            cp["channels"],
            cp["fno_hidden_channels"],
            cp["z_grid_size"],
            cp["spectral_groups"],
            cp["lateral_interlacing"],
        ) == (16, 16, 24, 1, 2)
        assert digest(cp["mace_model"]) == backbone_hash
        history = [
            json.loads(line)
            for line in (source.parent / "validation_history.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        history = [r for r in history if isinstance(r.get("step"), int)]
        through = [r for r in history if r["step"] <= cp["completed_steps"]]
        chosen = next(r for r in through if r["step"] == cp["best_step"])
        minimum = min(r["validation_objective"] for r in through)
        assert math.isclose(
            chosen["validation_objective"], minimum, rel_tol=1e-8, abs_tol=1e-12
        )
        assert math.isclose(
            cp["best_validation_objective"], minimum, rel_tol=1e-8, abs_tol=1e-12
        )
        save(root / "histories" / f"{name}.json", history)
        cfg = yaml.safe_load(Path(task["config"]).read_text())
        assert cfg["z_mixing"] == "full" and cfg["evaluation_batch_size"] == 1
        cfg.update(
            checkpoint=str(cp_path),
            mace_model=str(root / "models/mace_nl1.model"),
            device="cuda",
            dtype="float64",
        )
        config = root / "configs" / f"{name}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        tasks.append(
            dict(
                index=len(tasks),
                name=name,
                regime=task["opt_level"],
                seed=task["seed"],
                spectral_rank=rank,
                checkpoint=str(cp_path),
                checkpoint_sha256=checksum,
                source_checkpoint=str(source),
                selected_step=cp["best_step"],
                checkpoint_completed_steps=cp["completed_steps"],
                observed_step=max(r["step"] for r in history),
                budget=240000,
                validation=chosen["metrics"],
                config=str(config),
                preliminary=max(r["step"] for r in history) < 240000,
            )
        )
        if args.require_complete:
            state = json.loads((source.parent / "status.json").read_text())
            test_path = source.parent / "test.json"
            test = json.loads(test_path.read_text())
            assert state["state"] == state["stage"] == "complete", name
            assert state["checkpoint_sha256"] == checksum
            assert task["steps"] == max(r["step"] for r in history) == 240000
            assert (
                test["checkpoint"] == str(source)
                and test["best_step"] == cp["best_step"]
            )
            assert test["checkpoint_selection"] == "best_validation_loss"
            assert test["samples"] == 500 and not test["energy_shift_applied"]
            archived_test = root / "completed_tests" / f"{name}.json"
            shutil.copy2(test_path, archived_test)
            tasks[-1].update(
                completed_test=str(archived_test),
                completed_test_sha256=digest(archived_test),
                completed_test_source=str(test_path),
                training_complete=True,
            )
        print(
            name,
            "selected",
            cp["best_step"],
            "observed",
            tasks[-1]["observed_step"],
            flush=True,
        )
        del cp
    assert tasks
    assert len({(t["regime"], t["spectral_rank"], t["seed"]) for t in tasks}) == len(
        tasks
    )
    content = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(parent),
        wetting_parent=str(wetting),
        selection="current best validation loss, never test errors",
        z_mixing="full",
        spectral_ranks=sorted({t["spectral_rank"] for t in tasks}),
        mace_layers=1,
        seeds=sorted({t["seed"] for t in tasks}),
        test_structures=500,
        atoms=110,
        fmax=0.01,
        maximum_steps=500,
        raw_energies=True,
        tasks=tasks,
    )
    content["checksums"] = {
        str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()
    }
    save(root / "manifest.json", content)
    print(f"Prepared {len(tasks)} immutable checkpoints:", root, flush=True)


if __name__ == "__main__":
    main()
