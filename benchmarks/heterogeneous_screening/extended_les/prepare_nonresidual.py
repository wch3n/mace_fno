"""Make fresh nonresidual-first extended LES fits from the five-seed protocol."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml
from prepare_response import sha

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    args = parser.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    old = json.loads((parent / "manifest.json").read_text())
    for name, digest in old["sources"].items():
        assert sha(parent / name) == digest, name
    for name, digest in old["installed_sources"].items():
        assert sha(name) == digest, name
    root.mkdir(exist_ok=False)
    for directory in ("configs", "runs", "smoke", "logs", "results", "provenance"):
        (root / directory).mkdir()
    for name in ("data", "venv"):
        (root / name).symlink_to((parent / name).resolve(), target_is_directory=True)
    shutil.copy2(parent / "manifest.json", root / "provenance/parent_manifest.json")
    shutil.copy2(
        parent / "configs/les_response.yaml", root / "provenance/les_response.yaml"
    )
    shutil.copy2(
        parent / "configs/les_response.yaml", root / "configs/les_response.yaml"
    )
    shutil.copy2(
        old["tasks"][0]["smoke"]["model"], root / "provenance/residual_smoke.model"
    )
    tasks = deepcopy(old["tasks"])
    for task, original in zip(tasks, old["tasks"]):
        task["name"] = f"les_response_nonres_seed{task['seed']}"
        task["les_arguments"] = str(root / "configs/les_response.yaml")
        for mode in ("train", "smoke"):
            if mode not in task:
                continue
            source = Path(original[mode]["config"])
            shutil.copy2(
                source, root / "provenance" / f"seed{task['seed']}_{mode}.yaml"
            )
            config = yaml.safe_load(source.read_text())
            directory = root / ("runs" if mode == "train" else "smoke") / task["name"]
            directory.mkdir()
            config.update(
                name=task["name"],
                interaction_first="RealAgnosticInteractionBlock",
                restart_latest=False,
                max_num_epochs=1200 if mode == "train" else 4,
                work_dir=str(directory),
                log_dir=str(directory / "logs"),
                model_dir=str(directory / "models"),
                checkpoints_dir=str(directory / "checkpoints"),
                results_dir=str(directory / "results"),
                les_arguments=task["les_arguments"],
            )
            for key in ("train_file", "valid_file"):
                config[key] = str(root / "data" / Path(config[key]).name)
            path = root / "configs" / f"{task['name']}_{mode}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            task[mode] = dict(
                directory=str(directory),
                config=str(path),
                model=str(directory / "models" / f"{task['name']}_stagetwo.model"),
            )
    # Inherit the optimizer and conversion compatibility code byte-for-byte.
    for name in (
        "cueq_compat.py",
        "train_response.py",
        "optimizer_compat.py",
        "response.slurm",
        "collect.slurm",
        "eval_probe.py",
        "eval_probe2.py",
        "analyze_probe2.py",
    ):
        shutil.copy2(parent / name, root / name)
        assert sha(root / name) == old["sources"][name]
    for name in (
        "run_response.py",
        "evaluate.py",
        "collect.py",
        "prepare_response.py",
        "prepare_nonresidual.py",
        "check_first_block.py",
        "README_nonresidual.md",
    ):
        shutil.copy2(HERE / name, root / name)
    manifest = deepcopy(old)
    manifest.pop("budget_revision", None)
    manifest["training"].pop("restart", None)
    manifest["training"]["epochs"] = 1200
    manifest.pop("conversion_certificate")
    manifest.update(
        root=str(root),
        parent=str(parent),
        tasks=tasks,
        interaction_first="RealAgnosticInteractionBlock",
        expected_parameters=485376,
        first_block_comparison="Only interaction_first changes in the training model; sources, data, response settings and corrected optimizer are retained.",
        conversion_validation="Fresh GPU parity, optimizer-step and two-stage checks required for this architecture.",
        parent_results=str(parent / "results/summary.json"),
        sources={
            str(p.relative_to(root)): sha(p)
            for p in [
                *root.glob("*.py"),
                *root.glob("*.slurm"),
                *(root / "configs").glob("*.yaml"),
                *(root / "provenance").iterdir(),
            ]
            if p.is_file()
        },
    )
    for name, record in manifest["inputs"].items():
        assert sha(root / name) == record["sha256"], name
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                root=str(root),
                seeds=manifest["seeds"],
                interaction_first=manifest["interaction_first"],
                training=manifest["training"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
