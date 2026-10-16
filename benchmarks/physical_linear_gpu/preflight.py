"""Check the archived data, CLI settings, and exact authorized config changes."""

import json
from pathlib import Path

import yaml
from ase.io import read
from run import digest


def main():
    from mace_fno.cli.config import parse_arguments
    from mace_fno.training.configuration import TrainingConfig

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        assert digest(root / name) == expected, name
    for name, record in manifest["inputs"].items():
        assert digest(root / name) == record["sha256"], name
    for task in manifest["tasks"] + manifest["smoke_tasks"]:
        raw = yaml.safe_load(Path(task["config"]).read_text())
        cfg = TrainingConfig.from_namespace(
            parse_arguments(["--config", task["config"]])
        )
        assert cfg.model.architecture == "linear" and cfg.model.fno_layers == 1
        assert raw["spectral_groups"] == 1 and raw["spectral_diagnostic_samples"] == 0
        assert cfg.runtime.resume is None and raw["init_from"] is None
        assert cfg.data.test_file is None and cfg.data.train_cache is None
        assert (
            cfg.optimization.steps == task["steps"]
            and not Path(task["checkpoint"]).exists()
        )
        if "parent_config" in task:
            old = yaml.safe_load(Path(task["parent_config"]).read_text())
            for key, value in old.items():
                if key not in task["changes"]:
                    assert raw[key] == value, (task["name"], key)
        if task["system"] == "au2_mgo":
            assert (
                raw["batch_size"] == 4
                and raw["accumulation_steps"] == 1
                and raw["evaluation_batch_size"] == 1
            )
    counts = {}
    for system, expected in [("water", [471, 52, 134]), ("au2_mgo", [4275, 225, 500])]:
        counts[system] = [
            len(read(root / "data" / system / f"{s}.xyz", index=":"))
            for s in ["train", "validation", "test"]
        ]
        assert counts[system] == expected
    record = dict(
        state="passed",
        production_tasks=len(manifest["tasks"]),
        smoke_tasks=len(manifest["smoke_tasks"]),
        exact_changes_verified=True,
        source_hashes_verified=True,
        split_counts=counts,
        spectral_groups=1,
        spectral_checks=False,
    )
    (root / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
