"""Collect all 60 G=1 physical linear-FNO results after GPU completion."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    parent = Path(manifest["parent"])
    original = json.loads((parent / "manifest.json").read_text())
    tasks = [
        t for t in original["tasks"] if t["system"] != "water" and t["index"] < 51
    ] + manifest["tasks"]
    assert len(tasks) == 60
    rows, sources = [], {}

    def read(path):
        path = Path(path)
        data = path.read_bytes()
        sources[str(path)] = dict(
            sha256=hashlib.sha256(data).hexdigest(), text=data.decode()
        )
        return json.loads(data)

    for task in tasks:
        directory = Path(task["checkpoint"]).parent
        status = read(directory / "status.json")
        test = read(directory / "test.json")
        assert status["state"] == "complete" and not test["energy_shift_applied"]
        assert (
            test["checkpoint"] == task["checkpoint"]
            and test["best_step"] == status["selected_step"]
        )
        assert test["samples"] == dict(water=134, llzo=200, au2_mgo=500)[task["system"]]
        with Path(task["checkpoint"]).open("rb") as stream:
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest()
                == status["checkpoint_sha256"]
            )
        config = Path(task["config"]).read_bytes()
        assert (
            b"\nspectral_groups: 1\n" in config
            and b"\narchitecture: linear\n" in config
        )
        sources[task["config"]] = dict(
            sha256=hashlib.sha256(config).hexdigest(), text=config.decode()
        )
        values = test["splits"]["test"]["mace_fno"]
        row = dict(
            system=task["system"],
            depth=task["depth"],
            regime=task["opt_level"],
            seed=task["seed"],
            spectral_groups=1,
            selected_step=status["selected_step"],
            budget=task["steps"],
            checkpoint=task["checkpoint"],
            checkpoint_sha256=status["checkpoint_sha256"],
            energy_rmse_meV_atom=1000 * values["energy_rmse"],
            force_rmse_meV_A=1000 * values["force_rmse"],
        )
        assert math.isfinite(row["energy_rmse_meV_atom"]) and math.isfinite(
            row["force_rmse_meV_A"]
        )
        rows.append(row)
    rows.sort(key=lambda r: (r["system"], r["depth"], r["regime"], r["seed"]))
    summary = []
    for system, depth, regime in sorted(
        {(r["system"], r["depth"], r["regime"]) for r in rows}
    ):
        group = [
            r
            for r in rows
            if (r["system"], r["depth"], r["regime"]) == (system, depth, regime)
        ]
        assert sorted(r["seed"] for r in group) == [17, 29, 41, 53, 67]
        row = dict(system=system, depth=depth, regime=regime, n=5, spectral_groups=1)
        for key in ["energy_rmse_meV_atom", "force_rmse_meV_A"]:
            row[key + "_mean"] = statistics.mean(r[key] for r in group)
            row[key + "_sd"] = statistics.stdev(r[key] for r in group)
        summary.append(row)
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    for name, records in [("per_seed", rows), ("summary", summary)]:
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (output / f"{name}.json").write_text(
            json.dumps(records, indent=2, allow_nan=False) + "\n"
        )
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (root / "collection_status.json").write_text(
        json.dumps(
            dict(
                state="complete", models=60, spectral_groups=1, manuscript_edited=False
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Collected all 60 G=1 physical linear-FNO fits. Manuscript tables await review."
    )


if __name__ == "__main__":
    main()
