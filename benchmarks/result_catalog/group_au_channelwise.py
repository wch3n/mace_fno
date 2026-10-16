#!/usr/bin/env python3
"""Group the twenty archived channel-wise Au2/MgO FNO fits by variant and seed."""

import csv
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path("/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs")
REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "docs/source/data/au2_mgo_converged/test_per_seed.csv"
DEST = ROOT / "results/collections/au2_mgo_channelwise"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def link(target, source):
    if target.is_symlink():
        assert target.resolve() == source.resolve(), target
    else:
        assert not target.exists(), target
        target.symlink_to(
            os.path.relpath(source, target.parent), target_is_directory=source.is_dir()
        )


def main():
    rows = [
        r
        for r in csv.DictReader(SOURCE.open())
        if r["opt_level"] in {"frozen", "joint"}
    ]
    assert len(rows) == 20
    records = []
    for r in rows:
        checkpoint = Path(r["checkpoint"])
        config = ROOT / r["config_source"]
        metrics = ROOT / r["metrics_source"]
        status = checkpoint.parent / "status.json"
        cfg = yaml.safe_load(config.read_text())
        state = json.loads(status.read_text())
        assert state.get("state", state.get("status")) == "complete", status
        assert cfg["steps"] == 240000 and cfg["z_mixing"] == "global", config
        assert cfg.get("architecture", "nonlinear") == "nonlinear"
        assert checkpoint.exists() and metrics.exists()
        variant = f"mace{r['mace_layers']}_{r['opt_level']}"
        records.append(
            dict(
                variant=variant,
                seed=int(r["seed"]),
                run=str(checkpoint.parent),
                checkpoint=str(checkpoint),
                config=str(config),
                metrics=str(metrics),
                steps=cfg["steps"],
                z_mixing=cfg["z_mixing"],
                config_sha256=sha(config),
                metrics_sha256=sha(metrics),
                completion_sha256=sha(status),
                metrics_row=r,
            )
        )
    for variant in sorted({r["variant"] for r in records}):
        selected = [r for r in records if r["variant"] == variant]
        assert {r["seed"] for r in selected} == {17, 29, 41, 53, 67}
        folder = DEST / variant
        (folder / "configs").mkdir(parents=True, exist_ok=True)
        (folder / "test_results").mkdir(exist_ok=True)
        for r in selected:
            link(folder / f"seed{r['seed']}", Path(r["run"]))
            link(folder / "configs" / f"seed{r['seed']}.yaml", Path(r["config"]))
            link(folder / "test_results" / f"seed{r['seed']}.json", Path(r["metrics"]))
        lines = [
            f"# Au2/MgO channel-wise nonlinear FNO: {variant}",
            "",
            "All five fits completed their 240,000-step training budget.",
            "Open a seed directory for its checkpoints and training logs. "
            "The `configs/` and `test_results/` folders collect the corresponding configuration "
            "and held-out evaluation, including evaluations stored in a separate campaign.",
            "",
            "| Seed | Original run |",
            "|---|---|",
        ]
        lines.extend(
            f"| [{r['seed']}](seed{r['seed']}) | `{r['run']}` |" for r in selected
        )
        (folder / "README.md").write_text("\n".join(lines) + "\n")
    manifest = dict(
        source_csv=str(SOURCE),
        source_csv_sha256=sha(SOURCE),
        records=records,
        scope="Archived channel-wise nonlinear FNO results; distinct from the newer full-z fits.",
    )
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (DEST / "README.md").write_text(
        "# Au2/MgO channel-wise nonlinear FNO: all five seeds\n\n"
        "The original campaigns are combined here through links. Each variant contains "
        "`seed17`, `seed29`, `seed41`, `seed53` and `seed67`, plus configurations and test results.\n\n"
        "- [MACE1 joint](mace1_joint/README.md)\n"
        "- [MACE1 frozen](mace1_frozen/README.md)\n"
        "- [MACE2 joint](mace2_joint/README.md)\n"
        "- [MACE2 frozen](mace2_frozen/README.md)\n\n"
        "Every fit completed a 240,000-step budget. These are the archived channel-wise models "
        "identified by `au2_mgo_converged/test_per_seed.csv`; the newer full-z fits are separate. "
        "The MACE2 channel-wise models use the earlier scalar-only backbone.\n\n"
        "Seeds 17 and 67 were added in `au2_mgo_fno_5seed_20260926T200142Z`. "
        "The manifest records the earlier campaigns supplying seeds 29, 41 and 53. "
        "No original run files were moved or modified.\n"
    )
    for p in DEST.rglob("*"):
        if p.is_symlink():
            assert p.exists(), p
    print(DEST)
    print("Verified 20 runs, four five-seed groups, and 60 links.")


if __name__ == "__main__":
    main()
