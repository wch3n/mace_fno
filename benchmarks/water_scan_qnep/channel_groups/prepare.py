"""Prepare one G=1 water fit against the archived G=4 seed-29 control."""

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replace_once(path, before, after):
    text = path.read_text()
    assert text.count(before) == 1, (path, before)
    path.write_text(text.replace(before, after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    parent, root = args.reference_campaign.resolve(), args.output.resolve()
    previous = json.loads((parent / "manifest.json").read_text())
    reference = next(
        r
        for r in previous["references"]
        if r["system"] == "water_scan" and r["seed"] == 29
    )
    assert digest(reference["config"]) == reference["config_sha256"]
    assert digest(reference["checkpoint"]) == reference["checkpoint_sha256"]
    cfg = yaml.safe_load(Path(reference["config"]).read_text())
    assert cfg["seed"] == 29 and cfg["spectral_groups"] == 4
    assert cfg["mace_training"] == "frozen" and cfg["architecture"] == "nonlinear"
    assert cfg["steps"] == 120000 and cfg["test_file"] is None
    assert cfg["test_cache"] is None and cfg["checkpoint_interval"] == 500
    assert (
        cfg["grid"],
        cfg["modes"],
        cfg["channels"],
        cfg["fno_hidden_channels"],
        cfg["fno_layers"],
    ) == (24, 8, 16, 16, 2)
    wrappers = ("common.py", "train.py", "preflight.py", "smoke.py", "run.py")
    for relative, checksum in previous["source_checksums"].items():
        if relative.startswith("source/") or relative in wrappers:
            assert digest(parent / relative) == checksum, relative
    info = copy.deepcopy(previous["systems"]["water_scan"])
    for path, checksum in info["input_checksums"].items():
        assert digest(path) == checksum, path
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "configs",
        "references",
        "inputs",
        "runs",
        "smoke",
        "logs",
        "analysis",
    ):
        (root / name).mkdir()
    shutil.copytree(
        parent / "source",
        root / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in wrappers:
        shutil.copy2(parent / name, root / name)
    for name in ("prepare.py", "collect.py", "submit.py", "job.slurm", "README.md"):
        shutil.copy2(Path(__file__).parent / name, root / name)
    replace_once(
        root / "preflight.py",
        "tasks=51,references=9,",
        "tasks=len(m['tasks']),references=len(m['references']),",
    )
    replace_once(
        root / "run.py",
        "        assert state['seed']==t['seed'] and state['fno_layers']==t['fno_layers']\n",
        "        assert state['seed']==t['seed'] and state['fno_layers']==t['fno_layers']\n"
        "        assert state['spectral_groups']==t['spectral_groups']==1\n",
    )
    replace_once(
        root / "smoke.py",
        "            model.train()\n",
        "            model.train()\n"
        "            spectral=[v for v in model.modules() if hasattr(v,'radial_weight')]\n"
        "            assert len(spectral)==2 and all(v.groups==1 for v in spectral)\n",
    )
    replace_once(
        root / "smoke.py",
        "finite_force_gradients=True,backbone_unchanged=True,step_seconds=times,",
        "finite_force_gradients=True,backbone_unchanged=True,step_seconds=times,\n"
        "                spectral_groups=1,trainable_parameters=sum(p.numel() for p in parameters),\n"
        "                spectral_parameters=sum(v.radial_weight.numel() for v in spectral),",
    )
    ref = dict(reference)
    for field in ("config", "result", "history"):
        dest = root / "references" / Path(reference[field]).name
        shutil.copy2(reference[field], dest)
        ref[field] = str(dest)
    for field in ("mace_model", "train_file", "validation_file"):
        path = Path(cfg[field])
        shutil.copy2(path, root / "inputs" / (field + path.suffix))
    name = "water_scan_nl1_frozen_groups1_seed29"
    new = dict(
        cfg, spectral_groups=1, checkpoint=str(root / "runs" / name / "model.pt")
    )
    changes = {
        k: {"before": cfg.get(k), "after": v} for k, v in new.items() if cfg.get(k) != v
    }
    assert set(changes) == {"spectral_groups", "checkpoint"}
    path = root / "configs" / (name + ".yaml")
    path.write_text(yaml.safe_dump(new, sort_keys=False))
    task = dict(
        index=0,
        name=name,
        system="water_scan",
        variant="groups1",
        seed=29,
        regime="frozen",
        mace_layers=1,
        expected_steps=120000,
        config=str(path),
        config_sha256=digest(path),
        checkpoint=new["checkpoint"],
        changes={"spectral_groups": 1},
        **{
            k: new[k]
            for k in (
                "grid",
                "z_grid",
                "modes",
                "channels",
                "fno_layers",
                "spectral_groups",
            )
        },
    )
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        material_passport=dict(
            workflow="academic-research-suite experiment-agent",
            status="prepared",
            scope="one-seed frozen-MACE1 grouping sensitivity",
        ),
        parent_campaign=str(parent),
        parent_manifest_sha256=digest(parent / "manifest.json"),
        package_versions=previous["package_versions"],
        seeds=[29],
        tasks=[task],
        systems={"water_scan": info},
        references=[ref],
        scope="validation only; one seed; no automatic manuscript update",
        selection="minimum combined validation loss; raw energy RMSE; no fitted offset",
        config_changes=changes,
        checkpoint_interval=500,
    )
    manifest["source_checksums"] = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    dump(root / "manifest.json", manifest)
    print(json.dumps(dict(campaign=str(root), changes=changes), indent=2))


if __name__ == "__main__":
    main()
