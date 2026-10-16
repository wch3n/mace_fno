"""Verify unchanged input files, matched training settings, and exported models."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIRST = "RealAgnosticInteractionBlock"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify():
    import yaml
    from mace.tools import build_default_arg_parser

    assert Path.cwd() == ROOT
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, sha in manifest["inputs_and_source_checksums"].items():
        assert digest(ROOT / name) == sha, name
    for name, version in manifest["package_versions"].items():
        assert importlib.metadata.version(name) == version, name
    for name in ("mace", "les"):
        assert (
            Path(importlib.util.find_spec(name).origin).parent == ROOT / "source" / name
        )
    allowed = {
        "name",
        "work_dir",
        "log_dir",
        "model_dir",
        "checkpoints_dir",
        "results_dir",
        "train_file",
        "valid_file",
        "test_file",
        "interaction_first",
        "enable_cueq",
        "restart_latest",
        "les_arguments",
    }
    for task in manifest["tasks"]:
        old = yaml.safe_load((ROOT / "provenance" / f"{task['name']}.yaml").read_text())
        new = yaml.safe_load((ROOT / task["config"]).read_text())
        differences = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        assert differences <= allowed, differences
        assert (
            new["interaction_first"] == FIRST
            and new["enable_cueq"]
            and not new["restart_latest"]
        )
        for key in ("train_file", "valid_file", "test_file"):
            assert (key in old) == (key in new)
            if key in old:
                original = (Path(task["old_directory"]) / old[key]).resolve()
                rel = str(Path(new[key]).relative_to(ROOT))
                assert str(original) == manifest["original_data"][rel]
        args = build_default_arg_parser().parse_args(
            ["--config", str(ROOT / task["config"])]
        )
        assert (
            args.model == "MACELES"
            and args.num_interactions == 1
            and args.seed == task["seed"]
        )
        assert (
            args.interaction_first == FIRST
            and args.hidden_irreps.replace(" ", "") == "128x0e+128x1o"
        )
        assert (
            args.r_max == 4.5
            and args.batch_size == 4
            and args.default_dtype == "float64"
        )
        assert (
            args.swa
            and args.start_swa == 450
            and args.max_num_epochs == 1200
            and args.patience == 50
        )
        assert not args.foundation_model and not args.restart_latest
    return manifest


def check_model(model, task, control=False):
    import torch

    first = "RealAgnosticResidualInteractionBlock" if control else FIRST
    assert type(model).__name__ == "MACELES"
    assert int(model.num_interactions) == 1 and float(model.r_max) == 4.5
    assert [type(b).__name__ for b in model.interactions] == [first]
    assert [str(p.linear.irreps_out) for p in model.products] == ["128x0e"]
    assert next(model.parameters()).dtype == torch.float64
    assert not getattr(model, "keep_last_layer_irreps", False)
    assert model.atomic_numbers.tolist() == (
        [11, 17, 18] if task["dataset"] == "hetscreen" else [11, 17]
    )
    assert len(model.les_readouts) == len(model.readouts) == 1
    assert model.les.sigma == 1.0 and model.les.dl == 2.0
    assert model.les.remove_self_interaction and not model.les.use_atomwise
    assert model.les.ewald.norm_factor == 90.4756
    count = sum(p.numel() for p in model.parameters())
    assert (
        count
        == task["expected_control_parameters" if control else "expected_parameters"]
    ), count
    return dict(
        first_block=first,
        parameters=count,
        product_irreps="128x0e",
        cutoff_A=4.5,
        remove_self_interaction=True,
        ewald_norm_factor=90.4756,
    )


def smoke(index):
    import numpy as np
    import torch
    import yaml
    from ase.io import read
    from mace.calculators import MACECalculator
    from train_cueq import install

    install(yaml.safe_load((ROOT / "configs/les.yaml").read_text()))
    from mace.tools.scripts_utils import extract_config_mace_model

    task = verify()["tasks"][index]
    torch.set_default_dtype(torch.float64)
    details = {}
    for backend in ("native", "cueq"):
        directory = ROOT / "smoke" / task["dataset"] / backend / "models"
        paths = [p for p in directory.glob("*.model") if "compiled" not in p.name]
        assert len(paths) == 2 and any(
            p.name.endswith("_stagetwo.model") for p in paths
        )
        for path in paths:
            calc = MACECalculator(
                model_paths=str(path), device="cuda", default_dtype="float64"
            )
            model = calc.models[0]
            info = check_model(model, task)
            rebuilt = type(model)(**extract_config_mace_model(model))
            rebuilt.load_state_dict(model.state_dict(), strict=True)
            check_model(rebuilt, task)
            atoms = read(ROOT / "data" / task["dataset"] / "smoke_valid.xyz", 0)
            atoms.calc = calc
            assert (
                np.isfinite(atoms.get_potential_energy())
                and np.isfinite(atoms.get_forces()).all()
            )
            details[str(path.relative_to(ROOT))] = dict(sha256=digest(path), **info)
    dump(
        ROOT / "validation" / task["dataset"] / "smoke.json",
        dict(
            state="passed",
            two_stage_training=True,
            config_roundtrip=True,
            models=details,
        ),
    )
    print("TWO-STAGE EXPORT CHECK PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int)
    args = parser.parse_args()
    if args.smoke is None:
        print("PREFLIGHT PASSED:", len(verify()["tasks"]), "matched configurations")
    else:
        smoke(args.smoke)
