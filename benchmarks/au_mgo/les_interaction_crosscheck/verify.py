"""Validate the isolated residual-first LES cross-check and its saved models."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IRREPS = "128x0e+128x1o"
FIRST = "RealAgnosticResidualInteractionBlock"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, data):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify():
    import yaml

    assert Path.cwd() == ROOT
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, sha in manifest["inputs_and_source_checksums"].items():
        assert digest(ROOT / name) == sha, name
    for name, version in manifest["package_versions"].items():
        assert importlib.metadata.version(name) == version, name
    for package in ("mace", "les"):
        assert (
            Path(importlib.util.find_spec(package).origin).parent
            == ROOT / "source" / package
        )
    (task,) = manifest["tasks"]
    old = yaml.safe_load((ROOT / "provenance/control_config.yaml").read_text())
    new = yaml.safe_load((ROOT / task["config"]).read_text())
    differences = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    expected = {"name", "interaction_first"}
    if manifest.get("backend") == "cueq":
        expected.add("enable_cueq")
        assert new["enable_cueq"] is True
    assert differences == expected, differences
    assert old["interaction_first"] == "RealAgnosticInteractionBlock"
    from mace.tools import build_default_arg_parser

    args = build_default_arg_parser().parse_args(["--config", task["config"]])
    assert args.model == "MACELES" and args.num_interactions == 1 and args.seed == 17
    assert args.interaction_first == FIRST
    assert args.hidden_irreps.replace(" ", "") == IRREPS and args.keep_last_layer_irreps
    assert (
        args.r_max == 5.5 and args.batch_size == 10 and args.default_dtype == "float64"
    )
    assert args.swa and args.start_swa == 500 and args.max_num_epochs == 1000
    assert not args.foundation_model and not args.restart_latest
    return manifest


def check_model(model, task):
    import torch

    assert type(model).__name__ == "MACELES"
    assert int(model.num_interactions) == 1 and float(model.r_max) == 5.5
    assert [type(b).__name__ for b in model.interactions] == [FIRST]
    assert [str(p.linear.irreps_out) for p in model.products] == [IRREPS]
    assert (
        next(model.parameters()).dtype == torch.float64 and model.keep_last_layer_irreps
    )
    assert len(model.les_readouts) == 1 and len(model.readouts) == 1
    assert model.les.sigma == 1.0 and model.les.dl == 2.0
    assert model.les.remove_self_interaction and not model.les.use_atomwise
    assert model.les.ewald.norm_factor == 90.4756
    count = sum(p.numel() for p in model.parameters())
    assert count == 266496, count
    return dict(
        model_class=type(model).__name__,
        interaction_first=FIRST,
        product_irreps=[IRREPS],
        parameters=count,
        cutoff_A=5.5,
        remove_self_interaction=True,
        ewald_norm_factor=90.4756,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = verify()
    import numpy as np
    from ase.io import read

    if args.preflight:
        split = json.loads((ROOT / "data/split_manifest.json").read_text())
        assert set(split["train_indices"]).isdisjoint(split["validation_indices"])
        assert set(split["train_indices"]) | set(split["validation_indices"]) == set(
            range(4500)
        )
        for name, count in manifest["counts"].items():
            frames = read(ROOT / "data" / f"{name}.xyz", index=":")
            assert len(frames) == count
            for atoms in frames:
                assert (
                    len(atoms) == 110
                    and atoms.pbc.all()
                    and np.linalg.det(atoms.cell) > 0
                )
                assert (
                    np.isfinite(atoms.get_potential_energy())
                    and np.isfinite(atoms.get_forces()).all()
                )
        dump(
            ROOT / "preflight.json",
            dict(
                state="passed",
                counts=manifest["counts"],
                architecture_setting_changed="interaction_first",
                backend=manifest.get("backend", "e3nn"),
                immutable_parent_source_and_splits=True,
            ),
        )
        print("CROSS-CHECK PREFLIGHT PASSED", flush=True)
    if args.smoke:
        import torch

        torch.set_default_dtype(torch.float64)
        from mace.calculators import MACECalculator
        from mace.tools.scripts_utils import extract_config_mace_model

        assert torch.cuda.is_available()
        paths = [
            p
            for p in (ROOT / "smoke/les_nl1/models").glob("*.model")
            if "compiled" not in p.name
        ]
        assert len(paths) == 2 and any(
            p.name.endswith("_stagetwo.model") for p in paths
        )
        details = {}
        for path in paths:
            model = torch.load(path, map_location="cpu", weights_only=False)
            info = check_model(model, manifest["tasks"][0])
            rebuilt = type(model)(**extract_config_mace_model(model))
            rebuilt.load_state_dict(model.state_dict(), strict=True)
            check_model(rebuilt, manifest["tasks"][0])
            calc = MACECalculator(
                model_paths=str(path), device="cuda", default_dtype="float64"
            )
            for index in (0, 4):
                atoms = read(ROOT / "data/smoke_validation.xyz", index=index)
                atoms.calc = calc
                assert (
                    np.isfinite(atoms.get_potential_energy())
                    and np.isfinite(atoms.get_forces()).all()
                )
            details[str(path.relative_to(ROOT))] = dict(sha256=digest(path), **info)
        dump(
            ROOT / "smoke.json",
            dict(
                state="passed",
                two_stage_training=True,
                config_roundtrip_verified=True,
                both_compositions=True,
                models=details,
            ),
        )
        print("CROSS-CHECK TWO-STAGE SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
