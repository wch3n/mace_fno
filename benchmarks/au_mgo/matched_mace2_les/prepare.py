"""Prepare fresh scalar-vector MACE2-LES fits and validate the accelerated backend."""

import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    scripts = Path(__file__).resolve().parent
    base = Path.home() / "mace_fno_runs/_campaigns"
    parent = base / "au2_mgo_matched_irreps_20260929T182640Z/mace2_les"
    wetting = base / "au2_mgo_matched_les_wetting_20261011"
    root = base / "au2_mgo_matched_mace2_les_cueq_5seed_20261016"
    original = json.loads((parent / "manifest.json").read_text())
    root.mkdir(exist_ok=False)
    for directory in (
        "configs",
        "provenance",
        "inputs",
        "logs",
        "runs",
        "runtime",
        "smoke",
    ):
        (root / directory).mkdir()
    for directory in ("source", "data", "references"):
        shutil.copytree(
            parent / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for path in (root / directory).rglob("*"):
            if path.is_file():
                relative = str(path.relative_to(root))
                assert (
                    digest(path) == original["inputs_and_source_checksums"][relative]
                ), relative
    for path in (parent / "configs").glob("*.yaml"):
        relative = "configs/" + path.name
        assert digest(path) == original["inputs_and_source_checksums"][relative]
        config = yaml.safe_load(path.read_text())
        if path.name != "les.yaml":
            assert config["num_interactions"] == 2
            assert config["interaction_first"] == "RealAgnosticInteractionBlock"
            assert config["hidden_irreps"].replace(" ", "") == "128x0e+128x1o"
            assert config["keep_last_layer_irreps"] and not config["restart_latest"]
            config["enable_cueq"] = True
        (root / relative).write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copy2(path, root / "provenance" / path.name)
    for name in ("manifest.json", "smoke.json", "constructor_checks.json"):
        shutil.copy2(parent / name, root / "provenance" / ("parent_" + name))
    parity_model = (
        parent / "smoke/les_nl2/models/au2_mgo_matched_smoke_les_nl2_stagetwo.model"
    )
    smoke = json.loads((parent / "smoke.json").read_text())
    assert (
        digest(parity_model)
        == smoke["models"][str(parity_model.relative_to(parent))]["sha256"]
    )
    shutil.copy2(parity_model, root / "provenance/parity_reference.model")
    wet_manifest = json.loads((wetting / "manifest.json").read_text())
    for name in (
        "evaluate_adsorption_switch.py",
        "inputs/1-undoped.xyz",
        "inputs/3-undoped.xyz",
        "inputs/1-doped.xyz",
        "inputs/3-doped.xyz",
    ):
        assert digest(wetting / name) == wet_manifest["source_checksums"][name]
        shutil.copy2(wetting / name, root / name)
    for name in ("verify.py", "evaluate.py"):
        assert digest(parent / name) == original["inputs_and_source_checksums"][name]
        shutil.copy2(parent / name, root / name)
    verify_path = root / "verify.py"
    verify = verify_path.read_text()
    anchor = "assert args.seed == task['seed'] and args.r_max == 5.5 and args.batch_size == 10"
    assert verify.count(anchor) == 1
    verify = verify.replace(
        anchor,
        anchor
        + "\n        assert args.enable_cueq and args.interaction_first == 'RealAgnosticInteractionBlock'",
    )
    anchor = "assert model.les.remove_self_interaction and not model.les.use_atomwise"
    assert verify.count(anchor) == 1
    verify = verify.replace(
        anchor,
        anchor
        + "\n        assert model.les.ewald.norm_factor == 90.4756\n        assert [type(b).__name__ for b in model.interactions] == ['RealAgnosticInteractionBlock', 'RealAgnosticResidualInteractionBlock']",
    )
    verify_path.write_text(verify)
    crosscheck = scripts.parent / "les_interaction_crosscheck"
    shutil.copy2(crosscheck / "train_cueq.py", root / "train_cueq.py")
    parity = (crosscheck / "check_cueq.py").read_text()
    assert parity.count('(task,) = manifest["tasks"]') == 1
    parity = parity.replace(
        '(task,) = manifest["tasks"]', 'task = manifest["tasks"][0]'
    )
    (root / "check_cueq.py").write_text(parity)
    for name in ("prepare.py", "run.py", "job.slurm", "README.md"):
        shutil.copy2(scripts / name, root / name)
    versions = dict(original["package_versions"])
    for name in (
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
    ):
        versions[name] = importlib.metadata.version(name)
    manifest = dict(original)
    manifest.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        campaign_directory=str(root),
        parent_campaign=str(parent),
        backend="cueq",
        package_versions=versions,
        purpose="Fresh five-seed scalar-vector MACE2-LES fits for Tables VI and VII",
        changes_from_parent="Only enable_cueq=True; model, data, loss, epochs and first interaction block unchanged",
        prior_run_status="All five parent jobs timed out before final export; native checkpoints lack EMA and RNG history",
        expected_parameters=962336,
        wetting_parent=str(wetting),
        checkpoint_selection="Native stage-two minimum validation loss, then raw test and wetting evaluations",
    )
    manifest["inputs_and_source_checksums"] = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
