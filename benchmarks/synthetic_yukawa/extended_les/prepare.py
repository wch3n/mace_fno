"""Prepare isolated seed-17 static and MD fits at kappa=0.5 A^-1."""

import argparse
import ast
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml
from ase.io import read, write

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--parent", required=True, type=Path)
    args = ap.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    old = json.loads((parent / "manifest.json").read_text())
    for name, digest in old["sources"].items():
        assert sha(parent / name) == digest, name
    for name, digest in old["installed_sources"].items():
        assert sha(name) == digest, name
    base = root.parent.parent
    md = base / "yukawa_L32_md2800"
    root.mkdir(exist_ok=False)
    for name in (
        "data",
        "configs",
        "runs",
        "smoke",
        "logs",
        "results",
        "provenance",
        "md",
    ):
        (root / name).mkdir()
    (root / "venv").symlink_to((parent / "venv").resolve(), target_is_directory=True)
    shutil.copy2(parent / "manifest.json", root / "provenance/parent_manifest.json")
    shutil.copy2(
        parent / "smoke_passed.json", root / "provenance/parent_smoke_passed.json"
    )
    original = Path(old["tasks"][0]["train"]["config"])
    shutil.copy2(original, root / "provenance/parent_train.yaml")
    template = yaml.safe_load(original.read_text())
    shutil.copy2(
        parent / "configs/les_response.yaml", root / "configs/les_response.yaml"
    )
    for name in (
        "cueq_compat.py",
        "optimizer_compat.py",
        "train_response.py",
        "run_response.py",
    ):
        shutil.copy2(parent / name, root / name)
    for name in ("md_driver.py", "md_common.py", "md_szz_shells.py"):
        shutil.copy2(md / "md" / name, root / "provenance" / name)
    # Driver and completion checks are unchanged; the wrapper only selects cuEq.
    for name in ("md_driver.py", "md_common.py"):
        shutil.copy2(md / "md" / name, root / name)
    # Reuse the numerical routines without importing the legacy script's
    # top-level loop, which rewrites the manuscript-wide analysis directory.
    tree = ast.parse((md / "md/md_szz_shells.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("shell_vectors", "sk_frames", "mean_se")
    ]
    assert len(functions) == 3
    core = "import numpy as np\nN2 = [1,2,3,4,5,6,8,9,10,11,12,13,14,16,17]\n"
    core += "\n\n".join(ast.unparse(node) for node in functions)
    core += "\n\nVEC = shell_vectors()\n"
    (root / "szz_core.py").write_text(core)
    plot_data = HERE.parents[2] / "docs/source/data/md_structure/szz_shells.csv"
    shutil.copy2(plot_data, root / "provenance/baseline_szz.csv")
    tasks, inputs = [], {}
    published = {}
    for line in (HERE.parent / "checksums.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            published[name.lstrip("*")] = digest
    for index, dataset in enumerate(("static", "md")):
        source = (
            base / "yukawa_L32_kappa_production/data"
            if dataset == "static"
            else md / "data_k0p5"
        )
        data = root / "data" / dataset
        data.mkdir()
        files = dict(
            train="kappa0p5_train.xyz"
            if dataset == "static"
            else "kappa0p5_train_contact.xyz",
            valid="kappa0p5_valid.xyz",
            test="kappa0p5_test.xyz",
        )
        if dataset == "md":
            files["contact_test"] = "kappa0p5_contact_test.xyz"
        for split, filename in files.items():
            src, dst = source / filename, data / (split + ".xyz")
            digest = sha(src)
            key = ("static/" if dataset == "static" else "md2800/data_k0p5/") + filename
            assert digest == published[key], key
            dst.symlink_to(src)
            inputs[str(dst.relative_to(root))] = dict(source=str(src), sha256=digest)
        train = read(data / "train.xyz", ":")
        valid = read(data / "valid.xyz", ":")
        isolated = [a for a in train if len(a) == 1]
        thermal = [a for a in train if len(a) > 1]
        assert len(isolated) == 2 and len(thermal) == (
            660 if dataset == "static" else 810
        )
        assert len(valid) == 240 and all(len(a) == 1146 for a in thermal + valid)
        assert set(thermal[0].numbers.tolist()) == {11, 17}
        smoke = isolated + thermal[:8] + (thermal[-2:] if dataset == "md" else [])
        write(data / "smoke_train.xyz", smoke, format="extxyz")
        write(data / "smoke_valid.xyz", valid[:4], format="extxyz")
        for p in data.glob("smoke_*.xyz"):
            inputs[str(p.relative_to(root))] = dict(
                source="deterministic subset of the production split", sha256=sha(p)
            )
        name = f"les_response_k0p5_{dataset}_seed17"
        task = dict(
            index=index,
            name=name,
            dataset=dataset,
            seed=17,
            n_train=len(thermal),
            n_valid=240,
            n_test=300,
            contact_test=45 if dataset == "md" else 0,
        )
        for mode in ("train", "smoke"):
            directory = root / ("runs" if mode == "train" else "smoke") / name
            directory.mkdir()
            c = deepcopy(template)
            c.update(
                name=name,
                seed=17,
                restart_latest=False,
                max_num_epochs=1200,
                work_dir=str(directory),
                log_dir=str(directory / "logs"),
                model_dir=str(directory / "models"),
                checkpoints_dir=str(directory / "checkpoints"),
                results_dir=str(directory / "results"),
                train_file=str(
                    data / ("train.xyz" if mode == "train" else "smoke_train.xyz")
                ),
                valid_file=str(
                    data / ("valid.xyz" if mode == "train" else "smoke_valid.xyz")
                ),
                les_arguments=str(root / "configs/les_response.yaml"),
            )
            assert (
                c["interaction_first"] == "RealAgnosticInteractionBlock"
                and c["enable_cueq"]
            )
            if mode == "smoke":
                c.update(max_num_epochs=4, start_swa=2)
            cfg = root / "configs" / (name + "_" + mode + ".yaml")
            cfg.write_text(yaml.safe_dump(c, sort_keys=False))
            task[mode] = dict(
                config=str(cfg),
                directory=str(directory),
                model=str(directory / "models" / (name + "_stagetwo.model")),
            )
        tasks.append(task)
    for p in HERE.iterdir():
        if p.suffix in (".py", ".slurm", ".md"):
            shutil.copy2(p, root / p.name)
    manifest = dict(
        root=str(root),
        parent=str(parent),
        kappa_inv_A=0.5,
        seeds=[17],
        tasks=tasks,
        environment=old["environment"],
        installed_sources=old["installed_sources"],
        upstream_commits=old["upstream_commits"],
        inputs=inputs,
        interaction_first="RealAgnosticInteractionBlock",
        training=dict(
            epochs=1200,
            stage_two_start=450,
            patience=50,
            batch=4,
            selection="validation loss only; fresh fits, all response readouts optimized",
        ),
        md=dict(
            temperature_K=2800,
            dt_fs=2,
            friction_per_ps=2,
            equilibration_ps=2,
            production_ps=1000,
            save_fs=500,
            log_fs=10,
            seed=17,
            start_test_frame=0,
            charges="fixed reference +/-0.4 e, never learned latent sources",
            blocks=10,
            nve_ps=100,
            cell_A=32,
            atoms=1146,
        ),
        comparison_scope="Single seed; existing ordinary LES MD differs in first block and local representation. No causal attribution to induced response alone.",
        sources={
            str(p.relative_to(root)): sha(p)
            for p in [
                *root.glob("*.py"),
                *root.glob("*.slurm"),
                *root.glob("*.md"),
                *(root / "configs").glob("*.yaml"),
                *(root / "provenance").iterdir(),
            ]
        },
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(root=str(root), tasks=tasks, md=manifest["md"]), indent=2))


if __name__ == "__main__":
    main()
