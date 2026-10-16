"""Prepare matched contact-trained charge-only LES and Fig. 4 MD replacements."""

import argparse
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ase.io import read, write

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root, parent, runs):
    root, parent, runs = root.resolve(), parent.resolve(), runs.resolve()
    previous = json.loads((parent / "manifest.json").read_text())
    for name, digest in previous["inputs_and_source_checksums"].items():
        if name.startswith("source/") or name in (
            "verify.py",
            "train_cueq.py",
            "check_cueq.py",
            "configs/les.yaml",
        ):
            assert sha(parent / name) == digest, name
    root.mkdir(exist_ok=False)
    for name in (
        "configs",
        "provenance",
        "data",
        "runs",
        "smoke",
        "validation",
        "logs",
        "results",
        "md",
    ):
        (root / name).mkdir()
    shutil.copytree(
        parent / "source",
        root / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("verify.py", "train_cueq.py", "check_cueq.py", "configs/les.yaml"):
        shutil.copy2(parent / name, root / name)
    shutil.copy2(parent / "manifest.json", root / "provenance/parent_manifest.json")
    base = runs / "yukawa_L32_md2800"
    for name in ("md_driver.py", "md_common.py"):
        shutil.copy2(base / "md" / name, root / name)
    source = base / "md/md_szz_shells.py"
    shutil.copy2(source, root / "provenance/md_szz_shells.py")
    tree = ast.parse(source.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("shell_vectors", "sk_frames", "mean_se")
    ]
    assert len(functions) == 3
    core = "import numpy as np\nN2 = [1,2,3,4,5,6,8,9,10,11,12,13,14,16,17]\n"
    (root / "szz_core.py").write_text(
        core
        + "\n\n".join(ast.unparse(n) for n in functions)
        + "\n\nVEC = shell_vectors()\n"
    )
    shutil.copy2(
        REPO / "docs/source/data/md_structure/szz_shells.csv",
        root / "provenance/baseline_szz.csv",
    )
    shutil.copy2(
        REPO / "docs/source/data/md_structure/szz_shells_source.csv",
        root / "provenance/baseline_raw.csv",
    )
    checksums = {
        line.split(maxsplit=1)[1]: line.split()[0]
        for line in (HERE.parent / "checksums.sha256").read_text().splitlines()
        if line.strip()
    }
    tasks, counts, origins = [], {}, {}
    for index, kappa in enumerate(("0p05", "0p5")):
        dataset = f"kappa{kappa}"
        old = base / f"kappa_{kappa}/maceles1200c_seed17"
        name = f"les_nonres_contact_{dataset}_seed17"
        data = root / "data" / dataset
        data.mkdir()
        counts[dataset] = {}
        frames = {}
        for split, suffix in [
            ("train", "train_contact"),
            ("valid", "valid"),
            ("test", "test"),
            ("contact_test", "contact_test"),
        ]:
            src = base / f"data_k{kappa}" / f"{dataset}_{suffix}.xyz"
            assert sha(src) == checksums[f"md2800/data_k{kappa}/{src.name}"]
            dst = data / f"{split}.xyz"
            dst.symlink_to(src)
            origins[str(dst.relative_to(root))] = str(src)
            frames[split] = read(dst, ":")
            counts[dataset][split] = sum(len(a) > 1 for a in frames[split])
        assert counts[dataset] == dict(train=810, valid=240, test=300, contact_test=45)
        isolated = [a for a in frames["train"] if len(a) == 1]
        normal = [a for a in frames["train"] if len(a) > 1]
        assert len(isolated) == 2 and {len(a) for a in normal} == {1146}
        write(data / "smoke_train.xyz", isolated + normal[:8] + normal[-2:])
        write(data / "smoke_valid.xyz", frames["valid"][:4])
        config = yaml.safe_load((old / "config.yaml").read_text())
        shutil.copy2(old / "config.yaml", root / "provenance" / f"{name}.yaml")
        (control,) = [
            p
            for p in (old / "models").glob("*_stagetwo.model")
            if "compiled" not in p.name
        ]
        archived = root / "provenance" / f"{name}_residual.model"
        shutil.copy2(control, archived)
        output = root / "runs" / name
        config.update(
            name=name,
            work_dir=str(output),
            interaction_first="RealAgnosticInteractionBlock",
            enable_cueq=True,
            restart_latest=False,
            les_arguments=str(root / "configs/les.yaml"),
        )
        for key, child in [
            ("log_dir", "logs"),
            ("model_dir", "models"),
            ("checkpoints_dir", "checkpoints"),
            ("results_dir", "results"),
        ]:
            config[key] = str(output / child)
        for split in ("train", "valid", "test"):
            config[split + "_file"] = str(data / (split + ".xyz"))
        path = root / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        task = dict(
            index=index,
            name=name,
            dataset=dataset,
            kappa=kappa,
            seed=17,
            config=str(path.relative_to(root)),
            run_dir=str(output.relative_to(root)),
            old_directory=str(old),
            original_model=str(control),
            original_model_sha256=sha(control),
            control_model=str(archived.relative_to(root)),
            expected_parameters=278528,
            expected_control_parameters=180224,
            smoke_configs={},
        )
        for backend in ("native", "cueq"):
            smoke_dir = root / "smoke" / dataset / backend
            smoke = dict(
                config,
                name=f"smoke_{dataset}_{backend}",
                work_dir=str(smoke_dir),
                max_num_epochs=2,
                start_swa=1,
                enable_cueq=backend == "cueq",
                train_file=str(data / "smoke_train.xyz"),
                valid_file=str(data / "smoke_valid.xyz"),
            )
            smoke.pop("test_file")
            for key, child in [
                ("log_dir", "logs"),
                ("model_dir", "models"),
                ("checkpoints_dir", "checkpoints"),
                ("results_dir", "results"),
            ]:
                smoke[key] = str(smoke_dir / child)
            path = root / "configs" / f"smoke_{dataset}_{backend}.yaml"
            path.write_text(yaml.safe_dump(smoke, sort_keys=False))
            task["smoke_configs"][backend] = str(path.relative_to(root))
        tasks.append(task)
    for source in HERE.iterdir():
        if source.suffix in (".py", ".slurm", ".md"):
            shutil.copy2(source, root / source.name)
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        root=str(root),
        repo=str(REPO),
        parent=str(parent),
        tasks=tasks,
        counts=counts,
        original_data=origins,
        package_versions=previous["package_versions"],
        purpose="Replace only the ordinary LES curves in Fig. 4 with matched nonresidual contact-trained models",
        protocol=dict(max_num_epochs=1200, start_swa=450, patience=50, batch_size=4),
        checkpoint_selection="Validation-selected stage-two native export; no test or MD selection",
        md=dict(
            temperature_K=2800,
            dt_fs=2,
            friction_per_ps=2,
            equilibration_ps=2,
            production_ps=1000,
            save_fs=500,
            log_fs=10,
            seed=17,
            start_index=0,
            blocks=10,
            atoms=1146,
            cell_A=32,
            physical_charge_e=0.4,
        ),
        inputs_and_source_checksums={
            str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(root=str(root), tasks=[t["name"] for t in tasks], counts=counts),
            indent=2,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    args = ap.parse_args()
    prepare(args.root, args.parent, args.runs_root)
