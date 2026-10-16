"""Prepare a matched seed-17 fully periodic 3D Au2/MgO experiment."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

PARENT = Path(
    "/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs/_campaigns/au2_mgo_full_zmix_nl1_5seed_20261006T070300Z"
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--z-grid", type=int, default=24)
    parser.add_argument("--z-modes", type=int, default=8)
    args = parser.parse_args()
    assert args.z_grid >= 4 and 1 <= args.z_modes <= args.z_grid // 2
    root = args.campaign.resolve()
    assert not root.exists(), root
    parent = json.loads((PARENT / "manifest.json").read_text())
    original_path = PARENT / "configs/au2_mgo_nl1_joint_full_z_seed17.yaml"
    original = yaml.safe_load(original_path.read_text())
    root.mkdir(parents=True)
    for name in ("configs", "references", "models", "data", "logs", "runs", "smoke"):
        (root / name).mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for folder in ("source/src", "mace_source"):
        shutil.copytree(PARENT / folder, root / folder, ignore=ignore)
    inputs = {}
    for rel in (
        "models/mace_nl1.model",
        "data/train.xyz",
        "data/validation.xyz",
        "data/test.xyz",
        "data/smoke_train.xyz",
        "data/smoke_validation.xyz",
    ):
        src = PARENT / rel
        checksum = digest(src)
        if rel in parent["inputs"]:
            assert checksum == parent["inputs"][rel]["sha256"], rel
        shutil.copy2(src, root / rel)
        inputs[rel] = dict(source=str(src), sha256=checksum)
    shutil.copy2(original_path, root / "references/full_z_seed17.yaml")
    shutil.copy2(Path(__file__), root / "prepare.py")
    for name in ("train.py", "measure_train.py", "evaluate.py"):
        shutil.copy2(PARENT / name, root / name)
    runner = (PARENT / "run.py").read_text().replace("full-height", "periodic-3D")
    runner = runner.replace("cp['z_mixing']=='full'", "cp['spatial_scheme']=='3d'")
    runner = runner.replace(
        "assert cp['completed_steps']==task['steps']",
        f"assert cp['completed_steps']==task['steps']\n        assert cp['z_grid_size']=={args.z_grid} and tuple(cp['n_modes'])==({args.z_modes},8,8) and cp['spectral_groups']==1",
    )
    (root / "run.py").write_text(runner)
    cfg = dict(original)
    cfg.update(
        mace_model=str(root / "models/mace_nl1.model"),
        train_file=str(root / "data/train.xyz"),
        validation_file=str(root / "data/validation.xyz"),
        spatial_scheme="3d",
        grid=24,
        z_grid=args.z_grid,
        modes=8,
        z_modes=args.z_modes,
        z_extent=None,
        z_mixing="local",
        lateral_interlacing=1,
        volume_interlacing=2,
        interlacing_training="full",
        seed=17,
        test_file=None,
        train_cache=None,
        validation_cache=None,
        test_cache=None,
        resume=None,
        init_from=None,
    )
    tasks, smoke = [], []
    for mode, collection in (("train", tasks), ("smoke", smoke)):
        name = "au2_mgo_nl1_joint_periodic_3d_seed17" + (
            "_smoke" if mode == "smoke" else ""
        )
        checkpoint = root / ("smoke" if mode == "smoke" else "runs") / name / "model.pt"
        config = dict(
            cfg,
            checkpoint=str(checkpoint),
            last_checkpoint=str(checkpoint.with_name("model.last.pt")),
        )
        if mode == "smoke":
            config.update(
                steps=16,
                eval_interval=8,
                checkpoint_interval=8,
                mace_warmup_steps=2,
                train_file=str(root / "data/smoke_train.xyz"),
                validation_file=str(root / "data/smoke_validation.xyz"),
            )
        path = root / "configs" / (name + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        collection.append(
            dict(
                index=0,
                name=name,
                system="au2_mgo",
                depth=1,
                opt_level="joint",
                seed=17,
                steps=config["steps"],
                config=str(path),
                checkpoint=str(checkpoint),
                test_file=str(
                    root
                    / "data"
                    / ("smoke_validation.xyz" if mode == "smoke" else "test.xyz")
                ),
            )
        )
    job = """#!/usr/bin/env bash
#SBATCH --job-name=au-3d-fno
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
CAMPAIGN_ROOT=${1:?campaign directory}
MODE=${2:-train}
module --force purge
module load EasyBuild/2024a Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
export LD_LIBRARY_PATH="$EBROOTPYTHON/lib:${LD_LIBRARY_PATH:-}"
ulimit -s unlimited
export PYTHONPATH="${CAMPAIGN_ROOT}/source/src:${CAMPAIGN_ROOT}/mace_source"
export PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$CAMPAIGN_ROOT"
nvidia-smi
exec python3 -B -u run.py "$MODE" 0
"""
    (root / "job.slurm").write_text(job)
    manifest = dict(
        root=str(root),
        parent=str(PARENT),
        tasks=tasks,
        smoke_tasks=smoke,
        inputs=inputs,
        reference_config_sha256=digest(original_path),
        changes={
            key: dict(before=original.get(key), after=value)
            for key, value in cfg.items()
            if original.get(key) != value
        },
        geometry="Fully periodic original 9.04743 x 9.04743 x 26.45886 A cell, including vacuum.",
        grid_xyz=[24, 24, args.z_grid],
        modes_xyz=[8, 8, args.z_modes],
        interlacing="Eight volumetric origins, compared with four lateral origins for the slab.",
        protocol="Same data, MACE1 backbone, seed 17, 16 channels, G=1, four nonlinear layers, loss, optimizer and 240000-step cap as the full-z run.",
        batching="Batch one, four microbatches per optimizer step; float64; no cuEquivariance.",
        selection="Minimum validation loss. Test set excluded from optimization and model selection.",
        scope="Exploratory 3D comparison, not a replacement for manuscript results.",
    )
    manifest["source_hashes"] = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".slurm")
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Fully periodic 3D Au2/MgO test\n\n"
        f"Joint MACE1 nonlinear FNO, seed 17, G=1. Grid 24 x 24 x {args.z_grid} in physical x/y/z order, "
        f"mode counts 8 x 8 x {args.z_modes}, eight interlaced origins. The unchanged cell includes vacuum. "
        "The 16-step smoke test must pass before submitting the matched 240000-step training run. "
        "No paper results are replaced. See manifest.json for exact changes and hashes.\n"
    )
    print(root)


if __name__ == "__main__":
    main()
