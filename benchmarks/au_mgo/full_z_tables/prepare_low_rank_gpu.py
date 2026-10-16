"""Prepare the joint seed-17 dense/rank-32/rank-64 GPU cost comparison."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-campaign", type=Path, required=True)
    parser.add_argument("--low-rank-campaign", type=Path, required=True)
    parser.add_argument("--gpu-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dense, low, template, root = [
        p.resolve()
        for p in [
            args.dense_campaign,
            args.low_rank_campaign,
            args.gpu_template,
            args.output_dir,
        ]
    ]
    previous = json.loads((template / "manifest.json").read_text())
    lm = json.loads((low / "manifest.json").read_text())
    root.mkdir(exist_ok=False)
    for directory in ["models", "data", "logs", "measurements"]:
        (root / directory).mkdir()
    for name, expected in lm["checksums"].items():
        assert digest(low / name) == expected, name
    for directory in ["source", "mace_source"]:
        shutil.copytree(
            low / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ["measure_gpu.py", "run_gpu.py", "job.slurm"]:
        assert digest(template / name) == previous["snapshot_hashes"][name]
        shutil.copy2(template / name, root / name)
    # Extend only the accounting of spectral parameters to include both factors.
    p = root / "measure_gpu.py"
    text = p.read_text()
    old = 'name.endswith("radial_weight")'
    assert text.count(old) == 1
    p.write_text(
        text.replace(
            old,
            'name.endswith(("radial_weight", "radial_input_factor", "radial_output_factor"))',
        )
    )
    p = root / "job.slurm"
    text = p.read_text()
    assert 'export PYTHONPATH="${COMPARISON_ROOT}/source/src"' in text
    p.write_text(
        text.replace(
            'export PYTHONPATH="${COMPARISON_ROOT}/source/src"',
            'export PYTHONPATH="${COMPARISON_ROOT}/source/src:${COMPARISON_ROOT}/mace_source"',
        ).replace("#SBATCH --time=02:00:00", "#SBATCH --time=00:15:00")
    )
    shutil.copy2(low / "models/mace_nl1.model", root / "models/mace_nl1.model")
    frames = Path(previous["gpu"]["frames"])
    assert (
        digest(frames) == previous["snapshot_hashes"][str(frames.relative_to(template))]
    )
    shutil.copy2(frames, root / "data/benchmark_frames.xyz")
    records = []
    for parent in [dense, low]:
        manifest = json.loads((parent / "manifest.json").read_text())
        for task in manifest["tasks"]:
            if (task["regime"], task["seed"]) != ("joint", 17):
                continue
            name = task["name"]
            path = root / "models" / f"{name}.pt"
            shutil.copy2(task["checkpoint"], path)
            assert digest(path) == task["checkpoint_sha256"]
            assert manifest["checksums"]["models/mace_nl1.model"] == digest(
                root / "models/mace_nl1.model"
            )
            records.append(
                dict(
                    name=name,
                    regime="joint",
                    seed=17,
                    spectral_rank=task.get("spectral_rank", 0),
                    checkpoint=str(path),
                    checkpoint_provenance=dict(sha256=task["checkpoint_sha256"]),
                    mace_model=str(root / "models/mace_nl1.model"),
                )
            )
    assert sorted(r["spectral_rank"] for r in records) == [0, 32, 64]
    protocol = dict(previous["gpu"], frames=str(root / "data/benchmark_frames.xyz"))
    shutil.copy2(__file__, root / "prepare_low_rank_gpu.py")
    manifest = dict(
        records=records,
        gpu=protocol,
        template=str(template),
        accuracy_parents=[str(dense), str(low)],
        source_parent=str(low),
        snapshot_hashes={
            str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()
        },
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
