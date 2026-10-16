"""Benchmark a 3D checkpoint, without publishing interim accuracy."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("parent", type=Path)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Interim production checkpoint; defaults to smoke weights.",
)
args = parser.parse_args()
root, parent = args.root.resolve(), args.parent.resolve()
repo = Path(__file__).resolve().parents[3]
scripts = repo / "docs/source/data/inference"
previous = json.loads((scripts / "timing_summary.json").read_text())["au2_mgo"]
training_manifest = json.loads((parent / "manifest.json").read_text())
checkpoint = (
    args.checkpoint.resolve()
    if args.checkpoint
    else parent / "smoke/au2_mgo_nl1_joint_periodic_3d_seed17_smoke/model.pt"
)
status = json.loads((checkpoint.parent / "status.json").read_text())
checkpoint_hash = digest(checkpoint)
if args.checkpoint is None:
    assert status["state"] == "complete"
    assert checkpoint_hash == status["checkpoint_sha256"]
assert not root.exists()
root.mkdir(parents=True)
(root / "models").mkdir()
source_stat = checkpoint.stat()
shutil.copy2(checkpoint, root / "models/model.pt")
assert digest(root / "models/model.pt") == checkpoint_hash, (
    "Checkpoint changed while copying"
)
assert digest(checkpoint) == checkpoint_hash, "Source checkpoint changed while copying"
assert checkpoint.stat().st_mtime_ns == source_stat.st_mtime_ns
history = checkpoint.parent / "validation_history.jsonl"
if history.exists():
    shutil.copy2(history, root / "validation_history_snapshot.jsonl")
ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
shutil.copytree(parent / "source/src", root / "source/src", ignore=ignore)
for package in ("mace", "les"):
    shutil.copytree(
        parent / "mace_source" / package, root / "source/src" / package, ignore=ignore
    )
for name in ("time_inference.py", "measure_memory.py"):
    shutil.copy2(scripts / name, root / name)
shutil.copy2(Path(__file__), root / "prepare_inference.py")
local = previous["models"]["MACE_1"]
assert digest(local["path"]) == local["sha256"]
assert digest(previous["test_file"]) == previous["test_sha256"]
models = {
    "MACE_1": dict(kind="mace", path=local["path"], expected_sha256=local["sha256"]),
    "MACE_1-FNO_3d_joint": dict(
        kind="fno",
        path=str(root / "models/model.pt"),
        expected_sha256=checkpoint_hash,
        mace_model=local["path"],
        expected_architecture="nonlinear",
        expected_spectral_groups=1,
        expected_fno_layers=4,
    ),
}
manifest = dict(
    cpu_threads=4,
    structures=3,
    warmup=5,
    rounds=3,
    repeats=10,
    expected_gpu="NVIDIA A100-SXM4-40GB",
    systems=dict(
        au2_mgo=dict(
            test_file=previous["test_file"],
            test_sha256=previous["test_sha256"],
            models=models,
        )
    ),
)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
job = (
    (scripts / "job_linear.slurm")
    .read_text()
    .replace("linear-inference", "au-3d-inference")
)
job = job.replace("water_scan llzo au2_mgo", "au2_mgo").replace(
    "MACE_1-FNO_linear_joint", "MACE_1-FNO_3d_joint"
)
(root / "job.slurm").write_text(job)
scope = (
    "Architecture cost using an interim production checkpoint; accuracy not converged. Not added to manuscript."
    if args.checkpoint
    else "Architecture cost only: 16-step smoke weights, not a converged accuracy result. Not added to manuscript."
)
provenance = dict(
    source_campaign=str(parent),
    source_status=status,
    checkpoint_path=str(checkpoint),
    checkpoint_sha256=checkpoint_hash,
    checkpoint_mtime_ns=source_stat.st_mtime_ns,
    scope=scope,
    grid_xyz=training_manifest["grid_xyz"],
    modes_xyz=training_manifest["modes_xyz"],
    mesh_origins=8,
    protocol="Same three test frames and GPU inference protocol as manuscript Table VIII.",
)
(root / "selection_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
(root / "source_sha256.json").write_text(
    json.dumps(
        {
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
        indent=2,
    )
    + "\n"
)
print(root)
