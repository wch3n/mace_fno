"""Matched Table VII relaxations for completed, validation-selected linear FNOs."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

SEEDS = [17, 29, 41, 53, 67]
STATES = ["1-undoped", "3-undoped", "1-doped", "3-doped"]


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root, parent, gpu, wetting):
    a, b, old = [load(p / "manifest.json") for p in (parent, gpu, wetting)]
    for prefix in ("source/", "mace_source/"):
        ah = {k: v for k, v in a["source_hashes"].items() if k.startswith(prefix)}
        bh = {k: v for k, v in b["source_hashes"].items() if k.startswith(prefix)}
        assert ah and ah == bh, "Training source snapshots differ"
        for name, expected in ah.items():
            assert digest(parent / name) == digest(gpu / name) == expected, name
    tasks = [
        dict(t, training_root=str(parent))
        for t in a["tasks"]
        if t["system"] == "au2_mgo" and t["index"] <= 50
    ]
    tasks += [
        dict(t, training_root=str(gpu)) for t in b["tasks"] if t["system"] == "au2_mgo"
    ]
    tasks.sort(key=lambda t: (t["depth"], t["opt_level"], t["seed"]))
    assert len(tasks) == 20
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "inputs",
        "logs",
        "results",
        "analysis",
        "models",
        "configs",
        "provenance",
    ):
        (root / name).mkdir()
    for name in ("source", "mace_source"):
        shutil.copytree(
            parent / name,
            root / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for state in STATES:
        rel = f"inputs/{state}.xyz"
        assert digest(wetting / rel) == old["source_checksums"][rel]
        shutil.copy2(wetting / rel, root / rel)
    evaluator = "evaluate_adsorption_switch.py"
    assert digest(wetting / evaluator) == old["source_checksums"][evaluator]
    shutil.copy2(wetting / evaluator, root / evaluator)
    for filename in (
        "campaign.py",
        "collect.py",
        "submit.py",
        "job.slurm",
        "collect.slurm",
        "README.md",
    ):
        shutil.copy2(Path(__file__).with_name(filename), root / filename)
    for depth in (1, 2):
        rel = f"models/au2_mgo_nl{depth}.model"
        assert digest(parent / rel) == a["inputs"][rel]["sha256"]
        if depth == 2:
            assert digest(gpu / rel) == a["inputs"][rel]["sha256"]
        shutil.copy2(parent / rel, root / rel)
    for i, t in enumerate(tasks):
        t["training_index"], t["index"] = t["index"], i
        training = Path(t["training_root"])
        original_manifest = a if training == parent else b
        config = Path(t["config"])
        expected = original_manifest["source_hashes"][str(config.relative_to(training))]
        assert digest(config) == expected
        shutil.copy2(config, root / "configs" / (t["name"] + ".yaml"))
        t["config_sha256"] = expected
        t["backbone_sha256"] = digest(root / f"models/au2_mgo_nl{t['depth']}.model")
        t["test_sha256"] = digest(t["test_file"])
        assert t["test_sha256"] == a["inputs"]["data/au2_mgo/test.xyz"]["sha256"]
        status_path = Path(t["checkpoint"]).parent / "status.json"
        status = load(status_path) if status_path.exists() else {}
        t["ready_at_preparation"] = status.get("state") == "complete"
        if t["ready_at_preparation"]:
            assert digest(t["checkpoint"]) == status["checkpoint_sha256"]
            t["prepared_checkpoint_sha256"] = status["checkpoint_sha256"]
            t["training_dependency"] = None
        else:
            assert training == gpu and 22 <= t["training_index"] <= 28
            t["training_dependency"] = f"16191322_{t['training_index']}"
    for depth in (1, 2):
        for regime in ("frozen", "joint"):
            assert [
                t["seed"]
                for t in tasks
                if (t["depth"], t["opt_level"]) == (depth, regime)
            ] == SEEDS
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,
        parent=str(parent),
        gpu_parent=str(gpu),
        wetting_parent=str(wetting),
        seeds=SEEDS,
        fmax=0.01,
        maximum_steps=500,
        dtype="float64",
        selection="completed 240000-step fits; same best-validation-loss bytes as test evaluation",
        source_checksums={
            str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()
        },
    )
    save(root / "manifest.json", manifest)
    print(
        "Prepared",
        len(tasks),
        "models;",
        sum(t["ready_at_preparation"] for t in tasks),
        "ready",
    )


def verify(root):
    manifest = load(root / "manifest.json")
    for name, expected in manifest["source_checksums"].items():
        assert digest(root / name) == expected, name
    return manifest


def snapshot(root, task):
    """Require final training/test completion before copying a selected checkpoint."""
    import torch
    import yaml

    source = Path(task["checkpoint"])
    folder = source.parent
    status, test = load(folder / "status.json"), load(folder / "test.json")
    assert status["state"] == "complete" and task["steps"] == 240000
    assert test["checkpoint"] == str(source) and test["samples"] == 500
    assert (
        test["best_step"] == status["selected_step"]
        and not test["energy_shift_applied"]
    )
    assert digest(source) == status["checkpoint_sha256"]
    assert digest(task["config"]) == task["config_sha256"]
    assert digest(task["test_file"]) == task["test_sha256"]
    config = yaml.safe_load(Path(task["config"]).read_text())
    assert config["architecture"] == "linear" and config["spectral_groups"] == 1
    target = root / "models" / (task["name"] + ".pt")
    if not target.exists():
        shutil.copy2(source, target)
    assert digest(target) == digest(source) == status["checkpoint_sha256"]
    if "prepared_checkpoint_sha256" in task:
        assert digest(target) == task["prepared_checkpoint_sha256"]
    cp = torch.load(target, map_location="cpu", weights_only=False)
    assert cp["architecture"] == "linear" and cp["z_mixing"] == "spectral"
    assert cp["spectral_groups"] == 1 and cp["completed_steps"] == task["steps"]
    assert cp["seed"] == task["seed"] and cp["mace_training"] == task["opt_level"]
    assert bool(cp.get("mace_state_dict")) == (task["opt_level"] == "joint")
    assert (cp["channels"], cp["z_grid_size"], cp["lateral_interlacing"]) == (16, 24, 2)
    assert digest(cp["mace_model"]) == task["backbone_sha256"]
    history = [
        json.loads(line)
        for line in (folder / "validation_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert max(r["step"] for r in history) == task["steps"]
    best = min(r["validation_objective"] for r in history)
    selected = [r for r in history if r["step"] == cp["best_step"]][-1]
    assert math.isclose(
        best, cp["best_validation_objective"], rel_tol=2e-6, abs_tol=1e-10
    )
    assert math.isclose(
        best, selected["validation_objective"], rel_tol=2e-6, abs_tol=1e-10
    )
    assert cp["best_step"] == test["best_step"]
    provenance = dict(
        checkpoint=str(target),
        checkpoint_sha256=digest(target),
        source_checkpoint=str(source),
        completed_steps=cp["completed_steps"],
        selected_step=cp["best_step"],
        training_status=status,
        test=test,
        config=config,
        history=history,
        source_hashes={
            n: digest(folder / n)
            for n in ("status.json", "test.json", "validation_history.jsonl")
        },
    )
    save(root / "provenance" / (task["name"] + ".json"), provenance)
    return target, provenance


def worker(root, index):
    import evaluate_adsorption_switch as evaluator
    import mace
    import numpy as np
    import torch
    from ase.io import read

    import mace_fno

    manifest = verify(root)
    task = manifest["tasks"][index]
    assert torch.cuda.is_available()
    assert Path(mace_fno.__file__).resolve().is_relative_to(root / "source")
    assert Path(mace.__file__).resolve().is_relative_to(root / "mace_source")
    output = root / "results" / task["name"]
    output.mkdir(exist_ok=False)
    status = dict(
        state="running",
        task=task,
        job_id=os.environ.get("SLURM_JOB_ID"),
        started=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=digest(root / "manifest.json"),
    )
    save(output / "status.json", status)
    try:
        model, provenance = snapshot(root, task)
        args = argparse.Namespace(
            model=model,
            model_kind="mace-fno",
            device="cuda",
            dtype="float64",
            mace_head=None,
            movable_element="Au",
            energy_key="energy",
            fmax=0.01,
            steps=500,
            output_dir=output / "endpoints",
        )
        args.output_dir.mkdir()
        calc = evaluator.make_calculator(args)
        operators = [
            m
            for m in calc.model.modules()
            if type(m).__name__ == "LinearMetricSlabFNO2D"
        ]
        assert len(operators) == 1, "Expected one direct linear full-height operator"
        backbones = [
            m for m in calc.model.modules() if type(m).__name__ == "ScaleShiftMACE"
        ]
        assert len(backbones) == 1 and len(backbones[0].interactions) == task["depth"]
        atoms = read(root / "inputs/1-undoped.xyz")
        atoms.calc = calc
        i = evaluator.movable_indices(atoms, "Au")[0]
        force = float(atoms.get_forces()[i, 0])
        delta = 1e-4
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[i, 0] += delta
        minus.positions[i, 0] -= delta
        plus.calc = minus.calc = calc
        fd = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * delta)
        assert abs(force - fd) < 2e-5, (force, fd)
        save(
            output / "preflight.json",
            dict(
                state="passed",
                operator=type(operators[0]).__name__,
                mace_layers=task["depth"],
                force_fd_error_eV_A=abs(force - fd),
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                model_sha256=digest(model),
            ),
        )
        records, checks = {}, {}
        for state, composition, geometry in evaluator.STATE_DEFINITIONS:
            record = evaluator.relax_state(
                root / "inputs" / (state + ".xyz"),
                state,
                composition,
                geometry,
                calc,
                args,
            )
            records[state] = record
            initial = read(root / "inputs" / (state + ".xyz"))
            final = read(args.output_dir / (state + "-relaxed.xyz"))
            moving, fixed = record["movable_indices"], record["fixed_indices"]
            assert len(moving) == 2 and len(fixed) == 108 and len(final) == 110
            assert np.array_equal(initial.numbers, final.numbers)
            assert np.array_equal(initial.pbc, final.pbc)
            assert np.array_equal(initial.cell.array, final.cell.array)
            assert np.allclose(
                initial.positions[fixed], final.positions[fixed], atol=5e-8, rtol=0
            )
            normal = np.cross(final.cell[0], final.cell[1])
            normal /= np.linalg.norm(normal)
            distance = final.get_distance(*moving, mic=True, vector=True)
            angle = float(
                np.degrees(
                    np.arcsin(
                        np.clip(abs(distance @ normal) / np.linalg.norm(distance), 0, 1)
                    )
                )
            )
            checks[state] = dict(
                converged=record["converged"] and record["final_fmax_eV_per_A"] <= 0.01,
                expected_orientation=bool(
                    angle < 45 if geometry == "wetting" else angle > 45
                ),
                orientation_angle_from_plane_deg=angle,
                substrate_fixed=True,
                cell_fixed=True,
            )
        result = dict(
            model=str(model),
            checkpoint_sha256=provenance["checkpoint_sha256"],
            model_kind="mace-fno",
            device="cuda",
            dtype="float64",
            optimizer="ASE FIRE",
            fmax_eV_per_A=0.01,
            maximum_steps=500,
            movable_element="Au",
            records=records,
            summary=evaluator.summarize_switch(records),
        )
        save(args.output_dir / "results.json", result)
        clean = all(
            c["converged"] and c["expected_orientation"] for c in checks.values()
        )
        save(
            output / "checks.json",
            dict(all_converged_expected_basins=clean, states=checks),
        )
        assert clean, "Unconverged endpoint or changed orientation basin"
        assert (
            digest(model)
            == digest(task["checkpoint"])
            == provenance["checkpoint_sha256"]
        )
        status.update(
            state="complete", checks_passed=True, checkpoint_sha256=digest(model)
        )
        print(task["name"], result["summary"], flush=True)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        save(output / "status.json", status)


def ready(root):
    manifest = verify(root)
    indices = [t["index"] for t in manifest["tasks"] if t["ready_at_preparation"]]
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    def queue(device, assigned):
        for index in assigned:
            path = root / "logs" / f"evaluate-{index}.log"
            with path.open("x") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        str(root / "campaign.py"),
                        "worker",
                        str(root),
                        "--index",
                        str(index),
                    ],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(queue, device, indices[i :: len(devices)])
            for i, device in enumerate(devices)
        ]
        for future in futures:
            future.result()
    subprocess.run(
        [sys.executable, "-B", str(root / "collect.py"), str(root), "--allow-partial"],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "ready", "worker"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--wetting", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        prepare(root, args.parent.resolve(), args.gpu.resolve(), args.wetting.resolve())
    elif args.mode == "ready":
        ready(root)
    else:
        worker(root, args.index)


if __name__ == "__main__":
    main()
