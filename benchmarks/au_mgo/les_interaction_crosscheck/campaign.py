"""Prepare and run one Au2/MgO LES fit changing only interaction_first."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from verify import FIRST, digest, dump

STATES = ("1-undoped", "3-undoped", "1-doped", "3-doped")


def load(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    import importlib.metadata

    import yaml

    root, parent, wetting, published = (
        p.resolve() for p in (args.root, args.parent, args.wetting, args.published)
    )
    original = load(parent / "manifest.json")
    control = parent / "runs/les_nl1_seed17"
    assert load(control / "status.json")["state"] == "complete"
    metrics = load(control / "evaluation/metrics.json")
    assert digest(metrics["model"]) == metrics["model_sha256"]
    root.mkdir(parents=True, exist_ok=False)
    for directory in (
        "configs",
        "provenance",
        "logs",
        "runs",
        "smoke",
        "runtime",
        "inputs",
    ):
        (root / directory).mkdir()
    # Verify each copied parent source/data file against its immutable manifest.
    for directory in ("source", "data"):
        shutil.copytree(
            parent / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for path in (root / directory).rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(root))
                assert digest(path) == original["inputs_and_source_checksums"][rel], rel
    for name in ("les.yaml", "les_nl1_seed17.yaml", "smoke_les_nl1.yaml"):
        rel = f"configs/{name}"
        assert digest(parent / rel) == original["inputs_and_source_checksums"][rel]
        shutil.copy2(parent / rel, root / rel)
    shutil.copy2(
        root / "configs/les_nl1_seed17.yaml", root / "provenance/control_config.yaml"
    )
    for name in ("les_nl1_seed17", "smoke_les_nl1"):
        path = root / f"configs/{name}.yaml"
        config = yaml.safe_load(path.read_text())
        config["name"] = config["name"].replace("matched", "residual_first")
        config["interaction_first"] = FIRST
        if args.enable_cueq:
            config["enable_cueq"] = True
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    sources = {
        "control_metrics.json": control / "evaluation/metrics.json",
        "control_wetting.json": wetting
        / "results/au2_mgo_MACE1_LES_seed17/endpoints/results.json",
        "published_metrics.json": published / "results/test_metrics.json",
        "published_wetting.json": published / "results/wetting/results.json",
        "parent_manifest.json": parent / "manifest.json",
    }
    for name, path in sources.items():
        shutil.copy2(path, root / "provenance" / name)
    old_wetting = load(wetting / "manifest.json")
    for name in [f"inputs/{s}.xyz" for s in STATES] + ["evaluate_adsorption_switch.py"]:
        assert digest(wetting / name) == old_wetting["source_checksums"][name]
        if name.startswith("inputs/"):
            assert digest(wetting / name) == digest(published / name)
        shutil.copy2(wetting / name, root / name)
    # Reuse the original held-out evaluator byte-for-byte.
    assert (
        digest(parent / "evaluate.py")
        == original["inputs_and_source_checksums"]["evaluate.py"]
    )
    shutil.copy2(parent / "evaluate.py", root / "evaluate.py")
    for name in (
        "campaign.py",
        "verify.py",
        "job.slurm",
        "README.md",
        "train_cueq.py",
        "check_cueq.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), root / name)
    versions = dict(original["package_versions"])
    if args.enable_cueq:
        assert args.parity_reference is not None
        shutil.copy2(args.parity_reference, root / "provenance/parity_reference.model")
        for package in (
            "cuequivariance",
            "cuequivariance-torch",
            "cuequivariance-ops-torch-cu12",
        ):
            versions[package] = importlib.metadata.version(package)
    task = dict(original["tasks"][0])
    task["name"] = task["name"].replace("matched", "residual_first")
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        tasks=[task],
        parent_campaign=str(parent),
        wetting_campaign=str(wetting),
        published_campaign=str(published),
        purpose="Isolate the first interaction block; not a full published-protocol reproduction",
        architecture_setting_changed={
            "interaction_first": {
                "control": "RealAgnosticInteractionBlock",
                "crosscheck": FIRST,
            }
        },
        backend="cueq" if args.enable_cueq else "e3nn",
        parity_reference=str(args.parity_reference) if args.enable_cueq else None,
        counts=original["counts"],
        package_versions=versions,
        checkpoint_selection=original["checkpoint_selection"],
        unchanged_les=dict(remove_self_interaction=True, ewald_norm_factor=90.4756),
        expected_parameters=266496,
        control_model_sha256=metrics["model_sha256"],
        inputs_and_source_checksums={
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    )
    dump(root / "manifest.json", manifest)
    print(root)


def check_wetting(root, output):
    import numpy as np
    from ase.io import read

    result = load(output / "results.json")
    assert set(result["records"]) == set(STATES)
    checks = {}
    for name, rec in result["records"].items():
        initial = read(root / "inputs" / f"{name}.xyz")
        final = read(output / f"{name}-relaxed.xyz")
        moving, fixed = rec["movable_indices"], rec["fixed_indices"]
        assert len(moving) == 2 and all(initial[i].symbol == "Au" for i in moving)
        assert (
            len(fixed) == 108
            and rec["converged"]
            and rec["final_fmax_eV_per_A"] <= 0.01
        )
        assert np.array_equal(initial.numbers, final.numbers)
        assert np.array_equal(initial.pbc, final.pbc) and np.array_equal(
            initial.cell.array, final.cell.array
        )
        assert np.allclose(
            initial.positions[fixed], final.positions[fixed], atol=5e-8, rtol=0
        )
        normal = np.cross(final.cell[0], final.cell[1])
        normal /= np.linalg.norm(normal)
        d = final.get_distance(*moving, mic=True, vector=True)
        angle = float(
            np.degrees(np.arcsin(np.clip(abs(d @ normal) / np.linalg.norm(d), 0, 1)))
        )
        assert angle < 45 if rec["geometry"] == "wetting" else angle > 45
        checks[name] = dict(
            converged=True,
            substrate_and_cell_fixed=True,
            orientation_angle_from_plane_deg=angle,
        )
    dump(output / "checks.json", checks)
    return result


def run(root, validation_only=False):
    from verify import verify

    root = root.resolve()
    os.chdir(root)
    manifest = verify()
    (task,) = manifest["tasks"]
    output = root / ("cueq_validation" if validation_only else task["run_dir"])
    output.mkdir(exist_ok=False)
    status = dict(
        state="running",
        stage="preflight",
        job_id=os.environ.get("SLURM_JOB_ID"),
        started=datetime.now(timezone.utc).isoformat(),
        commands=[],
    )

    def command(stage, argv):
        status["stage"] = stage
        record = dict(stage=stage, argv=argv)
        status["commands"].append(record)
        dump(output / "status.json", status)
        with (output / f"{stage}.log").open("x") as handle:
            proc = subprocess.run(
                [sys.executable, "-B", "-u", *argv],
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=root,
            )
        record["exit_code"] = proc.returncode
        if proc.returncode:
            print(
                "\n".join((output / f"{stage}.log").read_text().splitlines()[-60:]),
                flush=True,
            )
            raise subprocess.CalledProcessError(proc.returncode, argv)
        print(stage, "passed", flush=True)

    try:
        import torch

        assert torch.cuda.is_available()
        dump(
            root / "runtime/hardware.json",
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                python=sys.version,
            ),
        )
        accelerated = manifest.get("backend") == "cueq"
        trainer = ["train_cueq.py"] if accelerated else ["-m", "mace.cli.run_train"]
        if validation_only or not accelerated:
            command("preflight", ["verify.py", "--preflight"])
            if accelerated:
                command("backend_parity", ["check_cueq.py"])
            command("smoke_train", [*trainer, "--config", "configs/smoke_les_nl1.yaml"])
            command("smoke_verify", ["verify.py", "--smoke"])
        if validation_only:
            assert accelerated
            status.update(state="complete", stage="validated")
            return
        if accelerated:
            assert load(root / "cueq_validation/status.json")["state"] == "complete"
            assert load(root / "backend_parity.json")["state"] == "passed"
            assert load(root / "smoke.json")["state"] == "passed"
        command("training", [*trainer, "--config", task["config"]])
        command("evaluation", ["evaluate.py", "0"])
        metrics = load(output / "evaluation/metrics.json")
        command(
            "wetting",
            [
                "evaluate_adsorption_switch.py",
                "--model",
                metrics["model"],
                "--model-kind",
                "mace",
                "--device",
                "cuda",
                "--dtype",
                "float64",
                "--structures-dir",
                str(root / "inputs"),
                "--output-dir",
                str(output / "wetting"),
                "--fmax",
                ".01",
                "--steps",
                "500",
            ],
        )
        wetting = check_wetting(root, output / "wetting")
        assert digest(metrics["model"]) == metrics["model_sha256"]
        rows = []
        for label, test, endpoints in (
            ("residual_first_seed17", metrics["splits"]["test"], wetting),
            (
                "nonresidual_first_seed17",
                load(root / "provenance/control_metrics.json")["splits"]["test"],
                load(root / "provenance/control_wetting.json"),
            ),
            (
                "published_checkpoint",
                load(root / "provenance/published_metrics.json"),
                load(root / "provenance/published_wetting.json"),
            ),
        ):
            rows.append(
                dict(
                    model=label,
                    energy_rmse_meV_per_atom=1000 * test["energy_rmse"],
                    force_rmse_meV_per_A=1000 * test["force_rmse"],
                    undoped_delta_meV=endpoints["summary"]["undoped"][
                        "predicted_delta_meV"
                    ],
                    doped_delta_meV=endpoints["summary"]["doped"][
                        "predicted_delta_meV"
                    ],
                )
            )
        dump(
            root / "comparison.json",
            dict(
                rows=rows,
                backend=manifest.get("backend", "e3nn"),
                interpretation="The seed17 fits compare interaction_first; accelerated runs additionally validate backend equivalence. The published checkpoint also differs in LES settings.",
            ),
        )
        status.update(
            state="complete", stage="complete", model_sha256=metrics["model_sha256"]
        )
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        dump(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    for name in ("parent", "wetting", "published"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--enable-cueq", action="store_true")
    p.add_argument("--parity-reference", type=Path)
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    p = sub.add_parser("validate-cueq")
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else run(
        args.root, validation_only=args.mode == "validate-cueq"
    )
