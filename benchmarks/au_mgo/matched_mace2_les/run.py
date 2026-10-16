"""Gate five matched MACE2-LES fits on GPU parity and two-stage export checks."""

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone

from verify import ROOT, digest, dump, verify


def check_wetting(output):
    import numpy as np
    from ase.io import read

    result = json.loads((output / "results.json").read_text())
    assert set(result["records"]) == {
        f"{i}-{s}" for i in (1, 3) for s in ("undoped", "doped")
    }
    checks = {}
    for name, record in result["records"].items():
        initial = read(ROOT / "inputs" / f"{name}.xyz")
        final = read(output / f"{name}-relaxed.xyz")
        moving, fixed = record["movable_indices"], record["fixed_indices"]
        assert len(moving) == 2 and all(initial[i].symbol == "Au" for i in moving)
        assert len(fixed) == 108
        assert np.array_equal(initial.numbers, final.numbers)
        assert np.array_equal(initial.pbc, final.pbc)
        assert np.array_equal(initial.cell.array, final.cell.array)
        assert np.allclose(
            initial.positions[fixed], final.positions[fixed], atol=5e-8, rtol=0
        )
        normal = np.cross(final.cell[0], final.cell[1])
        normal /= np.linalg.norm(normal)
        delta = final.get_distance(*moving, mic=True, vector=True)
        angle = float(
            np.degrees(
                np.arcsin(np.clip(abs(delta @ normal) / np.linalg.norm(delta), 0, 1))
            )
        )
        basin = angle < 45 if record["geometry"] == "wetting" else angle > 45
        checks[name] = dict(
            converged=record["converged"] and record["final_fmax_eV_per_A"] <= 0.01,
            expected_orientation=bool(basin),
            orientation_angle_from_plane_deg=angle,
            substrate_fixed=True,
            cell_fixed=True,
        )
    clean = all(c["converged"] and c["expected_orientation"] for c in checks.values())
    dump(
        output / "checks.json", dict(all_converged_expected_basins=clean, states=checks)
    )
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "train"))
    parser.add_argument("index", type=int, nargs="?", default=0, choices=range(5))
    args = parser.parse_args()
    manifest = verify()
    task = manifest["tasks"][args.index]
    validation = args.mode == "validate"
    output = ROOT / ("validation" if validation else task["run_dir"])
    output.mkdir(exist_ok=False)
    status = dict(
        state="running",
        stage="preflight",
        task=task,
        job_id=os.environ["SLURM_JOB_ID"],
        started=datetime.now(timezone.utc).isoformat(),
        commands=[],
    )

    def save():
        dump(output / "status.json", status)

    def command(stage, argv):
        status["stage"] = stage
        record = dict(stage=stage, command=[str(x) for x in argv])
        status["commands"].append(record)
        save()
        log = output / f"{stage}.log"
        with log.open("x") as handle:
            result = subprocess.run(
                [sys.executable, "-B", "-u", *map(str, argv)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
        record["exit_code"] = result.returncode
        save()
        if result.returncode:
            print(log.read_text()[-14000:], flush=True)
            raise RuntimeError(f"{stage} failed: {result.returncode}")

    save()
    try:
        import torch

        assert torch.cuda.is_available()
        status.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        if validation:
            command("preflight", ["verify.py", "--preflight"])
            command("backend_parity", ["check_cueq.py"])
            command(
                "smoke_training",
                ["train_cueq.py", "--config", "configs/smoke_les_nl2.yaml"],
            )
            command("smoke_verification", ["verify.py", "--smoke"])
            status.update(state="complete", stage="validated")
        else:
            assert (
                json.loads((ROOT / "validation/status.json").read_text())["state"]
                == "complete"
            )
            assert (
                json.loads((ROOT / "backend_parity.json").read_text())["state"]
                == "passed"
            )
            assert json.loads((ROOT / "smoke.json").read_text())["state"] == "passed"
            command("training", ["train_cueq.py", "--config", task["config"]])
            epochs = re.findall(r"Epoch (\d+):", (output / "training.log").read_text())
            assert epochs and int(epochs[-1]) == 999, epochs[-3:]
            status["completed_epochs"] = 1000
            command("evaluation", ["evaluate.py", str(args.index)])
            metrics = json.loads((output / "evaluation/metrics.json").read_text())
            assert (
                metrics["architecture"]["parameters"] == manifest["expected_parameters"]
            )
            status["model_sha256"] = metrics["model_sha256"]
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
                    ROOT / "inputs",
                    "--output-dir",
                    output / "wetting",
                    "--fmax",
                    "0.01",
                    "--steps",
                    "500",
                ],
            )
            clean = check_wetting(output / "wetting")
            assert digest(metrics["model"]) == metrics["model_sha256"]
            status.update(
                state="complete" if clean else "needs_review",
                stage="complete",
                wetting_checks_passed=clean,
            )
    except Exception as error:
        status.update(
            state="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
