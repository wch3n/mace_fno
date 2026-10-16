"""Validate, fit and run matched contact-trained nonresidual LES MD."""

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

from verify import ROOT, digest, dump, verify


def main(mode, index):
    manifest = verify()
    task = manifest["tasks"][index]
    output = ROOT / (
        f"validation/{task['dataset']}" if mode == "check" else task["run_dir"]
    )
    output.mkdir(parents=True, exist_ok=False)
    status = dict(
        state="running",
        stage="preflight",
        job_id=os.environ.get("SLURM_JOB_ID"),
        started=datetime.now(timezone.utc).isoformat(),
        task=task["name"],
        commands=[],
    )

    def command(stage, argv):
        status["stage"] = stage
        record = dict(stage=stage, argv=list(map(str, argv)))
        status["commands"].append(record)
        dump(output / "status.json", status)
        with (output / f"{stage}.log").open("x") as stream:
            result = subprocess.run(
                [sys.executable, "-B", "-u", *map(str, argv)],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        record["exit_code"] = result.returncode
        if result.returncode:
            print(
                "\n".join((output / f"{stage}.log").read_text().splitlines()[-50:]),
                flush=True,
            )
            raise subprocess.CalledProcessError(result.returncode, argv)
        print(stage, "passed", flush=True)

    try:
        import torch

        assert torch.cuda.is_available()
        status["gpu"] = torch.cuda.get_device_name()
        if mode == "check":
            command(
                "native_smoke",
                [
                    "-m",
                    "mace.cli.run_train",
                    "--config",
                    task["smoke_configs"]["native"],
                ],
            )
            command("backend_parity", ["check_cueq.py", index])
            command(
                "cueq_smoke",
                ["train_cueq.py", "--config", task["smoke_configs"]["cueq"]],
            )
            command("smoke_export", ["verify.py", "--smoke", index])
            command("model_validation", ["evaluate.py", mode, index])
            (model,) = (ROOT / "smoke" / task["dataset"] / "cueq/models").glob(
                "*_stagetwo.model"
            )
            command(
                "md_driver",
                [
                    "md_wrapper.py",
                    "equilibrate",
                    "--calc",
                    "mace",
                    "--model",
                    model,
                    "--kappa",
                    task["kappa"],
                    "--time",
                    "0.02",
                    "--dt",
                    "2",
                    "--seed",
                    "17",
                    "--out",
                    output / "md_driver",
                ],
            )
            from md_common import run_complete

            assert run_complete(str(output / "md_driver"))[0]
            command("szz_estimator", ["analyze.py", "check", index])
        else:
            checked = json.loads(
                (ROOT / "validation" / task["dataset"] / "status.json").read_text()
            )
            assert checked["state"] == "complete" and checked[
                "manifest_sha256"
            ] == digest(ROOT / "manifest.json")
            command("training", ["train_cueq.py", "--config", task["config"]])
            command("evaluation", ["evaluate.py", mode, index])
            (model,) = (output / "models").glob("*_stagetwo.model")
            status["model_sha256"] = digest(model)
            common = [
                "--calc",
                "mace",
                "--model",
                model,
                "--kappa",
                task["kappa"],
                "--dt",
                "2",
                "--temp",
                "2800",
                "--gamma",
                "2",
                "--seed",
                "17",
                "--log-fs",
                "10",
            ]
            md = ROOT / "md" / task["dataset"]
            command(
                "equilibration",
                [
                    "md_wrapper.py",
                    "equilibrate",
                    *common,
                    "--start-index",
                    "0",
                    "--time",
                    "2",
                    "--out",
                    md / "equil",
                ],
            )
            command(
                "nvt",
                [
                    "md_wrapper.py",
                    "run",
                    *common,
                    "--state",
                    md / "equil/state.npz",
                    "--ensemble",
                    "nvt",
                    "--time",
                    "1000",
                    "--save-fs",
                    "500",
                    "--out",
                    md / "nvt_1ns",
                ],
            )
            command("szz", ["analyze.py", "analyze", index])
        status.update(
            state="complete",
            stage="complete",
            manifest_sha256=digest(ROOT / "manifest.json"),
        )
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        dump(output / "status.json", status)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
