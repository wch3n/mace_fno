"""Run checked homogeneous extended-LES fits and their MD comparison."""

import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare import sha
from run_response import convergence

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify():
    m = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in m["sources"].items():
        assert sha(ROOT / name) == digest, name
    for name, record in m["inputs"].items():
        assert sha(ROOT / name) == record["sha256"], name
    for name, digest in m["installed_sources"].items():
        assert sha(name) == digest, name
    for package, version in m["environment"]["packages"].items():
        assert importlib.metadata.version(package) == version, package
    import les
    import mace

    for module in (mace, les):
        assert (
            Path(module.__file__).resolve()
            == Path(m["environment"][module.__name__ + "_path"]).resolve()
        )
    return m


def command(argv, log):
    print("RUN", argv, flush=True)
    with Path(log).open("x") as stream:
        subprocess.run(
            [sys.executable, "-B", "-u", *map(str, argv)],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def fit(m, task, mode):
    run = task[mode]
    directory = Path(run["directory"])
    path = directory / "status.json"
    assert not path.exists(), "Existing runs require explicit resume handling"
    record = dict(
        state="running",
        started=time.time(),
        job=os.environ.get("SLURM_JOB_ID"),
        dataset=task["dataset"],
        mode=mode,
        seed=task["seed"],
    )
    dump(path, record)
    try:
        command(
            [ROOT / "train_response.py", "--config", run["config"]],
            directory / "train.log",
        )
        assert Path(run["model"]).is_file()
        covered = json.loads((directory / "optimizer_coverage.json").read_text())
        updated = json.loads((directory / "first_response_updates.json").read_text())
        assert covered["all_trainable_parameters_covered"] and updated
        assert all(v["max_abs_update"] > 0 for v in updated.values())
        check = convergence(directory, 2 if mode == "smoke" else 450)
        dump(directory / "convergence.json", check)
        command(
            [ROOT / "evaluate.py", mode, task["index"]], directory / "evaluation.log"
        )
        record.update(
            state="complete", convergence=check, model_sha256=sha(run["model"])
        )
    except BaseException as exc:
        record.update(state="failed", error=repr(exc))
        raise
    finally:
        record["finished"] = time.time()
        dump(path, record)


def md(m, kind):
    task = m["tasks"][1]
    status = json.loads((Path(task["train"]["directory"]) / "status.json").read_text())
    assert status["state"] == "complete"
    model = task["train"]["model"]
    assert sha(model) == status["model_sha256"]
    common = [
        ROOT / "md_wrapper.py",
        "--calc",
        "mace",
        "--model",
        model,
        "--kappa",
        "0p5",
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
    equil = ROOT / "md/equil"
    if kind == "md":
        command(
            [
                common[0],
                "equilibrate",
                *common[1:],
                "--start-index",
                "0",
                "--time",
                "2",
                "--out",
                equil,
            ],
            ROOT / "md/equil.log",
        )
    else:
        from md_common import run_complete

        assert run_complete(str(equil))[0]
    name, ensemble, duration = (
        ("nvt_1ns", "nvt", "1000") if kind == "md" else ("nve_100ps", "nve", "100")
    )
    target = ROOT / "md" / name
    command(
        [
            common[0],
            "run",
            *common[1:],
            "--state",
            equil / "state.npz",
            "--ensemble",
            ensemble,
            "--time",
            duration,
            "--save-fs",
            "500",
            "--out",
            target,
        ],
        ROOT / "md" / (name + ".log"),
    )
    command([ROOT / "analyze.py", kind], ROOT / "md" / (name + "_analysis.log"))


def main():
    m = verify()
    mode = sys.argv[1]
    if mode == "check":
        for task in m["tasks"]:
            fit(m, task, "smoke")
        # Exercise the actual accelerated MD driver before long jobs become eligible.
        model = m["tasks"][1]["smoke"]["model"]
        command(
            [
                ROOT / "md_wrapper.py",
                "equilibrate",
                "--calc",
                "mace",
                "--model",
                model,
                "--kappa",
                "0p5",
                "--time",
                "0.02",
                "--dt",
                "2",
                "--seed",
                "17",
                "--out",
                ROOT / "smoke/md_driver",
            ],
            ROOT / "smoke/md_driver.log",
        )
        from md_common import run_complete

        assert run_complete(str(ROOT / "smoke/md_driver"))[0]
        command([ROOT / "analyze.py", "check"], ROOT / "smoke/szz_validation.log")
        dump(
            ROOT / "smoke_passed.json",
            dict(
                status="passed",
                manifest_sha256=sha(ROOT / "manifest.json"),
                checks="Both datasets: four epochs, stage two, all readout updates, native/cuEq parity, finite-difference forces; MD driver completion",
            ),
        )
    elif mode == "train":
        certificate = json.loads((ROOT / "smoke_passed.json").read_text())
        assert certificate["manifest_sha256"] == sha(ROOT / "manifest.json")
        fit(m, m["tasks"][int(sys.argv[2])], "train")
    elif mode in ("md", "nve"):
        md(m, mode)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
