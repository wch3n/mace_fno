"""Evaluate both first-block choices on identical frozen test data and probes."""

import json
import subprocess
import sys
import time

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator
from verify import ROOT, check_model, digest, dump


def expected_control(task):
    hetero = task["dataset"] == "hetscreen"
    filename = "heterogeneous_metrics.json" if hetero else "homogeneous_metrics.json"
    records = json.loads((ROOT / "provenance" / filename).read_text())
    if hetero:
        rows = [
            r
            for r in records
            if r["model"] == "mace1_les" and r["seed"] == task["seed"]
        ]
    else:
        rows = [
            r
            for r in records
            if r["kappa"] == task["dataset"].removeprefix("kappa")
            and r["seed"] == task["seed"]
            and r["split"] == "test"
        ]
    (row,) = rows
    if hetero:
        assert row["model_sha256"] == task["original_model_sha256"]
    return row


def main(index):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    task = manifest["tasks"][index]
    output = ROOT / task["run_dir"] / "evaluation"
    output.mkdir(exist_ok=False)
    (model,) = (ROOT / task["run_dir"] / "models").glob("*_stagetwo.model")
    control = ROOT / task["control_model"]
    assert digest(control) == task["original_model_sha256"]
    results = {}
    for label, path in (("nonresidual", model), ("residual", control)):
        start = time.time()
        calc = MACECalculator(
            model_paths=str(path), device="cuda", default_dtype="float64"
        )
        info = check_model(calc.models[0], task, control=label == "residual")
        result = dict(model=str(path), model_sha256=digest(path), **info, splits={})
        for split in ("valid", "test") if label == "nonresidual" else ("test",):
            frames = [
                a
                for a in read(ROOT / "data" / task["dataset"] / f"{split}.xyz", ":")
                if a.info.get("config_type") != "IsolatedAtom"
            ]
            assert len(frames) == manifest["counts"][task["dataset"]][split]
            energies, forces = [], []
            for a in frames:
                e, f = a.get_potential_energy(), a.get_forces()
                a.calc = calc
                energies.append((a.get_potential_energy() - e) / len(a) * 1000)
                forces.append((a.get_forces() - f).ravel() * 1000)
            de, df = np.asarray(energies), np.concatenate(forces)
            assert np.isfinite(de).all() and np.isfinite(df).all()
            result["splits"][split] = dict(
                n_frames=len(frames),
                energy_rmse_meV_atom=float(np.sqrt(np.mean(de**2))),
                force_rmse_meV_A=float(np.sqrt(np.mean(df**2))),
                energy_mean_error_meV_atom=float(de.mean()),
            )
            np.savez_compressed(
                output / f"{label}_{split}_errors.npz",
                energy_error_meV_atom=de,
                force_error_meV_A=df,
            )
        result["seconds"] = time.time() - start
        results[label] = result
        dump(output / f"{label}.json", result)
        del calc
        torch.cuda.empty_cache()
    expected = expected_control(task)
    reproduced = results["residual"]["splits"]["test"]
    for field, old in (
        ("energy_rmse_meV_atom", "e_rmse"),
        ("force_rmse_meV_A", "f_rmse"),
        ("energy_mean_error_meV_atom", "e_me"),
    ):
        np.testing.assert_allclose(
            reproduced[field], expected[old], atol=1e-5, rtol=1e-7
        )
    comparison = dict(
        dataset=task["dataset"],
        seed=task["seed"],
        residual_metrics_reproduced=True,
        results=results,
        differences={},
    )
    for field in ("energy_rmse_meV_atom", "force_rmse_meV_A"):
        old = results["residual"]["splits"]["test"][field]
        new = results["nonresidual"]["splits"]["test"][field]
        comparison["differences"][field] = dict(
            absolute=new - old, percent=100 * (new - old) / old
        )
    dump(output / "comparison.json", comparison)
    if task["dataset"] == "hetscreen":
        probe = output / "probe_models.json"
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-u",
                str(ROOT / "eval_probe2.py"),
                "--configs",
                str(ROOT / "data/probe.xyz"),
                "--reference",
                str(ROOT / "data/probe_reference.json"),
                "--mace-model",
                str(model),
                "--model",
                f"{task['name']}=mace:{model}",
                "--out",
                str(probe),
            ],
            check=True,
        )
        obs = json.loads(probe.read_text())[task["name"]]["obs"]
        assert len(obs) == 40 * 8 and all(
            np.isfinite(np.asarray(v)).all() for v in obs.values()
        )
        old = json.loads((ROOT / "provenance/probe_residual.json").read_text())
        record = json.loads(probe.read_text())
        record[f"residual_seed{task['seed']}"] = old[f"les_s{task['seed']}"]
        dump(output / "probe_paired.json", record)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-u",
                str(ROOT / "analyze_probe2.py"),
                "--reference",
                str(ROOT / "data/probe_reference.json"),
                "--models",
                str(output / "probe_paired.json"),
                "--out",
                str(output / "probe_analysis.json"),
            ],
            check=True,
        )
    print(json.dumps(comparison["differences"]), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
