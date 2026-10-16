"""Check exported architecture, backend parity, forces, and held-out accuracy."""

import json
import re
import sys

import numpy as np
import torch
import yaml
from ase.io import read
from mace.calculators import MACECalculator
from train_cueq import install
from verify import ROOT, check_model, digest, dump


def main(mode, index):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    task = manifest["tasks"][index]
    output = ROOT / (
        f"smoke/{task['dataset']}/cueq" if mode == "check" else task["run_dir"]
    )
    (path,) = (output / "models").glob("*_stagetwo.model")
    torch.set_default_dtype(torch.float64)
    install(yaml.safe_load((ROOT / "configs/les.yaml").read_text()))
    native = MACECalculator(
        model_paths=str(path), device="cuda", default_dtype="float64"
    )
    info = check_model(native.models[0], task)
    accelerated = MACECalculator(
        model_paths=str(path), device="cuda", default_dtype="float64", enable_cueq=True
    )
    assert accelerated.models[0].interactions[0].cueq_config.enabled
    frames = [
        read(ROOT / "data" / task["dataset"] / f"{s}.xyz", 0)
        for s in ("valid", "contact_test")
    ]
    errors = []
    for a in frames:
        a.calc = native
        en, fn = a.get_potential_energy(), a.get_forces().copy()
        a.calc = accelerated
        ec, fc = a.get_potential_energy(), a.get_forces().copy()
        np.testing.assert_allclose(en, ec, atol=1e-8, rtol=0)
        np.testing.assert_allclose(fn, fc, atol=1e-8, rtol=1e-9)
        errors.append(
            dict(energy_eV=abs(ec - en), force_eV_A=float(np.abs(fc - fn).max()))
        )
    a = frames[0].copy()
    a.calc = accelerated
    force = a.get_forces()
    atom, axis = np.unravel_index(np.argmax(np.abs(force)), force.shape)
    h = 1e-5
    original = a.positions[atom, axis]
    a.positions[atom, axis] = original + h
    plus = a.get_potential_energy()
    a.positions[atom, axis] = original - h
    minus = a.get_potential_energy()
    fd = abs(-(plus - minus) / (2 * h) - force[atom, axis])
    assert fd < 2e-5, fd
    report = dict(
        status="passed",
        model=str(path),
        model_sha256=digest(path),
        **info,
        backend_parity=errors,
        finite_difference_force_error_eV_A=float(fd),
        test={},
    )
    del accelerated
    torch.cuda.empty_cache()
    if mode != "check":
        for split in ("valid", "test", "contact_test"):
            frames = read(ROOT / "data" / task["dataset"] / (split + ".xyz"), ":")
            assert len(frames) == manifest["counts"][task["dataset"]][split]
            de, df = [], []
            for a in frames:
                e, f = a.get_potential_energy(), a.get_forces().copy()
                a.calc = native
                de.append(1000 * (a.get_potential_energy() - e) / len(a))
                df.append(1000 * (a.get_forces() - f).ravel())
            de, df = np.array(de), np.concatenate(df)
            assert np.isfinite(de).all() and np.isfinite(df).all()
            report["test"][split] = dict(
                n=len(frames),
                energy_RMSE_meV_atom=float(np.sqrt(np.mean(de**2))),
                force_RMSE_meV_A=float(np.sqrt(np.mean(df**2))),
            )
        epochs = re.findall(r"Epoch\s+(\d+)", (output / "training.log").read_text())
        report["last_logged_epoch"] = max(map(int, epochs)) if epochs else None
        report["epoch_budget"] = 1200
        report["selection"] = (
            "Validation-selected stage-two model; MD and test data never used for selection."
        )
    dump(output / "evaluation.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
