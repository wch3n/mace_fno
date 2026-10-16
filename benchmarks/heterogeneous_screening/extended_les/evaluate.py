"""Validate smoke models, or evaluate the unchanged test set and 40-set probe."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import iread, read
from mace.calculators import MACECalculator

ROOT = Path(__file__).resolve().parent


def main():
    mode, index = sys.argv[1], int(sys.argv[2])
    manifest = json.loads((ROOT / "manifest.json").read_text())
    task = manifest["tasks"][index]
    run = task[mode]
    directory = Path(run["directory"])
    calc = MACECalculator(
        model_paths=run["model"], device="cuda", default_dtype="float64"
    )
    model = calc.models[0]
    if manifest.get("interaction_first"):
        assert type(model.interactions[0]).__name__ == manifest["interaction_first"]
        assert [str(p.linear.irreps_out) for p in model.products] == ["128x0e+128x1o"]
        assert (
            sum(p.numel() for p in model.parameters())
            == manifest["expected_parameters"]
        )
    for attr, key in [
        ("use_dipoles", "use_dipole"),
        ("use_quads", "use_quad"),
        ("use_induced_charges", "use_induced_charge"),
        ("use_induced_dipoles", "use_induced_dipole"),
    ]:
        assert getattr(model, attr) == task["flags"][key], (attr, task["name"])
    result = dict(
        name=task["name"],
        total_parameters=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(),
        mode=mode,
    )
    torch.cuda.reset_peak_memory_stats()
    if mode == "smoke":
        atoms = read(ROOT / "data/smoke_valid.xyz", 0)
        atoms.calc = calc
        forces = atoms.get_forces()
        assert np.isfinite(forces).all()
        delta = 1e-4
        energies = []
        for sign in (-1, 1):
            shifted = atoms.copy()
            shifted.positions[0, 0] += sign * delta
            shifted.calc = calc
            energies.append(shifted.get_potential_energy())
        fd_force = -(energies[1] - energies[0]) / (2 * delta)
        assert np.isfinite(energies).all()
        error = abs(fd_force - forces[0, 0])
        assert error < 1e-5 + 1e-3 * abs(forces[0, 0]), (fd_force, forces[0, 0], error)
        result.update(
            fd_force_error_eV_A=error,
            max_force_eV_A=float(np.max(np.abs(forces))),
            peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
        )
        (directory / "smoke_check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return

    energies, forces = [], []
    start = time.time()
    for atoms in iread(ROOT / "data/test.xyz"):
        ref_e, ref_f = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = calc
        energies.append((atoms.get_potential_energy() - ref_e) / len(atoms))
        forces.append((atoms.get_forces() - ref_f).ravel())
    assert len(energies) == 300, len(energies)
    de, df = np.asarray(energies), np.concatenate(forces)
    assert np.isfinite(de).all() and np.isfinite(df).all()
    result.update(
        test_structures=len(de),
        energy_RMSE_meV_atom=float(1e3 * np.sqrt(np.mean(de**2))),
        force_RMSE_meV_A=float(1e3 * np.sqrt(np.mean(df**2))),
        signed_energy_error_meV_atom=float(1e3 * de.mean()),
        test_seconds=time.time() - start,
        peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
    )
    (directory / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    del calc, model
    torch.cuda.empty_cache()
    output = directory / "probe_models.json"
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "eval_probe2.py"),
            "--configs",
            str(ROOT / "data/probe.xyz"),
            "--reference",
            str(ROOT / "data/probe_reference.json"),
            "--mace-model",
            run["model"],
            "--model",
            f"{task['name']}=mace:{run['model']}",
            "--out",
            str(output),
        ],
        check=True,
    )
    obs = json.loads(output.read_text())[task["name"]]["obs"]
    assert len(obs) == 40 * 8
    assert all(np.isfinite(v).all() for v in map(np.asarray, obs.values()))
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "analyze_probe2.py"),
            "--reference",
            str(ROOT / "data/probe_reference.json"),
            "--models",
            str(output),
            "--out",
            str(directory / "probe_analysis.json"),
        ],
        check=True,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
