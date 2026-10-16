"""Check exported response models and evaluate independent Yukawa test sets."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.io import iread, read
from cueq_compat import install
from mace.calculators import MACECalculator
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.dataloader import DataLoader

ROOT = Path(__file__).resolve().parent


def main():
    mode, index = sys.argv[1], int(sys.argv[2])
    m = json.loads((ROOT / "manifest.json").read_text())
    task = m["tasks"][index]
    run = task[mode]
    directory = Path(run["directory"])
    torch.set_default_dtype(torch.float64)
    install(yaml.safe_load((ROOT / "configs/les_response.yaml").read_text()))
    native = MACECalculator(
        model_paths=run["model"], device="cuda", default_dtype="float64"
    )
    calc = MACECalculator(
        model_paths=run["model"],
        device="cuda",
        default_dtype="float64",
        enable_cueq=True,
    )
    model = calc.models[0]
    assert model.interactions[0].cueq_config.enabled
    assert type(model.interactions[0]).__name__ == m["interaction_first"]
    assert model.atomic_numbers.tolist() == [11, 17]
    assert [str(p.linear.irreps_out) for p in model.products] == ["128x0e+128x1o"]
    for attr in (
        "use_dipoles",
        "use_quads",
        "use_induced_charges",
        "use_induced_dipoles",
        "use_anisotropic_polarizability",
    ):
        assert getattr(model, attr)
    data = ROOT / "data" / task["dataset"]
    frames = read(data / "smoke_valid.xyz", ":3")
    table = AtomicNumberTable(model.atomic_numbers.tolist())

    def graph(a):
        d = AtomicData.from_config(
            config_from_atoms(a),
            z_table=table,
            cutoff=float(model.r_max),
            heads=model.heads,
        )
        return next(iter(DataLoader([d], batch_size=1))).to("cuda").to_dict()

    differences = {}
    for i, atoms in enumerate(frames):
        ref = native.models[0](graph(atoms), training=False)
        out = model(graph(atoms), training=False)
        for key in (
            "energy",
            "forces",
            "les_energy",
            "latent_charges",
            "latent_dipoles",
            "latent_quads",
            "latent_kappas",
            "latent_alphas",
        ):
            torch.testing.assert_close(
                out[key], ref[key], atol=1e-8, rtol=1e-9 if key != "energy" else 0
            )
            differences[f"{i}/{key}"] = float(
                (out[key] - ref[key]).abs().max().detach()
            )
    atoms = frames[0].copy()
    atoms.calc = calc
    force = atoms.get_forces()[0, 0]
    energies = []
    for sign in (-1, 1):
        shifted = atoms.copy()
        shifted.positions[0, 0] += sign * 1e-4
        shifted.calc = calc
        energies.append(shifted.get_potential_energy())
    fd_error = abs(-(energies[1] - energies[0]) / 2e-4 - force)
    assert fd_error < 1e-5 + 1e-3 * abs(force), fd_error
    result = dict(
        status="passed",
        mode=mode,
        dataset=task["dataset"],
        seed=17,
        parameters=sum(p.numel() for p in model.parameters()),
        interaction_first=m["interaction_first"],
        parity_max_errors=differences,
        fd_force_error_eV_A=float(fd_error),
        gpu=torch.cuda.get_device_name(),
    )
    del native, ref, out
    torch.cuda.empty_cache()
    if mode == "train":
        result["test"] = {}
        for split, expected in [("test", 300)] + (
            [("contact_test", 45)] if task["dataset"] == "md" else []
        ):
            de, df = [], []
            start = time.time()
            for a in iread(data / (split + ".xyz")):
                e0, f0 = a.get_potential_energy(), a.get_forces()
                a.calc = calc
                de.append((a.get_potential_energy() - e0) / len(a))
                df.append((a.get_forces() - f0).ravel())
            de, df = np.asarray(de), np.concatenate(df)
            assert (
                len(de) == expected and np.isfinite(de).all() and np.isfinite(df).all()
            )
            result["test"][split] = dict(
                n=expected,
                energy_RMSE_meV_atom=float(1000 * np.sqrt(np.mean(de**2))),
                force_RMSE_meV_A=float(1000 * np.sqrt(np.mean(df**2))),
                seconds=time.time() - start,
            )
    (directory / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
