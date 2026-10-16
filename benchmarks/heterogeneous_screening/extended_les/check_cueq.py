"""GPU parity, force backpropagation and four-epoch training/export check."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.io import read
from cueq_compat import install
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.dataloader import DataLoader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int, choices=(1, 2))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    task = manifest["tasks"][args.index]
    directory = root / "cueq_validation" / task["name"]
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "passed.json").exists()
    assert torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    les_arguments = yaml.safe_load(Path(task["les_arguments"]).read_text())
    install(les_arguments)
    from mace.cli.convert_cueq_e3nn import run as to_e3nn
    from mace.cli.convert_e3nn_cueq import run as to_cueq

    reference = torch.load(
        task["smoke"]["model"], map_location="cpu", weights_only=False
    )
    accelerated = to_cueq(copy.deepcopy(reference), device="cuda")
    assert accelerated.interactions[0].cueq_config.enabled
    assert any(
        type(m).__module__.startswith("cuequivariance") for m in accelerated.modules()
    )
    restored = to_e3nn(copy.deepcopy(accelerated), device="cuda").to("cuda")
    reference = reference.to("cuda")
    structures = read(root / "data/smoke_valid.xyz", ":3")
    z_table = AtomicNumberTable(reference.atomic_numbers.tolist())
    cutoff = float(reference.r_max)
    heads = reference.heads

    def batch(atoms):
        graph = AtomicData.from_config(
            config_from_atoms(atoms),
            z_table=z_table,
            cutoff=cutoff,
            heads=heads,
        )
        return next(iter(DataLoader([graph], batch_size=1))).to("cuda").to_dict()

    keys = (
        "energy",
        "forces",
        "les_energy",
        "latent_charges",
        "latent_dipoles",
        "latent_quads",
        "latent_kappas",
        "latent_alphas",
    )
    errors = {}
    for label, model in [("cueq", accelerated), ("roundtrip", restored)]:
        for n, atoms in enumerate(structures):
            ref = reference(batch(atoms), training=False)
            out = model(batch(atoms), training=False)
            for key in keys:
                if ref[key] is None:
                    assert out[key] is None, key
                    continue
                torch.testing.assert_close(out[key], ref[key], rtol=1e-9, atol=1e-8)
                errors[f"{label}/{n}/{key}"] = float(
                    (out[key] - ref[key]).abs().max().detach()
                )
    print("PARITY", json.dumps(errors), flush=True)

    # Force-loss training differentiates twice through the GPU operations.
    timings = {}
    for label, model in [("e3nn", reference), ("cueq", accelerated)]:
        samples = []
        for n in range(5):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(batch(structures[0]), training=True)
            loss = (out["energy"] / len(structures[0])).square().mean() + out[
                "forces"
            ].square().mean()
            loss.backward()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
        timings[label] = float(np.median(samples[1:]))
    print("BACKPROP_SECONDS", json.dumps(timings), flush=True)

    # Numerical derivative of the accelerated energy on a validation structure.
    force = float(accelerated(batch(structures[0]), training=False)["forces"][0, 0])
    energies = []
    for sign in (-1, 1):
        atoms = structures[0].copy()
        atoms.positions[0, 0] += sign * 1e-4
        energies.append(
            float(accelerated(batch(atoms), training=False)["energy"].sum().detach())
        )
    fd_error = abs(-(energies[1] - energies[0]) / 2e-4 - force)
    assert fd_error < 1e-5 + 1e-3 * abs(force), fd_error
    del reference, accelerated, restored, out, ref, grads, loss
    torch.cuda.empty_cache()

    # Exercise the actual training CLI, stage-two switch, EMA and e3nn export.
    config = yaml.safe_load(Path(task["smoke"]["config"]).read_text())
    config.update(
        enable_cueq=True,
        work_dir=str(directory),
        log_dir=str(directory / "logs"),
        model_dir=str(directory / "models"),
        checkpoints_dir=str(directory / "checkpoints"),
        results_dir=str(directory / "results"),
    )
    config_path = directory / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with (directory / "train.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "train_cueq.py"),
                "--config",
                str(config_path),
            ],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    model_path = directory / "models" / f"{task['name']}_stagetwo.model"
    exported = torch.load(model_path, map_location="cuda", weights_only=False)
    for attr, key in [
        ("use_dipoles", "use_dipole"),
        ("use_quads", "use_quad"),
        ("use_induced_charges", "use_induced_charge"),
        ("use_induced_dipoles", "use_induced_dipole"),
    ]:
        assert getattr(exported, attr) == task["flags"][key], attr
    # Construct the reload graph without keeping the original model alive.
    graph = AtomicData.from_config(
        config_from_atoms(structures[0]),
        z_table=z_table,
        cutoff=float(exported.r_max),
        heads=exported.heads,
    )
    out = exported(next(iter(DataLoader([graph], batch_size=1))).to("cuda").to_dict())
    assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()
    record = dict(
        task=task["name"],
        status="passed",
        time=time.time(),
        gpu=torch.cuda.get_device_name(),
        packages={
            p: importlib.metadata.version(p)
            for p in ("cuequivariance", "cuequivariance-torch")
        },
        parity_max_errors=errors,
        fd_force_error_eV_A=fd_error,
        backprop_seconds=timings,
        config_sha256=sha(config_path),
        les_config_sha256=sha(task["les_arguments"]),
        exported_model_sha256=sha(model_path),
        source_hashes={
            name: sha(root / name)
            for name in ("cueq_compat.py", "train_cueq.py", "check_cueq.py")
        },
    )
    (directory / "passed.json").write_text(json.dumps(record, indent=2) + "\n")
    print("PASSED", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
