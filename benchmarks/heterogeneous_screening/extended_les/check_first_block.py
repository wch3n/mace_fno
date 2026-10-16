"""Validate cuEquivariance for nonresidual extended LES and all response heads."""

import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from ase.io import read
from cueq_compat import install as conversion_install
from mace.data import AtomicData, config_from_atoms
from mace.modules.blocks import RealAgnosticInteractionBlock
from mace.tools import AtomicNumberTable, build_default_arg_parser
from mace.tools.torch_geometric.dataloader import DataLoader
from optimizer_compat import install as optimizer_install
from run_response import ROOT, sha, verify


def main():
    manifest = verify()
    assert torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    task = manifest["tasks"][0]
    les_args = yaml.safe_load(Path(task["les_arguments"]).read_text())
    conversion_install(les_args)
    from mace.cli.convert_cueq_e3nn import run as to_e3nn
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    from mace.tools.scripts_utils import extract_config_mace_model

    saved = torch.load(
        ROOT / "provenance/residual_smoke.model", map_location="cpu", weights_only=False
    )
    assert (
        type(saved.interactions[0]).__name__ == "RealAgnosticResidualInteractionBlock"
    )
    config = extract_config_mace_model(saved)
    config["interaction_cls_first"] = RealAgnosticInteractionBlock
    # Fresh random parameters: the residual checkpoint provides constructor metadata only.
    reference = type(saved)(**config)
    old_shapes = {n: tuple(p.shape) for n, p in saved.named_parameters()}
    new_shapes = {n: tuple(p.shape) for n, p in reference.named_parameters()}
    assert old_shapes.keys() == new_shapes.keys()
    changed = [n for n in old_shapes if old_shapes[n] != new_shapes[n]]
    assert changed == ["interactions.0.skip_tp.weight"], changed
    assert (
        sum(p.numel() for p in reference.parameters())
        == manifest["expected_parameters"]
    )
    del saved
    accelerated = to_cueq(deepcopy(reference), device="cuda")
    assert accelerated.interactions[0].cueq_config.enabled
    assert any(
        type(m).__module__.startswith("cuequivariance") for m in accelerated.modules()
    )
    restored = to_e3nn(deepcopy(accelerated), device="cuda").to("cuda")
    reference = reference.to("cuda")
    for model in (reference, accelerated, restored):
        assert type(model.interactions[0]).__name__ == manifest["interaction_first"]
        assert (
            sum(p.numel() for p in model.parameters())
            == manifest["expected_parameters"]
        )
        for attr, key in (
            ("use_dipoles", "use_dipole"),
            ("use_quads", "use_quad"),
            ("use_induced_charges", "use_induced_charge"),
            ("use_induced_dipoles", "use_induced_dipole"),
        ):
            assert getattr(model, attr) == task["flags"][key]
    atoms = read(ROOT / "data/smoke_valid.xyz", ":3")
    table = AtomicNumberTable(reference.atomic_numbers.tolist())

    def graph(structure):
        data = AtomicData.from_config(
            config_from_atoms(structure),
            z_table=table,
            cutoff=float(reference.r_max),
            heads=reference.heads,
        )
        return next(iter(DataLoader([data], batch_size=1))).to("cuda").to_dict()

    errors = {}

    def compare(model, label, atol=1e-8):
        for i, a in enumerate(atoms):
            ref, out = (
                reference(graph(a), training=False),
                model(graph(a), training=False),
            )
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
                assert ref[key] is not None and out[key] is not None, key
                torch.testing.assert_close(
                    out[key], ref[key], rtol=0 if key == "energy" else 1e-9, atol=atol
                )
                errors[f"{label}/{i}/{key}"] = float(
                    (out[key] - ref[key]).abs().max().detach()
                )

    compare(accelerated, "cueq")
    compare(restored, "roundtrip")
    del restored
    parity_dir = ROOT / "optimizer_parity"
    parity_dir.mkdir(exist_ok=False)
    optimizer_install(parity_dir)
    from mace.tools.scripts_utils import get_optimizer, get_params_options

    args = build_default_arg_parser().parse_args(["--config", task["train"]["config"]])
    for label, model in (("native", reference), ("cueq", accelerated)):
        optimizer = get_optimizer(args, get_params_options(args, model))
        optimizer.zero_grad(set_to_none=True)
        out = model(graph(atoms[0]), training=True)
        energy = torch.tensor(atoms[0].get_potential_energy(), device="cuda")
        forces = torch.tensor(atoms[0].get_forces(), device="cuda")
        loss = ((out["energy"] - energy) / len(atoms[0])).square().mean() + 100 * (
            out["forces"] - forces
        ).square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()
        for name in ("optimizer_coverage.json", "first_response_updates.json"):
            (parity_dir / f"{label}_{name}").write_bytes(
                (parity_dir / name).read_bytes()
            )
    compare(accelerated, "after_one_optimizer_step", atol=1e-7)
    record = dict(
        status="passed",
        manifest_sha256=sha(ROOT / "manifest.json"),
        interaction_first=manifest["interaction_first"],
        parameters=manifest["expected_parameters"],
        changed_parameter_shapes=changed,
        gpu=torch.cuda.get_device_name(),
        errors=errors,
    )
    (ROOT / "backend_parity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
