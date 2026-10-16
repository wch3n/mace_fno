"""GPU parity, one-step optimization, coverage and training-step timing."""

import sys
import time
from copy import deepcopy

import numpy as np
import torch
import yaml
from ase.io import read
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, build_default_arg_parser
from mace.tools.scripts_utils import get_optimizer, get_params_options
from mace.tools.torch_geometric.dataloader import DataLoader
from train_cueq import install
from verify import ROOT, check_model, digest, dump, verify


def main():
    manifest = verify()
    assert torch.cuda.is_available()
    task = manifest["tasks"][int(sys.argv[1])]
    torch.set_default_dtype(torch.float64)
    install(yaml.safe_load((ROOT / "configs/les.yaml").read_text()))
    from mace.cli.convert_cueq_e3nn import run as to_e3nn
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    from mace.tools.scripts_utils import extract_config_mace_model

    (reference_path,) = (ROOT / "smoke" / task["dataset"] / "native/models").glob(
        "*_stagetwo.model"
    )
    saved = torch.load(reference_path, map_location="cpu", weights_only=False)
    # Native stage-two exports can retain requires_grad=False from stage-one
    # evaluation. Rebuild the same weights with the constructor's training flags,
    # preserving genuinely frozen zero parameters in symmetric contractions.
    reference = type(saved)(**extract_config_mace_model(saved))
    reference.load_state_dict(saved.state_dict(), strict=True)
    del saved
    check_model(reference, task)
    accelerated = to_cueq(deepcopy(reference), device="cuda")
    assert accelerated.interactions[0].cueq_config.enabled
    assert any(
        type(m).__module__.startswith("cuequivariance") for m in accelerated.modules()
    )
    assert (
        accelerated.les.remove_self_interaction
        and accelerated.les.ewald.norm_factor == 90.4756
    )
    restored = to_e3nn(deepcopy(accelerated), device="cuda").to("cuda")
    check_model(restored, task)
    reference = reference.to("cuda")
    structures = read(ROOT / "data" / task["dataset"] / "smoke_valid.xyz", ":")
    table = AtomicNumberTable(reference.atomic_numbers.tolist())

    def batch(frames):
        graphs = [
            AtomicData.from_config(
                config_from_atoms(a),
                z_table=table,
                cutoff=float(reference.r_max),
                heads=reference.heads,
            )
            for a in frames
        ]
        return (
            next(iter(DataLoader(graphs, batch_size=len(graphs)))).to("cuda").to_dict()
        )

    errors = {}

    def compare(model, label, atol=1e-8):
        for index in (0, 3):
            ref = reference(batch([structures[index]]), training=False)
            out = model(batch([structures[index]]), training=False)
            for key in ("energy", "forces", "les_energy", "latent_charges"):
                # Absolute tolerance for total energies avoids hiding errors behind atomic offsets.
                torch.testing.assert_close(
                    out[key], ref[key], rtol=0 if key == "energy" else 1e-9, atol=atol
                )
                errors[f"{label}/{index}/{key}"] = float(
                    (out[key] - ref[key]).abs().max().detach()
                )

    compare(accelerated, "cueq")
    compare(restored, "roundtrip")
    del restored
    args = build_default_arg_parser().parse_args(["--config", task["config"]])
    coverage, timings = {}, {}

    def training_loss(model, frames):
        out = model(batch(frames), training=True)
        target_e = torch.tensor(
            [a.get_potential_energy() for a in frames], device="cuda"
        )
        target_f = torch.tensor(
            np.concatenate([a.get_forces() for a in frames]), device="cuda"
        )
        n = torch.tensor([len(a) for a in frames], device="cuda")
        return ((out["energy"] - target_e) / n).square().mean() + 100 * (
            out["forces"] - target_f
        ).square().mean()

    for label, model in (("e3nn", reference), ("cueq", accelerated)):
        optimizer = get_optimizer(args, get_params_options(args, model))
        included = [id(p) for g in optimizer.param_groups for p in g["params"]]
        needed = {id(p) for p in model.parameters() if p.requires_grad}
        missing = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and id(p) not in included
        ]
        assert len(included) == len(set(included)) and needed <= set(included), (
            label,
            missing,
        )
        coverage[label] = dict(
            all_trainable_parameters_included=True,
            parameters=sum(p.numel() for p in model.parameters()),
            les_readout_parameters=sum(
                p.numel() for p in model.les_readouts.parameters()
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        training_loss(model, [structures[0], structures[3]]).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()
    compare(accelerated, "after_one_adam_step", atol=1e-7)

    # Batch size 4 matches production; graph construction is excluded from timings.
    frames = [structures[i % len(structures)] for i in range(4)]
    for label, model in (("e3nn", reference), ("cueq", accelerated)):
        samples = []
        for _ in range(5):
            graph = batch(frames)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(graph, training=True)
            loss = out["les_energy"].square().mean() + out["forces"].square().mean()
            loss.backward()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        timings[label] = float(np.median(samples[1:]))
    record = dict(
        state="passed",
        gpu=torch.cuda.get_device_name(),
        dtype="float64",
        reference_sha256=digest(reference_path),
        max_absolute_errors=errors,
        optimizer_coverage=coverage,
        batch_size=4,
        training_step_seconds=timings,
        measured_step_speedup=timings["e3nn"] / timings["cueq"],
    )
    dump(ROOT / "validation" / task["dataset"] / "backend_parity.json", record)
    print(record, flush=True)


if __name__ == "__main__":
    main()
