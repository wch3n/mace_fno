"""Reconstruct a copied training checkpoint and evaluate the unchanged 40-set probe."""

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.io import read
from cueq_compat import install as install_conversion
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.dataloader import DataLoader
from optimizer_compat import install as install_optimizer

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


class ReconstructionComplete(Exception):
    """Stop the native CLI before any optimizer step or normal training export."""


def reconstruct(snapshot):
    from mace import tools
    from mace.cli import run_train
    from mace.cli.convert_cueq_e3nn import run as to_e3nn
    from mace.tools.train import evaluate

    original = yaml.safe_load((ROOT / "provenance/train.yaml").read_text())
    assert int(original["seed"]) == snapshot["seed"]
    les = yaml.safe_load((ROOT / "provenance/les.yaml").read_text())
    install_conversion(les)
    directory = ROOT / "reconstruction"
    directory.mkdir(exist_ok=False)
    # Rebuild through the original CLI so all data-dependent normalization is
    # recomputed from the production split, rather than inherited from a smoke fit.
    config = dict(original)
    config.update(
        name="interim_reconstruction",
        work_dir=str(directory),
        les_arguments=str(ROOT / "provenance/les.yaml"),
        restart_latest=False,
    )
    for key, child in (
        ("log_dir", "logs"),
        ("model_dir", "models"),
        ("checkpoints_dir", "checkpoints"),
        ("results_dir", "results"),
    ):
        config[key] = str(directory / child)
    for key in ("train_file", "valid_file"):
        config[key] = str(ROOT / "data" / Path(original[key]).name)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    install_optimizer(directory)
    # run_train imports these functions directly; refresh the corrected bindings.
    from mace.tools import scripts_utils

    run_train.get_params_options = scripts_utils.get_params_options
    run_train.get_optimizer = scripts_utils.get_optimizer
    run_train.run_cueq_to_e3nn = to_e3nn
    epoch = snapshot["epoch"]
    rows = [
        json.loads(line)
        for line in (ROOT / "provenance/training_metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    (expected,) = [
        r for r in rows if r.get("mode") == "eval" and r.get("epoch") == epoch
    ]
    checkpoint = torch.load(
        snapshot["checkpoint"], map_location="cpu", weights_only=False
    )
    assert set(checkpoint) == {"model", "optimizer", "lr_scheduler"}
    destination = ROOT / "interim.model"

    def capture(**kwargs):
        model = kwargs["model"]
        assert model.interactions[0].cueq_config.enabled
        model.load_state_dict(checkpoint["model"], strict=True)
        assert (
            type(model.interactions[0]).__name__
            == "RealAgnosticResidualInteractionBlock"
        )
        assert sum(p.numel() for p in model.parameters()) == 337920
        assert [str(p.linear.irreps_out) for p in model.products] == ["128x0e+128x1o"]
        for attr in (
            "use_dipoles",
            "use_quads",
            "use_induced_charges",
            "use_induced_dipoles",
            "use_anisotropic_polarizability",
        ):
            assert getattr(model, attr)
        logged_neighbors = float(
            re.search(
                r"Average number of neighbors: ([\d.eE+-]+)",
                (ROOT / "provenance/train.log").read_text(),
            )[1]
        )
        assert abs(model.interactions[0].avg_num_neighbors - logged_neighbors) < 1e-12
        loss, metrics = evaluate(
            model=model,
            loss_fn=kwargs["loss_fn"],
            data_loader=kwargs["valid_loaders"]["Default"],
            output_args=kwargs["output_args"],
            device=kwargs["device"],
        )
        for key in ("rmse_e_per_atom", "rmse_f"):
            np.testing.assert_allclose(
                metrics[key], expected[key], atol=1e-10, rtol=1e-8
            )
        native = to_e3nn(deepcopy(model), device="cuda").to("cuda")
        structures = read(ROOT / "data/smoke_valid.xyz", ":3")
        table = AtomicNumberTable(model.atomic_numbers.tolist())

        def graph(atoms):
            data = AtomicData.from_config(
                config_from_atoms(atoms),
                z_table=table,
                cutoff=float(model.r_max),
                heads=model.heads,
            )
            return next(iter(DataLoader([data], batch_size=1))).to("cuda").to_dict()

        errors = {}
        for i, atoms in enumerate(structures):
            ref, out = (
                model(graph(atoms), training=False),
                native(graph(atoms), training=False),
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
                torch.testing.assert_close(
                    out[key], ref[key], atol=1e-8, rtol=0 if key == "energy" else 1e-9
                )
                errors[f"{i}/{key}"] = float((out[key] - ref[key]).abs().max().detach())
        torch.save(native.to("cpu"), destination)
        record = dict(
            status="passed",
            epoch=epoch,
            training_stage=1 if epoch < original["start_swa"] else 2,
            checkpoint_sha256=snapshot["checkpoint_sha256"],
            model_sha256=sha(destination),
            weight_semantics="EMA validation-selected weights saved by the native training loop",
            native_validation_reproduced=True,
            optimizer_steps_taken=0,
            avg_num_neighbors=logged_neighbors,
            parameters=337920,
            interaction_first=type(model.interactions[0]).__name__,
            validation=dict(
                energy_RMSE_meV_atom=float(1000 * metrics["rmse_e_per_atom"]),
                force_RMSE_meV_A=float(1000 * metrics["rmse_f"]),
            ),
            conversion_max_errors=errors,
        )
        dump(ROOT / "reconstruction.json", record)
        raise ReconstructionComplete

    tools.train = capture
    argv = sys.argv
    try:
        sys.argv = ["mace_run_train", "--config", str(path)]
        run_train.main()
    except ReconstructionComplete:
        return destination
    finally:
        sys.argv = argv
    raise RuntimeError("Native CLI did not reach the reconstruction interception")


def run():
    snapshot = json.loads((ROOT / "snapshot.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    for name, digest in manifest["sources"].items():
        assert sha(ROOT / name) == digest, name
    parent = json.loads((ROOT / "provenance/parent_manifest.json").read_text())
    for path, digest in parent["installed_sources"].items():
        assert sha(path) == digest, path
    for name, record in parent["inputs"].items():
        assert sha(ROOT / name) == record["sha256"], name
    for package, version in parent["environment"]["packages"].items():
        assert importlib.metadata.version(package) == version, package
    assert sha(snapshot["checkpoint"]) == snapshot["checkpoint_sha256"]
    model = reconstruct(snapshot)
    torch.cuda.empty_cache()
    name = f"les_response_seed{snapshot['seed']}_epoch{snapshot['epoch']}"
    output = ROOT / "results/probe_models.json"
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
            f"{name}=mace:{model}",
            "--out",
            str(output),
        ],
        check=True,
    )
    obs = json.loads(output.read_text())[name]["obs"]
    assert len(obs) == 40 * 8 and all(
        np.isfinite(np.asarray(v)).all() for v in obs.values()
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-u",
            str(ROOT / "analyze_probe2.py"),
            "--reference",
            str(ROOT / "data/probe_reference.json"),
            "--models",
            str(output),
            "--out",
            str(ROOT / "results/probe_analysis.json"),
        ],
        check=True,
    )
    analysis = json.loads((ROOT / "results/probe_analysis.json").read_text())
    result = analysis["models"][name]
    null = [
        np.asarray(value) - np.asarray(obs[key.split("|")[0] + "|base"])
        for key, value in obs.items()
        if "|null" in key
    ]
    assert len(null) == 80
    # Per-set slopes diagnose whether pooled alignment is shared by the probe sets.
    from analyze_probe2 import centred, load

    _, ref, predictions = load(ROOT / "data/probe_reference.json", output)
    per_set = {}
    for key in sorted(ref):
        x = centred(ref, key, True)[0].ravel()
        y = centred(predictions[name], key, True)[0].ravel()
        per_set[key] = float(np.dot(x, y) / np.dot(x, x))
    summary = dict(
        name=name,
        epoch=snapshot["epoch"],
        seed=snapshot["seed"],
        preliminary=True,
        interaction_first="RealAgnosticResidualInteractionBlock",
        probe_sets=40,
        trajectories=20,
        medium_variants_per_set=6,
        ionic_null_variants=80,
        slope=result["with_base"]["slope"],
        slope_ci95=result["with_base"]["ci95"],
        captured=result["with_base"]["captured"],
        K_RMSE_meV_A2=1000 * result["with_base"]["raw_rmse"],
        reference_modulation_RMS_meV_A2=1000 * analysis["ref_rms_with_base"],
        null_RMS_meV_A2=float(1000 * np.sqrt(np.mean(np.asarray(null) ** 2))),
        null_max_meV_A2=1000 * result["null_max"],
        positive_slope_sets=sum(v > 0 for v in per_set.values()),
        per_set_slopes=per_set,
        reconstruction=json.loads((ROOT / "reconstruction.json").read_text()),
    )
    dump(ROOT / "results/summary.json", summary)
    print("INTERIM_PROBE_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    status = dict(state="running", started=time.time())
    dump(ROOT / "status.json", status)
    try:
        run()
        status["state"] = "complete"
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["finished"] = time.time()
        dump(ROOT / "status.json", status)
