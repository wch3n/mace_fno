"""Validate configurations and provenance before any GPU allocation."""

import json
from pathlib import Path

import torch
from run import digest

from mace_fno.cli.config import parse_arguments
from mace_fno.training.configuration import TrainingConfig

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for task in manifest["tasks"] + manifest["smoke_tasks"]:
    cfg = TrainingConfig.from_namespace(parse_arguments(["--config", task["config"]]))
    assert cfg.model.architecture == "nonlinear" and cfg.model.z_mixing == "full"
    assert cfg.model.fno_layers == 4 and cfg.model.fno_hidden_channels == 16
    assert cfg.model.channels == 16 and cfg.model.lateral_interlacing == 2
    assert (
        cfg.model.spectral_groups == 1
        and cfg.model.z_grid == 24
        and cfg.model.modes == 8
    )
    assert task["depth"] == 2 and cfg.data.mace_model.name == "mace_nl2.model"
    assert cfg.optimization.batch_size * cfg.optimization.accumulation_steps == 4
    assert cfg.runtime.resume is None and cfg.runtime.init_from is None
    assert cfg.data.test_file is None and cfg.data.train_cache is None
    assert cfg.optimization.steps == task["steps"]
    assert not Path(task["checkpoint"]).exists()
for rel, checksum in manifest["source_hashes"].items():
    assert digest(root / rel) == checksum, rel
for rel, record in manifest["inputs"].items():
    assert digest(root / rel) == record["sha256"], rel
model = torch.load(
    root / "models/mace_nl2.model", map_location="cpu", weights_only=False
)
assert len(model.interactions) == 2 and int(model.num_interactions) == 2
assert float(model.r_max) == 5.5
assert {t["seed"] for t in manifest["tasks"]} == {17, 29, 41, 53, 67}
assert len(manifest["tasks"]) == 10
result = dict(
    state="passed",
    production_tasks=len(manifest["tasks"]),
    smoke_tasks=len(manifest["smoke_tasks"]),
    backbone_interactions=len(model.interactions),
    backbone_parameters=sum(p.numel() for p in model.parameters()),
    interaction_first=type(model.interactions[0]).__name__,
    cutoff_A=float(model.r_max),
    smoke="Both modes use the complete production inputs and batching; joint gradients activate after two warmup steps.",
)
(root / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
