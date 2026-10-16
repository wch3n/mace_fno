"""Check immutable inputs and matched low-rank configurations before submission."""

import json
from pathlib import Path

from run import digest

from mace_fno.cli.config import parse_arguments
from mace_fno.training.configuration import TrainingConfig

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for task in manifest["tasks"] + manifest["smoke_tasks"]:
    cfg = TrainingConfig.from_namespace(parse_arguments(["--config", task["config"]]))
    assert cfg.model.spectral_rank == task["rank"] in (32, 64)
    assert cfg.model.z_mixing == "full" and cfg.model.architecture == "nonlinear"
    assert cfg.model.fno_layers == 4 and cfg.model.fno_hidden_channels == 16
    assert cfg.model.z_grid == 24 and cfg.model.modes == 8
    assert cfg.model.channels == 16 and cfg.model.lateral_interlacing == 2
    assert cfg.optimization.batch_size * cfg.optimization.accumulation_steps == 4
    assert cfg.optimization.steps == task["steps"]
    assert cfg.runtime.resume is None and cfg.runtime.init_from is None
    assert cfg.data.test_file is None and cfg.data.train_cache is None
    assert not Path(task["checkpoint"]).parent.exists()
for relative, checksum in manifest["source_hashes"].items():
    assert digest(root / relative) == checksum, relative
for relative, info in manifest["inputs"].items():
    assert digest(root / relative) == info["sha256"], relative
result = dict(
    state="passed",
    production_tasks=len(manifest["tasks"]),
    smoke_tasks=len(manifest["smoke_tasks"]),
)
(root / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
