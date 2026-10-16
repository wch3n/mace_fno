"""Run the native trainer for all nine distinct architecture/regime settings."""

import importlib.metadata
import sys
from pathlib import Path

from common import command, dump, load, now, parse_configuration, verify_inputs


def main():
    import torch
    import yaml

    root = Path(__file__).resolve().parent
    manifest = load(root / "manifest.json")
    verify_inputs(root)
    assert torch.cuda.is_available()
    versions = {k: importlib.metadata.version(k) for k in manifest["package_versions"]}
    assert versions == manifest["package_versions"]
    for task in manifest["tasks"]:
        parse_configuration(task["config"])
    records = []
    for task in [t for t in manifest["tasks"] if t["seed"] == 29]:
        cfg = yaml.safe_load(Path(task["config"]).read_text())
        run = root / "smoke" / task["name"]
        run.mkdir(exist_ok=False)
        cfg.update(
            steps=8,
            eval_interval=4,
            checkpoint_interval=4,
            mace_warmup_steps=2 if task["regime"] == "joint" else 0,
            checkpoint=str(run / "model.pt"),
            last_checkpoint=str(run / "model.last.pt"),
        )
        config = run / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        command(
            [sys.executable, "-B", "-u", root / "train.py", "--config", config],
            run / "training.log",
        )
        state = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        assert state["spectral_groups"] == 1 and state["completed_steps"] == 8
        assert state["mace_training"] == task["regime"]
        assert bool(state.get("mace_state_dict")) == (task["regime"] == "joint")

        def finite(value):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            elif isinstance(value, dict):
                for v in value.values():
                    finite(v)

        finite(state)
        assert (run / "model.last.pt").exists()
        records.append(
            dict(
                task=task["name"],
                regime=task["regime"],
                mace_layers=task["mace_layers"],
                variant=task["variant"],
                spectral_groups=1,
                state="passed",
            )
        )
        dump(root / "smoke" / "progress.json", records)
        print("SMOKE PASSED", task["name"], flush=True)
    assert len(records) == 9
    dump(
        root / "smoke.json",
        dict(
            state="passed",
            finished=now(),
            records=records,
            packages=versions,
            gpu=torch.cuda.get_device_name(),
        ),
    )


if __name__ == "__main__":
    main()
