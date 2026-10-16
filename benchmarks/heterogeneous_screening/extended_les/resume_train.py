"""Restore a native checkpoint and continue at its next epoch, capped at 1200."""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from cueq_compat import install as install_conversion
from optimizer_compat import install as install_optimizer

ROOT = Path(__file__).resolve().parent


def same_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            same_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            same_state(a, e)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    config = yaml.safe_load(Path(args.config).read_text())
    assert config["restart_latest"] and config["max_num_epochs"] == 1200
    assert config["start_swa"] == 450 and config["enable_cueq"]
    revision = json.loads((ROOT / "budget_resume.json").read_text())
    record = revision["runs"][config["name"]]
    checkpoint = torch.load(
        record["checkpoint"], map_location="cpu", weights_only=False
    )
    assert set(checkpoint) == {"model", "optimizer", "lr_scheduler"}
    install_conversion(yaml.safe_load(Path(config["les_arguments"]).read_text()))
    install_optimizer(config["work_dir"])
    from mace import tools
    from mace.cli import run_train

    original = tools.train

    def resumed(**kwargs):
        assert kwargs["start_epoch"] == record["epoch"], (
            "Native checkpoint restoration failed"
        )
        assert kwargs["max_num_epochs"] == 1200 and kwargs["swa"].start == 450
        same_state(kwargs["model"].state_dict(), checkpoint["model"])
        same_state(kwargs["optimizer"].state_dict(), checkpoint["optimizer"])
        same_state(kwargs["lr_scheduler"].state_dict(), checkpoint["lr_scheduler"])
        # Native MACE returns the saved (already completed) epoch number.
        kwargs["start_epoch"] = record["epoch"] + 1
        assert kwargs["start_epoch"] < 1200
        report = dict(
            status="passed",
            checkpoint_epoch=record["epoch"],
            next_epoch=kwargs["start_epoch"],
            max_num_epochs=1200,
            stage_two_start=450,
            model_optimizer_scheduler_restored_exactly=True,
            restart_semantics="Native saved EMA model and optimizer/LR state; EMA accumulator, RNG and patience history are not checkpointed and restart afresh.",
        )
        (Path(config["work_dir"]) / "resume_verified.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        logging.info(
            "Verified resume: checkpoint epoch %d; next epoch %d; total cap 1200",
            record["epoch"],
            kwargs["start_epoch"],
        )
        return original(**kwargs)

    tools.train = resumed
    run_train.main()


if __name__ == "__main__":
    main()
