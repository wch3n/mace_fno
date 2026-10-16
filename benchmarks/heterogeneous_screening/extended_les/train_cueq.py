"""Run standard MACE training with the extended-LES conversion fix installed."""

import argparse
from pathlib import Path

import yaml
from cueq_compat import install


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    config = yaml.safe_load(Path(args.config).read_text())
    assert config["model"] == "MACELES" and config["enable_cueq"] is True
    install(yaml.safe_load(Path(config["les_arguments"]).read_text()))
    from mace.cli.run_train import main as train

    train()


if __name__ == "__main__":
    main()
