"""Use native MACE training, preserving charge-only LES settings in conversion."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def install(les_arguments):
    from mace.cli import convert_cueq_e3nn, convert_e3nn_cueq
    from mace.tools import scripts_utils

    original = scripts_utils.extract_config_mace_model

    def extract(model):
        config = original(model)
        if type(model).__name__ == "MACELES":
            config["les_arguments"] = deepcopy(les_arguments)
        return config

    scripts_utils.extract_config_mace_model = extract
    convert_e3nn_cueq.extract_config_mace_model = extract
    convert_cueq_e3nn.extract_config_mace_model = extract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    config = yaml.safe_load(Path(args.config).read_text())
    assert config["model"] == "MACELES" and config["enable_cueq"] is True
    install(yaml.safe_load(Path(config["les_arguments"]).read_text()))
    from mace.cli.run_train import main

    main()
