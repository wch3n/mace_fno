"""Train full-response MACELES with validated conversion and complete optimizer."""

import argparse
from pathlib import Path

import yaml
from cueq_compat import install as install_conversion
from optimizer_compat import install as install_optimizer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    config = yaml.safe_load(Path(args.config).read_text())
    assert config["model"] == "MACELES" and config["enable_cueq"] is True
    les_config = yaml.safe_load(Path(config["les_arguments"]).read_text())
    assert all(
        les_config[key]
        for key in (
            "use_dipole",
            "use_quad",
            "use_induced_charge",
            "use_induced_dipole",
        )
    )
    install_conversion(les_config)
    install_optimizer(config["work_dir"])
    from mace.cli.run_train import main as train

    train()


if __name__ == "__main__":
    main()
