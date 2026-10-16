"""Keep the existing MD integrator and enable checked charge-only LES cuEq."""

from pathlib import Path

import md_driver
import yaml
from train_cueq import install

ROOT = Path(__file__).resolve().parent


def make_calc(args):
    from mace.calculators import MACECalculator

    assert args.calc == "mace" and args.kappa in ("0p05", "0p5")
    install(yaml.safe_load((ROOT / "configs/les.yaml").read_text()))
    calc = MACECalculator(
        model_paths=args.model, device="cuda", default_dtype="float64", enable_cueq=True
    )
    model = calc.models[0]
    assert type(model.interactions[0]).__name__ == "RealAgnosticInteractionBlock"
    assert model.interactions[0].cueq_config.enabled
    assert model.les.remove_self_interaction and model.les.ewald.norm_factor == 90.4756
    return calc, dict(
        calc="mace",
        model=args.model,
        model_sha256=md_driver.sha(args.model),
        interaction_first="RealAgnosticInteractionBlock",
        backend="cuequivariance",
        variant="ordinary charge-only LES, contact-trained",
    )


if __name__ == "__main__":
    md_driver.make_calc = make_calc
    md_driver.main()
