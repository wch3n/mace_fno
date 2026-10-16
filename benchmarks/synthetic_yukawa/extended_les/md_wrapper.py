"""Use the existing ASE MD integrator with checked cuEq extended LES."""

from pathlib import Path

import md_driver
import yaml
from cueq_compat import install

ROOT = Path(__file__).resolve().parent


def make_calc(args):
    from mace.calculators import MACECalculator

    assert args.calc == "mace" and args.kappa == "0p5"
    install(yaml.safe_load((ROOT / "configs/les_response.yaml").read_text()))
    calc = MACECalculator(
        model_paths=args.model, device="cuda", default_dtype="float64", enable_cueq=True
    )
    model = calc.models[0]
    assert model.interactions[0].cueq_config.enabled
    assert (
        model.use_induced_charges
        and model.use_induced_dipoles
        and model.use_dipoles
        and model.use_quads
    )
    return calc, dict(
        calc="mace",
        model=args.model,
        model_sha256=md_driver.sha(args.model),
        variant="extended LES full response",
        backend="cuequivariance",
    )


if __name__ == "__main__":
    md_driver.make_calc = make_calc
    md_driver.main()
