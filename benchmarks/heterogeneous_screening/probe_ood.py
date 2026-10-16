"""Are the probe's swapped states out of distribution for the models?

Labels every probe state with the reference, then compares each model's energies:
the unswapped configurations (state 00: the frame or its remote relabelling), the
single-swap energies E(10) - E(00) and E(01) - E(00), and the double swap
E(11) - E(00). Errors are compared with the model's error on the unswapped states.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from screening_reference import HeterogeneousScreeningCalculator  # noqa: E402

PRM = {}


def ref_energy(at):
    calc = HeterogeneousScreeningCalculator(
        32.0,
        PRM["kappa0"],
        PRM["kappa1"],
        PRM["ell"],
        scale=PRM["scale"],
        grid=48,
        workers=1,
    )
    b = at.copy()
    b.calc = calc
    return (
        f"{at.info['probe_set']}|{at.info['variant']}",
        str(at.info["state"]).zfill(2),
        float(b.get_potential_energy()),
    )


def init(prm):
    PRM.update(prm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, help="directory with probe_valid_* files")
    ap.add_argument("--data", required=True)
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    a = ap.parse_args()
    probe = Path(a.probe)
    lab = json.loads((Path(a.data) / "metadata.json").read_text())["labels"]
    prm = {k: lab[k] for k in ("kappa0", "kappa1", "ell", "scale")}
    cache = probe / "probe_valid_reference_energies.json"
    if cache.exists():
        E_ref = json.loads(cache.read_text())
    else:
        images = read(probe / "probe_valid_states.xyz", index=":")
        with mp.Pool(a.nproc, initializer=init, initargs=(prm,)) as pool:
            res = pool.map(ref_energy, images, chunksize=4)
        E_ref = defaultdict(dict)
        for key, state, e in res:
            E_ref[key][state] = e
        cache.write_text(json.dumps(E_ref, indent=1))
    models = json.loads((probe / "probe_valid_models.json").read_text())
    keys = sorted(E_ref)

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    ref_swap = np.array(
        [
            [E_ref[k]["10"] - E_ref[k]["00"], E_ref[k]["01"] - E_ref[k]["00"]]
            for k in keys
        ]
    )
    print(
        f"reference single-swap energies: median |dE| {1e3 * np.median(np.abs(ref_swap)):.0f} meV, "
        f"range {1e3 * ref_swap.min():.0f} to {1e3 * ref_swap.max():.0f} meV"
    )
    print(
        f"{'model':<10} {'00 RMSE':>9} {'00 centred':>11} {'swap RMSE':>10} {'double RMSE':>12} "
        f"{'D RMSE':>8}   (meV per configuration)"
    )
    out = {}
    for name, m in models.items():
        e00 = np.array([m["energies"][k]["00"] - E_ref[k]["00"] for k in keys])
        sw = np.array(
            [
                [
                    (m["energies"][k]["10"] - m["energies"][k]["00"])
                    - (E_ref[k]["10"] - E_ref[k]["00"]),
                    (m["energies"][k]["01"] - m["energies"][k]["00"])
                    - (E_ref[k]["01"] - E_ref[k]["00"]),
                ]
                for k in keys
            ]
        )
        dbl = np.array(
            [
                (m["energies"][k]["11"] - m["energies"][k]["00"])
                - (E_ref[k]["11"] - E_ref[k]["00"])
                for k in keys
            ]
        )
        dD = np.array(
            [
                m["D"][k]
                - (E_ref[k]["11"] - E_ref[k]["10"] - E_ref[k]["01"] + E_ref[k]["00"])
                for k in keys
            ]
        )
        out[name] = {
            "rmse_00": rms(e00),
            "rmse_00_centred": rms(e00 - e00.mean()),
            "rmse_swap": rms(sw),
            "rmse_double": rms(dbl),
            "rmse_D": rms(dD),
        }
        o = out[name]
        print(
            f"{name:<10} {1e3 * o['rmse_00']:9.1f} {1e3 * o['rmse_00_centred']:11.1f} "
            f"{1e3 * o['rmse_swap']:10.1f} {1e3 * o['rmse_double']:12.1f} {1e3 * o['rmse_D']:8.1f}"
        )
    (probe / "probe_valid_ood.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
