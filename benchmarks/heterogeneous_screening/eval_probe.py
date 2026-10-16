"""Evaluate trained models on probe states and form their D_AB per variant.

Models are given as NAME=KIND:PATH, with KIND mace (MACE or MACE-LES model file, via
mace.calculators.MACECalculator) or fno (MACE-FNO checkpoint, evaluated with eight
mesh origins as in the Yukawa MD). FNO checkpoints need --mace-model for the backbone.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read


def make_calc(kind, path, mace_model):
    if kind == "mace":
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=path, device="cuda", default_dtype="float64")
    if kind == "fno":
        from mace_fno.calculators import MACEFNOCalculator

        c = MACEFNOCalculator(
            checkpoint=path, device="cuda", dtype="float64", mace_model_path=mace_model
        )
        c.model.long_range.volume_interlacing = 2
        return c
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", required=True)
    ap.add_argument(
        "--mace-model", required=True, help="backbone for the FNO checkpoints"
    )
    ap.add_argument("--model", action="append", required=True, metavar="NAME=KIND:PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    images = read(a.states, index=":")
    results = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        if name in results:
            print(f"{name}: already evaluated, skipped", flush=True)
            continue
        calc = make_calc(kind, path, a.mace_model)
        t0 = time.time()
        energy = defaultdict(dict)
        for at in images:
            b = at.copy()
            b.calc = calc
            key = f"{at.info['probe_set']}|{at.info['variant']}"
            # extxyz reads labels like "01" back as integers; restore two characters.
            state = str(at.info["state"]).zfill(2)
            energy[key][state] = float(b.get_potential_energy())
        D = {k: e["11"] - e["10"] - e["01"] + e["00"] for k, e in energy.items()}
        results[name] = {
            "kind": kind,
            "path": path,
            "D": D,
            "energies": energy,
            "seconds": time.time() - t0,
        }
        Path(a.out).write_text(json.dumps(results, indent=1))
        print(
            f"{name}: {len(images)} states in {time.time() - t0:.0f} s; "
            f"|D| median {1e3 * np.median(np.abs(list(D.values()))):.2f} meV",
            flush=True,
        )


if __name__ == "__main__":
    main()
