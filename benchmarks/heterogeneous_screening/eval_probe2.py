"""Evaluate models on probe configurations (designs ii and iii).

Design iii configurations carry `probe_atom` (A) and `config` = d{x,y,z}{+,-}; the
model's force on A gives K_AB = -(F_A(+) - F_A(-)) / (2 delta). Design ii
configurations carry `config` = s00, s10, s01, s11; energies give
D_AB = E(s11) - E(s10) - E(s01) + E(s00). Models: NAME=KIND:PATH, KIND mace or fno
(FNO with eight mesh origins).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from eval_probe import make_calc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", required=True)
    ap.add_argument(
        "--reference", required=True, help="probe*_reference.json (for delta)"
    )
    ap.add_argument("--mace-model", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=KIND:PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.loads(Path(a.reference).read_text())
    delta = ref["params"]["delta"]
    images = read(a.configs, index=":")
    design = "iii" if "probe_atom" in images[0].info else "ii"
    results = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        if name in results:
            continue
        calc = make_calc(kind, path, a.mace_model)
        t0 = time.time()
        vals = defaultdict(dict)
        for at in images:
            b = at.copy()
            b.calc = calc
            key = f"{at.info['probe_set']}|{at.info['variant']}"
            if design == "iii":
                vals[key][at.info["config"]] = b.get_forces()[
                    int(at.info["probe_atom"])
                ].tolist()
                # energy of a configuration displaced by delta ~ E of the variant itself,
                # for the in-distribution check against the reference
                vals[key]["E_" + at.info["config"]] = float(b.get_potential_energy())
            else:
                vals[key][at.info["config"]] = float(b.get_potential_energy())
        obs = {}
        for key, v in vals.items():
            if design == "iii":
                K = np.zeros((3, 3))
                for j, ax in enumerate("xyz"):
                    K[:, j] = -(np.array(v[f"d{ax}+"]) - np.array(v[f"d{ax}-"])) / (
                        2 * delta
                    )
                obs[key] = K.tolist()
            else:
                obs[key] = v["s11"] - v["s10"] - v["s01"] + v["s00"]
        results[name] = {
            "kind": kind,
            "path": path,
            "design": design,
            "obs": obs,
            "raw": vals,
            "seconds": time.time() - t0,
        }
        Path(a.out).write_text(json.dumps(results))
        print(
            f"{name}: {len(images)} configurations in {time.time() - t0:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
