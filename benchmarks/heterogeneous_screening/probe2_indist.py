"""Model energy errors on the probe variants, by variant type (in-distribution check)."""

import json
import sys

import numpy as np

P = sys.argv[1]
for d, ekey in (("iii", "E_dx+"), ("ii", "s00")):
    ref = {
        f"{v['set']}|{v['kind']}": v["E_ref_00"]
        for v in json.load(open(f"{P}/probe{d}_valid_reference.json"))["variants"]
    }
    M = json.load(open(f"{P}/probe{d}_valid_models.json"))
    groups = {"base": "base", "field": "field", "uniform": "uniform", "null": "null"}
    print(
        f"design {d}: RMSE of model energy vs reference on unswapped/undisplaced variants (eV per configuration)"
    )
    print(f"  {'model':<9}" + "".join(f"{g:>9}" for g in groups))
    for name, m in M.items():
        row = f"  {name:<9}"
        for g in groups:
            e = [
                m["raw"][k][ekey] - ref[k] for k in ref if k.split("|")[1].startswith(g)
            ]
            row += f"{np.sqrt(np.mean(np.square(e))):9.2f}"
        print(row)
