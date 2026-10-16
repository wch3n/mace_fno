"""Errors of unswapped probe configurations, split by variant type."""

import json
import sys

import numpy as np

d = sys.argv[1]
Er = json.load(open(f"{d}/probe_valid_reference_energies.json"))
M = json.load(open(f"{d}/probe_valid_models.json"))
groups = {
    "base": lambda k: k == "base",
    "field": lambda k: k.startswith("field"),
    "uniform": lambda k: k == "uniform",
    "null": lambda k: k.startswith("null"),
}
base_e = {k.split("|")[0]: v["00"] for k, v in Er.items() if k.endswith("|base")}
print(
    "reference E(00) minus the base frame, mean by variant type (eV): "
    + "  ".join(
        f"{g} {np.mean([Er[k]['00'] - base_e[k.split('|')[0]] for k in Er if f(k.split('|')[1])]):+.1f}"
        for g, f in groups.items()
    )
)
print(
    f"{'model':<9}"
    + "".join(f"{g:>10}" for g in groups)
    + "   RMSE of E(00), eV per configuration"
)
for name, m in M.items():
    print(
        f"{name:<9}"
        + "".join(
            f"{np.sqrt(np.mean([(m['energies'][k]['00'] - Er[k]['00']) ** 2 for k in Er if f(k.split('|')[1])])):10.2f}"
            for g, f in groups.items()
        )
    )
