"""Per-set breakdown of a model's probe response against the reference."""

import json
import sys

import numpy as np

d = sys.argv[1]
ref = {}
for v in json.load(open(f"{d}/probe_valid_reference.json"))["variants"]:
    ref.setdefault(v["set"], {})[v["kind"]] = v["D_ref"]
M = json.load(open(f"{d}/probe_valid_models.json"))


def table(name):
    md = {}
    for k, val in M[name]["D"].items():
        s, kind = k.split("|")
        md.setdefault(s, {})[kind] = val
    rows = []
    for s in sorted(ref):
        ks = [k for k in ref[s] if not k.startswith("null")]
        r = np.array([ref[s][k] for k in ks])
        m = np.array([md[s][k] for k in ks])
        dr, dm = r - r.mean(), m - m.mean()
        null = np.array(
            [md[s][k] - md[s]["base"] for k in md[s] if k.startswith("null")]
        )
        corr = (
            np.corrcoef(dr, dm)[0, 1]
            if np.ptp(r) > 1e-6 and np.ptp(m) > 1e-9
            else np.nan
        )
        rows.append(
            (s, 1e3 * np.ptp(r), 1e3 * np.ptp(m), corr, 1e3 * np.abs(null).max())
        )
    return rows


r17, r29 = table("nl_s17"), table("nl_s29")
print(
    f"{'set':<14} {'ref range':>9} | {'nl17 range':>10} {'corr':>6} {'null':>6} | {'nl29 range':>10} {'corr':>6} {'null':>6}  (meV)"
)
for a, b in zip(r17, r29):
    print(
        f"{a[0]:<14} {a[1]:9.1f} | {a[2]:10.1f} {a[3]:6.2f} {a[4]:6.1f} | {b[2]:10.1f} {b[3]:6.2f} {b[4]:6.1f}"
    )
for lab, rows in (("s17", r17), ("s29", r29)):
    c = np.array([x[3] for x in rows if x[1] > 5])
    print(
        f"{lab}: sets with ref range > 5 meV: corr < 0 in {np.sum(c < 0)}/{len(c)}, median corr {np.median(c):.2f}"
    )
keys = sorted(M["nl_s17"]["D"])
x = np.array([M["nl_s17"]["D"][k] for k in keys])
y = np.array([M["nl_s29"]["D"][k] for k in keys])
print(
    f"corr of D between the two nonlinear seeds over all variants: {np.corrcoef(x, y)[0, 1]:.2f}"
)
