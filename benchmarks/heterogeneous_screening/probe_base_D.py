"""D_AB accuracy on the base variants only (real frames, swaps as the only perturbation)."""

import json
import sys

import numpy as np

d = sys.argv[1]
Er = json.load(open(f"{d}/probe_valid_reference_energies.json"))
M = json.load(open(f"{d}/probe_valid_models.json"))
keys = sorted(k for k in Er if k.endswith("|base"))
ref = np.array([Er[k]["11"] - Er[k]["10"] - Er[k]["01"] + Er[k]["00"] for k in keys])
sw = np.array([[Er[k]["10"] - Er[k]["00"], Er[k]["01"] - Er[k]["00"]] for k in keys])
print(
    f"{len(keys)} base sets: reference D RMS {1e3 * np.sqrt(np.mean(ref**2)):.1f} meV, range "
    f"{1e3 * ref.min():.1f} to {1e3 * ref.max():.1f}; single-swap energies {sw.min():.2f} to {sw.max():.2f} eV"
)
print(
    f"{'model':<9} {'D RMSE':>8} {'corr(D, ref)':>13} {'swap RMSE':>10}   (meV, meV, eV)"
)
for name, m in M.items():
    D = np.array([m["D"][k] for k in keys])
    s = np.array(
        [
            [
                m["energies"][k]["10"] - m["energies"][k]["00"],
                m["energies"][k]["01"] - m["energies"][k]["00"],
            ]
            for k in keys
        ]
    )
    c = np.corrcoef(D, ref)[0, 1] if np.ptp(D) > 1e-9 else float("nan")
    print(
        f"{name:<9} {1e3 * np.sqrt(np.mean((D - ref) ** 2)):8.1f} {c:13.2f} {np.sqrt(np.mean((s - sw) ** 2)):10.3f}"
    )
