"""Design iii details: null RMS, per-set slopes, seed agreement, linear-FNO floor."""

import sys

import numpy as np
from analyze_probe2 import centred, load

P = sys.argv[1]
refj, ref, models = load(
    f"{P}/probeiii_valid_reference.json", f"{P}/probeiii_valid_models.json"
)
sets = sorted(ref)
dr = {s: centred(ref, s, True)[0] for s in sets}
print(
    f"reference: RMS of centred K {1e3 * np.sqrt(np.mean(np.concatenate([dr[s].ravel() for s in sets]) ** 2)):.3f} meV/A^2; "
    f"median within-set range {1e3 * np.median([np.max(np.ptp(centred(ref, s, True)[1], axis=0)) for s in sets]):.3f}"
)
for name in ("lin_s17", "lin_s29", "nl_s17", "nl_s29"):
    md = models[name]
    dm = {s: centred(md, s, True)[0] for s in sets}
    null = np.concatenate(
        [
            (md[s][k] - md[s]["base"])
            for s in sets
            for k in md[s]
            if k.startswith("null")
        ]
    )
    resp = np.sqrt(np.mean(np.concatenate([dm[s].ravel() for s in sets]) ** 2))
    per = []
    for s in sets:
        x, y = dr[s].ravel(), dm[s].ravel()
        per.append(np.sum(x * y) / np.sum(x * x))
    per = np.array(per)
    print(
        f"{name}: response RMS {1e3 * resp:.3f}, null RMS {1e3 * np.sqrt(np.mean(null**2)):.3f} meV/A^2; "
        f"per-set slope median {np.median(per):.2f}, positive in {np.sum(per > 0)}/{len(per)}"
    )
a = np.concatenate([centred(models["nl_s17"], s, True)[0].ravel() for s in sets])
b = np.concatenate([centred(models["nl_s29"], s, True)[0].ravel() for s in sets])
r = np.concatenate([dr[s].ravel() for s in sets])
print(
    f"corr(dK) nl_s17 vs nl_s29: {np.corrcoef(a, b)[0, 1]:.2f}; each vs reference: {np.corrcoef(a, r)[0, 1]:.2f}, {np.corrcoef(b, r)[0, 1]:.2f}; "
    f"seed mean vs reference: {np.corrcoef((a + b) / 2, r)[0, 1]:.2f}"
)
