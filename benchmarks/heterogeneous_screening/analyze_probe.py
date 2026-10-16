"""Compare model and reference D_AB on the probe (protocol Sec. 9).

Per set, D is centred on its mean over the non-null variants (with or without the
base variant), so that only the response to the remote medium is compared. Reported
per model: the pooled within-set slope of model against reference (bootstrap 95 %
interval over sets), the captured fraction 1 - RMS(dD_model - dD_ref) / RMS(dD_ref),
the raw RMSE of D, the largest within-set range of D (exactly zero for local and
fixed-propagation models up to numerics), and the null-control response (remote ions
shuffled, S unchanged; the reference does not change).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def analyse(ref, model, sets, include_base, rng, n_boot=2000):
    dm, dr, raw = {}, {}, []
    for s in sets:
        kinds = [
            k
            for k in ref[s]
            if not k.startswith("null") and (include_base or k != "base")
        ]
        r = np.array([ref[s][k] for k in kinds])
        m = np.array([model[s][k] for k in kinds])
        dr[s], dm[s] = r - r.mean(), m - m.mean()
        raw.extend(m - r)

    def stats(sel):
        x = np.concatenate([dr[s] for s in sel])
        y = np.concatenate([dm[s] for s in sel])
        return np.sum(x * y) / np.sum(x * x), 1 - np.sqrt(
            np.mean((y - x) ** 2) / np.mean(x**2)
        )

    slope, captured = stats(sets)
    boot = [stats(rng.choice(sets, len(sets), replace=True))[0] for _ in range(n_boot)]
    return {
        "slope": float(slope),
        "slope_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "captured": float(captured),
        "raw_rmse_meV": float(1e3 * np.sqrt(np.mean(np.square(raw)))),
        "ref_rms_modulation_meV": float(
            1e3 * np.sqrt(np.mean(np.concatenate(list(dr.values())) ** 2))
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    refj = json.load(open(a.reference))
    ref = {}
    for v in refj["variants"]:
        ref.setdefault(v["set"], {})[v["kind"]] = v["D_ref"]
    sets = sorted(ref)
    models = json.load(open(a.models))
    rng = np.random.default_rng(0)
    report = {}
    print(
        f"{len(sets)} sets; reference modulation (RMS of centred D): "
        f"{analyse(ref, ref, sets, True, rng, 1)['ref_rms_modulation_meV']:.1f} meV with base, "
        f"{analyse(ref, ref, sets, False, rng, 1)['ref_rms_modulation_meV']:.1f} meV without"
    )
    print(
        f"{'model':<16} {'slope (95% CI), with base':>30} {'captured':>9} {'slope, no base':>16} "
        f"{'raw RMSE':>9} {'max range':>10} {'null RMS':>9}"
    )
    for name, m in models.items():
        md = {}
        for key, d in m["D"].items():
            s, k = key.split("|")
            md.setdefault(s, {})[k] = d
        wb = analyse(ref, md, sets, True, rng)
        nb = analyse(ref, md, sets, False, rng)
        rngs = [
            np.ptp([md[s][k] for k in md[s] if not k.startswith("null")]) for s in sets
        ]
        null = [
            md[s][k] - md[s]["base"]
            for s in sets
            for k in md[s]
            if k.startswith("null")
        ]
        report[name] = {
            "with_base": wb,
            "without_base": nb,
            "max_within_set_range_meV": float(1e3 * max(rngs)),
            "null_rms_meV": float(1e3 * np.sqrt(np.mean(np.square(null)))),
        }
        print(
            f"{name:<16} {wb['slope']:7.3f} ({wb['slope_ci95'][0]:6.3f}, {wb['slope_ci95'][1]:6.3f})"
            f"{'':>8} {wb['captured']:9.3f} {nb['slope']:16.3f} {wb['raw_rmse_meV']:8.2f} "
            f"{1e3 * max(rngs):10.3f} {1e3 * np.sqrt(np.mean(np.square(null))):9.3f}"
        )
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
