"""Model vs reference on the probe (designs ii and iii).

The observable is D_AB (design ii, one number per variant) or the 3 x 3 cross force
constant K_AB (design iii, nine numbers). Per set and component, values are centred
on their mean over the non-null variants, with or without the base variant.
Reported per model: pooled within-set slope of model on reference (bootstrap 95 %
interval over trajectories, keeping their sets together), captured fraction 1 - RMS(dX_model - dX_ref) / RMS(dX_ref),
raw RMSE, the largest within-set range (the noise floor for fixed-propagation
models, which must be flat), and the null response |X(null) - X(base)| (the
reference does not change under null variants).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load(ref_path, models_path):
    refj = json.load(open(ref_path))
    key = "K_ref" if refj["design"] == "iii" else "D_ref"
    ref = {}
    for v in refj["variants"]:
        ref.setdefault(v["set"], {})[v["kind"]] = np.atleast_1d(
            np.array(v[key], float)
        ).ravel()
    models = {}
    for name, m in json.load(open(models_path)).items():
        md = {}
        for k, x in m["obs"].items():
            s, kind = k.split("|")
            md.setdefault(s, {})[kind] = np.atleast_1d(np.array(x, float)).ravel()
        models[name] = md
    return refj, ref, models


def centred(d, s, include_base):
    kinds = [
        k for k in d[s] if not k.startswith("null") and (include_base or k != "base")
    ]
    X = np.array([d[s][k] for k in kinds])
    return X - X.mean(axis=0), X


def analyse(ref, md, sets, include_base, rng, n_boot=2000):
    dr = {s: centred(ref, s, include_base)[0] for s in sets}
    dm = {s: centred(md, s, include_base)[0] for s in sets}
    raw = np.concatenate(
        [
            (centred(md, s, include_base)[1] - centred(ref, s, include_base)[1]).ravel()
            for s in sets
        ]
    )

    def stats(sel):
        x = np.concatenate([dr[s].ravel() for s in sel])
        y = np.concatenate([dm[s].ravel() for s in sel])
        return np.sum(x * y) / np.sum(x * x), 1 - np.sqrt(
            np.mean((y - x) ** 2) / np.mean(x**2)
        )

    slope, cap = stats(sets)
    # resample trajectories, keeping all sets of a trajectory together: sets drawn
    # from the same frame are not independent
    groups = {}
    for s in sets:
        groups.setdefault(s.split("_")[0], []).append(s)
    keys = sorted(groups)
    boot = [
        stats(
            [s for g in rng.choice(keys, len(keys), replace=True) for s in groups[g]]
        )[0]
        for _ in range(n_boot)
    ]
    return {
        "slope": float(slope),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "captured": float(cap),
        "raw_rmse": float(np.sqrt(np.mean(raw**2))),
        "ref_rms": float(
            np.sqrt(np.mean(np.concatenate([dr[s].ravel() for s in sets]) ** 2))
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    refj, ref, models = load(a.reference, a.models)
    sets = sorted(ref)
    unit, fac = ("meV/A^2", 1e3) if refj["design"] == "iii" else ("meV", 1e3)
    rng = np.random.default_rng(0)
    r_wb = analyse(ref, ref, sets, True, rng, 1)["ref_rms"]
    r_nb = analyse(ref, ref, sets, False, rng, 1)["ref_rms"]
    print(
        f"design {refj['design']}: {len(sets)} sets; reference modulation (RMS of centred values) "
        f"{fac * r_wb:.3f} {unit} with base, {fac * r_nb:.3f} without; "
        f"in-distribution check: {refj.get('in_distribution')}"
    )
    print(
        f"{'model':<10} {'slope (95% CI) with base':>28} {'captured':>9} {'slope no base':>14} "
        f"{'raw RMSE':>9} {'max range':>10} {'null max':>9}   ({unit})"
    )
    report = {}
    for name, md in models.items():
        wb, nb = analyse(ref, md, sets, True, rng), analyse(ref, md, sets, False, rng)
        span = max(
            np.max(
                np.ptp(
                    np.array([md[s][k] for k in md[s] if not k.startswith("null")]),
                    axis=0,
                )
            )
            for s in sets
        )
        null = max(
            np.max(np.abs(md[s][k] - md[s]["base"]))
            for s in sets
            for k in md[s]
            if k.startswith("null")
        )
        report[name] = {
            "with_base": wb,
            "without_base": nb,
            "max_range": float(span),
            "null_max": float(null),
        }
        print(
            f"{name:<10} {wb['slope']:8.3f} ({wb['ci95'][0]:7.3f}, {wb['ci95'][1]:7.3f})   "
            f"{wb['captured']:9.3f} {nb['slope']:14.3f} {fac * wb['raw_rmse']:9.3f} "
            f"{fac * span:10.4f} {fac * null:9.4f}"
        )
    if a.out:
        json.dump(
            {
                "design": refj["design"],
                "ref_rms_with_base": r_wb,
                "ref_rms_without_base": r_nb,
                "models": report,
            },
            open(a.out, "w"),
            indent=2,
        )


if __name__ == "__main__":
    main()
