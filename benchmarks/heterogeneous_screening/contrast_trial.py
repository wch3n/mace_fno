"""Screening-contrast trial (protocol Sec. 6) without MD.

The large-scale S distribution is set by the driver field, not by the local
ionic structure, so contrast can be explored on random dense packings. For a
species-blind mixture, the equilibrium labelling in a field V(r) acting on S
is P(S set) ~ exp(-sum_{i in S} V(R_i) / kT); it is sampled here by
Metropolis identity swaps at fixed composition.

For each setting (S fraction, smoothing l, field shells, amplitude A/kT),
kappa1 is chosen so that the pooled volume 95th percentile of kappa is
0.5 A^-1, and the volume statistics of kappa(r) are reported together with
the driven power of c_S in shells 1-2 relative to independent particles.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from screening_reference import ScreenedMedium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic_yukawa"))
from gen_yukawa import init_positions  # noqa: E402

SHELLS = {
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)],
}


def random_field(rng, shells, L):
    """Unit-variance sum of cosines over the given shells, random phases."""
    modes = np.array([n for s in shells for n in SHELLS[s]], float)
    amp = rng.normal(size=len(modes))
    amp /= np.sqrt(np.sum(amp**2) / 2.0)
    phase = rng.uniform(0, 2 * np.pi, len(modes))
    k = 2 * np.pi * modes / L
    return lambda r: (np.cos(r @ k.T + phase) * amp).sum(axis=1)


def sample_labels(rng, v, n_s, sweeps=60):
    """Metropolis identity swaps for P(S set) ~ exp(-sum_S v)."""
    n = len(v)
    is_s = np.zeros(n, bool)
    is_s[rng.choice(n, n_s, replace=False)] = True
    s_idx, o_idx = np.flatnonzero(is_s), np.flatnonzero(~is_s)
    for _ in range(sweeps * n):
        a, b = rng.integers(len(s_idx)), rng.integers(len(o_idx))
        i, j = s_idx[a], o_idx[b]
        dv = v[j] - v[i]
        if dv <= 0 or rng.random() < np.exp(-dv):
            s_idx[a], o_idx[b] = j, i
    out = np.zeros(n, bool)
    out[s_idx] = True
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--L", type=float, default=32.0)
    ap.add_argument("--n", type=int, default=1146)
    ap.add_argument("--kappa0", type=float, default=0.05)
    ap.add_argument("--kappa95", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="contrast_trial.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    templates = [init_positions(rng, a.n, a.L, dmin=2.3) for _ in range(a.frames)]
    fields = {
        sh: [random_field(rng, sh, a.L) for _ in range(a.frames)]
        for sh in ((1,), (1, 2))
    }

    rows = []
    for frac, ell, sh, amp in itertools.product(
        (1 / 2, 1 / 3, 1 / 4), (4.0, 3.0), ((1,), (1, 2)), (1.0, 2.0, 4.0)
    ):
        n_s = int(round(frac * a.n))
        n_s += (a.n - n_s) % 2  # even ion count
        med = ScreenedMedium(a.L, a.kappa0, 1.0, ell)
        c_all, power = [], []
        shell_sel = np.isin(np.sum(med.n_c**2, axis=1), (1, 2))
        k2 = np.sum(med.k_c[shell_sel] ** 2, axis=1)
        for t in range(a.frames):
            pos = templates[t]
            v = amp * fields[sh][t](pos)
            is_s = sample_labels(np.random.default_rng(1000 + t), v, n_s)
            med.set_medium(pos[is_s])
            c_all.append(med.c_r.ravel())
            ideal = np.exp(-(ell**2) * k2) / n_s
            power.append(np.mean(np.abs(med.c_k[shell_sel]) ** 2 / ideal))
        c = np.clip(np.concatenate(c_all), 0, None)
        c95 = np.percentile(c, 95)
        kappa1 = np.sqrt((a.kappa95**2 - a.kappa0**2) / c95)
        kap = np.sqrt(a.kappa0**2 + kappa1**2 * c)
        rows.append(
            {
                "s_fraction": round(frac, 3),
                "n_s": n_s,
                "ell": ell,
                "shells": "1" if sh == (1,) else "1-2",
                "amp_kT": amp,
                "kappa1": float(kappa1),
                "kappa_p5": float(np.percentile(kap, 5)),
                "kappa_p50": float(np.percentile(kap, 50)),
                "vol_kappa_lt_0p10": float(np.mean(kap < 0.10)),
                "vol_kappa_lt_0p15": float(np.mean(kap < 0.15)),
                "vol_kappa_gt_0p40": float(np.mean(kap > 0.40)),
                "driven_power_ratio": float(np.mean(power)),
            }
        )
        r = rows[-1]
        print(
            f"S {r['s_fraction']:.2f}  l {ell:.0f}  shells {r['shells']:3s}  A {amp:.0f}  "
            f"kappa1 {r['kappa1']:.3f}  p5 {r['kappa_p5']:.3f}  p50 {r['kappa_p50']:.3f}  "
            f"V(<0.10) {r['vol_kappa_lt_0p10']:.2f}  V(<0.15) {r['vol_kappa_lt_0p15']:.2f}  "
            f"V(>0.40) {r['vol_kappa_gt_0p40']:.2f}  power {r['driven_power_ratio']:.0f}",
            flush=True,
        )
    Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
