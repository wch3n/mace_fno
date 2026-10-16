"""Probe trial: energy spread, self-energy share and remote-screening probe.

Templates are equilibrated 2800 K frames of the Yukawa MD data (1146 ions,
L = 32 A), relabelled: S identities are sampled in a shell-1 driver field
(Metropolis swaps, as in contrast_trial.py), the remaining sites keep their
original species, and the few sites needed to restore neutrality are flipped
at random. This keeps the ionic charge order of the template.

For each setting (S fraction, l, amplitude A/kT):
  1. kappa1 so that the pooled volume kappa_95 is 0.5 A^-1;
  2. E_scr' = COUL (E_scr - e_self(kappa0) sum q^2) on every frame, and the
     energy scale s that sets its per-atom spread to 5 meV/atom;
  3. the self-energy share: a fit E_scr' ~ sum_i f(c_S(R_i)) with a
     piecewise-linear f, scored by leave-one-out residuals;
  4. design (i) probe sets: for a few ion pairs (A, B), six remote variants
     (positions fixed, identities outside the frozen region reassigned) and
     the reference D_AB = 4 q_A q_B (P G P)_AB split into the pair term
     4 q_A q_B G_AB, the neutralizing-background term and the diagonal part
     4 q_A q_B [-(G_AA + G_BB)/N + sum_j G_jj / N^2];
  5. design (ii) probe sets: neutral compensated swaps A<->C, B<->D at
     FCC-like offsets (cross-group distances > 2 r_max), six variants each,
     D_AB = 4 q_A q_B (G_AB - G_AD - G_CB + G_CD), which vanishes in a uniform
     medium and measures how differently the two path pairs are screened.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from contrast_trial import random_field  # noqa: E402
from screening_reference import (  # noqa: E402
    COUL,
    FOUR_PI,
    ScreenedMedium,
    self_energy_discrete,
)

L, KAPPA0, SIGMA, KMAX, CHARGE = 32.0, 0.05, 0.6, 3.5, 0.4
FROZEN_RADIUS = 10.0


def mic(d):
    return d - L * np.round(d / L)


def metropolis_labels(rng, v, n_pick, sweeps=60):
    """Choose n_pick of len(v) sites with P ~ exp(-sum v) by swap Metropolis."""
    n = len(v)
    chosen = rng.choice(n, n_pick, replace=False)
    mask = np.zeros(n, bool)
    mask[chosen] = True
    sel, rest = np.flatnonzero(mask), np.flatnonzero(~mask)
    for _ in range(sweeps * n):
        a, b = rng.integers(len(sel)), rng.integers(len(rest))
        dv = v[rest[b]] - v[sel[a]]
        if dv <= 0 or rng.random() < np.exp(-dv):
            sel[a], rest[b] = rest[b], sel[a]
    out = np.zeros(n, bool)
    out[sel] = True
    return out


def relabel(template_symbols, pos, n_s, amp, seed):
    """Symbols with n_s S particles placed in a shell-1 field of amplitude amp."""
    rng = np.random.default_rng(seed)
    field = random_field(rng, (1,), L)
    is_s = metropolis_labels(rng, amp * field(pos), n_s)
    sym = np.array(template_symbols, dtype=object)
    sym[is_s] = "Ar"
    target = (len(sym) - n_s) // 2
    excess = int(np.sum(sym == "Na")) - target
    if excess:
        src, dst = ("Na", "Cl") if excess > 0 else ("Cl", "Na")
        flip = rng.choice(np.flatnonzero(sym == src), abs(excess), replace=False)
        sym[flip] = dst
    return sym


def charges(sym):
    return np.where(sym == "Na", CHARGE, np.where(sym == "Cl", -CHARGE, 0.0))


def c_at(med, r):
    """c_S evaluated at positions r from its plane-wave coefficients."""
    return np.real(np.exp(1j * (r @ med.k_c.T)) @ med.c_k)


# -- stage 1: energies --------------------------------------------------------
def energy_task(job):
    sym, pos, kappa1, ell = job
    med = ScreenedMedium(L, KAPPA0, kappa1, ell, grid=48, workers=1)
    is_s = sym == "Ar"
    q = charges(sym)
    ions = ~is_s
    med.set_medium(pos[is_s])
    e, _, _ = med.energy_forces(pos[ions], q[ions], pos[is_s])
    e_self = self_energy_discrete(KAPPA0, SIGMA, KMAX, L, None)
    return COUL * (e - e_self * np.sum(q**2)), c_at(med, pos[ions])


def c_field_task(job):
    sym, pos, ell = job
    med = ScreenedMedium(L, KAPPA0, 1.0, ell, grid=48, workers=1)
    med.set_medium(pos[sym == "Ar"])
    return np.clip(med.c_r.ravel(), 0, None)


def self_energy_share(e_per_atom, c_ions, n_knots=7):
    """LOO-scored fit of E/atom ~ sum_i f(c_i)/N with piecewise-linear f."""
    allc = np.concatenate(c_ions)
    knots = np.quantile(allc, np.linspace(0, 1, n_knots))
    n_atoms = 1146

    def hats(c):
        return np.stack(
            [np.interp(c, knots, np.eye(n_knots)[m]) for m in range(n_knots)], 1
        )

    X = np.stack([hats(c).sum(0) / n_atoms for c in c_ions])
    y = np.asarray(e_per_atom)
    if len(y) <= 2 * n_knots:
        return {
            "r2": None,
            "loo_r2": None,
            "unexplained_rms_frac": None,
            "note": f"{len(y)} frames for {n_knots} parameters: not fitted",
        }
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    H = X @ np.linalg.pinv(X)
    resid = y - X @ coef
    loo = resid / (1 - np.clip(np.diag(H), None, 1 - 1e-9))
    # Predictive scores use mean squared residuals, not residual variances.
    mse_loo = float(np.mean(loo**2))
    return {
        "r2": float(1 - np.mean(resid**2) / y.var()),
        "loo_r2": float(1 - mse_loo / y.var()),
        "unexplained_rms_frac": float(np.sqrt(mse_loo) / np.std(y)),
    }


# -- stage 2: probe -----------------------------------------------------------
def variant_symbols(sym, pos, A, B, kind, seed):
    """Reassign identities outside the frozen region around A and B."""
    rng = np.random.default_rng(seed)
    dA = np.linalg.norm(mic(pos - pos[A]), axis=1)
    dB = np.linalg.norm(mic(pos - pos[B]), axis=1)
    out = np.flatnonzero((dA > FROZEN_RADIUS) & (dB > FROZEN_RADIUS))
    new = sym.copy()
    if kind == "base":
        return new, len(out)
    pool = new[out]
    n_s, n_na = int(np.sum(pool == "Ar")), int(np.sum(pool == "Na"))
    rp = pos[out]
    if kind in ("wall", "antiwall"):
        mid = pos[A] + 0.5 * mic(pos[B] - pos[A])
        nrm = mic(pos[B] - pos[A])
        nrm /= np.linalg.norm(nrm)
        s = np.abs(mic(rp - mid) @ nrm)
        g = (s - L / 4) / (L / 8)
        v = 4.0 * (g if kind == "wall" else -g)
    elif kind.startswith("field"):
        v = 4.0 * random_field(rng, (1,), L)(rp)
    else:  # uniform
        v = np.zeros(len(out))
    is_s = metropolis_labels(rng, v, n_s)
    labels = np.empty(len(out), dtype=object)
    labels[is_s] = "Ar"
    ion_slots = np.flatnonzero(~is_s)
    rng.shuffle(ion_slots)
    labels[ion_slots[:n_na]] = "Na"
    labels[ion_slots[n_na:]] = "Cl"
    new[out] = labels
    return new, len(out)


def probe_task(job):
    tag, sym, pos, A, B, kind, seed, kappa1, ell, scale = job
    t0 = time.time()
    vsym, n_out = variant_symbols(sym, pos, A, B, kind, seed)
    med = ScreenedMedium(L, KAPPA0, kappa1, ell, grid=48, workers=1)
    med.set_medium(pos[vsym == "Ar"])
    n = len(pos)
    unit = FOUR_PI * med.V

    def rho_site(j):
        return med.density(pos[[j]], [1.0])

    rho_A, rho_B = rho_site(A), rho_site(B)
    rho_u = med.density(pos, np.full(n, 1.0 / n))
    x_A, x_B, x_u = med.solve(rho_A), med.solve(rho_B), med.solve(rho_u)
    G_AB = unit * np.vdot(rho_A, x_B).real
    G_AA = unit * np.vdot(rho_A, x_A).real
    G_BB = unit * np.vdot(rho_B, x_B).real
    G_Au = unit * np.vdot(rho_A, x_u).real
    G_uB = unit * np.vdot(rho_u, x_B).real
    G_uu = unit * np.vdot(rho_u, x_u).real
    diag_sum = 0.0
    for j in range(n):
        r = rho_site(j)
        diag_sum += unit * np.vdot(r, med.solve(r)).real
    qa, qb = charges(sym[[A]])[0], charges(sym[[B]])[0]
    pref = scale * COUL * 4 * qa * qb
    D = pref * (G_AB - G_Au - G_uB + G_uu)
    D_diag = pref * (-(G_AA + G_BB) / n + diag_sum / n**2)
    pair = pref * G_AB
    kap = med.kappa_field()
    return {
        "design": "i",
        "tag": tag,
        "kind": kind,
        "D": D,
        "D_diag": D_diag,
        "D_off": D - D_diag,
        "pair_term": pair,
        "background_term": D - D_diag - pair,
        "n_outside": n_out,
        "kappa_p5": float(np.percentile(kap, 5)),
        "kappa_p95": float(np.percentile(kap, 95)),
        "seconds": time.time() - t0,
    }


def compensated_sites(sym, pos, rng):
    """A plus B, C, D near the FCC-like offsets, q_B = -q_A, q_C = -q_A, q_D = q_A."""
    q = charges(sym)
    ions = np.flatnonzero(q != 0)
    A = rng.choice(ions)
    sites = [A]
    for off, sign in (((16, 16, 0), -1), ((16, 0, 16), -1), ((0, 16, 16), 1)):
        target = pos[A] + np.array(off, float)
        cand = ions[np.sign(q[ions]) == sign * np.sign(q[A])]
        d = np.linalg.norm(mic(pos[cand] - target), axis=1)
        sites.append(cand[np.argmin(d)])
    A, B, C, D = sites
    cross = [np.linalg.norm(mic(pos[i] - pos[j])) for i in (A, C) for j in (B, D)]
    return A, B, C, D, float(min(cross))


def plane_attraction(rp, pairs, pos):
    """Low values near the bisector planes of the given site pairs."""
    g = []
    for i, j in pairs:
        mid = pos[i] + 0.5 * mic(pos[j] - pos[i])
        nrm = mic(pos[j] - pos[i])
        nrm /= np.linalg.norm(nrm)
        g.append((np.abs(mic(rp - mid) @ nrm) - L / 4) / (L / 8))
    return np.min(g, axis=0)


def probe2_task(job):
    """Design (ii): neutral compensated swaps A<->C and B<->D."""
    tag, sym, pos, sites, kind, seed, kappa1, ell, scale = job
    A, B, C, D = sites
    rng = np.random.default_rng(seed)
    frozen = np.zeros(len(pos), bool)
    for s in sites:
        frozen |= np.linalg.norm(mic(pos - pos[s]), axis=1) <= FROZEN_RADIUS
    out = np.flatnonzero(~frozen)
    vsym = sym.copy()
    if kind != "base":
        pool = vsym[out]
        n_s, n_na = int(np.sum(pool == "Ar")), int(np.sum(pool == "Na"))
        rp = pos[out]
        if kind == "paths_AB_CD":
            v = 4.0 * plane_attraction(rp, ((A, B), (C, D)), pos)
        elif kind == "paths_AD_CB":
            v = 4.0 * plane_attraction(rp, ((A, D), (C, B)), pos)
        elif kind.startswith("field"):
            v = 4.0 * random_field(rng, (1,), L)(rp)
        else:
            v = np.zeros(len(out))
        is_s = metropolis_labels(rng, v, n_s)
        labels = np.empty(len(out), dtype=object)
        labels[is_s] = "Ar"
        slots = np.flatnonzero(~is_s)
        rng.shuffle(slots)
        labels[slots[:n_na]] = "Na"
        labels[slots[n_na:]] = "Cl"
        vsym[out] = labels
    med = ScreenedMedium(L, KAPPA0, kappa1, ell, grid=48, workers=1)
    med.set_medium(pos[vsym == "Ar"])
    unit = FOUR_PI * med.V
    rho = {s: med.density(pos[[s]], [1.0]) for s in sites}
    x_B, x_D = med.solve(rho[B]), med.solve(rho[D])
    G = {
        "AB": unit * np.vdot(rho[A], x_B).real,
        "CB": unit * np.vdot(rho[C], x_B).real,
        "AD": unit * np.vdot(rho[A], x_D).real,
        "CD": unit * np.vdot(rho[C], x_D).real,
    }
    qa, qb = charges(sym[[A]])[0], charges(sym[[B]])[0]
    pref = scale * COUL * 4 * qa * qb
    return {
        "design": "ii",
        "tag": tag,
        "kind": kind,
        "D": pref * (G["AB"] - G["AD"] - G["CB"] + G["CD"]),
        "terms": {k: pref * v for k, v in G.items()},
        "n_outside": int(len(out)),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--templates", required=True, help="2800 K extxyz (Na/Cl frames)")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--probe-frames", type=int, default=2)
    ap.add_argument(
        "--pairs", type=int, default=2, help="per probe frame, both designs"
    )
    ap.add_argument("--compensated-frames", type=int, default=10)
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="probe_trial.json")
    a = ap.parse_args()

    frames = [f for f in read(a.templates, index=slice(2, None, 15)) if len(f) == 1146][
        : a.frames
    ]
    print(f"{len(frames)} templates", flush=True)
    settings = [
        (1 / 4, 3.0, 4.0),
        (1 / 3, 3.0, 4.0),
        (1 / 4, 3.0, 2.0),
        (1 / 3, 3.0, 2.0),
    ]
    probe_settings = settings[:2]
    kinds = ["base", "wall", "antiwall", "field1", "field2", "uniform"]
    summary = {"settings": []}

    with mp.Pool(a.nproc) as pool:
        for si, (frac, ell, amp) in enumerate(settings):
            n_s = int(round(frac * 1146))
            n_s += (1146 - n_s) % 2
            syms = [
                relabel(
                    f.get_chemical_symbols(), f.positions % L, n_s, amp, 100 * si + t
                )
                for t, f in enumerate(frames)
            ]
            pos = [f.positions % L for f in frames]
            c = np.concatenate(
                pool.map(c_field_task, [(s, p, ell) for s, p in zip(syms, pos)])
            )
            kappa1 = float(np.sqrt((0.25 - KAPPA0**2) / np.percentile(c, 95)))
            kap = np.sqrt(KAPPA0**2 + kappa1**2 * c)
            res = pool.map(
                energy_task, [(s, p, kappa1, ell) for s, p in zip(syms, pos)]
            )
            e_atom = np.array([r[0] for r in res]) / 1146
            scale = 5e-3 / e_atom.std()
            share = self_energy_share(e_atom, [r[1] for r in res])
            entry = {
                "s_fraction": round(frac, 3),
                "n_s": n_s,
                "ell": ell,
                "amp_kT": amp,
                "kappa1": kappa1,
                "scale": float(scale),
                "kappa_p5": float(np.percentile(kap, 5)),
                "kappa_p50": float(np.percentile(kap, 50)),
                "vol_kappa_lt_0p10": float(np.mean(kap < 0.10)),
                "vol_kappa_gt_0p40": float(np.mean(kap > 0.40)),
                "e_scr_per_atom_std_meV_unscaled": float(1e3 * e_atom.std()),
                "self_energy_share": share,
            }
            print(json.dumps(entry), flush=True)

            if (frac, ell, amp) in probe_settings:
                jobs = []
                for t in range(a.probe_frames):
                    fidx = t * (len(frames) // a.probe_frames)
                    sym, p = syms[fidx], pos[fidx]
                    rng = np.random.default_rng(500 + 10 * si + t)
                    ions = np.flatnonzero(sym != "Ar")
                    d = np.linalg.norm(mic(p[ions][:, None] - p[ions][None]), axis=2)
                    q = charges(sym[ions])
                    cand = np.argwhere((d > 10) & (d < 16) & (q[:, None] * q[None] < 0))
                    for pi, k in enumerate(
                        rng.choice(len(cand), a.pairs, replace=False)
                    ):
                        A, B = ions[cand[k, 0]], ions[cand[k, 1]]
                        for vi, kind in enumerate(kinds):
                            jobs.append(
                                (
                                    f"set{si}_frame{fidx}_pair{pi}",
                                    sym,
                                    p,
                                    A,
                                    B,
                                    kind,
                                    1000 * si + 100 * t + 10 * pi + vi,
                                    kappa1,
                                    ell,
                                    scale,
                                )
                            )
                probes = pool.map(probe_task, jobs, chunksize=1)
                entry["probes"] = probes
                for tag in sorted({r["tag"] for r in probes}):
                    rows = [r for r in probes if r["tag"] == tag]
                    rng_ = {
                        k: 1e3 * np.ptp([r[k] for r in rows])
                        for k in ("D", "pair_term", "background_term", "D_diag")
                    }
                    print(
                        f"  (i) {tag}: range D {rng_['D']:7.2f} meV = pair {rng_['pair_term']:7.2f}"
                        f" / background {rng_['background_term']:7.2f} / diag {rng_['D_diag']:5.2f};  "
                        + " ".join(
                            f"{r['kind']}={1e3 * r['D']:.1f}({1e3 * r['pair_term']:.1f})"
                            for r in rows
                        ),
                        flush=True,
                    )

            kinds2 = [
                "base",
                "paths_AB_CD",
                "paths_AD_CB",
                "field1",
                "field2",
                "uniform",
            ]
            jobs2 = []
            for t in range(a.compensated_frames):
                fidx = t * (len(frames) // a.compensated_frames)
                sym, p = syms[fidx], pos[fidx]
                rng = np.random.default_rng(700 + 10 * si + t)
                for qi in range(a.pairs):
                    A, B, C, D, cross = compensated_sites(sym, p, rng)
                    if cross <= 9.0:
                        continue
                    for vi, kind in enumerate(kinds2):
                        jobs2.append(
                            (
                                f"set{si}_frame{fidx}_quad{qi}",
                                sym,
                                p,
                                (A, B, C, D),
                                kind,
                                2000 * si + 100 * t + 10 * qi + vi,
                                kappa1,
                                ell,
                                scale,
                            )
                        )
            probes2 = pool.map(probe2_task, jobs2, chunksize=1)
            entry["probes_compensated"] = probes2
            spans, mags = [], []
            for tag in sorted({r["tag"] for r in probes2}):
                rows = [r for r in probes2 if r["tag"] == tag]
                D = np.array([r["D"] for r in rows])
                spans.append(np.ptp(D))
                mags.append(np.max(np.abs([r["terms"]["AB"] for r in rows])))
                print(
                    f"  (ii) {tag}: range D {1e3 * np.ptp(D):7.2f} meV  |pair AB| <= "
                    f"{1e3 * mags[-1]:7.2f};  "
                    + " ".join(f"{r['kind']}={1e3 * r['D']:.1f}" for r in rows),
                    flush=True,
                )
            entry["compensated_range_meV_median"] = float(1e3 * np.median(spans))
            print(
                f"  (ii) median range of D over {len(spans)} quartets: "
                f"{1e3 * np.median(spans):.2f} meV",
                flush=True,
            )
            summary["settings"].append(entry)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
