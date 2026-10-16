"""Build design (ii) remote-screening probe sets and their reference D_AB (protocol Sec. 9).

From one frame per trajectory of a split (default: validation), two four-site sets
(A, B, C, D) are chosen at FCC-like offsets from A, with q_B = q_C = -q_A and
q_D = q_A, so that every cross-group distance exceeds 2 r_max. Everything within
--frozen A of the four sites keeps its species. Outside, identities are reassigned
at fixed positions:

  base       the frame itself
  fieldN     remote S placed by swap Metropolis in independent shell-1 fields (A = 4 kT)
  uniform    remote S placed at random
  nullN      remote Na/Cl identities shuffled, S unchanged (the reference D_AB is
             unchanged by construction; a model that responds is reacting to remote
             charges, not to the medium)

For each variant the four states (original, A<->C, B<->D, both) are written for model
evaluation, and the reference

  D_AB = E(both) - E(A<->C) - E(B<->D) + E(orig) = 4 q_A q_B (G_AB - G_AD - G_CB + G_CD)

is computed from two unit-charge solves. A few variants are checked against the four
reference energies.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read, write  # noqa: E402
from gen_heterogeneous import ShellField, metropolis_labels  # noqa: E402
from screening_reference import (  # noqa: E402
    COUL,
    FOUR_PI,
    SCREENING_SYMBOL,
    HeterogeneousScreeningCalculator,
    ScreenedMedium,
    reference_charges,
)

L = 32.0
OFFSETS = (
    ((16, 16, 0), -1),
    ((16, 0, 16), -1),
    ((0, 16, 16), 1),
)  # B, C, D relative to A


def mic(d):
    return d - L * np.round(d / L)


def choose_sites(sym, pos, rng, min_cross=9.0, tries=200):
    q = reference_charges(sym)
    ions = np.flatnonzero(q != 0)
    for _ in range(tries):
        A = int(rng.choice(ions))
        sites = [A]
        for off, sign in OFFSETS:
            cand = ions[np.sign(q[ions]) == sign * np.sign(q[A])]
            d = np.linalg.norm(mic(pos[cand] - (pos[A] + np.array(off, float))), axis=1)
            sites.append(int(cand[np.argmin(d)]))
        A, B, C, D = sites
        cross = min(
            np.linalg.norm(mic(pos[i] - pos[j])) for i in (A, C) for j in (B, D)
        )
        if cross > min_cross and len(set(sites)) == 4:
            return (A, B, C, D), float(cross)
    raise RuntimeError("no admissible site set found")


def variant_symbols(sym, pos, sites, kind, rng, frozen_radius):
    frozen = np.zeros(len(pos), bool)
    for s in sites:
        frozen |= np.linalg.norm(mic(pos - pos[s]), axis=1) <= frozen_radius
    out = np.flatnonzero(~frozen)
    new = sym.copy()
    pool = new[out]
    if kind == "base":
        return new, int(len(out))
    if kind.startswith("null"):
        ion_slots = out[pool != SCREENING_SYMBOL]
        labels = new[ion_slots].copy()
        rng.shuffle(labels)
        new[ion_slots] = labels
        return new, int(len(out))
    n_s, n_na = int(np.sum(pool == SCREENING_SYMBOL)), int(np.sum(pool == "Na"))
    if kind.startswith("field"):
        v = ShellField(rng, 4.0, 2800.0).energy(pos[out]) / (8.617333e-5 * 2800.0)
    else:
        v = np.zeros(len(out))
    is_s = metropolis_labels(rng, v, n_s)
    labels = np.empty(len(out), dtype=object)
    labels[is_s] = SCREENING_SYMBOL
    slots = np.flatnonzero(~is_s)
    rng.shuffle(slots)
    labels[slots[:n_na]] = "Na"
    labels[slots[n_na:]] = "Cl"
    new[out] = labels
    return new, int(len(out))


def states(sym, sites):
    A, B, C, D = sites
    out = {}
    for fa in (0, 1):
        for fb in (0, 1):
            s = sym.copy()
            if fa:
                s[A], s[C] = s[C], s[A]
            if fb:
                s[B], s[D] = s[D], s[B]
            out[(fa, fb)] = s
    return out


def task(job):
    set_id, kind, seed, sym0, pos, sites, prm = job
    rng = np.random.default_rng(seed)
    sym, n_out = variant_symbols(sym0, pos, sites, kind, rng, prm["frozen"])
    med = ScreenedMedium(
        L, prm["kappa0"], prm["kappa1"], prm["ell"], grid=48, workers=1
    )
    med.set_medium(pos[sym == SCREENING_SYMBOL])
    A, B, C, D = sites
    rho = {s: med.density(pos[[s]], [1.0]) for s in sites}
    xB, xD = med.solve(rho[B]), med.solve(rho[D])
    unit = FOUR_PI * med.V
    G = {
        "AB": unit * np.vdot(rho[A], xB).real,
        "CB": unit * np.vdot(rho[C], xB).real,
        "AD": unit * np.vdot(rho[A], xD).real,
        "CD": unit * np.vdot(rho[C], xD).real,
    }
    q = reference_charges(sym)
    pref = prm["scale"] * COUL * 4 * q[A] * q[B]
    d_ref = pref * (G["AB"] - G["AD"] - G["CB"] + G["CD"])
    kap = med.kappa_field()
    rec = {
        "set": set_id,
        "kind": kind,
        "D_ref": float(d_ref),
        "n_remote": n_out,
        "terms": {k: float(pref * v) for k, v in G.items()},
        "kappa_p5": float(np.percentile(kap, 5)),
        "kappa_p95": float(np.percentile(kap, 95)),
    }
    if prm["check"]:
        calc = HeterogeneousScreeningCalculator(
            L,
            prm["kappa0"],
            prm["kappa1"],
            prm["ell"],
            scale=prm["scale"],
            grid=48,
            workers=1,
        )
        e = {}
        for key, s in states(sym, sites).items():
            at = Atoms(symbols=list(s), positions=pos, cell=[L] * 3, pbc=True)
            at.calc = calc
            e[key] = at.get_potential_energy()
        rec["D_ref_from_energies"] = float(
            e[(1, 1)] - e[(1, 0)] - e[(0, 1)] + e[(0, 0)]
        )
    return rec, sym


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data",
        required=True,
        help="dataset directory (hetscreen_*.xyz, metadata.json)",
    )
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--frame-in-traj", type=int, default=7)
    ap.add_argument("--sets-per-frame", type=int, default=2)
    ap.add_argument("--n-field", type=int, default=4)
    ap.add_argument("--n-null", type=int, default=2)
    ap.add_argument("--frozen", type=float, default=10.0)
    ap.add_argument(
        "--n-check",
        type=int,
        default=3,
        help="variants checked with four reference energies",
    )
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = json.loads((data / "metadata.json").read_text())["labels"]
    prm = {
        "kappa0": lab["kappa0"],
        "kappa1": lab["kappa1"],
        "ell": lab["ell"],
        "scale": lab["scale"],
        "frozen": a.frozen,
    }
    frames = [
        f
        for f in read(data / f"hetscreen_{a.split}.xyz", index=":")
        if f.info.get("config_type") != "IsolatedAtom"
    ]
    by_traj = {}
    for f in frames:
        by_traj.setdefault(int(f.info["traj"]), []).append(f)
    trajs = sorted(by_traj)[: a.n_frames]
    rng = np.random.default_rng(a.seed)
    kinds = (
        ["base"]
        + [f"field{i}" for i in range(a.n_field)]
        + ["uniform"]
        + [f"null{i}" for i in range(a.n_null)]
    )
    jobs, sets = [], []
    for t in trajs:
        fr = by_traj[t][a.frame_in_traj]
        sym = np.array(fr.get_chemical_symbols(), dtype=object)
        pos = fr.positions % L
        for k in range(a.sets_per_frame):
            sites, cross = choose_sites(sym, pos, rng)
            set_id = f"traj{t:03d}_set{k}"
            sets.append(
                {
                    "set": set_id,
                    "traj": t,
                    "time_ps": float(fr.info["time_ps"]),
                    "sites": list(sites),
                    "min_cross_A": cross,
                }
            )
            for kind in kinds:
                jobs.append(
                    (set_id, kind, int(rng.integers(2**31)), sym, pos, sites, prm)
                )
    for i, j in enumerate(jobs):
        jobs[i] = j[:-1] + (
            {
                **prm,
                "check": i < a.n_check * len(kinds)
                and j[1] in ("base", "field0", "null0"),
            },
        )
    with mp.Pool(a.nproc) as pool:
        res = pool.map(task, jobs, chunksize=1)

    images = []
    site_of = {s["set"]: s["sites"] for s in sets}
    for (set_id, kind, _, _, pos, _, _), (rec, sym) in zip(jobs, res):
        for (fa, fb), s in states(sym, site_of[set_id]).items():
            at = Atoms(symbols=list(s), positions=pos, cell=[L] * 3, pbc=True)
            at.info.update(probe_set=set_id, variant=kind, state=f"{fa}{fb}")
            images.append(at)
    write(out / f"probe_{a.split}_states.xyz", images, format="extxyz")
    checks = [r for r, _ in res if "D_ref_from_energies" in r]
    summary = {
        "split": a.split,
        "data": str(data),
        "params": prm,
        "sets": sets,
        "variants": [r for r, _ in res],
        "check_max_abs_meV": float(
            1e3 * max(abs(r["D_ref"] - r["D_ref_from_energies"]) for r in checks)
        )
        if checks
        else None,
    }
    (out / f"probe_{a.split}_reference.json").write_text(json.dumps(summary, indent=2))
    D = {}
    for r, _ in res:
        D.setdefault(r["set"], {})[r["kind"]] = r["D_ref"]
    print(f"{len(sets)} sets x {len(kinds)} variants, {len(images)} states written")
    print(
        f"four-energy check on {len(checks)} variants: max |dD| = {summary['check_max_abs_meV']:.2e} meV"
    )
    for s, d in D.items():
        rng_all = 1e3 * np.ptp([v for k, v in d.items() if not k.startswith("null")])
        rng_nb = 1e3 * np.ptp(
            [v for k, v in d.items() if k not in ("base",) and not k.startswith("null")]
        )
        null = 1e3 * max(abs(d[k] - d["base"]) for k in d if k.startswith("null"))
        print(
            f"{s}: D_ref range {rng_all:6.1f} meV (without base {rng_nb:6.1f}); null |dD| {null:.1e} meV"
        )


if __name__ == "__main__":
    main()
