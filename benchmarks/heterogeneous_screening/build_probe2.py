"""Remote-screening probe with equilibrated variants (protocol Sec. 9a).

Design iii (primary): cross force constant between two ions A and B, 10-12 A apart,

    K_AB[a, b] = d^2 E / dR_A,a dR_B,b = -(F_A,a(R_B + d e_b) - F_A,a(R_B - d e_b)) / (2 d),

from six configurations per variant (B displaced by +-d along x, y, z).
Design ii (secondary): the four-site charge-swap difference D_AB of build_probe.py.

Remote variants (probe_mc.py): identities outside the frozen region resampled by
Monte Carlo under the driver energy: base, fieldN (new shell-1 field, A = 4 kT),
uniform (no field), nullN (Na/Cl only, S fixed; the reference does not change).

Writes the configurations for model evaluation, the reference observables (K from
exact reference forces, D from unit-charge solves), and the reference energy of
every unswapped, undisplaced variant, to check that the variants are in
distribution.
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
from build_probe import choose_sites, states  # noqa: E402
from probe_mc import make_variant  # noqa: E402
from screening_reference import (  # noqa: E402
    COUL,
    FOUR_PI,
    SCREENING_SYMBOL,
    HeterogeneousScreeningCalculator,
    ScreenedMedium,
    reference_charges,
)

L = 32.0


def mic(d):
    return d - L * np.round(d / L)


def choose_pair(sym, pos, rng, dmin=10.0, dmax=12.0):
    ions = np.flatnonzero(reference_charges(sym) != 0)
    for _ in range(1000):
        A = int(rng.choice(ions))
        d = np.linalg.norm(mic(pos[ions] - pos[A]), axis=1)
        cand = ions[(d > dmin) & (d < dmax)]
        if len(cand):
            B = int(rng.choice(cand))
            return (A, B), float(np.linalg.norm(mic(pos[B] - pos[A])))
    raise RuntimeError("no pair found")


def displaced(pos, B, delta):
    out = {}
    for b, ax in enumerate("xyz"):
        for sgn, tag in ((1, "+"), (-1, "-")):
            p = pos.copy()
            p[B, b] += sgn * delta
            out[ax + tag] = p % L
    return out


def calc_for(prm):
    return HeterogeneousScreeningCalculator(
        L,
        prm["kappa0"],
        prm["kappa1"],
        prm["ell"],
        scale=prm["scale"],
        grid=48,
        workers=1,
    )


def task(job):
    design, set_id, kind, seed, sym0, pos, sites, prm = job
    rng = np.random.default_rng(seed)
    sym, diag = make_variant(
        sym0, pos, sites, kind, rng, radius=prm["frozen"], sweeps=prm["sweeps"]
    )
    calc = calc_for(prm)
    at = Atoms(symbols=list(sym), positions=pos, cell=[L] * 3, pbc=True)
    at.calc = calc
    rec = {
        "design": design,
        "set": set_id,
        "kind": kind,
        "E_ref_00": float(at.get_potential_energy()),
        **diag,
    }
    if design == "iii":
        A, B = sites
        K = np.zeros((3, 3))
        for b, ax in enumerate("xyz"):
            f = {}
            for tag, p in displaced(pos, B, prm["delta"]).items():
                if tag[0] != ax:
                    continue
                c = Atoms(symbols=list(sym), positions=p, cell=[L] * 3, pbc=True)
                c.calc = calc
                f[tag[1]] = c.get_forces()[A]
            K[:, b] = -(f["+"] - f["-"]) / (2 * prm["delta"])
        rec["K_ref"] = K.tolist()
    else:
        A, B, C, D = sites
        med = ScreenedMedium(
            L, prm["kappa0"], prm["kappa1"], prm["ell"], grid=48, workers=1
        )
        med.set_medium(pos[sym == SCREENING_SYMBOL])
        rho = {s: med.density(pos[[s]], [1.0]) for s in sites}
        xB, xD = med.solve(rho[B]), med.solve(rho[D])
        unit = FOUR_PI * med.V
        G = {
            k: unit * np.vdot(rho[x], v).real
            for k, x, v in (("AB", A, xB), ("CB", C, xB), ("AD", A, xD), ("CD", C, xD))
        }
        q = reference_charges(sym)
        rec["D_ref"] = float(
            prm["scale"]
            * COUL
            * 4
            * q[A]
            * q[B]
            * (G["AB"] - G["AD"] - G["CB"] + G["CD"])
        )
    return rec, sym


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--design", required=True, choices=("ii", "iii"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--frame-in-traj", type=int, default=7)
    ap.add_argument("--sets-per-frame", type=int, default=2)
    ap.add_argument("--n-field", type=int, default=4)
    ap.add_argument("--n-null", type=int, default=2)
    ap.add_argument("--frozen", type=float, default=10.0)
    ap.add_argument("--sweeps", type=int, default=100)
    ap.add_argument(
        "--delta", type=float, default=0.05, help="displacement of B (A), design iii"
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
        "sweeps": a.sweeps,
        "delta": a.delta,
    }
    frames = [
        f
        for f in read(data / f"hetscreen_{a.split}.xyz", index=":")
        if f.info.get("config_type") != "IsolatedAtom"
    ]
    by_traj = {}
    for f in frames:
        by_traj.setdefault(int(f.info["traj"]), []).append(f)
    rng = np.random.default_rng(a.seed + (0 if a.design == "iii" else 1))
    kinds = (
        ["base"]
        + [f"field{i}" for i in range(a.n_field)]
        + ["uniform"]
        + [f"null{i}" for i in range(a.n_null)]
    )
    jobs, sets, base_E = [], [], {}
    for t in sorted(by_traj)[: a.n_frames]:
        fr = by_traj[t][a.frame_in_traj]
        sym = np.array(fr.get_chemical_symbols(), dtype=object)
        pos = fr.positions % L
        base_E[t] = [float(f.get_potential_energy()) for f in by_traj[t]]
        for k in range(a.sets_per_frame):
            if a.design == "iii":
                sites, dist = choose_pair(sym, pos, rng)
            else:
                sites, dist = choose_sites(sym, pos, rng)
            set_id = f"traj{t:03d}_set{k}"
            sets.append(
                {"set": set_id, "traj": t, "sites": list(sites), "distance_A": dist}
            )
            for kind in kinds:
                jobs.append(
                    (
                        a.design,
                        set_id,
                        kind,
                        int(rng.integers(2**31)),
                        sym,
                        pos,
                        sites,
                        prm,
                    )
                )
    with mp.Pool(a.nproc) as pool:
        res = pool.map(task, jobs, chunksize=1)

    images = []
    for job, (rec, sym) in zip(jobs, res):
        _, set_id, kind, _, _, pos, sites, _ = job
        if a.design == "iii":
            A, B = sites
            for tag, p in displaced(pos, B, a.delta).items():
                at = Atoms(symbols=list(sym), positions=p, cell=[L] * 3, pbc=True)
                at.info.update(
                    probe_set=set_id, variant=kind, config=f"d{tag}", probe_atom=A
                )
                images.append(at)
        else:
            for (fa, fb), s in states(sym, sites).items():
                at = Atoms(symbols=list(s), positions=pos, cell=[L] * 3, pbc=True)
                at.info.update(probe_set=set_id, variant=kind, config=f"s{fa}{fb}")
                images.append(at)
    write(out / f"probe{a.design}_{a.split}_configs.xyz", images, format="extxyz")
    recs = [r for r, _ in res]
    # in-distribution check: unswapped, undisplaced variants against the spread of
    # reference energies along the base frame's own trajectory
    base_of = {r["set"]: r["E_ref_00"] for r in recs if r["kind"] == "base"}
    traj_of = {s["set"]: s["traj"] for s in sets}
    dE = {
        k: [r["E_ref_00"] - base_of[r["set"]] for r in recs if r["kind"].startswith(k)]
        for k in ("field", "uniform", "null")
    }
    spread = float(np.mean([np.std(base_E[traj_of[s]]) for s in base_of]))
    summary = {
        "design": a.design,
        "split": a.split,
        "params": prm,
        "sets": sets,
        "variants": recs,
        "in_distribution": {
            "traj_energy_std_eV": spread,
            **{f"{k}_dE_mean_eV": float(np.mean(v)) for k, v in dE.items()},
            **{
                f"{k}_dE_rms_eV": float(np.sqrt(np.mean(np.square(v))))
                for k, v in dE.items()
            },
        },
    }
    (out / f"probe{a.design}_{a.split}_reference.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        f"design {a.design}: {len(sets)} sets x {len(kinds)} variants, {len(images)} configurations"
    )
    print(
        "in-distribution check: E_ref(variant) - E_ref(base), mean / RMS (eV): "
        + "  ".join(
            f"{k} {np.mean(v):+.2f} / {np.sqrt(np.mean(np.square(v))):.2f}"
            for k, v in dE.items()
        )
        + f";  energy std along a trajectory {spread:.2f} eV"
    )
    acc = [r.get("acceptance") for r in recs if r.get("acceptance") is not None]
    print(f"MC acceptance {np.mean(acc):.3f}")
    obs = "K_ref" if a.design == "iii" else "D_ref"
    for s in sets:
        vals = [
            np.array(r[obs])
            for r in recs
            if r["set"] == s["set"] and not r["kind"].startswith("null")
        ]
        nulls = [
            np.array(r[obs])
            for r in recs
            if r["set"] == s["set"] and r["kind"].startswith("null")
        ]
        base = [
            np.array(r[obs])
            for r in recs
            if r["set"] == s["set"] and r["kind"] == "base"
        ][0]
        span = np.max(np.ptp(np.array(vals), axis=0))
        null = max(np.max(np.abs(n - base)) for n in nulls)
        unit, fac = ("meV/A^2", 1e3) if a.design == "iii" else ("meV", 1e3)
        print(
            f"{s['set']}: d = {s['distance_A']:.1f} A, max range {fac * span:7.3f} {unit}, "
            f"|base| {fac * np.max(np.abs(base)):7.3f}, null {fac * null:.1e}"
        )


if __name__ == "__main__":
    main()
