"""Numerical checks of the design iii probe.

delta: rebuild the variants of selected sets from the probe configurations and
       write displaced configurations and reference K_AB for several displacement
       steps, one file pair per step, for eval_probe2.py / analyze_probe2.py.
mc:    run identity Monte Carlo chains for selected sets and variant kinds,
       recording the driver Yukawa energy every 25 sweeps and the reference
       energy and K_AB at sweep counts 200, 400, 800, 1600.
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
from build_probe2 import calc_for, displaced  # noqa: E402
from probe_mc import IdentityMC, frozen_mask  # noqa: E402

L = 32.0


def ref_K(sym, pos, A, B, delta, prm):
    calc = calc_for(prm)
    K = np.zeros((3, 3))
    for j, ax in enumerate("xyz"):
        f = {}
        for tag, p in displaced(pos, B, delta).items():
            if tag[0] == ax:
                c = Atoms(symbols=list(sym), positions=p, cell=[L] * 3, pbc=True)
                c.calc = calc
                f[tag[1]] = c.get_forces()[A]
        K[:, j] = -(f["+"] - f["-"]) / (2 * delta)
    return K


def delta_task(job):
    set_id, kind, sym, pos, A, B, delta, prm = job
    return set_id, kind, delta, ref_K(sym, pos, A, B, delta, prm).tolist()


def mc_task(job):
    from gen_heterogeneous import ShellField

    set_id, kind, sym0, pos, sites, seed, prm, marks = job
    rng = np.random.default_rng(seed)
    movable = ~frozen_mask(pos, sites, prm["frozen"])
    field, moves = None, ("swap_s", "swap_q")
    if kind == "field":
        field = ShellField(rng, 4.0, 2800.0).energy
    elif kind == "null":
        moves = ("swap_q",)
    mc = IdentityMC(pos, sym0, movable, field=field, rng=rng)
    e0 = mc.energy_yukawa()
    trace, snaps, done = [(0, 0.0)], {}, 0
    while done < marks[-1]:
        sym = mc.run(25, moves)
        done += 25
        trace.append((done, mc.energy_yukawa() - e0))
        if done in marks:
            calc = calc_for(prm)
            at = Atoms(symbols=list(sym), positions=pos, cell=[L] * 3, pbc=True)
            at.calc = calc
            snaps[done] = {
                "E_ref": float(at.get_potential_energy()),
                "K_ref": ref_K(
                    sym.copy(), pos, sites[0], sites[1], prm["delta"], prm
                ).tolist(),
            }
    return set_id, kind, trace, snaps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("check", choices=("delta", "mc"))
    ap.add_argument("--probe", required=True, help="probe2 directory")
    ap.add_argument("--data", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.025, 0.05, 0.10])
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    a = ap.parse_args()
    probe = Path(a.probe)
    refj = json.loads((probe / "probeiii_valid_reference.json").read_text())
    prm = refj["params"]
    sites = {s["set"]: s["sites"] for s in refj["sets"]}

    # undisplaced positions and variant symbols: the "dx+" configuration with B moved back
    var = {}
    for at in read(probe / "probeiii_valid_configs.xyz", index=":"):
        s, k = at.info["probe_set"], at.info["variant"]
        if s in a.sets and at.info["config"] == "dx+":
            pos = at.positions.copy()
            B = sites[s][1]
            pos[B, 0] -= prm["delta"]
            var[(s, k)] = (np.array(at.get_chemical_symbols(), dtype=object), pos % L)

    if a.check == "delta":
        jobs = [
            (s, k, sym, pos, sites[s][0], sites[s][1], d, prm)
            for (s, k), (sym, pos) in var.items()
            for d in a.deltas
        ]
        with mp.Pool(a.nproc) as pool:
            res = pool.map(delta_task, jobs, chunksize=1)
        for d in a.deltas:
            images = []
            for (s, k), (sym, pos) in var.items():
                A, B = sites[s]
                for tag, p in displaced(pos, B, d).items():
                    at = Atoms(symbols=list(sym), positions=p, cell=[L] * 3, pbc=True)
                    at.info.update(
                        probe_set=s, variant=k, config=f"d{tag}", probe_atom=A
                    )
                    images.append(at)
            tag = f"d{d:g}".replace(".", "p")
            write(probe / f"probeiii_check_{tag}_configs.xyz", images, format="extxyz")
            sub = {
                **refj,
                "params": {**prm, "delta": d},
                "sets": [x for x in refj["sets"] if x["set"] in a.sets],
                "variants": [
                    {"set": s, "kind": k, "K_ref": K} for s, k, dd, K in res if dd == d
                ],
            }
            (probe / f"probeiii_check_{tag}_reference.json").write_text(json.dumps(sub))
        K = {(s, k, d): np.array(v) for s, k, d, v in res}
        base = {(s, k): K[(s, k, 0.05)] for (s, k) in var} if 0.05 in a.deltas else None
        for d in a.deltas:
            if base is None or d == 0.05:
                continue
            diff = max(np.max(np.abs(K[(s, k, d)] - base[(s, k)])) for (s, k) in var)
            mod = max(
                np.max(
                    np.ptp(
                        np.array([K[(s, k, 0.05)] for (ss, k) in var if ss == s]),
                        axis=0,
                    )
                )
                for s in a.sets
            )
            print(
                f"reference: max |K(delta={d}) - K(0.05)| = {1e3 * diff:.4f} meV/A^2 "
                f"(largest within-set modulation {1e3 * mod:.3f} meV/A^2)"
            )
    else:
        kinds = ("field", "uniform", "null")
        marks = [200, 400, 800, 1600]
        rng = np.random.default_rng(7)
        jobs = [
            (
                s,
                k,
                var[(s, "base")][0],
                var[(s, "base")][1],
                sites[s],
                int(rng.integers(2**31)),
                prm,
                marks,
            )
            for s in a.sets
            for k in kinds
        ]
        with mp.Pool(a.nproc) as pool:
            res = pool.map(mc_task, jobs, chunksize=1)
        out = []
        for s, k, trace, snaps in res:
            t = np.array(trace)
            late = t[t[:, 0] > 400, 1]
            mid = t[(t[:, 0] > 200) & (t[:, 0] <= 400), 1]
            Ks = {m: np.array(v["K_ref"]) for m, v in snaps.items()}
            dK = {m: float(1e3 * np.max(np.abs(Ks[m] - Ks[1600]))) for m in marks}
            dE = {m: float(snaps[m]["E_ref"] - snaps[1600]["E_ref"]) for m in marks}
            print(
                f"{s} {k:<8}: Yukawa dE mean 200-400 sw {mid.mean():+7.2f}, 400-1600 sw {late.mean():+7.2f} "
                f"(sd {late.std():.2f}) eV; |K(m) - K(1600)| meV/A^2: "
                + " ".join(f"{m}:{dK[m]:.3f}" for m in marks[:-1])
                + "; E_ref(m) - E_ref(1600) eV: "
                + " ".join(f"{m}:{dE[m]:+.1f}" for m in marks[:-1])
            )
            out.append({"set": s, "kind": k, "trace": trace, "snapshots": snaps})
        (probe / "probeiii_check_mc.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
