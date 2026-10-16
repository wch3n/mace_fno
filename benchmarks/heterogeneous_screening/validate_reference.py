"""Validation suite for the heterogeneous-screening reference (protocol Sec. 4).

Runs on a test configuration with strong S segregation (sites ranked against a
smooth field), the worst case for solver conditioning:

  uniform   c_S = 1 reproduces gen_yukawa's unsubtracted Yukawa sum and forces
  grid      energies and forces agree between 48^3, 64^3 and 96^3 grids
  fd        forces on ions and S particles match central differences
  symmetry  rigid translation and 90 degree rotations; total force vanishes
  tolerance tightening the CG tolerance leaves the forces unchanged
  flip      the four-state charge-flip difference equals a^T (P G P) b

Writes a JSON summary and exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from screening_reference import (
    COUL,
    SCREENING_SYMBOL,
    HeterogeneousScreeningCalculator,
    reference_charges,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic_yukawa"))
from gen_yukawa import (  # noqa: E402
    init_positions,
    kernel,
    kgrid,
    lj_energy_forces,
    nl_energy_forces,
)


def test_configuration(L, n_ion_pairs, n_s, seed):
    """Random packing; S on the sites with the largest smooth-field values."""
    rng = np.random.default_rng(seed)
    n = 2 * n_ion_pairs + n_s
    pos = init_positions(rng, n, L, dmin=2.3)
    field = np.cos(2 * np.pi * pos[:, 0] / L) + 0.5 * np.cos(
        2 * np.pi * (pos[:, 1] + pos[:, 2]) / L
    )
    order = np.argsort(-field)
    sym = np.empty(n, dtype=object)
    sym[order[:n_s]] = SCREENING_SYMBOL
    rest = order[n_s:].copy()
    rng.shuffle(rest)
    sym[rest[:n_ion_pairs]] = "Na"
    sym[rest[n_ion_pairs:]] = "Cl"
    return Atoms(symbols=list(sym), positions=pos, cell=[L] * 3, pbc=True)


def evaluate(atoms, **kw):
    a = atoms.copy()
    a.calc = HeterogeneousScreeningCalculator(**kw)
    e = a.get_potential_energy()
    return e, a.get_forces(), dict(a.calc.results)


def screening_forces(atoms, forces, settings):
    """Total forces minus the LJ part."""
    _, f_lj = lj_energy_forces(
        atoms.get_positions() % settings["L"], settings["L"], 0.20, 3.0, 4.5, 3.6
    )
    return forces - f_lj


def max_rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--L", type=float, default=32.0)
    ap.add_argument("--kappa0", type=float, default=0.05)
    ap.add_argument("--kappa1", type=float, default=0.35)
    ap.add_argument("--ell", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--n-screening",
        type=int,
        default=572,
        help="S particles; the rest are equal numbers of Na and Cl",
    )
    ap.add_argument("--fd-atoms", type=int, default=6, help="per species group")
    ap.add_argument("--fd-step", type=float, default=1e-4, help="ions (A)")
    ap.add_argument("--fd-step-s", type=float, default=1e-3, help="S particles (A)")
    ap.add_argument("--out", default="validation_summary.json")
    a = ap.parse_args()

    settings = dict(L=a.L, kappa0=a.kappa0, kappa1=a.kappa1, ell=a.ell, scale=1.0)
    atoms = test_configuration(a.L, (1146 - a.n_screening) // 2, a.n_screening, a.seed)
    sym = np.array(atoms.get_chemical_symbols())
    is_s = sym == SCREENING_SYMBOL
    q0 = reference_charges(sym)
    results, ok = {"n_screening": a.n_screening}, True

    def report(name, passed, **vals):
        nonlocal ok
        ok &= bool(passed)
        results[name] = {"pass": bool(passed), **vals}
        shown = "  ".join(
            f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in vals.items()
        )
        print(f"{name:10s} {'PASS' if passed else 'FAIL'}  {shown}", flush=True)

    t0 = time.time()
    e64, f64, r64 = evaluate(atoms, grid=64, **settings)
    results["timing_s_per_eval"] = time.time() - t0
    results["cg_iterations"] = int(r64["cg_iterations"])

    # uniform limit against gen_yukawa (unsubtracted pair sum, same modes)
    kap = np.hypot(a.kappa0, a.kappa1)
    eu, fu, ru = evaluate(atoms, grid=64, uniform=True, **settings)
    kvec, kmag, _ = kgrid(a.L, 3.5)
    w = COUL * kernel(kmag, kap) * np.exp(-(0.6**2) * kmag**2)
    ions = ~is_s
    pos = atoms.get_positions() % a.L
    e_ref, f_ref = nl_energy_forces(pos[ions], q0[ions], a.L**3, kvec, w, 0.0)
    fs_u = screening_forces(atoms, fu, settings)
    report(
        "uniform",
        abs(ru["e_screened_unsubtracted"] - e_ref) <= 1e-10 * abs(e_ref)
        and max_rel(fs_u[ions], f_ref) <= 1e-10
        and np.max(np.abs(fs_u[is_s])) <= 1e-12,
        energy_rel=abs(ru["e_screened_unsubtracted"] - e_ref) / abs(e_ref),
        force_rel=max_rel(fs_u[ions], f_ref),
        s_force_max=float(np.max(np.abs(fs_u[is_s]))),
    )

    # grid independence
    worst_e, worst_f = 0.0, 0.0
    for grid in (48, 96):
        e, f, r = evaluate(atoms, grid=grid, **settings)
        worst_e = max(
            worst_e, abs(r["e_screened"] - r64["e_screened"]) / abs(r64["e_screened"])
        )
        worst_f = max(worst_f, max_rel(f, f64))
    report(
        "grid",
        worst_e <= 1e-10 and worst_f <= 1e-10,
        energy_rel=worst_e,
        force_rel=worst_f,
    )

    # finite differences of E_screened (ions and S particles, all directions).
    # Mixed tolerance: double-precision noise on E_scr (~1e-12 eV) sets an
    # absolute floor of ~1e-8 eV/A at h = 1e-4. S forces vary on the scale l,
    # so a larger step tests them more sharply.
    fs64 = screening_forces(atoms, f64, settings)
    rng = np.random.default_rng(a.seed + 1)
    picks = np.concatenate(
        [
            rng.choice(np.flatnonzero(ions), a.fd_atoms, replace=False),
            rng.choice(np.flatnonzero(is_s), a.fd_atoms, replace=False),
        ]
    )
    rtol, atol = 1e-7, 1e-8
    worst_abs, worst_ratio = 0.0, 0.0
    for i in picks:
        h = a.fd_step_s if is_s[i] else a.fd_step
        for d in range(3):
            vals = []
            for sgn in (1, -1):
                b = atoms.copy()
                b.positions[i, d] += sgn * h
                vals.append(evaluate(b, grid=64, **settings)[2]["e_screened"])
            err = abs(-(vals[0] - vals[1]) / (2 * h) - fs64[i, d])
            worst_abs = max(worst_abs, err)
            worst_ratio = max(worst_ratio, err / (atol + rtol * abs(fs64[i, d])))
    report(
        "fd",
        worst_ratio <= 1.0,
        worst_abs_err=worst_abs,
        worst_err_over_tol=worst_ratio,
        n_derivatives=int(3 * len(picks)),
    )

    # symmetry: translation, rotations about z and x, total force
    shift = rng.uniform(0, a.L, 3)
    b = atoms.copy()
    b.positions = (b.positions + shift) % a.L
    et, ft, rt = evaluate(b, grid=64, **settings)
    rot_err_e, rot_err_f = 0.0, 0.0
    for R in (
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    ):
        c = atoms.copy()
        c.positions = (c.positions @ R.T) % a.L
        er, fr, rr = evaluate(c, grid=64, **settings)
        rot_err_e = max(
            rot_err_e,
            abs(rr["e_screened"] - r64["e_screened"]) / abs(r64["e_screened"]),
        )
        rot_err_f = max(rot_err_f, max_rel(fr, f64 @ R.T))
    tr_e = abs(rt["e_screened"] - r64["e_screened"]) / abs(r64["e_screened"])
    tr_f = max_rel(ft, f64)
    total = float(np.linalg.norm(fs64.sum(axis=0)))
    report(
        "symmetry",
        max(tr_e, rot_err_e) <= 1e-10
        and max(tr_f, rot_err_f) <= 1e-9
        and total <= 1e-9 * np.abs(fs64).sum(),
        translation_e=tr_e,
        translation_f=tr_f,
        rotation_e=rot_err_e,
        rotation_f=rot_err_f,
        total_force=total,
    )

    # solver tolerance
    _, f10, _ = evaluate(atoms, grid=64, tol=1e-10, **settings)
    _, f13, _ = evaluate(atoms, grid=64, tol=1e-13, **settings)
    dtol = float(np.max(np.abs(f10 - f13)))
    report("tolerance", dtol < 1e-8, max_force_change=dtol)

    # four-state charge flip with site neutralization
    ion_idx = np.flatnonzero(ions)
    d = pos[ion_idx][:, None, :] - pos[ion_idx][None, :, :]
    d -= a.L * np.round(d / a.L)
    r = np.linalg.norm(d, axis=2)
    cand = np.argwhere(
        (r > 10) & (r < 16) & (q0[ion_idx][:, None] * q0[ion_idx][None, :] < 0)
    )
    A, B = ion_idx[cand[0, 0]], ion_idx[cand[0, 1]]
    swap = {"Na": "Cl", "Cl": "Na"}
    energies = {}
    for fa in (0, 1):
        for fb in (0, 1):
            s = list(sym)
            if fa:
                s[A] = swap[s[A]]
            if fb:
                s[B] = swap[s[B]]
            c = Atoms(symbols=s, positions=atoms.positions, cell=atoms.cell, pbc=True)
            energies[(fa, fb)] = evaluate(c, grid=64, neutralize=True, **settings)[0]
    D = energies[(1, 1)] - energies[(1, 0)] - energies[(0, 1)] + energies[(0, 0)]
    calc = HeterogeneousScreeningCalculator(grid=64, **settings)
    med = calc.medium
    med.set_medium(pos[is_s])
    n = len(atoms)
    va = np.full(n, 2 * q0[A] / n)
    va[A] -= 2 * q0[A]
    vb = np.full(n, 2 * q0[B] / n)
    vb[B] -= 2 * q0[B]
    D_pred = COUL * med.green(pos, va, pos, vb)
    report(
        "flip",
        abs(D - D_pred) <= 1e-8 * abs(D_pred) + 1e-9,
        D=float(D),
        D_pred=float(D_pred),
        rel_err=abs(D - D_pred) / abs(D_pred),
        r_AB=float(
            np.linalg.norm((pos[A] - pos[B]) - a.L * np.round((pos[A] - pos[B]) / a.L))
        ),
    )

    kfield = med.kappa_field()
    results["kappa_percentiles_5_50_95"] = [
        float(np.percentile(kfield, p)) for p in (5, 50, 95)
    ]
    results["settings"] = settings
    results["all_pass"] = bool(ok)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}; summary in {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
