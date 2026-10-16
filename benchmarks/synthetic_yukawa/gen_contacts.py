#!/usr/bin/env python3
"""Close-contact configurations for the synthetic MD training data.

Models trained only on the 2800 K data turn attractive just below the closest
pair separations sampled there (2.58 A for Na-Cl and 2.62 A for like pairs),
which MD at 2800 K occasionally reaches. These frames fill that gap.

Each frame is a 2800 K configuration in which the closest pair of one type
(Na-Cl, Na-Na and Cl-Cl in turn) is moved symmetrically along its axis to a
separation r ~ U(2.25, 2.55) A, with all other atoms fixed. A move is rejected
if either atom comes closer than 2.55 A to a third atom. Frames are labelled
with the analytical reference of the dataset, E_LJ + s E_Yukawa at the
dataset's kappa and fixed scale s, after checking that the calculator
reproduces the stored labels of unmodified frames.

Training contacts come from training frames and the contact test set from
test frames. The thermal validation and test sets are not modified.

Inputs, per kappa tag (default 0p05 and 0p5):
    <data-root>/data_k<tag>/kappa<tag>_{train,test}.xyz and metadata.json
    (written by gen_yukawa.py with --fixed-scale, see prepare_dataset.sh)
Outputs, per tag, in <out-root>/data_k<tag>/:
    kappa<tag>_contact_train.xyz   150 frames
    kappa<tag>_contact_test.xyz     45 frames
    kappa<tag>_train_contact.xyz    the training file followed by the contacts
and <out-root>/data_contact_summary.json.

Usage:
    python3 gen_contacts.py --data-root DATA_ROOT/md2800 [--out-root DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
R_LO, R_HI, R_OTHER = 2.25, 2.55, 2.55
N_TRAIN, N_TEST, SEED = 150, 45, 20260927
TYPES = [("Na", "Cl"), ("Na", "Na"), ("Cl", "Cl")]
# The random stream of each kappa is SEED + offset. These offsets reproduce the
# published contact sets and must not change.
SEED_OFFSETS = {"0p05": 0, "0p5": 1}


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_yukawa", HERE / "gen_yukawa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_calculator(gy, data_dir: Path):
    """The analytical reference that drove and labelled the fixed-scale dataset."""
    meta = json.loads((data_dir / "metadata.json").read_text())
    g = meta["generation"]
    if g.get("fixed_scale") is None:
        raise SystemExit(f"{data_dir}: not a --fixed-scale dataset")
    a = argparse.Namespace(
        sigma=meta["sigma"],
        kmax=meta["kmax"],
        band_limit=meta["band_limit"],
        lj_eps=g["lj_eps_eV"],
        lj_sigma=g["lj_sigma_A"],
        lj_rcut=g["lj_rcut_A"],
        lj_ron=g["lj_ron_A"],
    )
    kappa = g["drive_kappa"]
    calc = gy.YukawaLJCalculator(
        kappa, a, meta["fixed_cell_angstrom"], scale=g["fixed_scale"]
    )
    return calc, kappa


def label(gy, calc, at):
    """Energy, forces and the local/nonlocal split, as the dataset generator stores them."""
    at.calc = calc
    e = at.get_potential_energy()
    f = at.get_forces().copy()
    q = at.arrays["ref_charges"]
    pos = at.get_positions()
    a = calc.a
    e_nl, _ = gy.nl_energy_forces(pos, q, calc.L**3, calc.kvec, calc.w, calc.es)
    e_lj, _ = gy.lj_energy_forces(
        pos, calc.L, a.lj_eps, a.lj_sigma, a.lj_rcut, a.lj_ron
    )
    assert abs(e_lj + calc.scale * e_nl - e) < 1e-8
    at.calc = None
    return e, f, e_lj, calc.scale * e_nl


def contacts(gy, frames, n, calc, rng, kappa):
    out = []
    order = rng.permutation(len(frames))
    for idx in order:
        if len(out) == n:
            break
        src = frames[idx]
        t = TYPES[len(out) % 3]
        sym = np.array(src.get_chemical_symbols())
        x = src.positions.copy()
        L = src.cell.lengths()[0]
        d = x[:, None] - x[None]
        d -= L * np.round(d / L)
        r = np.sqrt((d**2).sum(-1))
        np.fill_diagonal(r, np.inf)
        mask = (sym[:, None] == t[0]) & (sym[None, :] == t[1])
        if t[0] == t[1]:
            mask &= np.triu(np.ones_like(mask, bool), 1)
        cand = np.where(mask, r, np.inf)
        rt = rng.uniform(R_LO, R_HI)
        ok = False
        for flat in np.argsort(cand, axis=None)[:20]:  # closest pair of this type first
            i, j = np.unravel_index(flat, cand.shape)
            v = x[j] - x[i]
            v -= L * np.round(v / L)
            r0 = np.linalg.norm(v)
            u = v / r0
            c = x[i] + 0.5 * v
            y = x.copy()
            y[i] = c - 0.5 * rt * u
            y[j] = c + 0.5 * rt * u
            dy = y[[i, j]][:, None] - y[None]
            dy -= L * np.round(dy / L)
            ry = np.sqrt((dy**2).sum(-1))
            ry[0, i] = ry[0, j] = ry[1, i] = ry[1, j] = np.inf
            if ry.min() >= R_OTHER:
                ok = True
                break
        if not ok:
            continue
        at = src.copy()
        at.positions = y
        at.info = {}
        e, f, el, enl = label(gy, calc, at)
        at.info.update(
            energy=e,
            e_local=el,
            e_nonlocal=enl,
            traj=int(src.info["traj"]),
            kappa=kappa,
            config_type="contact",
            contact_pair=f"{t[0]}-{t[1]}",
            contact_r=round(float(rt), 4),
            contact_r0=round(float(r0), 4),
            source_index=int(idx),
            source_energy=float(src.get_potential_energy()),
        )
        at.arrays["forces"] = f
        out.append(at)
    assert len(out) == n, (len(out), n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory holding data_k<tag>/ from the fixed-scale generation",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="where to write data_k<tag>/ outputs (default: --data-root)",
    )
    ap.add_argument(
        "--kappa",
        nargs="+",
        default=list(SEED_OFFSETS),
        choices=list(SEED_OFFSETS),
        help="kappa tags to process",
    )
    args = ap.parse_args()
    out_root = args.out_root or args.data_root
    gy = load_generator()
    summary = {}
    for k in args.kappa:
        data_dir = args.data_root / f"data_k{k}"
        out_dir = out_root / f"data_k{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        calc, kappa = reference_calculator(gy, data_dir)
        split = {
            s: [
                a
                for a in read(data_dir / f"kappa{k}_{s}.xyz", ":")
                if a.info.get("config_type") != "IsolatedAtom"
            ]
            for s in ("train", "test")
        }
        for a in split["train"][:3]:  # the calculator must reproduce the stored labels
            b = a.copy()
            e, f, _, _ = label(gy, calc, b)
            e0, f0 = a.get_potential_energy(), a.get_forces()
            assert abs(e - e0) / len(a) < 1e-9 and np.abs(f - f0).max() < 1e-6, (e, e0)
        rng = np.random.default_rng(SEED + SEED_OFFSETS[k])
        tr = contacts(gy, split["train"], N_TRAIN, calc, rng, kappa)
        te = contacts(gy, split["test"], N_TEST, calc, rng, kappa)
        write(out_dir / f"kappa{k}_contact_train.xyz", tr, format="extxyz")
        write(out_dir / f"kappa{k}_contact_test.xyz", te, format="extxyz")
        orig = read(data_dir / f"kappa{k}_train.xyz", ":")
        write(out_dir / f"kappa{k}_train_contact.xyz", orig + tr, format="extxyz")
        for name, s in (("train", tr), ("test", te)):
            de = np.array([a.info["energy"] - a.info["source_energy"] for a in s])
            fm = np.array([np.abs(a.arrays["forces"]).max() for a in s])
            rr = np.array([a.info["contact_r"] for a in s])
            summary[f"{k}_{name}"] = dict(
                n=len(s),
                r_range=[float(rr.min()), float(rr.max())],
                dE_eV=[float(de.min()), float(np.median(de)), float(de.max())],
                max_abs_force_eV_A=[
                    float(fm.min()),
                    float(np.median(fm)),
                    float(fm.max()),
                ],
                pairs={
                    p: sum(a.info["contact_pair"] == p for a in s)
                    for p in ("Na-Cl", "Na-Na", "Cl-Cl")
                },
            )
            print(f"kappa {k} contact {name}: {summary[f'{k}_{name}']}")
        print(
            f"kappa {k}: wrote kappa{k}_train_contact.xyz with {len(orig)} + {len(tr)} entries"
        )
    (out_root / "data_contact_summary.json").write_text(
        json.dumps(
            dict(
                R_LO=R_LO,
                R_HI=R_HI,
                R_OTHER=R_OTHER,
                N_TRAIN=N_TRAIN,
                N_TEST=N_TEST,
                SEED=SEED,
                summary=summary,
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
