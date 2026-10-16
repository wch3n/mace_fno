"""Generate the heterogeneous-screening dataset (protocol Secs. 5-6, 11a).

Stage md. Each new trajectory starts from an equilibrated 2800 K frame of the
Yukawa MD data (kappa = 0.05, 1146 ions, L = 32 A). Two template frames,
15 ps apart, are taken from each of the 40 template trajectories, and every
new trajectory inherits the train/valid/test split of its template
trajectory, so no positions are shared across splits (44/16/20 trajectories).

S identities are assigned by Metropolis swaps in a random shell-1 field
V(r) = A kT f(r) (unit-variance f, A uniform in [amp-min, amp-max]); the
remaining sites keep their template species, and the few flips needed for
neutrality are made at random. Building the profile by relabelling avoids
waiting for it to form by diffusion. The driver potential is

    E_drive = E_LJ + s_d E_Yukawa(ions; kappa_d) + sum_{m in S} V(R_m),

with species-blind switched LJ for all particles and the uniform Yukawa of
gen_yukawa.py among the ions. The default driver Yukawa (kappa_d = 0.05,
s_d = 1.851) is the Hamiltonian the templates were equilibrated with, so only
the charge order around the relabelled sites has to relax. Langevin MD at
2800 K relaxes for --relax-ps, then --frames frames are taken every
--stride-ps. V_ext is never part of the labels. --traj-range runs a subset of
trajectories (e.g. one task of a job array); each trajectory writes its own
metadata file, which the label stage collects.

Stage label. kappa1 is set on the training frames so that the pooled volume
95th percentile of kappa is --kappa95, every frame is labelled with the
heterogeneous-screening reference, and the energy scale s is set so that the
per-atom standard deviation of the screened energy over the training frames is
--target-rms meV. The Section 6 acceptance statistics are written to
metadata.json.
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
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from ase import Atoms, units  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import (  # noqa: E402
    MaxwellBoltzmannDistribution,
    Stationary,
)
from screening_reference import (  # noqa: E402
    COUL,
    SCREENING_SYMBOL,
    ScreenedMedium,
    reference_charges,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic_yukawa"))
from gen_yukawa import (  # noqa: E402
    kernel,
    kgrid,
    lj_energy_forces,
    nl_energy_forces,
    self_energy_discrete,
)

L, KB = 32.0, 8.617333e-5
SHELL1 = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)], float)
SPLITS = ("train", "valid", "test")


# Sampling driver.
class ShellField:
    """V(r) = A kT sum_n a_n cos(k_n.r + theta_n), unit variance over the cell."""

    def __init__(self, rng, amp_kT, temp):
        a = rng.normal(size=len(SHELL1))
        self.a = a / np.sqrt(np.sum(a**2) / 2.0)
        self.theta = rng.uniform(0, 2 * np.pi, len(SHELL1))
        self.k = 2 * np.pi * SHELL1 / L
        self.scale = amp_kT * KB * temp
        self.amp_kT = amp_kT

    def energy(self, r):
        return self.scale * np.cos(r @ self.k.T + self.theta) @ self.a

    def forces(self, r):
        return self.scale * (np.sin(r @ self.k.T + self.theta) * self.a) @ self.k

    def as_dict(self):
        return {
            "amp_kT": self.amp_kT,
            "a": self.a.tolist(),
            "theta": self.theta.tolist(),
        }


class DriverCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, field, kappa_d, scale_d, charge=0.4, sigma=0.6, kmax=3.5, **kw):
        super().__init__(**kw)
        self.field, self.charge = field, charge
        kvec, kmag, _ = kgrid(L, kmax)
        self.kvec = kvec
        self.w = scale_d * COUL * kernel(kmag, kappa_d) * np.exp(-(sigma**2) * kmag**2)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions() % L
        sym = np.array(self.atoms.get_chemical_symbols())
        q = reference_charges(sym, self.charge)
        ions, is_s = q != 0, sym == SCREENING_SYMBOL
        e_y, f_y = nl_energy_forces(pos[ions], q[ions], L**3, self.kvec, self.w, 0.0)
        e_lj, forces = lj_energy_forces(pos, L, 0.20, 3.0, 4.5, 3.6)
        forces[ions] += f_y
        forces[is_s] += self.field.forces(pos[is_s])
        e = e_lj + e_y + float(np.sum(self.field.energy(pos[is_s])))
        self.results = {"energy": e, "free_energy": e, "forces": forces}


def metropolis_labels(rng, v, n_pick, sweeps=60):
    """Choose n_pick of len(v) sites with P ~ exp(-sum v) (v in units of kT)."""
    n = len(v)
    sel = rng.choice(n, n_pick, replace=False)
    mask = np.zeros(n, bool)
    mask[sel] = True
    sel, rest = np.flatnonzero(mask), np.flatnonzero(~mask)
    for _ in range(sweeps * n):
        i, j = rng.integers(len(sel)), rng.integers(len(rest))
        dv = v[rest[j]] - v[sel[i]]
        if dv <= 0 or rng.random() < np.exp(-dv):
            sel[i], rest[j] = rest[j], sel[i]
    out = np.zeros(n, bool)
    out[sel] = True
    return out


def md_task(job):
    t, split, tmpl_traj, symbols, pos, a = job
    t0 = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([a.seed, t]))
    amp = rng.uniform(a.amp_min, a.amp_max)
    field = ShellField(rng, amp, a.temp)
    n_s = a.n_screening
    sym = np.array(symbols, dtype=object)
    is_s = metropolis_labels(rng, field.energy(pos) / (KB * a.temp), n_s)
    sym[is_s] = SCREENING_SYMBOL
    target = (len(sym) - n_s) // 2
    excess = int(np.sum(sym == "Na")) - target
    if excess:
        src, dst = ("Na", "Cl") if excess > 0 else ("Cl", "Na")
        sym[rng.choice(np.flatnonzero(sym == src), abs(excess), replace=False)] = dst

    atoms = Atoms(symbols=list(sym), positions=pos, cell=[L] * 3, pbc=True)
    atoms.calc = DriverCalculator(field, a.drive_kappa, a.drive_scale)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temp, rng=rng)
    Stationary(atoms)
    dyn = Langevin(
        atoms,
        timestep=a.dt * units.fs,
        temperature_K=a.temp,
        friction=a.gamma / (1000 * units.fs),
        rng=rng,
    )
    relax = int(round(a.relax_ps * 1000 / a.dt))
    stride = int(round(a.stride_ps * 1000 / a.dt))
    total = relax + stride * (a.frames - 1)
    frames = []
    out = Path(a.out) / "md"
    out.mkdir(parents=True, exist_ok=True)
    # The driver log is written as the run proceeds, so progress can be followed
    # with tail; a one-line summary goes to the job log every progress_every steps.
    logf = open(out / f"traj{t:03d}_log.txt", "w", buffering=1)
    logf.write("# time_ps  E_drive_per_atom_eV  T_K\n")
    t_md = time.time()

    def record():
        n = dyn.get_number_of_steps()
        if n % 50 == 0:
            e, temp = atoms.get_potential_energy() / len(atoms), atoms.get_temperature()
            logf.write(f"{n * a.dt / 1000:.3f} {e:.8f} {temp:.2f}\n")
            if n and n % a.progress_every == 0:
                el = time.time() - t_md
                print(
                    f"traj {t:3d}: step {n}/{total} ({100 * n / total:3.0f}%)  "
                    f"{el / 60:6.1f} min elapsed, ETA {el * (total - n) / n / 60:6.1f} min  "
                    f"E/atom {e:.4f} eV  T {temp:6.0f} K",
                    flush=True,
                )
        if n >= relax and (n - relax) % stride == 0 and len(frames) < a.frames:
            fr = Atoms(
                symbols=atoms.get_chemical_symbols(),
                positions=atoms.get_positions() % L,
                cell=[L] * 3,
                pbc=True,
            )
            fr.info.update(
                traj=t,
                split=split,
                template_traj=tmpl_traj,
                time_ps=n * a.dt / 1000,
                amp_kT=amp,
            )
            frames.append(fr)

    dyn.attach(record, interval=1)
    dyn.run(total)
    logf.close()
    record_last = len(frames) == a.frames
    write(out / f"traj{t:03d}.xyz", frames, format="extxyz")
    info = {
        "traj": t,
        "split": split,
        "template_traj": tmpl_traj,
        "field": field.as_dict(),
        "n_frames": len(frames),
        "complete": record_last,
        "seconds": time.time() - t0,
        "args": {
            k: v
            for k, v in vars(a).items()
            if k not in ("stage", "traj_range", "nproc")
        },
    }
    (out / f"traj{t:03d}_meta.json").write_text(json.dumps(info, indent=2))
    print(f"traj {t:3d}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return info


# Reference labels.
def c_task(job):
    pos, sym, ell = job
    med = ScreenedMedium(L, 0.05, 1.0, ell, grid=48, workers=1)
    med.set_medium(pos[sym == SCREENING_SYMBOL])
    return np.clip(med.c_r.ravel(), 0, None)


def label_task(job):
    pos, sym, kappa0, kappa1, ell = job
    med = ScreenedMedium(L, kappa0, kappa1, ell, grid=48, workers=1)
    q = reference_charges(sym)
    is_s, ions = sym == SCREENING_SYMBOL, q != 0
    med.set_medium(pos[is_s])
    e, f_q, f_s = med.energy_forces(pos[ions], q[ions], pos[is_s])
    e_self = self_energy_discrete(kappa0, 0.6, 3.5, L, None)
    e_scr = COUL * (e - e_self * np.sum(q**2))
    f_scr = np.zeros((len(pos), 3))
    f_scr[ions] = COUL * f_q
    f_scr[is_s] = COUL * f_s
    e_lj, f_lj = lj_energy_forces(pos, L, 0.20, 3.0, 4.5, 3.6)
    k_c2 = np.sum(med.n_c**2, axis=1)
    sel = np.isin(k_c2, (1, 2))
    ideal = np.exp(-(ell**2) * np.sum(med.k_c[sel] ** 2, axis=1)) / int(is_s.sum())
    c_ions = np.real(np.exp(1j * (pos[ions] @ med.k_c.T)) @ med.c_k)
    kap = med.kappa_field()
    return {
        "e_scr": e_scr,
        "f_scr": f_scr,
        "e_lj": e_lj,
        "f_lj": f_lj,
        "power": float(np.mean(np.abs(med.c_k[sel]) ** 2 / ideal)),
        "c_std": float(np.std(med.c_r)),
        "c_ions": c_ions,
        "kappa_hist": np.histogram(kap, bins=200, range=(0, 0.8))[0],
        "cg": med.iterations,
    }


def self_energy_share(e_per_atom, c_ions, n_atoms, n_knots=7):
    if len(e_per_atom) <= 2 * n_knots:
        return {
            "note": f"{len(e_per_atom)} frames for {n_knots} parameters: not fitted"
        }
    allc = np.concatenate(c_ions)
    knots = np.quantile(allc, np.linspace(0, 1, n_knots))
    eye = np.eye(n_knots)
    X = np.stack(
        [
            np.stack([np.interp(c, knots, eye[m]) for m in range(n_knots)], 1).sum(0)
            / n_atoms
            for c in c_ions
        ]
    )
    y = np.asarray(e_per_atom)
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
        "n_frames": int(len(y)),
    }


def hist_percentile(hist, edges, p):
    cdf = np.cumsum(hist) / hist.sum()
    i = int(np.searchsorted(cdf, p / 100.0))
    return float(0.5 * (edges[i] + edges[i + 1]))


def collect_md(out, n_traj):
    """Settings and frames of exactly trajectories 0..n_traj-1.

    Every trajectory must have its metadata, identical MD settings, the expected
    number of frames and the supported cell. Only the listed files are read, so
    stale files in the directory cannot enter the dataset.
    """
    md = out / "md"
    expected = list(range(n_traj))
    found = sorted(int(p.name[4:7]) for p in md.glob("traj[0-9][0-9][0-9]_meta.json"))
    if found != expected:
        raise RuntimeError(
            "trajectory metadata mismatch: missing "
            f"{sorted(set(expected) - set(found))}, unexpected "
            f"{sorted(set(found) - set(expected))}"
        )
    metas = [json.loads((md / f"traj{t:03d}_meta.json").read_text()) for t in expected]
    ref = metas[0]["args"]
    for t, m in zip(expected, metas):
        diff = sorted(
            k for k in set(ref) | set(m["args"]) if ref.get(k) != m["args"].get(k)
        )
        if m["traj"] != t or diff:
            raise RuntimeError(
                f"trajectory {t} differs from trajectory 0 in {diff or ['traj']}"
            )
        if not m["complete"] or m["n_frames"] != ref["frames"]:
            raise RuntimeError(f"trajectory {t} is incomplete ({m['n_frames']} frames)")
    frames = []
    for t in expected:
        fr = read(md / f"traj{t:03d}.xyz", index=":")
        if len(fr) != ref["frames"] or any(f.info["traj"] != t for f in fr):
            raise RuntimeError(f"traj{t:03d}.xyz does not match its metadata")
        for f in fr:
            if not (
                np.allclose(f.cell.array, L * np.eye(3), rtol=0, atol=1e-8)
                and all(f.pbc)
            ):
                raise RuntimeError(f"trajectory {t}: unsupported cell or periodicity")
        frames += fr
    stale = sorted(
        p.name
        for p in md.glob("traj*.xyz")
        if not (p.name[4:7].isdigit() and int(p.name[4:7]) in expected)
    )
    if stale:
        print(
            f"warning: ignoring files not in the trajectory list: {stale}", flush=True
        )
    return ref, metas, frames


def driver_drift(out, n_traj, args):
    """Mean change of the driver energy per atom between the halves of the sampling window."""
    t0 = args["relax_ps"]
    t1 = t0 + args["stride_ps"] * (args["frames"] - 1)
    tm = 0.5 * (t0 + t1)
    d = []
    for t in range(n_traj):
        log = np.loadtxt(out / "md" / f"traj{t:03d}_log.txt")
        time, e = log[:, 0], log[:, 1]
        d.append(
            e[(time >= tm) & (time <= t1)].mean() - e[(time >= t0) & (time < tm)].mean()
        )
    d = 1e3 * np.array(d)
    return float(d.mean()), float(np.abs(d).max())


# Section 6 of the protocol (provisional thresholds).
ACCEPTANCE = {
    "contrast": "kappa_p5 <= 0.12 and kappa_p95 >= 0.45 (A^-1)",
    "driven_power": "mean driven power of c_S in shells 1-2 >= 10x independent particles",
    "spread": "per-trajectory std of c_S: min <= 1.5x the independent-particle floor, max >= 1",
    "proxy": "local-concentration fit leaves >= 40 % (RMS) of the screened-energy spread unexplained",
    "nonlocal_spread": "valid and test per-atom spread of the screened energy within 4-6 meV",
    "stationarity": "abs(mean driver-energy drift between sampling halves) <= 1 meV/atom",
}


def acceptance(stats, floor):
    proxy = stats["self_energy_share_train"].get("unexplained_rms_frac")
    spread = stats["e_nonlocal_per_atom_std_meV"]
    checks = {
        "contrast": stats["kappa_p5"] <= 0.12 and stats["kappa_p95"] >= 0.45,
        "driven_power": stats["driven_power_ratio_mean"] >= 10,
        "spread": stats["traj_c_std_min"] <= 1.5 * floor
        and stats["traj_c_std_max"] >= 1.0,
        "proxy": proxy is not None and proxy >= 0.4,
        "nonlocal_spread": all(4.0 <= spread[s] <= 6.0 for s in ("valid", "test")),
        "stationarity": abs(stats["driver_drift_meV_per_atom_mean"]) <= 1.0,
    }
    return {k: {"pass": bool(v), "criterion": ACCEPTANCE[k]} for k, v in checks.items()}


def acceptance_report(out, stats, md_args, written):
    """Human-readable acceptance report, written next to the dataset."""
    sh = stats["self_energy_share_train"]
    sp = stats["e_nonlocal_per_atom_std_meV"]
    values = {
        "contrast": f"kappa_5 = {stats['kappa_p5']:.3f}, kappa_95 = {stats['kappa_p95']:.3f} A^-1",
        "driven_power": f"{stats['driven_power_ratio_mean']:.1f}x",
        "spread": f"{stats['traj_c_std_min']:.2f}-{stats['traj_c_std_max']:.2f} "
        f"(floor {stats['c_std_independent_floor']:.2f})",
        "proxy": (
            f"{100 * sh['unexplained_rms_frac']:.0f} % unexplained (LOO R^2 = {sh['loo_r2']:.2f})"
            if sh.get("unexplained_rms_frac") is not None
            else sh.get("note", "not fitted")
        ),
        "nonlocal_spread": f"train {sp['train']:.2f}, valid {sp['valid']:.2f}, test {sp['test']:.2f} meV/atom",
        "stationarity": f"mean {stats['driver_drift_meV_per_atom_mean']:+.2f}, "
        f"max abs {stats['driver_drift_meV_per_atom_max_abs']:.2f} meV/atom",
    }
    lines = [
        f"# Acceptance report: {out}",
        "",
        f"Result: {'ACCEPTED' if stats['accepted'] else 'REJECTED'}; dataset files "
        f"{'written' if written else 'not written'}.",
        "",
        f"MD: driver kappa_d = {md_args['drive_kappa']}, s_d = {md_args['drive_scale']}; "
        f"A in [{md_args['amp_min']}, {md_args['amp_max']}] kT; relaxation {md_args['relax_ps']} ps; "
        f"{md_args['frames']} frames every {md_args['stride_ps']} ps; {md_args['n_screening']} S.",
        f"Labels: kappa1 = {stats['kappa1']:.4f} A^-1, s = {stats['scale']:.4f}; "
        f"frames {stats['n_frames']}; CG iterations <= {stats['cg_iterations_max']}.",
        "",
        "| check | value | criterion | pass |",
        "|---|---|---|---|",
    ]
    for k, v in stats["acceptance"].items():
        lines.append(
            f"| {k} | {values[k]} | {v['criterion']} | {'yes' if v['pass'] else 'NO'} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "acceptance_report.md").write_text(text)
    return text


def label_stage(a, pool, frames, md_args):
    print(f"labelling {len(frames)} frames", flush=True)
    data = [(fr.positions % L, np.array(fr.get_chemical_symbols())) for fr in frames]
    train = [i for i, fr in enumerate(frames) if fr.info["split"] == "train"]
    c = np.concatenate(pool.map(c_task, [(*data[i], a.ell) for i in train]))
    kappa1 = float(np.sqrt((a.kappa95**2 - a.kappa0**2) / np.percentile(c, 95)))
    res = pool.map(
        label_task, [(*d, a.kappa0, kappa1, a.ell) for d in data], chunksize=4
    )
    n_atoms = len(frames[0])
    e_atom = np.array([r["e_scr"] for r in res]) / n_atoms
    scale = (a.target_rms * 1e-3) / e_atom[train].std()

    parts = {s: [] for s in SPLITS}
    for fr, r in zip(frames, res):
        at = fr.copy()
        at.info.update(
            energy=r["e_lj"] + scale * r["e_scr"],
            e_local=r["e_lj"],
            e_nonlocal=scale * r["e_scr"],
            kappa0=a.kappa0,
            kappa1=kappa1,
            ell=a.ell,
            scale=scale,
        )
        at.arrays["forces"] = r["f_lj"] + scale * r["f_scr"]
        at.arrays["ref_charges"] = reference_charges(at.get_chemical_symbols())
        if not (
            np.isfinite(at.info["energy"]) and np.all(np.isfinite(at.arrays["forces"]))
        ):
            raise RuntimeError(f"non-finite label in trajectory {fr.info['traj']}")
        parts[fr.info["split"]].append(at)
    iso = []
    for s in ("Na", "Cl", SCREENING_SYMBOL):
        at = Atoms(symbols=[s], positions=[[0, 0, 0]], cell=[30.0] * 3, pbc=True)
        at.info.update(
            energy=0.0, e_local=0.0, e_nonlocal=0.0, traj=-1, config_type="IsolatedAtom"
        )
        at.arrays["forces"] = np.zeros((1, 3))
        at.arrays["ref_charges"] = np.zeros(1)
        iso.append(at)
    parts["train"] = iso + parts["train"]

    edges = np.linspace(0, 0.8, 201)
    hist = np.sum([r["kappa_hist"] for r in res], axis=0)
    by_traj = {}
    for fr, r in zip(frames, res):
        by_traj.setdefault(fr.info["traj"], []).append(r["c_std"])
    traj_c_std = np.array([np.mean(v) for v in by_traj.values()])
    stats = {
        "kappa1": kappa1,
        "scale": float(scale),
        "kappa_p5": hist_percentile(hist, edges, 5),
        "kappa_p50": hist_percentile(hist, edges, 50),
        "kappa_p95": hist_percentile(hist, edges, 95),
        "vol_kappa_lt_0p10": float(hist[edges[1:] <= 0.10].sum() / hist.sum()),
        "vol_kappa_gt_0p40": float(hist[edges[:-1] >= 0.40].sum() / hist.sum()),
        "driven_power_ratio_mean": float(np.mean([r["power"] for r in res])),
        "traj_c_std_min": float(traj_c_std.min()),
        "traj_c_std_max": float(traj_c_std.max()),
        "e_nonlocal_per_atom_std_meV": {
            s: float(
                1e3
                * scale
                * e_atom[
                    [i for i, fr in enumerate(frames) if fr.info["split"] == s]
                ].std()
            )
            for s in SPLITS
        },
        "self_energy_share_train": self_energy_share(
            e_atom[train], [res[i]["c_ions"] for i in train], n_atoms
        ),
        "cg_iterations_max": int(max(r["cg"] for r in res)),
        "n_frames": {s: len(parts[s]) for s in SPLITS},
    }
    drift_mean, drift_max = driver_drift(Path(a.out), a.n_traj, md_args)
    stats["driver_drift_meV_per_atom_mean"] = drift_mean
    stats["driver_drift_meV_per_atom_max_abs"] = drift_max
    n_s = int(md_args["n_screening"])
    floor = 1.0 / np.sqrt(n_s / L**3 * (4 * np.pi * a.ell**2) ** 1.5)
    stats["c_std_independent_floor"] = float(floor)
    stats["acceptance"] = acceptance(stats, floor)
    stats["accepted"] = all(v["pass"] for v in stats["acceptance"].values())
    if stats["accepted"] or a.accept_failed:
        for s, frs in parts.items():
            write(Path(a.out) / f"hetscreen_{s}.xyz", frs, format="extxyz")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=("md", "label", "all"), default="all")
    ap.add_argument(
        "--templates", help="directory with kappa0p05_{train,valid,test}.xyz"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-traj", type=int, default=80)
    ap.add_argument(
        "--template-frames",
        type=int,
        nargs=2,
        default=(0, 15),
        help="frame indices taken from each template trajectory",
    )
    ap.add_argument("--n-screening", type=int, default=286)
    ap.add_argument("--amp-min", type=float, default=0.0)
    ap.add_argument("--amp-max", type=float, default=4.0)
    ap.add_argument("--temp", type=float, default=2800.0)
    ap.add_argument("--dt", type=float, default=2.0, help="fs")
    ap.add_argument("--gamma", type=float, default=2.0, help="1/ps")
    ap.add_argument("--relax-ps", type=float, default=6.0)
    ap.add_argument("--stride-ps", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--drive-kappa", type=float, default=0.05)
    ap.add_argument("--drive-scale", type=float, default=1.851)
    ap.add_argument(
        "--traj-range",
        default=None,
        metavar="START:END",
        help="md stage: run only trajectories START <= t < END",
    )
    ap.add_argument("--kappa0", type=float, default=0.05)
    ap.add_argument("--kappa95", type=float, default=0.5)
    ap.add_argument("--ell", type=float, default=3.0)
    ap.add_argument("--target-rms", type=float, default=5.0, help="meV/atom")
    ap.add_argument(
        "--accept-failed",
        action="store_true",
        help="write the dataset even if an acceptance check fails",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="steps between progress lines in the job log (multiple of 50)",
    )
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with mp.Pool(a.nproc) as pool:
        if a.stage in ("md", "all"):
            jobs, t = [], 0
            for split in SPLITS:
                tmpl = [
                    f
                    for f in read(
                        Path(a.templates) / f"kappa0p05_{split}.xyz", index=":"
                    )
                    if f.info.get("config_type") != "IsolatedAtom"
                ]
                by_traj = {}
                for f in tmpl:
                    by_traj.setdefault(f.info["traj"], []).append(f)
                for tt in sorted(by_traj):
                    for k in a.template_frames:
                        if t < a.n_traj:
                            f = by_traj[tt][k]
                            jobs.append(
                                (
                                    t,
                                    split,
                                    int(tt),
                                    f.get_chemical_symbols(),
                                    f.positions % L,
                                    a,
                                )
                            )
                            t += 1
            if a.traj_range:
                lo, hi = (int(x) for x in a.traj_range.split(":"))
                jobs = [j for j in jobs if lo <= j[0] < hi]
            print(
                f"running {len(jobs)} trajectories on {a.nproc} processes", flush=True
            )
            md = pool.map(md_task, jobs, chunksize=1)
            bad = [m["traj"] for m in md if not m["complete"]]
            if bad:
                raise RuntimeError(f"incomplete trajectories: {bad}")
        if a.stage in ("label", "all"):
            md_args, md_meta, frames = collect_md(out, a.n_traj)
            meta["md"] = {
                "args": md_args,
                "trajectories": [
                    {k: v for k, v in m.items() if k != "args"} for m in md_meta
                ],
            }
            stats = label_stage(a, pool, frames, md_args)
            meta["labels"] = {
                "kappa0": a.kappa0,
                "ell": a.ell,
                "kappa95_target": a.kappa95,
                "target_rms_meV": a.target_rms,
                **stats,
            }
            meta_path.write_text(json.dumps(meta, indent=2, default=float))
            print(json.dumps(stats, indent=2, default=float), flush=True)
            print(
                acceptance_report(
                    out, stats, md_args, stats["accepted"] or a.accept_failed
                ),
                flush=True,
            )
            if not stats["accepted"]:
                failed = [k for k, v in stats["acceptance"].items() if not v["pass"]]
                if a.accept_failed:
                    print(
                        f"acceptance failed for {failed}; dataset written (--accept-failed)",
                        flush=True,
                    )
                else:
                    sys.exit(
                        f"acceptance failed for {failed}: dataset files not written "
                        "(statistics in metadata.json; rerun with --accept-failed to override)"
                    )


if __name__ == "__main__":
    main()
