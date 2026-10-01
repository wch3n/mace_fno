"""Synthetic 3D-periodic benchmark with a Yukawa kernel of known asymptote.

The nonlocal energy is defined directly in reciprocal space,

    E_nl = 1/(2V) sum_{k!=0} V(k) g(k) |rho(k)|^2  -  N e_self ,
    V(k) = 4 pi / (k^2 + kappa^2),   g(k) = exp(-sigma^2 k^2),
    rho(k) = sum_i q_i exp(-i k.r_i),

so the propagation kernel is ground truth. For k >> kappa the kernel is
4 pi / k^2 and the spectral exponent p = -dlnV/dlnk approaches 2, while the
finite kappa removes the k -> 0 divergence. Sweeping kappa moves the crossover
through the sampled band: kappa well below k_min gives a Coulomb-like target,
kappa inside the band gives a screened one.

Because E_nl is quadratic in the sources, the source-space Hessian used by the
spectral diagnostic is exactly V(k): p has a known analytic value at every k.

All kappa variants are evaluated on the SAME trajectories, so comparing them
changes only the labels, never the geometry distribution.  Pass --num-atoms
to keep the graph size fixed while varying the periodic cell size; otherwise
the atom count is inferred from --density as before.
"""

from __future__ import annotations

import argparse
import json
import os

# Each worker process is single-threaded: the dominant cost is np.exp on an
# (N, K) complex array, which is elementwise and not threaded, so BLAS threads
# would only oversubscribe the cores. Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
from ase import Atoms, units  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.io import write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import (  # noqa: E402
    MaxwellBoltzmannDistribution,
    Stationary,
)

_KGRID_CACHE: dict = {}

COUL = 14.399645  # e^2/(4 pi eps0) in eV A


def kernel(k, kappa):
    """Yukawa multiplier; kappa -> 0 recovers bare Coulomb."""
    return 4.0 * np.pi / (k**2 + kappa**2)


def exponent(k, kappa):
    """Local logarithmic slope p = -dlnV/dlnk of the Yukawa kernel."""
    return 2.0 * k**2 / (k**2 + kappa**2)


def kgrid(L, kmax, nmax_band=None):
    """Half-space integer modes with 0 < |k| <= kmax for a cubic cell.

    Only one of each +-k pair is kept; the caller doubles the sums, which is
    exact because |rho(k)|^2 = |rho(-k)|^2 for real sources.
    """
    m = int(np.ceil(kmax * L / (2 * np.pi)))
    n = np.arange(-m, m + 1)
    N1, N2, N3 = np.meshgrid(n, n, n, indexing="ij")
    n3 = np.stack([N1.ravel(), N2.ravel(), N3.ravel()], axis=1)
    kvec = 2 * np.pi * n3 / L
    kmag = np.linalg.norm(kvec, axis=1)

    keep = (kmag > 1e-12) & (kmag <= kmax)
    # half space: first nonzero component positive
    lead = np.zeros(len(n3), dtype=int)
    for col in (0, 1, 2):
        unset = lead == 0
        lead[unset] = n3[unset, col]
    keep &= lead > 0
    if nmax_band is not None:
        keep &= np.all(np.abs(n3) <= nmax_band, axis=1)
    return kvec[keep], kmag[keep], n3[keep]


def kgrid_cached(L, kmax, nmax_band=None):
    """kgrid memoised per process; with a fixed cell it is built once."""
    key = (float(L), float(kmax), nmax_band)
    if key not in _KGRID_CACHE:
        _KGRID_CACHE[key] = kgrid(L, kmax, nmax_band)
    return _KGRID_CACHE[key]


def _self_energy(kappa, a, L):
    """Continuum self term, or the band-matched discrete one under --band-limit."""
    if a.band_limit is None:
        return COUL * self_energy_unit(kappa, a.sigma, a.kmax)
    return COUL * self_energy_discrete(kappa, a.sigma, a.kmax, L, a.band_limit)


def _drive_scale(a):
    """Scale on E_nl in the MD driving potential: 1 unless --fixed-scale."""
    return 1.0 if getattr(a, "fixed_scale", None) is None else a.fixed_scale


def _traj_worker(job):
    """Run one independent trajectory. Module level so it pickles."""
    t, L, n, seed_seq, a = job
    rng = np.random.default_rng(seed_seq)
    q = a.charge * np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
    rng.shuffle(q)
    kdrive = min(a.kappa)
    kvec, kmag, _ = kgrid_cached(L, a.kmax, a.band_limit)
    pos = init_positions(rng, n, L)
    if a.md_engine == "ase":
        traj = run_md_ase(rng, pos, q, L, a, a.nsteps, a.nsample)
    else:
        w = (_drive_scale(a) * COUL * kernel(kmag, kdrive)
             * np.exp(-(a.sigma**2) * kmag**2))
        es = _self_energy(kdrive, a, L)
        traj = run_md(rng, pos, q, L, kvec, w, es, a, a.nsteps, a.nsample)
    return t, L, n, q, traj, len(kmag)


def _label_worker(job):
    """Label one frame with one kernel. Module level so it pickles."""
    kap, t, L, pos, q, a = job
    kvec, kmag, _ = kgrid_cached(L, a.kmax, a.band_limit)
    w = COUL * kernel(kmag, kap) * np.exp(-(a.sigma**2) * kmag**2)
    e_nl, f_nl = nl_energy_forces(pos, q, L**3, kvec, w, _self_energy(kap, a, L))
    e_lj, f_lj = lj_energy_forces(pos, L, a.lj_eps, a.lj_sigma,
                                  a.lj_rcut, a.lj_ron)
    return t, L, pos, q, e_nl, f_nl, e_lj, f_lj


def _run(func, jobs, nproc, chunksize=1):
    """Serial map for nproc == 1, otherwise a process pool."""
    if nproc == 1:
        return [func(j) for j in jobs]
    with mp.Pool(nproc) as pool:
        return pool.map(func, jobs, chunksize=chunksize)


def self_energy_discrete(kappa, sigma, kmax, L, nmax_band):
    """Self term matched to a band-limited pair sum, so that E0 stays zero.

    With --band-limit the pair sum runs over a cell-dependent mode set, so the
    continuum self-energy no longer matches its k coverage and the isolated-atom
    energy drifts (-386 meV at L = 20 A, -842 meV at L = 80 A for q = 0.4).
    Using the same discrete mode set restores E0 = 0 in every cell, at the cost
    of a cell-dependent self term: size-extensivity under replication is then
    lost, because a band fixed in integer index covers a different physical
    region once the cell is doubled. Band-limited runs are therefore only
    self-consistent at a single cell size.
    """
    _, kmag, _ = kgrid(L, kmax, nmax_band)
    w = kernel(kmag, kappa) * np.exp(-(sigma**2) * kmag**2)
    return np.sum(w) / L**3          # half-space sum, doubled, over 2V


def self_energy_unit(kappa, sigma, kmax, npts=400000):
    """Cell-independent self-interaction of one smeared unit charge,

        e_self = 1/2 (2 pi)^-3 int d^3k V(k) g(k)
               = 1/2 (2 pi^2)^-1 int_0^kmax k^2 V(k) g(k) dk .

    Using the continuum integral rather than a discrete k-sum is what makes the
    labels exactly size-extensive: the pair term is extensive by construction,
    so the self term must not depend on the cell.
    """
    k = np.linspace(1e-8, kmax, npts)
    integ = k**2 * kernel(k, kappa) * np.exp(-(sigma**2) * k**2)
    return 0.5 * np.trapezoid(integ, k) / (2.0 * np.pi**2)


def nl_energy_forces(pos, q, vol, kvec, w, e_self_unit):
    """E_nl and forces from a half-space mode list (sums doubled)."""
    phase = np.exp(-1j * (pos @ kvec.T))          # (N,K)
    rho = q @ phase                                # (K,)
    e_pair = (1.0 / vol) * np.sum(w * np.abs(rho) ** 2)   # 2 x half space
    energy = e_pair - e_self_unit * np.sum(q**2)
    imag = np.imag(np.conj(rho)[None, :] * phase)
    grad = (2.0 * q[:, None] / vol) * ((w[None, :] * imag) @ kvec)
    return energy, -grad


def lj_energy_forces(pos, L, eps, sig, rcut, ron=3.6):
    """Lennard-Jones with a C^2 switching function over [ron, rcut].

    An energy shift alone leaves a finite force at the cutoff. Here the
    potential is multiplied by S(y) = 1 - 6y^5 + 15y^4 - 10y^3, y =
    (r-ron)/(rcut-ron), so energy, force, and force slope all vanish
    continuously at rcut. This matters because MACE's radial envelope also
    vanishes with zero derivative at its cutoff: a force step at rcut would
    lie outside MACE's function class, leaving a short-ranged remainder that
    the FNO could absorb and that would masquerade as a nonlocal gain.
    """
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(len(pos), k=1)
    rij, dij = r[iu], d[iu]
    m = rij < rcut
    rij, dij = rij[m], dij[m]
    if rij.size and rij.min() < 1.0:
        raise RuntimeError(f"atoms within {rij.min():.3f} A: core too soft "
                           "for this charge, or timestep too large")
    sr6 = (sig / rij) ** 6
    e_lj = 4 * eps * (sr6**2 - sr6)            # plain LJ energy
    f_lj = 24 * eps * (2 * sr6**2 - sr6) / rij  # plain LJ force magnitude

    y = np.clip((rij - ron) / (rcut - ron), 0.0, 1.0)
    S = 1 - 6 * y**5 + 15 * y**4 - 10 * y**3
    dS = -30 * y**2 * (y - 1) ** 2 / (rcut - ron)

    energy = np.sum(e_lj * S)
    fscal = f_lj * S - e_lj * dS               # = -d(e_lj * S)/dr
    fvec = (fscal / rij)[:, None] * dij
    forces = np.zeros_like(pos)
    np.add.at(forces, iu[0][m], fvec)
    np.add.at(forces, iu[1][m], -fvec)
    return energy, forces


class YukawaLJCalculator(Calculator):
    """ASE calculator for E = LJ(switched) + E_nl[Yukawa].

    Wrapping the analytic functions in a Calculator lets the trajectories come
    from ASE's tested integrators rather than a hand-rolled one, and makes the
    reference potential reusable -- in particular for the NVE drift test in
    verify_yukawa.py, which is the standard check that the potential is
    conservative and which catches any discontinuity at the LJ cutoff.
    """

    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, kappa, a, L, scale=1.0, **kw):
        super().__init__(**kw)
        self.L = float(L)
        self.a = a
        # The dataset labels are E_LJ + s*E_nl with s fitted per kappa, so MD
        # that must reproduce a trained model's Hamiltonian needs the same s.
        # The default 1.0 keeps the unscaled driving potential used to
        # generate the published configurations.
        self.scale = float(scale)
        kvec, kmag, _ = kgrid_cached(L, a.kmax, a.band_limit)
        self.kvec = kvec
        self.w = COUL * kernel(kmag, kappa) * np.exp(-(a.sigma**2) * kmag**2)
        self.es = _self_energy(kappa, a, L)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        L, a = self.L, self.a
        if not np.allclose(self.atoms.cell.lengths(), L):
            raise ValueError("calculator was built for a different cell")
        q = self.atoms.arrays["ref_charges"]
        pos = self.atoms.get_positions()
        e_nl, f_nl = nl_energy_forces(pos, q, L**3, self.kvec, self.w, self.es)
        e_lj, f_lj = lj_energy_forces(pos, L, a.lj_eps, a.lj_sigma,
                                      a.lj_rcut, a.lj_ron)
        e = e_lj + self.scale * e_nl
        self.results = {"energy": e, "free_energy": e,
                        "forces": f_lj + self.scale * f_nl}


def init_positions(rng, n, L, dmin=2.3, tries=40000):
    pos = np.empty((n, 3))
    c = 0
    for _ in range(tries):
        if c == n:
            break
        p = rng.uniform(0, L, size=3)
        if c:
            d = pos[:c] - p
            d -= L * np.round(d / L)
            if np.min(np.linalg.norm(d, axis=1)) < dmin:
                continue
        pos[c] = p
        c += 1
    if c < n:
        raise RuntimeError(f"placed only {c}/{n}; lower --density")
    return pos


def run_md_ase(rng, pos, q, L, a, nsteps, nsample):
    """Langevin dynamics via ASE, sampling nsample frames after a burn-in.

    Uses real Na/Cl masses from ASE rather than a single averaged mass. The
    sampled configurational distribution is unaffected by the mass choice; only
    the dynamics differ.
    """
    syms = ["Na" if qi > 0 else "Cl" for qi in q]
    atoms = Atoms(symbols=syms, positions=pos, cell=[L, L, L], pbc=True)
    atoms.arrays["ref_charges"] = q
    atoms.calc = YukawaLJCalculator(min(a.kappa), a, L, scale=_drive_scale(a))

    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temp,
                                 rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, timestep=a.dt * 1000 * units.fs,
                   temperature_K=a.temp,
                   friction=a.gamma / (1000 * units.fs), rng=rng)

    nb = getattr(a, "nburn", None)          # None -> legacy nsteps//4
    burn = nsteps // 4 if nb is None else nb
    stride = max(1, (nsteps - burn) // nsample)
    out = []

    def sample():
        step = dyn.get_number_of_steps()
        if step >= burn and (step - burn) % stride == 0 and len(out) < nsample:
            out.append(atoms.get_positions() % L)

    dyn.attach(sample, interval=1)
    dyn.run(nsteps)
    return out[:nsample]


def run_md(rng, pos, q, L, kvec, w, e_self, a, nsteps, nsample):
    vol = L**3
    mass = 23.0 * 1.0364e-4                       # amu -> eV ps^2/A^2
    kT = 8.617333e-5 * a.temp
    vel = rng.normal(0, np.sqrt(kT / mass), size=pos.shape)

    def force(p):
        _, f1 = lj_energy_forces(p, L, a.lj_eps, a.lj_sigma, a.lj_rcut, a.lj_ron)
        _, f2 = nl_energy_forces(p, q, vol, kvec, w, e_self)
        return f1 + f2

    f = force(pos)
    c1 = np.exp(-a.gamma * a.dt)
    c2 = np.sqrt((1 - c1**2) * kT / mass)
    nb = getattr(a, "nburn", None)          # None -> legacy nsteps//4
    burn = nsteps // 4 if nb is None else nb
    out, stride = [], max(1, (nsteps - burn) // nsample)
    for step in range(nsteps):
        vel += 0.5 * a.dt * f / mass
        pos = pos + 0.5 * a.dt * vel
        vel = c1 * vel + c2 * rng.normal(size=vel.shape)
        pos = (pos + 0.5 * a.dt * vel) % L
        f = force(pos)
        vel += 0.5 * a.dt * f / mass
        if step >= burn and (step - burn) % stride == 0:
            out.append(pos.copy())
    return out[:nsample]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_yukawa")
    ap.add_argument("--kappa", nargs="+", type=float,
                    default=[0.05, 0.5, 1.0, 2.0],
                    help="screening wavevectors (1/A); the smallest gives the "
                         "Coulomb-like 1/k^2 target")
    ap.add_argument("--band-limit", type=int, default=None, metavar="NMAX",
                    help="zero the kernel outside |n_a| <= NMAX, making the "
                         "target exactly representable by an FNO with that "
                         "mode band. WARNING: the band is fixed in integer "
                         "index, so its physical extent scales as 1/L. E0 = 0 "
                         "is preserved by a matched discrete self term, but "
                         "size-extensivity is NOT: use a single cell size.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--nproc", type=int, default=1, metavar="P",
                    help="worker processes for the trajectory and labelling "
                         "loops; 0 uses every detected core. Each worker is "
                         "pinned to one thread. Results are independent of P: "
                         "every trajectory gets its own spawned RNG stream.")
    ap.add_argument("--cells", nargs="+", type=float,
                    default=[14.0, 17.0, 20.0, 23.0],
                    help="cubic cell lengths (A), cycled over trajectories; "
                         "pass one value to use a fixed cell throughout")
    ap.add_argument("--density", type=float, default=0.035, help="atoms/A^3")
    ap.add_argument("--num-atoms", type=int, default=None, metavar="N",
                    help="use exactly N atoms in every configuration, "
                         "independent of cell size; N must be positive and "
                         "even so the two species remain charge neutral")
    ap.add_argument("--charge", type=float, default=0.4)
    ap.add_argument("--sigma", type=float, default=0.6, help="smearing (A)")
    ap.add_argument("--kmax", type=float, default=3.5)
    ap.add_argument("--ntraj", type=int, default=8)
    ap.add_argument("--nvalid", type=int, default=None, metavar="T",
                    help="trajectories held out for validation "
                         "(default: ntraj // 5). Splitting by whole trajectory "
                         "rather than by frame is deliberate: frames within a "
                         "run are strongly correlated.")
    ap.add_argument("--ntest", type=int, default=None, metavar="T",
                    help="trajectories held out for test (default: ntraj // 4)")
    ap.add_argument("--nsample", type=int, default=30)
    ap.add_argument("--nsteps", type=int, default=1500)
    ap.add_argument("--nburn", type=int, default=None, metavar="B",
                    help="equilibration steps discarded before sampling "
                         "(default: nsteps // 4). Sampling spans "
                         "[nburn, nsteps), so nsteps is still the TOTAL run "
                         "length. Raise this when the potential energy is "
                         "still drifting across the sampled window.")
    ap.add_argument("--no-isolated-atoms", action="store_true",
                    help="omit the config_type=IsolatedAtom reference frames "
                         "that anchor E0 = 0 for both species")
    ap.add_argument("--target-rms", type=float, default=5.0,
                    help="target std of E_nl per atom (meV)")
    ap.add_argument("--md-engine", choices=("ase", "internal"), default="ase",
                    help="'ase' uses ase.md.langevin.Langevin through a "
                         "YukawaLJCalculator (default); 'internal' uses the "
                         "built-in integrator. The two sample the same "
                         "ensemble but give different trajectories.")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--temp", type=float, default=400.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--lj-eps", type=float, default=0.20)
    ap.add_argument("--lj-sigma", type=float, default=3.0)
    ap.add_argument("--lj-rcut", type=float, default=4.5)
    ap.add_argument("--lj-ron", type=float, default=3.6,
                    help="switching starts here; keep above the LJ minimum "
                         "at 2^(1/6) sigma")
    ap.add_argument("--fixed-scale", type=float, default=None, metavar="S",
                    help="drive the MD with, and label using, E_LJ + S*E_nl at "
                         "the single --kappa given, instead of driving at scale 1 "
                         "with min(--kappa) and fitting the scale afterwards. The "
                         "configurations are then an equilibrium sample of the "
                         "labelled potential, which MD with a trained model needs. "
                         "Requires exactly one --kappa.")
    a = ap.parse_args()

    if a.fixed_scale is not None:
        if len(a.kappa) != 1:
            ap.error("--fixed-scale drives and labels one potential; pass exactly "
                     "one --kappa")
        if not a.fixed_scale > 0:
            ap.error("--fixed-scale must be positive")

    if a.num_atoms is not None and (a.num_atoms < 2 or a.num_atoms % 2):
        ap.error("--num-atoms must be a positive even integer")
    if a.nburn is not None:
        if not 0 <= a.nburn < a.nsteps:
            ap.error(f"--nburn {a.nburn} must satisfy 0 <= nburn < --nsteps "
                     f"{a.nsteps}")
        if a.nsteps - a.nburn < a.nsample:
            ap.error(f"--nburn {a.nburn} leaves {a.nsteps - a.nburn} sampling "
                     f"steps for --nsample {a.nsample} frames")
    if any(L <= 0 for L in a.cells):
        ap.error("all --cells values must be positive")

    # Three contiguous blocks of trajectories: train | valid | test.
    n_test = a.ntest if a.ntest is not None else max(1, a.ntraj // 4)
    n_valid = a.nvalid if a.nvalid is not None else max(1, a.ntraj // 5)
    if n_valid < 0 or n_test < 0:
        ap.error("--nvalid and --ntest must be non-negative")
    n_train = a.ntraj - n_valid - n_test
    if n_train < 1:
        ap.error(f"--nvalid {n_valid} + --ntest {n_test} leaves {n_train} "
                 f"training trajectories out of --ntraj {a.ntraj}")
    split = {"train": set(range(n_train)),
             "valid": set(range(n_train, n_train + n_valid)),
             "test": set(range(n_train + n_valid, a.ntraj))}
    for name in ("valid", "test"):
        if 0 < len(split[name]) < 4:
            print(f"note: only {len(split[name])} {name} trajectories; the "
                  f"RMSE on that set will carry ~{100/np.sqrt(10*len(split[name])):.0f}% "
                  "statistical noise", flush=True)
    # A contiguous block covers every cell size only if it is at least as long
    # as the cell cycle, since cells are assigned as cells[t % len(cells)].
    for name in ("valid", "test"):
        got = {a.cells[t % len(a.cells)] for t in split[name]}
        missing = sorted(set(a.cells) - got)
        if missing:
            print(f"note: {name} split misses cell size(s) {missing} A; raise "
                  f"--n{name} to at least {len(a.cells)} to cover them all",
                  flush=True)

    os.makedirs(a.out, exist_ok=True)
    nproc = a.nproc if a.nproc > 0 else (os.cpu_count() or 1)

    # One independent, reproducible RNG stream per trajectory, so the data does
    # not depend on how the pool happens to schedule the work.
    children = np.random.SeedSequence(a.seed).spawn(a.ntraj)

    jobs = []
    for t in range(a.ntraj):
        L = a.cells[t % len(a.cells)]
        if a.num_atoms is None:
            n = int(round(a.density * L**3))
            n -= n % 2
        else:
            n = a.num_atoms
        if n < 2:
            raise ValueError(f"cell L={L:g} A gives N={n}; increase --density "
                             "or set --num-atoms to a positive even value")
        jobs.append((t, L, n, children[t], a))

    print(f"running {a.ntraj} trajectories on {nproc} process(es)", flush=True)
    results = sorted(_run(_traj_worker, jobs, nproc), key=lambda r: r[0])

    frames, cell_specs = [], []
    for t, L, n, q, traj, nmodes in results:
        for p in traj:
            frames.append((t, L, p, q))
        cell_specs.append({"trajectory": t, "cell_angstrom": L,
                           "num_atoms": n,
                           "density_atoms_per_angstrom3": n / L**3})
        print(f"traj {t:3d}  L = {L:5.1f} A  N = {n:4d}  modes = {nmodes:6d}"
              f"  frames = {len(traj)}", flush=True)

    meta = {"generation": {
                "temp_K": a.temp, "dt_ps": a.dt, "gamma_per_ps": a.gamma,
                "nsteps": a.nsteps,
                "nburn_effective": (a.nsteps // 4 if a.nburn is None else a.nburn),
                "nsample": a.nsample, "md_engine": a.md_engine,
                "lj_eps_eV": a.lj_eps, "lj_sigma_A": a.lj_sigma,
                "lj_rcut_A": a.lj_rcut, "lj_ron_A": a.lj_ron,
                "drive_kappa": min(a.kappa), "drive_scale": _drive_scale(a),
                "fixed_scale": a.fixed_scale, "seed": a.seed},
            "n_frames": len(frames), "sigma": a.sigma, "kmax": a.kmax,
            "band_limit": a.band_limit, "nburn": a.nburn, "charge": a.charge,
            "cells": a.cells,
            "fixed_cell_angstrom": (a.cells[0] if len(a.cells) == 1 else None),
            "fixed_num_atoms": a.num_atoms,
            "nominal_density_atoms_per_angstrom3": (
                a.density if a.num_atoms is None else None),
            "cell_specs": cell_specs,
            "split_by": "trajectory",
            "split_trajectories": {k: sorted(v) for k, v in split.items()},
            "kernels": {}}

    for kap in a.kappa:
        label_jobs = [(kap, t, L, pos, q, a) for (t, L, pos, q) in frames]
        recs = _run(_label_worker, label_jobs, nproc,
                    chunksize=max(1, len(label_jobs) // (4 * nproc)))

        per_atom = np.array([r[4] / len(r[3]) for r in recs])
        scale = (a.fixed_scale if a.fixed_scale is not None
                 else (a.target_rms * 1e-3) / per_atom.std())

        images = []
        for (t, L, pos, q, e_nl, f_nl, e_lj, f_lj) in recs:
            syms = ["Na" if qi > 0 else "Cl" for qi in q]
            at = Atoms(symbols=syms, positions=pos, cell=[L, L, L], pbc=True)
            at.info.update(energy=e_lj + scale * e_nl, e_local=e_lj,
                           e_nonlocal=scale * e_nl, traj=t, kappa=kap)
            at.arrays["forces"] = f_lj + scale * f_nl
            at.arrays["ref_charges"] = q
            images.append(at)
        for at in images:
            if not (np.isfinite(at.info["energy"]).all()
                    and np.all(np.isfinite(at.arrays["forces"]))):
                raise RuntimeError(f"non-finite label at kappa={kap}")

        # Split by whole trajectory. Frames within one trajectory are strongly
        # correlated (lag-1 autocorrelation of E_nl/atom ~0.8, decorrelation
        # ~440 fs), so a random frame split would put near-duplicates on both
        # sides of every boundary: validation would flatter itself and
        # checkpoint selection would happen on data the model has seen.
        parts = {"train": [x for x in images if x.info["traj"] in split["train"]],
                 "valid": [x for x in images if x.info["traj"] in split["valid"]],
                 "test": [x for x in images if x.info["traj"] in split["test"]]}

        if not a.no_isolated_atoms:
            # E0 = 0 exactly for both species: an isolated atom has no LJ
            # neighbour, and the self-energy convention makes its nonlocal
            # energy vanish. Supplied explicitly because every structure is
            # 50/50, which makes a least-squares E0 fit degenerate (the two
            # design-matrix columns are identical). Train only -- MACE reads
            # E0s from the training file.
            iso = []
            for sym in ("Na", "Cl"):
                at = Atoms(symbols=[sym], positions=[[0.0, 0.0, 0.0]],
                           cell=[30.0, 30.0, 30.0], pbc=True)
                at.info.update(energy=0.0, e_local=0.0, e_nonlocal=0.0,
                               traj=-1, kappa=kap, config_type="IsolatedAtom")
                at.arrays["forces"] = np.zeros((1, 3))
                at.arrays["ref_charges"] = np.zeros(1)
                iso.append(at)
            parts["train"] = iso + parts["train"]

        tag = f"kappa{kap:g}".replace(".", "p")
        for name, frs in parts.items():
            write(f"{a.out}/{tag}_{name}.xyz", frs, format="extxyz")

        kmin = 2 * np.pi / max(a.cells)
        kbig = 2 * np.pi * np.sqrt(3) * 3 / min(a.cells)
        # `images` never holds the isolated-atom frames, but filter anyway:
        # two E_nl = 0 outliers against a mean of ~-100 meV/atom would wreck
        # this statistic, and the coupling is easy to reintroduce by accident.
        enl = np.array([x.info["e_nonlocal"] / len(x) for x in images
                        if x.info.get("config_type") != "IsolatedAtom"]) * 1e3
        meta["kernels"][tag] = {
            "kappa": kap, "scale": float(scale),
            "p_at_kmin": float(exponent(kmin, kap)),
            "p_at_band_edge": float(exponent(kbig, kap)),
            "e_nonlocal_per_atom_meV_std": float(enl.std()),
            "n_train": len(parts["train"]), "n_valid": len(parts["valid"]),
            "n_test": len(parts["test"]),
            "n_isolated_atom_refs": 0 if a.no_isolated_atoms else 2}
        print(f"kappa={kap:<5g} scale={scale:9.4g}  E_nl/atom sd="
              f"{enl.std():6.3f} meV  p(k_min)={exponent(kmin, kap):.3f} "
              f" p(band edge)={exponent(kbig, kap):.3f}"
              f"  train/valid/test={len(parts['train'])}/"
              f"{len(parts['valid'])}/{len(parts['test'])}", flush=True)

    with open(f"{a.out}/metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {a.out}/metadata.json")


if __name__ == "__main__":
    main()
