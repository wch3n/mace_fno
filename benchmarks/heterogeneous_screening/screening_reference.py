"""Reference energy of the heterogeneous-screening benchmark.

Neutral screening-medium particles S (symbol Ar) set a position-dependent
screening constant through their Gaussian-smoothed concentration,

    kappa^2(r) = kappa0^2 + kappa1^2 c_S(r),
    c_S(r) = (1/nbar_S) sum_{m in S} w_l(r - R_m),      <c_S> = 1,

while the ions (Na +q, Cl -q) carry Gaussian charges of width sigma. The
potential solves

    A phi = 4 pi rho_sigma,      A = -lap + kappa^2(r),

and the screened energy is E_scr = 1/2 <rho_sigma, phi>, self terms included.
For c_S = 1 this is the Yukawa energy of gen_yukawa.py at
kappa^2 = kappa0^2 + kappa1^2, before that script subtracts its self term.

The reference is a fixed Galerkin problem. phi is expanded in the plane waves
|k| <= kmax, including k = 0, which is nonzero once c_S varies; c_S is
expanded in |k| <= cmax, beyond which its Gaussian factor is below 1e-14 and
the truncation is part of the definition. Products are formed on a grid and
projected back onto the phi modes at every operator application. With G
points per axis the projections are alias-free when G > 2 n_phi + n_c, the
largest integer mode components, so energies and forces do not depend on G
beyond solver precision.

Fourier convention: f(r) = sum_k f_k exp(i k.r), f_k = (1/V) int f exp(-i k.r),
so int f g dr = V sum_k conj(f_k) g_k for real f and g.

Forces follow from dE = int phi d(rho) - (1/8 pi) int phi^2 d(kappa^2). Ions
feel the smoothed field; S particles feel the gradient of phi^2 smoothed by
w_l, which pulls them toward ions. Solving the linear system once per
configuration gives both, without differentiating through the solver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from ase.calculators.calculator import Calculator, all_changes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic_yukawa"))
from gen_yukawa import COUL, lj_energy_forces, self_energy_discrete  # noqa: E402

SCREENING_SYMBOL = "Ar"
FOUR_PI = 4.0 * np.pi


def plane_waves(L, kmax):
    """Integer modes and wavevectors with |k| <= kmax (k = 0 included).

    The selection matches gen_yukawa.kgrid (|k| <= kmax evaluated the same
    way) apart from keeping k = 0 and both members of each +-k pair.
    """
    m = int(np.ceil(kmax * L / (2 * np.pi)))
    n = np.arange(-m, m + 1)
    n3 = np.stack(np.meshgrid(n, n, n, indexing="ij"), axis=-1).reshape(-1, 3)
    kvec = 2 * np.pi * n3 / L
    keep = np.linalg.norm(kvec, axis=1) <= kmax
    return n3[keep], kvec[keep]


def structure_sum(pos, weights, kvec, chunk=128):
    """sum_i w_i exp(-i k.R_i) for every k, in atom chunks to bound memory."""
    out = np.zeros(len(kvec), dtype=complex)
    for s in range(0, len(pos), chunk):
        phase = np.exp(-1j * (pos[s : s + chunk] @ kvec.T))
        out += weights[s : s + chunk] @ phase
    return out


def gradient_sum(pos, coeff, kvec, chunk=128):
    """Im sum_k coeff_k exp(-i k.R) k for every position, shape (N, 3)."""
    out = np.empty((len(pos), 3))
    for s in range(0, len(pos), chunk):
        phase = np.exp(-1j * (pos[s : s + chunk] @ kvec.T)) * coeff[None, :]
        out[s : s + chunk] = np.imag(phase @ kvec)
    return out


class ScreenedMedium:
    """Galerkin solver for (-lap + kappa0^2 + kappa1^2 c_S) phi = 4 pi rho.

    All quantities are in solver units (charges in e, lengths in A); physical
    energies follow by multiplying with the energy scale times COUL.
    """

    def __init__(
        self,
        L,
        kappa0,
        kappa1,
        ell,
        sigma=0.6,
        kmax=3.5,
        cmax=None,
        grid=64,
        tol=1e-12,
        maxiter=5000,
        workers=-1,
    ):
        if kappa0 <= 0:
            raise ValueError("kappa0 must be positive to keep A positive definite")
        self.L = float(L)
        self.V = self.L**3
        self.kappa0, self.kappa1 = float(kappa0), float(kappa1)
        self.ell, self.sigma = float(ell), float(sigma)
        self.kmax = float(kmax)
        self.cmax = (
            np.sqrt(2.0 * np.log(1e14)) / self.ell if cmax is None else float(cmax)
        )
        self.grid, self.tol, self.maxiter = int(grid), float(tol), int(maxiter)
        self.workers = workers

        self.n_phi, self.k_phi = plane_waves(self.L, self.kmax)
        self.n_c, self.k_c = plane_waves(self.L, self.cmax)
        a = int(np.abs(self.n_phi).max())
        b = int(np.abs(self.n_c).max())
        if self.grid <= 2 * a + b:
            raise ValueError(
                f"grid {self.grid} aliases the projected products; "
                f"need more than 2*{a} + {b} = {2 * a + b} points"
            )
        self.k2_phi = np.sum(self.k_phi**2, axis=1)
        self.g_phi = np.exp(-0.5 * self.sigma**2 * self.k2_phi)
        self.g_c = np.exp(-0.5 * self.ell**2 * np.sum(self.k_c**2, axis=1))
        self.idx_phi = tuple((self.n_phi % self.grid).T)
        self.idx_c = tuple((self.n_c % self.grid).T)
        self.c0 = int(np.flatnonzero(np.all(self.n_c == 0, axis=1))[0])
        self.diag = self.k2_phi + self.kappa0**2
        self.c_k = None
        self.c_r = None
        self.iterations = 0

    # Source and concentration fields.
    def _to_real(self, coeff, idx):
        g = np.zeros((self.grid,) * 3, dtype=complex)
        g[idx] = coeff
        return sfft.ifftn(g, workers=self.workers) * self.grid**3

    def _to_coeff(self, values, idx):
        return sfft.fftn(values, workers=self.workers)[idx] / self.grid**3

    def set_medium(self, s_pos, uniform=False):
        """Concentration c_S of the screening particles (uniform: c_S = 1)."""
        if uniform:
            c_k = np.zeros(len(self.k_c), dtype=complex)
            c_k[self.c0] = 1.0
        else:
            if len(s_pos) == 0:
                raise ValueError("no screening particles; use uniform=True")
            c_k = (
                structure_sum(s_pos, np.ones(len(s_pos)), self.k_c)
                * self.g_c
                / len(s_pos)
            )
        self.c_k = c_k
        self.c_r = np.real(self._to_real(c_k, self.idx_c))
        self.preconditioner = 1.0 / (self.diag + self.kappa1**2 * np.real(c_k[self.c0]))

    def kappa_field(self):
        """kappa(r) on the grid for the current medium."""
        return np.sqrt(self.kappa0**2 + self.kappa1**2 * np.clip(self.c_r, 0, None))

    def density(self, pos, weights):
        """Coefficients of sum_i w_i g_sigma(r - R_i)."""
        return (
            structure_sum(pos, np.asarray(weights, float), self.k_phi)
            * self.g_phi
            / self.V
        )

    # Operator application and solution.
    def apply(self, x):
        prod = self._to_coeff(self.c_r * self._to_real(x, self.idx_phi), self.idx_phi)
        return self.diag * x + self.kappa1**2 * prod

    def solve(self, rhs, tol=None):
        """Preconditioned conjugate gradients for A x = rhs (A Hermitian PD)."""
        tol = self.tol if tol is None else float(tol)
        x = np.zeros_like(rhs)
        r = rhs.copy()
        z = self.preconditioner * r
        p = z.copy()
        rz = np.vdot(r, z).real
        bnorm = np.linalg.norm(rhs)
        if bnorm == 0:
            return x
        for it in range(1, self.maxiter + 1):
            Ap = self.apply(p)
            alpha = rz / np.vdot(p, Ap).real
            x += alpha * p
            r -= alpha * Ap
            if np.linalg.norm(r) <= tol * bnorm:
                self.iterations = it
                return x
            z = self.preconditioner * r
            rz_new = np.vdot(r, z).real
            p = z + (rz_new / rz) * p
            rz = rz_new
        raise RuntimeError(f"CG did not reach tol={tol:g} in {self.maxiter} iterations")

    # Energy, forces, and Green's function.
    def energy_forces(self, q_pos, q, s_pos, tol=None):
        """E_scr and forces on the charges and on the S particles.

        The energy is evaluated in the stationary form
        V [<rho, phi> - <phi, A phi> / (8 pi)], whose error is second order in
        the solver error, so finite-difference checks see the exact energy.
        """
        rho = self.density(q_pos, q)
        phi = FOUR_PI * self.solve(rho, tol)
        e = self.V * (
            np.vdot(rho, phi).real - np.vdot(phi, self.apply(phi)).real / (2 * FOUR_PI)
        )
        f_q = -np.asarray(q, float)[:, None] * gradient_sum(
            q_pos, np.conj(phi) * self.g_phi, self.k_phi
        )
        if len(s_pos):
            phi_r = np.real(self._to_real(phi, self.idx_phi))
            sq_k = self._to_coeff(phi_r**2, self.idx_c)
            f_s = (self.kappa1**2 * self.V / (2 * FOUR_PI * len(s_pos))) * gradient_sum(
                s_pos, np.conj(sq_k) * self.g_c, self.k_c
            )
        else:
            f_s = np.zeros((0, 3))
        return e, f_q, f_s

    def green(self, a_pos, a_w, b_pos, b_w, tol=None):
        """4 pi <g_a, A^{-1} g_b> for Gaussian source combinations a and b."""
        x = self.solve(self.density(b_pos, b_w), tol)
        return FOUR_PI * self.V * np.vdot(self.density(a_pos, a_w), x).real


def reference_charges(symbols, charge=0.4):
    """Na +q, Cl -q, screening particles 0."""
    table = {"Na": charge, "Cl": -charge, SCREENING_SYMBOL: 0.0}
    try:
        return np.array([table[s] for s in symbols])
    except KeyError as err:
        raise ValueError(f"unsupported species {err}") from None


class HeterogeneousScreeningCalculator(Calculator):
    """ASE calculator for E = E_LJ + s [E_scr - e_self(kappa0) sum q_i^2].

    e_self(kappa0) is the band-matched self energy of a unit Gaussian charge
    in the S-free medium (gen_yukawa.self_energy_discrete), so the constant
    removed is the same for every frame and probe state: it always uses the
    original ionic charges, never neutralized probe charges.

    neutralize=True adds -Q/N to every atom, the probe convention of the
    protocol for flipped states with net charge Q. uniform=True replaces c_S
    by 1 (validation against gen_yukawa).
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        L,
        kappa0,
        kappa1,
        ell,
        scale=1.0,
        charge=0.4,
        sigma=0.6,
        kmax=3.5,
        grid=64,
        tol=1e-12,
        lj_eps=0.20,
        lj_sigma=3.0,
        lj_rcut=4.5,
        lj_ron=3.6,
        neutralize=False,
        uniform=False,
        workers=-1,
        **kw,
    ):
        super().__init__(**kw)
        self.medium = ScreenedMedium(
            L,
            kappa0,
            kappa1,
            ell,
            sigma=sigma,
            kmax=kmax,
            grid=grid,
            tol=tol,
            workers=workers,
        )
        self.L, self.scale, self.charge = float(L), float(scale), float(charge)
        self.lj = (lj_eps, lj_sigma, lj_rcut, lj_ron)
        self.neutralize, self.uniform = bool(neutralize), bool(uniform)
        self.e_self = self_energy_discrete(kappa0, sigma, kmax, L, None)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # The plane waves are k = 2 pi n / L along the Cartesian axes, so only the
        # axis-aligned cubic cell L * I with full periodicity is supported. A rotated
        # cube has the right lengths and angles but a different reciprocal lattice.
        if not np.allclose(
            self.atoms.cell.array, self.L * np.eye(3), rtol=0, atol=1e-8
        ):
            raise ValueError(
                "calculator requires the axis-aligned cubic cell L * I it was built for"
            )
        if not np.all(self.atoms.pbc):
            raise ValueError(
                "calculator requires periodic boundary conditions along x, y and z"
            )
        pos = self.atoms.get_positions() % self.L
        q0 = reference_charges(self.atoms.get_chemical_symbols(), self.charge)
        is_s = np.array(
            [s == SCREENING_SYMBOL for s in self.atoms.get_chemical_symbols()]
        )
        q = q0 - q0.mean() if self.neutralize else q0
        if not self.neutralize and abs(q.sum()) > 1e-9:
            raise ValueError(f"net charge {q.sum():.3g} e; pass neutralize=True")
        carries = q != 0

        m = self.medium
        m.set_medium(pos[is_s], uniform=self.uniform)
        e, f_q, f_s = m.energy_forces(pos[carries], q[carries], pos[is_s])
        unit = self.scale * COUL
        e_scr = unit * e
        e_const = unit * self.e_self * np.sum(q0**2)
        e_lj, f_lj = lj_energy_forces(pos, self.L, *self.lj)

        forces = f_lj.copy()
        forces[carries] += unit * f_q
        if not self.uniform:
            forces[is_s] += unit * f_s
        energy = e_lj + e_scr - e_const
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "e_lj": e_lj,
            "e_screened": e_scr - e_const,
            "e_screened_unsubtracted": e_scr,
            "cg_iterations": m.iterations,
        }
