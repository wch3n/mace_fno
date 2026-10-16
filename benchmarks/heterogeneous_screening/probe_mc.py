"""Equilibrated remote variants for the remote-screening probe.

Identities are resampled at fixed positions by Metropolis Monte Carlo under the
driver energy of the dataset,

    E = s_d E_Yukawa(ions; kappa_d) + sum_{m in S} V(R_m)     (+ E_LJ, unchanged:
                                                                it is species-blind)

with only sites outside the frozen region allowed to change. Two moves:
S <-> ion exchange (an ion's charge moves to an S site and back) and Na <-> Cl
exchange. The Yukawa energy change is evaluated from the structure factor,
dE = (s_d / V) sum_{half k} w_k (2 Re(conj(rho_k) drho_k) + |drho_k|^2), with
the mode set and weights of gen_yukawa.py, so a move costs O(N_k).

The remote charge order thus re-equilibrates around each new S arrangement, as
in the training frames, instead of being assigned at random (which put the
first probe's variants ~0.25 eV/atom above their base frames).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from screening_reference import SCREENING_SYMBOL, reference_charges

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic_yukawa"))
from gen_yukawa import COUL, kernel, kgrid  # noqa: E402

L, KT = 32.0, 8.617333e-5 * 2800.0


class IdentityMC:
    def __init__(
        self,
        pos,
        sym,
        movable,
        kappa_d=0.05,
        scale_d=1.851,
        sigma=0.6,
        kmax=3.5,
        field=None,
        rng=None,
    ):
        self.pos, self.sym = pos, np.array(sym, dtype=object)
        self.movable = np.flatnonzero(movable)
        self.rng = rng or np.random.default_rng()
        kvec, kmag, _ = kgrid(L, kmax)
        self.w = (
            scale_d
            * COUL
            * kernel(kmag, kappa_d)
            * np.exp(-(sigma**2) * kmag**2)
            / L**3
        )
        self.kvec = kvec
        self.phase = np.exp(-1j * (pos[self.movable] @ kvec.T))  # movable sites only
        self.slot = {int(i): n for n, i in enumerate(self.movable)}
        q = reference_charges(self.sym)
        self.rho = q @ np.exp(-1j * (pos @ kvec.T))
        self.v = (field(pos) / KT) if field is not None else np.zeros(len(pos))  # in kT
        self.accepted = 0

    def _de_yukawa(self, drho):
        return float(
            np.sum(self.w * (2 * np.real(np.conj(self.rho) * drho) + np.abs(drho) ** 2))
        )

    def energy_yukawa(self):
        return float(np.sum(self.w * np.abs(self.rho) ** 2))

    def run(self, sweeps, moves=("swap_s", "swap_q")):
        sym, rng = self.sym, self.rng
        for _ in range(sweeps * len(self.movable)):
            kind = moves[rng.integers(len(moves))]
            i = int(self.movable[rng.integers(len(self.movable))])
            j = int(self.movable[rng.integers(len(self.movable))])
            si, sj = sym[i], sym[j]
            if kind == "swap_s":  # S at i, ion at j: the ion's charge moves j -> i
                if si != SCREENING_SYMBOL:
                    si, sj, i, j = sj, si, j, i
                if si != SCREENING_SYMBOL or sj == SCREENING_SYMBOL:
                    continue
                q = 0.4 if sj == "Na" else -0.4
                drho = q * (self.phase[self.slot[i]] - self.phase[self.slot[j]])
                dv = self.v[j] - self.v[i]  # S moves i -> j
            else:  # Na at i, Cl at j
                if {si, sj} != {"Na", "Cl"}:
                    continue
                if si != "Na":
                    i, j = j, i
                drho = -0.8 * self.phase[self.slot[i]] + 0.8 * self.phase[self.slot[j]]
                dv = 0.0
            de = self._de_yukawa(drho) / KT + dv
            if de <= 0 or rng.random() < np.exp(-de):
                sym[i], sym[j] = sym[j], sym[i]
                self.rho += drho
                self.accepted += 1
        return sym


def frozen_mask(pos, sites, radius):
    m = np.zeros(len(pos), bool)
    for s in sites:
        d = pos - pos[s]
        d -= L * np.round(d / L)
        m |= np.linalg.norm(d, axis=1) <= radius
    return m


def make_variant(sym0, pos, sites, kind, rng, radius=10.0, sweeps=100, field_amp=4.0):
    """Equilibrated variant; returns symbols and diagnostics."""
    from gen_heterogeneous import ShellField

    movable = ~frozen_mask(pos, sites, radius)
    if kind == "base":
        return np.array(sym0, dtype=object), {"n_remote": int(movable.sum())}
    field = None
    moves = ("swap_s", "swap_q")
    if kind.startswith("field"):
        f = ShellField(rng, field_amp, 2800.0)
        field = f.energy
    elif kind.startswith("null"):
        moves = ("swap_q",)
    mc = IdentityMC(pos, sym0, movable, field=field, rng=rng)
    e0 = mc.energy_yukawa()
    sym = mc.run(sweeps, moves)
    q = reference_charges(sym)
    assert abs(q.sum()) < 1e-9 and np.all(
        sym[~movable] == np.array(sym0, dtype=object)[~movable]
    )
    return sym, {
        "n_remote": int(movable.sum()),
        "acceptance": mc.accepted / (sweeps * movable.sum()),
        "dE_yukawa_eV": mc.energy_yukawa() - e0,
    }
