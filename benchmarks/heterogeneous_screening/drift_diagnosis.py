"""Decompose the driver-energy drift over the sampled frames of every trajectory."""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

# Configure BLAS threading before importing numerical libraries.
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from screening_reference import COUL, ScreenedMedium, reference_charges  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic_yukawa"))
from gen_yukawa import kernel, kgrid, lj_energy_forces, nl_energy_forces  # noqa: E402

L, KT = 32.0, 8.617333e-5 * 2800
R = Path(sys.argv[1])
meta = json.loads((R / "metadata.json").read_text())
fields = {m["traj"]: m["field"] for m in meta["md"]["trajectories"]}
args = meta["md"]["args"]
kvec, kmag, _ = kgrid(L, 3.5)
w = (
    args["drive_scale"]
    * COUL
    * kernel(kmag, args["drive_kappa"])
    * np.exp(-0.36 * kmag**2)
)
print(
    f"driver: kappa_d = {args['drive_kappa']}, s_d = {args['drive_scale']}, relax {args['relax_ps']} ps"
)
k1 = 2 * np.pi * np.eye(3) / L


def task(f):
    frames = read(f, index=":")
    t = frames[0].info["traj"]
    fd = fields[t]
    a, th, amp = np.array(fd["a"]), np.array(fd["theta"]), fd["amp_kT"]
    med = ScreenedMedium(L, 0.05, 1.0, 3.0, grid=48, workers=1)
    rows = []
    for fr in frames:
        pos = fr.positions % L
        sym = np.array(fr.get_chemical_symbols())
        q = reference_charges(sym)
        ion, s = q != 0, sym == "Ar"
        e_y, _ = nl_energy_forces(pos[ion], q[ion], L**3, kvec, w, 0.0)
        e_lj, _ = lj_energy_forces(pos, L, 0.20, 3.0, 4.5, 3.6)
        v = amp * KT * np.sum(np.cos(pos[s] @ k1.T + th) @ a)
        med.set_medium(pos[s])
        rows.append((fr.info["time_ps"], e_lj, e_y, v, np.std(med.c_r)))
    return t, np.array(rows)


with mp.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
    res = dict(pool.map(task, sorted((R / "md").glob("traj*.xyz"))))
arr = np.stack([res[t] for t in sorted(res)])  # (traj, frame, 5)
n = 1146
t = arr[0, :, 0]
names = ["E_LJ", "E_Yukawa", "V_ext"]
print(
    "mean over trajectories, meV/atom relative to the first sampled frame (t = %.1f ps):"
    % t[0]
)
for i, nm in enumerate(names, start=1):
    rel = 1e3 * (arr[:, :, i] - arr[:, :1, i]) / n
    print(
        f"  {nm:9s}: "
        + " ".join(f"{x:6.1f}" for x in rel.mean(0)[::2])
        + f"   (frames {t[::2].round(1).tolist()} ps)"
    )
tot = 1e3 * (arr[:, :, 1:4].sum(2) - arr[:, :1, 1:4].sum(2)) / n
print(f"  {'total':9s}: " + " ".join(f"{x:6.1f}" for x in tot.mean(0)[::2]))
c = arr[:, :, 4]
print(
    "std of c_S over the cell (mean over trajectories): "
    + " ".join(f"{x:.3f}" for x in c.mean(0)[::2])
)
half = arr.shape[1] // 2
dc = c[:, half:].mean(1) - c[:, :half].mean(1)
print(
    f"change of c_S std between sampling halves: mean {dc.mean():+.3f}, range {dc.min():+.3f} to {dc.max():+.3f}"
)
amps = np.array([fields[t]["amp_kT"] for t in sorted(res)])
dtot = tot[:, half:].mean(1) - tot[:, :half].mean(1)
print(
    f"total drift between halves vs field amplitude: corr {np.corrcoef(amps, dtot)[0, 1]:+.2f}; "
    f"A<2: {dtot[amps < 2].mean():+.2f}, A>3: {dtot[amps > 3].mean():+.2f} meV/atom"
)
