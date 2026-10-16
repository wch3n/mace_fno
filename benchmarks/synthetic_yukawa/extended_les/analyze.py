"""Analyze complete homogeneous extended-LES MD against frozen baseline curves."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from md_common import read_log, run_complete, unstable
from prepare import sha
from szz_core import N2, mean_se, sk_frames

ROOT = Path(__file__).resolve().parent


def shell_statistics(run):
    ok, why = run_complete(str(run))
    assert ok, why
    bad, why = unstable(str(run))
    assert not bad, why
    meta = json.loads((run / "meta.json").read_text())
    state = np.load(meta["from_state"]["path"])
    charges = state["ref_charges"]
    assert len(charges) == 1146 and abs(charges.sum()) < 1e-10
    np.testing.assert_allclose(abs(charges), 0.4, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state["cell"], np.eye(3) * 32, atol=1e-12, rtol=0)
    positions = np.load(run / "positions.npz")["positions"].astype(float)
    z, n = sk_frames(positions, charges, 32.0)
    zm, ze = mean_se(np.array_split(z, 10))
    nm, ne = mean_se(np.array_split(n, 10))
    return zm, ze, nm, ne, meta


def baselines():
    with (ROOT / "provenance/baseline_szz.csv").open() as stream:
        return [r for r in csv.DictReader(stream) if float(r["kappa"]) == 0.5]


def selfcheck():
    # A complete existing trajectory must reproduce the published CSV exactly
    # under the copied shell routines, charge assignment and ten-block grouping.
    run = ROOT.parent.parent / "yukawa_L32_md2800/md/macelesc/kappa_0p5/nvt_1ns"
    zm, ze, _, _, _ = shell_statistics(run)
    rows = sorted(
        (r for r in baselines() if r["model"] == "maceles"),
        key=lambda r: int(r["shell"]),
    )
    assert len(rows) == 15
    np.testing.assert_allclose(
        zm, [float(r["szz"]) for r in rows], atol=1e-12, rtol=1e-10
    )
    np.testing.assert_allclose(
        ze, [float(r["szz_se"]) for r in rows], atol=1e-12, rtol=1e-10
    )
    report = dict(
        status="passed",
        legacy_run=str(run),
        positions_sha256=sha(run / "positions.npz"),
        checks="All 15 shell means and sampling SE reproduce the existing ordinary LES CSV",
    )
    (ROOT / "szz_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    kind = sys.argv[1]
    if kind == "check":
        selfcheck()
        return
    run = ROOT / "md" / ("nvt_1ns" if kind == "md" else "nve_100ps")
    ok, why = run_complete(str(run))
    assert ok, why
    bad, why = unstable(str(run))
    assert not bad, why
    log = read_log(str(run))
    result = dict(
        run=str(run),
        complete=True,
        stable=True,
        mean_temperature_K=float(log["T_K"].mean()),
        minimum_pair_A=float(log["rmin_A"].min()),
        maximum_force_eV_A=float(log["fmax_eV_A"].max()),
    )
    if kind == "nve":
        slope = np.polyfit(log["t_ps"], log["etot_eV"] / 1146 * 1000, 1)[0]
        result.update(
            energy_slope_meV_atom_ps=float(slope),
            fitted_drift_over_100ps_meV_atom=float(100 * slope),
            energy_std_meV_atom=float(log["etot_eV"].std() / 1146 * 1000),
        )
        (ROOT / "results/nve.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        zm, ze, nm, ne, meta = shell_statistics(run)
        rows = baselines()
        reference = sorted(
            (r for r in rows if r["model"] == "reference_md"),
            key=lambda r: int(r["shell"]),
        )
        assert len(reference) == 15 and [int(r["n2"]) for r in reference] == N2
        ref = np.array([float(r["szz"]) for r in reference])
        ref_se = np.array([float(r["szz_se"]) for r in reference])
        ratio = zm / ref
        ratio_se = ratio * np.hypot(ze / zm, ref_se / ref)
        for j, n2 in enumerate(N2):
            rows.append(
                dict(
                    kappa=0.5,
                    model="extended_les",
                    label="Extended LES (induced response)",
                    shell=j + 1,
                    n2=n2,
                    k_inv_angstrom=float(2 * np.pi / 32 * np.sqrt(n2)),
                    szz=float(zm[j]),
                    szz_se=float(ze[j]),
                    ratio=float(ratio[j]),
                    ratio_se=float(ratio_se[j]),
                    n_groups=10,
                    error_basis="SE over ten blocks of one 1 ns trajectory; one trained seed",
                    status="complete",
                )
            )
        with (ROOT / "results/szz_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = json.loads((ROOT / "manifest.json").read_text())
        model_dir = Path(manifest["tasks"][1]["train"]["directory"])
        energy_halves = [
            float(x.mean() / 1146 * 1000) for x in np.array_split(log["epot_eV"], 2)
        ]
        result.update(
            n2=N2,
            szz=zm.tolist(),
            szz_se=ze.tolist(),
            snn=nm.tolist(),
            snn_se=ne.tolist(),
            ratio=ratio.tolist(),
            ratio_se=ratio_se.tolist(),
            k_inv_A=(2 * np.pi / 32 * np.sqrt(N2)).tolist(),
            n_groups=10,
            sampling="Ten time blocks, single training seed; fixed physical charges",
            mean_potential_energy_halves_meV_atom=energy_halves,
            convergence=json.loads((model_dir / "convergence.json").read_text()),
            model_sha256=meta["calculator"]["model_sha256"],
            positions_sha256=sha(run / "positions.npz"),
            comparison_scope=manifest["comparison_scope"],
        )
        (ROOT / "results/szz.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
