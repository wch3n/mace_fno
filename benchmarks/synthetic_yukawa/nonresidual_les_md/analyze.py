"""Use the original shell estimator and reject incomplete or unstable runs."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from md_common import read_log, run_complete, unstable
from szz_core import N2, mean_se, sk_frames
from verify import ROOT, digest, dump


def statistics(run):
    complete, why = run_complete(str(run))
    assert complete, why
    bad, why = unstable(str(run))
    assert not bad, why
    meta = json.loads((run / "meta.json").read_text())
    state = np.load(meta["from_state"]["path"])
    charges = state["ref_charges"]
    assert charges.shape == (1146,) and abs(charges.sum()) < 1e-10
    np.testing.assert_allclose(abs(charges), 0.4, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state["cell"], np.eye(3) * 32, rtol=0, atol=1e-12)
    assert meta["args"]["time"] == 1000 and meta["args"]["dt"] == 2
    assert meta["args"]["ensemble"] == "nvt" and meta["args"]["save_fs"] == 500
    positions = np.load(run / "positions.npz")["positions"].astype(float)
    assert positions.shape == (2001, 1146, 3)
    z, n = sk_frames(positions, charges, 32.0)
    zm, ze = mean_se(np.array_split(z, 10))
    nm, ne = mean_se(np.array_split(n, 10))
    assert np.isfinite([zm, ze, nm, ne]).all() and (zm > 0).all() and (ze >= 0).all()
    return zm, ze, nm, ne, meta


def main(mode, index):
    m = json.loads((ROOT / "manifest.json").read_text())
    task = m["tasks"][index]
    kappa = float(task["kappa"].replace("p", "."))
    rows = list(csv.DictReader((ROOT / "provenance/baseline_szz.csv").open()))
    baseline = [r for r in rows if float(r["kappa"]) == kappa]
    if mode == "check":
        run = (
            Path(task["old_directory"]).parents[1]
            / "md/macelesc"
            / f"kappa_{task['kappa']}"
            / "nvt_1ns"
        )
        zm, ze, _, _, _ = statistics(run)
        old = sorted(
            (r for r in baseline if r["model"] == "maceles"),
            key=lambda r: int(r["shell"]),
        )
        assert len(old) == 15
        np.testing.assert_allclose(
            zm, [float(r["szz"]) for r in old], atol=1e-12, rtol=1e-10
        )
        np.testing.assert_allclose(
            ze, [float(r["szz_se"]) for r in old], atol=1e-12, rtol=1e-10
        )
        report = dict(
            status="passed",
            legacy_run=str(run),
            positions_sha256=digest(run / "positions.npz"),
            check="All 15 ordinary LES shell means and sampling errors reproduce the current Fig. 4 source.",
        )
        dump(ROOT / "validation" / task["dataset"] / "szz_estimator.json", report)
    else:
        run = ROOT / "md" / task["dataset"] / "nvt_1ns"
        zm, ze, nm, ne, meta = statistics(run)
        model = Path(meta["calculator"]["model"])
        evaluation = json.loads(
            (ROOT / task["run_dir"] / "evaluation.json").read_text()
        )
        assert (
            meta["calculator"]["model_sha256"]
            == evaluation["model_sha256"]
            == digest(model)
        )
        assert evaluation["first_block"] == "RealAgnosticInteractionBlock"
        reference = sorted(
            (r for r in baseline if r["model"] == "reference_md"),
            key=lambda r: int(r["shell"]),
        )
        assert [int(r["n2"]) for r in reference] == N2
        ref = np.array([float(r["szz"]) for r in reference])
        ref_se = np.array([float(r["szz_se"]) for r in reference])
        ratio = zm / ref
        ratio_se = ratio * np.hypot(ze / zm, ref_se / ref)
        log = read_log(str(run))
        report = dict(
            kappa=kappa,
            seed=17,
            complete=True,
            stable=True,
            run=str(run),
            model_sha256=digest(model),
            positions_sha256=digest(run / "positions.npz"),
            state_sha256=digest(Path(meta["from_state"]["path"])),
            metadata_sha256=digest(run / "meta.json"),
            interaction_first=evaluation["first_block"],
            evaluation=evaluation,
            n2=N2,
            k_inv_A=(2 * np.pi / 32 * np.sqrt(N2)).tolist(),
            szz=zm.tolist(),
            szz_se=ze.tolist(),
            snn=nm.tolist(),
            snn_se=ne.tolist(),
            ratio=ratio.tolist(),
            ratio_se=ratio_se.tolist(),
            n_groups=10,
            n_frames=2001,
            production_ps=1000,
            reference_rows=reference,
            mean_temperature_K=float(log["T_K"].mean()),
            minimum_pair_A=float(log["rmin_A"].min()),
            maximum_force_eV_A=float(log["fmax_eV_A"].max()),
            mean_potential_energy_halves_meV_atom=[
                float(v.mean() / 1146 * 1000) for v in np.array_split(log["epot_eV"], 2)
            ],
            sampling="Ten time blocks of one 1 ns trajectory; one trained seed; fixed physical +/-0.4 e charges",
        )
        dump(ROOT / "results" / f"{task['dataset']}_szz.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
