"""Archive matched wetting energies; summarize only complete five-seed groups."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from campaign import SEEDS, STATES, digest, load, save


def collect(root, output, allow_partial=False):
    manifest = load(root / "manifest.json")
    rows, records, missing = [], [], []
    for t in manifest["tasks"]:
        folder = root / "results" / t["name"]
        if (
            not (folder / "status.json").exists()
            or load(folder / "status.json")["state"] != "complete"
        ):
            missing.append(t["name"])
            continue
        status, checks, result = [
            load(folder / name)
            for name in ["status.json", "checks.json", "endpoints/results.json"]
        ]
        provenance = load(root / "provenance" / (t["name"] + ".json"))
        assert status["checks_passed"] and checks["all_converged_expected_basins"]
        assert status["manifest_sha256"] == digest(root / "manifest.json")
        assert (
            digest(result["model"])
            == provenance["checkpoint_sha256"]
            == status["checkpoint_sha256"]
        )
        assert (
            provenance["completed_steps"] == 240000
            and provenance["test"]["samples"] == 500
        )
        assert (
            provenance["source_checkpoint"]
            == provenance["test"]["checkpoint"]
            == t["checkpoint"]
        )
        assert not provenance["test"]["energy_shift_applied"]
        assert (
            result["fmax_eV_per_A"],
            result["maximum_steps"],
            result["dtype"],
            result["optimizer"],
        ) == (0.01, 500, "float64", "ASE FIRE")
        assert set(result["records"]) == set(checks["states"]) == set(STATES)
        row = dict(
            depth=t["depth"],
            regime=t["opt_level"],
            seed=t["seed"],
            checkpoint_sha256=status["checkpoint_sha256"],
        )
        for state in STATES:
            r, c = result["records"][state], checks["states"][state]
            assert all(
                c[k]
                for k in [
                    "converged",
                    "expected_orientation",
                    "substrate_fixed",
                    "cell_fixed",
                ]
            )
            assert r["converged"] and r["final_fmax_eV_per_A"] <= 0.01
            assert len(r["movable_indices"]) == 2 and len(r["fixed_indices"]) == 108
        for surface in ["undoped", "doped"]:
            nw, w = [result["records"][f"{i}-{surface}"] for i in [1, 3]]
            reference = 1000 * (w["reference_energy_eV"] - nw["reference_energy_eV"])
            difference = 1000 * (w["final_energy_eV"] - nw["final_energy_eV"])
            assert math.isclose(
                difference,
                result["summary"][surface]["predicted_delta_meV"],
                abs_tol=1e-8,
            )
            row.update(
                {
                    surface + "_nonwetting_meV": 1000
                    * (nw["final_energy_eV"] - nw["reference_energy_eV"]),
                    surface + "_wetting_meV": 1000
                    * (w["final_energy_eV"] - nw["reference_energy_eV"]),
                    surface + "_preference_meV": difference,
                    surface + "_reference_meV": reference,
                    surface + "_deviation_meV": difference - reference,
                }
            )
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, float))
        rows.append(row)
        records.append(
            dict(
                task=t,
                status=status,
                checks=checks,
                result=result,
                provenance=provenance,
            )
        )
    assert allow_partial or not missing, missing
    assert rows, "No complete evaluations to collect"
    summaries = []
    for depth in [1, 2]:
        for regime in ["frozen", "joint"]:
            group = [r for r in rows if (r["depth"], r["regime"]) == (depth, regime)]
            if sorted(r["seed"] for r in group) != SEEDS:
                continue
            summary = dict(depth=depth, regime=regime, n=5)
            for key in [k for k in rows[0] if k.endswith("_meV")]:
                summary[key + "_mean"] = statistics.mean(r[key] for r in group)
                summary[key + "_sd"] = statistics.stdev(r[key] for r in group)
            summaries.append(summary)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [("per_seed", rows), ("summary", summaries)]:
        save(output / (name + ".json"), values)
        if values:
            with (output / (name + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    save(
        output / "sources.json",
        dict(campaign=str(root), manifest=manifest, records=records, missing=missing),
    )
    save(
        output / "status.json",
        dict(
            state="partial" if missing else "complete",
            completed_models=len(rows),
            complete_groups=len(summaries),
            missing=missing,
            manuscript_edited=False,
        ),
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    collect(
        args.root.resolve(),
        args.output_dir or args.root / "analysis",
        args.allow_partial,
    )
