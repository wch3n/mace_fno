"""Summarize current full-z test and wetting results without changing the paper."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    tests, wettings = [], []
    for task in manifest["tasks"]:
        folder = root / "results" / task["name"]
        status = json.loads((folder / "status.json").read_text())
        assert status["state"] == "complete" and status["checks_passed"], task["name"]
        test = json.loads((folder / "test.json").read_text())
        wet = json.loads((folder / "endpoints/results.json").read_text())
        with Path(task["checkpoint"]).open("rb") as f:
            assert (
                hashlib.file_digest(f, "sha256").hexdigest()
                == task["checkpoint_sha256"]
            )
        assert test["checkpoint"] == wet["model"] == task["checkpoint"]
        assert test["samples"] == 500 and not test["energy_shift_applied"]
        assert wet["fmax_eV_per_A"] == 0.01 and wet["maximum_steps"] == 500
        metrics = test["splits"]["test"]["mace_fno"]
        meta = dict(
            regime=task["regime"],
            spectral_rank=task.get("spectral_rank", 0),
            seed=task["seed"],
            preliminary=task["preliminary"],
            selected_step=task["selected_step"],
            observed_step=task["observed_step"],
            checkpoint_sha256=task["checkpoint_sha256"],
        )
        tests.append(
            dict(
                **meta,
                energy_rmse_meV_atom=1000 * metrics["energy_rmse"],
                force_rmse_meV_A=1000 * metrics["force_rmse"],
                energy_ME_meV_atom=1000 * metrics["energy_bias"],
            )
        )
        row = dict(meta)
        for surface in ["undoped", "doped"]:
            nonwet, yeswet = [
                wet["records"][f"{state}-{surface}"] for state in ["1", "3"]
            ]
            assert all(
                r["converged"] and r["final_fmax_eV_per_A"] <= 0.01
                for r in [nonwet, yeswet]
            )
            ref = nonwet["reference_energy_eV"]
            nw, w = [1000 * (r["final_energy_eV"] - ref) for r in [nonwet, yeswet]]
            delta = 1000 * (yeswet["final_energy_eV"] - nonwet["final_energy_eV"])
            refdelta = 1000 * (yeswet["reference_energy_eV"] - ref)
            assert math.isclose(
                delta, wet["summary"][surface]["predicted_delta_meV"], abs_tol=1e-8
            )
            row.update(
                {
                    surface + "_nonwetting_meV": nw,
                    surface + "_wetting_meV": w,
                    surface + "_preference_meV": delta,
                    surface + "_deviation_meV": delta - refdelta,
                }
            )
        wettings.append(row)
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "test_per_seed.csv", tests)
    write_csv(output / "wetting_per_seed.csv", wettings)
    summaries = {}
    for name, rows in [("test", tests), ("wetting", wettings)]:
        result = []
        keys = [
            k
            for k in rows[0]
            if k
            not in [
                "regime",
                "spectral_rank",
                "seed",
                "preliminary",
                "selected_step",
                "observed_step",
                "checkpoint_sha256",
            ]
        ]
        for rank, regime in sorted({(r["spectral_rank"], r["regime"]) for r in rows}):
            group = [
                r for r in rows if r["regime"] == regime and r["spectral_rank"] == rank
            ]
            assert sorted(r["seed"] for r in group) == manifest["seeds"]
            record = dict(
                regime=regime,
                n=len(group),
                preliminary=any(r["preliminary"] for r in group),
                z_mixing="full",
                spectral_rank=rank,
            )
            for key in keys:
                record[key + "_mean"] = statistics.mean(r[key] for r in group)
                record[key + "_sd"] = (
                    statistics.stdev(r[key] for r in group) if len(group) > 1 else None
                )
            result.append(record)
        summaries[name] = result
        write_csv(output / f"{name}_summary.csv", result)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
