"""Archive matched dense/low-rank test and wetting comparisons (no paper edits)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-campaign", type=Path, required=True)
    parser.add_argument("--low-rank-campaign", type=Path, required=True)
    parser.add_argument("--dense-costs", type=Path, required=True)
    parser.add_argument("--gpu-campaign", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def read(path):
        path = path.resolve()
        content = path.read_bytes()
        sources[str(path)] = dict(
            sha256=hashlib.sha256(content).hexdigest(), text=content.decode()
        )
        return json.loads(content)

    dense, low = [p.resolve() for p in [args.dense_campaign, args.low_rank_campaign]]
    dm, lm = [read(p / "manifest.json") for p in [dense, low]]
    for name in ["data/test.xyz", "models/mace_nl1.model"] + [
        f"inputs/{n}.xyz" for n in ["1-undoped", "3-undoped", "1-doped", "3-doped"]
    ]:
        assert dm["checksums"][name] == lm["checksums"][name], name
    for root in [dense, low]:
        read(root / "submission.json")
    counts = {
        r["name"]: r["counts"] for r in read(low / "preflight.json")["checkpoint_loads"]
    }
    dense_counts = {
        r["regime"]: r["parameters"]
        for r in read(args.dense_costs)
        if r["mixing"] == "full"
    }
    rows, baselines = [], {}
    for root, manifest in [(dense, dm), (low, lm)]:
        for task in manifest["tasks"]:
            if task["seed"] not in lm["seeds"]:
                continue
            folder = root / "results" / task["name"]
            status, checks, test, wet = [
                read(folder / name)
                for name in [
                    "status.json",
                    "checks.json",
                    "test.json",
                    "endpoints/results.json",
                ]
            ]
            assert status["state"] == "complete" and status["checks_passed"]
            assert checks["all_converged_expected_basins"]
            assert test["checkpoint"] == wet["model"] == task["checkpoint"]
            with Path(task["checkpoint"]).open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    == task["checkpoint_sha256"]
                )
            assert test["samples"] == 500 and not test["energy_shift_applied"]
            assert wet["fmax_eV_per_A"] == 0.01 and wet["maximum_steps"] == 500
            baseline = test["splits"]["test"]["frozen_mace"]
            if baselines:
                for key in ["energy_rmse", "force_rmse"]:
                    assert math.isclose(
                        baseline[key],
                        next(iter(baselines.values()))[key],
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    )
            baselines[task["name"]] = baseline
            rank = task.get("spectral_rank", 0)
            metrics = test["splits"]["test"]["mace_fno"]
            c = dense_counts[task["regime"]] if rank == 0 else counts[task["name"]]
            fno = c["added"] if rank == 0 else c["source_head"] + c["long_range"]
            row = dict(
                regime=task["regime"],
                seed=task["seed"],
                spectral_rank=rank,
                selected_step=task["selected_step"],
                observed_step=task["observed_step"],
                preliminary=task["preliminary"],
                checkpoint_sha256=task["checkpoint_sha256"],
                energy_rmse_meV_atom=1000 * metrics["energy_rmse"],
                force_rmse_meV_A=1000 * metrics["force_rmse"],
                fno_parameters=fno,
                total_parameters=fno + c["backbone"],
            )
            for surface in ["undoped", "doped"]:
                nw, w = [wet["records"][f"{state}-{surface}"] for state in ["1", "3"]]
                assert all(
                    r["converged"] and r["final_fmax_eV_per_A"] <= 0.01 for r in [nw, w]
                )
                pred = 1000 * (w["final_energy_eV"] - nw["final_energy_eV"])
                ref = 1000 * (w["reference_energy_eV"] - nw["reference_energy_eV"])
                assert math.isclose(
                    pred, wet["summary"][surface]["predicted_delta_meV"], abs_tol=1e-8
                )
                row.update(
                    {
                        surface + "_nonwetting_meV": 1000
                        * (nw["final_energy_eV"] - nw["reference_energy_eV"]),
                        surface + "_wetting_meV": 1000
                        * (w["final_energy_eV"] - nw["reference_energy_eV"]),
                        surface + "_preference_meV": pred,
                        surface + "_reference_preference_meV": ref,
                    }
                )
            rows.append(row)
    dense_rows = {
        (r["regime"], r["seed"]): r.copy() for r in rows if r["spectral_rank"] == 0
    }
    for row in rows:
        reference = dense_rows[row["regime"], row["seed"]]
        for key in ["energy_rmse_meV_atom", "force_rmse_meV_A"]:
            row[key + "_change_pct"] = 100 * (row[key] / reference[key] - 1)
        row["fno_parameter_reduction_factor"] = (
            reference["fno_parameters"] / row["fno_parameters"]
        )
        for surface in ["undoped", "doped"]:
            key = surface + "_preference_meV"
            assert (
                row[surface + "_reference_preference_meV"]
                == reference[surface + "_reference_preference_meV"]
            )
            row[key + "_difference_from_dense"] = row[key] - reference[key]
    rows.sort(key=lambda r: (r["regime"], r["seed"], r["spectral_rank"]))
    gpu_results = []
    if args.gpu_campaign:
        gpu = args.gpu_campaign.resolve()
        gm = read(gpu / "manifest.json")
        assert read(gpu / "gpu_status.json")["state"] == "complete"
        read(gpu / "submission.json")
        for name in ["measure_gpu.py", "run_gpu.py", "job.slurm"]:
            payload = (gpu / name).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == gm["snapshot_hashes"][name]
            sources[str(gpu / name)] = dict(
                sha256=gm["snapshot_hashes"][name], text=payload.decode()
            )
        for record in gm["records"]:
            measured = read(gpu / "measurements" / f"{record['name']}.json")
            row = next(
                r
                for r in rows
                if (r["regime"], r["seed"], r["spectral_rank"])
                == (record["regime"], record["seed"], record["spectral_rank"])
            )
            assert record["checkpoint_provenance"]["sha256"] == row["checkpoint_sha256"]
            assert (
                measured["state"] == "complete"
                and len(measured["timings"]) == 90
                and len(measured["memory_calls"]) == 9
            )
            assert measured["parameters"]["added"] == row["fno_parameters"]
            assert measured["parameters"]["total"] == row["total_parameters"]
            row.update(
                gpu_runtime_ms=measured["median_ms"],
                gpu_peak_tensor_MiB=measured["peak_tensor_MiB"],
            )
            gpu_results.append(measured)
        assert len(gpu_results) == 3
        assert len({m["environment"]["gpu"] for m in gpu_results}) == 1
        for row in rows:
            row.setdefault("gpu_runtime_ms", None)
            row.setdefault("gpu_peak_tensor_MiB", None)
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    script = Path(__file__).resolve()
    payload = script.read_bytes()
    sources[str(script)] = dict(
        sha256=hashlib.sha256(payload).hexdigest(), text=payload.decode()
    )
    (output / "sources.json").write_text(
        json.dumps(sources, indent=2, allow_nan=False) + "\n"
    )
    if gpu_results:
        (output / "gpu_measurements.json").write_text(
            json.dumps(gpu_results, indent=2, allow_nan=False) + "\n"
        )
    (output / "verification.json").write_text(
        json.dumps(
            dict(
                matched_seed=True,
                matched_input_hashes=True,
                unshifted_test_structures=500,
                baseline_metrics_agree=True,
                all_endpoints_pass=True,
                models=len(rows),
                same_checkpoint_for_test_and_wetting=True,
                unequal_training_progress=True,
                joint_gpu_models=len(gpu_results),
                gpu_checkpoints_match_accuracy=bool(gpu_results),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
