"""Collect completed extended-LES fits, including the probe null control."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "manifest.json").read_text())
rows = []
for task in manifest["tasks"]:
    run = Path(task["train"]["directory"])
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "completed"
    metrics = json.loads((run / "test.json").read_text())
    analysis = json.loads((run / "probe_analysis.json").read_text())["models"][
        task["name"]
    ]
    obs = json.loads((run / "probe_models.json").read_text())[task["name"]]["obs"]
    null = [
        np.asarray(value) - np.asarray(obs[key.split("|")[0] + "|base"])
        for key, value in obs.items()
        if "|null" in key
    ]
    assert len(null) == 80
    rows.append(
        dict(
            name=task["name"],
            seed=task["seed"],
            training_backend=task.get("training_backend", "e3nn"),
            fit_scope=manifest.get("fit_scope", "original full budget"),
            training_budget_epochs=manifest["training"]["epochs"],
            stage_two_start=manifest["training"]["stage_two_start"],
            energy_RMSE_meV_atom=metrics["energy_RMSE_meV_atom"],
            force_RMSE_meV_A=metrics["force_RMSE_meV_A"],
            parameters=metrics["total_parameters"],
            slope=analysis["with_base"]["slope"],
            slope_ci95_lo=analysis["with_base"]["ci95"][0],
            slope_ci95_hi=analysis["with_base"]["ci95"][1],
            captured=analysis["with_base"]["captured"],
            K_RMSE_meV_A2=1e3 * analysis["with_base"]["raw_rmse"],
            null_RMS_meV_A2=float(1e3 * np.sqrt(np.mean(np.asarray(null) ** 2))),
        )
    )
    if "convergence" in status:
        rows[-1].update(
            converged=status["convergence"]["converged"],
            stop_reason=status["convergence"]["stop_reason"],
            last_epoch=status["convergence"]["last"]["epoch"],
        )
with (ROOT / "results/summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(ROOT / "results/summary.json").write_text(json.dumps(rows, indent=2) + "\n")
if len({task["variant"] for task in manifest["tasks"]}) == 1 and len(rows) > 1:
    fields = (
        "energy_RMSE_meV_atom",
        "force_RMSE_meV_A",
        "slope",
        "captured",
        "K_RMSE_meV_A2",
        "null_RMS_meV_A2",
    )
    aggregate = dict(
        seeds=[r["seed"] for r in rows],
        n_seeds=len(rows),
        all_converged=all(r.get("converged", False) for r in rows),
        uncertainty="sample SD across independent training seeds; shared test/probe data",
        metrics={
            k: dict(
                mean=float(np.mean([r[k] for r in rows])),
                sd=float(np.std([r[k] for r in rows], ddof=1)),
            )
            for k in fields
        },
    )
    (ROOT / "results/summary_seeds.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
if manifest.get("parent_results"):
    parent_path = Path(manifest["parent_results"])
    comparison = dict(
        status="awaiting_residual_results", parent_results=str(parent_path)
    )
    if parent_path.is_file():
        old = {r["seed"]: r for r in json.loads(parent_path.read_text())}
        assert set(old) == {r["seed"] for r in rows}
        metrics = (
            "energy_RMSE_meV_atom",
            "force_RMSE_meV_A",
            "slope",
            "K_RMSE_meV_A2",
            "null_RMS_meV_A2",
        )
        paired = [
            dict(
                seed=r["seed"],
                differences={k: r[k] - old[r["seed"]][k] for k in metrics},
                residual_converged=old[r["seed"]].get("converged", False),
                nonresidual_converged=r.get("converged", False),
            )
            for r in rows
        ]
        comparison.update(
            status="complete",
            convention="nonresidual minus residual",
            per_seed=paired,
            summary={
                k: dict(
                    mean=float(np.mean([r["differences"][k] for r in paired])),
                    sample_sd=float(
                        np.std([r["differences"][k] for r in paired], ddof=1)
                    ),
                )
                for k in metrics
            },
        )
    (ROOT / "results/first_block_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
print(json.dumps(rows, indent=2))
