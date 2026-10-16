"""Collect paired-seed changes without modifying manuscript result tables."""

import csv
import json
from statistics import mean, stdev

from verify import ROOT, dump


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rows = []
    for task in manifest["tasks"]:
        directory = ROOT / task["run_dir"]
        assert (
            json.loads((directory / "status.json").read_text())["state"] == "complete"
        )
        comparison = json.loads((directory / "evaluation/comparison.json").read_text())
        assert comparison["residual_metrics_reproduced"]
        row = dict(dataset=task["dataset"], seed=task["seed"])
        for metric in ("energy_rmse_meV_atom", "force_rmse_meV_A"):
            for block in ("residual", "nonresidual"):
                row[f"{block}_{metric}"] = comparison["results"][block]["splits"][
                    "test"
                ][metric]
            for change in ("absolute", "percent"):
                row[f"change_{change}_{metric}"] = comparison["differences"][metric][
                    change
                ]
        rows.append(row)
    assert len(rows) == 20 and len({(r["dataset"], r["seed"]) for r in rows}) == 20
    output = ROOT / "comparison"
    output.mkdir(exist_ok=True)
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        selected = [r for r in rows if r["dataset"] == dataset]
        assert {r["seed"] for r in selected} == {17, 29, 41, 53, 67}
        summary[dataset] = {
            field: dict(
                mean=mean(r[field] for r in selected),
                sample_sd=stdev(r[field] for r in selected),
                n_seeds=5,
            )
            for field in rows[0]
            if field not in ("dataset", "seed")
        }
    dump(
        output / "summary.json",
        dict(
            datasets=summary,
            convention="Change = nonresidual minus residual; negative means improved RMSE.",
            scope="Matched seeds and protocol; backend parity checked, training trajectories may differ.",
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
