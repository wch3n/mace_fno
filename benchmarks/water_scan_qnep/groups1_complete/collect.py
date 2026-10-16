"""Export complete G=1 accuracy and convergence results without editing the paper."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    rows, missing = [], []
    for task in manifest["tasks"]:
        run = Path(task["checkpoint"]).parent
        if (
            not (run / "status.json").exists()
            or json.loads((run / "status.json").read_text())["state"] != "complete"
        ):
            missing.append(task["name"])
            continue
        result = json.loads((run / "results.json").read_text())
        best = result["selections"]["best_loss"]
        assert result["completed_steps"] == task["expected_steps"]
        assert best["energy_shift_applied"] is False
        metrics = (
            result["primary_test"]["mace_fno"]
            if task["role"] == "accuracy"
            else best["metrics"]
        )
        rows.append(
            dict(
                name=task["name"],
                role=task["role"],
                mace_layers=task["mace_layers"],
                regime=task["regime"],
                variant=task["variant"],
                seed=task["seed"],
                spectral_groups=1,
                split="test" if task["role"] == "accuracy" else "validation",
                selected_step=best["selected_step"],
                steps=result["completed_steps"],
                energy_rmse_meV_atom=metrics["energy_rmse"] * 1000,
                force_rmse_meV_A=metrics["force_rmse"] * 1000,
                checkpoint=task["checkpoint"],
                checkpoint_sha256=best["checkpoint_sha256"],
            )
        )
    progress = dict(
        complete=len(rows), expected=len(manifest["tasks"]), missing=missing
    )
    (root / "analysis" / "progress.json").write_text(
        json.dumps(progress, indent=2) + "\n"
    )
    if missing:
        raise SystemExit(
            f"Incomplete campaign: {len(rows)}/{len(manifest['tasks'])}; no final export."
        )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["role"], row["mace_layers"], row["regime"], row["variant"])].append(
            row
        )
    summaries = []
    for (role, layer, regime, variant), group in groups.items():
        assert sorted(r["seed"] for r in group) == (
            [17, 29, 41, 53, 67] if role == "accuracy" else [29, 41, 53]
        )
        summary = dict(
            role=role,
            mace_layers=layer,
            regime=regime,
            variant=variant,
            n=len(group),
            spectral_groups=1,
        )
        for metric in ("energy_rmse_meV_atom", "force_rmse_meV_A"):
            summary[metric + "_mean"] = statistics.mean(r[metric] for r in group)
            summary[metric + "_sd"] = statistics.stdev(r[metric] for r in group)
        summaries.append(summary)
    write_csv(root / "analysis" / "per_seed.csv", rows)
    write_csv(root / "analysis" / "summary.csv", summaries)
    (root / "analysis" / "results.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "spectral_groups": 1,
                "rows": rows,
                "summary": summaries,
                "manuscript_updated": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Exported 20 accuracy models and 15 convergence variants; paper update awaits review."
    )


if __name__ == "__main__":
    main()
