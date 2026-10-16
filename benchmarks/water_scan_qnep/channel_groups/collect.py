"""Collect the one-seed G=1 versus G=4 validation comparison."""

import csv
import json
from pathlib import Path

from common import digest, dump, load, metric_row, now


def main():
    root = Path(__file__).resolve().parent
    manifest = load(root / "manifest.json")
    ref, task = manifest["references"][0], manifest["tasks"][0]
    rows = []
    for groups, checkpoint, result_file in (
        (4, ref["checkpoint"], Path(ref["result"])),
        (1, task["checkpoint"], Path(task["checkpoint"]).parent / "results.json"),
    ):
        if not result_file.exists():
            continue
        result = load(result_file)
        if groups == 1:
            assert result["completed_steps"] == task["expected_steps"]
            assert result["test_evaluated"] is False
        selected = result["selections"]["best_loss"]
        assert digest(checkpoint) == selected["checkpoint_sha256"]
        assert selected["energy_shift_applied"] is False
        rows.append(
            dict(
                spectral_groups=groups,
                seed=29,
                steps=120000,
                selected_step=selected["selected_step"],
                validation_objective=selected["validation_objective"],
                **metric_row(selected["metrics"]),
                checkpoint=checkpoint,
            )
        )
    output = dict(
        collected=now(),
        state="complete" if len(rows) == 2 else "pending",
        scope="single-seed validation comparison; no statistical significance claim",
        rows=rows,
        test_evaluated=False,
    )
    if len(rows) == 2:
        control, trial = rows
        output["groups1_minus_groups4"] = {
            key: dict(
                absolute=trial[key] - control[key],
                percent=100 * (trial[key] / control[key] - 1),
            )
            for key in ("energy_raw_meV_atom", "force_meV_A", "validation_objective")
        }
    dump(root / "analysis" / "comparison.json", output)
    with (root / "analysis" / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
