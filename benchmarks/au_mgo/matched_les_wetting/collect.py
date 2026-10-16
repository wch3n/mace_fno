"""Replace only the five MACE1-LES wetting records, preserving the old snapshot."""

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root, target = args.campaign.resolve(), args.data_dir.resolve()
    sys.path.insert(0, str(target))
    from collect_results import Sources, digest, write_csv
    from collect_wetting import collect, latex, summarize
    from result_macros import write_result_macros

    assert json.loads((root / "complete.json").read_text())["state"] == "complete"
    manifest = json.loads((root / "manifest.json").read_text())
    assert sorted(t["seed"] for t in manifest["tasks"]) == [17, 29, 41, 53, 67]
    old = json.loads((target / "wetting_sources.json").read_text())
    snapshot = copy.deepcopy(old)
    old_rows, _ = collect(Sources(copy.deepcopy(old)))
    run_root = Path(snapshot["run_root"])
    relative = str(root.relative_to(run_root))
    accuracy = json.loads((target / "sources.json").read_text())
    accuracy_rows = list(csv.DictReader((target / "test_per_seed.csv").open()))
    replaced = set()
    for task in manifest["tasks"]:
        name = task["name"]
        assert name == f"au2_mgo_MACE1_LES_seed{task['seed']}"
        (row,) = [
            r
            for r in accuracy_rows
            if r["mace_layers"] == "1"
            and r["opt_level"] == "les"
            and int(r["seed"]) == task["seed"]
        ]
        assert row["checkpoint"] == task["model"]
        model_key = str(Path(task["model"]).relative_to(run_root))
        assert (
            digest(Path(task["model"]))
            == task["model_sha256"]
            == accuracy["verified_sha256"][model_key]
        )
        snapshot["verified_sha256"][model_key] = task["model_sha256"]
        snapshot["accuracy_models"][name] = dict(
            checkpoint=model_key, sha256=task["model_sha256"]
        )
        snapshot.setdefault("wetting_updates", {})[name] = relative
        replaced.add((1, "les", task["seed"]))
    src = Sources(snapshot, run_root, refresh_prefix=relative)
    rows, endpoints = collect(src)
    assert [
        r
        for r in old_rows
        if (r["mace_layers"], r["opt_level"], r["seed"]) not in replaced
    ] == [
        r for r in rows if (r["mace_layers"], r["opt_level"], r["seed"]) not in replaced
    ]
    # Confirm offline reproduction before replacing any table source data.
    assert collect(Sources(copy.deepcopy(snapshot))) == (rows, endpoints)
    archive = (
        target
        / "archive"
        / (
            "wetting_"
            + hashlib.sha256(
                (target / "wetting_sources.json").read_bytes()
            ).hexdigest()[:12]
        )
    )
    archive.mkdir(parents=True, exist_ok=True)
    for name in (
        "wetting_sources.json",
        "wetting_per_seed.csv",
        "wetting_endpoints.csv",
        "wetting_summary.csv",
        "wetting_values.tex",
    ):
        if not (archive / name).exists():
            shutil.copy2(target / name, archive / name)
    summary = summarize(rows)
    write_csv(target / "wetting_per_seed.csv", rows)
    write_csv(target / "wetting_endpoints.csv", endpoints)
    write_csv(target / "wetting_summary.csv", summary)
    literature = json.loads((target / "wetting_reference.json").read_text())
    write_result_macros(target / "wetting_values.tex", latex(summary, literature))
    (target / "wetting_sources.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(
        json.dumps(
            next(
                r for r in summary if r["mace_layers"] == 1 and r["opt_level"] == "les"
            ),
            indent=2,
        )
    )
    print("Archived previous wetting results:", archive)
    print("Verified all 128 endpoints; replaced only the five MACE1-LES records.")


if __name__ == "__main__":
    main()
