"""Prepare the user-approved full-z budget revision without editing parent campaigns."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    scripts = Path(__file__).resolve().parent
    base = Path.home() / "mace_fno_runs/_campaigns"
    root = base / "au2_mgo_full_z_finalize_20261016"
    root.mkdir(exist_ok=False)
    for folder in ("logs", "results", "inputs", "preflight"):
        (root / folder).mkdir()
    for name in ("run.py", "prepare.py", "job.slurm"):
        shutil.copy2(scripts / name, root / name)
    shutil.copy2(
        scripts.parent / "full_z_mace2/resume_run.py", root / "resume_helpers.py"
    )
    shutil.copy2(scripts.parent / "full_z_tables/run.py", root / "table_helpers.py")
    wetting = base / "au2_mgo_full_z_frozen_wetting_completed_20261015"
    wet_manifest = json.loads((wetting / "manifest.json").read_text())
    for name in (
        "evaluate_adsorption_switch.py",
        "inputs/1-undoped.xyz",
        "inputs/3-undoped.xyz",
        "inputs/1-doped.xyz",
        "inputs/3-doped.xyz",
    ):
        assert digest(wetting / name) == wet_manifest["checksums"][name]
        shutil.copy2(wetting / name, root / name)
    tasks = []
    parents = [
        base / "au2_mgo_full_zmix_nl1_5seed_20261006T070300Z",
        base / "au2_mgo_full_zmix_nl2_5seed_20261013",
    ]
    mace1_jobs = ["16234048", "16234049", "16190334", "16206375", "16206376"]
    for depth, parent in enumerate(parents, 1):
        manifest = json.loads((parent / "manifest.json").read_text())
        for original in manifest["tasks"]:
            index = original["index"]
            if depth == 1 and index < 5:
                continue
            mode = (
                "continue" if depth == 2 else ("evaluate" if index < 7 else "finalize")
            )
            tasks.append(
                dict(
                    index=len(tasks),
                    name=original["name"],
                    parent=str(parent),
                    parent_index=index,
                    parent_manifest_sha256=digest(parent / "manifest.json"),
                    depth=depth,
                    seed=original["seed"],
                    regime=original["opt_level"],
                    mode=mode,
                    target_steps=100000
                    if depth == 2
                    else (240000 if mode == "evaluate" else None),
                    previous_job=f"16208990_{index}"
                    if depth == 2
                    else mace1_jobs[index - 5],
                    reason="User-approved 100000-step cap after validation plateau"
                    if depth == 2
                    else (
                        "Completed fit: finish wetting evaluation"
                        if mode == "evaluate"
                        else "User-approved finalization at latest committed step after validation plateau"
                    ),
                )
            )
    content = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        original_budget_steps=240000,
        selection="Best validation objective; held-out test never used for selection",
        protocol="Preserve native optimizer/scheduler/RNG; change only total step budget. MACE1 finalization takes zero additional optimizer steps.",
        wetting_parent=str(wetting),
        tasks=tasks,
        obsolete_pending_jobs=["16234050", "16206377", "16206378"]
        + [str(i) for i in range(16208991, 16209011)],
        stopping_jobs=["16190334", "16206375", "16206376"],
    )
    content["checksums"] = {
        str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()
    }
    (root / "manifest.json").write_text(json.dumps(content, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
