"""Exercise all six model/training combinations on one 40-GB A100."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
records = []
for task in manifest["smoke_tasks"]:
    subprocess.run(
        [sys.executable, "-B", "-u", root / "run.py", "smoke", str(task["index"])],
        check=True,
    )
    directory = Path(task["checkpoint"]).parent
    assert json.loads((directory / "status.json").read_text())["state"] == "complete"
    record = dict(name=task["name"])
    for stage in ["train", "evaluate"]:
        memory = json.loads((directory / f"{stage}_memory.json").read_text())
        assert "A100" in memory["gpu"] and memory["total_memory_bytes"] < 50 * 2**30
        record[stage] = memory
    records.append(record)
(root / "smoke_passed.json").write_text(
    json.dumps(dict(state="passed", checks=records, spectral_checks=False), indent=2)
    + "\n"
)
print(
    "PASS: all six linear G=1 cases train, reload, and evaluate on a 40-GB A100.",
    flush=True,
)
