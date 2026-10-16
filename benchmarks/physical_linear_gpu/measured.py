"""Run the original trainer/evaluator and record actual GPU memory peaks."""

import json
import runpy
import sys
from pathlib import Path

import torch
import yaml

root = Path(__file__).resolve().parent
mode = sys.argv.pop(1)
assert mode in ["train", "evaluate"]
args = sys.argv[1:]
cfg = yaml.safe_load(
    Path(args[args.index("--config") + 1] if mode == "train" else args[0]).read_text()
)
assert torch.cuda.is_available()
torch.cuda.reset_peak_memory_stats()
sys.argv = [str(root / f"{mode}.py"), *args]
runpy.run_path(str(root / f"{mode}.py"), run_name="__main__")
torch.cuda.synchronize()
record = dict(
    state="complete",
    stage=mode,
    gpu=torch.cuda.get_device_name(),
    total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    training_batch_size=cfg["batch_size"],
    accumulation_steps=cfg["accumulation_steps"],
    evaluation_batch_size=cfg["evaluation_batch_size"],
)
(Path(cfg["checkpoint"]).parent / f"{mode}_memory.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(json.dumps(record), flush=True)
