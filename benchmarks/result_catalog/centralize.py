#!/usr/bin/env python3
"""Build a browsable, linked collection of completed MACE-FNO calculations.

Original runs are read only. Each refresh creates a new catalog snapshot and
atomically changes the central current link. Completion is not convergence.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_RUNS = Path("/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs")
SKIP = {
    "_archive",
    "_scratch",
    "releases",
    "results_catalog",
    "source",
    "src",
    "mace_source",
    "les_source",
    "cache",
    "caches",
    "data",
    "checkpoints",
    "models",
    "references",
    "provenance",
    "archive",
    "snapshots",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "site-packages",
    "audits",  # Directory name in archived campaigns.
    "checks",
    "diagnostics",
}
TERMINAL = {
    "complete",
    "completed",
    "done",
    "success",
    "succeeded",
    "early_stopped_plateau",
    "epoch_cap_reached",
}
COLUMNS = [
    "system",
    "category",
    "model_family",
    "campaign",
    "run",
    "seed",
    "mace_layers",
    "regime",
    "architecture",
    "spectral_groups",
    "z_mixing",
    "spatial_scheme",
    "interaction_first",
    "dataset",
    "recorded_state",
    "completion_basis",
    "directory",
    "source",
    "config",
    "evidence",
]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unspecified"


def read_tail(path, limit=32768):
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - limit))
        return f.read().decode(errors="replace")


def state_of(data):
    return str(data.get("state", data.get("status", ""))).lower()


def live_jobs():
    out = subprocess.check_output(
        ["squeue", "--me", "--noheader", "--format=%i|%T|%j|%Z|%o"], text=True
    )
    return [
        dict(zip(["id", "state", "name", "workdir", "command"], line.split("|", 4)))
        for line in out.splitlines()
        if line.strip()
    ]


def campaign_for(path, root):
    rel = path.relative_to(root)
    if rel.parts[0] == "_campaigns":
        return root / rel.parts[0] / rel.parts[1]
    # Legacy layouts have either a campaign directory or a system root.
    if len(rel.parts) > 1 and re.search(r"20\d{6}", rel.parts[1]):
        return root / rel.parts[0] / rel.parts[1]
    return root / rel.parts[0]


def task_dict(data):
    task = data.get("task", {})
    return task if isinstance(task, dict) else {}


def resolve_existing(value, run, campaign):
    if not isinstance(value, str) or not value or "\n" in value:
        return None
    p = Path(value).expanduser()
    if p.is_absolute():
        return p if p.exists() else None
    for base in [run, campaign, *run.parents]:
        candidate = base / p
        if candidate.exists():
            return candidate
    return None


def config_for(run, campaign, data):
    task = task_dict(data)
    candidates = []
    for obj in (task, data):
        for key in ("config", "config_source"):
            p = resolve_existing(obj.get(key), run, campaign)
            if p:
                candidates.append(p)
    candidates.extend(
        [
            run / "config.yaml",
            campaign / "configs" / f"{run.name}.yaml",
            run / "model.config.yaml",
        ]
    )
    if run.parent.name == "mace":
        candidates.append(campaign / "configs" / f"mace_{run.name}.yaml")
    candidates.extend(sorted(run.glob("*.config.yaml")))
    for p in candidates:
        if p.is_file() and "energy.config" not in p.name:
            d = yaml.safe_load(p.read_text())
            if isinstance(d, dict):
                return p, d
    return None, {}


def system_for(run, task, cfg):
    texts = [
        str(task.get("system", "")),
        run.name,
        str(cfg.get("train_file", "")),
        str(cfg.get("test_file", "")),
        str(run),
    ]
    for text in texts:
        low = text.lower()
        if "au2_mgo" in low or "au_mgo" in low or "inference_au_" in low:
            return "au2_mgo"
        if "llzo" in low:
            return "llzo"
        if "water_rpbe" in low:
            return "water_rpbe_d3"
        if "water" in low:
            return "liquid_water"
        if any(
            x in low
            for x in (
                "hetscreen",
                "heterogeneous",
                "extended_les_response",
                "interim_probe",
                "nonres_hetero",
            )
        ):
            return "heterogeneous_screening"
        if any(
            x in low
            for x in ("yukawa", "kappa", "nonres_k0", "synthetic_les_nonresidual_md")
        ):
            return "homogeneous_yukawa"
        if "ti2co2" in low:
            return "ti2co2"
    return "cross_system"


def describe(run, campaign, data, cfg):
    task = task_dict(data)
    text = str(run).lower()
    system = system_for(run, task, cfg)
    fno = "fno" in run.name.lower() or any(
        k in cfg for k in ("fno_layers", "mace_training", "fno_hidden_channels")
    )
    architecture = cfg.get("architecture", task.get("architecture", ""))
    if fno and not architecture:
        architecture = (
            "linear"
            if "linear" in run.name and "nonlinear" not in run.name
            else "nonlinear"
        )
    regime = cfg.get("mace_training", task.get("regime", task.get("opt_level", "")))
    if not regime:
        regime = next((x for x in ("frozen", "joint") if x in run.name), "")
    layers = task.get(
        "mace_layers",
        task.get("depth", cfg.get("num_interactions", data.get("mace_layers", ""))),
    )
    if not layers:
        match = re.search(
            r"(?:mace_?nl|mace|_nl)([12])(?:_|/|\.)",
            run.name + " " + str(cfg.get("mace_model", "")),
        )
        layers = int(match.group(1)) if match else ""
    seed = task.get("seed", cfg.get("seed", data.get("seed", "")))
    if seed == "":
        match = re.search(r"seed_?(\d+)", run.name)
        seed = int(match.group(1)) if match else ""
    if "extended_les" in text:
        family = "extended_les"
    elif "cace" in text:
        family = "cace_sog" if "sog" in text else "cace_local"
    elif fno:
        family = "_".join(
            str(x)
            for x in (
                f"mace{layers}" if layers else "mace",
                "fno",
                architecture,
                regime,
            )
            if x
        )
    elif "les" in text or "les" in str(cfg.get("model", "")).lower():
        family = f"mace{layers}_les" if layers else "mace_les"
    elif (
        cfg.get("model") == "MACE"
        or run.parent.name == "mace"
        or re.search(r"mace[12]|mace_nl[12]", run.name)
    ):
        family = f"mace{layers}" if layers else "mace"
    else:
        family = "other"
    if (
        "inference" in campaign.name
        or "mixing_test_cost" in campaign.name
        or "gpu_comparison" in campaign.name
        or run.name == "measurements"
    ):
        category = "runtime_memory"
    elif "wetting" in text:
        category = "wetting"
    elif "probe" in text:
        category = "remote_medium_probe"
    elif "md" in campaign.name.split("_") or "_md" in text:
        category = "molecular_dynamics"
    elif any(
        x in text
        for x in (
            "ablation",
            "hyperparameter",
            "fourier_depth",
            "fourier_layers",
            "depth_confirmation",
            "convergence_matched",
            "interlacing",
            "loss_scales",
        )
    ):
        category = "convergence_and_ablations"
    elif any(x in text for x in ("spectral", "lowk", "parity")):
        category = "diagnostics"
    elif (
        run.parent.name in {"results", "raw", "reports"}
        or "tables_" in text
        or "heldout" in text
        or "energy_test" in run.name
    ):
        category = "evaluations"
    else:
        category = "training"
    dataset = ""
    if system == "liquid_water":
        data_path = str(cfg.get("train_file", cfg.get("train_cache", "")))
        dataset = (
            "clean"
            if any(x in text + data_path for x in ("clean", "groups1", "water_linear"))
            else "original"
            if "/water_scan/data/" in data_path or "/water_scan/cache/" in data_path
            else "see_config"
        )
    elif system == "homogeneous_yukawa":
        match = re.search(r"(?:kappa_?|_k)(0p\d+)", text)
        dataset = "kappa_" + match.group(1) if match else "see_config"
    return dict(
        system=system,
        category=category,
        model_family=family,
        seed=seed,
        mace_layers=layers,
        regime=regime,
        architecture=architecture,
        spectral_groups=cfg.get("spectral_groups", task.get("spectral_groups", "")),
        z_mixing=cfg.get("z_mixing", ""),
        spatial_scheme=cfg.get("spatial_scheme", ""),
        interaction_first=cfg.get("interaction_first", ""),
        dataset=dataset,
    )


def discover(root, output, jobs):
    candidates, states, excluded, errors = {}, {}, [], []
    folders = []
    for base, dirs, files in os.walk(root, followlinks=False):
        run = Path(base)
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in SKIP
            and not d.startswith("resume_")
            and run / d != output
            and not (run / d).is_symlink()
        )
        folders.append((run, set(files)))
        if "status.json" in files:
            try:
                states[run] = load(run / "status.json")
            except (OSError, ValueError) as exc:
                errors.append(dict(path=str(run / "status.json"), error=str(exc)))
    active_ids = {j["id"] for j in jobs}
    for run, files in folders:
        relative = run.relative_to(root)
        if any(
            any(
                tag in part.lower()
                for tag in ("smoke", "pilot", "preflight", "preparation_incomplete")
            )
            or part in {"prepare", "validation"}
            for part in relative.parts
        ):
            continue
        data = states.get(run, {})
        recorded_state = state_of(data)
        if recorded_state and recorded_state not in TERMINAL:
            excluded.append(
                dict(
                    source=str(run),
                    reason="nonterminal_or_unsuccessful_status",
                    recorded_state=recorded_state,
                )
            )
            continue
        if any(
            state_of(states[parent]) not in TERMINAL
            for parent in run.parents
            if parent in states and state_of(states[parent])
        ):
            continue
        evidence = None
        basis = ""
        if recorded_state in TERMINAL:
            evidence, basis = run / "status.json", "terminal_status"
        elif "complete.json" in files:
            d = load(run / "complete.json")
            if isinstance(d, dict) and (
                state_of(d) in TERMINAL
                or ("elapsed_seconds" in d and d.get("pilot") is False)
            ):
                evidence, basis, data = run / "complete.json", "completion_marker", d
                recorded_state = state_of(d) or "complete_marker"
        elif "gpu_after.txt" in files and (run / "timing/summary.json").exists():
            evidence, basis = run / "gpu_after.txt", "gpu_benchmark_completion"
            recorded_state = "benchmark_complete"
        if evidence is None:
            # Old synthetic and physical jobs record finalization only in logs.
            for name in ("training.log", "train.log"):
                if name not in files:
                    continue
                tail = read_tail(run / name)
                fno_done = re.search(r"timings:.*final-evaluation=.*total=", tail)
                mace_done = re.search(r"INFO:\s+Done\s*(?:\n|$)", tail)
                if (
                    not (fno_done or mace_done)
                    or "Traceback (most recent call last)" in tail
                ):
                    continue
                model_exists = (
                    any(p.is_file() for p in run.glob("*.pt"))
                    or (run / "models").is_dir()
                )
                if model_exists:
                    evidence, basis, recorded_state = (
                        run / name,
                        "final_training_log",
                        "final_log_complete",
                    )
                    break
        if evidence is None:
            continue
        job_id = str(data.get("job_id", data.get("slurm_job_id", data.get("job", ""))))
        array_task = data.get("array_task", data.get("array_task_id"))
        full_job_id = (
            f"{job_id}_{array_task}"
            if array_task is not None and "_" not in job_id
            else job_id
        )
        if full_job_id in active_ids or job_id in active_ids:
            excluded.append(
                dict(
                    source=str(run),
                    reason="job_in_live_queue",
                    recorded_state=recorded_state,
                )
            )
            continue
        # Do not expose an aggregate directory that also contains unfinished runs.
        if (run / "runs").is_dir() and any(
            child.is_relative_to(run / "runs") and state_of(d) not in TERMINAL
            for child, d in states.items()
            if state_of(d)
        ):
            excluded.append(
                dict(
                    source=str(run),
                    reason="contains_unfinished_runs",
                    recorded_state=recorded_state,
                )
            )
            continue
        campaign = campaign_for(run, root)
        cfg_path, cfg = config_for(run, campaign, data)
        meta = describe(run, campaign, data, cfg)
        if basis == "final_training_log" and cfg.get("steps"):
            progress = re.findall(r"step (\d+)/(\d+):", read_tail(evidence))
            early_stop = re.search(r"early stopping at step \d+", read_tail(evidence))
            if not early_stop and (
                not progress or int(progress[-1][0]) < int(cfg["steps"])
            ):
                excluded.append(
                    dict(
                        source=str(run),
                        reason="log_does_not_reach_configured_steps",
                        recorded_state=recorded_state,
                    )
                )
                continue
        within = run.relative_to(campaign)
        leaf = "__".join(within.parts) if within.parts else "summary"
        directory = (
            Path(meta["system"])
            / meta["category"]
            / meta["model_family"]
            / campaign.name
            / slug(leaf)
        )
        candidates[run] = dict(
            **meta,
            campaign=campaign.name,
            run=run.name,
            source=str(run),
            directory=str(directory),
            recorded_state=recorded_state,
            completion_basis=basis,
            evidence=str(evidence),
            evidence_sha256=sha(evidence),
            config=str(cfg_path) if cfg_path else "",
            config_sha256=sha(cfg_path) if cfg_path else "",
            campaign_path=str(campaign),
            completion_record=data,
            configured_checkpoint=str(
                resolve_existing(cfg.get("checkpoint"), run, campaign) or ""
            ),
        )
    # An aggregate and its finished children would otherwise duplicate the same files.
    for parent in list(candidates):
        if candidates[parent]["category"] != "training" and any(
            child != parent and child.is_relative_to(parent) for child in candidates
        ):
            excluded.append(
                dict(
                    source=str(parent),
                    reason="aggregate_represented_by_individual_results",
                    recorded_state=candidates[parent]["recorded_state"],
                )
            )
            del candidates[parent]
    return sorted(candidates.values(), key=lambda r: r["directory"]), excluded, errors


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def link(path, target):
    assert target.exists(), target
    path.symlink_to(
        os.path.relpath(target, path.parent), target_is_directory=target.is_dir()
    )


def build(root, output, records, excluded, errors, jobs):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=True)
    ownership = output / ".completed-results-catalog"
    if any(output.iterdir()) and not ownership.exists():
        raise RuntimeError(f"Refusing to use an existing unmanaged directory: {output}")
    ownership.touch(exist_ok=True)
    snapshot = output / "snapshots" / stamp
    snapshot.mkdir(parents=True)
    for rec in records:
        target = snapshot / rec["directory"]
        target.mkdir(parents=True)
        source = Path(rec["source"])
        link(target / "files", source)
        link(target / "completion_evidence", Path(rec["evidence"]))
        if rec["config"]:
            link(target / "config.yaml", Path(rec["config"]))
        if rec["configured_checkpoint"]:
            link(target / "configured_checkpoint", Path(rec["configured_checkpoint"]))
        campaign = Path(rec["campaign_path"])
        if (campaign / "manifest.json").is_file():
            link(target / "campaign_manifest.json", campaign / "manifest.json")
        write_json(target / "record.json", rec)
    counts = Counter(r["system"] for r in records)
    with (snapshot / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    manifest = dict(
        created_utc=stamp,
        root=str(root),
        records=records,
        excluded=excluded,
        errors=errors,
        live_jobs=jobs,
        counts=dict(counts),
        original_files_moved=0,
        original_files_modified=0,
        scope="Completed calculations, not a convergence certification or a selection of manuscript production models.",
    )
    write_json(snapshot / "manifest.json", manifest)
    write_json(snapshot / "excluded.json", excluded)
    with (snapshot / "excluded.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "reason", "recorded_state"])
        w.writeheader()
        w.writerows(excluded)
    for system in counts:
        rows = [r for r in records if r["system"] == system]
        lines = [
            f"# {system}: completed calculations",
            "",
            "Each result contains `files/`, a link to its original run, and a provenance `record.json`.",
            "Campaigns and parameter variants are kept separate. Finished does not imply converged.",
            "",
            "| Category | Model | Campaign / run | Seed |",
            "|---|---|---|---|",
        ]
        for r in rows:
            rel = Path(r["directory"]).relative_to(system)
            lines.append(
                f"| {r['category']} | {r['model_family']} | [{r['campaign']} / {r['run']}]({rel}) | {r['seed']} |"
            )
        (snapshot / system / "README.md").write_text("\n".join(lines) + "\n")
    lines = [
        "# Completed MACE-FNO results",
        "",
        f"Catalog snapshot: {stamp}.",
        "",
        f"{len(records)} completed calculation directories are organized below.",
        "",
        "| System | Finished calculation directories |",
        "|---|---:|",
    ]
    for system, count in sorted(counts.items()):
        lines.append(f"| [{system}]({system}/README.md) | {count} |")
    lines += [
        "",
        "## Finding results",
        "",
        "Browse `system/category/model/campaign/run/`. Open `files/` for the original checkpoints, logs and results. "
        "The adjacent `record.json`, configuration link and completion-evidence link document the selection.",
        "",
        "[index.csv](index.csv) is a filterable list. It records seeds, architecture, training regime, channel groups, "
        "height mixing, dataset labels and the LES first interaction block when explicitly configured. "
        "Blank fields mean unspecified, not a guessed default. `z_mixing` records the YAML value; "
        "the linear slab dispatch can implement full channel-height mixing even when the YAML says `global`.",
        "",
        "## Scope",
        "",
        "This collection uses directory and file links. Existing run paths, queued jobs and manuscript references remain valid. "
        "It uses negligible space compared with copying checkpoints or trajectories. It is not an independent backup.",
        "",
        "Completed training, evaluations, MD, probes and supporting analyses remain separate. "
        "A completed evaluation of an interim checkpoint is not a completed training run. "
        "Historical variants are retained under their campaign names; this is not a recommendation to pool their metrics.",
        "",
        "Smoke tests, pilots, source snapshots, caches, archives, scratch space and release-upload bundles are outside this collection. "
        "Explicit running/failed/cancelled states are excluded even when an old checkpoint exists. "
        "Other accepted evidence is a final training log, a documented completion marker, or the final GPU benchmark marker. "
        "Jobs found in the live queue are excluded. [excluded.csv](excluded.csv) lists recorded exclusions.",
        "",
        "Large model and trajectory files were not rehashed. Completion evidence and configurations were hashed; "
        "all generated links were checked. This inventory does not reevaluate accuracy.",
        "",
        "## Refresh",
        "",
        "From any directory:",
        "",
        "```bash",
        f"python3 -B {Path(__file__).resolve()} --runs-root {root} --output-dir {output}",
        "```",
        "",
        "Each refresh creates a new snapshot and switches `current` atomically. "
        "Earlier catalog snapshots are retained; original runs are never moved or edited.",
    ]
    (snapshot / "README.md").write_text("\n".join(lines) + "\n")
    # Validate everything before exposing the snapshot through current.
    broken = [str(p) for p in snapshot.rglob("*") if p.is_symlink() and not p.exists()]
    assert not broken, broken
    for rec in records:
        assert sha(Path(rec["evidence"])) == rec["evidence_sha256"], rec["source"]
        if rec["config"]:
            assert sha(Path(rec["config"])) == rec["config_sha256"], rec["config"]
    write_json(
        snapshot / "verification.json",
        dict(
            records=len(records),
            broken_links=broken,
            evidence_and_config_hashes_verified=True,
            original_files_moved=0,
            original_files_modified=0,
        ),
    )
    for name in [
        "README.md",
        "index.csv",
        "manifest.json",
        "excluded.json",
        "excluded.csv",
        "verification.json",
        *counts,
    ]:
        alias = output / name
        if not alias.is_symlink() and alias.exists():
            raise RuntimeError(f"Refusing to overwrite {alias}")
        if not alias.is_symlink():
            alias.symlink_to(Path("current") / name)
    pending = output / f".current-{stamp}"
    pending.symlink_to(Path("snapshots") / stamp, target_is_directory=True)
    os.replace(pending, output / "current")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--inventory-only",
        type=Path,
        help="Write a review inventory without building the linked collection.",
    )
    args = parser.parse_args()
    root = args.runs_root.resolve()
    output = (args.output_dir or root / "results").resolve()
    assert output != root and not root.is_relative_to(output)
    jobs = live_jobs()
    records, excluded, errors = discover(root, output, jobs)
    if errors:
        raise RuntimeError(f"Unreadable completion records: {errors}")
    if args.inventory_only:
        write_json(
            args.inventory_only,
            dict(records=records, excluded=excluded, errors=errors, live_jobs=jobs),
        )
        destination = args.inventory_only
    else:
        destination = build(root, output, records, excluded, errors, jobs)
    print(
        json.dumps(
            dict(
                destination=str(destination),
                count=len(records),
                systems=dict(Counter(r["system"] for r in records)),
                categories=dict(Counter(r["category"] for r in records)),
                excluded=len(excluded),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
