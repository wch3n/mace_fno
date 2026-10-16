#!/usr/bin/env python3
"""Audit manuscript-selected accuracy records; optionally link cross-campaign seed sets.

Does not run inference/training or change manuscript data. Checkpoint hashes are
checked when archived hashes are available; legacy logs are checked numerically.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs/source/data"
RUNS = Path("/gpfs/home/acad/ucl-modl/wchen/mace_fno_runs")
SEEDS = {17, 29, 41, 53, 67}
sys.path.insert(0, str(DATA.parent))


def read(path):
    return json.loads(Path(path).read_text())


def csvrows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def module(name):
    spec = importlib.util.spec_from_file_location(
        "audit_" + name, DATA / name / "collect_results.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(actual, saved):
    def key(r):
        return tuple(
            str(r[k])
            for k in [
                "system",
                "mace_layers",
                "opt_level",
                "depth",
                "regime",
                "model",
                "seed",
            ]
            if k in r
        )

    aa, ss = {key(r): r for r in actual}, {key(r): r for r in saved}
    assert len(aa) == len(actual) == len(saved) == len(ss) and aa.keys() == ss.keys()
    for k, row in aa.items():
        for field, value in row.items():
            if field not in ss[k]:
                continue
            observed = ss[k][field]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert math.isclose(
                    float(observed), value, rel_tol=1e-11, abs_tol=1e-11
                ), (k, field)
            else:
                assert str(value) == observed or value is None and observed == "", (
                    k,
                    field,
                )


class Audit:
    def __init__(self):
        self.records, self.checks, self.hashes = [], [], {}
        self.verified_expected_hashes = set()

    def sha(self, path, expected=None):
        path = Path(path).resolve()
        assert path.is_file(), path
        if str(path) not in self.hashes:
            with path.open("rb") as stream:
                self.hashes[str(path)] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        actual = self.hashes[str(path)]
        if expected:
            assert actual == expected, (
                "Live file differs from archive",
                str(path),
                actual,
                expected,
            )
            self.verified_expected_hashes.add(str(path))
        return actual

    def add(
        self,
        system,
        group,
        seed,
        checkpoint,
        metrics,
        source,
        config=None,
        expected=None,
        **extra,
    ):
        cp, metric = Path(checkpoint).resolve(), Path(metrics).resolve()
        assert cp.is_file() and metric.is_file(), (cp, metric)
        run = cp.parent.parent if cp.parent.name == "models" else cp.parent
        r = dict(
            system=system,
            group=group,
            seed=int(seed or 17),
            checkpoint=str(cp),
            run=str(run),
            metrics=str(metric),
            source_table=str(source),
            **extra,
        )
        self.sha(metric)
        if config:
            config = Path(config).resolve()
            r["config"] = str(config)
            self.sha(config)
        if expected:
            r["checkpoint_sha256"] = self.sha(cp, expected)
        self.records.append(r)

    def archived_tables(self):
        for name in [
            "water_clean",
            "llzo_converged",
            "au2_mgo_converged",
            "water_linear_g1",
            "physical_linear_g1",
            "au2_mgo_full_z_current",
            "au2_mgo_linear_wetting",
        ]:
            mod, snapshot = module(name), read(DATA / name / "sources.json")
            if name in ["water_clean", "llzo_converged", "au2_mgo_converged"]:
                kwargs = (
                    {"include_les2": snapshot["include_les2"]}
                    if name == "llzo_converged"
                    else {}
                )
                rows = mod.collect(mod.Sources(snapshot), **kwargs)
                tables = {
                    "test_per_seed.csv": [
                        r for r in rows if r.get("split", "test") == "test"
                    ]
                }
            elif name == "au2_mgo_full_z_current":
                rows, _, _ = mod.collect(snapshot)
                tables = {k + "_per_seed.csv": v for k, v in rows.items()}
            elif name == "au2_mgo_linear_wetting":
                tables = {"per_seed.csv": mod.summarize(snapshot)[0]}
            else:
                tables = {"per_seed.csv": mod.collect(snapshot)[0]}
            for file, rows in tables.items():
                compare(rows, csvrows(DATA / name / file))
                self.checks.append(
                    dict(
                        check="collector_matches_saved_rows",
                        dataset=name,
                        file=file,
                        rows=len(rows),
                    )
                )

    def physical(self):
        for name, system in [
            ("water_clean", "water_scan_g1"),
            ("llzo_converged", "llzo"),
            ("au2_mgo_converged", "au2_mgo"),
        ]:
            snapshot = read(DATA / name / "sources.json")
            root = Path(snapshot.get("campaign_root", snapshot.get("run_root", "")))
            hashes = snapshot.get("checkpoints", snapshot.get("verified_sha256", {}))
            source = DATA / name / "test_per_seed.csv"
            for row in csvrows(source):
                cp = root / row["checkpoint"]
                group = f"mace{row['mace_layers']}_{row['opt_level']}"
                if row["opt_level"] in ["frozen", "joint"]:
                    group += "_nonlinear" + (
                        "_channelwise_archive" if system == "au2_mgo" else ""
                    )
                elif (
                    system == "au2_mgo"
                    and row["mace_layers"] == "2"
                    and row["opt_level"] == "baseline"
                ):
                    group += "_scalar_archive"
                cfg = root / row["config_source"] if row["config_source"] else None
                for field in ["metrics_source", "config_source"]:
                    if row[field]:
                        path = root / row[field]
                        entry = snapshot["files"].get(str(path.relative_to(root)))
                        if entry:
                            self.sha(path, entry["sha256"])
                if system == "water_scan_g1" and row["opt_level"] in [
                    "frozen",
                    "joint",
                ]:
                    assert yaml.safe_load(cfg.read_text())["spectral_groups"] == 1
                self.add(
                    system,
                    group,
                    row["seed"],
                    cp,
                    root / row["metrics_source"],
                    source,
                    config=cfg,
                    expected=hashes.get(str(cp.relative_to(root))),
                    metrics_row=row,
                )
        for name in ["water_linear_g1", "physical_linear_g1"]:
            snap = read(DATA / name / "sources.json")
            for record in snap["records"]:
                task = record["task"]
                cp = Path(task["checkpoint"])
                self.sha(task["config"], record["config_sha256"])
                test_hash = record.get("test_result_sha256") or record.get(
                    "source_files", {}
                ).get(str(cp.parent / "test.json"))
                self.sha(cp.parent / "test.json", test_hash)
                assert read(cp.parent / "status.json")["state"] == "complete"
                self.add(
                    "water_scan_g1" if task["system"] == "water" else task["system"],
                    f"mace{task['depth']}_{task['opt_level']}_linear",
                    task["seed"],
                    cp,
                    cp.parent / "test.json",
                    DATA / name / "per_seed.csv",
                    config=task["config"],
                    expected=record["checkpoint_sha256"],
                )
        base = read(DATA / "physical_linear_g1/matched_baseline.json")
        self.add(
            "au2_mgo",
            "mace2_baseline_matched",
            17,
            base["model"],
            base["metric_source"],
            DATA / "physical_linear_g1/matched_baseline.json",
            expected=base["model_sha256"],
        )

    def log_record(self, system, group, row, log, source):
        log = Path(log)
        cfg = log.parent / "config.yaml"
        config = yaml.safe_load(cfg.read_text())
        cp = log.parent / config["checkpoint"]
        matches = re.findall(
            r"^selected held-out test\s+\|.*?E_RMSE=\s*([-\d.e+]+).*?E_ME=\s*([-\d.e+]+).*?F_RMSE=\s*([-\d.e+]+)",
            log.read_text(),
            re.M,
        )
        assert matches, log
        for value, keys in zip(
            matches[-1],
            [
                ("e_rmse", "energy_rmse_meV_atom"),
                ("e_me", "energy_me_meV_atom"),
                ("f_rmse", "force_rmse_meV_A"),
            ],
        ):
            target = next(row[k] for k in keys if k in row)
            assert math.isclose(float(value), float(target), abs_tol=1e-10), (
                log,
                keys,
                value,
                target,
            )
        self.add(
            system, group, row["seed"], cp, log, source, config=cfg, metrics_row=row
        )

    def synthetic(self):
        provenance = read(DATA / "synthetic_les_nonresidual/provenance.json")
        for path, expected in provenance["source_sha256"].items():
            self.sha(path, expected)
        les = {
            (r["dataset"], r["seed"]): r
            for r in read(DATA / "synthetic_les_nonresidual/evaluations.json")
        }
        root = RUNS / "yukawa_L32_kappa_production"
        local = read(root / "analysis/mace2_metrics.json")

        def les_record(system, dataset, row, source):
            item = les[dataset, int(row["seed"])]
            model = item["results"]["nonresidual"]
            assert model["first_block"] == "RealAgnosticInteractionBlock"
            assert item["training_status"]["state"] == "complete"
            cfg = yaml.safe_load(item["configuration"])
            assert cfg["max_num_epochs"] == 1200
            for src, fields in [
                ("energy_rmse_meV_atom", ["e_rmse", "energy_rmse_meV_atom"]),
                ("force_rmse_meV_A", ["f_rmse", "force_rmse_meV_A"]),
            ]:
                assert math.isclose(
                    model["splits"]["test"][src],
                    float(next(row[k] for k in fields if k in row)),
                    rel_tol=1e-11,
                )
            cp = Path(model["model"])
            self.add(
                system,
                "mace1_les_nonresidual",
                row["seed"],
                cp,
                cp.parent.parent / "evaluation/comparison.json",
                source,
                config=Path(provenance["campaign"])
                / "configs"
                / (cfg["name"] + ".yaml"),
                expected=model["model_sha256"],
                epoch_budget=1200,
            )

        for file in [
            "synthetic_accuracy/per_seed.csv",
            "synthetic_accuracy/linear_fno/linear_fno_per_seed.csv",
        ]:
            source = DATA / file
            rows = [r for r in csvrows(source) if r["primary"] == "True"]
            assert len(rows) == (
                51 if file == "synthetic_accuracy/per_seed.csv" else 30
            )
            for row in rows:
                k = row["kappa"].replace(".", "p")
                system = "homogeneous_yukawa/kappa_" + k
                if row["model"] in ["maceles", "mace1_les"]:
                    les_record(system, "kappa" + k, row, source)
                elif row["model"] in ["mace1", "mace2"]:
                    m = next(
                        m
                        for m in local
                        if (m["kappa"], m["model"], m["stage"])
                        == (k, row["model"], row["checkpoint"])
                    )
                    for f, target in [
                        ("e_rmse", "energy_rmse_meV_atom"),
                        ("f_rmse", "force_rmse_meV_A"),
                    ]:
                        assert math.isclose(m[f], float(row[target]), abs_tol=5.1e-5)
                    self.add(
                        system,
                        row["model"] + "_baseline",
                        17,
                        root / m["model_file"],
                        root / "analysis/mace2_metrics.json",
                        source,
                        expected=m["model_sha256"],
                        selected_stage=m["stage"],
                    )
                else:
                    group = row["model"] + (
                        "" if "linear" in row["model"] else "_nonlinear"
                    )
                    self.log_record(system, group, row, root / row["source"], source)
        root = RUNS / "hetscreen_v2"
        source = DATA / "inhomogeneous_accuracy/final_test_accuracy.csv"
        local = read(root / "final_eval_20261006/test_metrics_mace.json")
        sog = read(root / "final_eval_20261006/test_metrics_cace.json")
        for row in csvrows(source):
            if row["checkpoint"] == "energy_selected":
                continue
            model, seed = row["model"], int(row["seed"])
            if model == "MACE1-LES":
                les_record("heterogeneous_screening", "hetscreen", row, source)
            elif model in ["MACE1", "MACE2"]:
                m = next(m for m in local if m["model"] == model.lower())
                for f in ["e_rmse", "e_me", "f_rmse"]:
                    assert math.isclose(m[f], float(row[f]), rel_tol=1e-11)
                self.add(
                    "heterogeneous_screening",
                    model.lower() + "_baseline",
                    seed,
                    root / m["model_file"],
                    root / "final_eval_20261006/test_metrics_mace.json",
                    source,
                    expected=m["model_sha256"],
                )
            elif model == "CACE-SOG":
                m = sog[f"cace_sog_seed{seed}"]
                for f, target in [
                    ("energy_rmse_meV_atom", "e_rmse"),
                    ("force_rmse_meV_A", "f_rmse"),
                ]:
                    assert math.isclose(m[f], float(row[target]), rel_tol=1e-11)
                run = root / f"cace_sog_e1000_5seed_20261005/runs/sog_seed{seed}"
                assert (
                    "early_stopped" in read(run / "status.json")["status"]
                    or read(run / "status.json")["status"] == "complete"
                )
                self.add(
                    "heterogeneous_screening",
                    "cace_sog",
                    seed,
                    run / "best.pt",
                    root / "final_eval_20261006/test_metrics_cace.json",
                    source,
                    config=run / "provenance.json",
                )
            else:
                regime, architecture, _ = model.split()
                folder = (
                    "fno_"
                    + ("joint_" if regime == "joint" else "")
                    + architecture
                    + ("_s200k_" if regime == "joint" else "_s100k_")
                    + f"seed{seed}"
                )
                self.log_record(
                    "heterogeneous_screening",
                    f"fno_{regime}_{architecture}",
                    row,
                    root / folder / "train.log",
                    source,
                )
        source = DATA / "inhomogeneous_accuracy/cace_local_test.json"
        for r in read(source)["runs"]:
            assert r["training_status"]["status"] == "early_stopped_plateau"
            cp = Path(r["checkpoint_path"])
            self.add(
                "heterogeneous_screening",
                "cace_local",
                r["seed"],
                cp,
                cp.parent / "test_metrics.json",
                source,
                config=cp.parent / "provenance.json",
                expected=r["checkpoint_sha256"],
            )

    def coverage(self):
        result = []
        for system, group in sorted({(r["system"], r["group"]) for r in self.records}):
            rows = [
                r for r in self.records if (r["system"], r["group"]) == (system, group)
            ]
            seeds = [r["seed"] for r in rows]
            assert len(set(seeds)) == len(seeds)
            assert set(seeds) == ({17} if "baseline" in group else SEEDS), (
                system,
                group,
                seeds,
            )
            result.append(
                dict(system=system, group=group, n=len(rows), seeds=sorted(seeds))
            )
        return result


def link(path, target):
    target = Path(target)
    assert target.exists(), target
    if path.is_symlink():
        assert path.resolve() == target.resolve(), path
    else:
        assert not path.exists(), path
        path.symlink_to(
            os.path.relpath(target, path.parent), target_is_directory=target.is_dir()
        )


def publish(audit, destination, report):
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manuscript benchmark seed collections",
        "",
        "Selected accuracy records, grouped across campaigns. Each seed link opens its original run; "
        "`checkpoints/`, `configs/` and `test_results/` identify the exact selected files. "
        "Local MACE baselines have one checkpoint; other groups have seeds 17, 29, 41, 53 and 67.",
        "",
        "Au2/MgO nonlinear entries here are the archived channel-wise runs. "
        "Dense full-z accuracy/wetting snapshots and ongoing jobs are separate; see the audit report. "
        "The old scalar-only MACE2 baseline is explicitly marked `scalar_archive`.",
        "",
        "| System | Model | Records |",
        "|---|---|---:|",
    ]
    for group in report["coverage"]:
        path = destination / group["system"] / group["group"]
        path.mkdir(parents=True, exist_ok=True)
        selected = [
            r
            for r in audit.records
            if (r["system"], r["group"]) == (group["system"], group["group"])
        ]
        for r in selected:
            link(path / f"seed{r['seed']}", r["run"])
            for key, folder in [
                ("checkpoint", "checkpoints"),
                ("config", "configs"),
                ("metrics", "test_results"),
            ]:
                if key in r:
                    (path / folder).mkdir(exist_ok=True)
                    link(
                        path / folder / (f"seed{r['seed']}" + Path(r[key]).suffix),
                        r[key],
                    )
        (path / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
        (path / "README.md").write_text(
            f"# {group['system']}: {group['group']}\n\n"
            + "\n".join(
                f"- [Seed {r['seed']}](seed{r['seed']}): [selected checkpoint](checkpoints/seed{r['seed']}{Path(r['checkpoint']).suffix})"
                for r in selected
            )
            + "\n\nExact source tables, configurations and metrics are recorded in `manifest.json`.\n"
        )
        relative = path.relative_to(destination)
        lines.append(
            f"| {group['system']} | [{group['group']}]({relative}/README.md) | {group['n']} |"
        )
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for path in destination.rglob("*"):
        if path.is_symlink():
            assert path.exists(), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    args = parser.parse_args()
    audit = Audit()
    audit.archived_tables()
    audit.physical()
    audit.synthetic()
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        verification_status="ANALYZED",
        scope="Saved-result consistency and selected-file availability; no new inference or training.",
        checks=audit.checks,
        coverage=audit.coverage(),
        records=audit.records,
        hashes=audit.hashes,
        matched_archived_hashes=len(audit.verified_expected_hashes),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.collection:
        publish(audit, args.collection, report)
    print(
        json.dumps(
            dict(
                records=len(audit.records),
                groups=len(report["coverage"]),
                saved_rows_checked=sum(c["rows"] for c in audit.checks),
                matched_archived_hashes=report["matched_archived_hashes"],
                files_hashed=len(audit.hashes),
                output=str(args.output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
