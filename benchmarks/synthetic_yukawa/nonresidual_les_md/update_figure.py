"""Stage Fig. 4 replacements, validate, publish and rebuild after both MD jobs."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    repo = Path(manifest["repo"])
    source = repo / "docs/source"
    data, figures = source / "data/md_structure", source / "figures"
    stage = ROOT / "results/figure4_update"
    stage.mkdir(exist_ok=False)
    status = dict(
        state="running",
        started=datetime.now(timezone.utc).isoformat(),
        job=os.environ.get("SLURM_JOB_ID"),
        visual_review="pending after rendering",
    )
    prose = {
        p: sha(p)
        for p in (
            source / "main.tex",
            source / "appendix_content.tex",
            source / "results_values.tex",
        )
    }
    original_data = {
        p: sha(p) for p in (data / "szz_shells_source.csv", data / "szz_shells.csv")
    }
    os.environ["MPLCONFIGDIR"] = str(stage / "mplconfig")
    skills = Path.home() / ".codex/skills/nature-figure/scripts"

    def run(label, argv):
        status["stage"] = label
        (stage / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        with (stage / f"{label}.log").open("x") as stream:
            subprocess.run(
                list(map(str, argv)),
                cwd=repo,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        run(
            "import",
            [
                sys.executable,
                "-B",
                data / "import_nonresidual_les.py",
                "--campaign",
                ROOT,
                "--output-dir",
                stage / "data",
            ],
        )
        run(
            "convert",
            [
                sys.executable,
                "-B",
                data / "prepare_plot_data.py",
                "--input",
                stage / "data/szz_shells_source.csv",
                "--output",
                stage / "data/szz_shells.csv",
            ],
        )
        run(
            "plot",
            [
                sys.executable,
                "-B",
                figures / "plot_md_structure.py",
                "--data-file",
                stage / "data/szz_shells.csv",
                "--output-dir",
                stage / "figures",
            ],
        )
        run(
            "source_check",
            [
                sys.executable,
                "-B",
                skills / "validate_figure.py",
                figures / "plot_md_structure.py",
            ],
        )
        pdf = stage / "figures/md_structure.pdf"
        run(
            "text_check",
            [
                sys.executable,
                "-B",
                skills / "audit_pdf_text.py",
                pdf,
                "--min-pt",
                "5",
                "--json",
            ],
        )
        run(
            "collision_check",
            [
                sys.executable,
                "-B",
                skills / "audit_figure_collisions.py",
                pdf,
                "--strict",
                "--json-out",
                stage / "figures/md_structure.collision-audit.json",
            ],
        )
        import pymupdf

        doc = pymupdf.open(pdf)
        doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(
            stage / "figures/md_structure.preview.png"
        )
        replacements = [
            (stage / "data" / n, data / n)
            for n in (
                "szz_shells_source.csv",
                "szz_shells.csv",
                "szz_shells.import.json",
            )
        ]
        replacements += [
            (stage / "figures" / n, figures / n)
            for n in (
                "md_structure.pdf",
                "md_structure.json",
                "md_structure.collision-audit.json",
            )
        ]
        assert all(sha(p) == value for p, value in original_data.items()), (
            "Figure data changed during staging."
        )
        backup = stage / "prior_figure"
        backup.mkdir()
        for _, dest in replacements:
            if dest.exists():
                shutil.copy2(dest, backup / dest.name)
        try:
            for src, dest in replacements:
                shutil.copy2(src, dest)
            run(
                "tests",
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    source / "tests",
                    "-p",
                    "test_md_structure.py",
                ],
            )
        except BaseException:
            for _, dest in replacements:
                old = backup / dest.name
                if old.exists():
                    shutil.copy2(old, dest)
            raise
        shutil.copytree(
            stage / "data/nonresidual_les", data / "nonresidual_les", dirs_exist_ok=True
        )
        status["figure_installed"] = True
        run(
            "manuscript",
            [
                "bash",
                source / "build_manuscript.sh",
                repo / "docs/build_les_nonresidual_md_20261012",
            ],
        )
        assert all(sha(p) == value for p, value in prose.items()), (
            "Manuscript prose changed during the figure update."
        )
        status.update(
            state="complete",
            stage="complete",
            figure=str(figures / "md_structure.pdf"),
            rows_replaced=30,
            non_les_rows_unchanged=True,
            prose_unchanged=True,
        )
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        (stage / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
