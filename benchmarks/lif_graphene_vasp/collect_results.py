#!/usr/bin/env python3
"""Collect completed VASP energies and symmetric LiF-pair forces into CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from validate_vasprun import VasprunValidationError, load_successful_vasprun


ROOT = Path(__file__).resolve().parent
FUNCTIONALS = ("pbe", "pbe0", "pbe0_rvv10")


def main() -> None:
    with (ROOT / "cases.tsv").open(encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle, delimiter="\t"))

    raw = []
    for case in cases:
        for functional in FUNCTIONALS:
            directory = ROOT / "calculations" / case["case"] / functional
            vasprun_path = directory / "vasprun.xml"
            row = {**case, "functional": functional, "status": "missing"}
            if vasprun_path.is_file():
                try:
                    run = load_successful_vasprun(
                        vasprun_path,
                        expected_atoms=36,
                        require_static=True,
                    )
                    forces = np.asarray(run.ionic_steps[-1]["forces"])
                    top_fz = float(forces[[32, 34], 2].sum())
                    bottom_fz = float(forces[[33, 35], 2].sum())
                    row.update(
                        {
                            "status": "complete",
                            "energy_eV": float(run.final_energy),
                            "top_pair_Fz_eV_A": top_fz,
                            "bottom_pair_Fz_eV_A": bottom_fz,
                            "antisymmetric_pair_F_eV_A": 0.5
                            * (top_fz - bottom_fz),
                            "pair_force_sum_error_eV_A": top_fz + bottom_fz,
                        }
                    )
                except VasprunValidationError as error:
                    row.update({"status": error.status, "error": str(error)})
            raw.append(row)

    references = {
        (row["functional"], row["orientation"]): row.get("energy_eV")
        for row in raw
        if row["reference"] == "True" and row["status"] == "complete"
    }
    for row in raw:
        key = (row["functional"], row["orientation"])
        if row["status"] == "complete" and key in references:
            row["relative_energy_per_LiF_meV"] = (
                500.0 * (float(row["energy_eV"]) - references[key])
            )

    fields = sorted({key for row in raw for key in row})
    output = ROOT / "results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw)
    complete = sum(row["status"] == "complete" for row in raw)
    print(f"Wrote {output}: {complete}/{len(raw)} stages complete")


if __name__ == "__main__":
    main()
