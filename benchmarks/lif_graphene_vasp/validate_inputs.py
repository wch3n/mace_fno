#!/usr/bin/env python3
"""Fail-fast validation for the generated LiF/graphene/LiF VASP inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar


ROOT = Path(__file__).resolve().parent
FUNCTIONALS = ("pbe", "pbe0", "pbe0_rvv10")
FORBIDDEN_ELECTROSTATIC_TAGS = {
    "LDIPOL",
    "IDIPOL",
    "DIPOL",
    "KERNEL_TRUNCATION/LTRUNCATE",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    with (ROOT / "cases.tsv").open(encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle, delimiter="\t"))
    metadata = json.loads((ROOT / "generation.json").read_text(encoding="utf-8"))
    if len(cases) != 21 or metadata["number_of_cases"] != len(cases):
        fail(f"expected 21 cases, found {len(cases)}")

    reference_cell = None
    reference_kpoints = None
    reference_potcar_titles = None
    checked = 0
    for case in cases:
        stage_structures = []
        for functional in FUNCTIONALS:
            directory = ROOT / "calculations" / case["case"] / functional
            required = ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "POTCAR.spec")
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                fail(f"{directory}: missing {missing}")

            incar = Incar.from_file(directory / "INCAR")
            forbidden = FORBIDDEN_ELECTROSTATIC_TAGS.intersection(incar)
            if forbidden:
                fail(f"{directory}: forbidden electrostatic tags {sorted(forbidden)}")
            if incar.get("ENCUT") != 600 or incar.get("NSW") != 0:
                fail(f"{directory}: unexpected ENCUT/NSW")
            is_hybrid = functional != "pbe"
            if bool(incar.get("LHFCALC", False)) != is_hybrid:
                fail(f"{directory}: LHFCALC does not match stage")
            if functional == "pbe0_rvv10":
                expected = {
                    "LUSE_VDW": True,
                    "IVDW_NL": 2,
                    "BPARAM": 10.0,
                    "CPARAM": 0.0093,
                }
                for tag, value in expected.items():
                    if incar.get(tag) != value:
                        fail(f"{directory}: expected {tag}={value}")
            elif incar.get("LUSE_VDW", False):
                fail(f"{directory}: unexpected LUSE_VDW")

            kpoints = Kpoints.from_file(directory / "KPOINTS")
            mesh = tuple(int(value) for value in kpoints.kpts[0])
            if mesh != (4, 4, 1):
                fail(f"{directory}: unexpected k mesh {mesh}")
            if reference_kpoints is None:
                reference_kpoints = mesh
            elif mesh != reference_kpoints:
                fail(f"{directory}: inconsistent k mesh")

            structure = Poscar.from_file(directory / "POSCAR").structure
            stage_structures.append(structure)
            expected_composition = {"C": 32.0, "Li": 2.0, "F": 2.0}
            actual_composition = structure.composition.get_el_amt_dict()
            if len(structure) != 36 or actual_composition != expected_composition:
                fail(f"{directory}: unexpected composition {structure.composition}")
            symbols = [site.specie.symbol for site in structure]
            if symbols != ["C"] * 32 + ["Li", "Li", "F", "F"]:
                fail(f"{directory}: unexpected site/species order")
            cell = np.asarray(structure.lattice.matrix)
            if reference_cell is None:
                reference_cell = cell
            elif not np.allclose(cell, reference_cell, atol=1.0e-12, rtol=0.0):
                fail(f"{directory}: cell mismatch")

            # Generator order: C32, Li_top, Li_bottom, F_top, F_bottom.
            top = np.asarray(structure.cart_coords)[[32, 34]]
            bottom = np.asarray(structure.cart_coords)[[33, 35]]
            midpoint = structure.lattice.get_cartesian_coords(
                metadata["hollow_fractional"]
            )
            top_midpoint = top.mean(axis=0)
            bottom_midpoint = bottom.mean(axis=0)
            graphene_z = float(np.asarray(structure.cart_coords)[:32, 2].mean())
            expected_height = float(case["height_A"])
            if not np.allclose(
                [top_midpoint[2] - graphene_z, graphene_z - bottom_midpoint[2]],
                expected_height,
                atol=1.0e-10,
                rtol=0.0,
            ):
                fail(f"{directory}: incorrect LiF midpoint height")
            bonds = [np.linalg.norm(top[1] - top[0]), np.linalg.norm(bottom[1] - bottom[0])]
            if not np.allclose(bonds, metadata["lif_bond_A"], atol=1.0e-10, rtol=0.0):
                fail(f"{directory}: incorrect LiF bond length {bonds}")
            target = 2.0 * midpoint - top
            delta_fractional = structure.lattice.get_fractional_coords(bottom - target)
            delta_fractional -= np.rint(delta_fractional)
            mismatch = np.linalg.norm(delta_fractional @ cell, axis=1).max()
            if mismatch > 1.0e-7:
                fail(f"{directory}: LiF inversion mismatch {mismatch:.3e} A")

            potcar = Potcar.from_file(directory / "POTCAR")
            titles = tuple(entry.TITEL for entry in potcar)
            if tuple(entry.symbol for entry in potcar) != ("C", "Li_sv", "F"):
                fail(f"{directory}: unexpected POTCAR order")
            if reference_potcar_titles is None:
                reference_potcar_titles = titles
            elif titles != reference_potcar_titles:
                fail(f"{directory}: inconsistent POTCAR datasets")
            if max(float(entry.enmax) for entry in potcar) > float(incar["ENCUT"]):
                fail(f"{directory}: ENCUT is below a POTCAR ENMAX")
            checked += 1

        fractional = [np.asarray(structure.frac_coords) for structure in stage_structures]
        if not all(np.allclose(fractional[0], value, atol=1.0e-13) for value in fractional[1:]):
            fail(f"{case['case']}: geometry differs between functional stages")

    print(
        f"PASS: {checked} VASP stages ({len(cases)} cases x {len(FUNCTIONALS)}) "
        "have matched cells/geometries/k meshes, inversion-paired LiF, valid "
        "POTCAR order, and no dipole/kernel-truncation tags."
    )


if __name__ == "__main__":
    main()
