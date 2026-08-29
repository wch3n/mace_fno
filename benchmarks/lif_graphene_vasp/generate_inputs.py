#!/usr/bin/env python3
"""Generate symmetric LiF/graphene/LiF VASP single-point calculations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar


ROOT = Path(__file__).resolve().parent
CALCULATIONS = ROOT / "calculations"
FUNCTIONALS = ("pbe", "pbe0", "pbe0_rvv10")
HEIGHTS_ANGSTROM = (3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
ORIENTATIONS = ("li_near", "f_near", "parallel_a1")

GRAPHENE_A = 2.460
GRAPHENE_REPETITIONS = (4, 4)
CELL_C = 40.0
LIF_BOND = 1.564
KPOINT_MESH = (4, 4, 1)
ENCUT_EV = 600
RVV10_B = 10.0
RVV10_C = 0.0093


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--potcar-functional",
        default="PBE_54",
        help="pymatgen POTCAR functional (default: PBE_54)",
    )
    return parser.parse_args()


def graphene_supercell() -> tuple[Structure, np.ndarray]:
    """Return 4x4 graphene and an exact hollow-site inversion centre."""
    a = GRAPHENE_A
    lattice = Lattice(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, CELL_C],
        ]
    )
    primitive = Structure(
        lattice,
        ["C", "C"],
        [[0.0, 0.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]],
    )
    primitive.make_supercell([*GRAPHENE_REPETITIONS, 1])

    # (1/3, 2/3) is a hollow site in the chosen graphene primitive cell.
    # The integer translations place it near the centre of the 4x4 supercell.
    hollow_fractional = np.array(
        [(2.0 + 1.0 / 3.0) / 4.0, (1.0 + 2.0 / 3.0) / 4.0, 0.5]
    )
    distances = []
    for site in primitive:
        delta = np.asarray(site.frac_coords) - hollow_fractional
        delta -= np.rint(delta)
        distances.append(np.linalg.norm(delta @ primitive.lattice.matrix))
    nearest = np.sort(np.asarray(distances))[:6]
    expected = GRAPHENE_A / math.sqrt(3.0)
    if not np.allclose(nearest, expected, atol=1.0e-8, rtol=0.0):
        raise RuntimeError(f"invalid hollow site: six nearest C distances are {nearest}")
    return primitive, hollow_fractional


def paired_lif_structure(
    graphene: Structure,
    hollow_fractional: np.ndarray,
    height: float,
    orientation: str,
) -> Structure:
    """Place an inversion-related LiF pair on opposite graphene faces."""
    centre = graphene.lattice.get_cartesian_coords(hollow_fractional)
    normal = np.array([0.0, 0.0, 1.0])
    a1 = np.array(graphene.lattice.matrix[0], dtype=float, copy=True)
    a1 /= np.linalg.norm(a1)

    if orientation == "li_near":
        top_li = centre + (height - 0.5 * LIF_BOND) * normal
        top_f = centre + (height + 0.5 * LIF_BOND) * normal
    elif orientation == "f_near":
        top_li = centre + (height + 0.5 * LIF_BOND) * normal
        top_f = centre + (height - 0.5 * LIF_BOND) * normal
    elif orientation == "parallel_a1":
        top_midpoint = centre + height * normal
        top_li = top_midpoint - 0.5 * LIF_BOND * a1
        top_f = top_midpoint + 0.5 * LIF_BOND * a1
    else:
        raise ValueError(f"unknown orientation {orientation}")

    bottom_li = 2.0 * centre - top_li
    bottom_f = 2.0 * centre - top_f
    species = ["C"] * len(graphene) + ["Li", "Li", "F", "F"]
    cart_coords = [site.coords for site in graphene] + [
        top_li,
        bottom_li,
        top_f,
        bottom_f,
    ]
    structure = Structure(
        graphene.lattice,
        species,
        cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )
    assert_inversion_symmetry(structure, hollow_fractional)
    return structure


def assert_inversion_symmetry(
    structure: Structure, centre_fractional: np.ndarray, tolerance: float = 1.0e-7
) -> None:
    """Check that every site maps to an identical species under inversion."""
    fractional = np.asarray(structure.frac_coords)
    species = np.asarray([site.specie.symbol for site in structure])
    for index, (coords, symbol) in enumerate(zip(fractional, species)):
        target = np.mod(2.0 * centre_fractional - coords, 1.0)
        candidates = fractional[species == symbol]
        delta = candidates - target
        delta -= np.rint(delta)
        distances = np.linalg.norm(delta @ structure.lattice.matrix, axis=1)
        if float(distances.min()) > tolerance:
            raise RuntimeError(
                f"site {index} ({symbol}) lacks an inversion partner; "
                f"minimum mismatch={distances.min():.3e} A"
            )


def common_incar(system: str) -> dict[str, object]:
    return {
        "SYSTEM": system,
        "GGA": "PE",
        "ENCUT": ENCUT_EV,
        "PREC": "Accurate",
        "EDIFF": 1.0e-7,
        "NELM": 200,
        "NELMIN": 6,
        "ALGO": "Normal",
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ISPIN": 1,
        # Full k mesh makes all cases use the same GPU decomposition and leaves
        # the inversion-force check independent of VASP force symmetrization.
        "ISYM": 0,
        "LREAL": False,
        "LASPH": True,
        "ADDGRID": True,
        "IBRION": -1,
        "NSW": 0,
        "ISIF": 2,
        "LWAVE": True,
        "LCHARG": True,
        "LORBIT": 11,
        "KPAR": 4,
        "NCORE": 1,
    }


def incar_for(functional: str, system: str) -> Incar:
    settings = common_incar(system)
    if functional == "pbe":
        settings.update({"ISTART": 0, "ICHARG": 2})
    elif functional in {"pbe0", "pbe0_rvv10"}:
        settings.update(
            {
                "ISTART": 1,
                "ICHARG": 1,
                "LHFCALC": True,
                "AEXX": 0.25,
                "LFOCKACE": True,
                "PRECFOCK": "Normal",
                # Graphene is semimetallic; the standard G=0 treatment is the
                # more rapidly convergent HFRCUT choice for metals.
                "HFRCUT": 0,
            }
        )
        if functional == "pbe0_rvv10":
            settings.update(
                {
                    "LUSE_VDW": True,
                    "IVDW_NL": 2,
                    "BPARAM": RVV10_B,
                    "CPARAM": RVV10_C,
                }
            )
    else:
        raise ValueError(functional)
    return Incar(settings)


def case_name(index: int, height: float, orientation: str) -> str:
    height_token = f"{height:05.2f}".replace(".", "p")
    return f"{index:03d}_h{height_token}_{orientation}"


def main() -> None:
    args = parse_args()
    graphene, hollow_fractional = graphene_supercell()
    potcar = Potcar(["C", "Li_sv", "F"], functional=args.potcar_functional)
    kpoints = Kpoints.gamma_automatic(KPOINT_MESH)

    CALCULATIONS.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    index = 0
    for orientation in ORIENTATIONS:
        for height in HEIGHTS_ANGSTROM:
            name = case_name(index, height, orientation)
            structure = paired_lif_structure(
                graphene, hollow_fractional, height, orientation
            )
            cases.append(
                {
                    "index": index,
                    "case": name,
                    "orientation": orientation,
                    "height_A": height,
                    "reference": height == max(HEIGHTS_ANGSTROM),
                    "atoms": len(structure),
                }
            )
            for functional in FUNCTIONALS:
                directory = CALCULATIONS / name / functional
                directory.mkdir(parents=True, exist_ok=True)
                Poscar(structure, comment=f"LiF_G_LiF {orientation} h={height:.2f} A").write_file(
                    directory / "POSCAR", significant_figures=16
                )
                incar_for(functional, f"LiF_G_LiF_{orientation}_h{height:.2f}").write_file(
                    directory / "INCAR"
                )
                kpoints.write_file(directory / "KPOINTS")
                potcar.write_file(directory / "POTCAR")
                (directory / "POTCAR.spec").write_text(
                    "C\nLi_sv\nF\n", encoding="utf-8"
                )
            index += 1

    with (ROOT / "cases.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cases[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(cases)
    (ROOT / "cases.list").write_text(
        "".join(f"{case['case']}\n" for case in cases), encoding="utf-8"
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).name),
        "graphene_lattice_A": GRAPHENE_A,
        "graphene_repetitions": GRAPHENE_REPETITIONS,
        "cell_c_A": CELL_C,
        "lif_bond_A": LIF_BOND,
        "height_definition": "LiF bond midpoint to graphene plane",
        "heights_A": HEIGHTS_ANGSTROM,
        "orientations": ORIENTATIONS,
        "hollow_fractional": hollow_fractional.tolist(),
        "kpoint_mesh": KPOINT_MESH,
        "encut_eV": ENCUT_EV,
        "functionals": FUNCTIONALS,
        "pbe0_rvv10_label": "exploratory PBE0+rVV10L(b=10.0)",
        "potcar_functional": args.potcar_functional,
        "potcars": [
            {
                "symbol": entry.symbol,
                "titel": entry.TITEL,
                "enmax_eV": float(entry.enmax),
            }
            for entry in potcar
        ],
        "dipole_correction": False,
        "coulomb_kernel_truncation": False,
        "number_of_cases": len(cases),
    }
    (ROOT / "generation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Generated {len(cases)} cases x {len(FUNCTIONALS)} stages "
        f"under {CALCULATIONS}"
    )


if __name__ == "__main__":
    main()
