from __future__ import annotations

import unittest
from pathlib import Path
from runpy import run_path

import numpy as np
from ase import Atoms

BENCHMARK = run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "au_mgo"
        / "evaluate_adsorption_switch.py"
    )
)
force_max = BENCHMARK["force_max"]
movable_indices = BENCHMARK["movable_indices"]
summarize_switch = BENCHMARK["summarize_switch"]


def record(reference: float, predicted: float) -> dict[str, float]:
    return {
        "reference_energy_eV": reference,
        "final_energy_eV": predicted,
    }


class AdsorptionSwitchTests(unittest.TestCase):
    def test_movable_indices_select_only_requested_element(self) -> None:
        atoms = Atoms("MgAuOAuAl")
        self.assertEqual(movable_indices(atoms, "Au"), [1, 3])

    def test_force_max_uses_only_movable_atoms(self) -> None:
        forces = np.asarray([[100.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
        self.assertEqual(force_max(forces, [1]), 5.0)

    def test_summary_detects_dopant_induced_sign_reversal(self) -> None:
        records = {
            "1-undoped": record(-10.0, -20.0),
            "3-undoped": record(-9.0, -19.1),
            "1-doped": record(-12.0, -22.0),
            "3-doped": record(-12.1, -22.05),
        }
        summary = summarize_switch(records)

        self.assertAlmostEqual(summary["undoped"]["reference_delta_meV"], 1000.0)
        self.assertAlmostEqual(summary["undoped"]["predicted_delta_meV"], 900.0)
        self.assertAlmostEqual(summary["doped"]["reference_delta_meV"], -100.0)
        self.assertAlmostEqual(summary["doped"]["predicted_delta_meV"], -50.0)
        self.assertTrue(summary["undoped"]["sign_match"])
        self.assertTrue(summary["doped"]["sign_match"])
        self.assertTrue(summary["sign_reversal_reproduced"])


if __name__ == "__main__":
    unittest.main()
