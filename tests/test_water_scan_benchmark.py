"""Regression checks for exact-geometry-disjoint water-SCAN preparation."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

spec = importlib.util.spec_from_file_location(
    "water_scan_prepare",
    Path(__file__).resolve().parents[1]
    / "benchmarks/water_scan_qnep/prepare_dataset.py",
)
PREPARE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PREPARE)


def frame(index, *, permutation=False, energy_shift=0.0, force_shift=0.0):
    atoms = Atoms(
        "H2O",
        positions=[[0.1 + index * 0.01, 0, 0], [1.0, 0, 0], [0, 1, 0]],
        cell=np.eye(3) * (10 + index * 0.1),
        pbc=True,
    )
    forces = np.arange(9, dtype=float).reshape(3, 3) * 0.01
    if permutation:
        atoms = atoms[[2, 0, 1]]
        forces = forces[[2, 0, 1]]
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(index) + energy_shift,
        forces=forces + force_shift,
    )
    return atoms


class WaterScanPreparationTests(unittest.TestCase):
    def test_exact_identity_includes_geometry_cell_species_and_pbc(self):
        first = frame(0)
        fingerprint = PREPARE.geometry_fingerprint(first)
        self.assertEqual(
            fingerprint, PREPARE.geometry_fingerprint(frame(0, permutation=True))
        )
        for field in ("positions", "cell", "numbers", "pbc"):
            changed = first.copy()
            if field == "positions":
                changed.positions[0, 0] += 1e-10
            elif field == "cell":
                changed.cell[0, 0] += 1e-10
            elif field == "numbers":
                changed.numbers[0] = 8
            else:
                changed.pbc[2] = False
            self.assertNotEqual(fingerprint, PREPARE.geometry_fingerprint(changed))

    def test_disjoint_deterministic_complete_split_preserves_source_roles(self):
        train = [frame(i) for i in range(20)] + [frame(0), frame(1, permutation=True)]
        test = [frame(0), frame(0), frame(4), frame(20), frame(20), frame(21)]
        splits, report = PREPARE.geometry_disjoint_split(train, test, 0.1, 17)
        again, repeated = PREPARE.geometry_disjoint_split(train, test, 0.1, 17)
        self.assertEqual(report, repeated)
        self.assertEqual(report["counts"], {"train": 18, "validation": 2, "test": 2})
        self.assertEqual(report["unique_union"], 22)
        self.assertEqual(report["shared_unique_geometries"], 2)
        self.assertEqual(report["source_validation_entries_seen_in_training"], 3)
        self.assertEqual(report["source_indices"]["test"], [3, 5])
        self.assertFalse(any(report["pairwise_geometry_overlap"].values()))
        for name, structures in splits.items():
            self.assertEqual(len(structures), len(again[name]))
            self.assertEqual(
                len(structures),
                len({PREPARE.geometry_fingerprint(a) for a in structures}),
            )
            source = test if name == "test" else train
            for atoms, index in zip(structures, report["source_indices"][name]):
                self.assertEqual(
                    atoms.get_potential_energy(), source[index].get_potential_energy()
                )
                np.testing.assert_array_equal(
                    atoms.get_forces(), source[index].get_forces()
                )
        self.assertNotIn("benchmark_split", train[0].info)

    def test_conflicting_labels_are_not_silently_averaged(self):
        for kwargs in ({"energy_shift": 0.01}, {"force_shift": 0.01}):
            with self.assertRaisesRegex(ValueError, "conflicting duplicate labels"):
                PREPARE.geometry_disjoint_split(
                    [frame(i) for i in range(10)] + [frame(0, **kwargs)],
                    [frame(10)],
                    0.1,
                    17,
                )

    def test_roundoff_label_variation_keeps_first_representative(self):
        train = [frame(i) for i in range(10)]
        train.append(frame(0, force_shift=2e-6, permutation=True))
        splits, report = PREPARE.geometry_disjoint_split(train, [frame(10)], 0.1, 17)
        first = next(
            a
            for name in ("train", "validation")
            for a in splits[name]
            if a.info["source_index"] == 0
        )
        np.testing.assert_array_equal(first.get_forces(), train[0].get_forces())
        self.assertAlmostEqual(
            report["label_consistency"]["max_force_component_span_eV_per_A"], 2e-6
        )

    def test_empty_unseen_test_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no source-validation geometries"):
            PREPARE.geometry_disjoint_split(
                [frame(i) for i in range(10)], [frame(0)], 0.1, 17
            )

    def test_single_unique_training_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two unique"):
            PREPARE.geometry_disjoint_split([frame(0), frame(0)], [frame(1)], 0.1, 17)
