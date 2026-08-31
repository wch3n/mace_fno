from __future__ import annotations

import unittest

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from torch import nn

from mace_fno import MACEFNOCalculator


class _ToyCombinedModel(nn.Module):
    def __init__(self, spatial_scheme: str = "2d") -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.spatial_scheme = spatial_scheme

    def forward(
        self,
        graph,
        *,
        training: bool,
        compute_force: bool,
    ):
        del training
        positions = graph["positions"]
        positions.requires_grad_(True)
        base_energy = self.scale * positions.square().sum().reshape(1)
        residual_energy = 0.25 * self.scale * positions.square().sum().reshape(1)
        energy = base_energy + residual_energy
        forces = None
        if compute_force:
            forces = -torch.autograd.grad(energy.sum(), positions)[0]
        return {
            "energy": energy,
            "forces": forces,
            "base_energy": base_energy,
            "residual_energy": residual_energy,
        }


def _graph_converter(atoms: Atoms):
    count = len(atoms)
    return {
        "positions": torch.as_tensor(atoms.positions, dtype=torch.float64),
        "cell": torch.as_tensor(atoms.cell.array, dtype=torch.float64).unsqueeze(0),
        "batch": torch.zeros(count, dtype=torch.long),
        "ptr": torch.tensor((0, count), dtype=torch.long),
    }


class CalculatorTests(unittest.TestCase):
    def test_combined_energy_and_conservative_forces_are_exposed_to_ase(self) -> None:
        atoms = Atoms(
            "H2",
            positions=((0.2, 0.3, 0.1), (1.1, 0.4, 0.2)),
            cell=(5.0, 5.0, 12.0),
            pbc=(True, True, False),
        )
        atoms.calc = MACEFNOCalculator(
            model=_ToyCombinedModel(),
            graph_converter=_graph_converter,
        )

        expected_energy = 1.25 * 0.4 * np.square(atoms.positions).sum()
        self.assertAlmostEqual(atoms.get_potential_energy(), expected_energy)
        np.testing.assert_allclose(
            atoms.get_forces(),
            -2.0 * 1.25 * 0.4 * atoms.positions,
            atol=1.0e-14,
            rtol=0.0,
        )
        self.assertAlmostEqual(
            atoms.calc.results["mace_energy"], expected_energy / 1.25
        )
        self.assertAlmostEqual(
            atoms.calc.results["residual_energy"], expected_energy / 5.0
        )
        self.assertEqual(atoms.calc.results["free_energy"], expected_energy)

    def test_periodicity_and_stress_contracts_are_explicit(self) -> None:
        slab = Atoms("H", cell=(5.0, 5.0, 12.0), pbc=(True, False, False))
        slab.calc = MACEFNOCalculator(
            model=_ToyCombinedModel("2.5d"),
            graph_converter=_graph_converter,
        )
        with self.assertRaisesRegex(ValueError, "periodicity on xy"):
            slab.get_potential_energy()

        bulk = Atoms("H", cell=(5.0, 5.0, 5.0), pbc=True)
        bulk.calc = MACEFNOCalculator(
            model=_ToyCombinedModel("3d"),
            graph_converter=_graph_converter,
        )
        with self.assertRaises(PropertyNotImplementedError):
            bulk.get_stress()

    def test_construction_requires_one_complete_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint is required"):
            MACEFNOCalculator()
        with self.assertRaisesRegex(ValueError, "not both"):
            MACEFNOCalculator(
                "checkpoint.pt",
                model=_ToyCombinedModel(),
                graph_converter=_graph_converter,
            )


if __name__ == "__main__":
    unittest.main()
