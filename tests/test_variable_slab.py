"""Variable in-plane slab cells through data loading, batching, and training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from mace_fno_test_helpers import FakeIrreps, FakeMACE, FakeProduct, train_arguments
from torch import nn

from mace_fno.training.configuration import TrainingConfig
from mace_fno.training.data import (
    collate_samples,
    load_or_create_samples,
    load_samples,
    save_sample_cache,
)
from mace_fno.training.setup import build_training_model

DTYPE = torch.float64


def backbone():
    model = FakeMACE()
    model.products = nn.ModuleList([FakeProduct(FakeIrreps([(4, 0, 1)]))])
    return model


class Calculator:
    models = [backbone()]

    def _atoms_to_batch(self, atoms):
        n = len(atoms)
        graph = {
            "positions": torch.tensor(atoms.positions, dtype=DTYPE),
            "cell": torch.tensor(atoms.cell.array, dtype=DTYPE)[None],
            "batch": torch.zeros(n, dtype=torch.long),
            "ptr": torch.tensor([0, n]),
            "node_attrs": torch.ones((n, 1), dtype=DTYPE),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "shifts": torch.empty((0, 3), dtype=DTYPE),
            "head": torch.zeros(1, dtype=torch.long),
        }
        return SimpleNamespace(to=lambda device: SimpleNamespace(to_dict=lambda: graph))


class VariableSlabTests(unittest.TestCase):
    def configuration(self, *extra):
        return TrainingConfig.from_namespace(
            train_arguments(
                "--spatial-scheme",
                "2.5d",
                "--cell-mode",
                "anisotropic",
                "--spectral-symmetry",
                "metric_eqgino",
                "--z-grid",
                "4",
                "--z-extent",
                "10",
                "--grid",
                "8",
                "--modes",
                "2",
                "--channels",
                "2",
                "--source-hidden-channels",
                "4",
                "--fno-hidden-channels",
                "4",
                "--fno-layers",
                "1",
                "--random-residual-initialization",
                *extra,
            )
        )

    @staticmethod
    def atoms():
        frames = []
        for n, cell in (
            (2, [[12.0, 0, 0], [0, 14.0, 0], [0, 0, 20.0]]),
            (3, [[18.0, 0, 0], [3.0, 22.0, 0], [2.0, 0, 25.0]]),
        ):
            atoms = Atoms(
                "H" * n,
                positions=[[1 + i * 2.3, 2 + i * 0.3, 9 + i * 0.2] for i in range(n)],
                cell=cell,
                pbc=[True, True, False],
            )
            atoms.calc = SinglePointCalculator(
                atoms, energy=1.0, forces=np.zeros((n, 3))
            )
            frames.append(atoms)
        return frames

    def test_invalid_combinations_still_rejected(self):
        for extra in (
            ("--cell-mode", "isotropic"),
            ("--spectral-symmetry", "none"),
            ("--architecture", "linear"),
            ("--planar-symmetry", "d4"),
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.configuration(*extra)

    def test_mixed_shapes_sizes_cache_batching_and_force_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xyz, mace, cache = (
                root / "data.xyz",
                root / "mace.model",
                root / "samples.pt",
            )
            write(xyz, self.atoms())
            mace.touch()
            args = (
                Calculator(),
                xyz,
                "energy",
                "forces",
                DTYPE,
                None,
                False,
                False,
                mace,
                cache,
                False,
            )
            samples, reference, metadata, hit = load_or_create_samples(
                *args, spatial_scheme="2.5d", cell_mode="anisotropic"
            )
            self.assertFalse(hit)
            self.assertEqual([s["num_atoms"] for s in samples], [2, 3])
            save_sample_cache(cache, metadata, samples, reference)
            cached, _, _, hit = load_or_create_samples(
                *args,
                reference_cell=reference * 1.1,
                spatial_scheme="2.5d",
                cell_mode="anisotropic",
            )
            self.assertTrue(hit)
            self.assertEqual(len(cached), 2)
            model = build_training_model(
                backbone(),
                self.configuration(),
                reference,
                device=torch.device("cpu"),
                dtype=DTYPE,
            )
            graph, _, _ = collate_samples(samples, torch.device("cpu"), DTYPE)
            output = model(graph, training=True, compute_force=True)
            for i, sample in enumerate(samples):
                single, _, _ = collate_samples([sample], torch.device("cpu"), DTYPE)
                expected = model(single, compute_force=True)
                torch.testing.assert_close(
                    output["energy"][i : i + 1], expected["energy"]
                )
                torch.testing.assert_close(
                    output["forces"][graph["batch"] == i], expected["forces"]
                )

            # A position FD checks the complete mixed-cell energy/force path.
            def displaced(delta):
                graph, _, _ = collate_samples(samples, torch.device("cpu"), DTYPE)
                graph["positions"][3, 0] += delta
                return model(graph, compute_force=False)["energy"].sum().item()

            fd = -(displaced(1e-5) - displaced(-1e-5)) / 2e-5
            self.assertAlmostEqual(fd, output["forces"][3, 0].item(), delta=1e-7)
            output["forces"].square().sum().backward()
            radial = model.long_range.field_operator.fno.blocks[
                0
            ].spectral.radial_weight
            self.assertTrue(torch.isfinite(radial.grad).all())
            self.assertGreater(radial.grad.abs().max().item(), 0)
            self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
            with self.assertRaisesRegex(ValueError, "cell mismatch"):
                load_samples(
                    Calculator(),
                    xyz,
                    "energy",
                    "forces",
                    DTYPE,
                    None,
                    False,
                    False,
                    spatial_scheme="2.5d",
                    cell_mode="fixed",
                )
            bad = graph["cell"].clone()
            bad[1, 1] = 2 * bad[1, 0]
            with self.assertRaisesRegex(ValueError, "nondegenerate"):
                model._validate_cells(bad)


if __name__ == "__main__":
    unittest.main()
