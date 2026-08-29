from __future__ import annotations

import unittest

import torch

from mace_fno import (
    cubic_signed_permutation_matrices,
    is_cubic_cell,
    transform_in_cell_axis_basis,
)


DTYPE = torch.float64


class CubicSymmetryTests(unittest.TestCase):
    def test_group_sizes_determinants_and_uniqueness(self) -> None:
        proper = cubic_signed_permutation_matrices(
            include_reflections=False, dtype=DTYPE
        )
        full = cubic_signed_permutation_matrices(dtype=DTYPE)
        self.assertEqual(proper.shape, (24, 3, 3))
        self.assertEqual(full.shape, (48, 3, 3))
        self.assertEqual(torch.unique(full.reshape(48, -1), dim=0).shape[0], 48)
        torch.testing.assert_close(
            torch.linalg.det(proper), torch.ones(24, dtype=DTYPE)
        )
        determinants = torch.linalg.det(full)
        self.assertEqual(int((determinants > 0).sum()), 24)
        self.assertEqual(int((determinants < 0).sum()), 24)

    def test_full_group_is_closed_and_contains_inverses(self) -> None:
        group = cubic_signed_permutation_matrices(dtype=DTYPE)
        keys = {tuple(matrix.reshape(-1).tolist()) for matrix in group}
        for first in group:
            self.assertIn(tuple(first.T.reshape(-1).tolist()), keys)
            for second in group:
                self.assertIn(tuple((first @ second).reshape(-1).tolist()), keys)

    def test_transform_batch_matches_individual_operations(self) -> None:
        cell = torch.tensor(
            ((0.0, 0.0, 8.0), (8.0, 0.0, 0.0), (0.0, 8.0, 0.0)),
            dtype=DTYPE,
        )
        vectors = torch.tensor(
            ((1.2, -0.4, 0.7), (-2.1, 1.3, 0.2)), dtype=DTYPE
        )
        group = cubic_signed_permutation_matrices(dtype=DTYPE)[:7]
        batched = transform_in_cell_axis_basis(vectors, cell, group)
        expected = torch.stack(
            [transform_in_cell_axis_basis(vectors, cell, item) for item in group]
        )
        torch.testing.assert_close(batched, expected, atol=0.0, rtol=0.0)
        original_norms = vectors.square().sum(dim=-1)
        torch.testing.assert_close(
            batched.square().sum(dim=-1), original_norms.expand_as(batched[..., 0])
        )

    def test_cubic_cell_detection(self) -> None:
        cubic = torch.diag(torch.tensor((12.4, 12.4, 12.4), dtype=DTYPE))
        orthorhombic = torch.diag(torch.tensor((12.4, 12.4, 13.0), dtype=DTYPE))
        tilted = cubic.clone()
        tilted[1, 0] = 0.2
        self.assertTrue(is_cubic_cell(cubic))
        self.assertFalse(is_cubic_cell(orthorhombic))
        self.assertFalse(is_cubic_cell(tilted))


if __name__ == "__main__":
    unittest.main()
