from __future__ import annotations

import unittest

from mace_fno import (
    FNO2D,
    FNO2d,
    FNO3D,
    FNO3d,
    FNOFieldOperator2D,
    FNOFieldOperator3D,
    FNOFieldOperator3d,
    LinearFNO2D,
    LinearFNO2d,
    LinearFNO3D,
    LinearFNO3d,
)
from mace_fno.cli.config import parse_arguments


class PublicAPITests(unittest.TestCase):
    def test_canonical_dimension_aliases_preserve_legacy_api(self) -> None:
        self.assertIs(FNO2D, FNO2d)
        self.assertIs(FNO3D, FNO3d)
        self.assertIs(LinearFNO2D, LinearFNO2d)
        self.assertIs(LinearFNO3D, LinearFNO3d)
        self.assertEqual(FNOFieldOperator2D.__name__, "FNOFieldOperator")
        self.assertIs(FNOFieldOperator3D, FNOFieldOperator3d)

    def test_implementations_are_split_by_geometry(self) -> None:
        self.assertEqual(FNO2D.__module__, "mace_fno.fno_2d")
        self.assertEqual(FNO3D.__module__, "mace_fno.fno_3d")

    def test_train_arguments_can_be_parsed_programmatically(self) -> None:
        args = parse_arguments(
            ["--mace-model", "model.pt", "--train-file", "train.xyz"]
        )
        self.assertEqual(args.spatial_scheme, "auto")
        self.assertEqual(args.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
