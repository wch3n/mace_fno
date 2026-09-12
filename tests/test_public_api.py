from __future__ import annotations

import unittest

import mace_fno
from mace_fno import (
    FNO2D,
    FNO3D,
    FNOFieldOperator2D,
    FNOFieldOperator3D,
    LearnedSlabParticleMeshLongRange,
    LinearFNO2D,
    LinearFNO3D,
    SlabFNO2D,
    SlabFNOFieldOperator2D,
    SlabParticleMesh,
    SlabParticleMeshEnergy,
)


class PublicAPITests(unittest.TestCase):
    def test_public_operator_names_are_canonical(self) -> None:
        expected = {
            "FNO2D": FNO2D,
            "FNO3D": FNO3D,
            "FNOFieldOperator2D": FNOFieldOperator2D,
            "FNOFieldOperator3D": FNOFieldOperator3D,
            "LinearFNO2D": LinearFNO2D,
            "LinearFNO3D": LinearFNO3D,
            "LearnedSlabParticleMeshLongRange": LearnedSlabParticleMeshLongRange,
            "SlabFNO2D": SlabFNO2D,
            "SlabFNOFieldOperator2D": SlabFNOFieldOperator2D,
            "SlabParticleMesh": SlabParticleMesh,
            "SlabParticleMeshEnergy": SlabParticleMeshEnergy,
        }
        for name, implementation in expected.items():
            with self.subTest(name=name):
                self.assertEqual(implementation.__name__, name)

        for legacy in (
            "FNO2d",
            "FNO3d",
            "FNO2p5D",
            "FNOFieldOperator3d",
            "LearnedParticleMeshLongRange2p5D",
            "SlabParticleMesh2p5D",
            "EqGINOSpectralConv3d",
            "CubicAdaptiveSpectralConv3d",
        ):
            with self.subTest(legacy=legacy):
                self.assertFalse(hasattr(mace_fno, legacy))


if __name__ == "__main__":
    unittest.main()
