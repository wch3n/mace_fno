"""Backward-compatible public imports for Fourier neural operators.

Implementations are separated by geometry in :mod:`mace_fno.fno_2d`,
:mod:`mace_fno.fno_slab`, and :mod:`mace_fno.fno_3d`.
"""

from .fno_2d import (
    FNO2d,
    FNOBlock2d,
    FNOFieldOperator,
    LinearFNO2d,
    SpectralConv2d,
)
from .fno_3d import (
    CubicAdaptiveSpectralConv3d,
    EqGINOSpectralConv3d,
    FNO3d,
    FNOBlock3d,
    FNOFieldOperator3d,
    LinearFNO3d,
    MetricEqGINOSpectralConv3d,
    SpectralConv3d,
)
from .fno_slab import (
    FNO2p5D,
    FNOBlock2p5d,
    FNOFieldOperator2p5D,
    GlobalZMixing,
    LinearFNO2p5D,
    PlanarSpectralConv2p5d,
    SpectralConv2p5d,
)

# Canonical dimensional suffixes for new code. The historical names remain
# supported because existing scripts and checkpoints already use them.
FNO2D = FNO2d
FNO3D = FNO3d
FNOFieldOperator2D = FNOFieldOperator
FNOFieldOperator3D = FNOFieldOperator3d
LinearFNO2D = LinearFNO2d
LinearFNO3D = LinearFNO3d
PlanarSpectralConv2p5D = PlanarSpectralConv2p5d
SpectralConv2D = SpectralConv2d
SpectralConv2p5D = SpectralConv2p5d
SpectralConv3D = SpectralConv3d

__all__ = [
    "CubicAdaptiveSpectralConv3d",
    "EqGINOSpectralConv3d",
    "FNO2d",
    "FNO2D",
    "FNO2p5D",
    "FNO3d",
    "FNO3D",
    "FNOBlock2d",
    "FNOBlock2p5d",
    "FNOBlock3d",
    "FNOFieldOperator",
    "FNOFieldOperator2D",
    "FNOFieldOperator2p5D",
    "FNOFieldOperator3d",
    "FNOFieldOperator3D",
    "GlobalZMixing",
    "LinearFNO2d",
    "LinearFNO2D",
    "LinearFNO2p5D",
    "LinearFNO3d",
    "LinearFNO3D",
    "MetricEqGINOSpectralConv3d",
    "PlanarSpectralConv2p5d",
    "PlanarSpectralConv2p5D",
    "SpectralConv2d",
    "SpectralConv2D",
    "SpectralConv2p5d",
    "SpectralConv2p5D",
    "SpectralConv3d",
    "SpectralConv3D",
]
