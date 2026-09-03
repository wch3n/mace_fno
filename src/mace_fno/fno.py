"""Public Fourier neural operators grouped by physical geometry."""

from .fno_2d import (
    FNO2D,
    FNOBlock2D,
    FNOFieldOperator2D,
    LinearFNO2D,
    SpectralConv2D,
)
from .fno_3d import (
    FNO3D,
    FNOBlock3D,
    FNOFieldOperator3D,
    LinearFNO3D,
    MetricEqGINOSpectralConv3D,
    SpectralConv3D,
)
from .fno_slab import (
    GlobalZMixing,
    LinearSlabFNO2D,
    SlabFNO2D,
    SlabFNOBlock2D,
    SlabFNOFieldOperator2D,
    SlabPlanarSpectralConv2D,
    SlabSpectralConv2D,
)

__all__ = [
    "FNO2D",
    "FNO3D",
    "FNOBlock2D",
    "FNOBlock3D",
    "FNOFieldOperator2D",
    "FNOFieldOperator3D",
    "GlobalZMixing",
    "LinearFNO2D",
    "LinearFNO3D",
    "LinearSlabFNO2D",
    "MetricEqGINOSpectralConv3D",
    "SlabFNO2D",
    "SlabFNOBlock2D",
    "SlabFNOFieldOperator2D",
    "SlabPlanarSpectralConv2D",
    "SlabSpectralConv2D",
    "SpectralConv2D",
    "SpectralConv3D",
]
