"""Continuous Gaussian synthesis; no experiments run on import.

Set BLAS thread environment variables before importing this package when using
the library directly. The supplied runners do this automatically.
"""

from .api import SynthesisConfig, SynthesisResult, synthesize
from .extent import ExtentConfig, ExtentResult, estimate_extent
from .basis import FermionSpace, FockBasis
from .sectors import OccupationSector, NumberSector
from .metrics import fidelity, phase_aligned_distance
from .gaussian import GaussianState
from .measure import GaussianCoefficientMeasure

__all__ = [
    "FockBasis", "GaussianState", "GaussianCoefficientMeasure",
    "SynthesisConfig", "SynthesisResult", "synthesize",
    "ExtentConfig", "ExtentResult", "estimate_extent",
    "FermionSpace", "OccupationSector", "NumberSector",
    "fidelity", "phase_aligned_distance",
]
