"""Continuous Gaussian synthesis; no experiments run on import.

Set BLAS thread environment variables before importing this package when using
the library directly. The supplied runners do this automatically.
"""

from .api import SynthesisConfig, SynthesisResult, synthesize
from .extent import ExtentConfig, ExtentResult, estimate_extent
from .basis import FockBasis
from .gaussian import GaussianState
from .measure import GaussianCoefficientMeasure

__all__ = [
    "FockBasis", "GaussianState", "GaussianCoefficientMeasure",
    "SynthesisConfig", "SynthesisResult", "synthesize",
    "ExtentConfig", "ExtentResult", "estimate_extent",
]
