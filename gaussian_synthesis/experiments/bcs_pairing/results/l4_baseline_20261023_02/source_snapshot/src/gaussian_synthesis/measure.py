"""An implicit continuous coefficient measure relative to invariant Haar mu."""

from dataclasses import dataclass
import math
import numpy as np
from .basis import FockBasis
from .gaussian import GaussianState
from .moments import estimate_moments
from .theory import check_order, log_dimension


@dataclass(frozen=True)
class GaussianCoefficientMeasure:
    basis: FockBasis
    k: int
    u: np.ndarray
    reference_measure: str = "normalized invariant four-mode even Gaussian-ray Haar measure"

    def __post_init__(self):
        self.basis.require_four_mode_even()
        check_order(self.k)
        u = self.basis.vector(self.u)
        u.flags.writeable = False
        object.__setattr__(self, "u", u)

    def _overlap(self, g):
        if not isinstance(g, GaussianState) or g.basis != self.basis:
            raise ValueError("Expected GaussianState in the same basis")
        return np.vdot(g.amplitudes, self.u)

    def log_density(self, g):
        a = abs(self._overlap(g))
        return -math.inf if a == 0 else log_dimension(self.basis.n, self.k)+(2*self.k-1)*math.log(a)

    def density(self, g):
        return math.exp(self.log_density(g))

    def aligned_state(self, g):
        a = self._overlap(g)
        # Zero-density rays may use any phase.
        return GaussianState(self.basis, g.amplitudes if a == 0 else (a/abs(a))*g.amplitudes)

    def probability_density_estimate(self, g, *, mass_estimate):
        """Estimated normalized density; the supplied estimated mass is explicit."""
        if not math.isfinite(mass_estimate) or mass_estimate <= 0:
            raise ValueError("A positive finite coefficient-mass estimate is required")
        return self.density(g)/mass_estimate

    def estimate_mass(self, sampler, count, *, batch_size=8192, check=None):
        estimate = estimate_moments(self.u, self.k, sampler, count,
                                    batch_size=batch_size, check=check)
        return {"estimate": estimate.mass, "standard_error": estimate.mass_standard_error,
                "samples": estimate.samples, "is_rigorous_interval": False}
