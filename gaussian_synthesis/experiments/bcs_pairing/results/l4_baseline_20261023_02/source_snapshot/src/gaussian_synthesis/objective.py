"""Real-Hilbert gradient: dJ(u)[v] = Re <C(u)-psi | v>."""

from dataclasses import dataclass
import numpy as np
from .moments import batch_moments
from .theory import check_order


@dataclass
class ConvexObjective:
    psi: np.ndarray
    k: int
    n: int = 4

    def __post_init__(self):
        check_order(self.k)
        self.psi = np.array(self.psi, dtype=np.complex128, copy=True)

    def value_gradient(self, u, samples=None):
        if self.k == 1:
            return (float(np.vdot(u, u).real/2-np.vdot(u, self.psi).real),
                    u-self.psi)
        if samples is None:
            raise ValueError("k>1 requires a fresh Gaussian batch")
        estimate = batch_moments(u, self.k, samples, n=self.n)
        value = estimate.q/(2*self.k)-np.vdot(u, self.psi).real
        return float(value), estimate.C-self.psi
