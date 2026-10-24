"""Matrix-free moments from an explicitly normalized importance proposal."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass
class Value:
    q: float
    gradient: np.ndarray
    hessian: np.ndarray | None
    mass: float
    ess: float


class WeightedMoments:
    def __init__(self, overlaps, log_measure_weights, k, log_dimension):
        self.h = np.asarray(overlaps, complex)
        self.log_weights = np.asarray(log_measure_weights, float)
        if self.log_weights.shape != (len(self.h),):
            raise ValueError("one log measure weight per atom required")
        self.k = int(k)
        self.log_d = float(log_dimension)
        self.jr = np.concatenate((self.h.real, -self.h.imag), axis=1)
        self.ji = np.concatenate((self.h.imag, self.h.real), axis=1)

    def evaluate(self, z, hessian=True):
        a = np.einsum("ni,i->n", self.h, z, optimize=False)
        r = abs(a)**2
        log_abs = np.log(np.maximum(abs(a), 1e-300))
        log_w = self.log_d+self.log_weights+2*self.k*log_abs
        q = float(np.exp(logsumexp(log_w)))
        weights = np.exp(log_w)
        # d|a|^(2k-2) avoids a singular ratio for exactly zero overlaps.
        base = np.exp(self.log_d+self.log_weights+(2*self.k-2)*log_abs)
        b = a.real[:, None]*self.jr+a.imag[:, None]*self.ji
        gradient = base @ b
        hes = None
        if hessian:
            hes = self.jr.T @ (base[:, None]*self.jr)+self.ji.T @ (base[:, None]*self.ji)
            if self.k > 1:
                curvature = np.exp(self.log_d+self.log_weights+(2*self.k-4)*log_abs)
                hes += 2*(self.k-1)*b.T @ (curvature[:, None]*b)
        mass = float(np.exp(logsumexp(self.log_d+self.log_weights+(2*self.k-1)*log_abs)))
        ess = float(q*q/np.sum(weights**2)) if q else 0.
        return Value(q, gradient, hes, mass, ess)

    def objective(self, z, target):
        return self.evaluate(z, hessian=False).q/(2*self.k)-float(np.vdot(target, z).real)

