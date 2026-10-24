"""Additional exact antiunitary symmetry of the physical real FK targets.

T = -P_A K acts as complex conjugation on the three singlet coordinates.
Both occupation conjugation and local fermionic parity preserve Gaussian
states and invariant measure. Unique minimizers for T-invariant targets are
therefore fixed by T, i.e. their z coordinates are real. Pairing an empirical
quadrature with its T image preserves this exact symmetry without assuming a
Gaussian contact family. It cannot remove scalar normalization uncertainty.
"""
import numpy as np

from .weighted import Value


def time_reverse(kets):
    masks = [m for m in range(256) if m.bit_count()%2 == 0]
    signs = np.array([-(-1)**((m & 15).bit_count()) for m in masks])
    return np.asarray(kets).conj()*signs


class TimeReversalAveraged:
    def __init__(self, backend):
        self.backend = backend
        self.k = backend.k
        d = backend.h.shape[1]
        self.signs = np.r_[np.ones(d), -np.ones(d)]

    def evaluate(self, z, hessian=True):
        a = self.backend.evaluate(z, hessian=hessian)
        b = a if np.all(np.imag(z) == 0) else self.backend.evaluate(np.conj(z), hessian=hessian)
        gradient = .5*(a.gradient+self.signs*b.gradient)
        hes = None if not hessian else .5*(a.hessian+self.signs[:, None]*b.hessian*self.signs[None, :])
        return Value(.5*(a.q+b.q), gradient, hes, .5*(a.mass+b.mass), min(a.ess, b.ess))

    def objective(self, z, target):
        return self.evaluate(z, hessian=False).q/(2*self.k)-np.vdot(target, z).real

