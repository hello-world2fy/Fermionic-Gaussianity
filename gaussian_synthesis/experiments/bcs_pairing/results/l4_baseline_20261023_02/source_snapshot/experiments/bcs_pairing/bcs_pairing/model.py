"""Reduced BCS model in the fixed-number seniority-zero sector."""

from dataclasses import dataclass, field
import numpy as np
from .sectors import PairBasis


@dataclass(frozen=True)
class BCSModel:
    basis: PairBasis
    energies: np.ndarray
    H0: np.ndarray = field(init=False, repr=False, compare=False)
    K: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        eps = np.asarray(self.energies)
        if np.iscomplexobj(eps) or eps.shape != (self.basis.L,) or not np.all(np.isfinite(eps)):
            raise ValueError('energies must be a finite real array with length L')
        eps = np.array(eps, dtype=float, copy=True)
        h0 = np.diag(2*self.basis.occupations @ eps-eps.sum())
        coupling = np.eye(self.basis.dimension)*self.basis.m
        indices = {mask: i for i, mask in enumerate(self.basis.pair_masks)}
        for col, mask in enumerate(self.basis.pair_masks):
            for j in range(self.basis.L):
                if mask & (1 << j):
                    for i in range(self.basis.L):
                        if not mask & (1 << i):
                            coupling[indices[mask ^ (1 << j) ^ (1 << i)], col] = 1
        for array in (eps, h0, coupling):
            array.flags.writeable = False
        object.__setattr__(self, 'energies', eps)
        object.__setattr__(self, 'H0', h0)
        object.__setattr__(self, 'K', coupling)

    def hamiltonian(self, g):
        if not np.isscalar(g) or not np.isreal(g) or not np.isfinite(g):
            raise ValueError('g must be a finite real scalar')
        return self.H0-float(g)*self.K

    def dicke(self):
        return np.ones(self.basis.dimension, dtype=complex)/np.sqrt(self.basis.dimension)


def uniform_model(L=4, m=2, bandwidth=1.):
    if L < 2 or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Uniform nondegenerate spectra require L>=2 and positive bandwidth')
    return BCSModel(PairBasis(L, m), np.linspace(-bandwidth/2, bandwidth/2, L))
