"""Exact fixed-number, seniority-zero reduced BCS Hamiltonian.

L pair levels = 2L physical fermionic modes; N=L physical particles.
All energy and interaction conventions follow RESEARCH_BRIEF.txt.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply


@dataclass
class PairModel:
    energies: np.ndarray
    G: float

    def __post_init__(self):
        self.energies = np.asarray(self.energies, dtype=float)
        self.L = len(self.energies)
        if self.L % 2 or self.L < 4 or not np.all(np.diff(self.energies) >= 0):
            raise ValueError('Need an even number of ordered pair levels >=4')
        self.r = self.L // 2
        self.g = self.G / self.L
        self.masks = np.array([sum(1 << j for j in c)
                              for c in combinations(range(self.L), self.r)], dtype=np.int64)
        self.index = {int(x): i for i, x in enumerate(self.masks)}
        self.occ = ((self.masks[:, None] >> np.arange(self.L)) & 1).astype(float)
        self.m = self.occ[:, self.r:].sum(axis=1).astype(int)
        self.dim = len(self.masks)
        self.fermi_index = self.index[(1 << self.r)-1]
        self.initial = np.zeros(self.dim, dtype=complex)
        self.initial[self.fermi_index] = 1
        self.transitions = {}
        rows, cols = [], []
        for i in range(self.L):
            for j in range(self.L):
                if i == j:
                    continue
                src = np.flatnonzero((self.occ[:, j] == 1) & (self.occ[:, i] == 0))
                dst = np.array([self.index[int(self.masks[x]) ^ (1 << i) ^ (1 << j)]
                                for x in src])
                self.transitions[i, j] = (dst, src)
                rows.extend(dst)
                cols.extend(src)
        hopping = coo_matrix((-self.g*np.ones(len(rows)), (rows, cols)),
                             shape=(self.dim, self.dim)).tocsr()
        self.diagonal = 2*self.occ @ self.energies-self.energies.sum()-self.g*self.r
        self.H = hopping+diags(self.diagonal)
        self.blocks = []
        self.permanent_recursion = []
        # Hole indices retain the original lower-band ordering.
        for m in range(self.r+1):
            subsets = list(combinations(range(self.r), m))
            indices = np.empty((len(subsets), len(subsets)), dtype=int)
            for a, p in enumerate(subsets):
                for b, h in enumerate(subsets):
                    mask = (1 << self.r)-1
                    for j in h:
                        mask ^= 1 << j
                    for j in p:
                        mask ^= 1 << (j+self.r)
                    indices[a, b] = self.index[mask]
            self.blocks.append((subsets, indices))
            if m:
                current = indices.ravel()
                particle = np.repeat([c[0] for c in subsets], len(subsets))
                holes = np.tile(np.array(subsets), (len(subsets), 1))
                parents = np.array([[self.index[int(self.masks[x]) ^ (1 << (p+self.r)) ^ (1 << int(h))]
                                     for h in hs] for x, p, hs in zip(current, particle, holes)])
                self.permanent_recursion.append((current, particle, holes, parents))

    def evolve(self, times, initial=None):
        times = np.asarray(times)
        initial = self.initial if initial is None else initial
        if len(times) > 1 and np.allclose(np.diff(times), np.diff(times)[0]):
            # Start the interval at zero. scipy 1.17.1's nonzero-start interval
            # path can lose normalization when start is much larger than step.
            if times[0] != 0:
                initial = expm_multiply(-1j*times[0]*self.H, initial,
                                        traceA=-1j*times[0]*self.diagonal.sum())
            return expm_multiply(-1j*self.H, initial, start=0, stop=times[-1]-times[0],
                                 num=len(times), traceA=-1j*self.diagonal.sum())
        return np.array([expm_multiply(-1j*t*self.H, initial,
                         traceA=-1j*t*self.diagonal.sum()) for t in times])

    def observe(self, state):
        probability = abs(state)**2
        count = np.bincount(self.m, weights=probability, minlength=self.r+1)
        mean = float(probability @ self.m)
        variance = float(probability @ (self.m**2)-mean**2)
        coherence = 0j
        for p in range(self.r, self.L):
            for h in range(self.r):
                dst, src = self.transitions[p, h]
                coherence += np.vdot(state[dst], state[src])/self.r
        return {'mean': mean, 'variance': variance, 'excess': variance-mean,
                'fano': variance/mean if mean > 1e-20 else 1.,
                'cross_real': float(coherence.real), 'cross_imag': float(coherence.imag),
                'count': count.tolist()}

    def pair_density(self, state):
        matrix = np.diag(abs(state)**2 @ self.occ).astype(complex)
        for (i, j), (dst, src) in self.transitions.items():
            matrix[i, j] = np.vdot(state[dst], state[src])
        return matrix


def picket_fence(L, G, half_bandwidth=1.):
    return PairModel(half_bandwidth*(2*(np.arange(L)+.5)/L-1), G)
