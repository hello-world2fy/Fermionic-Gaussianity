import bootstrap
from dataclasses import dataclass
import numpy as np
from gaussian_synthesis import FockBasis
from gaussian_synthesis.gaussian import quadric_matrix

BASIS = FockBasis(4, 'even')
J = quadric_matrix()
MASKS = np.array(BASIS.occupations)
# Modes: d_up, bath_up, d_down, bath_down. Fock creation order ascending.
PHYSICAL = np.array([[BASIS.index(5), BASIS.index(9)],
                     [BASIS.index(6), BASIS.index(10)]])


def quadratic(h):
    matrix = np.zeros((8, 8), complex)
    for col, mask in enumerate(BASIS.occupations):
        for j in range(4):
            if not mask & (1 << j):
                continue
            removed = mask ^ (1 << j)
            for i in range(4):
                if removed & (1 << i):
                    continue
                parity = (mask & ((1 << j)-1)).bit_count() + (removed & ((1 << i)-1)).bit_count()
                row = BASIS.index(removed | (1 << i))
                matrix[row, col] += (-1)**parity*h[i, j]
    return matrix


def unitary(h, time):
    energies, vectors = np.linalg.eigh(h)
    return (vectors*np.exp(-1j*energies*time)) @ vectors.conj().T


@dataclass
class GaussianSum:
    atoms: np.ndarray  # columns, normalized and quadric-null
    coefficients: np.ndarray

    def vector(self):
        return self.atoms @ self.coefficients

    @property
    def count(self):
        return len(self.coefficients)

    def cost(self):
        vector = self.vector()
        return float(np.sum(np.abs(self.coefficients))**2/np.vdot(vector, vector).real)


class Anderson:
    def __init__(self, interaction=4., hopping=.6, impurity_energy=0., dt=.05):
        self.interaction = interaction
        self.dt = dt
        h = np.array([[impurity_energy, hopping], [hopping, 0.]], complex)
        self.h = np.zeros((4, 4), complex)
        self.h[:2, :2] = self.h[2:, 2:] = h
        self.free = quadratic(self.h)
        up = ((MASKS >> 0) & 1).astype(float)
        down = ((MASKS >> 2) & 1).astype(float)
        self.double = up*down
        self.density = up+down
        self.zz = (1-2*up)*(1-2*down)
        self.hamiltonian = self.free + np.diag(interaction*self.zz/4)
        self.half = unitary(self.free, dt/2)
        self.phase = np.exp(-1j*interaction*self.zz*dt/4)
        self.strang = self.half @ (self.phase[:, None]*self.half)
        _, eig = np.linalg.eigh(h)
        self.initial = np.zeros(8, complex)
        self.initial[PHYSICAL] = np.outer(eig[:, 0], eig[:, 0])
        self.physical_h = self.hamiltonian[np.ix_(PHYSICAL.ravel(), PHYSICAL.ravel())]

    def initial_sum(self):
        return GaussianSum(self.initial[:, None].copy(), np.ones(1, complex))

    def branch(self, state):
        atoms = self.half @ state.atoms
        theta = self.interaction*self.dt/4
        if abs(np.sin(theta)) < 1e-15:
            return GaussianSum(self.half @ atoms, np.cos(theta)*state.coefficients)
        return GaussianSum(self.half @ np.concatenate([atoms, self.zz[:, None]*atoms], axis=1),
                            np.concatenate([np.cos(theta)*state.coefficients,
                                            -1j*np.sin(theta)*state.coefficients]))


def schmidt_decompose(vector):
    matrix = vector[PHYSICAL]
    left, values, right_h = np.linalg.svd(matrix)
    atoms = np.zeros((8, 2), complex)
    for i in range(2):
        atoms[PHYSICAL, i] = np.outer(left[:, i], right_h[i])
    keep = values > 1e-13
    return GaussianSum(atoms[:, keep], values[keep].astype(complex))


def gaussian_defect(atoms):
    return float(np.max(np.abs(np.einsum('ia,ij,ja->a', atoms, J, atoms))))
