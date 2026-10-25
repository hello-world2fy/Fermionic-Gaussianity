"""Frozen collective Gaussian spans and Hamiltonian-only propagation.

q particle and q hole orbitals generate number-projected onsite Gaussians
with arbitrary coefficients in these orbitals. Expanding their polynomial
in those coefficients yields a finite span. Only a training-time matrix
sets the orbitals; all later states come from the projected Hamiltonian.
These are physically guided compressions of a witness, NOT samples from its
canonical measure. This distinction is part of the saved metadata.
"""
from itertools import product
import numpy as np
from .structure import collective_basis


def occupations(total, q):
    if q == 1:
        return [(total,)]
    return [(a,)+rest for a in range(total+1) for rest in occupations(total-a, q-1)]


def band_span(subsets, orbitals):
    q = orbitals.shape[1]
    m = len(subsets[0])
    powers = occupations(m, q)
    matrix = np.zeros((len(subsets), len(powers)), complex)
    for row, subset in enumerate(subsets):
        polynomial = {(0,)*q: 1.+0j}
        for j in subset:
            next_poly = {}
            for power, coefficient in polynomial.items():
                for a in range(q):
                    new = tuple(x+(i == a) for i, x in enumerate(power))
                    next_poly[new] = next_poly.get(new, 0j)+coefficient*orbitals[j, a]
            polynomial = next_poly
        matrix[row] = [polynomial.get(power, 0j) for power in powers]
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, s > s[0]*1e-10]


def channel_basis(model, T, q):
    u, singular, vh = np.linalg.svd(T, full_matrices=False)
    particle, hole = u[:, :q], vh[:q].T
    blocks = []
    for m, (subsets, indices) in enumerate(model.blocks):
        p = band_span(subsets, particle)
        h = band_span(subsets, hole)
        local = np.einsum('ia,jb->ijab', p, h).reshape(len(subsets)**2, -1)
        block = np.zeros((model.dim, local.shape[1]), complex)
        block[indices.ravel()] = local
        blocks.append(block)
    return np.concatenate(blocks, axis=1), singular


def propagate(H_reduced, initial_coefficients, times):
    energy, vectors = np.linalg.eigh(H_reduced)
    weights = vectors.conj().T@initial_coefficients
    return (np.exp(-1j*np.asarray(times)[:, None]*energy)*weights)@vectors.T


def count_and_coherence(model, states):
    rows = []
    for state in states:
        obs = model.observe(state)
        rows.append(obs)
    return rows
