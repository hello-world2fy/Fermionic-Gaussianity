"""Explicit Fock conventions for the isolated two-cluster FK experiment.

Mode j is bit j (little endian). A occupies modes 0..3, B modes 4..7.
gamma[2*j] = c_j + c_j^dagger and gamma[2*j+1] = i(c_j^dagger-c_j).
The covariance is Gamma_ab = i <gamma_a gamma_b> for a != b, so the
vacuum has Gamma[2*j, 2*j+1] = -1.
"""
from functools import lru_cache

import numpy as np


def parity_masks(n, parity=0):
    """Sorted occupation masks with the specified fermionic parity."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if parity not in (0, 1):
        raise ValueError("parity must be 0 (even) or 1 (odd)")
    return np.array([m for m in range(1 << int(n)) if m.bit_count() % 2 == parity],
                    dtype=np.int64)


def majoranas(n):
    """Dense Hermitian Jordan-Wigner matrices in little-endian bit order."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    dimension = 1 << int(n)
    result = np.zeros((2*n, dimension, dimension), dtype=np.complex128)
    for mode in range(n):
        low_mask = (1 << mode)-1
        for mask in range(dimension):
            sign = -1 if (mask & low_mask).bit_count() % 2 else 1
            occupied = (mask >> mode) & 1
            flipped = mask ^ (1 << mode)
            result[2*mode, flipped, mask] = sign
            result[2*mode+1, flipped, mask] = sign * (-1j if occupied else 1j)
    return result


@lru_cache(maxsize=1)
def _clifford():
    # Left multiplication by the seven imaginary octonion units. Alternativity
    # gives J_a J_b + J_b J_a = -2 delta_ab, verified again by the model tests.
    result = np.zeros((7, 8, 8), dtype=float)
    oriented_triples = ((1,2,3), (1,4,5), (1,7,6), (2,4,6),
                        (2,5,7), (3,4,7), (3,6,5))
    for a in range(1, 8):
        result[a-1, a, 0] = 1
        result[a-1, 0, a] = -1
        for x, y, z in oriented_triples:
            for u, v, w in ((x,y,z), (y,z,x), (z,x,y)):
                if a == u:
                    result[a-1, w, v] = 1
                if a == v:
                    result[a-1, w, u] = -1
    result.setflags(write=False)
    return result


def clifford():
    """Return a writable copy of the seven real antisymmetric Clifford matrices."""
    return _clifford().copy()


def matmul(left, right):
    """Small deterministic contraction, without host-specific BLAS warnings."""
    return np.einsum("ij,jk->ik", left, right, optimize=False)


def matvec(matrix, vector):
    return np.einsum("ij,j->i", matrix, vector, optimize=False)


def inner(left, right):
    return np.einsum("i,i->", np.conjugate(left), right, optimize=False)


def quadratic_operator(real_antisymmetric, gammas):
    """i/4 gamma^T J gamma; only nonzero upper triangular entries are used."""
    dimension = gammas.shape[1]
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for a in range(len(gammas)):
        for b in range(a+1, len(gammas)):
            if real_antisymmetric[a, b] != 0:
                result += .5j * real_antisymmetric[a, b] * matmul(gammas[a], gammas[b])
    return result
