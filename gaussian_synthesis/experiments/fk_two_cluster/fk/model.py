"""Eight complex modes, two SO(7)-symmetric FK clusters, with V=1."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .conventions import (clifford, inner, majoranas, matmul, matvec,
                          parity_masks, quadratic_operator)


@dataclass
class FKModel:
    even_masks: np.ndarray
    gammas: np.ndarray
    q_ops: np.ndarray
    h_int: np.ndarray
    h_hop: np.ndarray
    S: np.ndarray
    j_matrices: np.ndarray
    local_gammas: np.ndarray
    local_q_ops: np.ndarray
    local_interaction: np.ndarray
    local_magic: np.ndarray
    mass_correlation_operator: np.ndarray
    reduced_interaction: np.ndarray
    reduced_hopping: np.ndarray

    def full_hamiltonian(self, t):
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        return self.h_int+t*self.h_hop

    def target(self, t):
        """Normalized ground state in the even 128-dimensional coordinates."""
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        _, vectors = eigh(self.reduced_interaction+t*self.reduced_hopping)
        coordinates = vectors[:, 0]
        state = matvec(self.S, coordinates)
        # A stable phase convention relative to the product magic reference.
        overlap = inner(self.S[:, 0], state)
        if abs(overlap) > 1e-14:
            state *= np.conjugate(overlap)/abs(overlap)
        return state/np.linalg.norm(state)

    def embed(self, state):
        state = np.asarray(state, dtype=np.complex128)
        if state.shape == (256,):
            return state
        if state.shape != (128,):
            raise ValueError("state must have 128 even amplitudes or 256 full amplitudes")
        result = np.zeros(256, dtype=np.complex128)
        result[self.even_masks] = state
        return result


def build_model():
    """Construct independently from local operators, then form the Krylov basis."""
    j_matrices = clifford()
    local_gammas = majoranas(4)
    local_q = np.array([quadratic_operator(j, local_gammas) for j in j_matrices])
    local_interaction = -sum(matmul(q, q) for q in local_q)
    _, local_eigenvectors = eigh(local_interaction)
    magic = local_eigenvectors[:, 0].copy()
    pivot = np.argmax(np.abs(magic))
    magic *= np.conjugate(magic[pivot])/abs(magic[pivot])
    identity = np.eye(16)
    local_parity = np.diag([(-1)**m.bit_count() for m in range(16)])
    # A is the low bit factor: np.kron(B, A).
    gammas = np.array([np.kron(identity, g) for g in local_gammas]
                       + [np.kron(g, local_parity) for g in local_gammas])
    q_ops = np.array([[np.kron(identity, q) for q in local_q],
                      [np.kron(q, identity) for q in local_q]])
    h_int = np.kron(identity, local_interaction)+np.kron(local_interaction, identity)
    h_hop = .5j*sum(np.kron(g, matmul(g, local_parity)) for g in local_gammas)
    mass_operator = sum(np.kron(q, q) for q in local_q)
    masks = parity_masks(8)
    h0_even = h_int[np.ix_(masks, masks)]
    hop_even = h_hop[np.ix_(masks, masks)]
    s0 = np.kron(magic, magic)[masks]
    s0 /= np.linalg.norm(s0)
    s1 = matvec(hop_even, s0)
    s1 -= s0*inner(s0, s1)
    s1 /= np.linalg.norm(s1)
    s2 = matvec(hop_even, s1)
    for _ in range(2):
        s2 -= s0*inner(s0, s2)+s1*inner(s1, s2)
    s2 /= np.linalg.norm(s2)
    basis = np.column_stack([s0, s1, s2])
    reduced_interaction = matmul(matmul(basis.conj().T, h0_even), basis)
    reduced_hopping = matmul(matmul(basis.conj().T, hop_even), basis)
    return FKModel(masks, gammas, q_ops, h_int, h_hop, basis, j_matrices,
                   local_gammas, local_q, local_interaction, magic, mass_operator,
                   reduced_interaction, reduced_hopping)
