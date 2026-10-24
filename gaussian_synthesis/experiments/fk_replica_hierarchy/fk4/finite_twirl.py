"""Finite Spin(7) twirl for the entire even eight-mode Fock representation.

Under diagonal Spin(7), S_+(16)=3*1+3*7+2*21+27+35. All nontrivial
summands occur in vector tensors of degrees at most three. The 64 even sign
changes kill vector and exterior-power components; seven cyclic permutations
average the diagonal traceless symmetric component to zero. Their 448 group
elements therefore have exactly the same average as Haar on this
representation. The implementation ALSO checks the complete 128x128 operator
against S S^dagger, rather than testing only selected atoms.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh

from . import paths
from fk.conventions import quadratic_operator
from fk2.orbits import DiagonalSpin7, simplex_directions
from fk3.discovery import covariance_from_kets


def _rotation(model, a, b, angle):
    q = quadratic_operator(model.j_matrices[a]@model.j_matrices[b], model.local_gammas)
    values, vectors = eigh(q)
    return (vectors*np.exp(-.5j*angle*values))@vectors.conj().T


class FiniteSpin7Twirl:
    def __init__(self, model):
        self.model = model
        signs = [np.eye(16, dtype=complex)]
        for j in range(1, 7):
            flip = _rotation(model, 0, j, np.pi)
            signs += [flip@u for u in signs]
        cycle = np.eye(16, dtype=complex)
        for j in range(6):
            cycle = _rotation(model, j, j+1, np.pi/2)@cycle
        group = []
        power = np.eye(16, dtype=complex)
        for _ in range(7):
            group.extend([u@power for u in signs])
            power = cycle@power
        self.group = np.array(group)
        idx = model.even_masks
        average = sum(np.kron(u, u)[np.ix_(idx, idx)] for u in group)/len(group)
        self.operator_error = float(np.linalg.norm(average-model.S@model.S.conj().T))
        self.unitarity_error = float(np.max(abs(self.group.conj().transpose(0, 2, 1)@self.group-np.eye(16))))
        if self.operator_error > 1e-10 or self.unitarity_error > 1e-10:
            raise ArithmeticError("finite Spin(7) group does not reproduce the full singlet projector")
        self.simplex = DiagonalSpin7(model)

    def orbit(self, ket, *, try_aligned_simplex=True):
        ket = np.asarray(ket)
        projected = self.model.S@(self.model.S.conj().T@ket)
        if np.linalg.norm(ket-projected) < 1e-11:
            return ket[None], {"method": "already_invariant", "count": 1,
                               "twirl_error": float(np.linalg.norm(ket-projected))}
        if try_aligned_simplex:
            cov = covariance_from_kets(ket)[0]
            mass = np.einsum('kab,ab->k', self.model.j_matrices, cov[:8, :8])/8
            if np.linalg.norm(mass) > 1e-8:
                u, _ = self.simplex.rotation(mass/np.linalg.norm(mass))
                matrix = self.model.embed(ket).reshape(16, 16)
                aligned = (u.conj().T@matrix@u.conj()).ravel()[self.model.even_masks]
                candidates = np.array([self.simplex.act(aligned, direction)[0] for direction in simplex_directions()])
                error = float(np.linalg.norm(candidates.mean(axis=0)-projected))
                if error < 1e-10:
                    return candidates, {"method": "aligned_verified_simplex", "count": 8, "twirl_error": error}
        matrix = self.model.embed(ket).reshape(16, 16)
        full = (self.group@matrix@self.group.transpose(0, 2, 1)).reshape(-1, 256)
        result = full[:, self.model.even_masks]
        error = float(np.linalg.norm(result.mean(axis=0)-projected))
        if error > 1e-10:
            raise ArithmeticError("finite full-group orbit failed its ket-level projection check")
        return result, {"method": "full_448_element_twirl", "count": len(result), "twirl_error": error}

