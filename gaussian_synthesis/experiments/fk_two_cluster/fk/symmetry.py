"""Compute the entire diagonal Spin(7) fixed space, not just a Krylov orbit."""
import numpy as np
from scipy.linalg import eigh

from .conventions import matmul, quadratic_operator


def diagonal_generators(model):
    """The 21 Hermitian infinitesimal generators, restricted to global even parity."""
    generators = []
    identity = np.eye(16)
    for a in range(7):
        for b in range(a+1, 7):
            rotation = matmul(model.j_matrices[a], model.j_matrices[b])
            local = quadratic_operator(rotation, model.local_gammas)
            diagonal = np.kron(identity, local)+np.kron(local, identity)
            generators.append(diagonal[np.ix_(model.even_masks, model.even_masks)])
    return np.array(generators)


def fixed_space(model, tolerance=1e-10):
    """Common kernel of all generators via their positive Gram operator."""
    generators = diagonal_generators(model)
    gram = np.einsum("gji,gjk->ik", generators.conj(), generators, optimize=False)
    eigenvalues, eigenvectors = eigh(gram)
    threshold = tolerance*max(1., float(np.max(np.abs(eigenvalues))))
    return eigenvectors[:, np.abs(eigenvalues) <= threshold], eigenvalues, generators


def symmetry_report(model, tolerance=1e-10):
    basis, eigenvalues, generators = fixed_space(model, tolerance)
    projector = matmul(basis, basis.conj().T)
    krylov_projector = matmul(model.S, model.S.conj().T)
    action = np.einsum("gij,jk->gik", generators, model.S, optimize=False)
    scale = max(1., float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues[eigenvalues > tolerance*scale]
    return {
        "fixed_space_dimension": int(basis.shape[1]),
        "generator_count": int(len(generators)),
        "ambient_dimension": 128,
        "kernel_tolerance": float(tolerance*scale),
        "smallest_positive_gram_eigenvalue": float(positive[0]) if len(positive) else None,
        "fixed_space_krylov_projector_error": float(np.linalg.norm(projector-krylov_projector)),
        "generator_singlet_residual": float(np.linalg.norm(action)),
        "verified": bool(basis.shape[1] == 3
                         and np.linalg.norm(projector-krylov_projector) < tolerance
                         and np.linalg.norm(action) < tolerance),
        "scope": "numerical common-kernel check in the complete global-even sector",
    }
