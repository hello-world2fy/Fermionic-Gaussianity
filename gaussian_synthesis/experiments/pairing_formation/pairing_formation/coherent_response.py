"""Full k2 coherent response geometry inside a declared symmetry sector.

A is linear Hermitian positive and B is an antilinear symmetric matrix:
DC(u)[v] = 2 A v + B v*. Neither their eigenvectors nor their eigenvalues
are Gaussian atoms or canonical probability samples.
"""
import numpy as np
from scipy.linalg import null_space


def complement(base):
    return null_space(np.asarray(base).conj().T, rcond=1e-11)


def response_modes(A, B, base):
    """Basis-invariant leading modes and coupling to an existing subspace."""
    Z = complement(base)
    reduced = Z.conj().T @ A @ Z
    values, vectors = np.linalg.eigh((reduced + reduced.conj().T)/2)
    order = np.argsort(values)[::-1]
    anomalous = Z.conj().T @ B @ Z.conj()
    U, s, _ = np.linalg.svd(anomalous, full_matrices=False)
    coupling = Z.conj().T @ np.column_stack((A @ base, B @ base.conj()))
    V, t, _ = np.linalg.svd(coupling, full_matrices=False)
    return {'eigen': Z @ vectors[:, order], 'anomalous': Z @ U,
            'cross': Z @ V}, {'eigen': values[order], 'anomalous': s, 'cross': t}


def ordered_krylov(H, initial, base, count):
    """Ordinary H-Krylov directions added to a shared space, stably."""
    current = np.asarray(initial).copy()
    current /= np.linalg.norm(current)
    arnoldi = [current.copy()]
    directions = []
    for _ in range(len(initial)-1):
        current = H @ current
        for _ in range(2):
            for v in arnoldi:
                current -= v * np.vdot(v, current)
        norm = np.linalg.norm(current)
        if norm < 1e-11:
            break
        current /= norm
        arnoldi.append(current.copy())
        residual=current.copy()
        for _ in range(2):
            residual -= base @ (base.conj().T @ residual)
            for v in directions:
                residual -= v*np.vdot(v,residual)
        norm=np.linalg.norm(residual)
        if norm>1e-11:
            directions.append(residual/norm)
        if len(directions)==count:
            break
    return np.column_stack(directions) if directions else np.empty((len(initial), 0), complex)
