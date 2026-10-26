"""Signed Gaussian operator convex synthesis via LP, dual witnesses, atom extraction.

Primal:  R_dict(rho) = min sum_i |w_i|  s.t.  rho = sum_i w_i |g_i><g_i|
Dual :  R(rho)      = sup { Tr(W rho) : Tr(W |g><g|) <= 1 for ALL Gaussian g }.

The dictionary LP gives an UPPER bound on the true continuous R (any dictionary
gives an achievable decomposition).  A witness valid on the FULL continuous
Gaussian manifold gives a LOWER bound.  Agreement = certification.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .gaussian import density_matrix, EVEN_OCC


def flatten_hermitian(H):
    """Hermitian 8x8 -> real 64-vector: diag(8), upper-tri Re(28), upper-tri Im(28).

    Elementwise equality of two such vectors is exactly Hermitian matrix equality
    (diagonal real; off-diagonal (i,j) and (j,i) conjugate).  The Hilbert-Schmidt
    inner product carries the factor 2 on off-diagonals separately.
    """
    H = np.asarray(H, dtype=np.complex128)
    n = H.shape[0]
    diag = np.diag(H).real
    tri_r, tri_im = [], []
    for a in range(n):
        for b in range(a + 1, n):
            tri_r.append(H[a, b].real)
            tri_im.append(H[a, b].imag)
    return np.concatenate([diag, np.array(tri_r), np.array(tri_im)])


def unflatten_hermitian(v):
    v = np.asarray(v, dtype=np.float64)
    n = 8
    H = np.zeros((n, n), dtype=np.complex128)
    H[np.diag_indices(n)] = v[:n]
    idx = n
    for a in range(n):
        for b in range(a + 1, n):
            H[a, b] = v[idx] + 1j * v[idx + n * (n - 1) // 2]
            H[b, a] = np.conj(H[a, b])
            idx += 1
    return H


def solve_signed_lp(rho, atoms, *, atol=1e-9):
    """Minimize TV norm of weights reproducing `rho` as sum w_i |g_i><g_i|.

    atoms: (M, 8, 8) array of Gaussian density matrices.
    Returns dict with value, weights, residual, n_nonzero, status.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    atoms = np.asarray(atoms, dtype=np.complex128)
    M = atoms.shape[0]
    # FAIL-CLOSED audit entry: every ORIGINAL atom must be a legal pure even-Gaussian
    # projector (Hermitian/trace/purity/PSD/rank-1/outer), checked BEFORE any flatten/LP.
    from .audit import audit_projector
    for i in range(M):
        audit_projector(atoms[i], label=f"solver_atom#{i}")
    b = flatten_hermitian(rho)
    # column i of A = flatten(atom_i)
    A = np.stack([flatten_hermitian(atoms[i]) for i in range(M)], axis=1)  # (64, M)
    # variables x = [w+ ; w-], min 1^T(w+ + w-)
    c = np.ones(2 * M)
    A_eq = np.hstack([A, -A])
    bounds = [(0.0, None)] * (2 * M)
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        return {"status": "lp_failed", "message": res.message, "value": np.inf}
    w = res.x[:M] - res.x[M:]
    value = float(np.sum(np.abs(w)))
    recon = np.tensordot(w, atoms, axes=(0, 0))
    resid = np.linalg.norm(flatten_hermitian(recon - rho))
    frob = float(np.linalg.norm(recon - rho))   # FULL 8x8 Frobenius residual
    nz = int(np.sum(np.abs(w) > atol))
    return {
        "status": "ok",
        "value": value,
        "weights": w,
        "residual": float(resid),
        "frobenius_residual": frob,
        "upper_valid": bool(frob < 1e-8),
        "n_nonzero": nz,
        "n_atoms": M,
        "linprog_status": res.status,
        "linprog_msg": res.message,
    }


def witness_continuous_sup_grid(witness_fn, rng, n_grid, n_random=0):
    """Estimate sup_g Tr(W sigma_g) over even Gaussian rays.

    Structured part: the product family g(t,t) = [(|00>+t|11>)_{01}(|00>+t|11>)_{23}]/(1+t^2)
    which saturates the magic witness.  Random part: uniform even rays.
    This is a FINITE-POOL upper estimate, NOT a continuous certificate.
    witness_fn(ket) -> Tr(W |ket><ket|).
    """
    best = -np.inf
    ts = np.linspace(0.0, 8.0, n_grid)
    for t in ts:
        a = 1.0 / (1.0 + t ** 2)
        ket = np.zeros(8, dtype=np.complex128)
        ket[0] = a            # 0000
        ket[1] = a * t        # 0011 = modes 0,1
        ket[6] = a * t        # 1100 = modes 2,3
        ket[7] = a * t ** 2   # 1111
        best = max(best, float(np.real(witness_fn(ket))))
    if n_random:
        from .gaussian import sample_even_gaussian_rays
        rays = sample_even_gaussian_rays(rng, n_random)
        for ket in rays:
            best = max(best, float(np.real(witness_fn(ket))))
    return best
