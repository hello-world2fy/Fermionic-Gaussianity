"""Non-orthogonal Slater determinant dictionary + the actual convex-J
(replica synthesis) optimization.

Correction of the earlier power heuristic
------------------------------------------
The paper's constructive step (dual_core.tex S6.1) is NOT ``p_a ∝ |r_a|^{2k}``.
It is a convex optimization over a *non-orthogonal* dictionary: maximize
Re<w,r> subject to  N_k(w) = ( sum_a |<a,w>|^{2k} )^{1/(2k)} <= 1.
KKT stationarity gives the atom-synthesis fixed point
    r = t * sum_a |z_a|^{2k-2} z_a * a,   z_a = <a,w>.
For an ORTHONORMAL dictionary this collapses exactly to c_i = r_i (the
``u_i = phase(r_i)|r_i|^{1/(2k-1)}`` fixed point), i.e. it carries no
selection information -- which is why an orthonormal Fock-basis "power"
reweighting was never a replica synthesis. The method only becomes
non-trivial on a NON-ORTHOGONAL Slater dictionary (rotated orbitals), where
the tilted atom sum is a genuine convex gauge mixture.

Dictionary restriction
----------------------
We build Slater determinants from spin-conserving occupied subspaces
(U_up, U_down), each an L x (L/2) orthonormal matrix in the site basis,
sampled as a small Grassmannian neighborhood of the HF/Neel determinant
(real, occupied-virtual perturbations). This is a FINITE local dictionary,
NOT the full continuous Grassmannian; residual components outside its span
are reported as projected residual, never claimed as reconstructed.

The sum is complex/signed (here real): coefficients c_a are not non-negative;
the convex quantity is the phase-balanced gauge tau = sum |c_a|.
"""
from __future__ import annotations

import itertools
import numpy as np

from hubbard import HubbardModel


# ---------------------------------------------------------------------------
# Slater determinants in the Fock basis
# ---------------------------------------------------------------------------
def _det_amplitudes(L: int, n: int, U: np.ndarray) -> dict:
    """For an (L x n) orthonormal occupied-orbital matrix, return
    {bitmask: det(U[rows,:])} over all C(L,n) occupied configurations."""
    out = {}
    for comb in itertools.combinations(range(L), n):
        rows = list(comb)
        sub = U[rows, :]
        out[sum(1 << i for i in rows)] = float(np.linalg.det(sub))
    return out


def slater_fock_vector(model: HubbardModel, U_up: np.ndarray,
                       U_down: np.ndarray) -> np.ndarray:
    """Build the full Sz=0 Fock-basis vector of a Slater determinant."""
    L = model.L
    up_amp = _det_amplitudes(L, model.n_up, U_up)
    dn_amp = _det_amplitudes(L, model.n_down, U_down)
    vec = np.zeros(model.dim)
    upmask = (1 << L) - 1
    for idx, mask in enumerate(model.states):
        mask = int(mask)
        sup = mask & upmask
        sdn = (mask >> L) & upmask
        vec[idx] = up_amp[sup] * dn_amp[sdn]
    nrm = np.linalg.norm(vec)
    return vec / nrm if nrm > 0 else vec


def reference_U(model: HubbardModel, ref_mask: int):
    """Occupied (L x n) matrices for a given Fock determinant: columns are
    the occupied site orbitals (standard basis)."""
    L = model.L
    up = np.zeros((L, model.n_up))
    dn = np.zeros((L, model.n_down))
    ucol = dcol = 0
    for i in range(L):
        if (int(ref_mask) >> i) & 1:
            up[i, ucol] = 1.0; ucol += 1
        if (int(ref_mask) >> (L + i)) & 1:
            dn[i, dcol] = 1.0; dcol += 1
    return up, dn


def orth(A: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(A)
    return q


def sample_dictionary(model: HubbardModel, ref_mask: int, n_atoms: int,
                      eps: float, seed: int):
    """Reference + (n_atoms-1) nearby non-orthogonal Slater determinants.
    Returns A (dim x n_atoms), normalized columns."""
    rng = np.random.default_rng(seed)
    Uup0, Udn0 = reference_U(model, ref_mask)
    cols = [slater_fock_vector(model, Uup0, Udn0)]
    for _ in range(n_atoms - 1):
        # small random occupied-subspace perturbation -> non-orthogonal det
        Uup = orth(Uup0 + eps * rng.standard_normal(Uup0.shape))
        Udn = orth(Udn0 + eps * rng.standard_normal(Udn0.shape))
        cols.append(slater_fock_vector(model, Uup, Udn))
    A = np.array(cols).T
    # renormalize columns
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    return A


# ---------------------------------------------------------------------------
# Convex-J (replica synthesis) KKT fixed point
# ---------------------------------------------------------------------------
def convex_j_synthesize(A: np.ndarray, r: np.ndarray, k: float = 2.0,
                        n_iter: int = 2000, tol: float = 1e-12):
    """Solve the UNCONSTRAINED strictly-convex primal
        J(u) = (1/(2k)) sum_a |<a,u>|^{2k} - <u,r>,
    analytic gradient  grad J(u) = C(u) - r,  C(u)=sum_a |z_a|^{2k-2} z_a a,
    z_a=<a,u>.  The KKT stationarity point is C(u*) = r exactly.

    Normalized-witness convention (so the returned numbers are comparable to
    the dual support value): let N=N_k(u*), w=u*/N, z=A^T w,
    c_a=|z_a|^{2k-2} z_a, t=<w,r>=N^{2k-1}. Then  r = t A c  holds exactly at
    the solution, and the honest KKT residual is ||r - t A c||/||r||.
    Atom weights c_a are SIGNED (complex superposition); the convex object is
    the phase-balanced gauge tau=sum|c_a|, NOT a non-negative mixture.

    Special cases: k=1 gives the frame normal equations A A^T u=r; on an
    orthonormal dictionary u*=sign(r)|r|^{1/(2k-1)} and A c = r.
    """
    from scipy.optimize import minimize

    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    dim, nat = A.shape
    G = A.T @ A                       # nat x nat Gram; u = A x lives in row(A)
    At_r = A.T @ r

    # J(x) = (1/2k) sum_a |(G x)_a|^{2k} - x^T A^T r ;  grad_x = G (|Gx|^{2k-2} Gx) - A^T r
    def J_and_grad(x):
        z = G @ x
        az = np.abs(z)
        cc = az ** (2 * k - 2) * z
        J = float(np.sum(az ** (2 * k)) / (2 * k) - x @ At_r)
        grad = G @ cc - At_r
        return J, grad

    x0 = np.linalg.lstsq(G + 1e-10 * np.eye(nat), At_r, rcond=None)[0]
    res = minimize(J_and_grad, x0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=n_iter, ftol=tol, gtol=1e-10))
    x = res.x
    u = A @ x                         # witness (un-normalized), in row(A)

    z_u = A.T @ u
    N = float(np.sum(np.abs(z_u) ** (2 * k)) ** (1.0 / (2 * k))) + 1e-300
    w = u / N
    z = A.T @ w
    c = np.abs(z) ** (2 * k - 2) * z            # normalized atom weights
    t = float(w @ r)                            # = N^{2k-1} = dual support value
    Ac = A @ c
    kkt_err = float(np.linalg.norm(r - t * Ac) / (np.linalg.norm(r) + 1e-30))
    synth = t * Ac
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    proj_residual = float(np.linalg.norm(r - A @ coef) / (np.linalg.norm(r) + 1e-30))
    return dict(w=w, z=z, c=c, synth=synth, t=t, kkt_err=kkt_err,
                proj_residual=proj_residual, iters=int(res.nit),
                span_dim=int(np.linalg.matrix_rank(A)), dual_obj=t,
                gauge=float(np.sum(np.abs(c))), J=float(res.fun))


# ---------------------------------------------------------------------------
# Non-orthogonal greedy baseline: 2x2 Ritz energy drop
# ---------------------------------------------------------------------------
def two_dim_ritz_drop(H: np.ndarray, psi: np.ndarray, E: float,
                      atom: np.ndarray) -> float:
    """Lowest variational energy of H in span{psi, atom} (NOT a drop).

    Orthonormalizes span{psi, atom} by QR, then takes the lowest eigenvalue of
    the Hermitian projected H = Q.conj().T @ H @ Q. Equivalent to the
    generalized eigenproblem H c = E S c but numerically stable. The current
    Rayleigh quotient is recomputed internally (passed E is diagnostic only).
    Complex-conjugate safe. Near-dependent (relative sigma-min < 1e-8) returns
    the current Rayleigh quotient (variational floor, no ridge artifact)."""
    psi = np.asarray(psi); atom = np.asarray(atom)
    if np.linalg.norm(psi) < 1e-14:
        raise ValueError("psi must be nonzero")
    psi = psi / np.linalg.norm(psi)
    if np.linalg.norm(atom) < 1e-14:
        return float(np.real(np.vdot(psi, H @ psi)))
    atom = atom / np.linalg.norm(atom)
    Ecur = float(np.real(np.vdot(psi, H @ psi)))
    # orthonormalize span{psi, atom} via QR, then Hermitian eigh (variational).
    W = np.column_stack([psi, atom])
    sv = np.linalg.svd(W, compute_uv=False)
    if sv[-1] / sv[0] < 1e-8:
        return Ecur
    Q, _ = np.linalg.qr(W)
    return float(np.real(np.linalg.eigvalsh(Q.conj().T @ H @ Q)[0]))


def _complement(Q0):
    """Orthonormal basis for the orthogonal complement of span(Q0)."""
    L, n = Q0.shape
    M = np.eye(L) - Q0 @ Q0.T
    U, s, _ = np.linalg.svd(M)
    return U[:, :L - n]   # largest singular values = virtual complement


def rotated_slater(model, Uup0, Udn0, th_up, th_dn):
    """Thouless rotation: occupied subspace Q0 rotated into virtual Qv via an
    exponentiated skew block K = [[0, Θ], [-Θ^T, 0]] (a genuine unitary)."""
    from scipy.linalg import expm

    def rot(Q0, Qv, th):
        nu = Q0.shape[1]; nv = Qv.shape[1]
        th = np.asarray(th, dtype=float).reshape(nu, nv)
        Q = np.hstack([Q0, Qv])          # L x L orthonormal
        K = np.zeros((nu + nv, nu + nv))
        K[:nu, nu:] = th
        K[nu:, :nu] = -th.T
        R = expm(K)                       # (nu+nv) x (nu+nv) orthogonal
        return Q @ R[:, :nu]
    return slater_fock_vector(model, rot(Uup0, _complement(Uup0), th_up),
                              rot(Udn0, _complement(Udn0), th_dn))


def orbital_optimize(model, H, psi, E, mask, deadline=None, seed=0):
    """Real spin-conserving orbital optimization: minimize the 2x2 Ritz energy
    in span{psi, rotated Slater} w.r.t. Thouless angles, 2 multi-starts.
    Deadline-aware. Returns (best_vector, iters)."""
    import time
    from scipy.optimize import minimize
    Uup0, Udn0 = reference_U(model, mask)
    nu, nv = Uup0.shape[1], model.L - Uup0.shape[1]
    nd, nvd = Udn0.shape[1], model.L - Udn0.shape[1]
    rng = np.random.default_rng(seed)
    best_E, best_v, it = E, None, 0

    def obj(x):
        if deadline is not None and time.perf_counter() > deadline:
            return 1e9
        v = rotated_slater(model, Uup0, Udn0,
                           x[:nu * nv].reshape(nu, nv),
                           x[nu * nv:].reshape(nd, nvd))
        return two_dim_ritz_drop(H, psi, E, v)

    for start in range(2):
        x0 = np.zeros(nu * nv + nd * nvd) if start == 0 else \
            0.3 * rng.standard_normal(nu * nv + nd * nvd)
        res = minimize(obj, x0, method="L-BFGS-B",
                       options=dict(maxiter=30, ftol=1e-6))
        it += res.nit
        if res.fun < best_E - 1e-12:
            best_E = res.fun
            best_v = rotated_slater(model, Uup0, Udn0,
                                    res.x[:nu * nv].reshape(nu, nv),
                                    res.x[nu * nv:].reshape(nd, nvd))
        if deadline is not None and time.perf_counter() > deadline:
            break
    return best_v, it
