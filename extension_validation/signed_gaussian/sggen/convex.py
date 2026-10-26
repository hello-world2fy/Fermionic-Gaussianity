"""Real convex program for the operator Gaussian extent dual witness.

Mirrors the PDF (Theorem 1) but for Hermitian operators W instead of kets:
    J_k(W) = d_{n,k}/(2k) * E_g |Tr(W sigma_g)|^{2k} - Tr(W rho)
is convex in W.  Its gradient is
    grad J_k(W) = d_{n,k} E_g |Tr(W sigma_g)|^{2k-2} Tr(W sigma_g) sigma_g - rho
                 = C_k(W) - rho.
At the minimizer, KKT gives C_k(W_k) = rho (the continuous Gaussian operator
measure reconstructs rho).  The normalized witness w_k = W_k / N_k(W_k),
N_k(W)=q_k(W)^{1/(2k)}, lower-bounds the signed synthesis cost R(rho).

This is NOT an LP: it is an unconstrained convex integral objective solved by
projected gradient with fresh Monte-Carlo samples, exactly the PDF baseline.
"""
from __future__ import annotations

import math
import numpy as np

N_MODES = 4
DIM = 8  # even sector


def log_dimension(n, k):
    s = 0.0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s += math.log1p(k / (2 * n - i - j))
    return s


def d_nk(n, k):
    return math.exp(log_dimension(n, k))


# ---------- Hermitian <-> real vector (correct Hilbert-Schmidt metric) ----------

def herm_to_real(W):
    """Hermitian 8x8 -> real vector x of length 8 + 2*28 = 64.

    Inner product: Tr(A^dag B) = sum_i A_ii B_ii + 2 sum_{i<j} Re(conj A_ij) B_ij.
    We store diag (8), off-diag Re (28), off-diag Im (28); the gradient w.r.t. x
    carries the factor 2 so that <grad, dx> = dJ.
    """
    d = np.diag(W).real
    re = np.array([W[i, j].real for i in range(8) for j in range(i + 1, 8)])
    im = np.array([W[i, j].imag for i in range(8) for j in range(i + 1, 8)])
    return np.concatenate([d, re, im])


def real_to_herm(x):
    W = np.zeros((8, 8), dtype=np.complex128)
    np.fill_diagonal(W, x[:8])
    idx = 8
    for i in range(8):
        for j in range(i + 1, 8):
            W[i, j] = x[idx] + 1j * x[idx + 28]
            W[j, i] = x[idx] - 1j * x[idx + 28]
            idx += 1
    return W


def hs_inner(A, B):
    return np.trace(A.conj().T @ B).real


def objective_and_grad(x, rho, rays, k):
    """J_k(x) and real gradient, with fresh rays (M,8) of normalized even kets."""
    W = real_to_herm(x)
    sigma_trace = rays.conj() @ W @ rays.T   # wait: need Tr(W |g><g|) = <g|W|g>
    # <g|W|g> for each ray:
    t = np.einsum("bi,ij,bj->b", rays.conj(), W, rays)
    d = d_nk(N_MODES, k)
    abs_t = np.abs(t)
    # J = d/(2k) E|t|^{2k} - Tr(W rho)
    J = d / (2 * k) * np.mean(abs_t ** (2 * k)) - np.trace(W @ rho).real
    # grad integrand: d |t|^{2k-2} t |g><g|  (complex t; use conj for real grad)
    coef = d * (abs_t ** (2 * k - 2)) * t
    # C_k(W) = E coef_b |g_b><g_b|
    C = np.einsum("b,bi,bj->ij", coef, rays, rays.conj()) / len(rays)
    G = C - rho
    # real gradient with HS factor 2 on off-diagonals
    gx = herm_to_real(G)
    # off-diag stored once but HS derivative has factor 2
    gx[8:] *= 2.0
    return J, gx


def solve_convex(rho, k, seed, *, iterations=3000, batch=2048, lr=None):
    """Projected-gradient minimization of J_k(W).  Returns dict with history."""
    rng = np.random.default_rng(seed)
    x = np.zeros(64)
    # warm start: witness proportional to rho for k>1
    if k > 1:
        x = herm_to_real(rho) * 0.5
    if lr is None:
        lr = 0.1 / (2 * k - 1)
    hist = []
    for it in range(iterations):
        rays = np.einsum("bi->bi", rng.standard_normal((batch, 8)))
        # normalized even rays via spin_to_fock on orthonormal pair
        r = rng.standard_normal((batch, 8)); r /= np.linalg.norm(r, axis=1, keepdims=True)
        s = rng.standard_normal((batch, 8))
        s -= np.einsum("bi,bi->b", r, s)[:, None] * r
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        z = (r + 1j * s) / np.sqrt(2)
        rays = spin_to_fock(z)
        J, gx = objective_and_grad(x, rho, rays, k)
        step = lr * (1 + it / 500) ** (-0.5)
        x = x - step * gx
        if it % 200 == 0 or it == iterations - 1:
            hist.append((it, J, float(np.linalg.norm(gx))))
    Wk = real_to_herm(x)
    # KKT residual with a large independent batch
    r = rng.standard_normal((40000, 8)); r /= np.linalg.norm(r, axis=1, keepdims=True)
    s = rng.standard_normal((40000, 8))
    s -= np.einsum("bi,bi->b", r, s)[:, None] * r
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    rays = spin_to_fock((r + 1j * s) / np.sqrt(2))
    t = np.einsum("bi,ij,bj->b", rays.conj(), Wk, rays)
    d = d_nk(N_MODES, k)
    coef = d * (np.abs(t) ** (2 * k - 2)) * t
    C = np.einsum("b,bi,bj->ij", coef, rays, rays.conj()) / len(rays)
    kkt = np.linalg.norm(C - rho)
    Nk = (d * np.mean(np.abs(t) ** (2 * k))) ** (1.0 / (2 * k))
    witness_value = np.trace(Wk @ rho).real / Nk
    return {"k": k, "Wk": Wk, "J_final": J, "kkt_residual": kkt,
            "Nk": Nk, "witness_lower": witness_value, "history": hist}


def spin_to_fock(z):
    pairs = ((0, 7, 1), (1, 6, -1), (2, 5, 1), (3, 4, -1))
    out = np.empty_like(z)
    for j, (a, b, sgn) in enumerate(pairs):
        out[..., a] = (z[..., 2 * j] + 1j * z[..., 2 * j + 1]) / np.sqrt(2)
        out[..., b] = sgn * (z[..., 2 * j] - 1j * z[..., 2 * j + 1]) / np.sqrt(2)
    return out
