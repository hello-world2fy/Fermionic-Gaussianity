"""Two-site Hubbard dimer: independent validation of the Slater extent
xi_S = 1 + U / sqrt(U^2 + 16 t^2).

Method (2-fermion canonical / Takagi decomposition)
---------------------------------------------------
A normalized 2-fermion state in 4 spin-orbitals is an antisymmetric matrix
M, psi = sum_{i<j} M_{ij} c_i^dag c_j |0>. Under a unitary change of
spin-orbital basis M block-diagonalizes as
    M = U^T (lambda_1 J) U  (+ lambda_2 J),  J = [[0,1],[-1,0]].
The two canonical pair amplitudes lambda_1, lambda_2 are the distinct
singular values of M (each repeated twice). The state is the orthogonal sum
psi = lambda_1 D_1 + lambda_2 D_2 of two orthonormal Slater determinants,
so the phase-balanced convex gauge (total coefficient mass) is
    tau = lambda_1 + lambda_2,   xi_S = tau^2 = 1 + 2 lambda_1 lambda_2
(using lambda_1^2 + lambda_2^2 = 1 for a unit state).

This is computed *independently* from the analytic formula: we diagonalize
the dimer Hamiltonian, take the exact ground vector, read off M, and form
xi_S numerically. Agreement with 1 + U/sqrt(U^2+16 t^2) is the check.
"""
from __future__ import annotations

import numpy as np

from hubbard import HubbardModel


def antisymmetric_matrix(state_vec: np.ndarray, masks: np.ndarray, n_orb: int) -> np.ndarray:
    """Build the antisymmetric coefficient matrix M (n_orb x n_orb) for a
    fixed-N Fock-basis vector. Only valid here because the basis spans the
    whole fixed-N sector (N=2)."""
    M = np.zeros((n_orb, n_orb), dtype=complex)
    for val, mask in zip(state_vec, masks):
        occ = [b for b in range(n_orb) if (int(mask) >> b) & 1]
        assert len(occ) == 2
        i, j = min(occ), max(occ)
        M[i, j] += val
        M[j, i] -= val
    return M


def canonical_pair_amplitudes(M: np.ndarray):
    """Return the two canonical singular values lambda_1 >= lambda_2 >= 0
    of an antisymmetric 2-fermion matrix."""
    sv = np.linalg.svd(M, compute_uv=False)
    # sv = [lambda_1, lambda_1, lambda_2, lambda_2] for a 4x4 antisym matrix
    sv = np.sort(np.real(sv))[::-1]
    lam1 = (sv[0] + sv[1]) / 2
    lam2 = (sv[2] + sv[3]) / 2
    return float(lam1), float(lam2)


def dimer_extent_numerical(U: float, t: float = 1.0):
    m = HubbardModel(L=2, t=t, U=U)
    E, psi = m.exact_ground()
    M = antisymmetric_matrix(psi, m.states, n_orb=2 * m.L)
    lam1, lam2 = canonical_pair_amplitudes(M)
    tau = lam1 + lam2
    xi = tau ** 2
    return dict(U=U, t=t, E=E, lam1=lam1, lam2=lam2, tau=tau, xi_S=xi,
                norm2=lam1 ** 2 + lam2 ** 2)


def xi_S_analytic(U: float, t: float = 1.0):
    return 1.0 + U / np.sqrt(U ** 2 + 16 * t ** 2)


if __name__ == "__main__":
    print(f"{'U':>6} {'lam1':>9} {'lam2':>9} {'xi_num':>10} {'xi_an':>10} {'rel.err':>10}")
    for U in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]:
        r = dimer_extent_numerical(U, t=1.0)
        xa = xi_S_analytic(U, 1.0)
        rel = abs(r["xi_S"] - xa) / xa if xa > 0 else abs(r["xi_S"] - xa)
        print(f"{U:6.1f} {r['lam1']:9.5f} {r['lam2']:9.5f} {r['xi_S']:10.6f} "
              f"{xa:10.6f} {rel:10.2e}")
