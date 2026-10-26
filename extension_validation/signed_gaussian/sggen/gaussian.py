"""Four-mode fermionic Gaussian kets, pairing chart, and uniform ray sampling.

Independent re-implementation from the conventions in the read-only reference
project (gaussian_synthesis/src/gaussian_synthesis/{basis,gaussian,sampling}.py)
and the PDF (Eqs. 20-22). No code is imported from the reference project.
"""
from __future__ import annotations

import numpy as np

N_MODES = 4
# bit j = mode j, ascending creation order
EVEN_OCC = (0, 3, 5, 6, 9, 10, 12, 15)   # |S| even, D = 8
ODD_OCC = (1, 2, 4, 7, 8, 11, 13, 14)    # |S| odd,  D = 8

# g^T J g = 2 (g0*g7 - g1*g6 + g2*g5 - g3*g4) = 0  (even Gaussian quadric)
QUADRIC_PAIRS = ((0, 7, 1), (1, 6, -1), (2, 5, 1), (3, 4, -1))


def fock_vector(amplitudes_by_occ, occupations):
    v = np.zeros(len(occupations), dtype=np.complex128)
    for occ, amp in amplitudes_by_occ.items():
        v[occupations.index(occ)] = amp
    return v


def pfaffian(a):
    """Pivoted O(n^3) Pfaffian of a skew-symmetric complex matrix."""
    m = np.array(a, dtype=np.complex128, copy=True)
    n = m.shape[0]
    if n % 2:
        return 0j
    result = 1 + 0j
    for k in range(0, n, 2):
        pivot = k + 1 + int(np.argmax(np.abs(m[k, k + 1:])))
        if m[k, pivot] == 0:
            return 0j
        if pivot != k + 1:
            m[[k + 1, pivot], :] = m[[pivot, k + 1], :]
            m[:, [k + 1, pivot]] = m[:, [pivot, k + 1]]
            result = -result
        entry = m[k, k + 1]
        result *= entry
        if k + 2 < n:
            x, y = m[k, k + 2:].copy(), m[k + 1, k + 2:].copy()
            m[k + 2:, k + 2:] += (np.outer(y, x) - np.outer(x, y)) / entry
    return result


def gaussian_ket_even(Z):
    """Even-parity Gaussian ket |g(Z)> = N exp(1/2 sum Z_ij c_i^dag c_j^dag)|0>.

    Z is a 4x4 complex antisymmetric matrix.  Returns a normalized vector in
    the even occupation basis (EVEN_OCC order).  Chart requires vacuum amp != 0.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    assert Z.shape == (4, 4)
    sign, logdet = np.linalg.slogdet(np.eye(4) + Z.conj().T @ Z)
    N = np.exp(-0.25 * logdet)
    amps = {}
    for S in EVEN_OCC:
        idx = [j for j in range(4) if S & (1 << j)]
        amps[S] = N * pfaffian(Z[np.ix_(idx, idx)])
    return fock_vector(amps, EVEN_OCC)


def spin_to_fock(z):
    """Real Spin(8) coordinates z in R^8 -> even Gaussian ket vector.

    Faithful to the reference project: pairs (z0,z1)->(g0,g7), (z2,z3)->(g1,g6)
    with sign -1, (z4,z5)->(g2,g5), (z6,z7)->(g3,g4).  Vectorized.
    """
    z = np.asarray(z, dtype=np.complex128)
    out = np.empty_like(z)
    for j, (a, b, sign) in enumerate(QUADRIC_PAIRS):
        out[..., a] = (z[..., 2 * j] + 1j * z[..., 2 * j + 1]) / np.sqrt(2)
        out[..., b] = sign * (z[..., 2 * j] - 1j * z[..., 2 * j + 1]) / np.sqrt(2)
    return out


def sample_even_gaussian_rays(rng, count):
    """Uniform (invariant) sampling of normalized even Gaussian rays.

    Port of the reference HaarFourModeSampler: draw orthonormal (r,s) on S^7,
    map z = (r + i s)/sqrt(2) to a normalized even Gaussian ket.
    """
    r = rng.standard_normal((count, 8))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    s = rng.standard_normal((count, 8))
    s -= np.einsum("bi,bi->b", r, s)[:, None] * r   # O(N) projection, not r@s.T
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    z = (r + 1j * s) / np.sqrt(2)
    return spin_to_fock(z)


def density_matrix(ket):
    return np.outer(ket, ket.conj())


# ---- named even-sector Fock states (Slater determinants = Gaussian) ----

def basis_ket(occ):
    v = np.zeros(len(EVEN_OCC), dtype=np.complex128)
    v[EVEN_OCC.index(occ)] = 1.0
    return v


def magic_ket():
    """M = (|0000> + |1111>)/sqrt(2)."""
    return (basis_ket(0) + basis_ket(15)) / np.sqrt(2)


def free_slater_atoms():
    """8 even Slater-determinant pure Gaussians from the SINGLE-PARTICLE quadratic h1.

    Orbitals (4 modes: 0=a_up,1=a_down,2=b_up,3=b_down):
      b_up=(e0+e2)/sqrt2, a_up=(e0-e2)/sqrt2, b_down=(e1+e3)/sqrt2, a_down=(e1-e3)/sqrt2.
    Even Slater determinants = wedges of n_occ = 0,2,4 orbitals; amplitudes are minors det.
    These are GUARANTEED Gaussian (single-particle wedge).  Arbitrary full-Fock eigenvectors
    in degenerate spaces are NOT (they fail the even quadric) and must not be used.
    """
    import itertools
    U = np.zeros((4, 4))
    U[0] = np.array([1, 0, 1, 0]) / np.sqrt(2)
    U[1] = np.array([1, 0, -1, 0]) / np.sqrt(2)
    U[2] = np.array([0, 1, 0, 1]) / np.sqrt(2)
    U[3] = np.array([0, 1, 0, -1]) / np.sqrt(2)
    out = []
    for nocc in (0, 2, 4):
        for J in itertools.combinations(range(4), nocc):
            ket = np.zeros(8, dtype=np.complex128)
            for si, S in enumerate(EVEN_OCC):
                modes = [m for m in range(4) if S & (1 << m)]
                if len(modes) != nocc:
                    continue
                ket[si] = np.linalg.det(U[list(J)][:, modes])
            nrm = np.linalg.norm(ket)
            if nrm > 1e-9:
                out.append(ket / nrm)
    return out


def project_even(ket16):
    """Pick the even-parity 8-vector out of a length-16 Fock vector."""
    return np.array([ket16[m] for m in EVEN_OCC], dtype=np.complex128)


def embed_even(ket8):
    out = np.zeros(16, dtype=np.complex128)
    for i, m in enumerate(EVEN_OCC):
        out[m] = ket8[i]
    return out


# ---- contact-set generation: even Gaussians saturating |<M|g>|^2 = 1/2 ----

J0_STD = np.zeros((4, 4), dtype=np.complex128)
J0_STD[0, 1] = 1; J0_STD[1, 0] = -1; J0_STD[2, 3] = 1; J0_STD[3, 2] = -1


def random_su4(rng):
    """Haar-random 4x4 special unitary (det = 1)."""
    a = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
    q = q * np.exp(-1j * np.angle(np.linalg.det(q)) / 4.0)
    return q


def contact_kets(rng, count, t_min=0.05):
    """Even Gaussian kets with |<M|g>|^2 = 1/2 exactly.

    Z = t V^T J0 V with V in SU(4): then Z^T=-Z, Z^dag Z = t^2 I and
    Pf(Z) = t^2 (phase-aligned), so g0=(1+t^2)^-1, g7=t^2(1+t^2)^-1.
    """
    out = []
    for _ in range(count):
        v = random_su4(rng)
        t = float(np.abs(rng.standard_normal())) + t_min
        out.append(gaussian_ket_even(t * (v.T @ J0_STD @ v)))
    return out


def product_family_kets(ts):
    """Real product BCS family g(t,t) = [(|00>+t|11>)_{01}(|00>+t|11>)_{23}]/(1+t^2)."""
    out = []
    for t in ts:
        k = np.zeros(8, dtype=np.complex128)
        k[0] = 1.0 / (1.0 + t ** 2)
        k[1] = t / (1.0 + t ** 2)   # 0011 = modes 0,1
        k[6] = t / (1.0 + t ** 2)   # 1100 = modes 2,3
        k[7] = t ** 2 / (1.0 + t ** 2)
        out.append(k)
    return out
