"""Phase-aware Gaussian kets, small pairing charts, and Spin(8) coordinates."""

from dataclasses import dataclass
import numpy as np
from .basis import FockBasis


# In ascending bitmask order [0,3,5,6,9,10,12,15], g.T @ J @ g = 0
# is twice g0*g1234 - g12*g34 + g13*g24 - g14*g23.
QUADRIC_PAIRS = ((0, 7, 1), (1, 6, -1), (2, 5, 1), (3, 4, -1))


def quadric_matrix():
    result = np.zeros((8, 8), dtype=np.complex128)
    for a, b, sign in QUADRIC_PAIRS:
        result[a, b] = result[b, a] = sign
    return result


def spin_to_fock(z):
    """Unitary U with U.T J U = I, U e0 = magic state; works on batches."""
    z = np.asarray(z, dtype=np.complex128)
    if z.shape[-1:] != (8,):
        raise ValueError("Spin coordinates must have final dimension 8")
    result = np.empty_like(z)
    for j, (a, b, sign) in enumerate(QUADRIC_PAIRS):
        result[..., a] = (z[..., 2*j] + 1j*z[..., 2*j+1]) / np.sqrt(2)
        result[..., b] = sign * (z[..., 2*j] - 1j*z[..., 2*j+1]) / np.sqrt(2)
    return result


def pfaffian(matrix):
    """Pivoted O(n^3) skew elimination; Pf(empty)=1, Pf(odd)=0."""
    a = np.array(matrix, dtype=np.complex128, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Pfaffian requires a square matrix")
    if not np.all(np.isfinite(a)) or not np.allclose(a, -a.T, atol=1e-12):
        raise ValueError("Pfaffian requires a finite antisymmetric matrix")
    size = len(a)
    if size % 2:
        return 0j
    result = 1+0j
    for k in range(0, size, 2):
        pivot = k + 1 + int(np.argmax(np.abs(a[k, k+1:])))
        if a[k, pivot] == 0:
            return 0j
        if pivot != k+1:
            a[[k+1, pivot], :] = a[[pivot, k+1], :]
            a[:, [k+1, pivot]] = a[:, [pivot, k+1]]
            result = -result
        entry = a[k, k+1]
        result *= entry
        if k+2 < size:
            x, y = a[k, k+2:].copy(), a[k+1, k+2:].copy()
            a[k+2:, k+2:] += (np.outer(y, x)-np.outer(x, y))/entry
    return result


@dataclass(frozen=True)
class GaussianState:
    basis: FockBasis
    amplitudes: np.ndarray

    def __post_init__(self):
        g = self.basis.vector(self.amplitudes, nonzero=True)
        if not np.isclose(np.linalg.norm(g), 1, atol=1e-10, rtol=0):
            raise ValueError("Gaussian kets must already be normalized")
        # The complete v1 public Fock-ket constructor is restricted to the
        # sector where we can check Gaussianity, not just normalization.
        self.basis.require_four_mode_even()
        if abs(g @ quadric_matrix() @ g) > 1e-9:
            raise ValueError("Ket fails the four-mode Gaussian quadric equation")
        g.flags.writeable = False
        object.__setattr__(self, "amplitudes", g)

    @classmethod
    def from_pairing(cls, z, basis=None):
        basis = basis or FockBasis()
        basis.require_four_mode_even()
        z = np.asarray(z, dtype=np.complex128)
        if z.shape != (basis.n, basis.n):
            raise ValueError("Pairing matrix shape must match n")
        if not np.all(np.isfinite(z)) or not np.allclose(z, -z.T, atol=1e-12):
            raise ValueError("Pairing matrix must be finite and antisymmetric")
        sign, logdet = np.linalg.slogdet(np.eye(basis.n) + z.conj().T @ z)
        if not np.isclose(sign, 1) or not np.isfinite(logdet):
            raise ValueError("Pairing matrix normalization is numerically singular")
        amplitudes = []
        for mask in basis.occupations:
            indices = [j for j in range(basis.n) if mask & (1 << j)]
            amplitudes.append(pfaffian(z[np.ix_(indices, indices)]))
        return cls(basis, np.exp(-0.25*logdet)*np.asarray(amplitudes))

    def overlap(self, u):
        return np.vdot(self.amplitudes, self.basis.vector(u))

    def to_fock(self):
        return self.amplitudes.copy()
