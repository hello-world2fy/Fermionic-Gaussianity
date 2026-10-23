"""Exact Haar Gaussian sampling by SO(2n) QR and a Givens Spin lift.

The covariance convention is Gamma[a,b] = i <gamma_a gamma_b> for a != b;
the vacuum has Gamma[2*j,2*j+1] = -1.  Kets use ascending occupation
bitmasks, with mode j at bit j.  The lift fixes an arbitrary ray phase.
"""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class GaussianBatch:
    kets: np.ndarray
    covariances: np.ndarray
    sectors: np.ndarray


@lru_cache(maxsize=None)
def _masks(n, parity):
    return np.array([m for m in range(1 << n)
                     if m.bit_count() % 2 == parity], dtype=np.int64)


def _majorana_action(masks, index):
    """Return destination masks and source phases of gamma_index."""
    mode = index // 2
    signs = np.array([1 - 2 * (int(m & ((1 << mode) - 1)).bit_count() % 2)
                      for m in masks], dtype=np.complex128)
    if index % 2:
        signs *= 1j * (1 - 2 * ((masks >> mode) & 1))
    return masks ^ (1 << mode), signs


@lru_cache(maxsize=None)
def _bivector_action(n, a, b):
    """Destination-indexed permutation and phase for gamma_a gamma_b."""
    masks = _masks(n, 0)
    intermediate, phase_b = _majorana_action(masks, b)
    destination, phase_a = _majorana_action(intermediate, a)
    inverse = np.empty(1 << n, dtype=np.int64)
    inverse[masks] = np.arange(len(masks))
    permutation = inverse[destination]
    # The occupation permutation is an involution, also for a and b in
    # the same complex mode.  Reindex the source phases by destination.
    phases = (phase_a * phase_b)[permutation]
    return permutation, phases


def _haar_so(n_majoranas, count, rng):
    raw = rng.standard_normal((count, n_majoranas, n_majoranas))
    orthogonal, triangular = np.linalg.qr(raw)
    diagonal = np.diagonal(triangular, axis1=1, axis2=2)
    # QR's conventional diagonal signs otherwise bias the orthogonal matrix.
    signs = np.where(diagonal < 0, -1., 1.)
    orthogonal *= signs[:, None, :]
    orientation = np.linalg.slogdet(orthogonal)[0]
    orthogonal[:, :, -1] *= orientation[:, None]
    return orthogonal


def _spin_lift(orthogonal, n):
    """Lift each SO matrix, acting on the vacuum in the even Fock sector."""
    count, size, _ = orthogonal.shape
    work = orthogonal.copy()
    rotations = []
    # G_last ... G_first O = I.  Each G has plane block [[c,s],[-s,c]].
    # Hence O = G_first.T ... G_last.T and these inverse rotations act on
    # the vacuum in reverse elimination order.
    for column in range(size - 1):
        for row in range(size - 1, column, -1):
            x, y = work[:, column, column], work[:, row, column]
            radius = np.hypot(x, y)
            cosine = np.divide(x, radius, out=np.ones_like(x), where=radius != 0)
            sine = np.divide(y, radius, out=np.zeros_like(y), where=radius != 0)
            left = work[:, column, column:].copy()
            right = work[:, row, column:].copy()
            work[:, column, column:] = cosine[:, None]*left + sine[:, None]*right
            work[:, row, column:] = -sine[:, None]*left + cosine[:, None]*right
            half_angle = .5*np.arctan2(sine, cosine)
            rotations.append((column, row, np.cos(half_angle), -np.sin(half_angle)))
    kets = np.zeros((count, 1 << (n - 1)), dtype=np.complex128)
    kets[:, 0] = 1.
    for a, b, cosine, sine in reversed(rotations):
        permutation, phases = _bivector_action(n, a, b)
        transformed = kets[:, permutation] * phases[None, :]
        kets = cosine[:, None]*kets + sine[:, None]*transformed
    return kets


def _sample_even(n, count, rng):
    orthogonal = _haar_so(2*n, count, rng)
    # O Gamma_vac has columns [O_odd, -O_even].
    rotated_vacuum = np.empty_like(orthogonal)
    rotated_vacuum[:, :, 0::2] = orthogonal[:, :, 1::2]
    rotated_vacuum[:, :, 1::2] = -orthogonal[:, :, 0::2]
    covariances = rotated_vacuum @ orthogonal.transpose(0, 2, 1)
    return _spin_lift(orthogonal, n), covariances


def sample_gaussian(n, count, rng, parity=0, *, batch_size=256):
    """Sample independent invariant pure Gaussian rays in one parity sector.

    QR of independent real Gaussian matrices supplies genuinely Haar SO(2n)
    rotations.  For odd parity, gamma_0 maps the even orbit to the odd orbit.
    The returned covariance is always in the physical 2n-Majorana order.
    ``sectors`` records the requested local parity (0 or 1).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 2 <= n <= 12:
        raise ValueError("n must be an integer in [2, 12]")
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
            or count < 1):
        raise ValueError("count must be a positive integer")
    if isinstance(parity, bool) or parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    if (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    kets = np.empty((count, 1 << (n - 1)), dtype=np.complex128)
    covariances = np.empty((count, 2*n, 2*n), dtype=float)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        kets[start:stop], covariances[start:stop] = _sample_even(n, stop-start, rng)
    if parity:
        even_masks, odd_masks = _masks(n, 0), _masks(n, 1)
        # gamma_0 flips bit zero, with unit phase in this convention.
        source = np.searchsorted(even_masks, odd_masks ^ 1)
        kets = kets[:, source]
        # gamma_0 conjugation fixes gamma_0 and reverses all other Majoranas.
        # Multiplying all signs by -1 leaves this covariance transformation
        # unchanged, so changing just row/column zero is sufficient.
        covariances[:, 0, :] *= -1
        covariances[:, :, 0] *= -1
    return GaussianBatch(kets, covariances, np.full(count, parity, dtype=np.int8))

