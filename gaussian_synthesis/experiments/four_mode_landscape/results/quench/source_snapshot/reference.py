import _bootstrap  # noqa: F401

"""Analytic four-mode references, independent of either numerical solver."""

from dataclasses import dataclass
from numbers import Real
import math
import numpy as np

from gaussian_synthesis import FockBasis
from gaussian_synthesis.gaussian import quadric_matrix


basis = FockBasis()
_J = quadric_matrix()
_LOGICAL_INDICES = tuple(basis.index(mask) for mask in (0, 12, 3, 15))
_ODD_PAIR_INDICES = tuple(basis.index(mask) for mask in (5, 6, 9, 10))


def _finite_real(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _cat_parameter(r):
    r = _finite_real(r, "r")
    if not 0 <= r <= .5:
        raise ValueError("r must be in [0, 1/2]")
    return r


def cat_state(r):
    """sqrt(1-r)|vac> + sqrt(r)|full>, in the repository bitmask order."""
    r = _cat_parameter(r)
    return math.sqrt(1-r)*basis.ket(0) + math.sqrt(r)*basis.ket(15)


def cat_witness(r):
    """Solve the strictly monotone scalar reference equation by bisection."""
    r = _cat_parameter(r)
    if r == 0:
        return basis.ket(0)
    if r == .5:
        return (8/7)**(1/3)*cat_state(r)
    rho = math.sqrt(r/(1-r))
    lo, hi = 0., 1.
    for _ in range(80):
        t = (lo+hi)/2
        ratio = t*(t*t+.75)/(1+.75*t*t)
        if ratio < rho:
            lo = t
        else:
            hi = t
    t = (lo+hi)/2
    a = (math.sqrt(1-r)/(1+.75*t*t))**(1/3)
    return a*basis.ket(0) + (t*a)*basis.ket(15)


def cat_extent(r):
    r = _cat_parameter(r)
    return 1 + 2*math.sqrt(r*(1-r))


def quench_state(phi):
    """Exact evolution under H/U = n_1 n_3, with phi = U t."""
    phi = _finite_real(phi, "phi")
    return (basis.ket(0)+basis.ket(3)+basis.ket(12)
            + np.exp(-1j*phi)*basis.ket(15))/2


@dataclass(frozen=True)
class QuenchReference:
    r: float
    unitary: np.ndarray
    state: np.ndarray
    witness: np.ndarray


def quench_reference(phi):
    """Schmidt reference with a legal Gaussian extension to all eight states.

    For A = L diag(s) Vh the two logical unitaries are L and Vh.T.
    An arbitrary logical W in U(2) acts by W on the even pair sector and
    by sqrt(det W) I on its odd sector. This is a Gaussian unitary up to
    global phase; leaving the odd sector equal to I would generally fail.
    The chosen square-root branches can jump at degeneracies, but the
    resulting physical state and witness are independent of that gauge.
    """
    phi = _finite_real(phi, "phi")
    coefficients = np.array([[1, 1], [1, np.exp(-1j*phi)]], dtype=complex)/2
    left, _, right_h = np.linalg.svd(coefficients)
    right = right_h.T
    unitary = np.zeros((8, 8), dtype=complex)
    unitary[np.ix_(_LOGICAL_INDICES, _LOGICAL_INDICES)] = np.kron(left, right)
    odd_phase = np.sqrt(complex(np.linalg.det(left)))*np.sqrt(complex(np.linalg.det(right)))
    unitary[np.ix_(_ODD_PAIR_INDICES, _ODD_PAIR_INDICES)] = odd_phase*np.eye(4)
    r = float(np.clip((1-abs(math.cos(phi/2)))/2, 0, .5))
    return QuenchReference(r, unitary, quench_state(phi), unitary@cat_witness(r))


def quadratic_unitary(phi):
    """exp(-i phi H_Q), H_Q = sum_{j=1}^4 j n_j, on the even sector."""
    phi = _finite_real(phi, "phi")
    energies = np.array([sum(j+1 for j in range(4) if mask & (1 << j))
                         for mask in basis.occupations])
    return np.diag(np.exp(-1j*phi*energies))


def quadratic_state(phi):
    return quadratic_unitary(phi)@quench_state(0)


def gaussianity_defect(psi):
    """A_empty A_full - A_12 A_34 + A_13 A_24 - A_14 A_23."""
    psi = basis.vector(psi)
    return complex(psi@_J@psi/2)
