"""Deterministic conversion of validated moments into extent certificates.

All probability statements belong to the caller: the bounds here hold on the
same event as ``q_lower <= q_k(u) <= q_upper`` and
``||psi - C_k(u)|| <= residual_bound``. No new samples or failure probability
are introduced. Inputs need not be normalized.
"""

from dataclasses import dataclass
import math
from numbers import Real
import numpy as np
from .theory import check_order, log_dimension


@dataclass(frozen=True)
class ExtentCertificate:
    k: int
    t_lower: float
    t_upper: float
    lower_bound: float
    upper_bound: float
    gamma: float


def _nonnegative_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return value


def _roundoff_slack(*values):
    # Only comparisons use this tolerance. A nearly inverted computed interval
    # is widened outwards; user-supplied q intervals are never silently reordered.
    return 64*np.finfo(float).eps*max(*(abs(x) for x in values), np.finfo(float).tiny)


def extent_certificate(psi, u, k, *, n=4, q_lower, q_upper, residual_bound):
    """Bound t_k=N_k^*(psi), then xi_G in [t_lower^2, Gamma*t_upper^2].

    Jensen and the frame identity give N_k(v) >= ||v||/c, with
    c=sqrt(D)*d_(n,k)^(-1/(2k)). Thus N_k^*(e) <= c||e||. Moreover
    N_k^*(C_k(u)) = q_k(u)^((2k-1)/(2k)), so the triangle inequality
    supplies both residual bounds. The witness quotient supplies an additional
    lower bound without assuming that u is an optimizer.

    The returned stage interval is not clipped by the target's trivial Fock
    bounds. Contradictory inputs raise ValueError; nonrepresentable arithmetic
    raises FloatingPointError. At most 64 machine epsilons of relative roundoff
    are allowed in derived comparisons, always by widening the upper endpoint.
    This is a statistical certificate in floating-point arithmetic, not an
    interval-arithmetic proof of rounding errors.
    """
    check_order(k)
    if (isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or n < 2):
        raise ValueError("n must be an integer >= 2")
    dimension = 1 << (int(n)-1)
    psi = np.asarray(psi, dtype=np.complex128)
    u = np.asarray(u, dtype=np.complex128)
    for name, vector in (("psi", psi), ("u", u)):
        if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite vector of shape ({dimension},)")
    q_lower = _nonnegative_scalar(q_lower, "q_lower")
    q_upper = _nonnegative_scalar(q_upper, "q_upper")
    rho = _nonnegative_scalar(residual_bound, "residual_bound")
    if q_lower > q_upper:
        raise ValueError("q_lower must not exceed q_upper")
    target_norm = float(np.linalg.norm(psi))
    u_norm = float(np.linalg.norm(u))
    if not math.isfinite(target_norm) or not math.isfinite(u_norm):
        raise FloatingPointError("Vector norm exceeds floating-point range")
    log_d = log_dimension(n, k)
    c = math.exp(.5*(n-1)*math.log(2)-log_d/(2*k))
    gamma = math.exp(log_d/k)
    exponent = (2*k-1)/(2*k)

    if not np.any(u):
        # q(0)=0 and C(0)=0 are exact: an interval containing zero may be
        # tightened to zero even when its supplied upper endpoint is positive.
        if q_lower > 0:
            raise ValueError("The zero witness has q_k(0)=0")
        if target_norm > rho+_roundoff_slack(target_norm, rho):
            raise ValueError("Zero-witness residual bound is smaller than ||psi||")
        rho = max(rho, target_norm)
        t_lower, t_upper = 0., c*rho
    else:
        if u_norm == 0 or q_upper == 0:
            raise ValueError("A nonzero witness must have a positive norm and q_upper")
        # The supplied interval must intersect the exact frame/projection
        # bounds d*||u||^(2k)/D^k <= q_k(u) <= ||u||^(2k).
        log_norm_power = 2*k*math.log(u_norm)
        log_q_floor = log_d+log_norm_power-k*(n-1)*math.log(2)
        log_q_upper = math.log(q_upper)
        if log_q_upper < log_q_floor-_roundoff_slack(log_q_upper, log_q_floor, 1.):
            raise ValueError("q interval lies below the frame lower bound")
        if q_lower > 0:
            log_q_lower = math.log(q_lower)
            if log_q_lower > log_norm_power+_roundoff_slack(log_q_lower, log_norm_power, 1.):
                raise ValueError("q interval lies above the projector upper bound")
        witness = float(abs(np.vdot(u, psi)))/q_upper**(1/(2*k))
        t_lower = max(witness, q_lower**exponent-c*rho, 0.)
        t_upper = q_upper**exponent+c*rho

    if not all(math.isfinite(x) for x in (t_lower, t_upper, gamma)):
        raise FloatingPointError("Extent certificate exceeds floating-point range")
    if t_lower > t_upper:
        if t_lower-t_upper > _roundoff_slack(t_lower, t_upper):
            raise ValueError("Moment and residual inputs imply an inconsistent dual interval")
        t_upper = t_lower
    lower_bound = t_lower*t_lower
    upper_bound = gamma*t_upper*t_upper
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
        raise FloatingPointError("Squared extent certificate exceeds floating-point range")
    return ExtentCertificate(int(k), t_lower, t_upper,
                             lower_bound, upper_bound, gamma)
