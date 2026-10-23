"""Scalar constants. All expensive replica operators are deliberately absent."""

import math
from fractions import Fraction
import numpy as np


def check_order(k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")


def log_dimension(n, k):
    check_order(k)
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    return sum(math.log1p(k/(2*n-i-j))
               for i in range(1, n+1) for j in range(i+1, n+1))


def dimension(n, k):
    """Exact small-system Weyl dimension, useful for checks and reporting."""
    log_dimension(n, k)
    result = Fraction(1)
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            result *= Fraction(2*n-i-j+int(k), 2*n-i-j)
    return int(result)


def gamma_factor(n, k):
    return math.exp(log_dimension(n, k)/k)


def search_radius(n, k, target_norm):
    check_order(k)
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError("Target norm must be positive and finite")
    return math.exp((k*(n-1)*math.log(2)+math.log(target_norm)
                     -log_dimension(n, k))/(2*k-1))


def log_second_moment_bound(n, k, u_norm):
    """log E||Y||^2 bound from Q_(2k-1) being an orthogonal projection."""
    check_order(k)
    if not math.isfinite(u_norm) or u_norm < 0:
        raise ValueError("u_norm must be finite and nonnegative")
    if u_norm == 0:
        return -math.inf
    return (2*log_dimension(n, k)+(4*k-2)*math.log(u_norm)
            -log_dimension(n, 2*k-1))
