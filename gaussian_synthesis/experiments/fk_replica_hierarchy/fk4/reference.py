"""Analytic regression references; never imported by stochastic optimizers."""
import math


def log_dimension(n, beta):
    """Weyl dimension polynomial continued to real beta >= 0."""
    if beta < 0 or int(n) != n or n < 2:
        raise ValueError("n >= 2 and beta >= 0 required")
    return sum(math.log1p(beta / (2*n-i-j))
               for i in range(1, n+1) for j in range(i+1, n+1))


def magic_log_z(beta):
    return -beta*math.log(2)+math.log(6)-sum(math.log(beta+j) for j in (1, 2, 3))


def magic_reference(k):
    log_q = log_dimension(4, k)+magic_log_z(k)
    log_radius = -log_q/(2*k-1)
    return {"log_q": log_q, "radius": math.exp(log_radius),
            "L": math.exp(-log_q/(2*k)),
            "B": math.exp(log_dimension(4, k)+magic_log_z(k-.5)+(2*k-1)*log_radius),
            "mass_squared_mean": (2*k+1)/(2*k+7), "tau": math.sqrt(2),
            "normal_width_rms": math.sqrt(6/(2*k+7))}


def gaussian_reference(n, k):
    return {"log_q": 0., "radius": 1., "L": 1., "tau": 1.,
            "B": math.exp(log_dimension(n, k)-log_dimension(n, k-.5))}

