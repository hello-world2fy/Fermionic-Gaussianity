import _bootstrap  # noqa: F401

"""Full complex eight-dimensional k=2 objective and damped Newton solver.

The solver deliberately does not import the family-specific references.
All sixteen real components are optimized, including off-family directions.
"""

from dataclasses import dataclass
import math
import numpy as np

from gaussian_synthesis import FockBasis
from gaussian_synthesis.gaussian import quadric_matrix
from gaussian_synthesis.solver import ResourceLimitError


_BASIS = FockBasis()
_J = quadric_matrix()


def exact_k2(u):
    """Return q_2 and C_2 = grad_real(q_2/4), with d_(4,2)=35."""
    u = _BASIS.vector(u)
    norm2 = float(np.vdot(u, u).real)
    trace = u@_J@u
    return norm2*norm2-abs(trace)**2/8, norm2*u-trace*(_J@u.conj())/8


def _real_vector(v):
    return np.concatenate((v.real, v.imag))


def _complex_vector(v):
    return v[:8]+1j*v[8:]


def exact_hessian(u):
    """Analytic real Hessian of q_2/4 in [Re u, Im u] coordinates."""
    u = _BASIS.vector(u)
    norm2 = float(np.vdot(u, u).real)
    trace = u@_J@u
    paired = _J@u.conj()
    result = np.empty((16, 16), dtype=float)
    for j in range(16):
        direction = np.zeros(8, dtype=complex)
        direction[j % 8] = 1 if j < 8 else 1j
        derivative = (norm2*direction + 2*np.vdot(u, direction).real*u
                      - (2*(u@_J@direction)*paired + trace*(_J@direction.conj()))/8)
        result[:, j] = _real_vector(derivative)
    return result


@dataclass(frozen=True)
class ExactSolveResult:
    u: np.ndarray
    residual: float
    iterations: int
    status: str


def solve_exact(psi, *, tolerance=1e-10, max_iterations=100, budget=None):
    """Damped Newton solve of C_2(u)=psi, starting at psi itself.

    The residual is the full Euclidean norm, not a family-restricted
    condition. Armijo backtracking includes a roundoff allowance once the
    predicted improvement is smaller than floating-point objective noise.
    """
    psi = _BASIS.vector(psi)
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance)
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be positive and finite")
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0):
        raise ValueError("max_iterations must be a nonnegative integer")
    u = psi.copy()
    completed = 0
    status = "max_iterations"
    try:
        for iteration in range(max_iterations+1):
            if budget is not None:
                budget.check()
            q, reconstruction = exact_k2(u)
            gradient = reconstruction-psi
            residual = float(np.linalg.norm(gradient))
            if residual <= tolerance:
                status = "converged"
                break
            if iteration == max_iterations:
                break
            real_gradient = _real_vector(gradient)
            direction = _complex_vector(np.linalg.solve(exact_hessian(u), -real_gradient))
            directional_derivative = float(np.vdot(gradient, direction).real)
            objective = float(q/4-np.vdot(u, psi).real)
            allowance = 8*np.finfo(float).eps*max(1., abs(objective))
            step = 1.
            accepted = False
            for _ in range(60):
                if budget is not None:
                    budget.check()
                candidate = u+step*direction
                candidate_q, _ = exact_k2(candidate)
                candidate_objective = float(candidate_q/4-np.vdot(candidate, psi).real)
                if (math.isfinite(candidate_objective)
                        and candidate_objective <= objective+1e-4*step*directional_derivative+allowance):
                    u = candidate
                    completed += 1
                    accepted = True
                    break
                step *= .5
            if not accepted:
                status = "line_search_failed"
                break
    except ResourceLimitError:
        status = "resource_limit"
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
        status = "numerical_error"
    _, reconstruction = exact_k2(u)
    return ExactSolveResult(u.copy(), float(np.linalg.norm(reconstruction-psi)), completed, status)
