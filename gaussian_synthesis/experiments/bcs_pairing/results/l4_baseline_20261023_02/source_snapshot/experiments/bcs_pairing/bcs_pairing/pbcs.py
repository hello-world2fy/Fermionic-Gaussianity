"""Onsite PBCS fitting. Values are achievable lower bounds, not global optima."""

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize
from gaussian_synthesis.metrics import normalized, fidelity
from .mean_field import initial_spinors, project_spinors


def parameters_to_spinors(parameters, L):
    theta = np.asarray(parameters[:L])
    phase = np.r_[parameters[L:], 0.]
    return np.column_stack((np.cos(theta), np.sin(theta)*np.exp(1j*phase)))


def spinors_to_parameters(spinors):
    theta = np.arctan2(np.abs(spinors[:, 1]), np.abs(spinors[:, 0]))
    phase = np.angle(spinors[:, 1])-np.angle(spinors[:, 0])
    phase -= phase[-1]
    return np.r_[theta, phase[:-1]]


def objective_gradient(parameters, basis, target):
    """Analytic real-coordinate gradient, including normalization derivatives."""
    L = basis.L
    spinors = parameters_to_spinors(parameters, L)
    theta, phase = parameters[:L], np.r_[parameters[L:], 0.]
    factors = np.where(basis.occupations, spinors[:, 1], spinors[:, 0])
    raw = np.prod(factors, axis=1)
    norm = np.linalg.norm(raw)
    if norm < 1e-150:
        # Zero particle-number probability is outside the ansatz domain.
        return 1., np.zeros(2*L-1)
    state = raw/norm
    derivatives = np.empty((basis.dimension, 2*L-1), dtype=complex)
    for j in range(L):
        rest = np.prod(np.delete(factors, j, axis=1), axis=1)
        derivatives[:, j] = rest*np.where(basis.occupations[:, j],
                                          np.cos(theta[j])*np.exp(1j*phase[j]), -np.sin(theta[j]))
        if j < L-1:
            derivatives[:, L+j] = 1j*basis.occupations[:, j]*raw
    derivatives = (derivatives-state[:, None]*np.real(state.conj() @ derivatives))/norm
    overlap = np.vdot(target, state)
    gradient = -2*np.real(overlap.conjugate()*(target.conj() @ derivatives))
    return -float(abs(overlap)**2), gradient


@dataclass
class PBCSFit:
    state: np.ndarray
    spinors: np.ndarray
    best_found_fidelity: float
    restarts: list
    elapsed_seconds: float
    status: str


def fit_pbcs(basis, target, *, seeds=(), seed=20261023, starts=16, max_seconds=60., maxiter=1000):
    target = normalized(basis.sector.vector(target, nonzero=True))
    if type(starts) is not int or starts < 1 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('starts and maxiter must be positive integers')
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be finite and positive')
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    # Include exact Fock boundary points even when an optimizer chart is singular.
    best_index = int(np.argmax(np.abs(target)))
    occupied = basis.occupations[best_index]
    best_spinors = np.column_stack((1-occupied, occupied)).astype(complex)
    best_state = project_spinors(basis, best_spinors)
    best_value = fidelity(target, best_state)
    candidates = [np.array(x, dtype=complex, copy=True) for x in seeds]+[initial_spinors(basis)]
    for spinors in candidates:
        state = project_spinors(basis, spinors)
        value = fidelity(target, state)
        if value > best_value:
            best_value, best_state, best_spinors = value, state, spinors.copy()
    parameters = [spinors_to_parameters(x) for x in candidates]
    while len(parameters) < starts:
        parameters.append(np.r_[rng.uniform(.1, np.pi/2-.1, basis.L),
                                rng.uniform(-np.pi, np.pi, basis.L-1)])
    records = []
    status = 'best_found_only'
    class Deadline(Exception):
        pass
    for index, initial in enumerate(parameters[:starts]):
        def function(x):
            nonlocal best_value, best_state, best_spinors
            if time.monotonic()-started >= max_seconds:
                raise Deadline()
            value, gradient = objective_gradient(x, basis, target)
            if -value > best_value:
                spinors = parameters_to_spinors(x, basis.L)
                state = project_spinors(basis, spinors)
                best_value, best_state, best_spinors = fidelity(target, state), state, spinors
            return value, gradient
        try:
            result = minimize(function, initial, method='L-BFGS-B', jac=True,
                              bounds=[(0., np.pi/2)]*basis.L+[(None, None)]*(basis.L-1),
                              options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': maxiter})
            records.append({'start': index, 'initial': initial.tolist(), 'final': result.x.tolist(),
                            'fidelity': max(0., -float(result.fun)), 'success': bool(result.success),
                            'iterations': int(result.nit), 'message': str(result.message)})
        except Deadline:
            status = 'resource_limit'
            records.append({'start': index, 'status': 'resource_limit', 'initial': initial.tolist()})
            break
    return PBCSFit(best_state, best_spinors, best_value, records, time.monotonic()-started, status)
