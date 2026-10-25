"""Early-state-only, state-averaged orbital extraction and tangent diagnostics.

This supplies conventional Hamiltonian-aware controls with exactly the same
two-orbit TDVP family as the witness/direct models. It is not Gaussian sampling.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import eye
from .model import PairModel
from .multiorbit import MultiOrbitDynamics, orbit_columns


def ensemble_objective(features, targets, parameters, rank):
    d = features.shape[1]
    z = parameters[:rank*d].reshape(rank, d)+1j*parameters[rank*d:].reshape(rank, d)
    Q = orbit_columns(features, z)
    coefficients = np.linalg.lstsq(Q, targets, rcond=1e-11)[0]
    residual = targets-Q@coefficients
    derivative = np.array([((residual.conj()@coefficients[a])*Q[:, a])@features for a in range(rank)])
    return float(np.vdot(residual, residual).real), -2*np.r_[derivative.real.ravel(), (1j*derivative).real.ravel()]


def response_basis(model, target, kind):
    """Orthonormal centered response vectors, with their generating operators."""
    target = target/np.linalg.norm(target)
    if kind == 'hamiltonian':
        operators = [model.H]
    elif kind == 'generators':
        free = PairModel(model.energies, 0.).H
        interaction = PairModel(model.energies, 1.).H-free
        operators = [free, interaction]
    else:
        raise ValueError(kind)
    identity = eye(model.dim, format='csr')
    centered = [H-np.vdot(target, H@target)*identity for H in operators]
    response = np.column_stack([H@target for H in centered])
    u, singular, vh = np.linalg.svd(response, full_matrices=False)
    keep = singular > max(1e-13, singular[0]*1e-11)
    coefficients = vh[keep].conj().T/singular[keep]
    whitened = [sum((coefficients[i, j]*H for i, H in enumerate(centered))) for j in range(int(keep.sum()))]
    return u[:, keep], whitened, {'kind': kind, 'rank': int(keep.sum()), 'singular_values': singular.tolist()}


def tangent_response_score(dynamics, coordinates, operators):
    """Mean squared normal response at the projected early state, not a forecast."""
    coefficients, logs = dynamics.unpack(coordinates)
    state = dynamics.state(coordinates)
    actions = np.column_stack([H@state for H in operators])
    residual = actions.copy()
    ranks = []
    for m, rows in enumerate(dynamics.groups):
        F = dynamics.features[rows]
        Q = orbit_columns(F, logs[m])
        means = abs(Q).T**2@F
        tangent = np.column_stack([Q]+[coefficients[m, a]*Q[:, a, None]*(F-means[a]) for a in range(dynamics.rank)])
        scales = np.linalg.norm(tangent, axis=0)
        scales[scales < 1e-14] = 1.
        u, singular, _ = np.linalg.svd(tangent/scales, full_matrices=False)
        keep = singular > max(1e-13, dynamics.svd_cutoff*singular[0])
        basis = u[:, keep]
        residual[rows] -= basis@(basis.conj().T@actions[rows])
        ranks.append(int(keep.sum()))
    return {'score': float(np.linalg.norm(residual)**2/len(operators)),
            'normal_squared_norms': np.sum(abs(residual)**2, axis=0).tolist(),
            'complex_tangent_rank': sum(ranks)}


def fit_response_orbits(model, target, responses, weight, initial_logs, starts=6, seed=1, maxiter=5000):
    target = target/np.linalg.norm(target)
    rank = initial_logs.shape[1]
    dynamics = MultiOrbitDynamics(model, rank)
    columns = np.column_stack((target, np.sqrt(weight/responses.shape[1])*responses))
    rng = np.random.default_rng(seed)
    logs = initial_logs.copy()
    records = []
    for m, rows in enumerate(dynamics.groups):
        if m in (0, model.r):
            records.append({'m': m, 'restarts': [], 'loss': 0.})
            continue
        block = columns[rows]
        scale = float(np.linalg.norm(block)**2)
        block = block/np.sqrt(max(scale, 1e-300))
        x0 = np.r_[logs[m].real.ravel(), logs[m].imag.ravel()]
        guesses = [x0]+[x0+rng.normal(0, .25, len(x0)) for _ in range(starts-1)]
        best, history = None, []
        for x in guesses:
            fit = minimize(lambda y: ensemble_objective(dynamics.features[rows], block, y, rank), x, jac=True,
                           method='L-BFGS-B', bounds=[(-14., 14.)]*(rank*model.L)+[(None, None)]*(rank*model.L),
                           options={'maxiter': maxiter, 'ftol': 2e-14, 'gtol': 2e-9, 'maxls': 40})
            history.append({'loss': float(fit.fun), 'iterations': int(fit.nit), 'success': bool(fit.success),
                            'gradient_norm': float(np.linalg.norm(fit.jac))})
            if best is None or fit.fun < best.fun:
                best = fit
        logs[m] = best.x[:rank*model.L].reshape(rank, model.L)+1j*best.x[rank*model.L:].reshape(rank, model.L)
        records.append({'m': m, 'loss': float(best.fun), 'weighted_block_norm': scale, 'restarts': history})
    coordinates, fidelity = dynamics.encode(logs, target)
    return coordinates, {'weight': weight, 'initial_fidelity': fidelity, 'rank': rank, 'blocks': records,
                         'global_optimum_certified': False, 'starts': starts, 'maxiter': maxiter}
