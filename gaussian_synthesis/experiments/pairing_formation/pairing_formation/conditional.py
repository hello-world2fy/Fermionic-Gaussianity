"""Transfer-number-conditioned pair orbitals and their frozen tangent spaces.

The blockwise product family is a coherent superposition of Gaussian symmetry
orbits. It is larger than a single radial direction, but smaller than arbitrary
Schmidt-rank-one blocks. Fits are feasible local results, never global bounds.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from .fitting import excitation_features


def normalized_product(features, parameters):
    d = features.shape[1]
    raw = features@(parameters[:d]+1j*parameters[d:])
    value = np.exp(raw-np.max(raw.real))
    return value/np.linalg.norm(value)


def product_objective(features, target, parameters):
    d = features.shape[1]
    value = normalized_product(features, parameters)
    overlap = np.vdot(target, value)
    centered = features-abs(value)**2@features
    derivative = (target.conj()*value)@centered
    return -float(abs(overlap)**2), -2*np.r_[np.real(overlap.conjugate()*derivative),
                                          np.real(1j*overlap.conjugate()*derivative)]


def _fit(features, target, initial, starts, rng, maxiter=600):
    d = features.shape[1]
    guesses = [initial, np.zeros(2*d)]
    guesses.extend(initial+np.r_[rng.normal(0, .4, d), rng.normal(0, .6, d)] for _ in range(max(0, starts-2)))
    best, history = None, []
    for x in guesses[:starts]:
        fit = minimize(lambda z: product_objective(features, target, z), x, jac=True,
                       method='L-BFGS-B', bounds=[(-14., 14.)]*d+[(None, None)]*d,
                       options={'maxiter': maxiter, 'ftol': 2e-14, 'gtol': 2e-9})
        history.append({'fidelity': float(-fit.fun), 'iterations': int(fit.nit), 'success': bool(fit.success)})
        if best is None or fit.fun < best.fun:
            best = fit
    return best.x, normalized_product(features, best.x), history


def fit_conditional_products(model, target, starts=12, seed=1):
    norm = np.linalg.norm(target)
    target = target/norm
    rng = np.random.default_rng(seed)
    features = excitation_features(model)
    basis = np.zeros((model.dim, model.r+1), complex)
    parameters = np.zeros((model.r+1, 2*model.L))
    records = []
    for m, (subsets, indices) in enumerate(model.blocks):
        rows = indices.ravel()
        probability = float(np.linalg.norm(target[rows])**2)
        if m in (0, model.r) or probability < 1e-28:
            basis[rows, m] = 1/np.sqrt(len(rows))
            records.append({'m': m, 'probability': probability, 'conditional_fidelity': 1., 'restarts': []})
            continue
        block = target[indices]/np.sqrt(probability)
        u, singular, vh = np.linalg.svd(block, full_matrices=False)
        F = np.array([[int(j in subset) for j in range(model.r)] for subset in subsets], float)
        guesses = []
        for vector in (u[:, 0], vh[0]):
            log = np.log(np.maximum(abs(vector), 1e-14))+1j*np.angle(vector)
            z = np.linalg.lstsq(F, log, rcond=None)[0]
            x = np.r_[z.real, z.imag]
            # Single-band fits supply useful seeds, not the final joint optimum.
            x, value, _ = _fit(F, vector, x, max(2, starts//3), rng)
            guesses.append(x)
        initial = np.r_[guesses[0][:model.r], guesses[1][:model.r],
                        guesses[0][model.r:], guesses[1][model.r:]]
        x, value, history = _fit(features[rows], block.ravel(), initial, starts, rng)
        basis[rows, m] = value
        parameters[m] = x
        records.append({'m': m, 'probability': probability,
                        'conditional_fidelity': float(abs(np.vdot(block.ravel(), value))**2),
                        'schmidt_rank1_fidelity_upper': float(singular[0]**2), 'restarts': history})
    coefficients = basis.conj().T@target
    fidelity = float(np.linalg.norm(coefficients)**2)
    approximation = basis@coefficients/np.sqrt(fidelity)
    count_preserved = basis@(np.sqrt([x['probability'] for x in records])*np.exp(1j*np.angle(coefficients)))
    return {'fidelity': fidelity, 'target_original_norm': float(norm), 'blocks': records,
            'global_optimum_certified': False}, basis, parameters, approximation, count_preserved


def conditioned_tangent_span(model, basis, order=1, tolerance=1e-10):
    """Freeze orbital profiles; include products of at most `order` densities.

    Basis vectors are Gaussian-orbit derivatives, not individually normalized
    Gaussian atoms. Polynomial order/dimension must be reported separately.
    """
    features = excitation_features(model)
    functions = [np.ones(model.dim)]
    for degree in range(1, order+1):
        functions.extend(np.prod(features[:, ids], axis=1) for ids in combinations(range(model.L), degree))
    F = np.column_stack(functions)
    columns, dimensions = [], []
    for m in range(model.r+1):
        rows = np.flatnonzero(model.m == m)
        raw = basis[rows, m, None]*F[rows]
        u, s, vh = np.linalg.svd(raw, full_matrices=False)
        active = s > tolerance*s[0]
        q = np.zeros((model.dim, int(active.sum())), complex)
        q[rows] = u[:, active]
        columns.append(q)
        dimensions.append(int(active.sum()))
    return np.column_stack(columns), dimensions


def energy_resolved_observables(model, state):
    """Physical one- and two-pair observables; no wavefunction fitting involved."""
    probability = abs(state)**2
    density = probability@model.occ
    connected = model.occ.T@(probability[:, None]*model.occ)-density[:, None]*density[None, :]
    pair = model.pair_density(state)
    cross = pair[model.r:, :model.r]
    energy_transfer = 2*((model.energies[model.r:][None, :]*model.occ[:, model.r:]).sum(axis=1)
                         -(model.energies[:model.r][None, :]*(1-model.occ[:, :model.r])).sum(axis=1))
    average = probability@energy_transfer
    return {'occupations': density.tolist(), 'connected_occupations': connected.tolist(),
            'cross_pair_real': cross.real.tolist(), 'cross_pair_imag': cross.imag.tolist(),
            'energy_transfer_mean': float(average),
            'energy_transfer_variance': float(probability@(energy_transfer**2)-average**2)}
