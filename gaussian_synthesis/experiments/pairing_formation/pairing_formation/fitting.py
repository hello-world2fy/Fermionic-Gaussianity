"""Feasible PBCS and collective radial fits, never global upper bounds."""
import numpy as np
from scipy.optimize import minimize


def excitation_features(model):
    return np.column_stack((model.occ[:, model.r:], 1-model.occ[:, :model.r]))


def fit_product_family(model, target, *, radial=False, slope=False, starts=8, seed=1, maxiter=600):
    """PBCS, a single direction with free radial weights, or m-dependent direction.

The slope option has log orbital z_j(m)=a_j+m*b_j and free complex radial
weights. It is a diagnostic family, not an unrestricted Gaussian family.
"""
    rng = np.random.default_rng(seed)
    features = excitation_features(model)
    if slope:
        features = np.column_stack((features, features*model.m[:, None]/model.r))
    d = features.shape[1]
    groups = [np.flatnonzero(model.m == m) for m in range(model.r+1)] if radial else [np.arange(model.dim)]

    def evaluate(parameters, get_state=False):
        z = parameters[:d]+1j*parameters[d:]
        lograw = features@z
        state = np.zeros(model.dim, complex)
        gradient = np.zeros(2*d)
        total = 0.
        for indices in groups:
            logv, f = lograw[indices], features[indices]
            v = np.exp(logv-np.max(logv.real))
            v /= np.linalg.norm(v)
            target_group = target[indices]
            overlap = np.vdot(target_group, v)
            centered = f-(abs(v)**2)@f
            derivatives = (target_group.conj()*v)@centered
            gradient[:d] -= 2*np.real(overlap.conjugate()*derivatives)
            gradient[d:] -= 2*np.real(1j*overlap.conjugate()*derivatives)
            total += abs(overlap)**2
            state[indices] = v*overlap.conjugate()
        if get_state:
            return state/np.linalg.norm(state)
        return -float(total), gradient

    density = abs(target)**2@excitation_features(model)
    a = .5*np.log(np.clip(density, 1e-5, 1-1e-5)/np.clip(1-density, 1e-5, 1-1e-5))
    _, ids = model.blocks[1]
    u, _, vh = np.linalg.svd(target[ids], full_matrices=False)
    orbital = np.r_[u[:, 0], vh[0]]
    base = np.log(np.maximum(abs(orbital), 1e-8))+1j*np.angle(orbital)
    if slope:
        a = np.r_[a, np.zeros(model.L)]
        base = np.r_[base, np.zeros(model.L)]
    candidates = [np.r_[base.real, base.imag], np.r_[a, np.zeros(d)]]
    candidates.extend(np.r_[a+rng.normal(0, .4, d), rng.normal(0, 1., d)] for _ in range(starts-2))
    best = None
    records = []
    for x in candidates[:starts]:
        result = minimize(evaluate, x, method='L-BFGS-B', jac=True,
                          bounds=[(-12, 12)]*d+[(None, None)]*d,
                          options={'ftol': 3e-13, 'gtol': 1e-8, 'maxiter': maxiter})
        records.append({'fidelity': float(-result.fun), 'success': bool(result.success),
                        'iterations': int(result.nit), 'gradient_norm': float(np.linalg.norm(result.jac))})
        if best is None or result.fun < best.fun:
            best = result
    return {'fidelity': float(-best.fun), 'parameters': best.x.tolist(), 'restarts': records,
            'global_optimum_certified': False}, evaluate(best.x, True)
