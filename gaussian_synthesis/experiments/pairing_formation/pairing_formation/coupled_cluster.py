"""Variational pair-CCD wavefunction fits with optional low-rank T.

These optimize overlap at selected times, and are intentionally stronger
than substituting the linear bosonic Riccati solution. They are not used as
predictions and no global-optimum claim is made.
"""
import numpy as np
from scipy.optimize import minimize


def objective_T(model, target, T, radial=False):
    raw = np.empty(model.dim, complex)
    raw[model.fermi_index] = 1.
    for current, particle, holes, parents in model.permanent_recursion:
        raw[current] = np.sum(T[particle[:, None], holes]*raw[parents], axis=1)
    if radial:
        adjoint = np.zeros_like(raw)
        state = np.zeros_like(raw)
        F = 0.
        for _, indices in model.blocks:
            rows = indices.ravel()
            norm2 = float(np.vdot(raw[rows], raw[rows]).real)
            if norm2 < 1e-280:
                continue
            overlap = np.vdot(target[rows], raw[rows])
            weight = abs(overlap)**2/norm2
            F += weight
            adjoint[rows] = -overlap*target[rows]/norm2+weight*raw[rows]/norm2
            state[rows] = raw[rows]*overlap.conjugate()/norm2
        state /= np.sqrt(F)
    else:
        norm2 = float(np.vdot(raw, raw).real)
        overlap = np.vdot(target, raw)
        F = abs(overlap)**2/norm2
        adjoint = -overlap*target/norm2+F*raw/norm2
        state = raw/np.sqrt(norm2)
    gradient = np.zeros_like(T)
    for current, particle, holes, parents in reversed(model.permanent_recursion):
        weight = adjoint[current, None]
        np.add.at(gradient, (particle[:, None], holes), weight*raw[parents].conj())
        np.add.at(adjoint, parents, weight*T[particle[:, None], holes].conj())
    return -float(F), gradient, state


def fit_cc(model, target, rank, *, starts=6, seed=1, maxiter=600, radial=False):
    r = model.r
    _, ids = model.blocks[1]
    ref = target[model.fermi_index]
    T0 = target[ids]/ref if abs(ref) > 1e-8 else target[ids]
    u, s, vh = np.linalg.svd(T0, full_matrices=False)
    rng = np.random.default_rng(seed)
    if rank == r:
        base = T0.ravel()
    else:
        base = np.r_[(u[:, :rank]*np.sqrt(s[:rank])).ravel(),
                      (vh[:rank].T*np.sqrt(s[:rank])).ravel()]
    d = len(base)
    def evaluate(x, save=False):
        z = x[:d]+1j*x[d:]
        if rank == r:
            T = z.reshape(r, r)
        else:
            A, B = z[:r*rank].reshape(r, rank), z[r*rank:].reshape(r, rank)
            T = A@B.T
        loss, grad, state = objective_T(model, target, T, radial=radial)
        if save:
            return T, state
        if rank == r:
            grad = grad.ravel()
        else:
            grad = np.r_[(grad@B.conj()).ravel(), (grad.T@A.conj()).ravel()]
        return loss, 2*np.r_[grad.real, grad.imag]
    best, results = None, []
    for k in range(starts):
        z = base if k == 0 else base+.1*(rng.normal(size=d)+1j*rng.normal(size=d))
        result = minimize(evaluate, np.r_[z.real, z.imag], jac=True, method='L-BFGS-B',
                          options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': maxiter})
        results.append({'fidelity': -float(result.fun), 'iterations': int(result.nit),
                        'success': bool(result.success), 'gradient_norm': float(np.linalg.norm(result.jac))})
        if best is None or result.fun < best.fun:
            best = result
    T, state = evaluate(best.x, save=True)
    return {'rank': rank, 'radial_free': radial, 'best_found_fidelity': -float(best.fun), 'restarts': results,
            'global_optimum_certified': False}, T, state
