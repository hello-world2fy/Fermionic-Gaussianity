"""Feasible Gaussian cost repairs and analytically supported magic witnesses.

Finite-dictionary SOCP duals are NOT full-Gaussian dual certificates. Only
the product-four-mode magic witnesses below have an analytic full-dictionary
support bound. All numerical outputs remain float64, not interval enclosures.
"""
import numpy as np
from scipy.optimize import minimize


def repair_minimum_mass(atoms, target):
    import cvxpy as cp
    c = cp.Variable(atoms.shape[1], complex=True)
    r = cp.Variable(len(target), complex=True)
    equality = atoms@c+r == target
    problem = cp.Problem(cp.Minimize(cp.norm(c, 1)+cp.norm(r, 1)), [equality])
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=300)
    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(problem.status)
    coefficients = np.asarray(c.value)
    # Recompute exact algebraic repair from saved coefficients, rather than
    # calling the solver's possibly infeasible r a valid decomposition.
    residual = target-atoms@coefficients
    mass = float(np.sum(abs(coefficients))+np.sum(abs(residual)))
    w = -np.asarray(equality.dual_value)
    report = {'mass': mass, 'extent_upper_numerical': mass**2,
              'gaussian_coefficient_mass': float(np.sum(abs(coefficients))),
              'fock_repair_mass': float(np.sum(abs(residual))),
              'solver_status': problem.status,
              'finite_dual_objective': float(np.vdot(w, target).real),
              'finite_dual_max_column_overlap': float(max(abs(atoms.conj().T@w))),
              'finite_dual_max_fock_overlap': float(max(abs(w))),
              'finite_dual_is_full_dictionary_certificate': False,
              'arithmetic': 'float64, no interval enclosure'}
    return report, coefficients, residual, w


def matched_magic_witness(model, target, *, permutations=None, starts=6, seed=1):
    r = model.r
    bits = ((np.arange(1 << r)[:, None] >> np.arange(r)) & 1)
    if permutations is None:
        permutations = [np.arange(r), np.arange(r)[::-1]]
    rng = np.random.default_rng(seed)
    best, witness = None, None
    for permutation in permutations:
        ids = []
        for row in bits:
            mask = (1 << r)-1
            for j in np.flatnonzero(row):
                mask ^= (1 << (j+r)) | (1 << int(permutation[j]))
            ids.append(model.index[mask])
        values = target[ids]
        def objective(theta):
            phase = np.exp(-1j*bits@theta)
            overlap = phase@values
            derivative = (-1j*phase*values)@bits
            return -float(abs(overlap)**2), -2*np.real(overlap.conjugate()*derivative)
        initial = [np.zeros(r), np.ones(r)*np.pi/2, np.ones(r)*np.pi]
        initial.extend(rng.uniform(-np.pi, np.pi, r) for _ in range(max(0, starts-3)))
        for theta in initial[:starts]:
            result = minimize(objective, theta, jac=True, method='BFGS', options={'gtol': 1e-9})
            if best is None or -result.fun > best['extent_lower_numerical']:
                witness = np.zeros(model.dim, complex)
                witness[ids] = np.exp(1j*bits@result.x)
                best = {'extent_lower_numerical': float(-result.fun), 'hole_permutation': np.asarray(permutation).tolist(),
                        'phases': result.x.tolist(), 'analytic_gaussian_support_bound': 1.,
                        'support_reason': 'tensor product of unnormalized four-mode magic states, up to a Gaussian mode permutation and phases; Gaussian fidelity multiplicativity',
                        'permutation_search_exhaustive': False, 'arithmetic': 'float64, no interval enclosure'}
    return best, witness
