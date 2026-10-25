"""Density-Jastrow controls with freely coherent transfer-number amplitudes.

All contractions enumerate the paired fixed-N sector. Parameter counts are not
claims of polynomial simulation cost. These are conventional exponential
correlator families, not a new Gaussian synthesis method.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from .conditional_dynamics import ConditionalDynamics
from .fitting import excitation_features


def centered_features(model, raw):
    """Quotient functions constant within m; these belong to radial amplitudes."""
    answer = np.asarray(raw).copy()
    for m in range(model.r+1):
        rows = model.m == m
        answer[rows] -= answer[rows].mean(axis=0)
    return answer


def feature_basis(model, family):
    if family in ('two_body', 'conditional_two_body'):
        # On a fixed-r slice sum_{j!=i} n_i n_j=(r-1)n_i, so single
        # occupations are already included for r>=2.
        raw = np.column_stack([model.occ[:, i]*model.occ[:, j]
                               for i, j in combinations(range(model.L), 2)])
        if family == 'conditional_two_body':
            X = excitation_features(model)
            raw = np.column_stack([raw]+[X*(model.m == m)[:, None] for m in range(1, model.r)])
    elif family.startswith('collective_'):
        degree = int(family.split('_')[-1])
        X = excitation_features(model)
        count = 2*model.m/model.r-1
        raw = np.column_stack([X*count[:, None]**p for p in range(degree+1)])
    else:
        raise ValueError(f'Unknown Jastrow family: {family}')
    raw = centered_features(model, raw)
    u, s, vh = np.linalg.svd(raw, full_matrices=False)
    keep = s > 1e-11*s[0]
    # Unit uniform variance stabilizes metrics across families and sizes.
    return u[:, keep]*np.sqrt(model.dim)


class JastrowDynamics(ConditionalDynamics):
    def __init__(self, model, family='two_body', metric_cutoff=1e-10, features=None):
        super().__init__(model, conditioned=False, metric_cutoff=metric_cutoff)
        self.family = family
        self.features = feature_basis(model, family) if features is None else np.asarray(features)
        if self.features.shape[0] != model.dim:
            raise ValueError('Feature basis and Hilbert space differ')
        self.shape = (1, self.features.shape[1])

    def project_logarithms(self, raw):
        """Least-squares projection of log amplitudes modulo radial functions."""
        raw = centered_features(self.model, np.asarray(raw))
        theta = self.features.T@raw/self.model.dim
        return theta, float(np.linalg.norm(self.features@theta-raw))

    def apply_diagonal_phase(self, coordinates, phase):
        """Apply exp(-i phase) exactly when phase is a retained density function."""
        result = coordinates.copy()
        coefficients, theta = self.unpack(result)
        projected, error = self.project_logarithms(phase)
        if error > 1e-9*max(1., np.linalg.norm(phase)):
            raise ValueError('Kick is outside the chosen variational family')
        theta[0] -= 1j*projected
        for m, rows in enumerate(self.groups):
            coefficients[m] *= np.exp(-1j*np.mean(phase[rows]))
        return result


def radial_objective(dynamics, target, parameters):
    d = dynamics.shape[1]
    theta = parameters[:d]+1j*parameters[d:]
    total, derivative = 0., np.zeros(d, complex)
    for rows in dynamics.groups:
        F = dynamics.features[rows]
        raw = F@theta
        q = np.exp(raw-np.max(raw.real))
        q /= np.linalg.norm(q)
        overlap = np.vdot(target[rows], q)
        centered = F-abs(q)**2@F
        dq = (target[rows].conj()*q)@centered
        total += abs(overlap)**2
        derivative += overlap.conjugate()*dq
    return -float(total), -2*np.r_[derivative.real, (1j*derivative).real]


def fit_jastrow(dynamics, target, initial_raw=None, starts=8, seed=1, maxiter=1000):
    target = target/np.linalg.norm(target)
    d = dynamics.shape[1]
    rng = np.random.default_rng(seed)
    initial = np.zeros(d, complex) if initial_raw is None else dynamics.project_logarithms(initial_raw)[0]
    x0 = np.r_[initial.real, initial.imag]
    guesses = [x0, np.zeros(2*d)]
    guesses.extend(x0+rng.normal(0., .15, 2*d) for _ in range(max(0, starts-2)))
    history, best = [], None
    for x in guesses[:starts]:
        result = minimize(lambda z: radial_objective(dynamics, target, z), x, jac=True,
                          method='L-BFGS-B', bounds=[(-20., 20.)]*d+[(None, None)]*d,
                          options={'maxiter': maxiter, 'ftol': 2e-14, 'gtol': 2e-9, 'maxls': 40})
        history.append({'fidelity': float(-result.fun), 'iterations': int(result.nit),
                        'success': bool(result.success), 'message': str(result.message),
                        'gradient_norm': float(np.linalg.norm(result.jac)),
                        'amplitude_bound_active': bool(np.any(abs(result.x[:d]) > 19.9999))})
        if best is None or result.fun < best.fun:
            best = result
    theta = best.x[:d]+1j*best.x[d:]
    coordinates, fidelity = dynamics.encode(theta, target)
    approximation = dynamics.state_and_blocks(coordinates)[0]
    report = {'family': dynamics.family, 'fidelity': fidelity, 'restarts': history,
              'log_features': d, 'complex_cone_dimension': dynamics.coefficient_count+d,
              'global_optimum_certified': False, 'full_sector_enumeration': True}
    return report, coordinates, approximation
