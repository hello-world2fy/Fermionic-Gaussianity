"""Several coherent onsite Gaussian symmetry orbits per transfer sector.

This is a conventional multiconfiguration TDVP control. The orbitals may be
learned from a target or canonical witness, but this is not canonical sampling.
Full paired-sector enumeration is used throughout.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from .conditional import _fit
from .fitting import excitation_features


def orbit_columns(features, logarithms):
    raw = features@logarithms.T
    value = np.exp(raw-np.max(raw.real, axis=0))
    return value/np.linalg.norm(value, axis=0)


def block_objective(features, target, parameters, rank):
    d = features.shape[1]
    z = parameters[:rank*d].reshape(rank, d)+1j*parameters[rank*d:].reshape(rank, d)
    Q = orbit_columns(features, z)
    coefficients = np.linalg.lstsq(Q, target, rcond=1e-11)[0]
    residual = target-Q@coefficients
    derivative = np.array([c*(residual.conj()*q)@features for c, q in zip(coefficients, Q.T)])
    return float(np.vdot(residual, residual).real), -2*np.r_[derivative.real.ravel(), (1j*derivative).real.ravel()]


def fit_orbits(model, target, rank=2, starts=6, seed=1, maxiter=1000):
    target = target/np.linalg.norm(target)
    F = excitation_features(model)
    rng = np.random.default_rng(seed)
    logs = np.zeros((model.r+1, rank, model.L), complex)
    records = []
    for m, (subsets, indices) in enumerate(model.blocks):
        rows = indices.ravel()
        p = float(np.linalg.norm(target[rows])**2)
        if m in (0, model.r) or p < 1e-28:
            records.append({'m': m, 'probability': p, 'fidelity': 1., 'restarts': []})
            continue
        block = target[indices]/np.sqrt(p)
        u, singular, vh = np.linalg.svd(block, full_matrices=False)
        side_features = np.array([[int(j in subset) for j in range(model.r)] for subset in subsets], float)
        for a in range(rank):
            local = []
            for vector in (u[:, a], vh[a]):
                guess = np.linalg.lstsq(side_features, np.log(np.maximum(abs(vector), 1e-12))+1j*np.angle(vector), rcond=None)[0]
                parameters = np.r_[guess.real, guess.imag]
                parameters, _, _ = _fit(side_features, vector, parameters, 3, rng)
                local.append(parameters[:model.r]+1j*parameters[model.r:])
            logs[m, a] = np.r_[local[0], local[1]]
        x0 = np.r_[logs[m].real.ravel(), logs[m].imag.ravel()]
        guesses = [x0]+[x0+rng.normal(0, .25, len(x0)) for _ in range(starts-1)]
        best, history = None, []
        for x in guesses:
            fit = minimize(lambda y: block_objective(F[rows], block.ravel(), y, rank), x, jac=True,
                           method='L-BFGS-B', bounds=[(-14., 14.)]*(rank*model.L)+[(None, None)]*(rank*model.L),
                           options={'maxiter': maxiter, 'ftol': 2e-14, 'gtol': 2e-9, 'maxls': 40})
            history.append({'fidelity': 1.-float(fit.fun), 'iterations': int(fit.nit), 'success': bool(fit.success),
                            'gradient_norm': float(np.linalg.norm(fit.jac))})
            if best is None or fit.fun < best.fun:
                best = fit
        logs[m] = best.x[:rank*model.L].reshape(rank, model.L)+1j*best.x[rank*model.L:].reshape(rank, model.L)
        records.append({'m': m, 'probability': p, 'fidelity': 1.-float(best.fun),
                        'schmidt_rank_bound': float(np.sum(singular[:rank]**2)), 'restarts': history})
    dynamics = MultiOrbitDynamics(model, rank)
    coordinates, fidelity = dynamics.encode(logs, target)
    return {'rank': rank, 'fidelity': fidelity, 'blocks': records, 'global_optimum_certified': False}, coordinates


class MultiOrbitDynamics:
    def __init__(self, model, rank=2, svd_cutoff=1e-8):
        self.model, self.rank, self.svd_cutoff = model, rank, svd_cutoff
        self.features = excitation_features(model)
        self.groups = [np.flatnonzero(model.m == m) for m in range(model.r+1)]
        self.coefficient_count = (model.r+1)*rank
        self.calls = 0

    def unpack(self, coordinates):
        return (coordinates[:self.coefficient_count].reshape(self.model.r+1, self.rank),
                coordinates[self.coefficient_count:].reshape(self.model.r+1, self.rank, self.model.L))

    def encode(self, logarithms, target):
        coefficients = np.array([np.linalg.lstsq(orbit_columns(self.features[rows], logarithms[m]), target[rows], rcond=1e-11)[0]
                                 for m, rows in enumerate(self.groups)])
        coordinates = np.r_[coefficients.ravel(), logarithms.ravel()]
        norm = np.linalg.norm(self.state(coordinates))
        coordinates[:self.coefficient_count] /= norm
        return coordinates, float(norm**2)

    def state(self, coordinates):
        coefficients, logs = self.unpack(coordinates)
        state = np.zeros(self.model.dim, complex)
        for m, rows in enumerate(self.groups):
            state[rows] = orbit_columns(self.features[rows], logs[m])@coefficients[m]
        return state

    def derivative(self, t, coordinates, diagnostics=False):
        self.calls += 1
        coefficients, logs = self.unpack(coordinates)
        state = self.state(coordinates)
        action = self.model.H@state
        c_rate, z_rate = np.zeros_like(coefficients), np.zeros_like(logs)
        physical = np.zeros_like(state)
        ranks = []
        for m, rows in enumerate(self.groups):
            F = self.features[rows]
            Q = orbit_columns(F, logs[m])
            means = abs(Q).T**2@F
            tangent = np.column_stack([Q]+[coefficients[m, a]*Q[:, a, None]*(F-means[a]) for a in range(self.rank)])
            scales = np.linalg.norm(tangent, axis=0)
            scales[scales < 1e-14] = 1.
            normalized = tangent/scales
            u, singular, vh = np.linalg.svd(normalized, full_matrices=False)
            keep = singular > max(1e-13, self.svd_cutoff*singular[0])
            horizontal = (vh[keep].conj().T@((u[:, keep].conj().T@(-1j*action[rows]))/singular[keep]))/scales
            c_rate[m] = horizontal[:self.rank]
            z_rate[m] = horizontal[self.rank:].reshape(self.rank, self.model.L)
            # Normalized orbit columns have a separate Berry phase connection.
            c_rate[m] -= 1j*coefficients[m]*np.sum(means*z_rate[m].imag, axis=1)
            physical[rows] = tangent@horizontal
            ranks.append(int(keep.sum()))
        rate = np.r_[c_rate.ravel(), z_rate.ravel()]
        if diagnostics:
            return rate, {'residual_norm': float(np.linalg.norm(1j*physical-action)),
                          'norm_rate': float(2*np.vdot(state, physical).real),
                          'energy_rate': float(2*np.vdot(action, physical).real),
                          'complex_tangent_rank': sum(ranks), 'block_tangent_ranks': ranks}, physical
        return rate

    def evolve(self, initial, elapsed, rtol=2e-8, atol=2e-10, max_step=.04):
        solution = solve_ivp(self.derivative, (0., float(elapsed[-1])), initial, t_eval=elapsed,
                             method='DOP853', rtol=rtol, atol=atol, max_step=max_step)
        if not solution.success:
            raise RuntimeError(solution.message)
        states = np.array([self.state(x) for x in solution.y.T])
        energy = np.array([np.vdot(x, self.model.H@x).real for x in states])
        report = {'calls': self.calls, 'rtol': rtol, 'atol': atol, 'max_step': max_step, 'svd_cutoff': self.svd_cutoff,
                  'max_norm_error': float(np.max(abs(np.linalg.norm(states, axis=1)-1))),
                  'max_energy_drift': float(np.max(abs(energy-energy[0]))),
                  'diagnostics': [self.derivative(t, x, True)[1] for t, x in zip(elapsed, solution.y.T)]}
        return states, solution.y.T, report
