"""Hamiltonian-only TDVP on radial and count-conditioned pair-orbital manifolds.

The chosen family is fixed at training time, while its orbitals move under the
Dirac-Frenkel projection. This is a stronger multiconfiguration baseline, not a
claim that canonical sampling discovered a new time-dependent variational method.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .fitting import excitation_features


class ConditionalDynamics:
    def __init__(self, model, conditioned=True, metric_cutoff=1e-10):
        self.model = model
        self.conditioned = conditioned
        self.metric_cutoff = metric_cutoff
        self.features = excitation_features(model)
        self.groups = [np.flatnonzero(model.m == m) for m in range(model.r+1)]
        self.coefficient_count = model.r+1
        self.shape = (model.r+1 if conditioned else 1, model.L)
        self.calls = 0

    def unpack(self, coordinates):
        return coordinates[:self.coefficient_count], coordinates[self.coefficient_count:].reshape(self.shape)

    def encode(self, logarithms, target):
        logs = np.asarray(logarithms).reshape(self.shape)
        provisional = np.r_[np.ones(self.coefficient_count, complex), logs.ravel()]
        _, data = self.state_and_blocks(provisional)
        coefficients = np.array([np.vdot(q, target[rows]) for rows, q, _, _, _ in data])
        initial_fidelity = float(np.linalg.norm(coefficients)**2)
        coefficients /= np.sqrt(initial_fidelity)
        return np.r_[coefficients, logs.ravel()], initial_fidelity

    def state_and_blocks(self, coordinates):
        coefficients, logarithms = self.unpack(coordinates)
        state = np.zeros(self.model.dim, complex)
        data = []
        for m, rows in enumerate(self.groups):
            F = self.features[rows]
            raw = F@logarithms[m if self.conditioned else 0]
            q = np.exp(raw-np.max(raw.real))
            q /= np.linalg.norm(q)
            mean = abs(q)**2@F
            centered = F-mean
            metric = centered.T@((abs(q)**2)[:, None]*centered)
            state[rows] = coefficients[m]*q
            data.append((rows, q, mean, centered, metric))
        return state, data

    def solve_metric(self, metric, force):
        values, vectors = np.linalg.eigh(metric)
        keep = values > max(1e-14, self.metric_cutoff*max(values[-1], 0.))
        if not np.any(keep):
            return np.zeros(len(force), complex)
        V = vectors[:, keep]
        return V@((V.T@force)/values[keep])

    def derivative(self, t, coordinates, diagnostics=False):
        self.calls += 1
        coefficients, logarithms = self.unpack(coordinates)
        state, data = self.state_and_blocks(coordinates)
        action = self.model.H@state
        forces, h = [], []
        for rows, q, mean, centered, metric in data:
            h.append(np.vdot(q, action[rows]))
            forces.append((q.conj()*action[rows])@centered)
        log_rate = np.zeros_like(logarithms)
        if self.conditioned:
            for m, (_, _, _, _, metric) in enumerate(data):
                if abs(coefficients[m]) > 1e-13:
                    log_rate[m] = -1j*self.solve_metric(metric, forces[m]/coefficients[m])
        else:
            metric = sum(abs(coefficients[m])**2*data[m][4] for m in range(self.coefficient_count))
            force = sum(coefficients[m].conjugate()*forces[m] for m in range(self.coefficient_count))
            log_rate[0] = -1j*self.solve_metric(metric, force)
        coefficient_rate = -1j*np.array(h)
        physical_rate = np.zeros_like(state)
        for m, (rows, q, mean, centered, metric) in enumerate(data):
            rate = log_rate[m if self.conditioned else 0]
            # q=normalized exp(F z) has Berry connection i <F> Im zdot.
            # Transfer that phase into c so the horizontal tangent is centered.
            coefficient_rate[m] -= 1j*coefficients[m]*(mean@rate.imag)
            q_rate = q*((centered+mean)@rate-mean@rate.real)
            physical_rate[rows] = coefficient_rate[m]*q+coefficients[m]*q_rate
        answer = np.r_[coefficient_rate, log_rate.ravel()]
        if diagnostics:
            return answer, {'residual_norm': float(np.linalg.norm(1j*physical_rate-action)),
                            'norm_rate': float(2*np.vdot(state, physical_rate).real),
                            'energy_rate': float(2*np.vdot(action, physical_rate).real)}, physical_rate
        return answer

    def evolve(self, initial, elapsed, rtol=2e-8, atol=2e-10, max_step=.04):
        solution = solve_ivp(self.derivative, (0., float(elapsed[-1])), initial, t_eval=elapsed,
                             method='DOP853', rtol=rtol, atol=atol, max_step=max_step)
        if not solution.success:
            raise RuntimeError(solution.message)
        states = np.array([self.state_and_blocks(x)[0] for x in solution.y.T])
        energy = np.array([np.vdot(x, self.model.H@x).real for x in states])
        residuals = [self.derivative(t, x, diagnostics=True)[1] for t, x in zip(elapsed, solution.y.T)]
        report = {'calls': self.calls, 'rtol': rtol, 'atol': atol, 'max_step': max_step,
                  'metric_cutoff': self.metric_cutoff,
                  'max_norm_error': float(np.max(abs(np.linalg.norm(states, axis=1)-1))),
                  'max_energy_drift': float(np.max(abs(energy-energy[0]))), 'diagnostics': residuals}
        return states, solution.y.T, report
