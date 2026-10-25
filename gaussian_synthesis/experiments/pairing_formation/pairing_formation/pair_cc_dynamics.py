"""Hermitian Dirac--Frenkel dynamics of exp(T_ph s_p^+ s_h^-) |F>.

This is a conventional variational pair-CC manifold, not projective or
bivariational time-dependent CC. Full fixed-number vectors are used here;
this implementation makes no polynomial-cost claim. An optional fixed linear
map restricts T to chosen particle/hole directions, with the same Fermi initial
state. The norm and Berry connection are retained explicitly.
"""
import numpy as np
from scipy.integrate import solve_ivp, DOP853


class PairCCDynamics:
    def __init__(self, model, parameter_map=None, metric_cutoff=1e-11, metric_solver='gram'):
        self.model = model
        r = model.r
        self.map = np.eye(r*r, dtype=complex) if parameter_map is None else np.asarray(parameter_map, complex)
        if self.map.shape[0] != r*r:
            raise ValueError('parameter_map maps coordinates to flattened T')
        self.metric_cutoff = metric_cutoff
        if metric_solver not in ('gram', 'svd'):
            raise ValueError('metric_solver must be gram or svd')
        self.metric_solver = metric_solver
        self.calls = 0
        # d permanent(T[P,H])/dT[p,h] = permanent of the minor.
        # The minor is another configuration in the same fixed-number basis.
        self.destinations, self.parents, self.columns = [], [], []
        for p in range(r):
            for h in range(r):
                rows = np.flatnonzero((model.occ[:, r+p] == 1) & (model.occ[:, h] == 0))
                parents = [model.index[int(model.masks[x]) ^ (1 << (r+p)) ^ (1 << h)] for x in rows]
                self.destinations.extend(rows)
                self.parents.extend(parents)
                self.columns.extend([p*r+h]*len(rows))
        self.destinations = np.asarray(self.destinations)
        self.parents = np.asarray(self.parents)
        self.columns = np.asarray(self.columns)

    def initial(self):
        return np.r_[1.+0j, np.zeros(self.map.shape[1], complex)]

    def raw_and_jacobian(self, coordinates):
        model = self.model
        T = (self.map@coordinates).reshape(model.r, model.r)
        raw = np.empty(model.dim, complex)
        raw[model.fermi_index] = 1.
        for current, particle, holes, parents in model.permanent_recursion:
            raw[current] = np.sum(T[particle[:, None], holes]*raw[parents], axis=1)
        jacobian = np.zeros((model.dim, model.r**2), complex)
        jacobian[self.destinations, self.columns] = raw[self.parents]
        if self.map.shape[1] != model.r**2:
            jacobian = jacobian@self.map
        else:
            # A square nonidentity map is permitted too.
            if not np.array_equal(self.map, np.eye(model.r**2)):
                jacobian = jacobian@self.map
        return raw, jacobian

    def state(self, coordinates):
        raw, _ = self.raw_and_jacobian(coordinates[1:])
        return coordinates[0]*raw/np.linalg.norm(raw)

    def derivative(self, time, coordinates, diagnostics=False):
        self.calls += 1
        c = coordinates[0]
        raw, D = self.raw_and_jacobian(coordinates[1:])
        norm = np.linalg.norm(raw)
        q = raw/norm
        D /= norm
        connection = q.conj()@D
        J = D-q[:, None]*connection
        action = self.model.H@q
        energy = float(np.vdot(q, action).real)
        if self.metric_solver == 'gram':
            metric = J.conj().T@J
            values, vectors = np.linalg.eigh(metric)
            keep = values > max(1e-15, self.metric_cutoff*max(values[-1], 0.))
            V = vectors[:, keep]
            rate = -1j*V@((V.conj().T@(J.conj().T@action))/values[keep])
        else:
            # The same spectral cutoff, applied without squaring the
            # condition number of a nearly rank-deficient tangent matrix.
            U, singular, Vh = np.linalg.svd(J, full_matrices=False)
            values = singular**2
            keep = values > max(1e-15, self.metric_cutoff*values[0])
            rate = -1j*Vh[keep].conj().T@((U[:, keep].conj().T@action)/singular[keep])
        cdot = -1j*c*(energy+(connection@rate).imag)
        answer = np.r_[cdot, rate]
        if not diagnostics:
            return answer
        physical_rate = cdot*q+c*(D@rate-q*(connection@rate).real)
        state = c*q
        return answer, {
            'residual_norm': float(np.linalg.norm(1j*physical_rate-self.model.H@state)),
            'norm_rate': float(2*np.vdot(state, physical_rate).real),
            'energy_rate': float(2*np.vdot(self.model.H@state, physical_rate).real),
            'metric_kept': int(keep.sum()),
            'metric_smallest': float(values.min()),
            'reference_probability': float(abs(state[self.model.fermi_index])**2),
        }, physical_rate

    def evolve(self, times, initial=None, rtol=2e-8, atol=2e-10, max_step=.04, error_control='coordinates'):
        times = np.asarray(times)
        initial = self.initial() if initial is None else np.asarray(initial)
        method='DOP853'
        if error_control == 'physical':
            dynamics=self
            class PhysicalDOP853(DOP853):
                """Use the physical state norm for DOP853's local error test.

                Near redundant or initially empty variational coordinates,
                coordinate error need not measure wavefunction error. Retain
                phase/norm, and check the finite-map nonlinear remainder too.
                """
                def _estimate_error_norm(self,K,h,scale):
                    raw,D=dynamics.raw_and_jacobian(self.y[1:]);norm=np.linalg.norm(raw)
                    q=raw/norm;D/=norm;connection=q.conj()@D;c=self.y[0]
                    def tangent(v):
                        return v[0]*q+c*(D@v[1:]-q*(connection@v[1:]).real)
                    e5=K.T@self.E5;e3=K.T@self.E3
                    p5=tangent(e5);p3=tangent(e3)
                    n5=float(np.vdot(p5,p5).real);n3=float(np.vdot(p3,p3).real)
                    tolerance=float(np.max(self.atol)+self.rtol*abs(c))
                    linear=abs(h)*n5/np.sqrt(n5+.01*n3)/tolerance if n5+n3 else 0.
                    remainder=np.linalg.norm(dynamics.state(self.y+h*e5)-c*q-h*p5)/tolerance
                    return max(linear,float(remainder))
            method=PhysicalDOP853
        elif error_control != 'coordinates':
            raise ValueError('error_control must be coordinates or physical')
        solution = solve_ivp(self.derivative, (0., float(times[-1])), initial, t_eval=times,
                             method=method, rtol=rtol, atol=atol, max_step=max_step)
        if not solution.success:
            raise RuntimeError(solution.message)
        states = np.array([self.state(x) for x in solution.y.T])
        energies = np.array([np.vdot(s, self.model.H@s).real for s in states])
        diagnostics = [self.derivative(t, x, diagnostics=True)[1] for t, x in zip(times, solution.y.T)]
        report = {'method': 'Hermitian Dirac-Frenkel pair-CC', 'calls': self.calls,
                  'complex_T_coordinates': self.map.shape[1], 'metric_cutoff': self.metric_cutoff,
                  'metric_solver': self.metric_solver,
                  'error_control': error_control,
                  'rtol': rtol, 'atol': atol, 'max_step': max_step,
                  'max_norm_error': float(np.max(abs(np.linalg.norm(states, axis=1)-1))),
                  'max_energy_drift': float(np.max(abs(energies-energies[0]))),
                  'minimum_reference_probability': min(x['reference_probability'] for x in diagnostics),
                  'diagnostics': diagnostics}
        return states, solution.y.T, report


def product_parameter_map(particle, hole):
    """Map X to T=particle X hole.T, using row-major vectorization."""
    return np.kron(particle, hole)


def circulant_parameter_map(r):
    """Translation-invariant T; applicable only when the Hamiltonian has it."""
    M = np.zeros((r*r,r),complex)
    p,h = np.indices((r,r))
    M[np.arange(r*r),((p-h)%r).ravel()] = 1/np.sqrt(r)
    return M


def projective_rhs(model, T):
    """Exact single-transfer projection of exp(-T) H exp(T), O(r^3).

    This is the usual projective right-ket CC equation. Normalized right-ket
    observables are a separate convention from bivariational CC observables.
    The two additional elementwise terms relative to bosonic Riccati are the
    hard-core corrections; no canonical synthesis information is involved.
    """
    from .pairing_channels import linear_matrices
    A, D, B, _ = linear_matrices(model)
    J = model.kernel[model.r:, :model.r]
    weighted = J.conj()*T
    row = weighted.sum(axis=1)
    column = weighted.sum(axis=0)
    pauli = 2*T*(row[:,None]+column[None,:]-weighted)
    return -1j*(B+A@T+T@D.T+T@B.conj().T@T+pauli)


def evolve_projective(model, times, rtol=2e-10, atol=2e-12, max_step=.02):
    def rhs(t, flat):
        return projective_rhs(model, flat.reshape(model.r, model.r)).ravel()
    def event(t, flat):
        return 1e6-np.linalg.norm(flat)
    event.terminal = True
    event.direction = -1
    sol = solve_ivp(rhs, (0.,float(times[-1])), np.zeros(model.r**2, complex), t_eval=times,
                    method='DOP853',rtol=rtol,atol=atol,max_step=max_step,events=event)
    return sol.t, sol.y.T.reshape(-1,model.r,model.r), {
        'success':bool(sol.success), 'message':sol.message, 'complete':bool(len(sol.t)==len(times)),
        'calls':sol.nfev,'rtol':rtol,'atol':atol,'max_step':max_step,
        'amplitude_limit_event':sol.t_events[0].tolist(),
        'largest_saved_T_norm':float(np.max(np.linalg.norm(sol.y.T,axis=1))),
        'observables':'Normalized right-ket observables, not bivariational CC expectations'}
