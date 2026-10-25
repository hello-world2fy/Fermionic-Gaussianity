"""Deterministic k=2 full-fermionic-Gaussian Cartan moment in the pair sector.

Q_2 is the zero eigenspace projector of Lambda=sum_a gamma_a (x) gamma_a.
On two Fock strings differing at d physical modes, Lambda is a sum of d
commuting signed Pauli X operators with coefficient 2. The zero Fourier
coefficient gives the projector. Between seniority-zero strings differing
at s pair levels (d=2s), exchanging j pairs has the positive matrix entry
Beta(j+1/2,s-j+1/2)/pi. Intermediate states are NOT restricted to paired
strings: these entries are those of the full projector, compressed afterwards.

Physical U(1) number and independent spin-difference rotations at each level
fix the target. Haar invariance and uniqueness let us restrict the witness
to the same invariant sector without restricting the Gaussian dictionary.
"""
import numpy as np
from scipy.special import beta
from scipy.optimize import minimize


class Replica2:
    def __init__(self, model):
        if model.L > 12:
            raise ValueError('Dense replica-pair index backend is limited to L<=12')
        self.model = model
        D = model.dim
        aa = np.repeat(np.arange(D, dtype=np.int32), D)
        bb = np.tile(np.arange(D, dtype=np.int32), D)
        ma, mb = model.masks[aa], model.masks[bb]
        difference = ma ^ mb
        shared = ma & mb
        keys = shared | (difference << model.L)
        sizes = np.bitwise_count(difference)
        self.blocks = []
        from math import comb
        for s in range(0, model.L+1, 2):
            chosen = np.flatnonzero(sizes == s)
            order = np.lexsort((ma[chosen], keys[chosen]))
            chosen = chosen[order]
            width = comb(s, s//2)
            a = aa[chosen].reshape(-1, width)
            b = bb[chosen].reshape(-1, width)
            masks = model.masks[a[0]]
            flipped = np.bitwise_count(masks[:, None] ^ masks[None, :])
            kernel = beta(flipped+.5, s-flipped+.5)/np.pi
            if len(a) > 1:
                # All fixed common/union sectors have the same ordering kernel.
                check = model.masks[a[-1]]
                assert np.array_equal(flipped, np.bitwise_count(check[:, None] ^ check[None, :]))
            self.blocks.append((a, b, kernel))

    def evaluate(self, u):
        C = np.zeros(self.model.dim, complex)
        for a, b, K in self.blocks:
            product = u[a]*u[b]
            action = product@K.T
            contribution = action*u[b].conj()
            C.real += np.bincount(a.ravel(), weights=contribution.real.ravel(), minlength=len(C))
            C.imag += np.bincount(a.ravel(), weights=contribution.imag.ravel(), minlength=len(C))
        q = float(np.vdot(u, C).real)
        return q, C

    def derivative(self, u, v):
        """Real-linear derivative D C_2(u)[v], including conjugate dependence."""
        result = np.zeros(self.model.dim, complex)
        for a, b, K in self.blocks:
            action = (u[a]*u[b])@K.T
            change = (v[a]*u[b]+u[a]*v[b])@K.T
            contribution = change*u[b].conj()+action*v[b].conj()
            result.real += np.bincount(a.ravel(), weights=contribution.real.ravel(), minlength=len(result))
            result.imag += np.bincount(a.ravel(), weights=contribution.imag.ravel(), minlength=len(result))
        return result

    def witness_response(self, u, target_response):
        from scipy.sparse.linalg import LinearOperator, cg
        D = len(u)
        def action(x):
            y = self.derivative(u, x[:D]+1j*x[D:])
            return np.r_[y.real, y.imag]
        operator = LinearOperator((2*D, 2*D), matvec=action, dtype=float)
        rhs = np.r_[target_response.real, target_response.imag]
        solution, status = cg(operator, rhs, rtol=1e-10, atol=1e-12, maxiter=200)
        v = solution[:D]+1j*solution[D:]
        return v, {'cg_status': int(status), 'response_residual': float(np.linalg.norm(self.derivative(u, v)-target_response))}

    def solve(self, target, maxiter=500):
        D = len(target)
        history = []
        q, _ = self.evaluate(target)
        initial = target/q**(1/3)
        def objective(x):
            u = x[:D]+1j*x[D:]
            q, C = self.evaluate(u)
            gradient = C-target
            return q/4-np.vdot(u, target).real, np.r_[gradient.real, gradient.imag]
        def callback(x):
            value, gradient = objective(x)
            history.append({'objective': float(value), 'residual': float(np.linalg.norm(gradient))})
        result = minimize(objective, np.r_[initial.real, initial.imag], jac=True,
                          method='L-BFGS-B', callback=callback,
                          options={'ftol': 1e-15, 'gtol': 2e-10, 'maxiter': maxiter, 'maxcor': 20})
        u = result.x[:D]+1j*result.x[D:]
        q, C = self.evaluate(u)
        lower = abs(np.vdot(u, target))**2/np.sqrt(q)
        return u, {'k': 2, 'physical_modes': 2*self.model.L,
                   'q': q, 'residual': float(np.linalg.norm(C-target)),
                   'moment_dual_extent_lower': float(lower),
                   'float64_deterministic_not_interval_certificate': True,
                   'full_gaussian_dictionary': True,
                   'iterations': int(result.nit), 'optimizer_success': bool(result.success),
                   'history': history}
