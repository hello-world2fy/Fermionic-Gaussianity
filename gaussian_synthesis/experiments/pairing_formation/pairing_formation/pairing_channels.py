"""Separable competing-pairing channels for a bounded exploratory extension.

The uniform reduced BCS model remains a checked special case. These finite
angular-patch models are not material models or thermodynamic phase claims.
"""
from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.integrate import solve_ivp
from .model import PairModel


class PairKernelModel(PairModel):
    def __init__(self, energies, kernel):
        super().__init__(np.asarray(energies), 0.)
        self.kernel = np.asarray(kernel, complex)
        if self.kernel.shape != (self.L, self.L):
            raise ValueError('Pairing kernel must have one index per pair level')
        if not np.allclose(self.kernel, self.kernel.conj().T, atol=1e-13):
            raise ValueError('Pairing kernel must be Hermitian')
        self.H = diags(2*self.occ@self.energies-self.energies.sum())-self.pair_operator(self.kernel)
        self.diagonal = self.H.diagonal()

    def pair_operator(self, matrix):
        """Sum matrix[i,j] s_i^+ s_j^- in the full fixed-pair-number space."""
        W = np.asarray(matrix)
        rows, cols, values = [], [], []
        for (i, j), (dst, src) in self.transitions.items():
            if abs(W[i, j]) < 1e-15:
                continue
            rows.append(dst); cols.append(src); values.append(np.full(len(src), W[i, j]))
        diagonal = self.occ@np.diag(W)
        if not rows:
            return diags(diagonal).tocsr()
        return (coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(self.dim, self.dim)).tocsr()+diags(diagonal)).tocsr()


def angular_model(L, G=4., relative_strength=1.):
    """Two radial shells, with real cos(2theta), sin(2theta) singlet channels.

    L=8 has disjoint support for the two sampled real form factors and serves
    as a finite-grid control, not evidence for generic competing channels.
    """
    r = L//2
    theta = np.pi*np.arange(r)/r
    forms = np.column_stack((np.sqrt(2)*np.cos(2*theta), np.sqrt(2)*np.sin(2*theta)))
    all_forms = np.tile(forms, (2, 1))
    J = (all_forms*np.array([G, G*relative_strength]))@all_forms.T/L
    model = PairKernelModel(np.r_[-.5*np.ones(r), .5*np.ones(r)], J)
    return model, forms


def linear_matrices(model):
    r, J, eps = model.r, model.kernel, model.energies
    A = np.diag(2*eps[r:])-J[r:, r:]
    D = np.diag(-2*eps[:r]+2*np.diag(J[:r, :r]))-J[:r, :r].T
    B = -J[r:, :r]
    return A, D, B, np.block([[A, B], [-B.conj().T, -D.T]])


def linear_pair_matrix(model, times):
    A, D, B, _ = linear_matrices(model)
    r = model.r
    def rhs(t, flat):
        T = flat.reshape(r, r)
        return (-1j*(B+A@T+T@D.T+T@B.conj().T@T)).ravel()
    times = np.asarray(times)
    if times[-1] == 0:
        return np.zeros((len(times), r, r), complex)
    sol = solve_ivp(rhs, (0., float(times[-1])), np.zeros(r*r, complex), t_eval=times,
                    method='DOP853', rtol=2e-11, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(-1, r, r)


def annihilator(L, number, weights):
    """Pair annihilation between complete adjacent fixed-pair sectors."""
    source = [sum(1 << j for j in c) for c in combinations(range(L), number)]
    destination = [sum(1 << j for j in c) for c in combinations(range(L), number-1)]
    lookup = {mask: j for j, mask in enumerate(destination)}
    rows, cols, values = [], [], []
    for col, mask in enumerate(source):
        for j, weight in enumerate(weights):
            if abs(weight) > 1e-15 and mask & (1 << j):
                rows.append(lookup[mask ^ (1 << j)]); cols.append(col); values.append(weight)
    return coo_matrix((values, (rows, cols)), shape=(len(destination), len(source))).tocsr()


class ChannelObservables:
    def __init__(self, model, forms):
        self.model = model
        weights = np.vstack((np.zeros_like(forms), forms))/np.sqrt(model.r)
        self.first = [annihilator(model.L, model.r, weights[:, a]) for a in (0, 1)]
        self.second = [annihilator(model.L, model.r-1, weights[:, a]) for a in (0, 1)]
        x, y = weights.T
        self.chirality = model.pair_operator(1j*(np.outer(x, y)-np.outer(y, x)))
        self.nematic = model.pair_operator(np.outer(x, x)-np.outer(y, y))

    def evaluate(self, state):
        one = [B@state for B in self.first]
        two = [B@v for B, v in zip(self.second, one)]
        nx, ny = [float(np.linalg.norm(v)**2) for v in one]
        second = [float(np.linalg.norm(v)**2) for v in two]
        locking = np.vdot(two[0], two[1])
        denominator = np.sqrt(second[0]*second[1])
        c = self.chirality@state
        n = self.nematic@state
        c2 = float(np.linalg.norm(c)**2)
        c4 = float(np.linalg.norm(self.chirality@c)**2)
        return {'upper_channel_occupations': [nx, ny],
                'upper_channel_two_pair_moments': second,
                'relative_phase_four_pair_real': float(locking.real),
                'relative_phase_four_pair_imag': float(locking.imag),
                'normalized_phase_locking': float(locking.real/denominator) if denominator > 1e-14 else None,
                'chirality_mean': float(np.vdot(state, c).real),
                'chirality_second': c2, 'chirality_fourth': c4,
                'chirality_binder': 1-c4/(3*c2*c2) if c2 > 1e-12 else None,
                'nematic_second': float(np.linalg.norm(n)**2)}
