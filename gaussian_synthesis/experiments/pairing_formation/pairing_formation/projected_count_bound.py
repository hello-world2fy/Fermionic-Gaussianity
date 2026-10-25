"""Independent full-Fock checks for a proposed orbital-invariant AGP bound.

The analytic argument, not random checks, establishes the claimed scope.
No extra seniority projection is applied to the measured number observable.
"""
from itertools import combinations
import numpy as np
from .gaussian import pfaffian_batch


class CanonicalAGPCheck:
    def __init__(self, pair_orbitals, occupied_pairs):
        self.K, self.M = pair_orbitals, occupied_pairs
        self.n, self.N = 2*pair_orbitals, 2*occupied_pairs
        self.subsets = list(combinations(range(self.K), self.M))
        self.occupations = np.array([[int(k in s) for k in range(self.K)] for s in self.subsets], float)
        self.configurations = np.array(list(combinations(range(self.n), self.N)), int)
        masks = [sum(1 << int(j) for j in c) for c in self.configurations]
        lookup = {mask: i for i, mask in enumerate(masks)}
        paired_masks = [sum(3 << (2*k) for k in s) for s in self.subsets]
        self.paired_indices = np.array([lookup[mask] for mask in paired_masks])
        dst, src, ci, cj, signs = [], [], [], [], []
        for a, mask in enumerate(paired_masks):
            for j in range(self.n):
                if not mask & (1 << j):
                    continue
                removed = mask ^ (1 << j)
                for i in range(self.n):
                    if removed & (1 << i):
                        continue
                    sign = (-1)**((mask & ((1 << j)-1)).bit_count()+(removed & ((1 << i)-1)).bit_count())
                    dst.append(lookup[removed | (1 << i)]); src.append(a)
                    ci.append(i); cj.append(j); signs.append(sign)
        self.dst, self.src, self.ci, self.cj, self.signs = [np.array(x) for x in (dst, src, ci, cj, signs)]

    def coefficients(self, logarithms):
        exponents = self.occupations@logarithms
        values = np.exp(exponents-exponents.real.max())
        return values/np.linalg.norm(values)

    def check(self, logarithms, projector):
        coefficients = self.coefficients(logarithms)
        probability = abs(coefficients)**2
        occupations = probability@self.occupations
        joint = self.occupations.T@(probability[:, None]*self.occupations)
        covariance = joint-np.outer(occupations, occupations)
        action = np.zeros(len(self.configurations), complex)
        np.add.at(action, self.dst, self.signs*projector[self.ci, self.cj]*coefficients[self.src])
        mean = float(np.vdot(coefficients, action[self.paired_indices]).real)
        variance = float(np.vdot(action, action).real-mean**2)
        diagonal = np.array([np.trace(projector[2*k:2*k+2, 2*k:2*k+2]).real for k in range(self.K)])
        paired_variance = float(diagonal@covariance@diagonal)
        broken_variance = float(np.vdot(action, action).real-np.vdot(action[self.paired_indices], action[self.paired_indices]).real)
        broken_bound = 0.
        for i in range(self.K):
            for j in range(i+1, self.K):
                norm2 = float(np.linalg.norm(projector[2*i:2*i+2, 2*j:2*j+2])**2)
                broken_bound += 2*norm2*(occupations[i]+occupations[j]-2*joint[i, j])
        diagonal_bound = float(np.sum(diagonal**2*occupations*(1-occupations)))
        simple_bound = 2*mean-float(np.sum((diagonal*occupations)**2))
        off_covariance = covariance.copy(); np.fill_diagonal(off_covariance, 0.)
        return {'mean_number': mean, 'variance_number': variance,
                'pair_fano': variance/(2*mean) if mean > 1e-15 else 0.,
                'diagonal_variance': paired_variance, 'broken_pair_variance': broken_variance,
                'diagonal_bound': diagonal_bound, 'broken_pair_bound': broken_bound,
                'simple_bound': simple_bound, 'universal_bound': 2*mean,
                'max_offdiagonal_pair_covariance': float(np.max(off_covariance)),
                'variance_decomposition_error': abs(variance-paired_variance-broken_variance),
                'projector_error': float(np.linalg.norm(projector@projector-projector))}

    def rotated_full_state(self, logarithms, orbital_unitary):
        Z = np.zeros((self.n, self.n), complex)
        weights = np.exp(logarithms-logarithms.real.max())
        Z[2*np.arange(self.K), 2*np.arange(self.K)+1] = weights
        Z -= Z.T
        Z = orbital_unitary@Z@orbital_unitary.T
        idx = self.configurations
        state = pfaffian_batch(Z[idx[:, :, None], idx[:, None, :]])
        return state/np.linalg.norm(state)


def quantum_fisher(density, generator):
    values, vectors = np.linalg.eigh(density)
    values = np.maximum(values, 0.)
    transformed = vectors.conj().T@generator@vectors
    denominator = values[:, None]+values[None, :]
    numerator = (values[:, None]-values[None, :])**2
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-13)
    return float(2*np.sum(ratio*abs(transformed)**2))


def coherence_lower_bound(density, generator):
    commutator = generator@density-density@generator
    return float(2*np.linalg.norm(commutator)**2)


def moment_fidelity_bound(mean, variance, max_pair_count):
    """Analytic global bound from Var_phi(m)<=mean_phi(m), evaluated in float64.

    D=sqrt(1-F), delta=Var_psi(m)-mean_psi(m), R=range[(m-mean)^2-m].
    delta <= R*D + r^2*D^2 for any normalized projected Gaussian.
    The continuous range covers both integer and half-integer counts.
    """
    excess = variance-mean
    if excess <= 0:
        return {'fidelity_upper_bound': 1., 'infidelity_lower_bound': 0., 'excess': excess}
    r = float(max_pair_count)
    vertex = np.clip(mean+.5, 0., r)
    function = lambda x: (x-mean)**2-x
    spectral_range = max(function(0.), function(r))-function(vertex)
    distance = 2*excess/(spectral_range+np.sqrt(spectral_range**2+4*r*r*excess))
    return {'fidelity_upper_bound': float(1-distance**2), 'infidelity_lower_bound': float(distance**2),
            'excess': float(excess), 'spectral_range': float(spectral_range),
            'scope': 'All normalized total-number-projected pure fermionic Gaussians; float64 evaluation, not interval arithmetic'}
