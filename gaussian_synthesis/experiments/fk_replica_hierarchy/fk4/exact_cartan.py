"""Directed interval evaluation of the integer Cartan Gram certificate.

The Gram is in unnormalized Cayley coordinates. No fitted rational or rounded
matrix coefficient enters a certificate. Targets mean their saved binary64
Fock amplitudes; an exact-Hamiltonian eigenstate enclosure is separate.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import numpy as np
from mpmath import iv
from .certificates import _c, _mul, _scale, _bounds, exact_basis_action


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _conj(a):
    return a[0], -a[1]


def _pow(a, n):
    out = (iv.mpf(1), iv.mpf(0))
    for _ in range(int(n)):
        out = _mul(out, a)
    return out


def _sum(values):
    out = (iv.mpf(0), iv.mpf(0))
    for value in values:
        out = _add(out, value)
    return out


def radical_transform():
    return [[iv.mpf(1)/4, iv.sqrt(14)/4, iv.mpf(1)/4],
            [-iv.sqrt(2)/2, iv.mpf(0), iv.sqrt(2)/2],
            [iv.sqrt(7)/4, -iv.sqrt(2)/4, iv.sqrt(7)/4]]


def dimension(n, k):
    d = Fraction(1)
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            d *= Fraction(k+2*n-i-j, 2*n-i-j)
    assert d.denominator == 1
    return d.numerator


class ExactCartan:
    def __init__(self, proof):
        self.path = Path(proof)
        self.proof = json.loads(self.path.read_text())
        if self.proof['status'] != 'complete' or not self.proof['exact_integer_gram_recovered']:
            raise ValueError('a completed exact modular Gram is required')
        self.k = self.proof['k']
        self.powers = self.proof['powers']
        self.numerators = [[int(x) for x in row] for row in self.proof['gram_numerator']]
        self.denominator = int(self.proof['projector_denominator'])

    def evaluate(self, z, digits=70):
        iv.dps = digits
        V = radical_transform()
        zz = [_c(x) for x in z]
        beta = [_sum(_scale(zz[j], V[j][i]) for j in range(3)) for i in range(3)]
        scales = [iv.mpf(1), 1/iv.sqrt(14), iv.mpf(1)]
        x = [_scale(b, s) for b, s in zip(beta, scales)]
        monomials = [_mul(_mul(_pow(x[0], p[0]), _pow(x[1], p[1])), _pow(x[2], p[2])) for p in self.powers]
        action = [_sum(_scale(m, iv.mpf(n)/self.denominator) for m, n in zip(monomials, row) if n)
                  for row in self.numerators]
        q = _sum(_mul(_conj(m), a) for m, a in zip(monomials, action))[0]
        Cbeta = []
        for j in range(3):
            derivatives = []
            for p, a in zip(self.powers, action):
                if p[j]:
                    reduced = p.copy()
                    reduced[j] -= 1
                    derivative = _mul(_mul(_pow(x[0], reduced[0]), _pow(x[1], reduced[1])), _pow(x[2], reduced[2]))
                    derivatives.append(_scale(_mul(_conj(derivative), a), p[j]*scales[j]/self.k))
            Cbeta.append(_sum(derivatives))
        Cz = [_sum(_scale(Cbeta[j], V[i][j]) for j in range(3)) for i in range(3)]
        return q, Cz

    def certificate(self, z, target, previous=None, digits=70):
        q, Cz = self.evaluate(z, digits)
        if _bounds(q)[0] <= 0:
            raise ValueError('positive moment required')
        upper = _bounds(q**(iv.mpf(1)/(2*self.k)))[1]
        safe = np.asarray(z, complex)/(upper*(1+32*np.finfo(float).eps))
        q_safe, _ = self.evaluate(safe, digits)
        if _bounds(q_safe)[1] > 1:
            raise ArithmeticError('normalized interval moment exceeds one')
        basis = [exact_basis_action(np.eye(3, dtype=complex)[j]) for j in range(3)]
        fullC = [_sum(_mul(basis[j][i], Cz[j]) for j in range(3)) for i in range(128)]
        residuals = [(c[0]-_c(t)[0], c[1]-_c(t)[1]) for c, t in zip(fullC, target)]
        magnitudes = [iv.sqrt(r**2+s**2) for r, s in residuals]
        l1 = sum(magnitudes, iv.mpf(0))
        rho = iv.sqrt(sum((r**2+s**2 for r,s in residuals), iv.mpf(0)))
        w = exact_basis_action(safe)
        inner = sum((a*_c(t)[0]+b*_c(t)[1] for (a,b),t in zip(w,target)), iv.mpf(0))
        norm = iv.sqrt(sum((a**2+b**2 for a,b in w),iv.mpf(0)))
        result = {'normalization_order': self.k, 'q_interval': _bounds(q),
                  'q_normalized_interval': _bounds(q_safe),
                  'L_certified_saved_target': _bounds(inner)[0],
                  'witness_hilbert_norm_upper': _bounds(norm)[1],
                  'full_reconstruction_residual_upper': _bounds(rho)[1],
                  'fock_residual_repair_cost_upper': _bounds(l1)[1],
                  'full_dictionary_feasible': True, 'gaussian_support_upper': 1.,
                  'high_order_moment_certified': True, 'interval_decimal_digits': digits,
                  'proof_file': str(self.path),
                  'target_scope': 'Exact vector of saved binary64 Fock entries; not an enclosed exact Hamiltonian ground state',
                  'reconstruction_scope': 'Full 128-dimensional even Fock space; Spin(7) invariance proves C lies in S',
                  'z_feasible': safe}
        if previous is not None:
            if previous.k != self.k-1:
                raise ValueError('adjacent integer order needed for Cauchy mass bound')
            qp, _ = previous.evaluate(z, digits)
            mass_upper = iv.sqrt(iv.mpf(dimension(8,self.k))/dimension(8,self.k-1)*q*qp)
            result['canonical_B_cauchy_upper'] = _bounds(mass_upper)[1]
            result['repaired_decomposition_cost_upper'] = _bounds(mass_upper+l1)[1]
            result['mass_bound_formula'] = 'B_k <= sqrt(d_k/d_(k-1) q_k(u) q_(k-1)(u)); add exact Fock residual for saved target'
        return result
