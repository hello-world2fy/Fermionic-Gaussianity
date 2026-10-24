"""Deterministic Cartan projection in small symmetric-replica weight sectors.

In hopping quasiparticle coordinates the FK singlet space is spanned by the
vacuum, a 14-term Cayley four-form, and minus the filled state. Total Cartan
weights are conserved by the replica Casimir, so only weight sectors touched
by these 16 Fock configurations are needed. This module handles k<=4; it is
not a proposal to build large replica tensors at k=64 or 128.
"""
from itertools import combinations_with_replacement
import math
import numpy as np
from numba import njit, prange
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from .energy_windows import hopping_eigenbasis

CAYLEY_MASKS = np.array([15, 51, 60, 85, 90, 102, 105, 150, 153, 165, 170, 195, 204, 240])
CAYLEY_SIGNS = np.array([-1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1])
EVEN_MASKS = np.array([m for m in range(256) if m.bit_count()%2 == 0])


def exact_hopping_fock_proof():
    """Signed-integer verification of the physical-to-Cayley change of basis.

    Every Fock basis vector has entries in (Z+iZ)/4. We retain those integer
    numerators, check every division by two, and use only int64 Gram products.
    """
    coefficients = np.zeros((3,256),np.int64)
    for mask,sign in ((119,1),(120,-1),(135,-1),(136,1)):
        coefficients[0,mask]=sign
    for mask,sign in ((15,-1),(60,1),(90,-1),(105,1),(150,1),(165,-1),(195,1),(240,-1)):
        coefficients[1,mask]=sign
    for mask,sign in ((30,2),(45,-2),(75,2),(180,2),(210,-2),(225,2),
                      (119,-1),(120,-1),(135,-1),(136,-1)):
        coefficients[2,mask]=sign
    if np.any((coefficients[0]+coefficients[2])%2):
        raise ArithmeticError('nonintegral vacuum numerator')
    F_real=np.array([(coefficients[0]+coefficients[2])//2,
                     7*coefficients[0]-coefficients[2],
                     (coefficients[0]+coefficients[2])//2]).T
    F_imag=np.array([-coefficients[1],np.zeros(256,np.int64),coefficients[1]]).T
    def gamma(a,r,i):
        mode,imaginary=divmod(a,2)
        out_r,out_i=np.zeros(256,np.int64),np.zeros(256,np.int64)
        for mask in range(256):
            sign=(-1)**((mask&((1<<mode)-1)).bit_count())
            destination=mask^(1<<mode)
            if imaginary:
                sign*=(-1 if (mask>>mode)&1 else 1)
                out_r[destination],out_i[destination]=-sign*i[mask],sign*r[mask]
            else:
                out_r[destination],out_i[destination]=sign*r[mask],sign*i[mask]
        return out_r,out_i
    real,imag=np.zeros((256,256),np.int64),np.zeros((256,256),np.int64)
    real[:,0],imag[:,0]=F_real[:,0],F_imag[:,0]
    for j in range(8):
        ar,ai=gamma(j,real[:,0],imag[:,0])
        br,bi=gamma(j+8,real[:,0],imag[:,0])
        if np.any(ar-bi) or np.any(ai+br):
            raise ArithmeticError('specified vacuum is not annihilated')
    for mask in range(1,256):
        bit=mask&-mask
        j=bit.bit_length()-1
        previous=mask^bit
        ar,ai=gamma(j,real[:,previous],imag[:,previous])
        br,bi=gamma(j+8,real[:,previous],imag[:,previous])
        rr,ii=ar+bi,ai-br
        if np.any(rr%2) or np.any(ii%2):
            raise ArithmeticError('Fock numerator division is not integral')
        real[:,mask],imag[:,mask]=rr//2,ii//2
    np.testing.assert_array_equal(real.T@real+imag.T@imag,16*np.eye(256,dtype=np.int64))
    np.testing.assert_array_equal(real.T@imag-imag.T@real,np.zeros((256,256),np.int64))
    expected=np.zeros((256,3),np.int64)
    expected[0,0],expected[255,2]=16,-16
    expected[CAYLEY_MASKS,1]=16*CAYLEY_SIGNS
    np.testing.assert_array_equal(real.T@F_real+imag.T@F_imag,expected)
    np.testing.assert_array_equal(real.T@F_imag-imag.T@F_real,np.zeros((256,3),np.int64))
    return {'verified':True,'arithmetic':'signed int64; all divisions checked exact',
            'vacuum_annihilation_exact':True,'fock_unitarity_exact':True,
            'cayley_basis_identity_exact':True,'basis_entry_denominator':4,
            'maximum_numerator_magnitude':int(max(abs(real).max(),abs(imag).max()))}


@njit(cache=True)
def _enumerate_codes(k, weights, allowed):
    maximum = 128*(128+1)//2 if k == 2 else 128*129*130//6 if k == 3 else 128*129*130*131//24
    output = np.empty(maximum, np.int64)
    count = 0
    for a in range(128):
        for b in range(a, 128):
            if k == 2:
                if allowed[weights[a]+weights[b]]:
                    output[count] = (a<<7)|b
                    count += 1
                continue
            for c in range(b, 128):
                if k == 3:
                    if allowed[weights[a]+weights[b]+weights[c]]:
                        output[count] = (a<<14)|(b<<7)|c
                        count += 1
                    continue
                for d in range(c, 128):
                    if allowed[weights[a]+weights[b]+weights[c]+weights[d]]:
                        output[count] = (a<<21)|(b<<14)|(c<<7)|d
                        count += 1
    return output[:count]


def weight_basis(k):
    if k not in (2, 3, 4):
        raise ValueError("the explicit symmetric weight backend is restricted to k=2,3,4")
    weights = np.array([sum(((int(m)>>j)&1)*(k+1)**j for j in range(8)) for m in EVEN_MASKS])
    support_masks = np.r_[0, CAYLEY_MASKS, 255]
    support = np.searchsorted(EVEN_MASKS, support_masks)
    allowed = np.zeros((k+1)**8, bool)
    for indices in combinations_with_replacement(support, k):
        allowed[sum(weights[list(indices)])] = True
    codes = _enumerate_codes(k, weights, allowed)
    tuples = np.array([(codes >> (7*(k-1-j))) & 127 for j in range(k)]).T.astype(np.int16)
    return codes, tuples, {"k": k, "allowed_weight_sectors": int(allowed.sum()),
                           "symmetric_basis_size": len(codes),
                           "ambient_symmetric_dimension": math.comb(127+k, k)}


@njit(cache=True)
def _multiplicity(tup):
    denominator = 1
    count = 1
    for j in range(1, len(tup)+1):
        if j < len(tup) and tup[j] == tup[j-1]:
            count += 1
        else:
            for a in range(2, count+1):
                denominator *= a
            count = 1
    factorial = 1
    for j in range(2, len(tup)+1):
        factorial *= j
    return factorial//denominator


@njit(cache=True, parallel=True)
def _counts(tuples, masks):
    counts = np.ones(len(tuples), np.int64)
    for i in prange(len(tuples)):
        for a in range(tuples.shape[1]):
            for b in range(a+1, tuples.shape[1]):
                diff = masks[tuples[i,a]] ^ masks[tuples[i,b]]
                d = 0
                for bit in range(8):
                    d += (diff>>bit)&1
                counts[i] += d*(d-1)//2
    return counts


@njit(cache=True, parallel=True)
def _build_columns(codes, tuples, masks, mask_to_index, indptr, row, value, integer=False):
    for source in prange(len(codes)):
        k = tuples.shape[1]
        tup = tuples[source]
        multiplicity = _multiplicity(tup)
        offset = indptr[source]
        row[offset] = source
        diagonal = 0.
        offset += 1
        for a in range(k):
            for b in range(a+1, k):
                x, y = masks[tup[a]], masks[tup[b]]
                diff = x^y
                bits = np.empty(8, np.int64)
                count = 0
                for bit in range(8):
                    if (diff>>bit)&1:
                        bits[count] = bit
                        count += 1
                diagonal += 4*count
                for j in range(count):
                    for ell in range(j+1, count):
                        flip = (1<<bits[j])|(1<<bits[ell])
                        changed = tup.copy()
                        changed[a] = mask_to_index[x^flip]
                        changed[b] = mask_to_index[y^flip]
                        changed.sort()
                        code = 0
                        for index in changed:
                            code = (code<<7)|index
                        destination = np.searchsorted(codes, code)
                        if destination >= len(codes) or codes[destination] != code:
                            raise ValueError("Cartan action escaped the conserved weight sectors")
                        row[offset] = destination
                        sign = 1 if (j+ell)%2 == 0 else -1
                        value[offset] = 8*sign if integer else 8*sign*math.sqrt(multiplicity/_multiplicity(changed))
                        offset += 1
        value[indptr[source]] = diagonal


def casimir_defect(k):
    codes, tuples, report = weight_basis(k)
    counts = _counts(tuples, EVEN_MASKS)
    indptr = np.r_[0, np.cumsum(counts)]
    row = np.empty(int(indptr[-1]), np.int32)
    value = np.empty(len(row), np.float64)
    mask_to_index = np.full(256, -1, np.int16)
    mask_to_index[EVEN_MASKS] = np.arange(128)
    _build_columns(codes, tuples, EVEN_MASKS, mask_to_index, indptr, row, value)
    A = csc_matrix((value, row, indptr), shape=(len(codes), len(codes))).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    asymmetry = A-A.T
    error = float(np.max(abs(asymmetry.data))) if asymmetry.nnz else 0.
    if error > 1e-12:
        raise ArithmeticError("replica Casimir is not symmetric in normalized tensor coordinates")
    report.update(nnz=A.nnz, symmetry_error=error,
                  casimir_max=2*k*k+28*k, defect_max=4*(2*k*k+28*k),
                  theoretical_nonzero_defect_lower=8 if k == 3 else 16 if k == 4 else 16)
    return A, codes, tuples, report


def integer_casimir_defect(k):
    """Same exact action in unnormalized ordered-amplitude coordinates."""
    codes, tuples, report = weight_basis(k)
    counts = _counts(tuples, EVEN_MASKS)
    indptr = np.r_[0, np.cumsum(counts)]
    row = np.empty(int(indptr[-1]), np.int32)
    values = np.empty(len(row), np.int64)
    lookup = np.full(256, -1, np.int16)
    lookup[EVEN_MASKS] = np.arange(128)
    _build_columns(codes, tuples, EVEN_MASKS, lookup, indptr, row, values, True)
    # Outgoing canonical-tuple transitions are columns above; the operator on
    # ordered amplitudes is their transpose (normalized tensor multiplicities
    # are now accounted for only by the weighted inner product).
    A = csc_matrix((values, row, indptr), shape=(len(codes), len(codes))).T.tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    multiplicities = np.array([_multiplicity(t) for t in tuples], dtype=np.int64)
    weighted = A.multiply(multiplicities[:, None]).tocsr()
    defect = weighted-weighted.T
    if defect.nnz and np.any(defect.data):
        raise ArithmeticError("integer Casimir violates exact weighted self-adjointness")
    row_norm = int(np.max(np.asarray(abs(A).sum(axis=1)).ravel()))
    report.update(nnz=A.nnz, max_absolute_row_sum=row_norm,
                  exact_weighted_symmetry_verified=True)
    return A, codes, tuples, multiplicities, report


def possible_defect_roots(k):
    """All nonzero Casimir defects allowed by tensor-spinor weight bounds.

Extra roots for absent irreps are harmless. Top chirality-reversed weight is
not in S_+^tensor k: saturating the first seven coordinates fixes the last.
"""
    top = 2*k*k+28*k
    if k == 2:
        casimirs = [b*(16-b) for b in range(0, 9, 2)]
    elif k == 3:
        casimirs = [30+17*m-m*m for m in range(9)]
    elif k == 4:
        casimirs = []
        for a in range(9):
            for b in range(0, 9-a, 2):
                weight = [2]*a+[1]*b+[0]*(8-a-b)
                casimirs.append(sum(lam*(lam+2*(7-j)) for j, lam in enumerate(weight)))
    else:
        raise ValueError(k)
    return sorted({4*(top-c) for c in casimirs if c < top}, reverse=True)


def monomial_indices(k):
    return np.array([(a, b, k-a-b) for a in range(k, -1, -1) for b in range(k-a, -1, -1)])


def replica_frame(k, tuples):
    categories = np.full(128, -1, int)
    amplitude = np.zeros(128)
    categories[0] = 0
    amplitude[0] = 1
    for mask, sign in zip(CAYLEY_MASKS, CAYLEY_SIGNS):
        index = np.searchsorted(EVEN_MASKS, mask)
        categories[index] = 1
        amplitude[index] = sign/math.sqrt(14)
    categories[-1] = 2
    amplitude[-1] = -1
    rows = np.flatnonzero(np.all(categories[tuples] >= 0, axis=1))
    powers = monomial_indices(k)
    lookup = {tuple(p): j for j, p in enumerate(powers)}
    columns, values = [], []
    for row in rows:
        tup = tuples[row]
        counts = np.bincount(categories[tup], minlength=3)
        column = lookup[tuple(counts)]
        normal = math.sqrt(math.factorial(k)/math.prod(math.factorial(int(v)) for v in counts))
        value = math.sqrt(_multiplicity(tup))*np.prod(amplitude[tup])/normal
        columns.append(column)
        values.append(value)
    V = csr_matrix((values, (rows, columns)), shape=(len(tuples), len(powers)))
    error = float(np.linalg.norm((V.T@V).toarray()-np.eye(len(powers))))
    if error > 1e-12:
        raise ArithmeticError("symmetric replica frame normalization failed")
    return V, rows, powers, error


def chebyshev_project(A, V, *, minimum, maximum, degree=100, callback=None):
    """Stable normalized Chebyshev filter, equal to one on the Cartan kernel.

The exact spectral remainder is <= 1/cosh(degree*acosh(a)) on every
nonzero eigenvalue, a=(maximum+minimum)/(maximum-minimum). Binary64
rounding is diagnosed separately; a small filter remainder alone is not an
interval arithmetic certificate.
"""
    a = (maximum+minimum)/(maximum-minimum)
    b = 2/(maximum-minimum)
    tau = math.acosh(a)
    previous = V.toarray()
    current = previous-b/a*(A@previous)
    for m in range(2, degree+1):
        ratio1 = math.cosh((m-1)*tau)/math.cosh(m*tau)
        ratio2 = math.cosh((m-2)*tau)/math.cosh(m*tau)
        nxt = 2*ratio1*(a*current-b*(A@current))-ratio2*previous
        previous, current = current, nxt
        if callback and m%10 == 0:
            callback(m)
    remainder = 1/math.cosh(degree*tau)
    residual = float(np.linalg.norm(A@current))
    return current, {"filter_degree": degree, "exact_arithmetic_spectral_remainder": remainder,
                     "projected_frame_defect_residual": residual,
                     "last_iterate_difference": float(np.linalg.norm(current-previous)),
                     "strict_interval_certified": False}


@dataclass
class DeterministicValue:
    q: float
    gradient: np.ndarray
    hessian: np.ndarray | None
    mass: None = None
    ess: None = None


class CartanMoments:
    def __init__(self, k, matrix):
        self.k = int(k)
        self.matrix = np.asarray(matrix)
        self.powers = monomial_indices(k)
        self.map = hopping_eigenbasis().T
        self.directions = self.map@np.concatenate((np.eye(3), 1j*np.eye(3)), axis=1)
        self.scales = np.array([math.sqrt(math.factorial(k)/math.prod(math.factorial(int(x)) for x in p)) for p in self.powers])

    def monomials(self, z, hessian=True):
        alpha = self.map@z
        values = self.scales*np.prod(alpha[None, :]**self.powers, axis=1)
        derivative = np.zeros((len(values), 3), complex)
        second = np.zeros((len(values), 3, 3), complex)
        for row, power in enumerate(self.powers):
            for j in range(3):
                if not power[j]:
                    continue
                p = power.copy()
                p[j] -= 1
                derivative[row, j] = self.scales[row]*power[j]*np.prod(alpha**p)
                if hessian:
                    for ell in range(3):
                        if not p[ell]:
                            continue
                        q = p.copy()
                        q[ell] -= 1
                        second[row, j, ell] = self.scales[row]*power[j]*p[ell]*np.prod(alpha**q)
        J = derivative@self.directions
        K = np.einsum('ajl,jr,ls->ars', second, self.directions, self.directions) if hessian else None
        return values, J, K

    def evaluate(self, z, hessian=True):
        values, jacobian, second = self.monomials(z, hessian=hessian)
        action = self.matrix@values
        q = float(np.vdot(values, action).real)
        gradient = (jacobian.conj().T@action).real/self.k
        hes = None
        if hessian:
            hes = (jacobian.conj().T@self.matrix@jacobian+np.einsum('ars,a->rs', second.conj(), action)).real/self.k
            hes = .5*(hes+hes.T)
        return DeterministicValue(q, gradient, hes)

    def objective(self, z, target):
        return self.evaluate(z, hessian=False).q/(2*self.k)-float(np.vdot(target, z).real)


def independent_full_reconstruction(k, z, powers, support_tuples, projected_support_rows):
    """Contract the projected symmetric tensor in all 128 new Fock directions.

The conserved Hamming-code character forces all other coordinates to zero;
within its 16 supported coordinates no singlet projection is inserted.
"""
    alpha = hopping_eigenbasis().T@z
    u = np.zeros(128, complex)
    u[0], u[-1] = alpha[0], -alpha[2]
    u[np.searchsorted(EVEN_MASKS, CAYLEY_MASKS)] = CAYLEY_SIGNS*alpha[1]/math.sqrt(14)
    scales = [math.sqrt(math.factorial(k)/math.prod(math.factorial(int(x)) for x in p)) for p in powers]
    v = np.array(scales)*np.prod(alpha[None, :]**powers, axis=1)
    projected = projected_support_rows@v
    C = np.zeros(128, complex)
    for tup, value in zip(support_tuples, projected):
        scale = math.sqrt(_multiplicity(tup))/k
        for position in range(k):
            others = np.delete(tup, position)
            C[tup[position]] += scale*np.prod(u[others]).conjugate()*value
    return C
