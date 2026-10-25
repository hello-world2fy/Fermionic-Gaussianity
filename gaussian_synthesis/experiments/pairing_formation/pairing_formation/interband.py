"""Fixed-count interband entanglement, local filters, and entangling sources."""
from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix

PAULI = [np.eye(2), np.array([[0., 1.], [1., 0.]]),
         np.array([[0., -1j], [1j, 0.]]), np.diag([1., -1.])]


def sector_entanglement(model, state):
    state = state/np.linalg.norm(state)
    records = []
    for m, (_, indices) in enumerate(model.blocks):
        singular = np.linalg.svd(state[indices], compute_uv=False)
        p = float(np.sum(singular**2))
        q = singular**2/max(p, 1e-300)
        positive = q[q > 1e-16]
        records.append({'m': m, 'probability': p, 'conditional_entropy_bits': float(-positive@np.log2(positive)),
                        'conditional_linear_entropy': float(1.-q@q) if p else 0.,
                        'rank1_weight': float(q[0]), 'rank2_weight': float(q[:2].sum()),
                        'schmidt_probabilities': q.tolist()})
    return {'accessible_entropy_bits': sum(r['probability']*r['conditional_entropy_bits'] for r in records),
            'weighted_linear_entropy': sum(r['probability']*r['conditional_linear_entropy'] for r in records),
            'rank1_fidelity_bound': sum(r['probability']*r['rank1_weight'] for r in records),
            'blocks': records}


def filtered_density(model, state_or_columns, particle, hole):
    """Local m=1 filters; columns represent an unnormalized pure-state ensemble."""
    rows = model.blocks[1][1][np.ix_(particle, hole)].ravel()
    X = np.asarray(state_or_columns)
    X = X[:, None] if X.ndim == 1 else X
    X = X[rows]
    rho = X@X.conj().T
    probability = float(np.trace(rho).real)
    return rho/max(probability, 1e-300), probability


def tomography(rho, probability=1.):
    moments = np.array([[np.trace(rho@np.kron(a, b)).real for b in PAULI] for a in PAULI])
    pt = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    eigenvalues = np.linalg.eigvalsh(pt)
    negativity = float(-np.minimum(eigenvalues, 0.).sum())
    singular = np.linalg.svd(moments[1:, 1:], compute_uv=False)
    chsh = float(2*np.sqrt(np.sum(singular[:2]**2)))
    return {'success_probability': probability, 'negativity': negativity,
            'weighted_negativity': probability*negativity, 'PT_min_eigenvalue': float(eigenvalues[0]),
            'tomographic_CHSH_max': chsh, 'pauli_moments': moments.tolist(),
            'CHSH_is_loophole_free_experiment': False}


def ppt_witness(predicted_density):
    """A linear witness chosen without access to the later measured density."""
    pt = predicted_density.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    values, vectors = np.linalg.eigh(pt)
    v = vectors[:, 0]
    witness = np.outer(v, v.conj()).reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    coefficients = np.array([[np.trace(witness@np.kron(a, b)).real/4 for b in PAULI] for a in PAULI])
    return witness, coefficients, float(values[0])


def choose_filter(model, state):
    """Maximize probability times negativity using training data alone."""
    state = state/np.linalg.norm(state)
    B = state[model.blocks[1][1]]
    choices = []
    for p in combinations(range(model.r), 2):
        for h in combinations(range(model.r), 2):
            value = B[np.ix_(p, h)]
            choices.append((float(abs(np.linalg.det(value))), p, h))
    score, particle, hole = max(choices)
    rho, probability = filtered_density(model, state, particle, hole)
    return {'particle': list(particle), 'hole': list(hole), 'score': score,
            'selection': 'training-state success probability times conditional negativity',
            'training_observables': tomography(rho, probability)}


def schmidt_dephasing_columns(model, state):
    """Local, number-conserving dephasing in state-adapted Schmidt bases.

    Kraus alpha is block-diagonal in local count and projects on Schmidt index
    alpha in every block. Its implementation is not assumed experimentally easy.
    This preserves both local reduced states and every p(m), while each fixed-m
    block becomes separable. Coherence between different m may also change.
    """
    rank = max(len(subsets) for subsets, _ in model.blocks)
    columns = np.zeros((model.dim, rank), complex)
    for m, (_, indices) in enumerate(model.blocks):
        u, singular, vh = np.linalg.svd(state[indices], full_matrices=False)
        for a, weight in enumerate(singular):
            columns[indices.ravel(), a] = (weight*u[:, a, None]*vh[a, None]).ravel()
    return columns


class EntanglingSources:
    def __init__(self, model):
        self.model = model
        dst, src = [], []
        for p in range(model.r, model.L):
            for h in range(model.r):
                d, s = model.transitions[p, h]
                dst.extend(d); src.extend(s)
        self.up = coo_matrix((-model.g*np.ones(len(dst)), (dst, src)), shape=(model.dim, model.dim)).tocsr()
        self.down = self.up.T
        self.local = model.H-self.up-self.down

    def analyze(self, state):
        product = np.zeros(self.model.dim, complex)
        bases = []
        for m, (_, indices) in enumerate(self.model.blocks):
            B = state[indices]
            u, s, vh = np.linalg.svd(B, full_matrices=False)
            product[indices.ravel()] = (np.linalg.norm(s)*u[:, :1]@vh[:1]).ravel()
            bases.append((u[:, 0], vh[0].conj()))
        actions = [operator@product for operator in (self.up, self.down, self.local)]
        records = []
        for m, (_, indices) in enumerate(self.model.blocks):
            u, v = bases[m]
            def normal(value):
                B = value[indices]
                return B-u[:, None]*(u.conj()@B)[None, :]-(B@v)[:, None]*v.conj()[None, :]+np.vdot(u, B@v)*u[:, None]*v.conj()[None, :]
            up, down, local = [normal(action) for action in actions]
            overlap = np.vdot(up, down)
            norm_up, norm_down = float(np.linalg.norm(up)**2), float(np.linalg.norm(down)**2)
            total = float(np.linalg.norm(up+down)**2)
            singular = np.linalg.svd(up+down, compute_uv=False)
            records.append({'m': m, 'from_lower_squared_norm': norm_up, 'from_upper_squared_norm': norm_down,
                            'interference_real': float(2*overlap.real), 'interference_imag': float(2*overlap.imag),
                            'normal_squared_norm': total, 'local_normal_norm': float(np.linalg.norm(local)),
                            'relative_interference': float(2*overlap.real/max(norm_up+norm_down, 1e-300)),
                            'rank_beyond_two_norm': float(np.linalg.norm(singular[2:]))})
        return {'records': records, 'total_normal_squared_norm': sum(r['normal_squared_norm'] for r in records),
                'scope': 'Hamiltonian normal source at the count-preserving best rank-one Schmidt approximation; not exact-state entanglement rate'}, product
