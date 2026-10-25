"""Count-coherence diagnostics, including mixed states as low-rank factors.

These are physical-state diagnostics, not canonical-distribution marginals.
V denotes rho=V V^dagger; its columns need not be orthogonal or normalized.
"""
import numpy as np
from scipy.sparse import diags


def factor_observables(count_labels, factors, theta=.2):
    V = np.asarray(factors)
    if V.ndim == 1:
        V = V[:, None]
    m = np.asarray(count_labels)
    norm = float(np.linalg.norm(V)**2)
    if abs(norm-1.) > 2e-7:
        raise ValueError(f'Expected a normalized density operator, trace={norm}')
    V = V/np.sqrt(norm)
    r = int(m.max())
    probability = np.bincount(m, weights=np.sum(abs(V)**2, axis=1), minlength=r+1)
    mean = float(probability@np.arange(r+1))
    variance = float(probability@np.arange(r+1)**2-mean**2)
    grams = np.array([V[m == j].conj().T@V[m == j] for j in range(r+1)])
    # ||P_a rho P_b||_HS^2 = Tr[(V_a^dagger V_a)(V_b^dagger V_b)].
    coherence = np.einsum('aij,bji->ab', grams, grams).real
    orders = np.array([np.trace(coherence, offset=q) for q in range(r+1)])
    purity = float(coherence.sum())
    echo_by_order = 2*(1-np.cos(theta*np.arange(r+1)))*orders
    echo = float(echo_by_order.sum())
    # Thin-factor QFI retains the contribution from the zero-eigenvalue space.
    U, singular, _ = np.linalg.svd(V, full_matrices=False)
    keep = singular > 1e-12
    U, eigenvalues = U[:, keep], singular[keep]**2
    transformed = U.conj().T@(m[:, None]*U)
    pair_weight = eigenvalues[:, None]*eigenvalues[None, :]
    pair_weight /= eigenvalues[:, None]+eigenvalues[None, :]
    fisher = float(4*probability@(np.arange(r+1)**2)-8*np.sum(pair_weight*abs(transformed)**2))
    return {'count': probability.tolist(), 'mean': mean, 'variance': variance,
            'fano': variance/mean if mean > 1e-14 else 1.,
            'purity': purity, 'QFI': fisher, 'QFI_bound': 4*mean,
            'theta': theta, 'echo_difference': echo,
            'echo_excess': echo-theta**2*mean,
            'coherence_orders': orders.tolist(), 'echo_by_order': echo_by_order.tolist(),
            'vacuum_echo_contribution': float(2*np.sum(coherence[0]*(1-np.cos(theta*np.arange(r+1))))),
            'norm_error_before_normalization': abs(norm-1.)}


def dephase_groups(state, groups):
    """A count-preserving projective dephasing channel, in factor form."""
    groups = np.asarray(groups)
    return np.asarray(state)[:, None]*(groups[:, None] == np.unique(groups)[None, :])


def flat_band_hamiltonian(r, G, energy=.5):
    """Exact permutation-symmetric two-band pairing chain, m=0,...,r.

    Each band has r pair levels of energy +/-energy. Initial m=0 is the
    normal Fermi sea. The basis vector is the normalized product of the
    two fixed-m Dicke states. No bosonic or mean-field approximation.
    """
    m = np.arange(r+1)
    g = G/(2*r)
    diagonal = 4*energy*m-2*energy*r-g*(r+2*m*(r-m))
    offdiagonal = -g*(m[:-1]+1)*(r-m[:-1])
    return diags([offdiagonal, diagonal, offdiagonal], [-1, 0, 1]).tocsr()
