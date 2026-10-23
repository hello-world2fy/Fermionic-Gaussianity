"""Number-conserving BCS observables; entropy cuts whole pair levels."""

import numpy as np
from gaussian_synthesis.metrics import normalized


def observables(basis, state, *, left_levels=None):
    state = normalized(basis.sector.vector(state, nonzero=True))
    probability = np.abs(state)**2
    occupied = basis.occupations
    number = probability @ occupied
    nn = np.einsum('b,bi,bj->ij', probability, occupied, occupied)
    pair = np.diag(number).astype(complex)
    index = {mask: i for i, mask in enumerate(basis.pair_masks)}
    for col, mask in enumerate(basis.pair_masks):
        for j in range(basis.L):
            if mask & (1 << j):
                for i in range(basis.L):
                    if not mask & (1 << i):
                        pair[i, j] += state[index[mask ^ (1 << j) ^ (1 << i)]].conj()*state[col]
    left = tuple(range(basis.L//2)) if left_levels is None else tuple(left_levels)
    if len(set(left)) != len(left) or any(type(j) is not int or not 0 <= j < basis.L for j in left):
        raise ValueError('Invalid subsystem levels')
    if basis.L > 16:
        raise ValueError('Dense entropy calculation is limited to L<=16')
    right = tuple(j for j in range(basis.L) if j not in left)
    matrix = np.zeros((1 << len(left), 1 << len(right)), dtype=complex)
    for mask, value in zip(basis.pair_masks, state):
        a = sum(((mask >> j) & 1) << t for t, j in enumerate(left))
        b = sum(((mask >> j) & 1) << t for t, j in enumerate(right))
        matrix[a, b] = value
    weights = np.linalg.svd(matrix, compute_uv=False)**2
    weights = weights[weights > 0]
    return {'probability': probability, 'number': number, 'number_correlation': nn,
            'pair_correlation': pair, 'entropy': float(-np.sum(weights*np.log(weights)))}
